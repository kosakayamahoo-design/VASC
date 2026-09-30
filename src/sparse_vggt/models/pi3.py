import sys
from functools import partial
from pathlib import Path
from types import MethodType
from typing import Literal

import torch

from sparse_vggt.models.attention import adaptive_sparse_attention_forward
from sparse_vggt.models.utils import print_sparse_info
from sparse_vggt.utils.hilbert import hilbert_permute
from sparse_vggt.utils.sparse_wrapper import (
    check_sparse_mode,
    finalize_cosa_value_risk_pair_debt,
    finalize_projected_qk_pv_debt,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(_PROJECT_ROOT / "external/Pi3"))
from pi3.models.pi3 import Pi3


def _finalize_pi3_sparse_debt(model):
    routing_state = getattr(model, "_sparse_routing_state", None)
    aux_output_store = getattr(model, "_sparse_aux_output_store", None)
    terminal_stats = None
    if aux_output_store:
        final_layer = max(aux_output_store)
        if isinstance(aux_output_store[final_layer], dict):
            terminal_stats = aux_output_store[final_layer]
    finalize_projected_qk_pv_debt(
        routing_state, stats_store=terminal_stats
    )
    finalize_cosa_value_risk_pair_debt(
        routing_state, stats_store=terminal_stats
    )


def pi3_sparse_attention_forward(
    self,
    x,
    xpos,  # replace pos with xpos
    sparse_ratio,
    cdf_threshold,
    pool_mode,
    aux_sparsity_only,
    aux_output_store,
    num_special_tokens,
    num_heads,
    layer_idx,
    num_global_layers,
    routing_state,
):
    """Wrapper for Pi3 attention because pi3 has `xpos` instead of `pos`"""
    return adaptive_sparse_attention_forward(
        self,
        x,
        xpos,
        sparse_ratio,
        cdf_threshold,
        pool_mode,
        aux_sparsity_only,
        aux_output_store,
        num_special_tokens,
        num_heads,
        layer_idx=layer_idx,
        dual_path_num_layers=num_global_layers,
        routing_state=routing_state,
    )


def sparse_pi3_decode_forward(self, hidden, N, H, W, use_hilbert=False):
    BN, hw, _ = hidden.shape
    B = BN // N

    final_output = []
    routing_state = getattr(self, "_sparse_routing_state", None)
    if routing_state is not None:
        routing_state.clear()

    hidden = hidden.reshape(B * N, hw, -1)

    register_token = self.register_token.repeat(B, N, 1, 1).reshape(
        B * N, *self.register_token.shape[-2:]
    )

    # Concatenate special tokens with patch tokens
    hidden = torch.cat([register_token, hidden], dim=1)
    hw = hidden.shape[1]

    # Convert HW to patch space
    H = H // self.patch_size
    W = W // self.patch_size
    pos = self.position_getter(B * N, H, W, hidden.device)

    if self.patch_start_idx > 0:
        # do not use position embedding for special tokens (camera and register tokens)
        # so set pos to 0 for the special tokens
        pos = pos + 1
        pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
        pos = torch.cat([pos_special, pos], dim=1)

    # add: hilbert permutation
    orig_pos = pos.clone()
    if use_hilbert:
        S = self.patch_start_idx
        hidden = hilbert_permute(hidden, H, W, S)  # (B * N, H * W + S, C)
        orig_pos = pos.clone()
        pos = hilbert_permute(pos, H, W, S)  # (B * N, H * W, 2)

    for i in range(len(self.decoder)):
        blk = self.decoder[i]

        if i % 2 == 0:
            pos = pos.reshape(B * N, hw, -1)
            hidden = hidden.reshape(B * N, hw, -1)
        else:
            pos = pos.reshape(B, N * hw, -1)
            hidden = hidden.reshape(B, N * hw, -1)

        hidden = blk(hidden, xpos=pos)

        if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
            # add: hilbert permutation
            if use_hilbert:
                S = self.patch_start_idx
                hidden_orig = hidden.reshape(B * N, hw, -1)
                hidden_orig = hilbert_permute(hidden_orig, H, W, S, reverse=True)
                final_output.append(hidden_orig)
            else:
                final_output.append(hidden.reshape(B * N, hw, -1))

    _finalize_pi3_sparse_debt(self)

    return torch.cat([final_output[0], final_output[1]], dim=-1), orig_pos.reshape(B * N, hw, -1)


def sparse_model_from_pi3(
    model: Pi3,
    use_hilbert: bool = False,
    sparse_ratio: float | None = None,
    layer_sparsity_ratios: list[float] | None = None,
    cdf_threshold: float | None = None,
    pool_mode: Literal["max", "avg"] = "avg",
    aux_output: bool = False,
    aux_sparsity_only: bool = True,
    num_special_tokens: int = 5,
    verbose: bool = True,
):
    """Convert the original Pi3 model to a sparse model.

    Args:
        model: Original Pi3 model.
        use_hilbert: If True, use Hilbert permutation on patch tokens.
        sparse_ratio: Sparse ratio for the global attention.
        cdf_threshold: CDF threshold for the global attention.
        pool_mode: Avg or Max pooling for the global attention.
        aux_output: If True, store auxiliary output from the global attention.
        aux_sparsity_only: If True, only store sparsity from the global attention.

    Returns:
        Aggregator: Modified Aggregator
        aux_output_store (dict | None): Auxiliary output storage.
    """
    # Check sparse mode
    check_sparse_mode(sparse_ratio, cdf_threshold)

    # Replace model decode function
    decode_fwd = partial(sparse_pi3_decode_forward, use_hilbert=use_hilbert)
    model.decode = MethodType(decode_fwd, model)

    # Auxiliary output
    modified_layers = [i for i in range(len(model.decoder)) if i % 2 != 0]
    num_global_layers = len(modified_layers)
    if layer_sparsity_ratios is not None:
        if len(layer_sparsity_ratios) != num_global_layers:
            raise ValueError(
                "Pi3 layer_sparsity_ratios must match the 18 global-attention layers"
            )
        if any(ratio < 0.0 or ratio > 1.0 for ratio in layer_sparsity_ratios):
            raise ValueError("Pi3 layer sparsity ratios must be in [0, 1]")
    routing_state = {}
    model._sparse_routing_state = routing_state
    if aux_output:
        # Pointers to store auxiliary output
        aux_output_store = {i: {} for i in range(num_global_layers)}
    else:
        # No auxiliary output``
        aux_output_store = {i: None for i in range(num_global_layers)}
    model._sparse_aux_output_store = aux_output_store

    for i in range(len(model.decoder)):
        if i not in modified_layers:
            continue

        # Replace attention forward function
        logical_layer_idx = i // 2
        layer_sparse_ratio = (
            layer_sparsity_ratios[logical_layer_idx]
            if layer_sparsity_ratios is not None
            else sparse_ratio
        )
        attn_fwd = partial(
            pi3_sparse_attention_forward,
            sparse_ratio=layer_sparse_ratio,
            cdf_threshold=cdf_threshold,
            pool_mode=pool_mode,
            aux_output_store=aux_output_store[logical_layer_idx],
            aux_sparsity_only=aux_sparsity_only,
            num_special_tokens=num_special_tokens,
            num_heads=model.decoder[i].attn.num_heads,
            layer_idx=logical_layer_idx,
            num_global_layers=num_global_layers,
            routing_state=routing_state,
        )

        # Replace attention
        model.decoder[i].attn.forward = MethodType(attn_fwd, model.decoder[i].attn)
        model.decoder[i].attn._sparse_routing_state = routing_state

    # Print some info
    if verbose:
        print_sparse_info(
            sparse_ratio,
            cdf_threshold,
            pool_mode,
            use_hilbert,
            aux_output,
            aux_sparsity_only,
        )
        if layer_sparsity_ratios is not None:
            print(f"Pi3 global layer sparsity: {layer_sparsity_ratios}")
    return model, aux_output_store
