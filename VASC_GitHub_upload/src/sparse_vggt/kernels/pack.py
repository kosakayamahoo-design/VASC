"""Triton kernels for packing debt-routed K/V buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_debt_kv_kernel(
    dense_key_ptr,
    dense_value_ptr,
    coarse_key_ptr,
    coarse_value_ptr,
    residual_key_ptr,
    residual_value_ptr,
    special_key_ptr,
    special_value_ptr,
    center_ptr,
    coarse_index_ptr,
    residual_index_ptr,
    output_key_ptr,
    output_value_ptr,
    dense_tokens: tl.constexpr,
    coarse_frames: tl.constexpr,
    coarse_cells: tl.constexpr,
    residual_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    packed_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    packed_token = row % packed_tokens
    outer = row // packed_tokens
    group = outer % groups
    outer = outer // groups
    head = outer % heads
    batch = outer // heads

    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    output_key = tl.zeros((block_dim,), dtype=tl.float32)
    output_value = tl.zeros((block_dim,), dtype=tl.float32)

    dense_mask = (packed_token < dense_tokens) & dim_mask
    center = tl.load(center_ptr + group)
    dense_source_token = center * dense_tokens + packed_token
    dense_offsets = (
        (
            (batch * heads + head) * num_frames * dense_tokens
            + dense_source_token
        )
        * head_dim
        + dim_offsets
    )
    output_key += tl.load(dense_key_ptr + dense_offsets, mask=dense_mask, other=0.0)
    output_value += tl.load(
        dense_value_ptr + dense_offsets, mask=dense_mask, other=0.0
    )

    coarse_start = dense_tokens
    coarse_token_count = coarse_frames * coarse_cells
    coarse_position = packed_token - coarse_start
    coarse_valid = (
        (coarse_position >= 0)
        & (coarse_position < coarse_token_count)
    )
    coarse_slot = coarse_position // coarse_cells
    coarse_cell = coarse_position % coarse_cells
    coarse_frame = tl.load(
        coarse_index_ptr + (batch * groups + group) * coarse_frames + coarse_slot,
        mask=coarse_valid,
        other=0,
    )
    coarse_mask = coarse_valid & dim_mask
    coarse_source_token = coarse_frame * coarse_cells + coarse_cell
    coarse_offsets = (
        (
            (batch * heads + head) * num_frames * coarse_cells
            + coarse_source_token
        )
        * head_dim
        + dim_offsets
    )
    output_key += tl.load(
        coarse_key_ptr + coarse_offsets, mask=coarse_mask, other=0.0
    )
    output_value += tl.load(
        coarse_value_ptr + coarse_offsets, mask=coarse_mask, other=0.0
    )

    residual_start = coarse_start + coarse_token_count
    residual_position = packed_token - residual_start
    residual_valid = (
        (residual_position >= 0)
        & (residual_position < residual_tokens)
    )
    residual_source_token = tl.load(
        residual_index_ptr
        + (batch * groups + group) * residual_tokens
        + residual_position,
        mask=residual_valid,
        other=0,
    )
    residual_mask = residual_valid & dim_mask
    residual_offsets = (
        (
            (batch * heads + head) * num_frames * coarse_cells
            + residual_source_token
        )
        * head_dim
        + dim_offsets
    )
    output_key += tl.load(
        residual_key_ptr + residual_offsets, mask=residual_mask, other=0.0
    )
    output_value += tl.load(
        residual_value_ptr + residual_offsets,
        mask=residual_mask,
        other=0.0,
    )

    if special_tokens > 0:
        special_start = residual_start + residual_tokens
        special_position = packed_token - special_start
        special_valid = (
            (special_position >= 0)
            & (special_position < special_tokens)
        )
        special_mask = special_valid & dim_mask
        special_offsets = (
            ((batch * heads + head) * special_tokens + special_position)
            * head_dim
            + dim_offsets
        )
        output_key += tl.load(
            special_key_ptr + special_offsets, mask=special_mask, other=0.0
        )
        output_value += tl.load(
            special_value_ptr + special_offsets,
            mask=special_mask,
            other=0.0,
        )

    output_offsets = row * head_dim + dim_offsets
    tl.store(output_key_ptr + output_offsets, output_key, mask=dim_mask)
    tl.store(output_value_ptr + output_offsets, output_value, mask=dim_mask)


def triton_pack_debt_kv(
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    coarse_key: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_key: torch.Tensor,
    residual_value: torch.Tensor,
    centers: torch.Tensor,
    coarse_indices: torch.Tensor,
    residual_indices: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the common debt-routing K/V layout with one fused kernel."""
    sources = (
        dense_key,
        dense_value,
        coarse_key,
        coarse_value,
        residual_key,
        residual_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("debt K/V sources must be five-dimensional")
    if any(not source.is_cuda for source in sources):
        raise ValueError("Triton debt packing requires CUDA tensors")
    if any(not source.is_contiguous() for source in sources):
        raise ValueError("Triton debt packing requires contiguous sources")
    if any(source.device != dense_key.device for source in sources):
        raise ValueError("all debt K/V sources must use the same device")
    if any(source.dtype != dense_key.dtype for source in sources):
        raise ValueError("all debt K/V sources must use the same dtype")
    if dense_key.shape != dense_value.shape:
        raise ValueError("dense key and value shapes must match")
    if coarse_key.shape != coarse_value.shape:
        raise ValueError("coarse key and value shapes must match")
    if residual_key.shape != residual_value.shape:
        raise ValueError("residual key and value shapes must match")
    if coarse_key.shape != residual_key.shape:
        raise ValueError("coarse and residual shapes must match")

    batch, heads, num_frames, dense_tokens, head_dim = dense_key.shape
    expected_prefix = (batch, heads, num_frames)
    if coarse_key.shape[:3] != expected_prefix:
        raise ValueError("dense, coarse, and residual prefixes must match")
    if coarse_key.shape[-1] != head_dim:
        raise ValueError("all sources must use the same head dimension")
    if centers.ndim != 1:
        raise ValueError("centers must be one-dimensional")
    groups = centers.numel()
    if coarse_indices.shape[:2] != (batch, groups):
        raise ValueError("coarse indices must have shape [batch, groups, frames]")
    if residual_indices.shape[:2] != (batch, groups):
        raise ValueError("residual indices must have shape [batch, groups, tokens]")
    if any(
        index.dtype not in {torch.int32, torch.int64}
        for index in (centers, coarse_indices, residual_indices)
    ):
        raise ValueError("debt pack indices must use an integer dtype")
    if any(
        index.device != dense_key.device
        for index in (centers, coarse_indices, residual_indices)
    ):
        raise ValueError("debt pack indices must use the source device")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("special key and value shapes must match")
        if special_key.shape[:2] != (batch, heads):
            raise ValueError("special K/V batch and head dimensions must match")
        if special_key.shape[-1] != head_dim:
            raise ValueError("special K/V head dimensions must match")
        if (
            not special_key.is_cuda
            or not special_value.is_cuda
            or not special_key.is_contiguous()
            or not special_value.is_contiguous()
            or special_key.device != dense_key.device
            or special_value.device != dense_key.device
            or special_key.dtype != dense_key.dtype
            or special_value.dtype != dense_key.dtype
        ):
            raise ValueError("special K/V must be contiguous CUDA tensors")
        special_tokens = special_key.shape[-2]

    coarse_frames = coarse_indices.shape[-1]
    coarse_cells = coarse_key.shape[-2]
    residual_tokens = residual_indices.shape[-1]
    packed_tokens = (
        dense_tokens
        + coarse_frames * coarse_cells
        + residual_tokens
        + special_tokens
    )
    output_shape = (batch, heads, groups, packed_tokens, head_dim)
    output_key = torch.empty(
        output_shape, device=dense_key.device, dtype=dense_key.dtype
    )
    output_value = torch.empty_like(output_key)
    if output_key.numel() == 0:
        return output_key, output_value

    centers = centers.contiguous()
    coarse_indices = coarse_indices.contiguous()
    residual_indices = residual_indices.contiguous()
    block_dim = triton.next_power_of_2(head_dim)
    grid = (batch * heads * groups * packed_tokens,)
    _pack_debt_kv_kernel[grid](
        dense_key,
        dense_value,
        coarse_key,
        coarse_value,
        residual_key,
        residual_value,
        special_key,
        special_value,
        centers,
        coarse_indices,
        residual_indices,
        output_key,
        output_value,
        dense_tokens=dense_tokens,
        coarse_frames=coarse_frames,
        coarse_cells=coarse_cells,
        residual_tokens=residual_tokens,
        special_tokens=special_tokens,
        packed_tokens=packed_tokens,
        num_frames=num_frames,
        head_dim=head_dim,
        heads=heads,
        groups=groups,
        block_dim=block_dim,
    )
    return output_key, output_value
