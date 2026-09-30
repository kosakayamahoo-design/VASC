"""FlashAttention coarse-carrier prefix with exact HPS corrections."""

import math

import torch
import triton
import triton.language as tl

from sparse_vggt.kernels.direct_attention import _online_attention_update


@triton.jit
def _subtract_attention_update(
    query,
    key,
    value,
    accumulator,
    denominator,
    row_max,
    query_mask,
    key_mask,
    softmax_scale_log2,
    weight: tl.constexpr,
):
    logits = tl.dot(query, tl.trans(key)) * softmax_scale_log2
    valid = query_mask[:, None] & key_mask[None, :]
    shifted = tl.where(
        valid,
        logits - row_max[:, None],
        -float("inf"),
    )
    probabilities = tl.math.exp2(shifted) * weight
    denominator -= tl.sum(probabilities, axis=1)
    accumulator -= tl.dot(probabilities.to(value.dtype), value)
    return accumulator, denominator


@triton.jit
def _hps_flash_correction_kernel(
    query_ptr,
    dense_key_ptr,
    dense_value_ptr,
    coarse_key_ptr,
    coarse_value_ptr,
    residual_key_ptr,
    residual_value_ptr,
    residual_index_ptr,
    complement_index_ptr,
    special_key_ptr,
    special_value_ptr,
    carrier_output_ptr,
    carrier_lse_ptr,
    output_ptr,
    softmax_scale,
    dense_tokens: tl.constexpr,
    coarse_cells: tl.constexpr,
    residual_tokens: tl.constexpr,
    complement_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    complement_correction: tl.constexpr,
):
    query_block = tl.program_id(0)
    frame_program = tl.program_id(1)
    frame = frame_program % num_frames
    outer = frame_program // num_frames
    head = outer % heads
    batch = outer // heads

    query_offsets = query_block * block_m + tl.arange(0, block_m)
    dim_offsets = tl.arange(0, block_d)
    query_mask = query_offsets < dense_tokens
    dim_mask = dim_offsets < head_dim
    query_base = (
        ((batch * heads + head) * num_frames + frame)
        * dense_tokens
        * head_dim
    )
    query = tl.load(
        query_ptr
        + query_base
        + query_offsets[:, None] * head_dim
        + dim_offsets[None, :],
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offsets = (
        query_base
        + query_offsets[:, None] * head_dim
        + dim_offsets[None, :]
    )
    carrier_lse_base = (
        ((batch * heads + head) * num_frames + frame) * dense_tokens
    )
    accumulator = tl.load(
        carrier_output_ptr + output_offsets,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    row_max = tl.load(
        carrier_lse_ptr + carrier_lse_base + query_offsets,
        mask=query_mask,
        other=0.0,
    ) * 1.4426950408889634
    if complement_correction:
        accumulator *= 0.5
        denominator = tl.where(query_mask, 0.5, 0.0)
    else:
        denominator = tl.where(query_mask, 1.0, 0.0)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    zero_bias = tl.zeros((block_n,), dtype=tl.float32)

    coarse_total = num_frames * coarse_cells
    coarse_source_base = (
        (batch * heads + head) * coarse_total * head_dim
    )
    self_coarse_start = frame * coarse_cells
    for start in tl.range(0, coarse_cells, block_n):
        local_offsets = start + tl.arange(0, block_n)
        key_mask = local_offsets < coarse_cells
        source_cells = self_coarse_start + local_offsets
        source_offsets = (
            coarse_source_base
            + source_cells[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            coarse_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            coarse_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator, denominator = _subtract_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            softmax_scale_log2,
            weight=0.5 if complement_correction else 1.0,
        )

    residual_index_base = (
        (batch * num_frames + frame) * residual_tokens
    )
    if complement_correction:
        complement_index_base = (
            (batch * num_frames + frame) * complement_tokens
        )
        complement_bias = tl.full(
            (block_n,), -0.6931471805599453, dtype=tl.float32
        )
        for start in tl.range(0, complement_tokens, block_n):
            complement_offsets = start + tl.arange(0, block_n)
            key_mask = complement_offsets < complement_tokens
            source_cells = tl.load(
                complement_index_ptr
                + complement_index_base
                + complement_offsets,
                mask=key_mask,
                other=0,
            )
            source_offsets = (
                coarse_source_base
                + source_cells[:, None] * head_dim
                + dim_offsets[None, :]
            )
            key = tl.load(
                coarse_key_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            value = tl.load(
                coarse_value_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator, denominator, row_max = (
                _online_attention_update(
                    query,
                    key,
                    value,
                    accumulator,
                    denominator,
                    row_max,
                    query_mask,
                    key_mask,
                    complement_bias,
                    softmax_scale_log2,
                )
            )
    else:
        for start in tl.range(0, residual_tokens, block_n):
            selected_offsets = start + tl.arange(0, block_n)
            key_mask = selected_offsets < residual_tokens
            source_cells = tl.load(
                residual_index_ptr
                + residual_index_base
                + selected_offsets,
                mask=key_mask,
                other=0,
            )
            source_offsets = (
                coarse_source_base
                + source_cells[:, None] * head_dim
                + dim_offsets[None, :]
            )
            key = tl.load(
                coarse_key_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            value = tl.load(
                coarse_value_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator, denominator = _subtract_attention_update(
                query,
                key,
                value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                softmax_scale_log2,
                weight=0.5,
            )

    for start in tl.range(0, dense_tokens, block_n):
        key_offsets = start + tl.arange(0, block_n)
        key_mask = key_offsets < dense_tokens
        source_offsets = (
            query_base
            + key_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            dense_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            dense_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            zero_bias,
            softmax_scale_log2,
        )

    residual_source_base = (
        (batch * heads + head) * coarse_total * head_dim
    )
    residual_bias = tl.full(
        (block_n,), -0.6931471805599453, dtype=tl.float32
    )
    for start in tl.range(0, residual_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < residual_tokens
        source_cells = tl.load(
            residual_index_ptr + residual_index_base + selected_offsets,
            mask=key_mask,
            other=0,
        )
        source_offsets = (
            residual_source_base
            + source_cells[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            residual_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            residual_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            residual_bias,
            softmax_scale_log2,
        )

    special_source_base = (
        (batch * heads + head) * special_tokens * head_dim
    )
    for start in tl.range(0, special_tokens, block_n):
        key_offsets = start + tl.arange(0, block_n)
        key_mask = key_offsets < special_tokens
        source_offsets = (
            special_source_base
            + key_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            special_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            special_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            zero_bias,
            softmax_scale_log2,
        )

    output = accumulator / denominator[:, None]
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def hps_flash_carrier_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    coarse_key: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_key: torch.Tensor,
    residual_value: torch.Tensor,
    residual_indices: torch.Tensor,
    complement_indices: torch.Tensor | None = None,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    output_dtype: torch.dtype | None = None,
    complement_correction: bool = False,
) -> torch.Tensor:
    """Run all coarse carriers with FlashAttention and correct HPS structure."""
    sources = (
        query,
        dense_key,
        dense_value,
        coarse_key,
        coarse_value,
        residual_key,
        residual_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("HPS flash sources must be five-dimensional")
    if any(not source.is_cuda or not source.is_contiguous() for source in sources):
        raise ValueError("HPS flash sources must be contiguous CUDA tensors")
    if any(source.device != query.device for source in sources):
        raise ValueError("HPS flash sources must use one CUDA device")
    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if query.shape != dense_key.shape or query.shape != dense_value.shape:
        raise ValueError("HPS flash dense Q/K/V shapes must match")
    if coarse_key.shape != coarse_value.shape:
        raise ValueError("HPS flash coarse K/V shapes must match")
    if residual_key.shape != residual_value.shape:
        raise ValueError("HPS flash residual K/V shapes must match")
    if residual_key.shape != coarse_key.shape:
        raise ValueError("HPS flash requires one residual phase")
    if coarse_key.shape[:3] != (batch, heads, num_frames):
        raise ValueError("HPS flash coarse prefix must match query")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("HPS flash head dimensions must match")
    if any(source.dtype != query.dtype for source in (
        dense_key, coarse_key, residual_key
    )):
        raise ValueError("HPS flash Q/K sources must use one dtype")
    if any(source.dtype != dense_value.dtype for source in (
        coarse_value, residual_value
    )):
        raise ValueError("HPS flash V sources must use one dtype")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("HPS flash requires FP16 or BF16 Q/K")

    coarse_cells = coarse_key.shape[-2]
    if residual_indices.ndim != 3 or residual_indices.shape[:2] != (
        batch,
        num_frames,
    ):
        raise ValueError("HPS flash residual indices must have shape [B, F, K]")
    if residual_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("HPS flash residual indices must be integer")
    if residual_indices.device != query.device:
        raise ValueError("HPS flash residual indices must use the query device")
    residual_tokens = residual_indices.shape[-1]
    if residual_tokens < 1:
        raise ValueError("HPS flash requires residual tokens")
    if complement_correction:
        if (
            complement_indices is None
            or complement_indices.ndim != 3
            or complement_indices.shape[:2] != (batch, num_frames)
            or complement_indices.dtype not in {torch.int32, torch.int64}
            or complement_indices.device != query.device
        ):
            raise ValueError(
                "HPS complement indices must have shape [B, F, K]"
            )
        complement_indices = complement_indices.contiguous()
        complement_tokens = complement_indices.shape[-1]
    else:
        complement_indices = residual_indices
        complement_tokens = 0

    if (special_key is None) != (special_value is None):
        raise ValueError("HPS flash special K/V must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("HPS flash special K/V shapes must match")
        if (
            special_key.shape[:2] != (batch, heads)
            or special_key.shape[-1] != head_dim
            or not special_key.is_cuda
            or not special_value.is_cuda
            or not special_key.is_contiguous()
            or not special_value.is_contiguous()
            or special_key.device != query.device
            or special_value.device != query.device
            or special_key.dtype != query.dtype
            or special_value.dtype != dense_value.dtype
        ):
            raise ValueError("HPS flash special K/V must match query")
        special_tokens = special_key.shape[-2]

    flat_query = query.reshape(
        batch,
        heads,
        num_frames * dense_tokens,
        head_dim,
    )
    flat_coarse_key = coarse_key.reshape(
        batch,
        heads,
        num_frames * coarse_cells,
        head_dim,
    )
    flat_coarse_value = coarse_value.reshape_as(flat_coarse_key)
    flash_result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        flat_query,
        flat_coarse_key,
        flat_coarse_value,
        0.0,
        False,
        False,
        scale=1.0 / math.sqrt(head_dim),
    )
    carrier_output = flash_result[0].reshape_as(query).contiguous()
    carrier_lse = flash_result[1][
        ..., : num_frames * dense_tokens
    ].reshape(
        batch,
        heads,
        num_frames,
        dense_tokens,
    ).contiguous()

    if output_dtype is None:
        output_dtype = query.dtype
    output = torch.empty(
        query.shape,
        device=query.device,
        dtype=output_dtype,
    )
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _hps_flash_correction_kernel[grid](
        query,
        dense_key,
        dense_value,
        coarse_key,
        coarse_value,
        residual_key,
        residual_value,
        residual_indices.contiguous(),
        complement_indices,
        special_key,
        special_value,
        carrier_output,
        carrier_lse,
        output,
        1.0 / math.sqrt(head_dim),
        dense_tokens=dense_tokens,
        coarse_cells=coarse_cells,
        residual_tokens=residual_tokens,
        complement_tokens=complement_tokens,
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        complement_correction=complement_correction,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
