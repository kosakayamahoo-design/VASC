"""Descriptor-direct Triton attention for the main residual-debt layout."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _online_attention_update_logits(
    logits,
    value,
    accumulator,
    denominator,
    row_max,
    query_mask,
    key_mask,
):
    valid = query_mask[:, None] & key_mask[None, :]
    logits = tl.where(valid, logits, -float("inf"))

    block_max = tl.max(logits, axis=1)
    row_has_key = query_mask & (
        tl.sum(key_mask.to(tl.int32), axis=0) > 0
    )
    next_max = tl.where(
        row_has_key,
        tl.maximum(row_max, block_max),
        row_max,
    )
    correction = tl.where(
        row_has_key,
        tl.math.exp2(row_max - next_max),
        1.0,
    )
    shifted_logits = tl.where(
        valid,
        logits - next_max[:, None],
        -float("inf"),
    )
    probabilities = tl.math.exp2(shifted_logits)
    next_denominator = denominator * correction + tl.sum(
        probabilities, axis=1
    )
    accumulator = accumulator * correction[:, None]
    accumulator += tl.dot(probabilities.to(value.dtype), value)
    return accumulator, next_denominator, next_max


@triton.jit
def _online_attention_prepare_logits(
    logits,
    accumulator,
    denominator,
    row_max,
    query_mask,
    key_mask,
):
    """Update exact online-softmax state before an optional PV product."""
    valid = query_mask[:, None] & key_mask[None, :]
    logits = tl.where(valid, logits, -float("inf"))

    block_max = tl.max(logits, axis=1)
    row_has_key = query_mask & (
        tl.sum(key_mask.to(tl.int32), axis=0) > 0
    )
    next_max = tl.where(
        row_has_key,
        tl.maximum(row_max, block_max),
        row_max,
    )
    correction = tl.where(
        row_has_key,
        tl.math.exp2(row_max - next_max),
        1.0,
    )
    shifted_logits = tl.where(
        valid,
        logits - next_max[:, None],
        -float("inf"),
    )
    probabilities = tl.math.exp2(shifted_logits)
    next_denominator = denominator * correction + tl.sum(
        probabilities, axis=1
    )
    accumulator = accumulator * correction[:, None]
    block_peak_log2 = tl.max(
        tl.where(query_mask, block_max - next_max, -float("inf")),
        axis=0,
    )
    return (
        accumulator,
        next_denominator,
        next_max,
        probabilities,
        block_peak_log2,
    )


@triton.jit
def _online_attention_update(
    query,
    key,
    value,
    accumulator,
    denominator,
    row_max,
    query_mask,
    key_mask,
    log_bias,
    softmax_scale_log2,
):
    logits = tl.dot(query, tl.trans(key)) * softmax_scale_log2
    logits += log_bias[None, :] * 1.4426950408889634
    return _online_attention_update_logits(
        logits,
        value,
        accumulator,
        denominator,
        row_max,
        query_mask,
        key_mask,
    )


@triton.jit
def _descriptor_direct_attention_kernel(
    query_ptr,
    dense_key_ptr,
    dense_value_ptr,
    coarse_key_ptr,
    coarse_value_ptr,
    residual_key_ptr,
    residual_value_ptr,
    residual_index_ptr,
    refined_map_ptr,
    special_key_ptr,
    special_value_ptr,
    output_ptr,
    softmax_scale,
    dense_tokens: tl.constexpr,
    coarse_cells: tl.constexpr,
    residual_cells_per_frame: tl.constexpr,
    residual_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    mass_conserving: tl.constexpr,
    exact_mass_conserving: tl.constexpr,
    single_phase_layout: tl.constexpr,
    dot_qk_bf16: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
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
    if dot_qk_bf16:
        query = query.to(tl.bfloat16)

    accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    row_max = tl.where(query_mask, -float("inf"), 0.0)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    zero_bias = tl.zeros((block_n,), dtype=tl.float32)

    dense_base = query_base
    for start in tl.range(0, dense_tokens, block_n):
        key_offsets = start + tl.arange(0, block_n)
        key_mask = key_offsets < dense_tokens
        source_offsets = (
            dense_base
            + key_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            dense_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if dot_qk_bf16:
            key = key.to(tl.bfloat16)
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

    coarse_total = num_frames * coarse_cells
    coarse_source_base = (
        (batch * heads + head) * coarse_total * head_dim
    )
    refined_base = (batch * num_frames + frame) * coarse_total
    for start in tl.range(0, coarse_total, block_n):
        cell_offsets = start + tl.arange(0, block_n)
        source_frames = cell_offsets // coarse_cells
        key_mask = (cell_offsets < coarse_total) & (source_frames != frame)
        source_offsets = (
            coarse_source_base
            + cell_offsets[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            coarse_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if dot_qk_bf16:
            key = key.to(tl.bfloat16)
        value = tl.load(
            coarse_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if exact_mass_conserving:
            refinement_count = tl.load(
                refined_map_ptr + refined_base + cell_offsets,
                mask=cell_offsets < coarse_total,
                other=0,
            )
            key_mask = key_mask & (refinement_count == 0)
            log_bias = zero_bias
        elif mass_conserving:
            refinement_count = tl.load(
                refined_map_ptr + refined_base + cell_offsets,
                mask=cell_offsets < coarse_total,
                other=0,
            )
            if residual_cells_per_frame == coarse_cells:
                log_bias = tl.where(
                    refinement_count != 0, -0.6931471805599453, 0.0
                )
            else:
                log_bias = -tl.log(
                    1.0 + refinement_count.to(tl.float32)
                )
        else:
            log_bias = zero_bias
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            log_bias,
            softmax_scale_log2,
        )

    residual_total = num_frames * residual_cells_per_frame
    residual_source_base = (
        (batch * heads + head) * residual_total * head_dim
    )
    residual_index_base = (
        (batch * num_frames + frame) * residual_tokens
    )
    for start in tl.range(0, residual_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < residual_tokens
        source_cells = tl.load(
            residual_index_ptr + residual_index_base + selected_offsets,
            mask=key_mask,
            other=0,
        )
        if single_phase_layout:
            source_coarse_cells = source_cells
        else:
            source_frames = source_cells // residual_cells_per_frame
            source_phase_cells = source_cells % residual_cells_per_frame
            source_coarse_cells = (
                source_frames * coarse_cells
                + source_phase_cells % coarse_cells
            )
        source_offsets = (
            residual_source_base
            + source_cells[:, None] * head_dim
            + dim_offsets[None, :]
        )
        residual_key = tl.load(
            residual_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if dot_qk_bf16:
            residual_key = residual_key.to(tl.bfloat16)
        residual_value = tl.load(
            residual_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if exact_mass_conserving:
            coarse_key = tl.load(
                coarse_key_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            if dot_qk_bf16:
                coarse_key = coarse_key.to(tl.bfloat16)
            coarse_value = tl.load(
                coarse_value_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            coarse_logits = (
                tl.dot(query, tl.trans(coarse_key)) * softmax_scale_log2
            )
            residual_logits = (
                tl.dot(query, tl.trans(residual_key))
                * softmax_scale_log2
            )
            pair_max = tl.maximum(coarse_logits, residual_logits)
            pair_logsumexp = pair_max + tl.log2(
                tl.math.exp2(coarse_logits - pair_max)
                + tl.math.exp2(residual_logits - pair_max)
            )
            correction = coarse_logits - pair_logsumexp
            accumulator, denominator, row_max = (
                _online_attention_update_logits(
                    coarse_logits + correction,
                    coarse_value,
                    accumulator,
                    denominator,
                    row_max,
                    query_mask,
                    key_mask,
                )
            )
            accumulator, denominator, row_max = (
                _online_attention_update_logits(
                    residual_logits + correction,
                    residual_value,
                    accumulator,
                    denominator,
                    row_max,
                    query_mask,
                    key_mask,
                )
            )
        elif mass_conserving:
            refinement_count = tl.load(
                refined_map_ptr + refined_base + source_coarse_cells,
                mask=key_mask,
                other=0,
            )
            if residual_cells_per_frame == coarse_cells:
                log_bias = tl.full(
                    (block_n,), -0.6931471805599453, dtype=tl.float32
                )
            else:
                log_bias = -tl.log(
                    1.0 + refinement_count.to(tl.float32)
                )
            accumulator, denominator, row_max = _online_attention_update(
                query,
                residual_key,
                residual_value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                log_bias,
                softmax_scale_log2,
            )
        else:
            log_bias = zero_bias
            accumulator, denominator, row_max = _online_attention_update(
                query,
                residual_key,
                residual_value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                log_bias,
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
        if dot_qk_bf16:
            key = key.to(tl.bfloat16)
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
    output_offsets = (
        query_base
        + query_offsets[:, None] * head_dim
        + dim_offsets[None, :]
    )
    tl.store(
        output_ptr + output_offsets,
        output,
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def descriptor_direct_debt_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    coarse_key: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_key: torch.Tensor,
    residual_value: torch.Tensor,
    residual_indices: torch.Tensor,
    refined_map: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    mass_conserving: bool,
    exact_mass_conserving: bool = False,
    single_phase_layout: bool = False,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dot_qk_bf16: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute the exact mainline debt layout without compact K/V buffers."""
    if exact_mass_conserving and not mass_conserving:
        raise ValueError(
            "exact mass conservation requires mass_conserving=True"
        )
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
        raise ValueError("descriptor-direct K/V sources must be five-dimensional")
    if any(not source.is_cuda for source in sources):
        raise ValueError("descriptor-direct attention requires CUDA tensors")
    if any(not source.is_contiguous() for source in sources):
        raise ValueError("descriptor-direct attention requires contiguous sources")
    if any(source.device != query.device for source in sources):
        raise ValueError("descriptor-direct sources must use the same device")
    key_sources = (dense_key, coarse_key, residual_key)
    value_sources = (dense_value, coarse_value, residual_value)
    if any(source.dtype != query.dtype for source in key_sources):
        raise ValueError("descriptor-direct Q/K sources must use one dtype")
    if any(source.dtype != dense_value.dtype for source in value_sources):
        raise ValueError("descriptor-direct V sources must use one dtype")
    if query.shape != dense_key.shape or query.shape != dense_value.shape:
        raise ValueError("query and dense K/V shapes must match")
    if coarse_key.shape != coarse_value.shape:
        raise ValueError("coarse K/V shapes must match")
    if residual_key.shape != residual_value.shape:
        raise ValueError("residual K/V shapes must match")
    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    coarse_cells = coarse_key.shape[-2]
    if coarse_key.shape[:3] != (batch, heads, num_frames):
        raise ValueError("coarse K/V prefix must match query")
    if coarse_key.shape[-1] != head_dim:
        raise ValueError("all sources must have the same head dimension")
    if residual_key.shape[:3] != (batch, heads, num_frames):
        raise ValueError("residual K/V prefix must match query")
    if residual_key.shape[-1] != head_dim:
        raise ValueError("all sources must have the same head dimension")
    residual_cells_per_frame = residual_key.shape[-2]
    if residual_cells_per_frame % coarse_cells != 0:
        raise ValueError(
            "residual cells per frame must be a multiple of coarse cells"
        )
    residual_phases = residual_cells_per_frame // coarse_cells
    if single_phase_layout and residual_phases != 1:
        raise ValueError(
            "single-phase layout requires exactly one residual phase"
        )
    if exact_mass_conserving and residual_phases != 1:
        raise ValueError("exact mass conservation supports one residual phase")
    if residual_indices.ndim != 3 or residual_indices.shape[:2] != (
        batch,
        num_frames,
    ):
        raise ValueError(
            "residual indices must have shape [batch, frames, selected]"
        )
    if residual_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("residual indices must use an integer dtype")
    if residual_indices.device != query.device:
        raise ValueError("residual indices must use the query device")
    expected_map_shape = (batch, num_frames, num_frames * coarse_cells)
    if refined_map.shape != expected_map_shape:
        raise ValueError(
            f"refined map must have shape {expected_map_shape}"
        )
    if refined_map.device != query.device or not refined_map.is_contiguous():
        raise ValueError("refined map must be contiguous on the query device")
    if refined_map.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    }:
        raise ValueError("refined map must contain integer refinement counts")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("special K/V shapes must match")
        if special_key.shape[:2] != (batch, heads):
            raise ValueError("special K/V prefix must match query")
        if special_key.shape[-1] != head_dim:
            raise ValueError("special K/V head dimension must match query")
        if (
            not special_key.is_cuda
            or not special_value.is_cuda
            or not special_key.is_contiguous()
            or not special_value.is_contiguous()
            or special_key.device != query.device
            or special_value.device != query.device
            or special_key.dtype != query.dtype
            or special_value.dtype != dense_value.dtype
        ):
            raise ValueError("special K/V must match query device and dtype")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("descriptor-direct attention supports head_dim <= 128")
    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported descriptor-direct block shape")
    launch_warps = 4 if block_m <= 32 else 8
    if num_warps is not None:
        if num_warps not in {4, 8}:
            raise ValueError("descriptor-direct num_warps must be 4 or 8")
        launch_warps = num_warps
    launch_stages = 2 if query.dtype == torch.float32 else 3
    if num_stages is not None:
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError(
                "descriptor-direct num_stages must be 1, 2, 3, or 4"
            )
        launch_stages = num_stages
    residual_tokens = residual_indices.shape[-1]
    if residual_tokens < 1:
        raise ValueError("descriptor-direct attention requires residual tokens")

    if output_dtype is None:
        output_dtype = torch.bfloat16 if dot_qk_bf16 else query.dtype
    if not output_dtype.is_floating_point:
        raise ValueError("descriptor-direct output must use a floating dtype")
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
    _descriptor_direct_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        coarse_key,
        coarse_value,
        residual_key,
        residual_value,
        residual_indices.contiguous(),
        refined_map,
        special_key,
        special_value,
        output,
        1.0 / math.sqrt(head_dim),
        dense_tokens=dense_tokens,
        coarse_cells=coarse_cells,
        residual_cells_per_frame=residual_cells_per_frame,
        residual_tokens=residual_tokens,
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        mass_conserving=mass_conserving,
        exact_mass_conserving=exact_mass_conserving,
        single_phase_layout=single_phase_layout,
        dot_qk_bf16=dot_qk_bf16,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    return output
