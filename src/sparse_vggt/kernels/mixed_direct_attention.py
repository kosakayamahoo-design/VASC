"""Descriptor-direct Triton attention for mixed 4x4/2x2 execution."""

import math

import torch
import triton
import triton.language as tl

from sparse_vggt.kernels.direct_attention import (
    _online_attention_prepare_logits,
    _online_attention_update,
)


@triton.jit
def _compiled_carrier_attention_kernel(
    query_ptr,
    dense_key_ptr,
    dense_value_ptr,
    parent_key_ptr,
    parent_value_ptr,
    child_key_ptr,
    child_value_ptr,
    parent_index_ptr,
    parent_log_bias_ptr,
    child_index_ptr,
    child_log_bias_ptr,
    special_key_ptr,
    special_value_ptr,
    output_ptr,
    softmax_scale,
    dense_tokens: tl.constexpr,
    parent_cells: tl.constexpr,
    parent_tokens: tl.constexpr,
    child_cells: tl.constexpr,
    child_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    prevalidated_remote_layout: tl.constexpr,
    value_bf16: tl.constexpr,
    block_m: tl.constexpr,
    dense_block_n: tl.constexpr,
    parent_block_n: tl.constexpr,
    child_block_n: tl.constexpr,
    block_d: tl.constexpr,
):
    """Carrier-specialized attention with a tile size per service segment."""
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

    accumulator = tl.zeros((block_m, block_d), dtype=tl.float32)
    denominator = tl.zeros((block_m,), dtype=tl.float32)
    row_max = tl.where(query_mask, -float("inf"), 0.0)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    dense_zero_bias = tl.zeros((dense_block_n,), dtype=tl.float32)

    for start in tl.range(0, dense_tokens, dense_block_n):
        key_offsets = start + tl.arange(0, dense_block_n)
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
        if value_bf16:
            value = value.to(tl.bfloat16)
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            dense_zero_bias,
            softmax_scale_log2,
        )

    parent_source_base = (
        (batch * heads + head) * num_frames * parent_cells * head_dim
    )
    parent_descriptor_base = (
        (batch * num_frames + frame) * parent_tokens
    )
    for start in tl.range(0, parent_tokens, parent_block_n):
        selected_offsets = start + tl.arange(0, parent_block_n)
        key_mask = selected_offsets < parent_tokens
        source_parents = tl.load(
            parent_index_ptr + parent_descriptor_base + selected_offsets,
            mask=key_mask,
            other=0,
        )
        parent_bias = tl.load(
            parent_log_bias_ptr
            + parent_descriptor_base
            + selected_offsets,
            mask=key_mask,
            other=-float("inf"),
        ).to(tl.float32)
        if not prevalidated_remote_layout:
            source_frames = source_parents // parent_cells
            key_mask = key_mask & (source_frames != frame)
        source_offsets = (
            parent_source_base
            + source_parents[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            parent_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            parent_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if value_bf16:
            value = value.to(tl.bfloat16)
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            parent_bias,
            softmax_scale_log2,
        )

    child_source_base = (
        (batch * heads + head) * num_frames * child_cells * head_dim
    )
    child_descriptor_base = (
        (batch * num_frames + frame) * child_tokens
    )
    for start in tl.range(0, child_tokens, child_block_n):
        selected_offsets = start + tl.arange(0, child_block_n)
        key_mask = selected_offsets < child_tokens
        source_children = tl.load(
            child_index_ptr + child_descriptor_base + selected_offsets,
            mask=key_mask,
            other=0,
        )
        child_bias = tl.load(
            child_log_bias_ptr
            + child_descriptor_base
            + selected_offsets,
            mask=key_mask,
            other=-float("inf"),
        ).to(tl.float32)
        if not prevalidated_remote_layout:
            source_frames = source_children // child_cells
            key_mask = key_mask & (source_frames != frame)
        source_offsets = (
            child_source_base
            + source_children[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            child_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            child_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if value_bf16:
            value = value.to(tl.bfloat16)
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            child_bias,
            softmax_scale_log2,
        )

    special_source_base = (
        (batch * heads + head) * special_tokens * head_dim
    )
    special_zero_bias = tl.zeros((dense_block_n,), dtype=tl.float32)
    for start in tl.range(0, special_tokens, dense_block_n):
        key_offsets = start + tl.arange(0, dense_block_n)
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
        if value_bf16:
            value = value.to(tl.bfloat16)
        accumulator, denominator, row_max = _online_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            special_zero_bias,
            softmax_scale_log2,
        )

    output = accumulator / denominator[:, None]
    output_base = (
        ((batch * heads + head) * num_frames + frame)
        * dense_tokens
        * head_dim
    )
    tl.store(
        output_ptr
        + output_base
        + query_offsets[:, None] * head_dim
        + dim_offsets[None, :],
        output,
        mask=query_mask[:, None] & dim_mask[None, :],
    )


@triton.jit
def _mixed_descriptor_direct_attention_kernel(
    query_ptr,
    dense_key_ptr,
    dense_value_ptr,
    parent_key_ptr,
    parent_value_ptr,
    child_key_ptr,
    child_value_ptr,
    residual_key_ptr,
    residual_value_ptr,
    easy_parent_index_ptr,
    hard_child_index_ptr,
    parent_log_bias_ptr,
    child_log_bias_ptr,
    residual_index_ptr,
    refined_child_map_ptr,
    service_parent_index_ptr,
    service_child_mask_ptr,
    parent_to_child_ptr,
    child_to_parent_ptr,
    child_log_mass_ptr,
    special_key_ptr,
    special_value_ptr,
    output_ptr,
    two_stage_stats_ptr,
    pv_execution_map_ptr,
    carrier_observer_stats_ptr,
    softmax_scale,
    pv_log2_threshold,
    dense_tokens: tl.constexpr,
    parent_cells: tl.constexpr,
    easy_parent_tokens: tl.constexpr,
    child_cells: tl.constexpr,
    hard_child_tokens: tl.constexpr,
    residual_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    dot_qk_bf16: tl.constexpr,
    indexed_service_bias: tl.constexpr,
    parent_mask_service: tl.constexpr,
    implicit_parent_layout: tl.constexpr,
    prevalidated_remote_layout: tl.constexpr,
    two_stage_qk_pv: tl.constexpr,
    collect_two_stage_stats: tl.constexpr,
    return_pv_execution_map: tl.constexpr,
    observe_carrier_compensation: tl.constexpr,
    carrier_compensated_pv: tl.constexpr,
    query_blocks: tl.constexpr,
    child_blocks: tl.constexpr,
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

    parent_total = num_frames * parent_cells
    parent_source_base = (
        (batch * heads + head) * parent_total * head_dim
    )
    easy_parent_base = (
        (batch * num_frames + frame) * easy_parent_tokens
    )
    parent_bias_base = (
        (batch * num_frames + frame) * easy_parent_tokens
    )
    for start in tl.range(0, easy_parent_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < easy_parent_tokens
        if implicit_parent_layout:
            source_parents = selected_offsets
        else:
            source_parents = tl.load(
                easy_parent_index_ptr
                + easy_parent_base
                + selected_offsets,
                mask=key_mask,
                other=0,
            )
        if indexed_service_bias:
            parent_bias = tl.load(
                parent_log_bias_ptr
                + parent_bias_base
                + selected_offsets,
                mask=key_mask,
                other=-float("inf"),
            ).to(tl.float32)
            if implicit_parent_layout:
                key_mask = key_mask & (
                    parent_bias != -float("inf")
                )
        else:
            parent_bias = tl.full(
                (block_n,), 1.3862943611198906, dtype=tl.float32
            )
        source_offsets = (
            parent_source_base
            + source_parents[:, None] * head_dim
            + dim_offsets[None, :]
        )
        key = tl.load(
            parent_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        if dot_qk_bf16:
            key = key.to(tl.bfloat16)
        value = tl.load(
            parent_value_ptr + source_offsets,
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
            parent_bias,
            softmax_scale_log2,
        )

    child_total = num_frames * child_cells
    child_source_base = (
        (batch * heads + head) * child_total * head_dim
    )
    hard_child_base = (
        (batch * num_frames + frame) * hard_child_tokens
    )
    child_bias_base = hard_child_base
    refined_child_base = (
        (batch * num_frames + frame) * child_total
    )
    if parent_mask_service:
        service_parent_tokens: tl.constexpr = hard_child_tokens // 3
        service_parent_base = (
            (batch * num_frames + frame) * service_parent_tokens
        )
        for child_rank in tl.static_range(0, 3):
            for start in tl.range(0, service_parent_tokens, block_n):
                service_offsets = start + tl.arange(0, block_n)
                key_mask = service_offsets < service_parent_tokens
                source_parents = tl.load(
                    service_parent_index_ptr
                    + service_parent_base
                    + service_offsets,
                    mask=key_mask,
                    other=0,
                )
                service_masks = tl.load(
                    service_child_mask_ptr
                    + service_parent_base
                    + service_offsets,
                    mask=key_mask,
                    other=0,
                )
                bit0 = service_masks & 1
                bit1 = (service_masks >> 1) & 1
                bit2 = (service_masks >> 2) & 1
                bit3 = (service_masks >> 3) & 1
                selected_count = bit0 + bit1 + bit2 + bit3
                key_mask = key_mask & (child_rank < selected_count)
                child_slots = tl.where(
                    child_rank < bit0,
                    0,
                    tl.where(
                        child_rank < bit0 + bit1,
                        1,
                        tl.where(child_rank < bit0 + bit1 + bit2, 2, 3),
                    ),
                )
                source_parent_ids = source_parents % parent_cells
                source_frames = source_parents // parent_cells
                source_child_ids = tl.load(
                    parent_to_child_ptr
                    + source_parent_ids * 4
                    + child_slots,
                    mask=key_mask,
                    other=0,
                )
                source_cells = (
                    source_frames * child_cells + source_child_ids
                )
                if not prevalidated_remote_layout:
                    key_mask = key_mask & (source_frames != frame)
                source_offsets = (
                    child_source_base
                    + source_cells[:, None] * head_dim
                    + dim_offsets[None, :]
                )
                key = tl.load(
                    child_key_ptr + source_offsets,
                    mask=key_mask[:, None] & dim_mask[None, :],
                    other=0.0,
                )
                if dot_qk_bf16:
                    key = key.to(tl.bfloat16)
                value = tl.load(
                    child_value_ptr + source_offsets,
                    mask=key_mask[:, None] & dim_mask[None, :],
                    other=0.0,
                )
                child_bias = tl.load(
                    child_log_mass_ptr + source_child_ids,
                    mask=key_mask,
                    other=-float("inf"),
                ).to(tl.float32)
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
                        child_bias,
                        softmax_scale_log2,
                    )
                )
    else:
        for start in tl.range(0, hard_child_tokens, block_n):
            selected_offsets = start + tl.arange(0, block_n)
            key_mask = selected_offsets < hard_child_tokens
            source_cells = tl.load(
                hard_child_index_ptr + hard_child_base + selected_offsets,
                mask=key_mask,
                other=0,
            )
            if not prevalidated_remote_layout:
                source_frames = source_cells // child_cells
                key_mask = key_mask & (source_frames != frame)
            source_offsets = (
                child_source_base
                + source_cells[:, None] * head_dim
                + dim_offsets[None, :]
            )
            key = tl.load(
                child_key_ptr + source_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            if dot_qk_bf16:
                key = key.to(tl.bfloat16)
            if indexed_service_bias:
                child_bias = tl.load(
                    child_log_bias_ptr
                    + child_bias_base
                    + selected_offsets,
                    mask=key_mask,
                    other=-float("inf"),
                ).to(tl.float32)
            else:
                refined = tl.load(
                    refined_child_map_ptr
                    + refined_child_base
                    + source_cells,
                    mask=key_mask,
                    other=0,
                )
                child_bias = tl.where(
                    refined != 0, -0.6931471805599453, 0.0
                )
            if two_stage_qk_pv:
                logits = (
                    tl.dot(query, tl.trans(key)) * softmax_scale_log2
                )
                logits += child_bias[None, :] * 1.4426950408889634
                (
                    accumulator,
                    denominator,
                    row_max,
                    probabilities,
                    block_peak_log2,
                ) = _online_attention_prepare_logits(
                    logits,
                    accumulator,
                    denominator,
                    row_max,
                    query_mask,
                    key_mask,
                )
                execute_pv = block_peak_log2 >= pv_log2_threshold
                if return_pv_execution_map:
                    child_block = start // block_n
                    execution_offset = (
                        (frame_program * query_blocks + query_block)
                        * child_blocks
                        + child_block
                    )
                    tl.store(
                        pv_execution_map_ptr + execution_offset,
                        execute_pv.to(tl.int32),
                    )
                if collect_two_stage_stats:
                    has_keys = (
                        tl.sum(key_mask.to(tl.int32), axis=0) > 0
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 0,
                        has_keys.to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 1,
                        (has_keys & execute_pv).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 2,
                        (has_keys & (block_peak_log2 < -1.0)).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 3,
                        (has_keys & (block_peak_log2 < -2.0)).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 4,
                        (has_keys & (block_peak_log2 < -3.0)).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 5,
                        (has_keys & (block_peak_log2 < -4.0)).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 6,
                        (has_keys & (block_peak_log2 < -6.0)).to(tl.int32),
                    )
                    tl.atomic_add(
                        two_stage_stats_ptr + 7,
                        (has_keys & (block_peak_log2 < -8.0)).to(tl.int32),
                    )
                if execute_pv:
                    value = tl.load(
                        child_value_ptr + source_offsets,
                        mask=key_mask[:, None] & dim_mask[None, :],
                        other=0.0,
                    )
                    accumulator += tl.dot(
                        probabilities.to(value.dtype), value
                    )
                elif observe_carrier_compensation or carrier_compensated_pv:
                    value = tl.load(
                        child_value_ptr + source_offsets,
                        mask=key_mask[:, None] & dim_mask[None, :],
                        other=0.0,
                    )
                    source_child_ids = source_cells % child_cells
                    source_frames = source_cells // child_cells
                    source_parent_ids = tl.load(
                        child_to_parent_ptr + source_child_ids,
                        mask=key_mask,
                        other=0,
                    )
                    carrier_cells = (
                        source_frames * parent_cells + source_parent_ids
                    )
                    carrier_offsets = (
                        parent_source_base
                        + carrier_cells[:, None] * head_dim
                        + dim_offsets[None, :]
                    )
                    carrier_value = tl.load(
                        parent_value_ptr + carrier_offsets,
                        mask=key_mask[:, None] & dim_mask[None, :],
                        other=0.0,
                    )
                    probability_value = probabilities.to(value.dtype)
                    exact_contribution = tl.dot(
                        probability_value, value
                    )
                    carrier_contribution = tl.dot(
                        probability_value, carrier_value
                    )
                    if carrier_compensated_pv:
                        accumulator += carrier_contribution
                    contribution_mask = (
                        query_mask[:, None] & dim_mask[None, :]
                    )
                    exact_contribution = tl.where(
                        contribution_mask, exact_contribution, 0.0
                    )
                    carrier_error = tl.where(
                        contribution_mask,
                        exact_contribution - carrier_contribution,
                        0.0,
                    )
                    value_innovation = tl.where(
                        key_mask[:, None] & dim_mask[None, :],
                        value - carrier_value,
                        0.0,
                    )
                    value_innovation_fp32 = value_innovation.to(tl.float32)
                    innovation_sq = tl.sum(
                        value_innovation_fp32 * value_innovation_fp32,
                        axis=1,
                    )
                    innovation_norm = tl.sqrt(innovation_sq)
                    weighted_innovation = tl.sum(
                        probabilities
                        * innovation_sq[None, :]
                        * query_mask[:, None].to(tl.float32),
                        axis=1,
                    )
                    bound_per_query = tl.sum(
                        probabilities
                        * innovation_norm[None, :]
                        * query_mask[:, None].to(tl.float32),
                        axis=1,
                    )
                    bound_norm = tl.sqrt(
                        tl.sum(bound_per_query * bound_per_query, axis=0)
                    )
                    zero_tile_sq = tl.sum(
                        tl.sum(
                            exact_contribution * exact_contribution,
                            axis=1,
                        ),
                        axis=0,
                    )
                    carrier_tile_sq = tl.sum(
                        tl.sum(carrier_error * carrier_error, axis=1),
                        axis=0,
                    )
                    zero_tile_norm = tl.sqrt(zero_tile_sq)
                    carrier_tile_norm = tl.sqrt(carrier_tile_sq)
                    tl.atomic_add(carrier_observer_stats_ptr + 0, 1.0)
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 1,
                        tl.sum(query_mask.to(tl.float32), axis=0),
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 2,
                        tl.sum(
                            tl.sum(
                                probabilities
                                * query_mask[:, None].to(tl.float32),
                                axis=1,
                            ),
                            axis=0,
                        ),
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 3,
                        zero_tile_sq,
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 4,
                        carrier_tile_sq,
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 5,
                        tl.sum(weighted_innovation, axis=0),
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 6,
                        (carrier_tile_sq > zero_tile_sq).to(tl.float32),
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 7, bound_norm
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 8,
                        carrier_tile_norm,
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 9,
                        bound_norm * bound_norm,
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 10,
                        bound_norm * carrier_tile_norm,
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 11, zero_tile_norm
                    )
                    tl.atomic_add(
                        carrier_observer_stats_ptr + 12,
                        bound_norm * zero_tile_norm,
                    )
            else:
                value = tl.load(
                    child_value_ptr + source_offsets,
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
                        child_bias,
                        softmax_scale_log2,
                    )
                )

    residual_source_base = (
        (batch * heads + head) * child_total * head_dim
    )
    residual_index_base = (
        (batch * num_frames + frame) * residual_tokens
    )
    if not parent_mask_service:
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
        if parent_mask_service:
            residual_bias = tl.load(
                refined_child_map_ptr
                + residual_index_base
                + selected_offsets,
                mask=key_mask,
                other=-float("inf"),
            ).to(tl.float32)
            key_mask = key_mask & (residual_bias != -float("inf"))
        if not prevalidated_remote_layout:
            source_frames = source_cells // child_cells
            key_mask = key_mask & (source_frames != frame)
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
        if dot_qk_bf16:
            key = key.to(tl.bfloat16)
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


def mixed_descriptor_direct_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    residual_key: torch.Tensor,
    residual_value: torch.Tensor,
    easy_parent_indices: torch.Tensor,
    hard_child_indices: torch.Tensor,
    residual_indices: torch.Tensor,
    refined_child_map: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dot_qk_bf16: bool = False,
    prevalidated_remote_layout: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute mixed parent/child attention without compact K/V buffers."""
    sources = (
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        residual_key,
        residual_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("mixed direct sources must be five-dimensional")
    if any(not source.is_cuda or not source.is_contiguous() for source in sources):
        raise ValueError("mixed direct sources must be contiguous CUDA tensors")
    if any(source.device != query.device for source in sources):
        raise ValueError("mixed direct sources must use one CUDA device")
    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("mixed direct dense Q/K/V shapes must match")
    parent_cells = parent_key.shape[-2]
    child_cells = child_key.shape[-2]
    if parent_key.shape != parent_value.shape:
        raise ValueError("mixed direct parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("mixed direct child K/V shapes must match")
    if residual_key.shape != residual_value.shape:
        raise ValueError("mixed direct residual K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("mixed direct frame prefixes must match")
    if residual_key.shape[-2] != child_cells:
        raise ValueError("mixed direct residual cells must match child cells")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("mixed direct head dimensions must match")
    if any(source.dtype != query.dtype for source in (
        dense_key, parent_key, child_key, residual_key
    )):
        raise ValueError("mixed direct Q/K sources must use one dtype")
    if any(source.dtype != dense_value.dtype for source in (
        parent_value, child_value, residual_value
    )):
        raise ValueError("mixed direct V sources must use one dtype")

    parent_total = num_frames * parent_cells
    child_total = num_frames * child_cells
    if refined_child_map.shape != (batch, num_frames, child_total):
        raise ValueError("mixed hard-child refinement mask has the wrong shape")
    index_tensors = (
        easy_parent_indices,
        hard_child_indices,
        residual_indices,
    )
    if any(
        tensor.ndim != 3
        or tensor.shape[:2] != (batch, num_frames)
        or tensor.device != query.device
        or tensor.dtype not in {torch.int32, torch.int64}
        for tensor in index_tensors
    ):
        raise ValueError("mixed direct indices must have shape [B, F, K]")
    if any(
        tensor.device != query.device or not tensor.is_contiguous()
        for tensor in (refined_child_map,)
    ):
        raise ValueError("mixed direct maps must be contiguous on the query device")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("mixed direct special K/V shapes must match")
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
            raise ValueError("mixed direct special K/V must match query")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("mixed direct attention supports head_dim <= 128")
    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported mixed direct block shape")
    launch_warps = 4 if block_m <= 32 else 8
    if num_warps is not None:
        if num_warps not in {4, 8}:
            raise ValueError("mixed direct num_warps must be 4 or 8")
        launch_warps = num_warps
    launch_stages = 2 if query.dtype == torch.float32 else 3
    if num_stages is not None:
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError("mixed direct num_stages must be in [1, 4]")
        launch_stages = num_stages

    if output_dtype is None:
        output_dtype = torch.bfloat16 if dot_qk_bf16 else query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _mixed_descriptor_direct_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        residual_key,
        residual_value,
        easy_parent_indices.contiguous(),
        hard_child_indices.contiguous(),
        easy_parent_indices,
        hard_child_indices,
        residual_indices.contiguous(),
        refined_child_map,
        residual_indices,
        residual_indices,
        residual_indices,
        residual_indices,
        residual_indices,
        special_key,
        special_value,
        output,
        output,
        output,
        output,
        1.0 / math.sqrt(head_dim),
        -float("inf"),
        dense_tokens=dense_tokens,
        parent_cells=parent_cells,
        easy_parent_tokens=easy_parent_indices.shape[-1],
        child_cells=child_cells,
        hard_child_tokens=hard_child_indices.shape[-1],
        residual_tokens=residual_indices.shape[-1],
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        dot_qk_bf16=dot_qk_bf16,
        indexed_service_bias=False,
        parent_mask_service=False,
        implicit_parent_layout=False,
        prevalidated_remote_layout=prevalidated_remote_layout,
        two_stage_qk_pv=False,
        collect_two_stage_stats=False,
        return_pv_execution_map=False,
        observe_carrier_compensation=False,
        carrier_compensated_pv=False,
        query_blocks=triton.cdiv(dense_tokens, block_m),
        child_blocks=triton.cdiv(hard_child_indices.shape[-1], block_n),
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    return output


def additive_descriptor_direct_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    parent_indices: torch.Tensor,
    parent_log_bias: torch.Tensor,
    child_indices: torch.Tensor,
    child_log_bias: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    child_to_parent: torch.Tensor | None = None,
    block_m: int = 128,
    block_n: int = 128,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dot_qk_bf16: bool = False,
    prevalidated_remote_layout: bool = False,
    two_stage_qk_pv: bool = False,
    pv_log2_threshold: float = -4.0,
    collect_two_stage_stats: bool = False,
    return_pv_execution_map: bool = False,
    observe_carrier_compensation: bool = False,
    carrier_compensated_pv: bool = False,
    output_dtype: torch.dtype | None = None,
) -> (
    torch.Tensor
    | tuple[torch.Tensor, dict[str, float]]
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, dict[str, float], torch.Tensor]
):
    """Execute additive parent/child service without compact K/V buffers."""
    sources = (
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("additive direct sources must be five-dimensional")
    if any(
        not source.is_cuda or not source.is_contiguous()
        for source in sources
    ):
        raise ValueError(
            "additive direct sources must be contiguous CUDA tensors"
        )
    if any(source.device != query.device for source in sources):
        raise ValueError("additive direct sources must use one CUDA device")

    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("additive direct dense Q/K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("additive direct frame prefixes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("additive direct parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("additive direct child K/V shapes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("additive direct head dimensions must match")
    if any(
        source.dtype != query.dtype
        for source in (dense_key, parent_key, child_key)
    ):
        raise ValueError("additive direct Q/K sources must use one dtype")
    if any(
        source.dtype != dense_value.dtype
        for source in (parent_value, child_value)
    ):
        raise ValueError("additive direct V sources must use one dtype")

    descriptor_pairs = (
        (parent_indices, parent_log_bias),
        (child_indices, child_log_bias),
    )
    for indices, log_bias in descriptor_pairs:
        if (
            indices.ndim != 3
            or indices.shape[:2] != (batch, num_frames)
            or indices.device != query.device
            or indices.dtype not in {
                torch.int32,
                torch.int64,
                torch.uint16,
            }
        ):
            raise ValueError(
                "additive direct indices must have shape [B, F, K]"
            )
        if (
            log_bias.shape != indices.shape
            or log_bias.device != query.device
            or not log_bias.dtype.is_floating_point
        ):
            raise ValueError(
                "additive direct log bias must match its indices"
            )

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("additive direct special K/V shapes must match")
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
            raise ValueError("additive direct special K/V must match query")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("additive direct attention supports head_dim <= 128")
    if two_stage_qk_pv and pv_log2_threshold > 0.0:
        raise ValueError("PV log2 threshold must be non-positive")
    if collect_two_stage_stats and not two_stage_qk_pv:
        raise ValueError("two-stage stats require two_stage_qk_pv=True")
    if return_pv_execution_map and not two_stage_qk_pv:
        raise ValueError("PV execution map requires two_stage_qk_pv=True")
    if observe_carrier_compensation and not (
        two_stage_qk_pv and collect_two_stage_stats
    ):
        raise ValueError(
            "carrier compensation observation requires two-stage stats"
        )
    if carrier_compensated_pv and not two_stage_qk_pv:
        raise ValueError("carrier-compensated PV requires two-stage execution")
    if observe_carrier_compensation or carrier_compensated_pv:
        if (
            child_to_parent is None
            or child_to_parent.shape != (child_key.shape[-2],)
            or child_to_parent.device != query.device
            or child_to_parent.dtype not in {torch.int32, torch.int64}
        ):
            raise ValueError(
                "child_to_parent must map every child cell on the CUDA device"
            )
        child_to_parent = child_to_parent.contiguous()
    else:
        child_to_parent = child_indices
    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported additive direct block shape")
    launch_warps = 4 if block_m <= 32 else 8
    if num_warps is not None:
        if num_warps not in {4, 8}:
            raise ValueError("additive direct num_warps must be 4 or 8")
        launch_warps = num_warps
    launch_stages = 2 if query.dtype == torch.float32 else 3
    if num_stages is not None:
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError("additive direct num_stages must be in [1, 4]")
        launch_stages = num_stages

    if output_dtype is None:
        output_dtype = torch.bfloat16 if dot_qk_bf16 else query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    two_stage_stats = (
        torch.zeros(8, device=query.device, dtype=torch.int32)
        if collect_two_stage_stats
        else output
    )
    block_d = triton.next_power_of_2(head_dim)
    query_blocks = triton.cdiv(dense_tokens, block_m)
    child_blocks = triton.cdiv(child_indices.shape[-1], block_n)
    pv_execution_map = (
        torch.empty(
            batch,
            heads,
            num_frames,
            query_blocks,
            child_blocks,
            device=query.device,
            dtype=torch.int32,
        )
        if return_pv_execution_map
        else output
    )
    carrier_observer_stats = (
        torch.zeros(13, device=query.device, dtype=torch.float32)
        if observe_carrier_compensation
        else output
    )
    grid = (
        query_blocks,
        batch * heads * num_frames,
    )
    _mixed_descriptor_direct_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        child_key,
        child_value,
        parent_indices.contiguous(),
        child_indices.contiguous(),
        parent_log_bias.contiguous(),
        child_log_bias.contiguous(),
        child_indices,
        child_indices,
        child_indices,
        child_indices,
        child_indices,
        child_to_parent,
        child_log_bias,
        special_key,
        special_value,
        output,
        two_stage_stats,
        pv_execution_map,
        carrier_observer_stats,
        1.0 / math.sqrt(head_dim),
        pv_log2_threshold,
        dense_tokens=dense_tokens,
        parent_cells=parent_key.shape[-2],
        easy_parent_tokens=parent_indices.shape[-1],
        child_cells=child_key.shape[-2],
        hard_child_tokens=child_indices.shape[-1],
        residual_tokens=0,
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        dot_qk_bf16=dot_qk_bf16,
        indexed_service_bias=True,
        parent_mask_service=False,
        implicit_parent_layout=False,
        prevalidated_remote_layout=prevalidated_remote_layout,
        two_stage_qk_pv=two_stage_qk_pv,
        collect_two_stage_stats=collect_two_stage_stats,
        return_pv_execution_map=return_pv_execution_map,
        observe_carrier_compensation=observe_carrier_compensation,
        carrier_compensated_pv=carrier_compensated_pv,
        query_blocks=query_blocks,
        child_blocks=child_blocks,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    execution_fraction = None
    if return_pv_execution_map:
        execution_fraction = pv_execution_map.float().mean(dim=(1, 3))
    if not collect_two_stage_stats and execution_fraction is None:
        return output
    if not collect_two_stage_stats:
        return output, execution_fraction
    counts = two_stage_stats.to(device="cpu", dtype=torch.float64).tolist()
    candidates = counts[0]
    denominator = max(candidates, 1.0)
    stats = {
        "qk_candidate_tiles": candidates,
        "pv_executed_tiles": counts[1],
        "pv_executed_fraction": counts[1] / denominator,
        "pv_skippable_fraction_gap_lt_1": counts[2] / denominator,
        "pv_skippable_fraction_gap_lt_2": counts[3] / denominator,
        "pv_skippable_fraction_gap_lt_3": counts[4] / denominator,
        "pv_skippable_fraction_gap_lt_4": counts[5] / denominator,
        "pv_skippable_fraction_gap_lt_6": counts[6] / denominator,
        "pv_skippable_fraction_gap_lt_8": counts[7] / denominator,
    }
    if observe_carrier_compensation:
        carrier_counts = carrier_observer_stats.to(
            device="cpu", dtype=torch.float64
        ).tolist()
        zero_sq = max(carrier_counts[3], 0.0)
        carrier_sq = max(carrier_counts[4], 0.0)
        probability_mass = max(carrier_counts[2], 0.0)
        zero_l2 = math.sqrt(zero_sq)
        carrier_l2 = math.sqrt(carrier_sq)
        residual_ratio = carrier_l2 / max(zero_l2, 1e-12)
        stats.update(
            {
                "carrier_observer_skipped_tiles": carrier_counts[0],
                "carrier_observer_skipped_query_rows": carrier_counts[1],
                "carrier_observer_probability_mass": probability_mass,
                "carrier_observer_zero_residual_l2": zero_l2,
                "carrier_observer_compensated_residual_l2": carrier_l2,
                "carrier_observer_residual_ratio": residual_ratio,
                "carrier_observer_relative_error_reduction": (
                    1.0 - residual_ratio
                ),
                "carrier_observer_value_innovation_rms": math.sqrt(
                    max(carrier_counts[5], 0.0)
                    / max(probability_mass, 1e-12)
                ),
            }
        )
        skipped_tiles = max(carrier_counts[0], 1.0)

        def correlation(sum_x, sum_y, sum_x2, sum_y2, sum_xy):
            covariance = sum_xy - sum_x * sum_y / skipped_tiles
            variance_x = max(
                sum_x2 - sum_x * sum_x / skipped_tiles, 0.0
            )
            variance_y = max(
                sum_y2 - sum_y * sum_y / skipped_tiles, 0.0
            )
            return covariance / max(
                math.sqrt(variance_x * variance_y), 1e-12
            )

        stats.update(
            {
                "carrier_observer_unsafe_tile_fraction": (
                    carrier_counts[6] / skipped_tiles
                ),
                "carrier_observer_residual_bound_mean": (
                    carrier_counts[7] / skipped_tiles
                ),
                "carrier_observer_bound_carrier_residual_correlation": (
                    correlation(
                        carrier_counts[7],
                        carrier_counts[8],
                        carrier_counts[9],
                        carrier_counts[4],
                        carrier_counts[10],
                    )
                ),
                "carrier_observer_bound_zero_residual_correlation": (
                    correlation(
                        carrier_counts[7],
                        carrier_counts[11],
                        carrier_counts[9],
                        carrier_counts[3],
                        carrier_counts[12],
                    )
                ),
            }
        )
    if execution_fraction is not None:
        return output, stats, execution_fraction
    return output, stats


def compiled_carrier_additive_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    parent_indices: torch.Tensor,
    parent_log_bias: torch.Tensor,
    child_indices: torch.Tensor,
    child_log_bias: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    dense_block_n: int = 32,
    parent_block_n: int = 32,
    child_block_n: int = 32,
    num_warps: int = 4,
    num_stages: int = 3,
    prevalidated_remote_layout: bool = False,
    value_bf16: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute CarrierSwap with a compile-time tile per service segment."""
    sources = (
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("compiled carrier sources must be five-dimensional")
    if any(
        not source.is_cuda or not source.is_contiguous()
        for source in sources
    ):
        raise ValueError(
            "compiled carrier sources must be contiguous CUDA tensors"
        )
    if any(source.device != query.device for source in sources):
        raise ValueError("compiled carrier sources must use one CUDA device")

    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("compiled carrier dense Q/K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("compiled carrier frame prefixes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("compiled carrier parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("compiled carrier child K/V shapes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("compiled carrier head dimensions must match")
    if any(
        source.dtype != query.dtype
        for source in (dense_key, parent_key, child_key)
    ):
        raise ValueError("compiled carrier Q/K sources must use one dtype")
    if any(
        source.dtype != dense_value.dtype
        for source in (parent_value, child_value)
    ):
        raise ValueError("compiled carrier V sources must use one dtype")

    descriptor_pairs = (
        (parent_indices, parent_log_bias),
        (child_indices, child_log_bias),
    )
    for indices, log_bias in descriptor_pairs:
        if (
            indices.ndim != 3
            or indices.shape[:2] != (batch, num_frames)
            or indices.device != query.device
            or indices.dtype not in {
                torch.int32,
                torch.int64,
                torch.uint16,
            }
            or not indices.is_contiguous()
        ):
            raise ValueError(
                "compiled carrier indices must have shape [B, F, K]"
            )
        if (
            log_bias.shape != indices.shape
            or log_bias.device != query.device
            or not log_bias.dtype.is_floating_point
            or not log_bias.is_contiguous()
        ):
            raise ValueError(
                "compiled carrier log bias must match its indices"
            )

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("compiled carrier special K/V shapes must match")
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
            raise ValueError("compiled carrier special K/V must match query")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("compiled carrier attention supports head_dim <= 128")
    valid_tiles = {32, 64, 128}
    if (
        block_m not in {16, 32, 64, 128}
        or dense_block_n not in valid_tiles
        or parent_block_n not in valid_tiles
        or child_block_n not in valid_tiles
    ):
        raise ValueError("unsupported compiled carrier block shape")
    if num_warps not in {4, 8}:
        raise ValueError("compiled carrier num_warps must be 4 or 8")
    if num_stages not in {1, 2, 3, 4}:
        raise ValueError("compiled carrier num_stages must be in [1, 4]")

    if output_dtype is None:
        output_dtype = query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _compiled_carrier_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        parent_indices,
        parent_log_bias,
        child_indices,
        child_log_bias,
        special_key,
        special_value,
        output,
        1.0 / math.sqrt(head_dim),
        dense_tokens=dense_tokens,
        parent_cells=parent_key.shape[-2],
        parent_tokens=parent_indices.shape[-1],
        child_cells=child_key.shape[-2],
        child_tokens=child_indices.shape[-1],
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        prevalidated_remote_layout=prevalidated_remote_layout,
        value_bf16=value_bf16,
        block_m=block_m,
        dense_block_n=dense_block_n,
        parent_block_n=parent_block_n,
        child_block_n=child_block_n,
        block_d=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output


def parent_mask_additive_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    parent_indices: torch.Tensor,
    parent_log_bias: torch.Tensor,
    service_parent_indices: torch.Tensor,
    service_child_masks: torch.Tensor,
    parent_to_children: torch.Tensor,
    child_log_masses: torch.Tensor,
    tail_child_indices: torch.Tensor,
    tail_child_log_bias: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dot_qk_bf16: bool = False,
    prevalidated_remote_layout: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute additive service from active parents and four-bit child masks."""
    sources = (
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("parent-mask sources must be five-dimensional")
    if any(
        not source.is_cuda or not source.is_contiguous()
        for source in sources
    ):
        raise ValueError("parent-mask sources must be contiguous CUDA tensors")
    if any(source.device != query.device for source in sources):
        raise ValueError("parent-mask sources must use one CUDA device")

    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("parent-mask dense Q/K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("parent-mask frame prefixes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("parent-mask parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("parent-mask child K/V shapes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("parent-mask head dimensions must match")
    if any(
        source.dtype != query.dtype
        for source in (dense_key, parent_key, child_key)
    ):
        raise ValueError("parent-mask Q/K sources must use one dtype")
    if any(
        source.dtype != dense_value.dtype
        for source in (parent_value, child_value)
    ):
        raise ValueError("parent-mask V sources must use one dtype")

    if (
        parent_indices.ndim != 3
        or parent_indices.shape[:2] != (batch, num_frames)
        or parent_indices.device != query.device
        or parent_indices.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("parent-mask parent indices must have shape [B, F, K]")
    if (
        parent_log_bias.shape != parent_indices.shape
        or parent_log_bias.device != query.device
        or not parent_log_bias.dtype.is_floating_point
    ):
        raise ValueError("parent-mask parent bias must match parent indices")
    if (
        service_parent_indices.ndim != 3
        or service_parent_indices.shape[:2] != (batch, num_frames)
        or service_parent_indices.device != query.device
        or service_parent_indices.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("service parent indices must have shape [B, F, K]")
    if (
        service_child_masks.shape != service_parent_indices.shape
        or service_child_masks.device != query.device
        or service_child_masks.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("four-bit service masks must match service parents")
    parent_cells = parent_key.shape[-2]
    if (
        parent_to_children.shape != (parent_cells, 4)
        or parent_to_children.device != query.device
        or parent_to_children.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("parent child map must have shape [P, 4]")
    child_cells = child_key.shape[-2]
    if (
        child_log_masses.shape != (child_cells,)
        or child_log_masses.device != query.device
        or not child_log_masses.dtype.is_floating_point
    ):
        raise ValueError("child log masses must have one value per child cell")
    if (
        tail_child_indices.ndim != 3
        or tail_child_indices.shape[:2] != (batch, num_frames)
        or tail_child_indices.device != query.device
        or tail_child_indices.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("parent-mask tail indices must have shape [B, F, K]")
    if (
        tail_child_log_bias.shape != tail_child_indices.shape
        or tail_child_log_bias.device != query.device
        or not tail_child_log_bias.dtype.is_floating_point
    ):
        raise ValueError("parent-mask tail bias must match tail indices")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("parent-mask special K/V shapes must match")
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
            raise ValueError("parent-mask special K/V must match query")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("parent-mask attention supports head_dim <= 128")
    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported parent-mask block shape")
    launch_warps = 4 if block_m <= 32 else 8
    if num_warps is not None:
        if num_warps not in {4, 8}:
            raise ValueError("parent-mask num_warps must be 4 or 8")
        launch_warps = num_warps
    launch_stages = 2 if query.dtype == torch.float32 else 3
    if num_stages is not None:
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError("parent-mask num_stages must be in [1, 4]")
        launch_stages = num_stages

    if output_dtype is None:
        output_dtype = torch.bfloat16 if dot_qk_bf16 else query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _mixed_descriptor_direct_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        child_key,
        child_value,
        parent_indices.contiguous(),
        service_parent_indices,
        parent_log_bias.contiguous(),
        child_log_masses,
        tail_child_indices.contiguous(),
        tail_child_log_bias.contiguous(),
        service_parent_indices.contiguous(),
        service_child_masks.contiguous(),
        parent_to_children.contiguous(),
        parent_to_children,
        child_log_masses.contiguous(),
        special_key,
        special_value,
        output,
        output,
        output,
        output,
        1.0 / math.sqrt(head_dim),
        -float("inf"),
        dense_tokens=dense_tokens,
        parent_cells=parent_cells,
        easy_parent_tokens=parent_indices.shape[-1],
        child_cells=child_cells,
        hard_child_tokens=service_parent_indices.shape[-1] * 3,
        residual_tokens=tail_child_indices.shape[-1],
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        dot_qk_bf16=dot_qk_bf16,
        indexed_service_bias=True,
        parent_mask_service=True,
        implicit_parent_layout=False,
        prevalidated_remote_layout=prevalidated_remote_layout,
        two_stage_qk_pv=False,
        collect_two_stage_stats=False,
        return_pv_execution_map=False,
        observe_carrier_compensation=False,
        carrier_compensated_pv=False,
        query_blocks=triton.cdiv(dense_tokens, block_m),
        child_blocks=triton.cdiv(
            service_parent_indices.shape[-1] * 3, block_n
        ),
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    return output


def implicit_parent_additive_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    parent_log_bias_map: torch.Tensor,
    child_indices: torch.Tensor,
    child_log_bias: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
    dot_qk_bf16: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Execute additive service with an implicit contiguous parent lattice."""
    sources = (
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
    )
    if any(source.ndim != 5 for source in sources):
        raise ValueError("implicit parent sources must be five-dimensional")
    if any(
        not source.is_cuda or not source.is_contiguous()
        for source in sources
    ):
        raise ValueError(
            "implicit parent sources must be contiguous CUDA tensors"
        )
    if any(source.device != query.device for source in sources):
        raise ValueError("implicit parent sources must use one CUDA device")

    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("implicit parent dense Q/K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("implicit parent frame prefixes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("implicit parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("implicit child K/V shapes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("implicit parent head dimensions must match")
    if any(
        source.dtype != query.dtype
        for source in (dense_key, parent_key, child_key)
    ):
        raise ValueError("implicit parent Q/K sources must use one dtype")
    if any(
        source.dtype != dense_value.dtype
        for source in (parent_value, child_value)
    ):
        raise ValueError("implicit parent V sources must use one dtype")

    parent_total = num_frames * parent_key.shape[-2]
    expected_parent_bias_shape = (batch, num_frames, parent_total)
    if (
        parent_log_bias_map.shape != expected_parent_bias_shape
        or parent_log_bias_map.device != query.device
        or not parent_log_bias_map.dtype.is_floating_point
        or not parent_log_bias_map.is_contiguous()
    ):
        raise ValueError(
            "implicit parent bias must have shape [B, F, F*P]"
        )
    if (
        child_indices.ndim != 3
        or child_indices.shape[:2] != (batch, num_frames)
        or child_indices.device != query.device
        or child_indices.dtype not in {torch.int32, torch.int64}
        or not child_indices.is_contiguous()
    ):
        raise ValueError(
            "implicit child indices must have shape [B, F, K]"
        )
    if (
        child_log_bias.shape != child_indices.shape
        or child_log_bias.device != query.device
        or not child_log_bias.dtype.is_floating_point
        or not child_log_bias.is_contiguous()
    ):
        raise ValueError("implicit child bias must match child indices")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("implicit special K/V shapes must match")
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
            raise ValueError("implicit special K/V must match query")
        special_tokens = special_key.shape[-2]

    if head_dim > 128:
        raise ValueError("implicit parent attention supports head_dim <= 128")
    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported implicit parent block shape")
    launch_warps = 4 if block_m <= 32 else 8
    if num_warps is not None:
        if num_warps not in {4, 8}:
            raise ValueError("implicit parent num_warps must be 4 or 8")
        launch_warps = num_warps
    launch_stages = 2 if query.dtype == torch.float32 else 3
    if num_stages is not None:
        if num_stages not in {1, 2, 3, 4}:
            raise ValueError("implicit parent num_stages must be in [1, 4]")
        launch_stages = num_stages

    if output_dtype is None:
        output_dtype = torch.bfloat16 if dot_qk_bf16 else query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _mixed_descriptor_direct_attention_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        child_key,
        child_value,
        child_indices,
        child_indices,
        parent_log_bias_map,
        child_log_bias,
        child_indices,
        child_indices,
        child_indices,
        child_indices,
        child_indices,
        child_indices,
        child_log_bias,
        special_key,
        special_value,
        output,
        output,
        output,
        output,
        1.0 / math.sqrt(head_dim),
        -float("inf"),
        dense_tokens=dense_tokens,
        parent_cells=parent_key.shape[-2],
        easy_parent_tokens=parent_total,
        child_cells=child_key.shape[-2],
        hard_child_tokens=child_indices.shape[-1],
        residual_tokens=0,
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        dot_qk_bf16=dot_qk_bf16,
        indexed_service_bias=True,
        parent_mask_service=False,
        implicit_parent_layout=True,
        prevalidated_remote_layout=False,
        two_stage_qk_pv=False,
        collect_two_stage_stats=False,
        return_pv_execution_map=False,
        observe_carrier_compensation=False,
        carrier_compensated_pv=False,
        query_blocks=triton.cdiv(dense_tokens, block_m),
        child_blocks=triton.cdiv(child_indices.shape[-1], block_n),
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    return output
