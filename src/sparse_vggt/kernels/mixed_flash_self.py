"""FlashAttention self-frame base with direct mixed remote correction."""

import math

import torch
import triton
import triton.language as tl

from sparse_vggt.kernels.direct_attention import _online_attention_update
from sparse_vggt.kernels.hps_flash_carrier import _subtract_attention_update


@triton.jit
def _mixed_flash_self_correction_kernel(
    query_ptr,
    parent_key_ptr,
    parent_value_ptr,
    child_key_ptr,
    child_value_ptr,
    residual_key_ptr,
    residual_value_ptr,
    easy_parent_index_ptr,
    hard_parent_index_ptr,
    residual_index_ptr,
    parent_to_children_ptr,
    special_key_ptr,
    special_value_ptr,
    self_output_ptr,
    self_lse_ptr,
    output_ptr,
    softmax_scale,
    dense_tokens: tl.constexpr,
    parent_cells: tl.constexpr,
    easy_parent_tokens: tl.constexpr,
    hard_parent_tokens: tl.constexpr,
    child_cells: tl.constexpr,
    children_per_parent: tl.constexpr,
    residual_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
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

    output_offsets = (
        query_base
        + query_offsets[:, None] * head_dim
        + dim_offsets[None, :]
    )
    lse_base = (
        ((batch * heads + head) * num_frames + frame) * dense_tokens
    )
    accumulator = tl.load(
        self_output_ptr + output_offsets,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    denominator = tl.where(query_mask, 1.0, 0.0)
    row_max = tl.load(
        self_lse_ptr + lse_base + query_offsets,
        mask=query_mask,
        other=0.0,
    ) * 1.4426950408889634
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    zero_bias = tl.zeros((block_n,), dtype=tl.float32)

    parent_total = num_frames * parent_cells
    parent_source_base = (
        (batch * heads + head) * parent_total * head_dim
    )
    easy_parent_base = (
        (batch * num_frames + frame) * easy_parent_tokens
    )
    parent_bias = tl.full(
        (block_n,), 1.3862943611198906, dtype=tl.float32
    )
    for start in tl.range(0, easy_parent_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < easy_parent_tokens
        source_parents = tl.load(
            easy_parent_index_ptr
            + easy_parent_base
            + selected_offsets,
            mask=key_mask,
            other=0,
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

    hard_parent_base = (
        (batch * num_frames + frame) * hard_parent_tokens
    )
    child_total = num_frames * child_cells
    child_source_base = (
        (batch * heads + head) * child_total * head_dim
    )
    for start in tl.range(0, hard_parent_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < hard_parent_tokens
        source_parents = tl.load(
            hard_parent_index_ptr
            + hard_parent_base
            + selected_offsets,
            mask=key_mask,
            other=0,
        )
        source_frames = source_parents // parent_cells
        local_parents = source_parents - source_frames * parent_cells
        key_mask = key_mask & (source_frames != frame)

        for child_slot in tl.static_range(0, children_per_parent):
            local_children = tl.load(
                parent_to_children_ptr
                + local_parents * children_per_parent
                + child_slot,
                mask=key_mask,
                other=0,
            )
            source_children = source_frames * child_cells + local_children
            child_offsets = (
                child_source_base
                + source_children[:, None] * head_dim
                + dim_offsets[None, :]
            )
            child_key = tl.load(
                child_key_ptr + child_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            child_value = tl.load(
                child_value_ptr + child_offsets,
                mask=key_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            accumulator, denominator, row_max = _online_attention_update(
                query,
                child_key,
                child_value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                zero_bias,
                softmax_scale_log2,
            )

    residual_index_base = (
        (batch * num_frames + frame) * residual_tokens
    )
    for start in tl.range(0, residual_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < residual_tokens
        source_children = tl.load(
            residual_index_ptr
            + residual_index_base
            + selected_offsets,
            mask=key_mask,
            other=0,
        )
        source_frames = source_children // child_cells
        key_mask = key_mask & (source_frames != frame)
        source_offsets = (
            child_source_base
            + source_children[:, None] * head_dim
            + dim_offsets[None, :]
        )
        child_key = tl.load(
            child_key_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        child_value = tl.load(
            child_value_ptr + source_offsets,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator, denominator = _subtract_attention_update(
            query,
            child_key,
            child_value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            softmax_scale_log2,
            weight=0.5,
        )

    residual_source_base = (
        (batch * heads + head) * child_total * head_dim
    )
    residual_bias = tl.full(
        (block_n,), -0.6931471805599453, dtype=tl.float32
    )
    for start in tl.range(0, residual_tokens, block_n):
        selected_offsets = start + tl.arange(0, block_n)
        key_mask = selected_offsets < residual_tokens
        source_children = tl.load(
            residual_index_ptr
            + residual_index_base
            + selected_offsets,
            mask=key_mask,
            other=0,
        )
        source_frames = source_children // child_cells
        key_mask = key_mask & (source_frames != frame)
        source_offsets = (
            residual_source_base
            + source_children[:, None] * head_dim
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


def mixed_flash_self_attention(
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
    hard_parent_indices: torch.Tensor,
    residual_indices: torch.Tensor,
    parent_to_children: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run dense self frames with FlashAttention, then add mixed remote K/V."""
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
        raise ValueError("mixed self-flash sources must be five-dimensional")
    if any(not source.is_cuda or not source.is_contiguous() for source in sources):
        raise ValueError(
            "mixed self-flash sources must be contiguous CUDA tensors"
        )
    if any(source.device != query.device for source in sources):
        raise ValueError("mixed self-flash sources must use one CUDA device")
    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("mixed self-flash dense Q/K/V shapes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("mixed self-flash parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("mixed self-flash child K/V shapes must match")
    if residual_key.shape != residual_value.shape:
        raise ValueError("mixed self-flash residual K/V shapes must match")
    if residual_key.shape != child_key.shape:
        raise ValueError(
            "mixed self-flash residual cells must match child cells"
        )
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("mixed self-flash frame prefixes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("mixed self-flash head dimensions must match")
    if any(source.dtype != query.dtype for source in (
        dense_key, parent_key, child_key, residual_key
    )):
        raise ValueError("mixed self-flash Q/K sources must use one dtype")
    if any(source.dtype != dense_value.dtype for source in (
        parent_value, child_value, residual_value
    )):
        raise ValueError("mixed self-flash V sources must use one dtype")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("mixed self-flash requires FP16 or BF16 Q/K")
    if head_dim > 128:
        raise ValueError("mixed self-flash supports head_dim <= 128")

    parent_cells = parent_key.shape[-2]
    child_cells = child_key.shape[-2]
    index_tensors = (
        easy_parent_indices,
        hard_parent_indices,
        residual_indices,
    )
    if any(
        tensor.ndim != 3
        or tensor.shape[:2] != (batch, num_frames)
        or tensor.device != query.device
        or tensor.dtype not in {torch.int32, torch.int64}
        for tensor in index_tensors
    ):
        raise ValueError(
            "mixed self-flash indices must have shape [B, F, K]"
        )
    if (
        parent_to_children.ndim != 2
        or parent_to_children.shape[0] != parent_cells
        or parent_to_children.device != query.device
        or parent_to_children.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError(
            "mixed self-flash parent-to-child map must have shape [P, C]"
        )
    children_per_parent = parent_to_children.shape[-1]
    if children_per_parent < 1:
        raise ValueError("mixed self-flash requires children per parent")

    if (special_key is None) != (special_value is None):
        raise ValueError(
            "mixed self-flash special K/V must be provided together"
        )
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError(
                "mixed self-flash special K/V shapes must match"
            )
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
            raise ValueError(
                "mixed self-flash special K/V must match query"
            )
        special_tokens = special_key.shape[-2]

    flat_query = query.permute(0, 2, 1, 3, 4).reshape(
        batch * num_frames,
        heads,
        dense_tokens,
        head_dim,
    ).contiguous()
    flat_dense_key = dense_key.permute(0, 2, 1, 3, 4).reshape_as(
        flat_query
    ).contiguous()
    flat_dense_value = dense_value.permute(0, 2, 1, 3, 4).reshape_as(
        flat_query
    ).contiguous()
    flash_result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        flat_query,
        flat_dense_key,
        flat_dense_value,
        0.0,
        False,
        False,
        scale=1.0 / math.sqrt(head_dim),
    )
    self_output = flash_result[0].reshape(
        batch,
        num_frames,
        heads,
        dense_tokens,
        head_dim,
    ).permute(0, 2, 1, 3, 4).contiguous()
    self_lse = flash_result[1][..., :dense_tokens].reshape(
        batch,
        num_frames,
        heads,
        dense_tokens,
    ).permute(0, 2, 1, 3).contiguous()

    if block_m not in {16, 32, 64, 128} or block_n not in {32, 64, 128}:
        raise ValueError("unsupported mixed self-flash block shape")
    launch_warps = 4 if num_warps is None else num_warps
    launch_stages = 3 if num_stages is None else num_stages
    if launch_warps not in {4, 8}:
        raise ValueError("mixed self-flash num_warps must be 4 or 8")
    if launch_stages not in {1, 2, 3, 4}:
        raise ValueError("mixed self-flash num_stages must be in [1, 4]")

    if output_dtype is None:
        output_dtype = query.dtype
    output = torch.empty(query.shape, device=query.device, dtype=output_dtype)
    block_d = triton.next_power_of_2(head_dim)
    grid = (
        triton.cdiv(dense_tokens, block_m),
        batch * heads * num_frames,
    )
    _mixed_flash_self_correction_kernel[grid](
        query,
        parent_key,
        parent_value,
        child_key,
        child_value,
        residual_key,
        residual_value,
        easy_parent_indices.contiguous(),
        hard_parent_indices.contiguous(),
        residual_indices.contiguous(),
        parent_to_children.contiguous(),
        special_key,
        special_value,
        self_output,
        self_lse,
        output,
        1.0 / math.sqrt(head_dim),
        dense_tokens=dense_tokens,
        parent_cells=parent_cells,
        easy_parent_tokens=easy_parent_indices.shape[-1],
        hard_parent_tokens=hard_parent_indices.shape[-1],
        child_cells=child_cells,
        children_per_parent=children_per_parent,
        residual_tokens=residual_indices.shape[-1],
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        num_warps=launch_warps,
        num_stages=launch_stages,
    )
    return output
