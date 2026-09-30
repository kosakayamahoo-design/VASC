"""FlashAttention parent-carrier base with additive-service corrections."""

import math

import torch
import triton
import triton.language as tl

from sparse_vggt.kernels.direct_attention import _online_attention_update


@triton.jit
def _subtract_weighted_attention_update(
    query,
    key,
    value,
    accumulator,
    denominator,
    row_max,
    query_mask,
    key_mask,
    weight,
    softmax_scale_log2,
):
    logits = tl.dot(query, tl.trans(key)) * softmax_scale_log2
    valid = query_mask[:, None] & key_mask[None, :]
    shifted = tl.where(valid, logits - row_max[:, None], -float("inf"))
    probabilities = tl.math.exp2(shifted) * weight[None, :]
    denominator -= tl.sum(probabilities, axis=1)
    accumulator -= tl.dot(probabilities.to(value.dtype), value)
    return accumulator, denominator


@triton.jit
def _additive_flash_carrier_correction_kernel(
    query_ptr,
    dense_key_ptr,
    dense_value_ptr,
    parent_key_ptr,
    parent_value_ptr,
    child_key_ptr,
    child_value_ptr,
    parent_correction_mass_ptr,
    child_index_ptr,
    child_log_bias_ptr,
    special_key_ptr,
    special_value_ptr,
    parent_output_ptr,
    parent_lse_ptr,
    output_ptr,
    softmax_scale,
    base_parent_mass: tl.constexpr,
    dense_tokens: tl.constexpr,
    parent_cells: tl.constexpr,
    child_cells: tl.constexpr,
    child_tokens: tl.constexpr,
    special_tokens: tl.constexpr,
    num_frames: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_d: tl.constexpr,
    positive_flash_groups: tl.constexpr,
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
        parent_output_ptr + output_offsets,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if positive_flash_groups:
        denominator = tl.where(query_mask, 1.0, 0.0)
    else:
        accumulator *= base_parent_mass
        denominator = tl.where(
            query_mask,
            float(base_parent_mass),
            0.0,
        )
    row_max = tl.load(
        parent_lse_ptr + lse_base + query_offsets,
        mask=query_mask,
        other=0.0,
    ) * 1.4426950408889634
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    zero_bias = tl.zeros((block_n,), dtype=tl.float32)

    parent_total = num_frames * parent_cells
    parent_source_base = (
        (batch * heads + head) * parent_total * head_dim
    )
    correction_base = (
        (batch * num_frames + frame) * parent_total
    )
    for start in tl.range(0, parent_total, block_n):
        parent_offsets = start + tl.arange(0, block_n)
        correction_mass = tl.load(
            parent_correction_mass_ptr
            + correction_base
            + parent_offsets,
            mask=parent_offsets < parent_total,
            other=0.0,
        )
        key_mask = (
            (parent_offsets < parent_total)
            & (correction_mass > 0.0)
        )
        source_offsets = (
            parent_source_base
            + parent_offsets[:, None] * head_dim
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
        accumulator, denominator = _subtract_weighted_attention_update(
            query,
            key,
            value,
            accumulator,
            denominator,
            row_max,
            query_mask,
            key_mask,
            correction_mass,
            softmax_scale_log2,
        )

    child_total = num_frames * child_cells
    child_source_base = (
        (batch * heads + head) * child_total * head_dim
    )
    child_descriptor_base = (
        (batch * num_frames + frame) * child_tokens
    )
    for start in tl.range(0, child_tokens, block_n):
        child_offsets = start + tl.arange(0, block_n)
        key_mask = child_offsets < child_tokens
        source_children = tl.load(
            child_index_ptr
            + child_descriptor_base
            + child_offsets,
            mask=key_mask,
            other=0,
        )
        child_descriptor = tl.load(
            child_log_bias_ptr
            + child_descriptor_base
            + child_offsets,
            mask=key_mask,
            other=0.0,
        )
        if positive_flash_groups:
            key_mask = key_mask & (child_descriptor > 0.0)
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
        if positive_flash_groups:
            accumulator, denominator = _subtract_weighted_attention_update(
                query,
                key,
                value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                child_descriptor,
                softmax_scale_log2,
            )
        else:
            accumulator, denominator, row_max = _online_attention_update(
                query,
                key,
                value,
                accumulator,
                denominator,
                row_max,
                query_mask,
                key_mask,
                child_descriptor,
                softmax_scale_log2,
            )

    if not positive_flash_groups:
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


def additive_flash_carrier_attention(
    query: torch.Tensor,
    dense_key: torch.Tensor,
    dense_value: torch.Tensor,
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    parent_correction_mass: torch.Tensor,
    child_indices: torch.Tensor,
    child_log_bias: torch.Tensor,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
    *,
    base_parent_mass: int,
    base_child_mass: int = 1,
    positive_flash_groups: bool = False,
    block_m: int = 128,
    block_n: int = 32,
    num_warps: int = 8,
    num_stages: int = 2,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run regular parent carriers with Flash and apply exact mass deltas."""
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
        raise ValueError("additive Flash sources must be five-dimensional")
    if any(
        not source.is_cuda or not source.is_contiguous()
        for source in sources
    ):
        raise ValueError(
            "additive Flash sources must be contiguous CUDA tensors"
        )
    if any(source.device != query.device for source in sources):
        raise ValueError("additive Flash sources must use one CUDA device")

    batch, heads, num_frames, dense_tokens, head_dim = query.shape
    if dense_key.shape != query.shape or dense_value.shape != query.shape:
        raise ValueError("additive Flash dense Q/K/V shapes must match")
    expected_prefix = (batch, heads, num_frames)
    if any(source.shape[:3] != expected_prefix for source in sources[3:]):
        raise ValueError("additive Flash frame prefixes must match")
    if parent_key.shape != parent_value.shape:
        raise ValueError("additive Flash parent K/V shapes must match")
    if child_key.shape != child_value.shape:
        raise ValueError("additive Flash child K/V shapes must match")
    if any(source.shape[-1] != head_dim for source in sources):
        raise ValueError("additive Flash head dimensions must match")
    if any(
        source.dtype != query.dtype
        for source in (dense_key, parent_key, child_key)
    ):
        raise ValueError("additive Flash Q/K sources must use one dtype")
    if any(
        source.dtype != dense_value.dtype
        for source in (parent_value, child_value)
    ):
        raise ValueError("additive Flash V sources must use one dtype")
    if query.dtype not in {torch.float16, torch.bfloat16}:
        raise ValueError("additive Flash requires FP16 or BF16 Q/K")
    if base_parent_mass < 1:
        raise ValueError("base parent mass must be positive")
    if base_child_mass < 1:
        raise ValueError("base child mass must be positive")

    parent_cells = parent_key.shape[-2]
    parent_total = num_frames * parent_cells
    if (
        parent_correction_mass.shape
        != (batch, num_frames, parent_total)
        or parent_correction_mass.device != query.device
        or not parent_correction_mass.dtype.is_floating_point
    ):
        raise ValueError(
            "parent correction mass must have shape [B, F, F*P]"
        )
    if (
        child_indices.ndim != 3
        or child_indices.shape[:2] != (batch, num_frames)
        or child_indices.device != query.device
        or child_indices.dtype not in {torch.int32, torch.int64}
    ):
        raise ValueError("child indices must have shape [B, F, K]")
    if (
        child_log_bias.shape != child_indices.shape
        or child_log_bias.device != query.device
        or not child_log_bias.dtype.is_floating_point
    ):
        raise ValueError("child log bias must match child indices")

    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    if special_key is None:
        special_tokens = 0
        special_key = dense_key
        special_value = dense_value
    else:
        if special_key.shape != special_value.shape:
            raise ValueError("special K/V shapes must match")
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
            raise ValueError("special K/V must match additive Flash sources")
        special_tokens = special_key.shape[-2]

    flat_query = query.reshape(
        batch,
        heads,
        num_frames * dense_tokens,
        head_dim,
    )
    flat_parent_key = parent_key.reshape(
        batch,
        heads,
        num_frames * parent_cells,
        head_dim,
    )
    flat_parent_value = parent_value.reshape_as(flat_parent_key)
    flash_result = torch.ops.aten._scaled_dot_product_flash_attention.default(
        flat_query,
        flat_parent_key,
        flat_parent_value,
        0.0,
        False,
        False,
        scale=1.0 / math.sqrt(head_dim),
    )
    parent_output = flash_result[0].reshape_as(query).contiguous()
    parent_lse = flash_result[1][
        ..., : num_frames * dense_tokens
    ].reshape(
        batch,
        heads,
        num_frames,
        dense_tokens,
    ).contiguous()

    correction_child_descriptor = child_log_bias
    if positive_flash_groups:
        frame_query = query.permute(0, 2, 1, 3, 4).reshape(
            batch * num_frames,
            heads,
            dense_tokens,
            head_dim,
        ).contiguous()
        frame_dense_key = dense_key.permute(0, 2, 1, 3, 4).reshape_as(
            frame_query
        ).contiguous()
        frame_dense_value = dense_value.permute(
            0, 2, 1, 3, 4
        ).reshape_as(frame_query).contiguous()
        dense_flash = torch.ops.aten._scaled_dot_product_flash_attention.default(
            frame_query,
            frame_dense_key,
            frame_dense_value,
            0.0,
            False,
            False,
            scale=1.0 / math.sqrt(head_dim),
        )
        dense_output = dense_flash[0].reshape(
            batch,
            num_frames,
            heads,
            dense_tokens,
            head_dim,
        ).permute(0, 2, 1, 3, 4).contiguous()
        dense_lse = dense_flash[1][..., :dense_tokens].reshape(
            batch,
            num_frames,
            heads,
            dense_tokens,
        ).permute(0, 2, 1, 3).contiguous()

        child_total = num_frames * child_key.shape[-2]
        flat_child_key = child_key.reshape(
            batch, heads, child_total, head_dim
        )
        flat_child_value = child_value.reshape_as(flat_child_key)
        child_outputs = []
        child_lses = []
        for frame in range(num_frames):
            frame_indices = child_indices[:, frame].to(torch.long)
            gather_index = frame_indices[:, None, :, None].expand(
                batch,
                heads,
                frame_indices.shape[-1],
                head_dim,
            )
            frame_child_key = torch.gather(
                flat_child_key, 2, gather_index
            ).contiguous()
            frame_child_value = torch.gather(
                flat_child_value, 2, gather_index
            ).contiguous()
            child_flash = (
                torch.ops.aten._scaled_dot_product_flash_attention.default(
                    query[:, :, frame],
                    frame_child_key,
                    frame_child_value,
                    0.0,
                    False,
                    False,
                    scale=1.0 / math.sqrt(head_dim),
                )
            )
            child_outputs.append(child_flash[0])
            child_lses.append(child_flash[1][..., :dense_tokens])
        child_output = torch.stack(child_outputs, dim=2).contiguous()
        child_lse = torch.stack(child_lses, dim=2).contiguous()

        component_outputs = [parent_output, dense_output, child_output]
        component_lses = [
            parent_lse.to(torch.float32) + math.log(base_parent_mass),
            dense_lse.to(torch.float32),
            child_lse.to(torch.float32) + math.log(base_child_mass),
        ]
        if special_tokens:
            special_flash = (
                torch.ops.aten._scaled_dot_product_flash_attention.default(
                    flat_query,
                    special_key,
                    special_value,
                    0.0,
                    False,
                    False,
                    scale=1.0 / math.sqrt(head_dim),
                )
            )
            component_outputs.append(
                special_flash[0].reshape_as(query).contiguous()
            )
            component_lses.append(
                special_flash[1][
                    ..., : num_frames * dense_tokens
                ].reshape(
                    batch,
                    heads,
                    num_frames,
                    dense_tokens,
                ).to(torch.float32)
            )
        stacked_lse = torch.stack(component_lses, dim=0)
        combined_lse = torch.logsumexp(stacked_lse, dim=0)
        component_weights = torch.exp(
            stacked_lse - combined_lse.unsqueeze(0)
        )
        parent_output = sum(
            output.to(torch.float32) * weight.unsqueeze(-1)
            for output, weight in zip(
                component_outputs,
                component_weights,
                strict=True,
            )
        ).contiguous()
        parent_lse = combined_lse.contiguous()
        correction_child_descriptor = (
            float(base_child_mass)
            - torch.exp(child_log_bias.to(torch.float32))
        ).clamp_min_(0.0)

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
    _additive_flash_carrier_correction_kernel[grid](
        query,
        dense_key,
        dense_value,
        parent_key,
        parent_value,
        child_key,
        child_value,
        parent_correction_mass.contiguous(),
        child_indices.contiguous(),
        correction_child_descriptor.contiguous(),
        special_key,
        special_value,
        parent_output,
        parent_lse,
        output,
        1.0 / math.sqrt(head_dim),
        base_parent_mass=base_parent_mass,
        dense_tokens=dense_tokens,
        parent_cells=parent_cells,
        child_cells=child_key.shape[-2],
        child_tokens=child_indices.shape[-1],
        special_tokens=special_tokens,
        num_frames=num_frames,
        heads=heads,
        head_dim=head_dim,
        block_m=block_m,
        block_n=block_n,
        block_d=block_d,
        positive_flash_groups=positive_flash_groups,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
