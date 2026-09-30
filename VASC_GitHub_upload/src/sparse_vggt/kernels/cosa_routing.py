"""Fused Triton primitives for CoSA pair routing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pair_innovation_kernel(
    value_ptr,
    innovation_ptr,
    key_blocks: tl.constexpr,
    pair_count: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    pair = tl.program_id(0)
    outer = tl.program_id(1)
    left_key = pair * 2
    right_key = tl.minimum(left_key + 1, key_blocks - 1)
    dim = tl.arange(0, block_dim)
    dim_mask = dim < head_dim
    base = outer * key_blocks * head_dim
    left = tl.load(value_ptr + base + left_key * head_dim + dim, mask=dim_mask, other=0.0)
    right = tl.load(value_ptr + base + right_key * head_dim + dim, mask=dim_mask, other=0.0)
    delta = left.to(tl.float32) - right.to(tl.float32)
    magnitude = tl.sqrt(tl.sum(delta * delta, axis=0) / head_dim) * 0.5
    output_base = outer * key_blocks
    tl.store(innovation_ptr + output_base + left_key, magnitude)
    tl.store(
        innovation_ptr + output_base + right_key,
        magnitude,
        mask=right_key != left_key,
    )


@triton.jit
def _pooled_pair_innovation_kernel(
    value_ptr,
    innovation_ptr,
    token_count: tl.constexpr,
    key_blocks: tl.constexpr,
    block_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    block_dim: tl.constexpr,
):
    pair = tl.program_id(0)
    outer = tl.program_id(1)
    token = tl.arange(0, block_tokens)[:, None]
    dim = tl.arange(0, block_dim)[None, :]
    dim_mask = dim < head_dim
    left_start = pair * 2 * block_tokens
    right_start = left_start + block_tokens
    left_count = tl.minimum(block_tokens, token_count - left_start)
    right_count = tl.maximum(0, tl.minimum(block_tokens, token_count - right_start))
    left_token = left_start + token
    right_token = right_start + token
    base = outer * token_count * head_dim
    left = tl.load(
        value_ptr + base + left_token * head_dim + dim,
        mask=(token < left_count) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    left_mean = (tl.sum(left, axis=0) / left_count).to(
        value_ptr.dtype.element_ty
    ).to(tl.float32)
    right = tl.load(
        value_ptr + base + right_token * head_dim + dim,
        mask=(token < right_count) & dim_mask,
        other=0.0,
    ).to(tl.float32)
    right_mean = tl.where(
        right_count > 0,
        tl.sum(right, axis=0) / tl.maximum(right_count, 1),
        left_mean,
    )
    right_mean = right_mean.to(value_ptr.dtype.element_ty).to(tl.float32)
    delta = left_mean - right_mean
    magnitude = tl.sqrt(tl.sum(delta * delta, axis=0) / head_dim) * 0.5
    left_key = pair * 2
    right_key = tl.minimum(left_key + 1, key_blocks - 1)
    output_base = outer * key_blocks
    tl.store(innovation_ptr + output_base + left_key, magnitude)
    tl.store(
        innovation_ptr + output_base + right_key,
        magnitude,
        mask=right_key != left_key,
    )


@triton.jit
def _conservative_arrival_kernel(
    score_ptr,
    innovation_ptr,
    capacity_ptr,
    output_ptr,
    query_blocks: tl.constexpr,
    key_blocks: tl.constexpr,
    block_keys: tl.constexpr,
):
    row = tl.program_id(0)
    key = tl.arange(0, block_keys)
    key_mask = key < key_blocks
    score = tl.load(score_ptr + row * key_blocks + key, mask=key_mask, other=0.0)
    score = score.to(tl.float32)
    score = tl.where(score == score, tl.maximum(score, 0.0), 0.0)
    outer = row // query_blocks
    innovation = tl.load(
        innovation_ptr + outer * key_blocks + key,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    risk = score * innovation
    total = tl.sum(risk, axis=0)
    capacity = tl.load(capacity_ptr + row).to(tl.float32)
    scale = tl.where(total > 0.0, capacity / tl.maximum(total, 1.1754944e-38), 0.0)
    projected = tl.minimum(tl.maximum(risk * scale, 0.0), 1.0)
    tl.store(output_ptr + row * key_blocks + key, projected, mask=key_mask)


def triton_pair_innovation(pooled_value: torch.Tensor) -> torch.Tensor:
    """Return the FP32 two-key innovation used by the CoSA pair ledger."""
    if not pooled_value.is_cuda or pooled_value.ndim != 4:
        raise ValueError("pooled_value must be a CUDA [B, H, K, D] tensor")
    value = pooled_value.contiguous()
    batch, heads, key_blocks, head_dim = value.shape
    output = torch.empty(
        batch,
        heads,
        key_blocks,
        device=value.device,
        dtype=torch.float32,
    )
    if output.numel() == 0:
        return output
    pair_count = (key_blocks + 1) // 2
    _pair_innovation_kernel[(pair_count, batch * heads)](
        value,
        output,
        key_blocks=key_blocks,
        pair_count=pair_count,
        head_dim=head_dim,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def triton_pooled_pair_innovation(
    value: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Fuse contiguous value pooling with adjacent-block innovation."""
    if not value.is_cuda or value.ndim != 4:
        raise ValueError("value must be a CUDA [B, H, T, D] tensor")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    value = value.contiguous()
    batch, heads, token_count, head_dim = value.shape
    key_blocks = (token_count + block_size - 1) // block_size
    output = torch.empty(
        batch,
        heads,
        key_blocks,
        device=value.device,
        dtype=torch.float32,
    )
    if output.numel() == 0:
        return output
    pair_count = (key_blocks + 1) // 2
    _pooled_pair_innovation_kernel[(pair_count, batch * heads)](
        value,
        output,
        token_count=token_count,
        key_blocks=key_blocks,
        block_tokens=block_size,
        head_dim=head_dim,
        block_dim=triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output


def triton_conservative_arrival(
    pooled_score: torch.Tensor,
    innovation: torch.Tensor,
    capacity: torch.Tensor,
) -> torch.Tensor:
    """Fuse value-risk construction and conservative capacity projection."""
    if not pooled_score.is_cuda or pooled_score.ndim != 4:
        raise ValueError("pooled_score must be a CUDA [B, H, Q, K] tensor")
    if innovation.shape != pooled_score.shape[:2] + pooled_score.shape[-1:]:
        raise ValueError("innovation must have shape [B, H, K]")
    if capacity.shape != pooled_score.shape[:-1] + (1,):
        raise ValueError("capacity must have shape [B, H, Q, 1]")
    if pooled_score.device != innovation.device or pooled_score.device != capacity.device:
        raise ValueError("routing tensors must share one CUDA device")
    score = pooled_score.contiguous()
    innovation = innovation.contiguous()
    capacity = capacity.contiguous()
    output = torch.empty(score.shape, device=score.device, dtype=torch.float32)
    if output.numel() == 0:
        return output
    _, _, query_blocks, key_blocks = score.shape
    rows = score.numel() // key_blocks
    _conservative_arrival_kernel[(rows,)](
        score,
        innovation,
        capacity,
        output,
        query_blocks=query_blocks,
        key_blocks=key_blocks,
        block_keys=triton.next_power_of_2(key_blocks),
        num_warps=8,
    )
    return output
