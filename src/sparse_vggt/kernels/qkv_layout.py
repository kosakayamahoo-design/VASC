"""Fused frame-major to patch/special QKV layout packing."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_qkv_layout_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    total_elements,
    tokens,
    channels,
    frames,
    tokens_per_frame,
    special_tokens,
    patch_tokens,
    q_stride_b,
    q_stride_h,
    q_stride_t,
    q_stride_c,
    k_stride_b,
    k_stride_h,
    k_stride_t,
    k_stride_c,
    v_stride_b,
    v_stride_h,
    v_stride_t,
    v_stride_c,
    heads: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < total_elements
    channel = offsets % channels
    token = (offsets // channels) % tokens
    head = (offsets // (channels * tokens)) % heads
    batch = offsets // (channels * tokens * heads)

    patch_count = frames * patch_tokens
    is_patch = token < patch_count
    patch_frame = token // patch_tokens
    patch_local = token - patch_frame * patch_tokens
    special_linear = token - patch_count
    special_frame = special_linear // special_tokens
    special_local = special_linear - special_frame * special_tokens
    source_token = tl.where(
        is_patch,
        patch_frame * tokens_per_frame + special_tokens + patch_local,
        special_frame * tokens_per_frame + special_local,
    )

    q_offset = (
        batch * q_stride_b
        + head * q_stride_h
        + source_token * q_stride_t
        + channel * q_stride_c
    )
    k_offset = (
        batch * k_stride_b
        + head * k_stride_h
        + source_token * k_stride_t
        + channel * k_stride_c
    )
    v_offset = (
        batch * v_stride_b
        + head * v_stride_h
        + source_token * v_stride_t
        + channel * v_stride_c
    )
    tl.store(q_out_ptr + offsets, tl.load(q_ptr + q_offset, mask=valid), mask=valid)
    tl.store(k_out_ptr + offsets, tl.load(k_ptr + k_offset, mask=valid), mask=valid)
    tl.store(v_out_ptr + offsets, tl.load(v_ptr + v_offset, mask=valid), mask=valid)


def pack_qkv_patch_then_special(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    special_tokens: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Pack QKV once and return zero-copy patch/special views."""
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("fused QKV layout packing requires CUDA tensors")
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Q, K, and V must share shape [B, H, T, C]")
    if query.shape[2] != num_frames * tokens_per_frame:
        raise ValueError("token count does not match the frame layout")
    if not 0 <= special_tokens < tokens_per_frame:
        raise ValueError("special token count must be inside each frame")

    patch_tokens = tokens_per_frame - special_tokens
    if special_tokens == 0:
        q_out = query.contiguous()
        k_out = key.contiguous()
        v_out = value.contiguous()
    else:
        q_out = torch.empty_like(query, memory_format=torch.contiguous_format)
        k_out = torch.empty_like(key, memory_format=torch.contiguous_format)
        v_out = torch.empty_like(value, memory_format=torch.contiguous_format)
        total = q_out.numel()
        _pack_qkv_layout_kernel[(triton.cdiv(total, 256),)](
            query,
            key,
            value,
            q_out,
            k_out,
            v_out,
            total,
            query.shape[2],
            query.shape[3],
            num_frames,
            tokens_per_frame,
            special_tokens,
            patch_tokens,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            query.stride(3),
            key.stride(0),
            key.stride(1),
            key.stride(2),
            key.stride(3),
            value.stride(0),
            value.stride(1),
            value.stride(2),
            value.stride(3),
            heads=query.shape[1],
            BLOCK=256,
            num_warps=4,
        )

    patch_count = num_frames * patch_tokens
    q_patch = q_out[..., :patch_count, :]
    q_special = q_out[..., patch_count:, :] if special_tokens else None
    return (
        q_patch,
        q_special,
        k_out[..., :patch_count, :],
        k_out,
        v_out[..., :patch_count, :],
        v_out,
    )
