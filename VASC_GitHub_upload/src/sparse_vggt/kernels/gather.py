"""Triton gather kernels for compact grouped attention buffers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_flat_tokens_kernel(
    source_ptr,
    index_ptr,
    output_ptr,
    source_tokens: tl.constexpr,
    selected_tokens: tl.constexpr,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    selected = row % selected_tokens
    outer = row // selected_tokens
    group = outer % groups
    outer = outer // groups
    head = outer % heads
    batch = outer // heads

    source_token = tl.load(
        index_ptr + (batch * groups + group) * selected_tokens + selected
    )
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    source_offsets = (
        ((batch * heads + head) * source_tokens + source_token) * head_dim
        + dim_offsets
    )
    values = tl.load(source_ptr + source_offsets, mask=dim_mask)
    tl.store(output_ptr + row * head_dim + dim_offsets, values, mask=dim_mask)


@triton.jit
def _gather_frame_tokens_kernel(
    source_ptr,
    index_ptr,
    output_ptr,
    num_frames: tl.constexpr,
    selected_frames: tl.constexpr,
    tokens_per_frame: tl.constexpr,
    head_dim: tl.constexpr,
    heads: tl.constexpr,
    groups: tl.constexpr,
    block_dim: tl.constexpr,
):
    row = tl.program_id(0)
    token = row % tokens_per_frame
    outer = row // tokens_per_frame
    selected_frame = outer % selected_frames
    outer = outer // selected_frames
    group = outer % groups
    outer = outer // groups
    head = outer % heads
    batch = outer // heads

    frame = tl.load(
        index_ptr + (batch * groups + group) * selected_frames + selected_frame
    )
    source_token = frame * tokens_per_frame + token
    source_tokens = num_frames * tokens_per_frame
    dim_offsets = tl.arange(0, block_dim)
    dim_mask = dim_offsets < head_dim
    source_offsets = (
        ((batch * heads + head) * source_tokens + source_token) * head_dim
        + dim_offsets
    )
    values = tl.load(source_ptr + source_offsets, mask=dim_mask)
    tl.store(output_ptr + row * head_dim + dim_offsets, values, mask=dim_mask)


def _validate_inputs(
    frame_tokens: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if not frame_tokens.is_cuda or not indices.is_cuda:
        raise ValueError("Triton gather requires CUDA tensors")
    if frame_tokens.ndim != 5:
        raise ValueError("frame_tokens must be five-dimensional")
    if indices.ndim != 3 or indices.shape[0] != frame_tokens.shape[0]:
        raise ValueError("indices must have shape [batch, groups, selected]")
    if indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("indices must use an integer dtype")
    if frame_tokens.device != indices.device:
        raise ValueError("frame_tokens and indices must use the same device")
    return tuple(frame_tokens.shape)


def triton_gather_flat_frame_tokens(
    frame_tokens: torch.Tensor,
    flat_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather arbitrary flattened frame tokens into grouped compact rows."""
    batch, heads, num_frames, tokens_per_frame, head_dim = _validate_inputs(
        frame_tokens, flat_token_indices
    )
    groups, selected_tokens = flat_token_indices.shape[1:]
    output = torch.empty(
        batch,
        heads,
        groups,
        selected_tokens,
        head_dim,
        device=frame_tokens.device,
        dtype=frame_tokens.dtype,
    )
    if output.numel() == 0:
        return output
    source = frame_tokens.contiguous()
    indices = flat_token_indices.contiguous()
    block_dim = triton.next_power_of_2(head_dim)
    grid = (batch * heads * groups * selected_tokens,)
    _gather_flat_tokens_kernel[grid](
        source,
        indices,
        output,
        source_tokens=num_frames * tokens_per_frame,
        selected_tokens=selected_tokens,
        head_dim=head_dim,
        heads=heads,
        groups=groups,
        block_dim=block_dim,
    )
    return output


def triton_gather_frame_tokens(
    frame_tokens: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather complete frame-token rows into grouped compact buffers."""
    batch, heads, num_frames, tokens_per_frame, head_dim = _validate_inputs(
        frame_tokens, frame_indices
    )
    groups, selected_frames = frame_indices.shape[1:]
    output = torch.empty(
        batch,
        heads,
        groups,
        selected_frames,
        tokens_per_frame,
        head_dim,
        device=frame_tokens.device,
        dtype=frame_tokens.dtype,
    )
    if output.numel() == 0:
        return output
    source = frame_tokens.contiguous()
    indices = frame_indices.contiguous()
    block_dim = triton.next_power_of_2(head_dim)
    grid = (
        batch * heads * groups * selected_frames * tokens_per_frame,
    )
    _gather_frame_tokens_kernel[grid](
        source,
        indices,
        output,
        num_frames=num_frames,
        selected_frames=selected_frames,
        tokens_per_frame=tokens_per_frame,
        head_dim=head_dim,
        heads=heads,
        groups=groups,
        block_dim=block_dim,
    )
    return output
