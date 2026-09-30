import math
import os

import spas_sage_attn._qattn as qattn
import torch
import torch.nn.functional as F
from spas_sage_attn.quant_per_block import per_block_int8
from spas_sage_attn.utils import (
    block_map_lut_triton,
    fill_block_map_triton,
    hyperparameter_check,
)

from collections import namedtuple
SortResult = namedtuple("SortResult", ["values", "indices"])

_RADIAL_BLOCK_MASK_CACHE = {}
_RADIAL_BLOCK_MASK_CACHE_SIZE = 8
_SOFT_GEOMETRY_CACHE = {}
_SOFT_GEOMETRY_CACHE_SIZE = 8
_FRAME_KEY_TABLE_CACHE = {}
_FRAME_KEY_TABLE_CACHE_SIZE = 16
_BLOCK_DEBT_LAYOUT_CACHE = {}
_BLOCK_DEBT_LAYOUT_CACHE_SIZE = 8


def radial_width(
    frame_i: int,
    frame_j: int,
    token_per_frame: int,
    block_size: int = 128,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
) -> int:
    """Compute radial attention window width for a frame pair."""
    dist = abs(frame_i - frame_j)
    if dist <= dense_neighbor:
        return token_per_frame

    group = dist.bit_length()
    width = (2 ** token_per_frame.bit_length()) / (2**group) * decay_factor
    return int(max(block_size, min(token_per_frame, width)))


def compute_block_to_frame_mapping(
    num_frames: int,
    tokens_per_frame: int,
    block_size: int,
) -> list:
    """
    Compute which frame each pooled block belongs to, accounting for cross-frame blocks.

    This handles the actual pooling behavior where tokens are concatenated before pooling,
    so blocks can span frame boundaries when tokens_per_frame % block_size != 0.

    Args:
        num_frames: Number of frames
        tokens_per_frame: Tokens per frame
        block_size: Pooling kernel size

    Returns:
        List of (frame_idx, start_token_in_frame, end_token_in_frame) for each block
        where start/end are relative to the frame's token range
    """
    import math

    total_tokens = num_frames * tokens_per_frame
    num_blocks = math.ceil(total_tokens / block_size)

    block_mapping = []
    for blk_idx in range(num_blocks):
        # Absolute token range for this block
        token_start = blk_idx * block_size
        token_end = min((blk_idx + 1) * block_size, total_tokens)

        # Find which frame(s) this block spans
        frame_start = token_start // tokens_per_frame
        frame_end = (token_end - 1) // tokens_per_frame

        # For simplicity, assign block to the frame containing most of its tokens
        if frame_start == frame_end:
            # Block is entirely within one frame
            primary_frame = frame_start
        else:
            # Block spans frames - assign to frame with more tokens
            tokens_in_start_frame = (frame_start + 1) * tokens_per_frame - token_start
            tokens_in_end_frame = token_end - frame_end * tokens_per_frame
            primary_frame = frame_start if tokens_in_start_frame >= tokens_in_end_frame else frame_end

        block_mapping.append(primary_frame)

    return block_mapping


# ============================================================
# 优化后的 build_radial_block_mask 实现
# 用于直接替换 sparse_wrapper.py 中的原函数（第87-266行）
# ============================================================
# 性能: 135,453ms → 3,750ms (36.1× 加速)
# 正确性: 4轮Codex审查验证
# 日期: 2026-06-02
# ============================================================

def bit_length_vectorized(x: torch.Tensor) -> torch.Tensor:
    """Compute bit_length for a tensor of positive integers using integer-safe logic.

    Equivalent to Python's int.bit_length() but vectorized.
    For x > 0: bit_length(x) = floor(log2(x)) + 1
    For x = 0: bit_length(0) = 0

    Uses pure integer operations to avoid ALL floating-point precision issues.

    Args:
        x: Tensor of non-negative integers

    Returns:
        bit_length for each element
    """
    result = torch.zeros_like(x, dtype=torch.long)
    mask = x > 0

    if mask.any():
        x_positive = x[mask]

        # Integer-safe bit_length: count leading zeros
        # For each value, find the position of the highest set bit
        bit_lengths = torch.zeros_like(x_positive, dtype=torch.long)

        # Iterate to find bit length (this is exact for all integers)
        for bit_pos in range(63, -1, -1):  # Check from highest bit down
            has_bit = (x_positive >> bit_pos) > 0
            bit_lengths = torch.where(has_bit & (bit_lengths == 0),
                                     torch.tensor(bit_pos + 1, dtype=torch.long, device=x.device),
                                     bit_lengths)

        result[mask] = bit_lengths

    return result


def radial_width_vectorized(
    frame_i: torch.Tensor,  # Shape: [num_frames, 1] or broadcastable
    frame_j: torch.Tensor,  # Shape: [1, num_frames] or broadcastable
    token_per_frame: int,
    block_size: int = 128,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
) -> torch.Tensor:
    """Vectorized computation of radial attention window widths.

    FIXED: Uses correct bit_length semantics matching original implementation.

    Args:
        frame_i: Query frame indices (broadcastable)
        frame_j: Key frame indices (broadcastable)
        token_per_frame: Number of tokens per frame
        block_size: Minimum block size
        decay_factor: Radial decay factor
        dense_neighbor: Dense neighbor threshold

    Returns:
        width_tokens: [*broadcast_shape] tensor of radial widths
    """
    dist = (frame_i - frame_j).abs()

    # Initialize with token_per_frame
    width = torch.full_like(dist, token_per_frame, dtype=torch.float32)

    # Mask for non-dense neighbors
    non_dense = dist > dense_neighbor

    if non_dense.any():
        # FIXED: Use bit_length semantics matching original
        # Original: group = dist.bit_length()
        # For integer distance values
        dist_non_dense = dist[non_dense]
        group = bit_length_vectorized(dist_non_dense).float()

        # FIXED: Use bit_length for token_per_frame too
        # Original: token_bits = token_per_frame.bit_length()
        token_bits = token_per_frame.bit_length() if token_per_frame > 0 else 0

        # Compute width for non-dense pairs
        base_width = (2 ** token_bits) / (2 ** group) * decay_factor

        # Clamp to [block_size, token_per_frame]
        computed_width = torch.clamp(base_width, min=block_size, max=token_per_frame)

        width[non_dense] = computed_width

    return width.long()


def estimate_memory_gb(total_q_blocks: int, total_k_blocks: int, num_frames: int) -> float:
    """Estimate peak memory usage for 4D tensor operations.

    Returns estimated peak memory in GB.
    """
    # 4D tensor shape: [total_q_blocks, total_k_blocks, num_frames, num_frames]
    num_elements = total_q_blocks * total_k_blocks * num_frames * num_frames

    # Conservative estimate: multiple 4D tensors coexist
    # - overlaps (bool): 1 byte
    # - dense_mask (bool): 1 byte
    # - non_dense_overlaps (bool): 1 byte
    # - In sparse path: dist1, dist2, min_dist (int64): 3 × 8 bytes
    # - Position tensors (int64): 4 × 8 bytes
    # Total per element: ~43 bytes (as found by Codex)

    bytes_per_element = 43
    total_bytes = num_elements * bytes_per_element
    total_gb = total_bytes / (1024 ** 3)

    return total_gb


def build_radial_block_mask_2d(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    device=None,
) -> torch.Tensor:
    """Build the exact conservative block mask using O(q_blocks * k_blocks) memory."""
    total_tokens = num_frames * tokens_per_frame
    total_q_blocks = math.ceil(total_tokens / q_block_size)
    total_k_blocks = math.ceil(total_tokens / k_block_size)

    # The fast endpoint formulation assumes a block spans at most two frames.
    if q_block_size > tokens_per_frame or k_block_size > tokens_per_frame:
        return build_radial_block_mask_exact(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            device=device,
        )

    q_start = torch.arange(total_q_blocks, dtype=torch.long) * q_block_size
    q_end = torch.clamp(q_start + q_block_size, max=total_tokens)
    k_start = torch.arange(total_k_blocks, dtype=torch.long) * k_block_size
    k_end = torch.clamp(k_start + k_block_size, max=total_tokens)

    q_frames = torch.stack((q_start // tokens_per_frame, (q_end - 1) // tokens_per_frame))
    k_frames = torch.stack((k_start // tokens_per_frame, (k_end - 1) // tokens_per_frame))
    mask = torch.zeros((total_q_blocks, total_k_blocks), dtype=torch.bool)

    for q_side in range(2):
        q_frame = q_frames[q_side]
        q_local_start = torch.clamp(
            q_start - q_frame * tokens_per_frame, min=0, max=tokens_per_frame
        )
        q_local_end = torch.clamp(
            q_end - q_frame * tokens_per_frame, min=0, max=tokens_per_frame
        )
        q_valid = q_local_end > q_local_start

        for k_side in range(2):
            k_frame = k_frames[k_side]
            k_local_start = torch.clamp(
                k_start - k_frame * tokens_per_frame, min=0, max=tokens_per_frame
            )
            k_local_end = torch.clamp(
                k_end - k_frame * tokens_per_frame, min=0, max=tokens_per_frame
            )
            k_valid = k_local_end > k_local_start

            qf = q_frame[:, None]
            kf = k_frame[None, :]
            frame_dist = (qf - kf).abs()
            width = radial_width_vectorized(
                qf,
                kf,
                tokens_per_frame,
                min(q_block_size, k_block_size),
                decay_factor,
                dense_neighbor,
            )

            min_dist = torch.maximum(
                torch.maximum(
                    k_local_start[None, :] - q_local_end[:, None] + 1,
                    q_local_start[:, None] - k_local_end[None, :] + 1,
                ),
                torch.zeros((), dtype=torch.long),
            )
            valid = q_valid[:, None] & k_valid[None, :]
            mask |= valid & (
                (frame_dist <= dense_neighbor) | (min_dist <= width)
            )

    return mask.to(device) if device is not None else mask


def build_radial_block_mask(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    device=None,
    max_memory_gb: float = 15.0,
) -> torch.Tensor:
    """
    Generate block-level radial spatial constraint mask (OPTIMIZED VERSION).

    This is the optimized 4D-tensor vectorized implementation that replaces
    the original O(num_frames²) double-loop version.

    Performance: 135,453ms → 3,750ms (36.1× speedup for 1280 frames)
    Correctness: Verified by 4 rounds of Codex review
    Memory: ~13 GB for 1280 frames

    Args:
        num_frames: Number of frames in the sequence (e.g., 1280)
        tokens_per_frame: Number of patch tokens per frame (e.g., 1)
        q_block_size: Query pooling kernel size (default 128)
        k_block_size: Key pooling kernel size (default 64)
        decay_factor: Radial decay factor (higher = wider windows)
        dense_neighbor: Number of neighboring frames with full attention
        device: torch device for the mask tensor
        max_memory_gb: Maximum allowed memory usage (default 15 GB, allows 1280 frames)

    Returns:
        mask: (total_q_blocks, total_k_blocks) bool tensor
              True = allowed to attend, False = masked out

    Raises:
        MemoryError: If estimated memory exceeds max_memory_gb
    """
    # The previous implementation materialized tensors shaped
    # [q_blocks, k_blocks, num_frames, num_frames]. The 2D endpoint
    # formulation is exact for VGGT's 128/64 blocks and avoids that N^2 factor.
    return build_radial_block_mask_2d(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        decay_factor=decay_factor,
        dense_neighbor=dense_neighbor,
        device=device,
    )

    # Compute dimensions
    total_q_tokens = num_frames * tokens_per_frame
    total_k_tokens = num_frames * tokens_per_frame
    total_q_blocks = math.ceil(total_q_tokens / q_block_size)
    total_k_blocks = math.ceil(total_k_tokens / k_block_size)

    # Memory safety check
    estimated_memory = estimate_memory_gb(total_q_blocks, total_k_blocks, num_frames)
    if estimated_memory > max_memory_gb:
        raise MemoryError(
            f"Estimated memory {estimated_memory:.1f} GB exceeds limit {max_memory_gb:.1f} GB. "
            f"Consider using original implementation or increasing max_memory_gb."
        )

    # Block indices
    q_blk_idx = torch.arange(total_q_blocks, dtype=torch.long).view(-1, 1)
    k_blk_idx = torch.arange(total_k_blocks, dtype=torch.long).view(1, -1)

    # Token ranges
    q_token_start = q_blk_idx * q_block_size
    q_token_end = torch.clamp((q_blk_idx + 1) * q_block_size, max=total_q_tokens)
    k_token_start = k_blk_idx * k_block_size
    k_token_end = torch.clamp((k_blk_idx + 1) * k_block_size, max=total_k_tokens)

    # Frame ranges
    q_frame_start = q_token_start // tokens_per_frame
    q_frame_end = (q_token_end - 1) // tokens_per_frame
    k_frame_start = k_token_start // tokens_per_frame
    k_frame_end = (k_token_end - 1) // tokens_per_frame

    # Frame grids
    q_frame_grid = torch.arange(num_frames, dtype=torch.long).view(-1, 1)
    k_frame_grid = torch.arange(num_frames, dtype=torch.long).view(1, -1)

    # Precompute radial widths
    radial_widths = radial_width_vectorized(
        q_frame_grid, k_frame_grid, tokens_per_frame,
        min(q_block_size, k_block_size), decay_factor, dense_neighbor
    )

    # Frame distances
    frame_dist = (q_frame_grid - k_frame_grid).abs()
    is_dense = frame_dist <= dense_neighbor

    # Overlap detection (4D broadcasting)
    q_overlaps = (q_frame_start.unsqueeze(-1).unsqueeze(-1) <= q_frame_grid.unsqueeze(0).unsqueeze(0)) & \
                 (q_frame_grid.unsqueeze(0).unsqueeze(0) <= q_frame_end.unsqueeze(-1).unsqueeze(-1))

    k_overlaps = (k_frame_start.unsqueeze(-1).unsqueeze(-1) <= k_frame_grid.unsqueeze(0).unsqueeze(0)) & \
                 (k_frame_grid.unsqueeze(0).unsqueeze(0) <= k_frame_end.unsqueeze(-1).unsqueeze(-1))

    overlaps = q_overlaps & k_overlaps

    # Dense mask
    is_dense_4d = is_dense.unsqueeze(0).unsqueeze(0)
    dense_mask = overlaps & is_dense_4d

    # Sparse radial constraint
    non_dense_overlaps = overlaps & (~is_dense_4d)

    if non_dense_overlaps.any():
        # Compute position ranges
        q_frame_offset = q_frame_grid.unsqueeze(0).unsqueeze(0) * tokens_per_frame
        zero_4d_q = torch.zeros(total_q_blocks, 1, num_frames, 1, dtype=torch.long)
        tpf_4d_q = torch.full((total_q_blocks, 1, num_frames, 1), tokens_per_frame, dtype=torch.long)

        q_pos_start = torch.maximum(zero_4d_q, q_token_start.unsqueeze(-1).unsqueeze(-1) - q_frame_offset)
        q_pos_end = torch.minimum(tpf_4d_q, q_token_end.unsqueeze(-1).unsqueeze(-1) - q_frame_offset)

        k_frame_offset = k_frame_grid.unsqueeze(0).unsqueeze(0) * tokens_per_frame
        zero_4d_k = torch.zeros(1, total_k_blocks, 1, num_frames, dtype=torch.long)
        tpf_4d_k = torch.full((1, total_k_blocks, 1, num_frames), tokens_per_frame, dtype=torch.long)

        k_pos_start = torch.maximum(zero_4d_k, k_token_start.unsqueeze(-1).unsqueeze(-1) - k_frame_offset)
        k_pos_end = torch.minimum(tpf_4d_k, k_token_end.unsqueeze(-1).unsqueeze(-1) - k_frame_offset)

        # Minimum distance
        dist1 = k_pos_start - q_pos_end + 1
        dist2 = q_pos_start - k_pos_end + 1
        zero_4d_full = torch.zeros(total_q_blocks, total_k_blocks, num_frames, num_frames, dtype=torch.long)
        min_dist = torch.maximum(torch.maximum(dist1, dist2), zero_4d_full)

        # Check radial width
        radial_widths_4d = radial_widths.unsqueeze(0).unsqueeze(0)
        within_width = min_dist <= radial_widths_4d
        sparse_mask = non_dense_overlaps & within_width
    else:
        sparse_mask = torch.zeros_like(overlaps)

    # Combine and reduce
    combined_mask = dense_mask | sparse_mask
    mask = combined_mask.any(dim=-1).any(dim=-1)

    # Transfer to device
    if device is not None:
        mask = mask.to(device)

    return mask



def build_radial_block_mask_exact(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    device=None,
) -> torch.Tensor:
    """
    Generate radial block mask using exact spatial position checking.

    This version computes the actual token positions for each block and checks
    if they fall within the radial window, rather than just checking frame distances.

    Args:
        Same as build_radial_block_mask()

    Returns:
        mask: (total_q_blocks, total_k_blocks) bool tensor
    """
    import math

    total_q_tokens = num_frames * tokens_per_frame
    total_k_tokens = num_frames * tokens_per_frame

    total_q_blocks = math.ceil(total_q_tokens / q_block_size)
    total_k_blocks = math.ceil(total_k_tokens / k_block_size)

    mask = torch.zeros((total_q_blocks, total_k_blocks), dtype=torch.bool)

    # For each query block
    for q_blk in range(total_q_blocks):
        # Compute token range for this query block
        q_token_start = q_blk * q_block_size
        q_token_end = min((q_blk + 1) * q_block_size, total_q_tokens)

        # For each key block
        for k_blk in range(total_k_blocks):
            # Compute token range for this key block
            k_token_start = k_blk * k_block_size
            k_token_end = min((k_blk + 1) * k_block_size, total_k_tokens)

            # Conservative: allow if ANY token pair in these blocks would be radial-allowed
            # Get frame ranges
            q_frame_start = q_token_start // tokens_per_frame
            q_frame_end = (q_token_end - 1) // tokens_per_frame
            k_frame_start = k_token_start // tokens_per_frame
            k_frame_end = (k_token_end - 1) // tokens_per_frame

            # Check all frame pair combinations
            allowed = False
            for q_frame in range(q_frame_start, min(q_frame_end + 1, num_frames)):
                for k_frame in range(k_frame_start, min(k_frame_end + 1, num_frames)):
                    # Compute radial width for this frame pair
                    width_tokens = radial_width(
                        q_frame, k_frame,
                        tokens_per_frame,
                        min(q_block_size, k_block_size),
                        decay_factor, dense_neighbor
                    )

                    frame_dist = abs(q_frame - k_frame)

                    if frame_dist <= dense_neighbor:
                        # Dense neighbor frames: allow
                        allowed = True
                        break
                    else:
                        # Check spatial distance for tokens in these frames
                        q_pos_start = max(0, q_token_start - q_frame * tokens_per_frame)
                        q_pos_end = min(tokens_per_frame, q_token_end - q_frame * tokens_per_frame)
                        k_pos_start = max(0, k_token_start - k_frame * tokens_per_frame)
                        k_pos_end = min(tokens_per_frame, k_token_end - k_frame * tokens_per_frame)

                        # Minimum distance between any token pair
                        # Note: pos_end is exclusive, so add 1 for correct distance
                        min_dist = max(
                            0,
                            k_pos_start - q_pos_end + 1,
                            q_pos_start - k_pos_end + 1
                        )
                        if min_dist <= width_tokens:
                            allowed = True
                            break

                if allowed:
                    break

            if allowed:
                mask[q_blk, k_blk] = True

    # Transfer to device if specified
    if device is not None:
        mask = mask.to(device)

    return mask


def _get_cached_radial_block_mask(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int,
    k_block_size: int,
    decay_factor: float,
    dense_neighbor: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an immutable-by-convention radial geometry mask for inference reuse."""
    import time
    normalized_device = torch.device(device)
    cache_key = (
        num_frames,
        tokens_per_frame,
        q_block_size,
        k_block_size,
        float(decay_factor),
        dense_neighbor,
        normalized_device.type,
        normalized_device.index,
    )
    radial_mask = _RADIAL_BLOCK_MASK_CACHE.get(cache_key)
    if radial_mask is None:
        t0 = time.perf_counter()
        radial_mask = build_radial_block_mask(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            device=normalized_device,
        )
        t1 = time.perf_counter()
        # print(f"[PERF] build_radial_block_mask: {(t1-t0)*1000:.2f}ms (CACHE MISS)")
        if len(_RADIAL_BLOCK_MASK_CACHE) >= _RADIAL_BLOCK_MASK_CACHE_SIZE:
            _RADIAL_BLOCK_MASK_CACHE.pop(next(iter(_RADIAL_BLOCK_MASK_CACHE)))
        _RADIAL_BLOCK_MASK_CACHE[cache_key] = radial_mask
    else:
        # Cache hit
        # print(f"[PERF] build_radial_block_mask: 0.00ms (CACHE HIT)")
        pass
    return radial_mask


def _select_importance_in_region(
    pooled_score: torch.Tensor,
    allowed_mask: torch.Tensor,
    sparse_ratio: float | None,
    cdf_threshold: float | None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Select important blocks only inside a shared 2D allowed region."""
    B, nh, q_blocks, k_blocks = pooled_score.shape
    allowed = allowed_mask.unsqueeze(0).unsqueeze(0).expand(B, nh, -1, -1)
    num_allowed = allowed_mask.sum(dim=-1)

    if sparse_ratio is None and cdf_threshold is None:
        raise ValueError("Either sparse_ratio or cdf_threshold must be specified")
    if sparse_ratio is not None and not (0.0 <= sparse_ratio <= 1.0):
        raise ValueError(f"sparse_ratio must be in [0, 1], got {sparse_ratio}")
    if cdf_threshold is not None and not (0.0 <= cdf_threshold <= 1.0):
        raise ValueError(f"cdf_threshold must be in [0, 1], got {cdf_threshold}")

    masked_scores = pooled_score.clone()
    masked_scores.nan_to_num_(nan=0.0)
    masked_scores.masked_fill_(~allowed, float("-inf"))

    if cdf_threshold is not None:
        normalized = pooled_score.clone()
        normalized.nan_to_num_(nan=0.0)
        normalized.masked_fill_(~allowed, 0.0)
        score_sum = normalized.sum(dim=-1, keepdim=True)
        zero_sum = score_sum.squeeze(-1) == 0
        normalized = normalized / torch.where(
            score_sum > 0, score_sum, torch.ones_like(score_sum)
        )
        _, sorted_indices = torch.sort(masked_scores, dim=-1, descending=True)
        sorted_normalized = torch.gather(normalized, -1, sorted_indices)
        cdf = torch.cumsum(sorted_normalized, dim=-1)
        if cdf_threshold <= 0:
            num_to_keep = torch.zeros_like(cdf[..., 0], dtype=torch.long)
        else:
            # Include the first block that reaches or crosses the target mass.
            num_to_keep = (cdf < (cdf_threshold - eps)).sum(dim=-1) + 1
        no_candidates = num_allowed.view(1, 1, q_blocks) == 0
        num_to_keep = torch.where(
            zero_sum | no_candidates,
            torch.zeros_like(num_to_keep),
            num_to_keep,
        )

        if sparse_ratio is not None:
            min_keep = (num_allowed.float() * (1.0 - sparse_ratio)).long()
            min_keep = min_keep.view(1, 1, q_blocks).expand(B, nh, -1)
            num_to_keep = torch.where(
                zero_sum,
                torch.zeros_like(num_to_keep),
                torch.maximum(num_to_keep, min_keep),
            )
        max_keep = num_allowed.view(1, 1, q_blocks).expand(B, nh, -1)
        num_to_keep = torch.minimum(num_to_keep, max_keep)
    else:
        num_to_keep = (num_allowed.float() * (1.0 - sparse_ratio)).long()
        num_to_keep = num_to_keep.view(1, 1, q_blocks).expand(B, nh, -1)
        sorted_indices = None

    max_k = int(num_to_keep.max().item())
    selected = torch.zeros_like(pooled_score, dtype=torch.bool)
    if max_k == 0:
        return selected

    if sorted_indices is None:
        _, selected_indices = torch.topk(
            masked_scores, k=max_k, dim=-1, largest=True, sorted=True
        )
    else:
        selected_indices = sorted_indices[..., :max_k]

    positions = torch.arange(max_k, device=pooled_score.device).view(1, 1, 1, -1)
    valid = positions < num_to_keep.unsqueeze(-1)
    selected.scatter_(-1, selected_indices, valid)
    selected.logical_and_(allowed)
    return selected


def get_distance_routed_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    route_frame_threshold: int = 4,
    use_covariance_aware_importance: bool = False,
    covariance_weight: float = 0.5,
    covariance_eps: float = 1e-8,
    key_block_variance: torch.Tensor | None = None,
    use_adaptive_slit_routing: bool = False,
    adaptive_slit_temporal_window: int = 10,
    adaptive_slit_stable_quantile: float = 0.6,
    adaptive_slit_change_quantile: float = 0.7,
    adaptive_slit_narrow_width: int = 1,
    adaptive_slit_base_width: int = 2,
    adaptive_slit_expand_width: int = 4,
    stats_store: dict | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Use radial blocks for near frame pairs and importance for far frame pairs."""
    if route_frame_threshold < 0:
        raise ValueError(
            f"route_frame_threshold must be non-negative, got {route_frame_threshold}"
        )
    if covariance_weight < 0:
        raise ValueError(f"covariance_weight must be non-negative, got {covariance_weight}")
    if covariance_eps <= 0:
        raise ValueError(f"covariance_eps must be positive, got {covariance_eps}")
    if adaptive_slit_temporal_window < 1:
        raise ValueError("adaptive_slit_temporal_window must be >= 1")
    for name, value in (
        ("adaptive_slit_narrow_width", adaptive_slit_narrow_width),
        ("adaptive_slit_base_width", adaptive_slit_base_width),
        ("adaptive_slit_expand_width", adaptive_slit_expand_width),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
    if adaptive_slit_narrow_width > adaptive_slit_base_width:
        raise ValueError("adaptive_slit_narrow_width must be <= adaptive_slit_base_width")
    if adaptive_slit_base_width > adaptive_slit_expand_width:
        raise ValueError("adaptive_slit_base_width must be <= adaptive_slit_expand_width")
    if not 0.0 <= adaptive_slit_stable_quantile <= 1.0:
        raise ValueError("adaptive_slit_stable_quantile must be in [0, 1]")
    if not 0.0 <= adaptive_slit_change_quantile <= 1.0:
        raise ValueError("adaptive_slit_change_quantile must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    radial_mask = _get_cached_radial_block_mask(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        decay_factor=decay_factor,
        dense_neighbor=dense_neighbor,
        device=device,
    )
    if radial_mask.shape != (q_blocks, k_blocks):
        raise ValueError(
            f"Radial mask shape {radial_mask.shape} does not match "
            f"pooled score blocks {(q_blocks, k_blocks)}"
        )

    total_tokens = num_frames * tokens_per_frame
    q_start = torch.arange(q_blocks, device=device) * q_block_size
    q_end = torch.clamp(q_start + q_block_size, max=total_tokens)
    k_start = torch.arange(k_blocks, device=device) * k_block_size
    k_end = torch.clamp(k_start + k_block_size, max=total_tokens)
    q_frame_start = q_start // tokens_per_frame
    q_frame_end = (q_end - 1) // tokens_per_frame
    k_frame_start = k_start // tokens_per_frame
    k_frame_end = (k_end - 1) // tokens_per_frame
    frame_distance = torch.maximum(
        torch.maximum(
            k_frame_start[None, :] - q_frame_end[:, None],
            q_frame_start[:, None] - k_frame_end[None, :],
        ),
        torch.zeros((), dtype=torch.long, device=device),
    )
    near_region = frame_distance <= route_frame_threshold

    if use_adaptive_slit_routing:
        q_center = (q_start + q_end - 1) // 2
        k_center = (k_start + k_end - 1) // 2
        q_local_bin = (q_center % tokens_per_frame) // k_block_size
        k_local_bin = (k_center % tokens_per_frame) // k_block_size
        local_distance = (q_local_bin[:, None] - k_local_bin[None, :]).abs()

        slit_region = near_region & (frame_distance > dense_neighbor)
        narrow_slit = slit_region & (local_distance <= adaptive_slit_narrow_width)
        base_slit = slit_region & (local_distance <= adaptive_slit_base_width)
        expand_slit = slit_region & (local_distance <= adaptive_slit_expand_width)

        stats_radius = min(
            route_frame_threshold,
            max(0, adaptive_slit_temporal_window // 2),
        )
        stats_region = frame_distance <= stats_radius
        stats_region = stats_region & (frame_distance > dense_neighbor)
        if not bool(stats_region.any()):
            stats_region = slit_region

        stats_allowed = stats_region.view(1, 1, q_blocks, k_blocks)
        stats_count = stats_region.sum(dim=-1).clamp_min(1).view(1, 1, q_blocks)
        score_for_stats = pooled_score.float()
        mean_score = (score_for_stats * stats_allowed).sum(dim=-1) / stats_count
        centered = (score_for_stats - mean_score.unsqueeze(-1)) * stats_allowed
        var_score = (centered * centered).sum(dim=-1) / stats_count

        mean_hi = torch.quantile(
            mean_score, adaptive_slit_stable_quantile, dim=-1, keepdim=True
        )
        mean_lo = torch.quantile(
            mean_score, 1.0 - adaptive_slit_change_quantile, dim=-1, keepdim=True
        )
        var_lo = torch.quantile(
            var_score, 1.0 - adaptive_slit_stable_quantile, dim=-1, keepdim=True
        )
        var_hi = torch.quantile(
            var_score, adaptive_slit_change_quantile, dim=-1, keepdim=True
        )
        stable_query = (mean_score >= mean_hi) & (var_score <= var_lo)
        changing_query = ((var_score >= var_hi) | (mean_score <= mean_lo)) & ~stable_query
        middle_query = ~(stable_query | changing_query)

        dense_core = radial_mask & near_region & (frame_distance <= dense_neighbor)
        adaptive_near = dense_core.view(1, 1, q_blocks, k_blocks).expand(B, nh, -1, -1)
        adaptive_near = adaptive_near | (
            stable_query.unsqueeze(-1)
            & narrow_slit.view(1, 1, q_blocks, k_blocks)
        )
        adaptive_near = adaptive_near | (
            middle_query.unsqueeze(-1)
            & base_slit.view(1, 1, q_blocks, k_blocks)
        )
        adaptive_near = adaptive_near | (
            changing_query.unsqueeze(-1)
            & expand_slit.view(1, 1, q_blocks, k_blocks)
        )

        if stats_store is not None:
            near_total = near_region.sum().clamp_min(1).float()
            stats_store.update(
                {
                    "adaptive_slit_stable_fraction": stable_query.float().mean().detach(),
                    "adaptive_slit_change_fraction": changing_query.float().mean().detach(),
                    "adaptive_slit_selected_near_fraction": (
                        adaptive_near.float().mean(dim=(0, 1)).sum() / near_total
                    ).detach(),
                    "adaptive_slit_dense_core_fraction": (
                        dense_core.float().sum() / near_total
                    ).detach(),
                }
            )
        near_mask = adaptive_near
    else:
        near_radial = radial_mask & near_region
        near_mask = near_radial.view(1, 1, q_blocks, k_blocks).expand(B, nh, -1, -1)

    far_score = pooled_score
    if use_covariance_aware_importance:
        if key_block_variance is None:
            raise ValueError(
                "key_block_variance is required for covariance-aware importance"
            )
        expected_shape = (B, nh, k_blocks)
        if key_block_variance.shape != expected_shape:
            raise ValueError(
                f"key_block_variance shape {key_block_variance.shape} does not match "
                f"expected {expected_shape}"
            )
        log_content = pooled_score.float().clamp_min(covariance_eps).log()
        log_variance = key_block_variance.float().clamp_min(covariance_eps).log()
        far_logits = log_content + covariance_weight * log_variance.unsqueeze(-2)
        far_score = torch.softmax(far_logits, dim=-1).to(pooled_score.dtype)

        if stats_store is not None:
            stats_store.update(
                {
                    "covariance_k_variance_mean": key_block_variance.float().mean().detach(),
                    "covariance_k_variance_std": key_block_variance.float().std().detach(),
                    "covariance_score_shift_std": (
                        covariance_weight * log_variance
                    ).std().detach(),
                }
            )

    far_importance = _select_importance_in_region(
        pooled_score=far_score,
        allowed_mask=~near_region,
        sparse_ratio=sparse_ratio,
        cdf_threshold=cdf_threshold,
        eps=eps,
    )
    return near_mask | far_importance


def get_radial_layerwise_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    layer_idx: int = 0,
    layer_sparsity_ratios: list = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fuse radial spatial constraint + layer-wise sparsity + importance-based selection.

    This function combines:
    1. Radial constraint: spatial locality mask from build_radial_block_mask()
    2. Layer-wise sparsity: layer-specific sparse_ratio or cdf_threshold
    3. Importance scoring: select top-k blocks based on pooled attention scores

    Args:
        pooled_score: Pooled attention scores, shape (B, nh, q_blocks, k_blocks)
        num_frames: Number of frames in the sequence
        tokens_per_frame: Number of patch tokens per frame
        q_block_size: Query pooling kernel size (default 128)
        k_block_size: Key pooling kernel size (default 64)
        sparse_ratio: Base sparsity ratio (fraction to remove, default None)
        cdf_threshold: CDF threshold for adaptive selection (default None)
        decay_factor: Radial decay factor (default 1.0)
        dense_neighbor: Number of neighboring frames with full attention (default 1)
        layer_idx: Current layer index (for layer-wise sparsity)
        layer_sparsity_ratios: List of per-layer sparsity ratios (optional)
        eps: Small epsilon for numerical stability (default 1e-5)

    Returns:
        block_mask: Boolean tensor of shape (B, nh, q_blocks, k_blocks)
                   True = block is selected for attention
    """
    import math

    import time
    t_start = time.perf_counter()

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device

    # Step 1: Generate radial spatial constraint mask
    # This mask defines which blocks are spatially allowed by radial attention
    t0 = time.perf_counter()
    radial_mask = _get_cached_radial_block_mask(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        decay_factor=decay_factor,
        dense_neighbor=dense_neighbor,
        device=device
    )  # Shape: (q_blocks, k_blocks)
    t1 = time.perf_counter()
    # print(f"[PERF] Step 1 (radial mask): {(t1-t0)*1000:.2f}ms")

    # Verify shape matches pooled_score
    assert radial_mask.shape == (q_blocks, k_blocks), \
        f"Radial mask shape {radial_mask.shape} doesn't match pooled_score {(q_blocks, k_blocks)}"

    # Validate layer_idx
    if layer_idx < 0:
        raise ValueError(f"layer_idx must be non-negative, got {layer_idx}")

    # Step 2: Get layer-specific sparsity ratio
    if layer_sparsity_ratios is not None and layer_idx < len(layer_sparsity_ratios):
        layer_sparse_ratio = layer_sparsity_ratios[layer_idx]
    else:
        layer_sparse_ratio = sparse_ratio

    # Validate parameters
    if layer_sparse_ratio is not None:
        if not (0.0 <= layer_sparse_ratio <= 1.0):
            raise ValueError(f"sparse_ratio must be in [0, 1], got {layer_sparse_ratio}")

    if cdf_threshold is not None:
        if not (0.0 <= cdf_threshold <= 1.0):
            raise ValueError(f"cdf_threshold must be in [0, 1], got {cdf_threshold}")

    # Check that at least one selection mode is specified
    if layer_sparse_ratio is None and cdf_threshold is None:
        raise ValueError("Either sparse_ratio or cdf_threshold must be specified")

    # Step 3: Select top-k blocks within radial-allowed region
    # Optimized vectorized implementation - avoid repeated tensor allocations

    # Expand radial mask for broadcasting (view, not copy)
    radial_mask_expanded = radial_mask.unsqueeze(0).unsqueeze(0).expand(B, nh, -1, -1)

    # Count radial-allowed blocks per query (used by both modes)
    num_allowed_per_q = radial_mask.sum(dim=-1)  # (q_blocks,)

    # Pre-allocate constant to avoid repeated tensor creation
    neg_inf = float('-inf')

    if cdf_threshold is not None:
        # CDF mode: renormalize within radial-allowed region per query
        # Use masked_fill to handle NaNs properly (multiplication doesn't suppress NaN * 0)
        # Shape: (B, nh, q_blocks, k_blocks)
        masked_scores = pooled_score.clone()
        masked_scores.masked_fill_(~radial_mask_expanded, 0.0)

        # Handle NaNs in allowed region (if any)
        masked_scores.nan_to_num_(nan=0.0)

        # Shape: (B, nh, q_blocks, 1)
        scores_sum = masked_scores.sum(dim=-1, keepdim=True)

        # Handle zero-sum case: when sum is 0, we have no importance information
        zero_sum_mask = (scores_sum.squeeze(-1) == 0)  # (B, nh, q_blocks)

        # Normalize scores within radial-allowed region
        # Use safe denominator to avoid divide-by-zero work
        # Shape: (B, nh, q_blocks, 1)
        denom = torch.where(scores_sum > 0, scores_sum, torch.ones_like(scores_sum))
        scores_normalized = masked_scores / denom
        scores_normalized.masked_fill_(zero_sum_mask.unsqueeze(-1), 0.0)

        # Create sorting key: use clone and masked_fill for efficiency
        # Shape: (B, nh, q_blocks, k_blocks)
        sort_key = scores_normalized.clone()
        sort_key.masked_fill_(~radial_mask_expanded, neg_inf)

        # Sort and compute CDF per query - use memory-efficient sort for large tensors
        # Shape: (B, nh, q_blocks, k_blocks)
        if B * nh * q_blocks > 1000:  # Use chunked sort for large batches
            sort_result = _mem_eff_sort(sort_key, chunks=4, dim=1)
            sorted_scores = sort_result.values
            # CRITICAL: _mem_eff_sort returns int32 indices, but torch.gather needs int64
            sorted_indices = sort_result.indices.long()
        else:
            sorted_scores, sorted_indices = torch.sort(sort_key, dim=-1, descending=True)

        # Only compute CDF on the normalized scores (not the sort key with -inf)
        sorted_normalized = torch.gather(scores_normalized, -1, sorted_indices)
        cdf = torch.cumsum(sorted_normalized, dim=-1)

        # Find threshold index per (B, nh, q)
        # Shape: (B, nh, q_blocks)
        cdf_mask = cdf <= (cdf_threshold + eps)
        num_to_keep = cdf_mask.sum(dim=-1)  # (B, nh, q_blocks)

        # Zero-sum case: set num_to_keep to 0 (no importance information)
        num_to_keep = torch.where(zero_sum_mask, torch.zeros_like(num_to_keep), num_to_keep)

        # Apply sparse_ratio as minimum if specified
        if layer_sparse_ratio is not None:
            keep_ratio = 1.0 - layer_sparse_ratio
            min_keep = (num_allowed_per_q.float() * keep_ratio).long()  # (q_blocks,)
            min_keep = min_keep.unsqueeze(0).unsqueeze(0).expand(B, nh, -1)  # (B, nh, q_blocks)
            # Only apply min_keep for non-zero-sum cases
            num_to_keep = torch.where(
                zero_sum_mask,
                torch.zeros_like(num_to_keep),
                torch.clamp(num_to_keep, min=min_keep)
            )

        # CRITICAL: Clamp num_to_keep to num_allowed to prevent selecting forbidden blocks
        # Shape: (B, nh, q_blocks)
        max_keep = num_allowed_per_q.unsqueeze(0).unsqueeze(0).expand(B, nh, -1)
        num_to_keep = torch.minimum(num_to_keep, max_keep)

        # Create block mask efficiently using scatter instead of advanced indexing
        # This avoids materializing large index tensors
        block_mask = torch.zeros((B, nh, q_blocks, k_blocks), dtype=torch.bool, device=device)

        # Create valid_mask for positions < num_to_keep
        # Shape: (B, nh, q_blocks, k_blocks)
        position_idx = torch.arange(k_blocks, device=device).view(1, 1, 1, k_blocks).expand(B, nh, q_blocks, k_blocks)
        valid_mask = position_idx < num_to_keep.unsqueeze(-1)

        # Use scatter to fill block_mask efficiently
        block_mask.scatter_(-1, sorted_indices, valid_mask)

        t2 = time.perf_counter()
        # print(f"[PERF] Step 2 (CDF mode): {(t2-t1)*1000:.2f}ms")

    else:
        # Ratio mode: fixed keep_ratio per query
        t2 = time.perf_counter()
        keep_ratio = 1.0 - layer_sparse_ratio

        # Fast path: if sparse_ratio=0 (no sparsity), return radial mask directly
        if layer_sparse_ratio == 0.0:
            # print(f"[PERF] Step 2 (ratio mode - fast path): {(time.perf_counter()-t1)*1000:.2f}ms")
            # print(f"[PERF] TOTAL get_radial_layerwise_block_mask: {(time.perf_counter()-t_start)*1000:.2f}ms")
            return radial_mask_expanded.clone()

        # Compute num_to_keep based on allowed blocks only
        num_to_keep = (num_allowed_per_q.float() * keep_ratio).long()  # (q_blocks,)
        num_to_keep = torch.minimum(num_to_keep.clamp_min(0), num_allowed_per_q)

        # Expand to (B, nh, q_blocks)
        num_to_keep = num_to_keep.unsqueeze(0).unsqueeze(0).expand(B, nh, -1)

        # Mask out non-radial-allowed blocks - use clone + masked_fill for efficiency
        # Shape: (B, nh, q_blocks, k_blocks)
        masked_scores = pooled_score.clone()
        masked_scores.masked_fill_(~radial_mask_expanded, neg_inf)

        # Use topk instead of full sort - much qk for ratio mode
        # Only need top max(num_to_keep) blocks, not full sort
        # Shape: (B, nh, q_blocks, k_blocks)
        max_k = num_to_keep.max().item()

        # Create block mask
        block_mask = torch.zeros((B, nh, q_blocks, k_blocks), dtype=torch.bool, device=device)

        if max_k > 0 and max_k < k_blocks:
            # topk is qk when k << k_blocks
            _, topk_indices = torch.topk(masked_scores, k=max_k, dim=-1, largest=True, sorted=False)

            # Create valid_mask for positions < num_to_keep
            # Use scatter to avoid large intermediate index tensors
            position_idx = torch.arange(max_k, device=device).view(1, 1, 1, max_k).expand(B, nh, q_blocks, max_k)
            valid_mask = position_idx < num_to_keep.unsqueeze(-1)

            # Use scatter to fill block_mask efficiently
            block_mask.scatter_(-1, topk_indices, valid_mask)
        elif max_k >= k_blocks:
            # Select all allowed blocks (sparse_ratio very low or 0)
            # Just return the radial mask
            block_mask = radial_mask_expanded.clone()

        # print(f"[PERF] Step 2 (ratio mode): {(time.perf_counter()-t2)*1000:.2f}ms")

    # Final safety check: ensure no forbidden blocks are selected
    # Use in-place operation for efficiency
    t3 = time.perf_counter()
    block_mask.logical_and_(radial_mask_expanded)
    t4 = time.perf_counter()

    # print(f"[PERF] Step 3 (final mask): {(t4-t3)*1000:.2f}ms")
    # print(f"[PERF] TOTAL get_radial_layerwise_block_mask: {(t4-t_start)*1000:.2f}ms")

    return block_mask


def _int32_idx(sort_result):
    return SortResult(
        sort_result.values,
        sort_result.indices.to(torch.int32),
    )


def _mem_eff_sort(t, chunks=4, dim=1):
    # Reduces max memory overhead of sorting large numbers of small-ish arrays
    # split along heads (number of heads is usually divisible by 4)
    sorted = [
        _int32_idx(torch.sort(tt, dim=-1, descending=True))
        for tt in torch.chunk(t, chunks, dim=dim)
    ]
    values = torch.cat([s.values for s in sorted], dim=dim)
    indices = torch.cat([s.indices for s in sorted], dim=dim)
    return SortResult(values, indices)


def check_sparse_mode(sparse_ratio, cdf_threshold):
    """Check the valid combinations of sparse_ratio, and cdf_threshold for sparse inference.

    Args:
        sparse_ratio (float | None): choose a ratio of top blocks
        cdf_threshold (float | None): choose blocks that accumulate to a certain threshold

    Four modes (combinations) are allowed:
        1. only specify sparse_ratio
        2. only specify cdf_threshold
        3. specify both sparse_ratio and cdf_threshold
            This means that the cdf threshold and sparse ratio are BOTH reached.
    """
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None

    # Modes
    only_use_ratio = use_ratio and (not use_cdf)
    only_use_cdf = (not use_ratio) and use_cdf
    use_ratio_and_cdf = use_ratio and use_cdf

    assert (
        only_use_ratio + only_use_cdf + use_ratio_and_cdf == 1
    ), f"Current: {sparse_ratio=}, {cdf_threshold=}"


def check_sparse_mode_three_type(topk, sparse_ratio, cdf_threshold):
    """Check the valid combinations of topk, sparse_ratio, and cdf_threshold for sparse inference.

    Args:
        topk (int | None): choose the top-k key blocks for each query block
        sparse_ratio (float | None): choose a ratio of top blocks
        cdf_threshold (float | None): choose blocks that accumulate to a certain threshold

    Four modes (combinations) are allowed:
        1. only specify topk
        2. only specify sparse_ratio
        3. only specify cdf_threshold
        4. specify both sparse_ratio and cdf_threshold
            This means that the cdf threshold and sparse ratio are BOTH reached.
    """
    use_topk = topk is not None
    use_ratio = sparse_ratio is not None
    use_cdf = cdf_threshold is not None

    # Modes
    only_use_topk = use_topk and (not use_ratio) and (not use_cdf)
    only_use_ratio = (not use_topk) and use_ratio and (not use_cdf)
    only_use_cdf = (not use_topk) and (not use_ratio) and use_cdf
    use_ratio_and_cdf = use_ratio and use_cdf and (not use_topk)

    assert (
        only_use_topk + only_use_ratio + only_use_cdf + use_ratio_and_cdf == 1
    ), f"Current: {topk=}, {sparse_ratio=}, {cdf_threshold=}"


def get_block_mask(
    pooled_score: torch.Tensor,
    sink_blocks: int,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Args:
        pooled_score (Tensor): Pooled attention scores after softmax
            Shape: (B, nh, q_blk, k_blk)
            where q_blk and k_blk are the number of query and key blocks.
            nh: number of heads
        sink_blocks (int): number of key blocks (from the beginning) to always be selected

    Returns:
        final_map (Bool Tensor): (B, nh, q_blk, k_blk + sink_blocks)
            True means the block is selected.

    """
    check_sparse_mode_three_type(topk, sparse_ratio, cdf_threshold)

    B, nh, q_blk, k_blk = pooled_score.shape
    assert sink_blocks >= 0 and sink_blocks <= k_blk

    if sparse_ratio is not None:
        # Convert sparse ratio to topk
        assert sparse_ratio >= 0 and sparse_ratio <= 1
        topk = int(k_blk * (1 - sparse_ratio))

    if topk is not None:
        assert topk >= 0 and topk <= k_blk

    if cdf_threshold is not None:
        assert cdf_threshold >= 0 and cdf_threshold <= 1

    # try to avoid large additional memory allocation of torch.sort
    # see also https://github.com/pytorch/pytorch/issues/77049
    if pooled_score.numel() < 2e8:
        sorted_score = torch.sort(pooled_score, dim=-1, descending=True)
    else:
        sorted_score = _mem_eff_sort(pooled_score)

    num_to_select = None
    if cdf_threshold is not None:
        cdf = torch.cumsum(sorted_score.values, dim=-1)
        cdfthreshd = hyperparameter_check(cdf_threshold, nh, pooled_score.device)
        cdfthreshd_ts = cdfthreshd.view(1, nh, 1, 1)
        cdfthreshd_ts = cdfthreshd_ts + eps  # to avoid numerical error in searchsorted
        cdfthreshd_ts = cdfthreshd_ts.expand(B, -1, q_blk, 1).contiguous()
        num_to_select = torch.searchsorted(cdf, cdfthreshd_ts, right=True).squeeze(-1)

    if topk is not None:
        if num_to_select is None:
            num_to_select = torch.full((B, nh, q_blk), topk, device=pooled_score.device)
        else:
            num_to_select = torch.clamp(num_to_select, min=topk)

    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map = fill_block_map_triton(final_map, num_to_select, sorted_score.indices)

    if sink_blocks > 0:
        # Always select special tokens/blocks
        ones_shape = list(final_map.shape)
        ones_shape[-1] = sink_blocks
        trailing_ones = torch.ones(ones_shape, device=final_map.device).bool()
        final_map = torch.cat([final_map, trailing_ones], dim=-1)

    return final_map


def _fixed_ratio_capacity_mask(
    pooled_score: torch.Tensor,
    *,
    sparse_ratio: float,
    sink_blocks: int,
    force_last_patch: bool,
) -> torch.Tensor:
    """Materialize a ratio budget without ranking disposable indices.

    This template is only valid for downstream selectors that consume the
    mask cardinality, not its identity. A forced mixed tail is charged as an
    additional protected block, matching the existing CoSA capacity hint.
    """
    if pooled_score.ndim != 4:
        raise ValueError("ratio capacity requires [B,H,Q,K] scores")
    if not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError("sparse ratio must be in [0, 1]")
    if sink_blocks < 0:
        raise ValueError("sink block count must be non-negative")

    patch_blocks = pooled_score.shape[-1]
    patch_budget = int(patch_blocks * (1.0 - sparse_ratio))
    patch_mask = torch.zeros_like(pooled_score, dtype=torch.bool)
    if patch_budget:
        patch_mask[..., :patch_budget] = True
    if force_last_patch and patch_blocks:
        patch_mask[..., -1] = True
    if not sink_blocks:
        return patch_mask
    sink_mask = torch.ones(
        *patch_mask.shape[:-1],
        sink_blocks,
        dtype=torch.bool,
        device=patch_mask.device,
    )
    return torch.cat([patch_mask, sink_mask], dim=-1)


def _fixed_ratio_capacity_sparsity(
    *,
    patch_blocks: int,
    sink_blocks: int,
    sparse_ratio: float,
    force_last_patch: bool,
) -> float:
    """Return the exact row sparsity of ``_fixed_ratio_capacity_mask``."""
    if patch_blocks < 0 or sink_blocks < 0:
        raise ValueError("block counts must be non-negative")
    total_blocks = patch_blocks + sink_blocks
    if total_blocks <= 0:
        raise ValueError("sparsity accounting requires at least one block")
    patch_budget = int(patch_blocks * (1.0 - sparse_ratio))
    protected_tail = int(
        force_last_patch and patch_blocks > 0 and patch_budget < patch_blocks
    )
    selected_blocks = patch_budget + protected_tail + sink_blocks
    return 1.0 - selected_blocks / total_blocks


def _block_debt_layout(
    *,
    num_frames: int,
    tokens_per_frame: int,
    query_blocks: int,
    key_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cache frame ownership and per-frame key-block tables."""
    cache_key = (
        num_frames,
        tokens_per_frame,
        query_blocks,
        key_blocks,
        str(device),
    )
    cached = _BLOCK_DEBT_LAYOUT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    query_frame = torch.tensor(
        [
            min(
                num_frames - 1,
                (block_id * 128 + 64) // tokens_per_frame,
            )
            for block_id in range(query_blocks)
        ],
        device=device,
        dtype=torch.long,
    )
    key_frame_ids = [
        min(
            num_frames - 1,
            (block_id * 64 + 32) // tokens_per_frame,
        )
        for block_id in range(key_blocks)
    ]
    key_frame = torch.tensor(
        key_frame_ids,
        device=device,
        dtype=torch.long,
    )
    frame_blocks = [
        [
            block_id
            for block_id, frame_id in enumerate(key_frame_ids)
            if frame_id == target_frame
        ]
        for target_frame in range(num_frames)
    ]
    max_blocks_per_frame = max(len(blocks) for blocks in frame_blocks)
    frame_key_table = torch.zeros(
        num_frames,
        max_blocks_per_frame,
        device=device,
        dtype=torch.long,
    )
    frame_key_valid = torch.zeros(
        num_frames,
        max_blocks_per_frame,
        device=device,
        dtype=torch.bool,
    )
    for frame_id, blocks in enumerate(frame_blocks):
        if not blocks:
            continue
        count = len(blocks)
        frame_key_table[frame_id, :count] = torch.tensor(
            blocks,
            device=device,
            dtype=torch.long,
        )
        frame_key_valid[frame_id, :count] = True

    cached = (query_frame, key_frame, frame_key_table, frame_key_valid)
    if len(_BLOCK_DEBT_LAYOUT_CACHE) >= _BLOCK_DEBT_LAYOUT_CACHE_SIZE:
        _BLOCK_DEBT_LAYOUT_CACHE.pop(next(iter(_BLOCK_DEBT_LAYOUT_CACHE)))
    _BLOCK_DEBT_LAYOUT_CACHE[cache_key] = cached
    return cached


def get_block_debt_mask(
    pooled_score: torch.Tensor,
    *,
    sparse_ratio: float,
    num_frames: int,
    tokens_per_frame: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float = 0.75,
    repayment: float = 1.0,
    service_credit_scale: float = 1.0,
    frame_balance_fraction: float = 0.2,
    stats_store: dict | None = None,
) -> torch.Tensor:
    """Allocate a fixed block budget with persistent non-negative debt."""
    if pooled_score.ndim != 4:
        raise ValueError("pooled_score must have shape [B, H, Q, K]")
    if not 0.0 <= sparse_ratio < 1.0:
        raise ValueError("block-debt sparse ratio must be in [0, 1)")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("block-debt momentum must be in [0, 1]")
    if not 0.0 <= repayment <= 1.0:
        raise ValueError("block-debt repayment must be in [0, 1]")
    if service_credit_scale < 0.0:
        raise ValueError("block-debt service credit must be non-negative")
    if not 0.0 <= frame_balance_fraction <= 1.0:
        raise ValueError("block-debt frame balance must be in [0, 1]")

    _, _, query_blocks, key_blocks = pooled_score.shape
    keep_blocks = max(1, int(key_blocks * (1.0 - sparse_ratio)))
    query_frame, key_frame, frame_key_table, frame_key_valid = (
        _block_debt_layout(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            query_blocks=query_blocks,
            key_blocks=key_blocks,
            device=pooled_score.device,
        )
    )

    current_need = pooled_score.float().clamp_min(0.0)
    current_need = current_need / current_need.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    state_key = "block_debt_scheduler"
    previous_debt = (
        None if routing_state is None else routing_state.get(state_key)
    )
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != tuple(current_need.shape)
    ):
        previous_debt = torch.zeros_like(current_need)
    else:
        previous_debt = previous_debt.to(current_need)
    debt = momentum * previous_debt + current_need

    same_frame = (
        query_frame[:, None] == key_frame[None, :]
    ).view(1, 1, query_blocks, key_blocks)
    candidate_score = debt.masked_fill(same_frame, float("-inf"))
    frame_candidates = candidate_score[..., frame_key_table]
    frame_candidates = frame_candidates.masked_fill(
        ~frame_key_valid.view(
            1, 1, 1, num_frames, frame_key_table.shape[-1]
        ),
        float("-inf"),
    )
    frame_best_score, frame_best_slot = frame_candidates.max(dim=-1)
    frame_key_expanded = frame_key_table.view(
        1, 1, 1, num_frames, -1
    ).expand(*frame_candidates.shape)
    frame_best_key = frame_key_expanded.gather(
        -1, frame_best_slot[..., None]
    ).squeeze(-1)
    query_remote_frame = (
        torch.arange(num_frames, device=pooled_score.device)[None, :]
        != query_frame[:, None]
    )
    frame_best_score = frame_best_score.masked_fill(
        ~query_remote_frame.view(1, 1, query_blocks, num_frames),
        float("-inf"),
    )

    structural_count = min(
        keep_blocks, math.ceil(tokens_per_frame / 64)
    )
    remote_budget = max(0, keep_blocks - structural_count)
    balanced_count = min(
        max(0, num_frames - 1),
        round(remote_budget * frame_balance_fraction),
    )
    balanced_mask = torch.zeros_like(pooled_score, dtype=torch.bool)
    if balanced_count > 0:
        balanced_frames = frame_best_score.topk(
            balanced_count, dim=-1
        ).indices
        balanced_keys = frame_best_key.gather(-1, balanced_frames)
        balanced_mask.scatter_(-1, balanced_keys, True)
        balanced_mask &= ~same_frame

    # Lexicographic priority: structural guarantee, frame reserve, then debt.
    priority_span = debt.detach().amax(dim=-1, keepdim=True) + 1.0
    ranked_score = debt + balanced_mask * priority_span
    ranked_score = ranked_score + same_frame * (2.0 * priority_span)
    selected_indices = ranked_score.topk(keep_blocks, dim=-1).indices
    final_map = torch.zeros_like(pooled_score, dtype=torch.bool)
    final_map.scatter_(-1, selected_indices, True)

    selected_need = current_need.gather(-1, selected_indices)
    credit = selected_need / selected_need.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    selected_debt = debt.gather(-1, selected_indices)
    selected_next_debt = (
        selected_debt
        - repayment * service_credit_scale * credit
    ).clamp_min(0.0)
    next_debt = debt.scatter(-1, selected_indices, selected_next_debt)
    if routing_state is not None:
        routing_state[state_key] = next_debt.detach()

    if stats_store is not None:
        stats_store.update({
            "block_debt_enabled": torch.tensor(
                1.0, device=pooled_score.device
            ),
            "block_debt_keep_blocks": torch.tensor(
                float(keep_blocks), device=pooled_score.device
            ),
            "block_debt_frame_reserve_blocks": torch.tensor(
                float(balanced_count), device=pooled_score.device
            ),
            "block_debt_negative_fraction": (
                (next_debt < 0).float().mean().detach()
            ),
            "block_debt_mean": debt.mean().detach(),
            "block_debt_after_service": next_debt.mean().detach(),
        })
    return final_map


def get_projected_qk_debt_mask(
    pooled_score: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    layer_idx: int,
    routing_state: dict | None,
    stats_store: dict | None = None,
    protected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reallocate an upstream QK sparse attention mask with service-unit debt.

    The upstream mask fixes the per-row QK capacity. Proxy mass creates exactly
    that many admission requests, while a selected block receives one service
    unit. Historical debt is admitted into MaxWeight ranking only up to the
    current row capacity; no momentum or dataset-specific coefficient is used.
    """
    if pooled_score.ndim != 4 or base_mask.shape != pooled_score.shape:
        raise ValueError(
            "projected QK debt requires matching [B, H, Q, K] tensors"
        )
    if base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be boolean")
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    if routing_state is None:
        raise ValueError("projected QK debt requires routing_state")

    capacity = base_mask.sum(dim=-1, keepdim=True)
    score = pooled_score.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    score_mass = score.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(score, 1.0 / max(score.shape[-1], 1))
    distribution = torch.where(
        score_mass > 0,
        score / score_mass.clamp_min(1e-8),
        uniform,
    )
    arrival = distribution * capacity.to(distribution)

    state_key = "projected_qk_qk_admission_debt"
    previous = routing_state.get(state_key)
    reset_to_base = (
        layer_idx == 0
        or previous is None
        or tuple(previous.shape) != tuple(score.shape)
    )
    if reset_to_base:
        previous = torch.zeros_like(score)
    else:
        previous = previous.to(score).clamp_min(0.0)

    previous_mass = previous.sum(dim=-1, keepdim=True)
    projection_scale = torch.where(
        previous_mass > 0,
        torch.minimum(
            torch.ones_like(previous_mass),
            capacity.to(previous) / previous_mass.clamp_min(1e-8),
        ),
        torch.zeros_like(previous_mass),
    )
    admitted = previous * projection_scale
    priority = arrival + admitted
    if protected_mask is not None:
        if protected_mask.shape != base_mask.shape:
            raise ValueError("protected mask must match the QK mask")
        priority = priority.masked_fill(
            protected_mask, torch.finfo(priority.dtype).max
        )

    max_capacity = int(capacity.max().item()) if capacity.numel() else 0
    if reset_to_base:
        selected = base_mask.clone()
    else:
        selected = torch.zeros_like(base_mask)
    if max_capacity > 0 and not reset_to_base:
        selected_indices = priority.topk(
            max_capacity, dim=-1, largest=True, sorted=True
        ).indices
        valid = (
            torch.arange(max_capacity, device=score.device)
            .view(1, 1, 1, -1)
            < capacity
        )
        selected.scatter_(-1, selected_indices, valid)

    service = selected.to(score)
    unreflected = previous + arrival - service
    reflection = (-unreflected).clamp_min(0.0)
    next_debt = unreflected.clamp_min(0.0)
    routing_state[state_key] = next_debt.detach()

    if stats_store is not None:
        overlap = (selected & base_mask).sum(dim=-1).float()
        denominator = capacity.squeeze(-1).clamp_min(1).float()
        conservation_error = (
            next_debt - previous - arrival + service - reflection
        ).abs().amax()
        stats_store.update({
            "projected_qk_debt_enabled": torch.tensor(
                1.0, device=score.device
            ),
            "projected_qk_debt_capacity": capacity.float().mean().detach(),
            "projected_qk_debt_arrival": arrival.sum(
                dim=-1
            ).mean().detach(),
            "projected_qk_debt_admitted": admitted.sum(
                dim=-1
            ).mean().detach(),
            "projected_qk_debt_before": previous.mean().detach(),
            "projected_qk_debt_after": next_debt.mean().detach(),
            "projected_qk_debt_projection_scale": projection_scale.mean().detach(),
            "projected_qk_debt_base_overlap": (
                overlap / denominator
            ).mean().detach(),
            "projected_qk_debt_swap_fraction": (
                1.0 - overlap / denominator
            ).mean().detach(),
            "projected_qk_debt_protected_retention": (
                (selected & protected_mask).sum().float()
                / protected_mask.sum().clamp_min(1).float()
                if protected_mask is not None
                else torch.tensor(1.0, device=score.device)
            ).detach(),
            "projected_qk_debt_reflection": reflection.sum(
                dim=-1
            ).mean().detach(),
            "projected_qk_debt_conservation_error": conservation_error.detach(),
        })
    return selected


def _project_capped_proportional_mass(
    weights: torch.Tensor,
    target_mass: torch.Tensor,
    *,
    fill_zero_rows: bool,
    compute_dtype: torch.dtype = torch.float64,
    max_saturated_hint: int | None = None,
) -> torch.Tensor:
    """Return `min(1, lambda * weights)` under a row-wise mass budget."""
    if compute_dtype not in {torch.float32, torch.float64}:
        raise ValueError("projection compute dtype must be float32 or float64")
    original_dtype = weights.dtype
    work_weights = weights.to(dtype=compute_dtype).clamp_min(0.0)
    work_target = target_mass.to(device=weights.device, dtype=compute_dtype)
    max_weight = work_weights.amax(dim=-1, keepdim=True)
    normalized = work_weights / max_weight.clamp_min(
        torch.finfo(compute_dtype).tiny
    )
    if fill_zero_rows:
        normalized = torch.where(
            max_weight > 0, normalized, torch.ones_like(normalized)
        )
    positive = normalized > 0
    feasible_mass = positive.sum(dim=-1, keepdim=True).to(compute_dtype)
    work_target = torch.minimum(work_target, feasible_mass)

    key_count = weights.shape[-1]
    if max_saturated_hint is None:
        sorted_weight = normalized.sort(dim=-1, descending=True).values
        remaining_weight = sorted_weight.flip(-1).cumsum(-1).flip(-1)
    else:
        if max_saturated_hint < 0:
            raise ValueError("max saturated hint must be non-negative")
        # At mass K, fewer than K positive entries can be strictly saturated;
        # only those top weights are needed to solve the exact water filling.
        candidate_count = min(max(max_saturated_hint, 1), key_count)
        sorted_weight = normalized.topk(
            candidate_count, dim=-1, sorted=True
        ).values
        prefix_before = F.pad(
            sorted_weight.cumsum(dim=-1)[..., :-1], (1, 0)
        )
        remaining_weight = normalized.sum(dim=-1, keepdim=True) - prefix_before
        key_count = candidate_count
    saturated = torch.arange(
        key_count, device=weights.device, dtype=compute_dtype
    ).view(*([1] * (weights.ndim - 1)), key_count)
    remaining_mass = work_target - saturated
    candidate_scale = remaining_mass / remaining_weight.clamp_min(
        torch.finfo(compute_dtype).tiny
    )
    previous_weight = torch.cat(
        [sorted_weight[..., :1], sorted_weight[..., :-1]], dim=-1
    )
    valid_scale = (
        (remaining_mass >= 0)
        & ((saturated == 0) | (candidate_scale * previous_weight >= 1.0))
        & (candidate_scale * sorted_weight <= 1.0)
    )
    scale_index = valid_scale.float().argmax(dim=-1, keepdim=True)
    scale = candidate_scale.gather(-1, scale_index)
    projected = (normalized * scale).clamp(min=0.0, max=1.0)
    projected = torch.where(
        work_target >= feasible_mass,
        positive.to(dtype=compute_dtype),
        projected,
    )
    return projected.to(dtype=original_dtype)


def _project_conservative_capped_mass(
    weights: torch.Tensor,
    target_mass: torch.Tensor,
) -> torch.Tensor:
    """Admit one proportional debt share without exceeding the capacity."""
    original_dtype = weights.dtype
    work_weights = weights.float().clamp_min(0.0)
    work_target = target_mass.to(
        device=weights.device, dtype=torch.float32
    ).clamp_min(0.0)
    total = work_weights.sum(dim=-1, keepdim=True)
    scale = work_target / total.clamp_min(torch.finfo(torch.float32).tiny)
    projected = (work_weights * scale).clamp(min=0.0, max=1.0)
    projected = torch.where(total > 0, projected, torch.zeros_like(projected))
    return projected.to(dtype=original_dtype)


def _project_conservative_capped_mass_inplace(
    weights: torch.Tensor,
    target_mass: torch.Tensor,
) -> torch.Tensor:
    """Consume a fresh FP32 weight tensor using the conservative projection."""
    if weights.dtype != torch.float32:
        raise ValueError("in-place conservative projection requires FP32 weights")
    work_target = target_mass.to(
        device=weights.device, dtype=torch.float32
    ).clamp_min(0.0)
    weights.clamp_min_(0.0)
    total = weights.sum(dim=-1, keepdim=True)
    scale = torch.where(
        total > 0,
        work_target / total.clamp_min(torch.finfo(torch.float32).tiny),
        torch.zeros_like(total),
    )
    return weights.mul_(scale).clamp_(min=0.0, max=1.0)


def _lexicographic_bundle_mask(
    bundle_priority: torch.Tensor,
    child_score: torch.Tensor,
    capacity: torch.Tensor,
    *,
    child_to_bundle: torch.Tensor,
    protected_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Compile logical bundle priorities back to an exact physical-tile mask."""
    if child_to_bundle.ndim != 1 or child_to_bundle.numel() != child_score.shape[-1]:
        raise ValueError("child-to-bundle map must identify every physical tile")
    primary = bundle_priority.index_select(-1, child_to_bundle)
    if protected_mask is not None:
        primary = primary.masked_fill(
            protected_mask, torch.finfo(primary.dtype).max
        )

    # Stable two-pass sorting implements exact lexicographic order without an
    # arbitrary epsilon: bundle priority first, child score second.
    child_order = torch.argsort(
        child_score, dim=-1, descending=True, stable=True
    )
    ordered_primary = primary.gather(-1, child_order)
    primary_order = torch.argsort(
        ordered_primary, dim=-1, descending=True, stable=True
    )
    indices = child_order.gather(-1, primary_order)

    max_capacity = int(capacity.max().item()) if capacity.numel() else 0
    selected = torch.zeros_like(child_score, dtype=torch.bool)
    if max_capacity > 0:
        selected_indices = indices[..., :max_capacity]
        valid = (
            torch.arange(max_capacity, device=child_score.device)
            .view(*([1] * (capacity.ndim - 1)), max_capacity)
            < capacity
        )
        selected.scatter_(-1, selected_indices, valid)
    return selected


def observe_qk_bundle_debt(
    pooled_score: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    layer_idx: int,
    routing_state: dict | None,
    stats_store: dict | None = None,
    protected_mask: torch.Tensor | None = None,
    bundle_size: int = 2,
    child_to_bundle: torch.Tensor | None = None,
) -> torch.Tensor:
    """Observe debt over logical bundles while executing the QK mask.

    A logical bundle contains ``bundle_size`` consecutive physical K tiles.
    The default size is derived from QK's 128-token Q tile and 64-token K
    tile. Arrivals and service remain measured in physical-tile units, while
    the persistent queue has one stable identity per logical bundle.
    """
    if pooled_score.ndim != 4 or base_mask.shape != pooled_score.shape:
        raise ValueError(
            "QK bundle observer requires matching [B, H, Q, K] tensors"
        )
    if base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be boolean")
    if routing_state is None:
        raise ValueError("QK bundle observer requires routing_state")
    if bundle_size < 1:
        raise ValueError("bundle_size must be positive")
    if protected_mask is not None and protected_mask.shape != base_mask.shape:
        raise ValueError("protected mask must match the QK mask")

    score = pooled_score.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    capacity = base_mask.sum(dim=-1, keepdim=True)
    key_blocks = score.shape[-1]
    if child_to_bundle is None:
        child_to_bundle = (
            torch.arange(key_blocks, device=score.device) // bundle_size
        )
    else:
        child_to_bundle = child_to_bundle.to(
            device=score.device, dtype=torch.long
        )
    if child_to_bundle.ndim != 1 or child_to_bundle.numel() != key_blocks:
        raise ValueError("child-to-bundle map must identify every physical tile")
    if bool((child_to_bundle < 0).any()):
        raise ValueError("bundle ids must be non-negative")
    bundle_count = int(child_to_bundle.max().item()) + 1 if key_blocks else 0
    if bundle_count < 1:
        raise ValueError("at least one bundle is required")
    expected_bundle_ids = torch.arange(bundle_count, device=score.device)
    if not torch.equal(torch.unique(child_to_bundle, sorted=True), expected_bundle_ids):
        raise ValueError("bundle ids must be contiguous and non-empty")
    bundle_index = child_to_bundle.view(
        *([1] * (score.ndim - 1)), key_blocks
    ).expand_as(score)
    child_count = torch.bincount(
        child_to_bundle, minlength=bundle_count
    ).to(score)

    child_arrival = _project_capped_proportional_mass(
        score, capacity.to(score), fill_zero_rows=True
    )
    bundle_arrival = score.new_zeros(*score.shape[:-1], bundle_count)
    bundle_arrival.scatter_add_(-1, bundle_index, child_arrival)
    bundle_service = score.new_zeros(*score.shape[:-1], bundle_count)
    bundle_service.scatter_add_(-1, bundle_index, base_mask.to(score))

    state_key = "qk_bundle_observer_debt"
    previous = routing_state.get(state_key)
    if (
        layer_idx == 0
        or previous is None
        or tuple(previous.shape) != tuple(bundle_arrival.shape)
    ):
        previous = torch.zeros_like(bundle_arrival)
    else:
        previous = previous.to(bundle_arrival).clamp_min(0.0)

    # Project bundle debt through physical child-service slots. This enforces
    # 0 <= admitted bundle mass <= number of children and total mass <= K.
    debt_per_bundle_child = previous / child_count.view(
        *([1] * (previous.ndim - 1)), bundle_count
    ).clamp_min(1.0)
    debt_slots = debt_per_bundle_child.gather(-1, bundle_index)
    admitted_slots = _project_capped_proportional_mass(
        debt_slots,
        capacity.to(score),
        fill_zero_rows=False,
    )
    admitted = score.new_zeros(*score.shape[:-1], bundle_count)
    admitted.scatter_add_(-1, bundle_index, admitted_slots)

    current_mask = _lexicographic_bundle_mask(
        bundle_arrival,
        score,
        capacity,
        child_to_bundle=child_to_bundle,
        protected_mask=protected_mask,
    )
    debt_mask = _lexicographic_bundle_mask(
        bundle_arrival + admitted,
        score,
        capacity,
        child_to_bundle=child_to_bundle,
        protected_mask=protected_mask,
    )

    unreflected = previous + bundle_arrival - bundle_service
    reflection = (-unreflected).clamp_min(0.0)
    next_debt = unreflected.clamp_min(0.0)
    routing_state[state_key] = next_debt.detach()

    pending_key = "qk_bundle_observer_pending_masks"
    pending = routing_state.get(pending_key)
    denominator = (score * base_mask).sum(dim=-1).clamp_min(1e-8)
    future_base_ratio = score.new_full((), float("nan"))
    future_current_ratio = score.new_full((), float("nan"))
    future_debt_ratio = score.new_full((), float("nan"))
    if isinstance(pending, dict):
        masks = (
            pending.get("base"),
            pending.get("current"),
            pending.get("debt"),
        )
        if all(mask is not None and mask.shape == base_mask.shape for mask in masks):
            future_base_ratio, future_current_ratio, future_debt_ratio = (
                ((score * mask.to(score)).sum(dim=-1) / denominator).mean()
                for mask in masks
            )
    routing_state[pending_key] = {
        "base": base_mask.detach(),
        "current": current_mask.detach(),
        "debt": debt_mask.detach(),
    }

    if stats_store is not None:
        base_denominator = capacity.squeeze(-1).clamp_min(1).float()
        current_overlap = (current_mask & base_mask).sum(dim=-1).float()
        debt_overlap = (debt_mask & base_mask).sum(dim=-1).float()
        debt_current_overlap = (debt_mask & current_mask).sum(dim=-1).float()
        base_score = (score * base_mask).sum(dim=-1).clamp_min(1e-8)
        base_touched_count = torch.zeros(
            *base_mask.shape[:-1], bundle_count,
            dtype=score.dtype,
            device=base_mask.device,
        )
        base_touched_count.scatter_add_(
            -1, bundle_index, base_mask.to(score)
        )
        base_touched = base_touched_count > 0
        base_touched_capacity = (
            base_touched.to(score)
            * child_count.view(*([1] * (score.ndim - 1)), bundle_count)
        ).sum(dim=-1).clamp_min(1.0)
        conservation_error = (
            next_debt - previous - bundle_arrival + bundle_service - reflection
        ).abs().amax()
        stats_store.update({
            "qk_bundle_observer_enabled": torch.tensor(1.0, device=score.device),
            "qk_bundle_observer_size": child_count.float().mean().detach(),
            "qk_bundle_observer_count": torch.tensor(float(bundle_count), device=score.device),
            "qk_bundle_observer_arrival": bundle_arrival.sum(dim=-1).mean().detach(),
            "qk_bundle_observer_service": bundle_service.sum(dim=-1).mean().detach(),
            "qk_bundle_observer_debt_before": previous.mean().detach(),
            "qk_bundle_observer_debt_after": next_debt.mean().detach(),
            "qk_bundle_observer_debt_max": next_debt.max().detach(),
            "qk_bundle_observer_admitted": admitted.sum(dim=-1).mean().detach(),
            "qk_bundle_observer_reflection": reflection.sum(dim=-1).mean().detach(),
            "qk_bundle_observer_conservation_error": conservation_error.detach(),
            "qk_bundle_observer_base_bundle_fill": (
                bundle_service.sum(dim=-1) / base_touched_capacity
            ).mean().detach(),
            "qk_bundle_observer_current_base_overlap": (
                current_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_debt_base_overlap": (
                debt_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_debt_current_overlap": (
                debt_current_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_current_swap_fraction": (
                1.0 - current_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_debt_swap_fraction": (
                1.0 - debt_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_history_swap_fraction": (
                1.0 - debt_current_overlap / base_denominator
            ).mean().detach(),
            "qk_bundle_observer_current_score_retention": (
                (score * current_mask).sum(dim=-1) / base_score
            ).mean().detach(),
            "qk_bundle_observer_debt_score_retention": (
                (score * debt_mask).sum(dim=-1) / base_score
            ).mean().detach(),
            "qk_bundle_observer_future_base_score_ratio": future_base_ratio.detach(),
            "qk_bundle_observer_future_current_score_ratio": future_current_ratio.detach(),
            "qk_bundle_observer_future_debt_score_ratio": future_debt_ratio.detach(),
            "qk_bundle_observer_budget_error": (
                (debt_mask.sum(dim=-1, keepdim=True) - capacity).abs().max().float()
            ).detach(),
            "qk_bundle_observer_protected_retention": (
                (debt_mask & protected_mask).sum().float()
                / protected_mask.sum().clamp_min(1).float()
                if protected_mask is not None
                else torch.tensor(1.0, device=score.device)
            ).detach(),
        })
    return base_mask


def _frame_reduce_block_map(
    matrix: torch.Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
) -> torch.Tensor:
    """Average a [B,H,Q,K] block map into frame-to-frame coordinates."""
    if matrix.ndim != 4:
        raise ValueError("attention-map matrix must have shape [B,H,Q,K]")
    if num_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError("frame reduction requires positive frame metadata")

    block_map = matrix.detach().float().mean(dim=(0, 1))
    q_blocks, k_blocks = block_map.shape
    device = block_map.device
    q_frame = (
        torch.arange(q_blocks, device=device) * q_block_size
        + q_block_size // 2
    ).div(tokens_per_frame, rounding_mode="floor").clamp_max(num_frames - 1)
    k_frame = (
        torch.arange(k_blocks, device=device) * k_block_size
        + k_block_size // 2
    ).div(tokens_per_frame, rounding_mode="floor").clamp_max(num_frames - 1)
    frame_pair = q_frame[:, None] * num_frames + k_frame[None, :]

    reduced = block_map.new_zeros(num_frames * num_frames)
    counts = block_map.new_zeros(num_frames * num_frames)
    reduced.scatter_add_(0, frame_pair.flatten(), block_map.flatten())
    counts.scatter_add_(0, frame_pair.flatten(), torch.ones_like(block_map).flatten())
    return (reduced / counts.clamp_min(1.0)).view(num_frames, num_frames).cpu()


def _frame_reduce_key_map(
    vector: torch.Tensor,
    *,
    num_frames: int,
    tokens_per_frame: int,
    k_block_size: int = 64,
) -> torch.Tensor:
    """Average a [B,H,K] key-block vector into frame coordinates."""
    if vector.ndim != 3:
        raise ValueError("attention-map vector must have shape [B,H,K]")
    key_map = vector.detach().float().mean(dim=(0, 1))
    device = key_map.device
    k_frame = (
        torch.arange(key_map.numel(), device=device) * k_block_size
        + k_block_size // 2
    ).div(tokens_per_frame, rounding_mode="floor").clamp_max(num_frames - 1)
    reduced = key_map.new_zeros(num_frames)
    counts = key_map.new_zeros(num_frames)
    reduced.scatter_add_(0, k_frame, key_map)
    counts.scatter_add_(0, k_frame, torch.ones_like(key_map))
    return (reduced / counts.clamp_min(1.0)).cpu()


def _save_qk_value_risk_attention_snapshot(
    *,
    output_dir: str,
    layer_idx: int,
    num_frames: int,
    tokens_per_frame: int,
    score: torch.Tensor,
    innovation: torch.Tensor,
    risk: torch.Tensor,
    arrival: torch.Tensor,
    admitted: torch.Tensor,
    previous_debt: torch.Tensor,
    next_debt: torch.Tensor,
    pair_ids: torch.Tensor,
    base_mask: torch.Tensor,
    current_mask: torch.Tensor,
    debt_mask: torch.Tensor,
    protected_mask: torch.Tensor | None,
    execution_mode: str,
    risk_definition: str,
) -> None:
    """Persist compact frame-level evidence for a paper attention-map figure."""
    q_blocks = score.shape[-2]
    previous_child = previous_debt.index_select(-1, pair_ids).squeeze(-2)
    next_child = next_debt.index_select(-1, pair_ids).squeeze(-2)
    innovation_map = innovation.unsqueeze(-2).expand(-1, -1, q_blocks, -1)
    previous_map = previous_child.unsqueeze(-2).expand(-1, -1, q_blocks, -1)
    admitted_map = admitted.expand(-1, -1, q_blocks, -1)
    next_map = next_child.unsqueeze(-2).expand(-1, -1, q_blocks, -1)
    protected = (
        torch.zeros_like(base_mask)
        if protected_mask is None
        else protected_mask
    )

    matrices = {
        "qk_probability": score,
        "value_innovation": innovation_map,
        "current_risk": risk,
        "projected_arrival": arrival,
        "previous_debt": previous_map,
        "admitted_debt": admitted_map,
        "current_priority": arrival,
        "debt_priority": arrival + admitted,
        "next_debt": next_map,
        "qk_support": base_mask,
        "current_support": current_mask,
        "debt_support": debt_mask,
        "promoted_support": debt_mask & ~current_mask,
        "evicted_support": current_mask & ~debt_mask,
        "protected_support": protected,
    }
    frame_maps = {
        name: _frame_reduce_block_map(
            value,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
        )
        for name, value in matrices.items()
    }
    frame_maps["support_delta"] = (
        frame_maps["promoted_support"] - frame_maps["evicted_support"]
    )

    snapshot = {
        "schema_version": 1,
        "layer": int(layer_idx),
        "num_frames": int(num_frames),
        "tokens_per_frame": int(tokens_per_frame),
        "q_block_size": 128,
        "k_block_size": 64,
        "execution_mode": execution_mode,
        "risk_definition": risk_definition,
        "source_shape": tuple(int(size) for size in score.shape),
        "capacity_mean": float(base_mask.sum(dim=-1).float().mean().item()),
        "current_debt_swap_fraction": float(
            (debt_mask & ~current_mask).sum().float()
            .div(current_mask.sum().clamp_min(1).float()).item()
        ),
        "qk_current_swap_fraction": float(
            (current_mask & ~base_mask).sum().float()
            .div(base_mask.sum().clamp_min(1).float()).item()
        ),
        "qk_debt_swap_fraction": float(
            (debt_mask & ~base_mask).sum().float()
            .div(base_mask.sum().clamp_min(1).float()).item()
        ),
        "frame_maps": frame_maps,
        "key_frame_vectors": {
            "value_innovation": _frame_reduce_key_map(
                innovation,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
            ),
            "previous_debt": _frame_reduce_key_map(
                previous_child,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
            ),
            "next_debt": _frame_reduce_key_map(
                next_child,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
            ),
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"layer_{layer_idx:02d}.pt")
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    torch.save(snapshot, temporary_path)
    os.replace(temporary_path, output_path)


def observe_qk_value_risk_debt(
    pooled_score: torch.Tensor,
    pooled_value: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    layer_idx: int,
    routing_state: dict | None,
    stats_store: dict | None = None,
    protected_mask: torch.Tensor | None = None,
    execution_mode: str = "observe",
    risk_definition: str = "innovation",
    production_fast_path: bool = False,
    attention_map_dir: str | None = None,
    attention_map_layers: set[int] | None = None,
    num_frames: int | None = None,
    tokens_per_frame: int | None = None,
) -> torch.Tensor:
    """Measure value-risk debt and optionally execute an equal-budget support."""
    if pooled_score.ndim != 4 or base_mask.shape != pooled_score.shape:
        raise ValueError("value-risk observer requires [B, H, Q, K] scores")
    if pooled_value.ndim != 4:
        raise ValueError("pooled value must have shape [B, H, K, D]")
    if pooled_value.shape[:2] != pooled_score.shape[:2]:
        raise ValueError("pooled value batch/head dimensions must match scores")
    if pooled_value.shape[-2] != pooled_score.shape[-1]:
        raise ValueError("pooled value must identify every physical K tile")
    if base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be boolean")
    if routing_state is None:
        raise ValueError("value-risk observer requires routing_state")
    if protected_mask is not None and protected_mask.shape != base_mask.shape:
        raise ValueError("protected mask must match the QK mask")
    if execution_mode not in {"observe", "current", "debt"}:
        raise ValueError("value-risk execution mode must be observe, current, or debt")
    if risk_definition not in {"innovation", "pair_contribution"}:
        raise ValueError("risk definition must be innovation or pair_contribution")
    if production_fast_path and execution_mode == "observe":
        raise ValueError("production fast path requires current or debt execution")
    if attention_map_dir and production_fast_path:
        raise ValueError("attention-map export requires the diagnostic path")
    if attention_map_dir and (num_frames is None or tokens_per_frame is None):
        raise ValueError("attention-map export requires frame metadata")

    score = pooled_score.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    native_innovation = os.environ.get(
        "SPARSE_VGGT_COSA_NATIVE_INNOVATION", "0"
    ).lower() in {"1", "true", "yes", "on"}
    value = pooled_value.nan_to_num(nan=0.0)
    if not native_innovation:
        value = value.float()
    key_blocks = score.shape[-1]
    capacity = base_mask.sum(dim=-1, keepdim=True)

    # A local two-tile carrier exposes value detail not represented by a
    # neighboring aggregate. This is a diagnostic identity, not a tunable
    # bundle size used by execution.
    pair_ids = torch.arange(key_blocks, device=score.device) // 2
    pair_count = (key_blocks + 1) // 2
    pair_index = pair_ids.view(1, 1, key_blocks, 1).expand_as(value)
    pair_sum = value.new_zeros(*value.shape[:-2], pair_count, value.shape[-1])
    pair_sum.scatter_add_(-2, pair_index, value)
    pair_size = torch.bincount(pair_ids, minlength=pair_count).to(value)
    carrier = pair_sum / pair_size.view(1, 1, pair_count, 1).clamp_min(1.0)
    carrier_per_tile = carrier.index_select(-2, pair_ids)
    innovation = (value - carrier_per_tile).square().mean(dim=-1).sqrt()
    relative_innovation = None
    if not production_fast_path:
        relative_innovation = (
            innovation / value.square().mean(dim=-1).sqrt().clamp_min(1e-8)
        )
    child_contribution = None
    if risk_definition == "innovation":
        # For a carrier-replaced tile, p(q,j) * ||V_j - V_c(j)|| is the
        # first-order output-residual contribution.
        risk_weight = score * innovation.unsqueeze(-2)
    else:
        # Compute ||p_0 V_0 + p_1 V_1|| without materializing Q x K x D.
        # The quadratic identity is exact and keeps memory at Q x pair_count.
        even_score = score[..., 0::2]
        odd_score = score[..., 1::2]
        even_value = value[..., 0::2, :]
        odd_value = value[..., 1::2, :]
        if odd_score.shape[-1] < pair_count:
            odd_score = torch.cat(
                (odd_score, odd_score.new_zeros(*odd_score.shape[:-1], 1)), dim=-1
            )
            odd_value = torch.cat(
                (
                    odd_value,
                    odd_value.new_zeros(
                        *odd_value.shape[:-2], 1, odd_value.shape[-1]
                    ),
                ),
                dim=-2,
            )
        even_energy = even_value.square().mean(dim=-1).unsqueeze(-2)
        odd_energy = odd_value.square().mean(dim=-1).unsqueeze(-2)
        cross_energy = (even_value * odd_value).mean(dim=-1).unsqueeze(-2)
        pair_contribution = (
            even_score.square() * even_energy
            + odd_score.square() * odd_energy
            + 2.0 * even_score * odd_score * cross_energy
        ).clamp_min_(0.0).sqrt_()
        child_contribution = score * value.square().mean(dim=-1).sqrt().unsqueeze(-2)
        # Repeat pair risk over its physical service slots; the capped
        # projection below still enforces the exact QK child-tile budget.
        risk_weight = pair_contribution.index_select(-1, pair_ids)
    row_arrival = _project_capped_proportional_mass(
        risk_weight, capacity.to(risk_weight), fill_zero_rows=True
    )
    # Query layouts can change across VGGT layers. A persistent liability must
    # therefore live on stable 128-token pair identities. Pair occupancy is
    # measured as the mean service of its physical children, then averaged over
    # current query rows.
    row_pair_arrival = row_arrival.new_zeros(*row_arrival.shape[:-1], pair_count)
    row_pair_arrival.scatter_add_(
        -1,
        pair_ids.view(1, 1, 1, key_blocks).expand_as(row_arrival),
        row_arrival,
    )
    row_pair_arrival = row_pair_arrival / pair_size.view(1, 1, 1, pair_count)

    def select(priority: torch.Tensor) -> torch.Tensor:
        ranked = priority
        if protected_mask is not None:
            ranked = ranked.masked_fill(
                protected_mask, torch.finfo(ranked.dtype).max
            )
        max_capacity = int(capacity.max().item()) if capacity.numel() else 0
        selected = torch.zeros_like(base_mask)
        if max_capacity:
            indices = ranked.topk(max_capacity, dim=-1).indices
            valid = (
                torch.arange(max_capacity, device=score.device)
                .view(1, 1, 1, -1)
                < capacity
            )
            selected.scatter_(-1, indices, valid)
        return selected

    if production_fast_path and execution_mode == "current":
        if risk_definition == "pair_contribution":
            return _lexicographic_bundle_mask(
                row_pair_arrival,
                child_contribution,
                capacity,
                child_to_bundle=pair_ids,
                protected_mask=protected_mask,
            )
        return select(row_arrival)

    arrival = row_pair_arrival.mean(dim=-2, keepdim=True)
    state_capacity = row_pair_arrival.sum(dim=-1, keepdim=True).mean(
        dim=-2, keepdim=True
    )

    ledger_key = "qk_value_risk_observer_ledger"
    if layer_idx == 0:
        routing_state[ledger_key] = {}
    ledger = routing_state.setdefault(ledger_key, {})
    state_key = "debt"
    checksum_key = "qk_value_risk_observer_debt_checksum"
    payload = ledger.get(state_key)
    if production_fast_path:
        stored_checksum = 0.0
        state_present = torch.is_tensor(payload)
        state_shape_match = (
            state_present and tuple(payload.shape) == tuple(arrival.shape)
        )
    else:
        stored_checksum = float(ledger.get(checksum_key, 0.0))
        state_present = isinstance(payload, dict) and "values" in payload
        state_shape_match = (
            state_present and tuple(payload.get("shape", ())) == tuple(arrival.shape)
        )
    if (
        layer_idx == 0
        or not state_shape_match
    ):
        previous = torch.zeros_like(arrival)
    elif production_fast_path:
        previous = payload.detach().to(
            device=arrival.device, dtype=arrival.dtype
        ).view_as(arrival).clamp_min(0.0)
    else:
        previous = torch.tensor(
            payload["values"], device=arrival.device, dtype=arrival.dtype
        ).view_as(arrival).clamp_min_(0.0)
    admitted_pair = _project_capped_proportional_mass(
        previous, state_capacity.to(previous), fill_zero_rows=False
    )
    admitted = admitted_pair.index_select(-1, pair_ids)

    if risk_definition == "pair_contribution":
        current_mask = None
        if not production_fast_path:
            current_mask = _lexicographic_bundle_mask(
                row_pair_arrival,
                child_contribution,
                capacity,
                child_to_bundle=pair_ids,
                protected_mask=protected_mask,
            )
        debt_mask = _lexicographic_bundle_mask(
            row_pair_arrival + admitted_pair,
            child_contribution,
            capacity,
            child_to_bundle=pair_ids,
            protected_mask=protected_mask,
        )
    else:
        current_mask = None if production_fast_path else select(row_arrival)
        debt_mask = select(row_arrival + admitted)
    if production_fast_path:
        execution_mask = debt_mask
    else:
        execution_mask = {
            "observe": base_mask,
            "current": current_mask,
            "debt": debt_mask,
        }[execution_mode]
    row_pair_service = row_pair_arrival.new_zeros(
        *row_pair_arrival.shape[:-1], pair_count
    )
    row_pair_service.scatter_add_(
        -1,
        pair_ids.view(1, 1, 1, key_blocks).expand_as(row_arrival),
        execution_mask.to(row_arrival),
    )
    row_pair_service = row_pair_service / pair_size.view(1, 1, 1, pair_count)
    service = row_pair_service.mean(dim=-2, keepdim=True)
    unreflected = previous + arrival - service
    reflection = (-unreflected).clamp_min(0.0)
    next_debt = unreflected.clamp_min(0.0)
    if production_fast_path:
        ledger[state_key] = next_debt.detach().clone()
    else:
        # The diagnostic path serializes state so custom CUDA buffers cannot
        # mutate observer evidence after a layer has completed.
        ledger[state_key] = {
            "shape": tuple(next_debt.shape),
            "values": next_debt.detach().float().cpu().flatten().tolist(),
        }
        ledger[checksum_key] = float(next_debt.mean().item())

    if (
        attention_map_dir
        and layer_idx in (attention_map_layers or {0, 7, 15, 23})
    ):
        _save_qk_value_risk_attention_snapshot(
            output_dir=attention_map_dir,
            layer_idx=layer_idx,
            num_frames=int(num_frames),
            tokens_per_frame=int(tokens_per_frame),
            score=score,
            innovation=innovation,
            risk=risk_weight,
            arrival=row_arrival,
            admitted=admitted,
            previous_debt=previous,
            next_debt=next_debt,
            pair_ids=pair_ids,
            base_mask=base_mask,
            current_mask=current_mask,
            debt_mask=debt_mask,
            protected_mask=protected_mask,
            execution_mode=execution_mode,
            risk_definition=risk_definition,
        )

    future_ratios = [score.new_full((), float("nan")) for _ in range(3)]
    if not production_fast_path:
        pending_key = "pending_masks"
        pending = ledger.get(pending_key)
        risk_denominator = (risk_weight * base_mask).sum(dim=-1).clamp_min(1e-8)
        if isinstance(pending, dict):
            masks = [pending.get(name) for name in ("base", "current", "debt")]
            if all(mask is not None and mask.shape == base_mask.shape for mask in masks):
                masks = [mask.to(device=base_mask.device) for mask in masks]
                future_ratios = [
                    (
                        (risk_weight * mask.to(risk_weight)).sum(dim=-1)
                        / risk_denominator
                    ).mean()
                    for mask in masks
                ]
        ledger[pending_key] = {
            "base": base_mask.detach().to("cpu", copy=True),
            "current": current_mask.detach().to("cpu", copy=True),
            "debt": debt_mask.detach().to("cpu", copy=True),
        }

    if stats_store is not None and not production_fast_path:
        def scalar(tensor: torch.Tensor) -> float:
            return float(tensor.detach().float().cpu().item())

        denominator = capacity.squeeze(-1).clamp_min(1).float()
        current_base = (current_mask & base_mask).sum(dim=-1).float()
        debt_base = (debt_mask & base_mask).sum(dim=-1).float()
        debt_current = (debt_mask & current_mask).sum(dim=-1).float()
        base_risk = (risk_weight * base_mask).sum(dim=-1).clamp_min(1e-8)
        conservation_error = (
            next_debt - previous - arrival + service - reflection
        ).abs().amax()
        stats_store.update({
            "qk_value_risk_observer_enabled": 1.0,
            "qk_value_risk_observer_execution_mode": float(
                {"observe": 0, "current": 1, "debt": 2}[execution_mode]
            ),
            "qk_value_risk_observer_definition": float(
                {"innovation": 0, "pair_contribution": 1}[risk_definition]
            ),
            "qk_value_risk_observer_state_present": float(state_present),
            "qk_value_risk_observer_state_shape_match": float(state_shape_match),
            "qk_value_risk_observer_stored_checksum": stored_checksum,
            "qk_value_risk_observer_key_blocks": float(key_blocks),
            "qk_value_risk_observer_pair_blocks": float(pair_count),
            "qk_value_risk_observer_query_rows": float(score.shape[-2]),
            "qk_value_risk_observer_innovation_mean": scalar(innovation.mean()),
            "qk_value_risk_observer_innovation_max": scalar(innovation.max()),
            "qk_value_risk_observer_relative_innovation_mean": scalar(relative_innovation.mean()),
            "qk_value_risk_observer_relative_innovation_max": scalar(relative_innovation.max()),
            "qk_value_risk_observer_arrival": scalar(arrival.sum(dim=-1).mean()),
            "qk_value_risk_observer_debt_before": scalar(previous.mean()),
            "qk_value_risk_observer_debt_after": scalar(next_debt.mean()),
            "qk_value_risk_observer_debt_max": scalar(next_debt.max()),
            "qk_value_risk_observer_admitted": scalar(admitted_pair.sum(dim=-1).mean()),
            "qk_value_risk_observer_reflection": scalar(reflection.sum(dim=-1).mean()),
            "qk_value_risk_observer_conservation_error": scalar(conservation_error),
            "qk_value_risk_observer_current_base_overlap": scalar((current_base / denominator).mean()),
            "qk_value_risk_observer_debt_base_overlap": scalar((debt_base / denominator).mean()),
            "qk_value_risk_observer_debt_current_overlap": scalar((debt_current / denominator).mean()),
            "qk_value_risk_observer_current_swap_fraction": scalar((1.0 - current_base / denominator).mean()),
            "qk_value_risk_observer_debt_swap_fraction": scalar((1.0 - debt_base / denominator).mean()),
            "qk_value_risk_observer_history_swap_fraction": scalar((1.0 - debt_current / denominator).mean()),
            "qk_value_risk_observer_current_risk_retention": scalar(((risk_weight * current_mask).sum(dim=-1) / base_risk).mean()),
            "qk_value_risk_observer_debt_risk_retention": scalar(((risk_weight * debt_mask).sum(dim=-1) / base_risk).mean()),
            "qk_value_risk_observer_future_base_risk_ratio": scalar(future_ratios[0]),
            "qk_value_risk_observer_future_current_risk_ratio": scalar(future_ratios[1]),
            "qk_value_risk_observer_future_debt_risk_ratio": scalar(future_ratios[2]),
            "qk_value_risk_observer_budget_error": scalar((debt_mask.sum(dim=-1, keepdim=True) - capacity).abs().max().float()),
            "qk_value_risk_observer_protected_retention": (
                (debt_mask & protected_mask).sum().float()
                / protected_mask.sum().clamp_min(1).float()
                if protected_mask is not None
                else torch.tensor(1.0, device=score.device)
            ).detach().float().cpu().item(),
        })
    return execution_mask


def get_projected_qk_pv_debt_mask(
    pooled_score: torch.Tensor,
    base_mask: torch.Tensor,
    *,
    layer_idx: int,
    routing_state: dict | None,
    stats_store: dict | None = None,
    protected_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate the QK support budget using exact unpaid PV service.

    The queue is updated one layer late because online-softmax service is only
    known after the current mask has executed. Each selected key block offers
    one service unit; fractional service is the mean execution indicator over
    the query warps in the corresponding Q block. Incoming queue mass is
    projected onto the current support capacity before MaxWeight selection.
    """
    if pooled_score.ndim != 4 or base_mask.shape != pooled_score.shape:
        raise ValueError(
            "projected QK PV debt requires matching [B, H, Q, K] tensors"
        )
    if base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be boolean")
    if routing_state is None:
        raise ValueError("projected QK PV debt requires routing_state")

    capacity = base_mask.sum(dim=-1, keepdim=True)
    score = pooled_score.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    # Map arbitrary positive QK scores to block-service units. The unique
    # proportional water-filling solution has total mass K and per-block mass
    # at most one, matching the units of exact PV execution service.
    target_mass = capacity.to(dtype=score.dtype)
    arrival = _project_capped_proportional_mass(
        score, target_mass, fill_zero_rows=True
    )

    state_key = "projected_qk_pv_debt"
    pending_key = "projected_qk_pv_pending_arrival"
    pending_layer_key = "projected_qk_pv_pending_layer"
    previous = routing_state.get(state_key)
    if previous is None or tuple(previous.shape) != tuple(score.shape):
        previous = torch.zeros_like(score)
    else:
        previous = previous.to(device=score.device, dtype=score.dtype)

    service_observation = routing_state.pop(
        "qk_pv_service_observation", None
    )
    pending_arrival = routing_state.get(pending_key)
    consumed_service = torch.zeros_like(score)
    previous_before_service = previous
    settled_arrival = torch.zeros_like(score)
    reflection = torch.zeros_like(score)
    if pending_arrival is not None:
        if service_observation is None or not service_observation.get(
            "identity_observed", False
        ):
            raise RuntimeError(
                "PV-service debt requires an exact per-key execution map"
            )
        pending_layer = routing_state.get(pending_layer_key)
        observed_layer = service_observation.get("layer_idx")
        if pending_layer is not None and observed_layer != pending_layer:
            raise RuntimeError(
                "PV service observation does not match the pending layer"
            )
        consumed_service = service_observation["service_fraction"].to(
            device=score.device, dtype=score.dtype
        )
        if tuple(consumed_service.shape) != tuple(score.shape):
            raise RuntimeError("PV service map changed routing shape")
        settled_arrival = pending_arrival.to(
            device=score.device, dtype=score.dtype
        )
        unreflected = previous_before_service + settled_arrival - consumed_service
        reflection = (-unreflected).clamp_min(0.0)
        previous = unreflected.clamp_min(0.0)

    debt_mass = previous.sum(dim=-1, keepdim=True)
    admitted = _project_capped_proportional_mass(
        previous, capacity.to(dtype=previous.dtype), fill_zero_rows=False
    )
    admission_scale = admitted.sum(dim=-1, keepdim=True) / debt_mass.clamp_min(
        1e-8
    )
    priority = arrival + admitted
    if protected_mask is not None:
        if protected_mask.shape != base_mask.shape:
            raise ValueError("protected mask must match the QK mask")
        priority = priority.masked_fill(
            protected_mask, torch.finfo(priority.dtype).max
        )

    reset_to_base = layer_idx == 0
    max_capacity = int(capacity.max().item()) if capacity.numel() else 0
    selected = torch.zeros_like(base_mask)
    if reset_to_base:
        selected.copy_(base_mask)
    elif max_capacity > 0:
        indices = priority.topk(
            max_capacity,
            dim=-1,
            sorted=True,
        ).indices
        valid = (
            torch.arange(max_capacity, device=score.device)
            .view(1, 1, 1, -1)
            < capacity
        )
        selected.scatter_(-1, indices, valid)

    routing_state[state_key] = previous.detach()
    routing_state[pending_key] = arrival.detach()
    routing_state[pending_layer_key] = int(layer_idx)
    if stats_store is not None:
        budget_error = (
            capacity.float() - selected.sum(dim=-1, keepdim=True).float()
        ).abs().max()
        queue_conservation_error = (
            previous
            - previous_before_service
            - settled_arrival
            + consumed_service
            - reflection
        ).abs().max()
        debt_available = previous_before_service + settled_arrival
        service_reduction = torch.minimum(
            debt_available, consumed_service
        ).sum(dim=-1).mean()
        overlap = (selected & base_mask).sum(dim=-1).float()
        denominator = capacity.squeeze(-1).clamp_min(1).float()
        stats_store.update({
            "projected_qk_pv_debt_enabled": torch.tensor(
                1.0, device=score.device
            ),
            "projected_qk_pv_debt_mean": previous.mean().detach(),
            "projected_qk_pv_debt_max": previous.max().detach(),
            "projected_qk_pv_debt_before_service": (
                previous_before_service.mean().detach()
            ),
            "projected_qk_pv_settled_arrival": (
                settled_arrival.sum(dim=-1).mean().detach()
            ),
            "projected_qk_pv_arrival_max": arrival.max().detach(),
            "projected_qk_pv_arrival_budget_error": (
                arrival.sum(dim=-1, keepdim=True) - target_mass
            ).abs().max().detach(),
            "projected_qk_pv_reflection": (
                reflection.sum(dim=-1).mean().detach()
            ),
            "projected_qk_pv_service_reduction": (
                service_reduction.detach()
            ),
            "projected_qk_pv_debt_admission_scale": (
                admission_scale.mean().detach()
            ),
            "projected_qk_pv_debt_admitted": admitted.sum(
                dim=-1
            ).mean().detach(),
            "projected_qk_pv_debt_admitted_max": admitted.max().detach(),
            "projected_qk_pv_service_fraction": (
                consumed_service.sum() / capacity.float().sum().clamp_min(1.0)
            ).detach(),
            "projected_qk_pv_base_overlap": (
                overlap / denominator
            ).mean().detach(),
            "projected_qk_pv_swap_fraction": (
                1.0 - overlap / denominator
            ).mean().detach(),
            "projected_qk_pv_protected_retention": (
                (selected & protected_mask).sum().float()
                / protected_mask.sum().clamp_min(1).float()
                if protected_mask is not None
                else torch.tensor(1.0, device=score.device)
            ).detach(),
            "projected_qk_pv_budget_error": budget_error.detach(),
            "projected_qk_pv_queue_conservation_error": (
                queue_conservation_error.detach()
            ),
        })
    return selected


def finalize_projected_qk_pv_debt(
    routing_state: dict | None,
    *,
    stats_store: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Settle the final layer's exact PV service and close the queue ledger."""
    if routing_state is None:
        return {}

    state_key = "projected_qk_pv_debt"
    pending_key = "projected_qk_pv_pending_arrival"
    pending_layer_key = "projected_qk_pv_pending_layer"
    service_key = "qk_pv_service_observation"
    pending_arrival = routing_state.pop(pending_key, None)
    pending_layer = routing_state.pop(pending_layer_key, None)
    if pending_arrival is None:
        # Read-only PV observers also publish service without enabling debt.
        return {}
    service_observation = routing_state.pop(service_key, None)
    if service_observation is None or not service_observation.get(
        "identity_observed", False
    ):
        raise RuntimeError(
            "terminal PV-debt settlement requires an exact execution map"
        )
    if service_observation.get("layer_idx") != pending_layer:
        raise RuntimeError(
            "terminal PV service does not match the pending layer"
        )

    previous = routing_state.get(state_key)
    if previous is None or tuple(previous.shape) != tuple(pending_arrival.shape):
        previous = torch.zeros_like(pending_arrival)
    else:
        previous = previous.to(
            device=pending_arrival.device, dtype=pending_arrival.dtype
        )
    consumed_service = service_observation["service_fraction"].to(
        device=pending_arrival.device, dtype=pending_arrival.dtype
    )
    if tuple(consumed_service.shape) != tuple(pending_arrival.shape):
        raise RuntimeError("terminal PV service map changed routing shape")

    unreflected = previous + pending_arrival - consumed_service
    reflection = (-unreflected).clamp_min(0.0)
    terminal_debt = unreflected.clamp_min(0.0)
    conservation_error = (
        terminal_debt
        - previous
        - pending_arrival
        + consumed_service
        - reflection
    ).abs().max()
    service_reduction = torch.minimum(
        previous + pending_arrival, consumed_service
    ).sum(dim=-1).mean()
    routing_state[state_key] = terminal_debt.detach()

    stats = {
        "projected_qk_pv_terminal_settled": torch.tensor(
            1.0, device=terminal_debt.device
        ),
        "projected_qk_pv_terminal_layer": torch.tensor(
            float(pending_layer), device=terminal_debt.device
        ),
        "projected_qk_pv_terminal_arrival": pending_arrival.sum(
            dim=-1
        ).mean().detach(),
        "projected_qk_pv_terminal_service": consumed_service.sum(
            dim=-1
        ).mean().detach(),
        "projected_qk_pv_terminal_service_reduction": (
            service_reduction.detach()
        ),
        "projected_qk_pv_terminal_reflection": reflection.sum(
            dim=-1
        ).mean().detach(),
        "projected_qk_pv_terminal_backlog": terminal_debt.sum(
            dim=-1
        ).mean().detach(),
        "projected_qk_pv_terminal_backlog_max": terminal_debt.max().detach(),
        "projected_qk_pv_terminal_conservation_error": (
            conservation_error.detach()
        ),
    }
    if stats_store is not None:
        stats_store.update(stats)
    return stats


def _mean_pool_contiguous_value_blocks(
    value: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Mean-pool contiguous token blocks without transposing the value tensor."""
    if value.ndim != 4:
        raise ValueError("value must have shape [B,H,T,D]")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    token_count = value.shape[-2]
    full_blocks, tail_tokens = divmod(token_count, block_size)
    pooled = []
    if full_blocks:
        pooled.append(
            value[..., : full_blocks * block_size, :]
            .reshape(*value.shape[:-2], full_blocks, block_size, value.shape[-1])
            .mean(dim=-2)
        )
    if tail_tokens:
        pooled.append(value[..., full_blocks * block_size :, :].mean(dim=-2, keepdim=True))
    if not pooled:
        return value.new_empty(*value.shape[:-2], 0, value.shape[-1])
    return pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=-2)


def _reduce_pv_service_to_pair(
    service_fraction: torch.Tensor,
    *,
    pair_ids: torch.Tensor,
    pair_size: torch.Tensor,
) -> torch.Tensor:
    """Map exact child-tile PV service to stable pair-service units."""
    if service_fraction.ndim != 4:
        raise ValueError("PV service fraction must have shape [B,H,Q,K]")
    if service_fraction.shape[-1] != pair_ids.numel():
        raise ValueError("PV service key layout does not match pair identities")
    expected_ids = torch.arange(
        pair_ids.numel(), device=pair_ids.device
    ) // 2
    if not torch.equal(pair_ids, expected_ids):
        raise ValueError("pair service requires stable adjacent child identities")
    pair_count = int(pair_size.numel())
    if service_fraction.shape[-1] % 2:
        service_fraction = F.pad(service_fraction, (0, 1))
    row_pair_service = service_fraction.reshape(
        *service_fraction.shape[:-1], pair_count, 2
    ).sum(dim=-1)
    row_pair_service = row_pair_service / pair_size.view(
        1, 1, 1, pair_count
    ).clamp_min(1.0)
    return row_pair_service.mean(dim=-2, keepdim=True)


def _reduce_qk_admission_to_pair(
    support: torch.Tensor,
    *,
    patch_blocks: int,
) -> torch.Tensor:
    """Reduce exact binary QK admission to stable pair-service units."""
    if support.ndim != 4 or support.dtype != torch.bool:
        raise ValueError("QK admission support must be boolean [B,H,Q,K]")
    if patch_blocks <= 0 or patch_blocks > support.shape[-1]:
        raise ValueError("invalid patch block count")
    patch_support = support[..., :patch_blocks]
    if patch_blocks % 2:
        patch_support = F.pad(patch_support, (0, 1))
    pair_count = (patch_blocks + 1) // 2
    pair_service = patch_support.reshape(
        *patch_support.shape[:-1], pair_count, 2
    ).sum(dim=-1, dtype=torch.float32)
    pair_size = torch.full(
        (pair_count,),
        2.0,
        device=support.device,
        dtype=torch.float32,
    )
    if patch_blocks % 2:
        pair_size[-1] = 1.0
    return (
        pair_service / pair_size.view(1, 1, 1, pair_count)
    ).mean(dim=-2, keepdim=True)


def _reduce_qk_indices_to_pair(
    selected_indices: torch.Tensor,
    capacity: torch.Tensor,
    *,
    patch_blocks: int,
) -> torch.Tensor:
    """Reduce exact top-k indices without rescanning the dense support."""
    if selected_indices.ndim != 4 or selected_indices.dtype not in {
        torch.int32,
        torch.int64,
    }:
        raise ValueError("QK indices must be integer [B,H,Q,M]")
    if capacity.shape != (*selected_indices.shape[:-1], 1):
        raise ValueError("QK index capacity does not match the selected rows")
    if patch_blocks <= 0:
        raise ValueError("invalid patch block count")
    if selected_indices.shape[-1] == 0:
        return torch.zeros(
            *selected_indices.shape[:2],
            1,
            (patch_blocks + 1) // 2,
            device=selected_indices.device,
            dtype=torch.float32,
        )
    pair_count = (patch_blocks + 1) // 2
    width = selected_indices.shape[-1]
    valid = (
        torch.arange(width, device=selected_indices.device)
        .view(1, 1, 1, width)
        < capacity
    )
    pair_indices = (selected_indices // 2).reshape(
        *selected_indices.shape[:2], -1
    )
    pair_counts = torch.zeros(
        *selected_indices.shape[:2],
        pair_count,
        device=selected_indices.device,
        dtype=torch.float32,
    )
    pair_counts.scatter_add_(
        -1,
        pair_indices,
        valid.expand_as(selected_indices).reshape(
            *selected_indices.shape[:2], -1
        ).to(torch.float32),
    )
    pair_size = torch.full(
        (pair_count,),
        2.0,
        device=selected_indices.device,
        dtype=torch.float32,
    )
    if patch_blocks % 2:
        pair_size[-1] = 1.0
    query_rows = max(selected_indices.shape[-2], 1)
    return (
        pair_counts / (pair_size.view(1, 1, pair_count) * query_rows)
    ).unsqueeze(-2)


def _reduce_exact_pv_execution_to_pair(
    execution_map: torch.Tensor,
    *,
    patch_blocks: int,
) -> torch.Tensor:
    """Reduce binary child execution before converting the compact result to float."""
    if execution_map.ndim != 5:
        raise ValueError("PV execution map must have shape [B,H,Q,W,K]")
    if patch_blocks <= 0 or patch_blocks > execution_map.shape[-1]:
        raise ValueError("invalid patch block count")
    patch_execution = execution_map[..., :patch_blocks]
    if patch_blocks % 2:
        patch_execution = F.pad(patch_execution, (0, 1))
    pair_count = (patch_blocks + 1) // 2
    # A single reduction over Q, warp, child, and every pair can materialize
    # a sequence-sized temporary in PyTorch. Keep the exact same integer sum
    # while bounding peak workspace independently of sequence length.
    pair_counts = torch.empty(
        *patch_execution.shape[:2],
        pair_count,
        device=execution_map.device,
        dtype=torch.float32,
    )
    for pair_start in range(0, pair_count, 32):
        pair_end = min(pair_start + 32, pair_count)
        child_start = 2 * pair_start
        child_end = 2 * pair_end
        pair_chunk = patch_execution[..., child_start:child_end].reshape(
            *patch_execution.shape[:-1], pair_end - pair_start, 2
        )
        pair_counts[..., pair_start:pair_end] = pair_chunk.sum(
            dim=(-1, -3, -4), dtype=torch.float32
        )
    pair_size = torch.full(
        (pair_count,), 2.0,
        device=execution_map.device,
        dtype=torch.float32,
    )
    if patch_blocks % 2:
        pair_size[-1] = 1.0
    query_warps = execution_map.shape[-3] * execution_map.shape[-2]
    return (
        pair_counts.float()
        / (float(query_warps) * pair_size.view(1, 1, -1))
    ).unsqueeze(-2)


def _reduce_kernel_pair_service_counts(
    pair_counts: torch.Tensor,
    *,
    patch_blocks: int,
    query_warps_per_block: int,
) -> torch.Tensor:
    """Normalize exact pair-service counts accumulated inside the QK/PV kernel."""
    if pair_counts.ndim != 4 or pair_counts.dtype != torch.int32:
        raise ValueError("kernel pair counts must be int32 [B,H,Q,pair]")
    if patch_blocks <= 0:
        raise ValueError("patch block count must be positive")
    pair_count = (patch_blocks + 1) // 2
    if pair_count > pair_counts.shape[-1]:
        raise ValueError("kernel pair counts do not cover all patch pairs")
    if query_warps_per_block <= 0:
        raise ValueError("query warps per block must be positive")
    pair_size = torch.full(
        (pair_count,),
        2.0,
        device=pair_counts.device,
        dtype=torch.float32,
    )
    if patch_blocks % 2:
        pair_size[-1] = 1.0
    denominator = (
        float(pair_counts.shape[-2] * query_warps_per_block)
        * pair_size.view(1, 1, pair_count)
    )
    return (
        pair_counts[..., :pair_count].sum(dim=-2, dtype=torch.float32)
        / denominator
    ).unsqueeze(-2)


def _resolve_qk_pv_threshold(spec: str, key_tokens: int) -> float:
    """Resolve the in-kernel online-softmax skip threshold.

    ``sequence_mass_bound`` instantiates the CoSA/BLASST condition with
    skip scale Delta=1, hence tau=log(N). A skipped key tile then contributes
    at most one uniform-block share under the running-max upper bound.
    ``natural_gap:x`` expresses a natural-logit gap and converts it to the
    base-2 score domain used by the CUDA online-softmax kernel.
    """
    if key_tokens <= 0:
        raise ValueError("key token count must be positive")
    if spec == "sequence_mass_bound":
        return math.log(float(max(key_tokens, 2)))
    if spec.startswith("natural_gap:"):
        natural_gap = float(spec.split(":", 1)[1])
        if not math.isfinite(natural_gap) or natural_gap < 0.0:
            raise ValueError("natural-logit PV gap must be finite and non-negative")
        return natural_gap * math.log2(math.e)
    value = float(spec)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "SPARSE_VGGT_QK_PV_THRESHOLD must be non-negative or "
            "sequence_mass_bound"
        )
    return value


def _pv_threshold_active_for_layer(spec: str, layer_idx: int) -> bool:
    """Return whether the optional PV post-cut runs at this global layer."""
    normalized = spec.strip()
    if not normalized:
        return True
    active_layers = {
        int(token.strip())
        for token in normalized.split(",")
        if token.strip()
    }
    return layer_idx in active_layers


@torch.no_grad()
def _cosa_hrm_order_priority(
    query: torch.Tensor,
    key: torch.Tensor,
    support: torch.Tensor,
    *,
    query_block_size: int = 128,
    key_block_size: int = 64,
    queries_per_block: int = 4,
    proxy_key_stride: int = 8,
) -> torch.Tensor:
    """Build a CoSA-inspired HRM-first traversal priority.

    A small, fixed query/key sample estimates which key blocks contain a row
    maximum. Those blocks are visited first, then the remaining blocks follow
    proxy probability mass. The support itself is never changed.
    """
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("HRM order requires [B,H,T,D] query and key tensors")
    if support.ndim != 4 or support.dtype != torch.bool:
        raise ValueError("HRM order requires boolean [B,H,Qblk,Kblk] support")
    if query.shape[:2] != support.shape[:2] or key.shape[:2] != support.shape[:2]:
        raise ValueError("HRM order batch/head layouts do not match support")
    if min(
        query_block_size,
        key_block_size,
        queries_per_block,
        proxy_key_stride,
    ) <= 0:
        raise ValueError("HRM order sampling parameters must be positive")

    query_blocks = support.shape[-2]
    key_blocks = support.shape[-1]
    query_offsets = torch.linspace(
        0,
        query_block_size - 1,
        queries_per_block,
        device=query.device,
    ).round().long()
    query_indices = (
        torch.arange(query_blocks, device=query.device)[:, None]
        * query_block_size
        + query_offsets[None, :]
    ).clamp_max(query.shape[-2] - 1)
    key_offsets = torch.arange(
        0, key_block_size, proxy_key_stride, device=key.device
    )
    key_indices = (
        torch.arange(key_blocks, device=key.device)[:, None] * key_block_size
        + key_offsets[None, :]
    ).clamp_max(key.shape[-2] - 1)

    sampled_query = query.index_select(-2, query_indices.flatten()).reshape(
        *query.shape[:2], query_blocks, queries_per_block, query.shape[-1]
    )
    sampled_key = key.index_select(-2, key_indices.flatten()).reshape(
        *key.shape[:2], key_blocks, key_offsets.numel(), key.shape[-1]
    )
    proxy_logits = torch.einsum(
        "bhqrd,bhksd->bhqrks", sampled_query.float(), sampled_key.float()
    ) * (query.shape[-1] ** -0.5)
    proxy_logits = proxy_logits.masked_fill(
        ~support.unsqueeze(-2).unsqueeze(-1), float("-inf")
    )
    proxy_probability = proxy_logits.flatten(-2).softmax(dim=-1).reshape_as(
        proxy_logits
    )
    proxy_mass = proxy_probability.sum(dim=(-3, -1))
    estimated_rowmax = proxy_logits.amax(dim=-1).argmax(dim=-1)
    has_estimated_rowmax = torch.zeros_like(support)
    has_estimated_rowmax.scatter_(-1, estimated_rowmax, True)
    priority_span = proxy_mass.sum(dim=-1, keepdim=True) + 1.0
    return (
        proxy_mass + priority_span * has_estimated_rowmax.float()
    ).masked_fill(~support, float("-inf"))


def _priority_ordered_block_lut(
    block_map: torch.Tensor,
    patch_priority: torch.Tensor,
    *,
    max_selected_blocks: int | None = None,
    selected_patch_indices: torch.Tensor | None = None,
    frontier_only: bool = False,
    frontier_anchor_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode selected blocks as signed jumps in descending priority order.

    The selected support is unchanged. Appended sink blocks are visited first
    because they have no pooled QK proxy and are always retained. The physical
    final block is forced to the end so the CUDA kernel's final OOB predicate
    remains valid for a partially filled sequence tail.
    """
    if block_map.ndim != 4 or block_map.dtype != torch.bool:
        raise ValueError("ordered LUT requires a boolean [B,H,Q,K] block map")
    if patch_priority.ndim != 4:
        raise ValueError("ordered LUT requires [B,H,Q,K_patch] priorities")
    if patch_priority.shape[:-1] != block_map.shape[:-1]:
        raise ValueError("ordered LUT priority rows do not match the block map")
    patch_blocks = patch_priority.shape[-1]
    total_blocks = block_map.shape[-1]
    if patch_blocks > total_blocks:
        raise ValueError("ordered LUT has more patch priorities than blocks")
    if max_selected_blocks is not None and max_selected_blocks < 0:
        raise ValueError("ordered LUT selection bound must be non-negative")

    if selected_patch_indices is not None:
        if selected_patch_indices.shape[:-1] != block_map.shape[:-1]:
            raise ValueError(
                "compact ordered LUT indices do not match the block-map rows"
            )
        if selected_patch_indices.dtype not in {torch.int32, torch.int64}:
            raise ValueError("compact ordered LUT indices must be integers")
        if selected_patch_indices.shape[-1] > patch_blocks:
            raise ValueError(
                "compact ordered LUT has more candidates than patch blocks"
            )

        patch_count = block_map[..., :patch_blocks].sum(
            dim=-1, dtype=torch.int32
        )
        candidate_width = selected_patch_indices.shape[-1]
        patch_positions = torch.arange(
            candidate_width, device=block_map.device
        ).view(1, 1, 1, candidate_width)
        valid_patch = patch_positions < patch_count.unsqueeze(-1)
        patch_indices = selected_patch_indices.to(dtype=torch.int64)
        safe_patch_indices = patch_indices.clamp(0, max(patch_blocks - 1, 0))
        selected_priority = patch_priority.gather(
            -1, safe_patch_indices
        ).nan_to_num(nan=float("-inf"))

        sink_blocks = total_blocks - patch_blocks
        if sink_blocks > 0:
            sink_indices = torch.arange(
                patch_blocks, total_blocks, device=block_map.device
            ).view(1, 1, 1, sink_blocks).expand(
                *block_map.shape[:-1], sink_blocks
            )
            sink_valid = block_map[..., patch_blocks:]
            sink_priority = torch.full(
                sink_valid.shape,
                torch.finfo(patch_priority.dtype).max,
                dtype=patch_priority.dtype,
                device=block_map.device,
            )
            candidate_indices = torch.cat(
                [safe_patch_indices, sink_indices], dim=-1
            )
            candidate_valid = torch.cat([valid_patch, sink_valid], dim=-1)
            candidate_priority = torch.cat(
                [selected_priority, sink_priority], dim=-1
            )
        else:
            candidate_indices = safe_patch_indices
            candidate_valid = valid_patch
            candidate_priority = selected_priority

        candidate_priority = candidate_priority.masked_fill(
            ~candidate_valid, float("-inf")
        )
        tail_candidate = candidate_indices == total_blocks - 1
        candidate_priority = torch.where(
            tail_candidate & candidate_valid,
            torch.full_like(
                candidate_priority, torch.finfo(candidate_priority.dtype).min
            ),
            candidate_priority,
        )
        if frontier_only or frontier_anchor_only:
            # Online softmax needs an early high-logit tile, not a globally
            # random traversal. Visit sinks, then the strongest pooled-QK
            # patch, and stream the remainder in physical order.
            patch_priority_only = selected_priority.masked_fill(
                ~valid_patch, float("-inf")
            )
            patch_priority_only = patch_priority_only.masked_fill(
                safe_patch_indices == total_blocks - 1,
                float("-inf"),
            )
            frontier_position = patch_priority_only.argmax(
                dim=-1, keepdim=True
            )
            candidate_position = torch.arange(
                candidate_indices.shape[-1], device=block_map.device
            ).view(1, 1, 1, -1)
            is_frontier = (
                candidate_position == frontier_position
            ) & candidate_valid & (candidate_indices < patch_blocks)
            is_sink = candidate_valid & (candidate_indices >= patch_blocks)
            order_key = candidate_indices.to(torch.int64)
            order_key = torch.where(
                is_frontier,
                torch.full_like(order_key, -1),
                order_key,
            )
            if frontier_only:
                order_key = torch.where(
                    is_sink,
                    candidate_indices.to(torch.int64)
                    - patch_blocks
                    - total_blocks,
                    order_key,
                )
            order_key = torch.where(
                tail_candidate & candidate_valid,
                torch.full_like(order_key, 2 * total_blocks),
                order_key,
            )
            order_key = torch.where(
                candidate_valid,
                order_key,
                torch.full_like(order_key, 3 * total_blocks),
            )
            order = order_key.argsort(dim=-1)
        else:
            order = candidate_priority.argsort(dim=-1, descending=True)
        ordered_candidates = candidate_indices.gather(-1, order)
        valid_block_num = candidate_valid.sum(dim=-1, dtype=torch.int32)
        ordered_absolute = F.pad(
            ordered_candidates,
            (0, total_blocks - ordered_candidates.shape[-1]),
        )
        positions = torch.arange(total_blocks, device=block_map.device).view(
            1, 1, 1, total_blocks
        )
        valid = positions < valid_block_num.unsqueeze(-1)
        previous = torch.cat(
            [
                torch.zeros_like(ordered_absolute[..., :1]),
                ordered_absolute[..., :-1],
            ],
            dim=-1,
        )
        signed_jumps = ordered_absolute - previous
        lut = torch.where(
            valid, signed_jumps, torch.zeros_like(signed_jumps)
        ).to(torch.int32)
        return lut.contiguous(), valid_block_num.contiguous()

    priority = torch.full(
        block_map.shape,
        float("-inf"),
        dtype=patch_priority.dtype,
        device=block_map.device,
    )
    priority[..., :patch_blocks] = patch_priority.nan_to_num(
        nan=float("-inf")
    )
    if patch_blocks < total_blocks:
        priority[..., patch_blocks:] = torch.finfo(priority.dtype).max

    # The kernel predicates the first and final loads. Keeping the physical
    # sequence tail final is sufficient for arbitrary signed traversal jumps.
    tail_selected = block_map[..., -1]
    priority[..., -1] = torch.where(
        tail_selected,
        torch.full_like(priority[..., -1], torch.finfo(priority.dtype).min),
        priority[..., -1],
    )
    priority = priority.masked_fill(~block_map, float("-inf"))
    valid_block_num = block_map.sum(dim=-1, dtype=torch.int32)
    sort_width = total_blocks
    if max_selected_blocks is not None:
        sort_width = min(int(max_selected_blocks), total_blocks)
    if sort_width < total_blocks:
        # Ratio routing gives a strict row-wise visit bound. Sorting only that
        # prefix is equivalent to a full argsort because every omitted entry
        # has -inf priority and is masked out by valid_block_num below.
        ordered_prefix = priority.topk(
            sort_width, dim=-1, largest=True, sorted=True
        ).indices
        ordered_absolute = F.pad(
            ordered_prefix, (0, total_blocks - sort_width)
        )
    else:
        ordered_absolute = priority.argsort(dim=-1, descending=True)
    positions = torch.arange(total_blocks, device=block_map.device).view(
        1, 1, 1, total_blocks
    )
    valid = positions < valid_block_num.unsqueeze(-1)
    previous = torch.cat(
        [torch.zeros_like(ordered_absolute[..., :1]), ordered_absolute[..., :-1]],
        dim=-1,
    )
    signed_jumps = ordered_absolute - previous
    lut = torch.where(valid, signed_jumps, torch.zeros_like(signed_jumps)).to(
        torch.int32
    )
    return lut.contiguous(), valid_block_num.contiguous()


def _observe_cosa_endpoint_omission_risk(
    residual_risk: torch.Tensor,
    current_selected: torch.Tensor,
    debt_selected: torch.Tensor,
    *,
    protected_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compare Current and Debt supports under one residual-risk measure.

    This is an endpoint observer only. It never changes either support. The
    caller may later use the same omission-risk functional to define a
    parameter-free continuation between the two frozen endpoints.
    """
    if residual_risk.shape != current_selected.shape:
        raise ValueError("residual risk and Current support must match")
    if debt_selected.shape != current_selected.shape:
        raise ValueError("Current and Debt supports must match")
    if current_selected.dtype != torch.bool or debt_selected.dtype != torch.bool:
        raise ValueError("CoSA endpoint supports must be boolean")
    eligible = torch.ones_like(current_selected)
    if protected_mask is not None:
        if protected_mask.shape != current_selected.shape:
            raise ValueError("protected mask must match CoSA endpoint supports")
        eligible = ~protected_mask

    risk = residual_risk.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    current_omission = (risk * (eligible & ~current_selected)).sum(dim=-1)
    debt_omission = (risk * (eligible & ~debt_selected)).sum(dim=-1)
    scale = current_omission.abs().clamp_min(1.0)
    tolerance = 8.0 * torch.finfo(risk.dtype).eps * scale
    changed = ((current_selected ^ debt_selected) & eligible).any(dim=-1)
    debt_safe = debt_omission <= current_omission + tolerance
    return {
        "cosa_pair_risk_observer_enabled": torch.tensor(
            1.0, device=risk.device
        ),
        "cosa_pair_risk_current_omission": current_omission.mean().detach(),
        "cosa_pair_risk_debt_omission": debt_omission.mean().detach(),
        "cosa_pair_risk_debt_delta": (
            debt_omission - current_omission
        ).mean().detach(),
        "cosa_pair_risk_debt_safe_fraction": debt_safe.float().mean().detach(),
        "cosa_pair_risk_support_changed_fraction": (
            changed.float().mean().detach()
        ),
    }


@torch.no_grad()
def _observe_cosa_global_service_ledger(
    *,
    selected: torch.Tensor,
    current_selected: torch.Tensor,
    debt_selected: torch.Tensor,
    score: torch.Tensor,
    routing_state: dict,
    service_gate_uses_debt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Record cross-global-event service history without changing support."""
    key_blocks = selected.shape[-1]
    pair_count = (key_blocks + 1) // 2
    padding = key_blocks % 2

    def to_pairs(mask: torch.Tensor) -> torch.Tensor:
        padded = F.pad(mask, (0, padding), value=False)
        return padded.reshape(*mask.shape[:-1], pair_count, 2).any(dim=-1)

    pair_selected = to_pairs(selected)
    pair_current = to_pairs(current_selected)
    pair_debt = to_pairs(debt_selected)
    pair_union = pair_current | pair_debt
    pair_shared = pair_current & pair_debt
    pair_debt_only = pair_debt & ~pair_current
    pair_current_only = pair_current & ~pair_debt

    padded_score = F.pad(score.float().clamp_min(0.0), (0, padding))
    pair_score = padded_score.reshape(
        *score.shape[:-1], pair_count, 2
    ).sum(dim=-1)

    count_key = "cosa_service_ledger_pair_count"
    previous_count = routing_state.get(count_key)
    if previous_count is None or previous_count.shape != pair_selected.shape:
        previous_count = torch.zeros_like(pair_score)
    else:
        previous_count = previous_count.to(pair_score)
    next_count = previous_count + pair_selected.float()
    routing_state[count_key] = next_count.detach()

    event_key = "cosa_service_ledger_event_count"
    event_count = int(routing_state.get(event_key, 0)) + 1
    routing_state[event_key] = event_count

    previous_selected_key = "cosa_service_ledger_previous_support"
    previous_selected = routing_state.get(previous_selected_key)
    if previous_selected is None or previous_selected.shape != pair_selected.shape:
        previous_selected = torch.zeros_like(pair_selected)
    else:
        previous_selected = previous_selected.to(pair_selected.device).bool()
    routing_state[previous_selected_key] = pair_selected.detach()

    age_key = "cosa_service_ledger_wait_age"
    previous_age = routing_state.get(age_key)
    if previous_age is None or previous_age.shape != pair_selected.shape:
        previous_age = torch.zeros_like(pair_score)
    else:
        previous_age = previous_age.to(pair_score)
    next_age = torch.where(
        pair_selected,
        torch.zeros_like(previous_age),
        previous_age + 1.0,
    )
    routing_state[age_key] = next_age.detach()

    selected_count = pair_selected.sum(dim=-1).float().clamp_min(1.0)
    first_selected = pair_selected & (previous_count == 0)
    repeated_selected = pair_selected & (previous_count > 0)
    union_count = pair_union.sum(dim=-1).float().clamp_min(1.0)
    selected_score = pair_score.masked_fill(~pair_selected, 0.0).sum(dim=-1)
    first_score = pair_score.masked_fill(~first_selected, 0.0).sum(dim=-1)
    repeated_score = pair_score.masked_fill(~repeated_selected, 0.0).sum(dim=-1)
    total_score = pair_score.sum(dim=-1).clamp_min(1e-12)
    sorted_pair_score = pair_score.sort(dim=-1, descending=True).values
    pair_rank = torch.arange(pair_count, device=pair_score.device).view(
        *([1] * (pair_score.ndim - 1)),
        pair_count,
    )
    top_budget_mask = pair_rank < selected_count.long().unsqueeze(-1)
    top_budget_score = sorted_pair_score.masked_fill(
        ~top_budget_mask,
        0.0,
    ).sum(dim=-1)
    score_probability = pair_score / total_score.unsqueeze(-1)
    score_entropy = -torch.where(
        score_probability > 0,
        score_probability * score_probability.clamp_min(1e-12).log(),
        torch.zeros_like(score_probability),
    ).sum(dim=-1)
    normalized_score_entropy = (
        score_entropy / math.log(pair_count)
        if pair_count > 1
        else torch.zeros_like(score_entropy)
    )

    service_total = next_count.sum(dim=-1)
    normalized_hhi = (
        next_count.square().sum(dim=-1)
        / service_total.square().clamp_min(1e-12)
        * pair_count
    )
    effective_fraction = torch.where(
        normalized_hhi > 0,
        normalized_hhi.reciprocal(),
        torch.zeros_like(normalized_hhi),
    )

    overlap = (pair_selected & previous_selected).sum(dim=-1).float()
    temporal_union = (pair_selected | previous_selected).sum(dim=-1).float()
    retention = torch.where(
        temporal_union > 0,
        overlap / temporal_union.clamp_min(1.0),
        torch.zeros_like(temporal_union),
    )
    selected_wait = previous_age.masked_select(pair_selected)

    return {
        "service_obs_schema_version": torch.tensor(
            8.0, device=score.device
        ),
        "service_obs_mainline_event_count": torch.tensor(
            float(event_count), device=score.device
        ),
        "service_obs_mainline_gate_uses_debt": (
            service_gate_uses_debt.float().detach()
        ),
        "service_obs_mainline_repeat_pair_fraction": (
            repeated_selected.sum(dim=-1).float() / selected_count
        ).mean().detach(),
        "service_obs_mainline_first_pair_fraction": (
            first_selected.sum(dim=-1).float() / selected_count
        ).mean().detach(),
        "service_obs_mainline_unique_pair_coverage": (
            (next_count > 0).float().mean()
        ).detach(),
        "service_obs_mainline_normalized_hhi": normalized_hhi.mean().detach(),
        "service_obs_mainline_effective_service_fraction": (
            effective_fraction.mean().detach()
        ),
        "service_obs_mainline_max_service_count": next_count.max().detach(),
        "service_obs_mainline_service_count_p90": torch.quantile(
            next_count.flatten(), 0.90
        ).detach(),
        "service_obs_mainline_temporal_jaccard": retention.mean().detach(),
        "service_obs_mainline_temporal_churn": (
            1.0 - retention.mean()
        ).detach(),
        "service_obs_mainline_selected_wait_age": (
            selected_wait.mean().detach()
            if selected_wait.numel()
            else torch.tensor(0.0, device=score.device)
        ),
        "service_obs_mainline_max_wait_age": next_age.max().detach(),
        "service_obs_mainline_endpoint_shared_fraction": (
            pair_shared.sum(dim=-1).float() / union_count
        ).mean().detach(),
        "service_obs_mainline_endpoint_debt_only_fraction": (
            pair_debt_only.sum(dim=-1).float() / union_count
        ).mean().detach(),
        "service_obs_mainline_endpoint_current_only_fraction": (
            pair_current_only.sum(dim=-1).float() / union_count
        ).mean().detach(),
        "service_obs_mainline_selected_score_mass": (
            selected_score / total_score
        ).mean().detach(),
        "service_obs_mainline_first_score_share": (
            first_score / selected_score.clamp_min(1e-12)
        ).mean().detach(),
        "service_obs_mainline_repeat_score_share": (
            repeated_score / selected_score.clamp_min(1e-12)
        ).mean().detach(),
        "service_obs_mainline_debt_only_score_share": (
            pair_score.masked_fill(~pair_debt_only, 0.0).sum(dim=-1)
            / pair_score.masked_fill(~pair_union, 0.0).sum(dim=-1).clamp_min(1e-12)
        ).mean().detach(),
        "service_obs_mainline_score_topbudget_mass": (
            top_budget_score / total_score
        ).mean().detach(),
        "service_obs_mainline_score_selection_efficiency": (
            selected_score / top_budget_score.clamp_min(1e-12)
        ).mean().detach(),
        "service_obs_mainline_score_normalized_entropy": (
            normalized_score_entropy.mean().detach()
        ),
    }


def get_cosa_value_risk_pair_debt_mask(
    pooled_score: torch.Tensor,
    pooled_value: torch.Tensor | None,
    base_mask: torch.Tensor,
    *,
    layer_idx: int,
    routing_state: dict | None,
    stats_store: dict | None = None,
    protected_mask: torch.Tensor | None = None,
    max_capacity_hint: int | None = None,
    fp32_projection: bool = False,
    risk_safe_observer: bool = False,
    raw_value: torch.Tensor | None = None,
    value_block_size: int = 64,
) -> torch.Tensor:
    """Select value-aware candidates with execution-conditioned pair memory.

    The QK support is selected before the kernel. By default, exact
    online-softmax PV service from that kernel is settled one layer later and
    is the only event that repays the stable pair ledger. The default-off
    ``qk_admission`` service domain instead settles exact first-stage support;
    this makes the conservative PV post-cut independent of scheduler debt.
    """
    if pooled_score.ndim != 4 or base_mask.shape != pooled_score.shape:
        raise ValueError("CoSA pair debt requires matching [B,H,Q,K] scores")
    fused_routing = os.environ.get(
        "SPARSE_VGGT_COSA_FUSED_ROUTING", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if pooled_value is None and not (fused_routing and raw_value is not None):
        raise ValueError("CoSA pair debt requires pooled or raw values")
    if pooled_value is not None and pooled_value.ndim != 4:
        raise ValueError("CoSA pair debt requires pooled [B,H,K,D] values")
    if pooled_value is not None and pooled_value.shape[:2] != pooled_score.shape[:2]:
        raise ValueError("pooled score/value batch and head dimensions differ")
    if pooled_value is not None and pooled_value.shape[-2] != pooled_score.shape[-1]:
        raise ValueError("pooled score/value key layouts differ")
    if raw_value is not None and raw_value.shape[:2] != pooled_score.shape[:2]:
        raise ValueError("raw value batch and head dimensions differ")
    if base_mask.dtype != torch.bool:
        raise ValueError("base_mask must be boolean")
    if layer_idx < 0:
        raise ValueError("layer_idx must be non-negative")
    if routing_state is None:
        raise ValueError("CoSA pair debt requires routing_state")
    if protected_mask is not None and protected_mask.shape != base_mask.shape:
        raise ValueError("protected mask must match the QK support")
    if max_capacity_hint is not None and max_capacity_hint < 0:
        raise ValueError("max capacity hint must be non-negative")

    score = (
        pooled_score
        if fused_routing
        else pooled_score.float().clamp_min(0.0).nan_to_num_(nan=0.0)
    )
    value = (
        None
        if fused_routing
        else pooled_value.float().nan_to_num_(nan=0.0)
    )
    key_blocks = score.shape[-1]
    max_capacity = (
        min(int(max_capacity_hint), key_blocks)
        if max_capacity_hint is not None
        else None
    )
    uniform_capacity_hint = os.environ.get(
        "SPARSE_VGGT_COSA_UNIFORM_CAPACITY_HINT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if uniform_capacity_hint:
        if max_capacity is None:
            raise RuntimeError("uniform capacity requires an explicit hint")
        capacity = torch.full(
            (*base_mask.shape[:-1], 1),
            max_capacity,
            dtype=torch.int64,
            device=base_mask.device,
        )
    else:
        capacity = base_mask.sum(dim=-1, keepdim=True)
        if max_capacity is None:
            max_capacity = int(capacity.max().item()) if capacity.numel() else 0
    assert max_capacity is not None

    pair_count = (key_blocks + 1) // 2
    pair_ids = torch.arange(key_blocks, device=score.device) // 2
    pair_size = torch.bincount(pair_ids, minlength=pair_count).to(score)
    if fused_routing:
        from sparse_vggt.kernels.cosa_routing import (
            triton_pair_innovation,
            triton_pooled_pair_innovation,
        )

        innovation = (
            triton_pooled_pair_innovation(raw_value, value_block_size)
            if raw_value is not None
            else triton_pair_innovation(pooled_value)
        )
        if innovation.shape[-1] != key_blocks:
            raise RuntimeError("fused value pooling changed the QK block layout")
    else:
        assert value is not None
        left = value[..., 0::2, :]
        right = value[..., 1::2, :]
        if right.shape[-2] != left.shape[-2]:
            right = torch.cat([right, left[..., -1:, :]], dim=-2)
        pair_innovation = (left - right).square().mean(dim=-1).sqrt() * 0.5
        innovation = pair_innovation.repeat_interleave(2, dim=-1)[
            ..., :key_blocks
        ].float()
    risk = None if fused_routing else score * innovation.unsqueeze(-2)

    projection_dtype = torch.float32 if fp32_projection else torch.float64
    conservative_arrival_projection = os.environ.get(
        "SPARSE_VGGT_COSA_CONSERVATIVE_ARRIVAL_PROJECTION", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if conservative_arrival_projection:
        inplace_arrival_projection = os.environ.get(
            "SPARSE_VGGT_COSA_INPLACE_ARRIVAL_PROJECTION", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if fused_routing and not risk_safe_observer:
            from sparse_vggt.kernels.cosa_routing import triton_conservative_arrival

            row_arrival = triton_conservative_arrival(
                score,
                innovation,
                capacity,
            )
        elif inplace_arrival_projection and not risk_safe_observer:
            assert risk is not None
            row_arrival = _project_conservative_capped_mass_inplace(
                risk,
                capacity,
            )
        else:
            assert risk is not None
            row_arrival = _project_conservative_capped_mass(
                risk,
                capacity.to(risk),
            )
    else:
        if risk is None:
            raise RuntimeError("fused CoSA routing requires conservative arrival")
        row_arrival = _project_capped_proportional_mass(
            risk,
            capacity.to(risk),
            fill_zero_rows=True,
            compute_dtype=projection_dtype,
            max_saturated_hint=max_capacity,
        )
    padded_arrival = F.pad(row_arrival, (0, key_blocks % 2))
    row_pair_arrival = padded_arrival.reshape(
        *row_arrival.shape[:-1], pair_count, 2
    ).sum(dim=-1)
    row_pair_arrival = row_pair_arrival / pair_size.view(
        1, 1, 1, pair_count
    ).clamp_min(1.0)
    arrival = row_pair_arrival.mean(dim=-2, keepdim=True)
    pair_capacity = row_pair_arrival.sum(dim=-1, keepdim=True).mean(
        dim=-2, keepdim=True
    )

    state_key = "cosa_value_risk_pair_debt"
    pending_key = "cosa_value_risk_pair_pending_arrival"
    pending_layer_key = "cosa_value_risk_pair_pending_layer"
    service_key = "qk_pv_service_observation"
    previous = routing_state.get(state_key)
    if previous is None or tuple(previous.shape) != tuple(arrival.shape):
        previous = torch.zeros_like(arrival)
    else:
        previous = previous.to(arrival).clamp_min(0.0)

    previous_before_service = previous
    settled_arrival = torch.zeros_like(arrival)
    consumed_service = torch.zeros_like(arrival)
    reflection = torch.zeros_like(arrival)
    pending_arrival = routing_state.get(pending_key)
    service_observation = routing_state.pop(service_key, None)
    if pending_arrival is not None:
        if service_observation is None or not service_observation.get(
            "identity_observed", False
        ):
            raise RuntimeError("CoSA pair debt requires exact PV service identity")
        pending_layer = routing_state.get(pending_layer_key)
        if service_observation.get("layer_idx") != pending_layer:
            raise RuntimeError("PV service does not match pending CoSA layer")
        settled_arrival = pending_arrival.to(arrival)
        if "pair_service_fraction" in service_observation:
            consumed_service = service_observation[
                "pair_service_fraction"
            ].to(arrival)
        else:
            consumed_service = _reduce_pv_service_to_pair(
                service_observation["service_fraction"].to(arrival),
                pair_ids=pair_ids,
                pair_size=pair_size,
            )
        if tuple(consumed_service.shape) != tuple(arrival.shape):
            raise RuntimeError("PV service changed the stable pair layout")
        unreflected = previous_before_service + settled_arrival - consumed_service
        reflection = (-unreflected).clamp_min(0.0)
        previous = unreflected.clamp_min(0.0)

    # Memory is admitted for execution only when
    # the previous layer's useful repayment is at least its reflected
    # (wasted) service. With no settled service, the path starts at Current.
    useful_repayment = (consumed_service - reflection).clamp_min(0.0).sum()
    wasted_service = reflection.sum()
    consumed_service_total = consumed_service.sum()
    service_gate_uses_debt = (consumed_service_total > 0) & (
        useful_repayment >= wasted_service
    )
    service_utilization = torch.where(
        consumed_service_total > 0,
        useful_repayment / consumed_service_total.clamp_min(1e-8),
        torch.zeros_like(consumed_service_total),
    )

    conservative_debt_projection = os.environ.get(
        "SPARSE_VGGT_COSA_CONSERVATIVE_DEBT_PROJECTION", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if conservative_debt_projection:
        admitted_pair = _project_conservative_capped_mass(
            previous,
            pair_capacity.to(previous),
        )
    else:
        admitted_pair = _project_capped_proportional_mass(
            previous,
            pair_capacity.to(previous),
            fill_zero_rows=False,
            compute_dtype=projection_dtype,
            max_saturated_hint=(max_capacity + 1) // 2,
        )
    admitted = admitted_pair.repeat_interleave(2, dim=-1)[..., :key_blocks]
    current_priority = row_arrival
    debt_priority = row_arrival + admitted
    if protected_mask is not None:
        current_priority = current_priority.masked_fill(
            protected_mask, torch.finfo(current_priority.dtype).max
        )
        debt_priority = debt_priority.masked_fill(
            protected_mask, torch.finfo(debt_priority.dtype).max
        )

    if stats_store is not None and bool((capacity > max_capacity).any()):
        raise RuntimeError("capacity hint is smaller than the QK budget")
    def select(priority: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        support = torch.zeros_like(base_mask)
        if not max_capacity:
            return support, torch.empty(
                *base_mask.shape[:-1],
                0,
                dtype=torch.int64,
                device=base_mask.device,
            )
        sort_selected_indices = os.environ.get(
            "SPARSE_VGGT_COSA_SORT_SELECTED_INDICES", "1"
        ).lower() in {"1", "true", "yes", "on"}
        indices = priority.topk(
            max_capacity,
            dim=-1,
            sorted=sort_selected_indices,
        ).indices
        valid = (
            torch.arange(max_capacity, device=score.device)
            .view(1, 1, 1, -1)
            < capacity
        )
        support.scatter_(-1, indices, valid)
        return support, indices

    execution_priority = torch.where(
        service_gate_uses_debt,
        debt_priority,
        current_priority,
    )

    collect_counterfactual_stats = os.environ.get(
        "SPARSE_VGGT_COSA_COUNTERFACTUAL_STATS", "1"
    ).lower() in {"1", "true", "yes", "on"}
    observe_service_ledger = os.environ.get(
        "SPARSE_VGGT_COSA_SERVICE_LEDGER_OBSERVER", "0"
    ).lower() in {"1", "true", "yes", "on"}
    need_counterfactual = risk_safe_observer or observe_service_ledger or (
        stats_store is not None and collect_counterfactual_stats
    )
    current_result = (
        select(current_priority)
        if need_counterfactual
        else None
    )
    debt_result = (
        select(debt_priority)
        if need_counterfactual
        else None
    )
    current_selected = None if current_result is None else current_result[0]
    current_indices = None if current_result is None else current_result[1]
    debt_selected = None if debt_result is None else debt_result[0]
    debt_indices = None if debt_result is None else debt_result[1]
    if need_counterfactual:
        assert current_selected is not None and debt_selected is not None
        assert current_indices is not None and debt_indices is not None
        selected = torch.where(
            service_gate_uses_debt,
            debt_selected,
            current_selected,
        )
        selected_indices = torch.where(
            service_gate_uses_debt,
            debt_indices,
            current_indices,
        )
    else:
        selected, selected_indices = select(execution_priority)
    assert selected is not None and selected_indices is not None

    if observe_service_ledger:
        if stats_store is None:
            raise RuntimeError(
                "CoSA service ledger observer requires routing statistics"
            )
        assert current_selected is not None and debt_selected is not None
        stats_store.update(
            _observe_cosa_global_service_ledger(
                selected=selected,
                current_selected=current_selected,
                debt_selected=debt_selected,
                score=score,
                routing_state=routing_state,
                service_gate_uses_debt=service_gate_uses_debt,
            )
        )

    postcut_observer_layers = {
        int(item)
        for item in os.environ.get(
            "SPARSE_VGGT_COSA_POSTCUT_ORDER_OBSERVER_LAYERS", "0,7,15,23"
        ).split(",")
        if item.strip()
    }
    observe_postcut_order = os.environ.get(
        "SPARSE_VGGT_COSA_POSTCUT_ORDER_OBSERVER", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if observe_postcut_order and layer_idx in postcut_observer_layers:
        routing_state["cosa_postcut_order_observer_snapshot"] = {
            "layer_idx": int(layer_idx),
            "support": selected.detach(),
            "pooled_score": score.detach(),
            "debt_priority": debt_priority.detach(),
        }

    if risk_safe_observer:
        if current_selected is None or debt_selected is None:
            raise RuntimeError("risk observer requires both endpoint supports")
        routing_state["cosa_exact_pair_risk_endpoint_supports"] = {
            "layer_idx": int(layer_idx),
            "current": current_selected.detach(),
            "debt": debt_selected.detach(),
        }

    routing_state[state_key] = previous.detach()
    routing_state[pending_key] = arrival.detach()
    routing_state[pending_layer_key] = int(layer_idx)
    debt_service_domain = os.environ.get(
        "SPARSE_VGGT_COSA_DEBT_SERVICE_DOMAIN", "pv_execution"
    ).lower()
    if debt_service_domain not in {"pv_execution", "qk_admission"}:
        raise ValueError("unsupported CoSA debt service domain")
    if debt_service_domain == "qk_admission":
        index_service_reduction = os.environ.get(
            "SPARSE_VGGT_COSA_INDEX_SERVICE_REDUCTION", "0"
        ).lower() in {"1", "true", "yes", "on"}
        routing_state[service_key] = {
            "layer_idx": int(layer_idx),
            "identity_observed": True,
            "service_domain": "qk_admission",
            "pair_service_fraction": (
                _reduce_qk_indices_to_pair(
                    selected_indices,
                    capacity,
                    patch_blocks=key_blocks,
                )
                if index_service_reduction
                else _reduce_qk_admission_to_pair(
                    selected,
                    patch_blocks=key_blocks,
                )
            ).detach(),
        }
    routing_state["cosa_value_risk_pair_priority"] = execution_priority.detach()
    routing_state["cosa_value_risk_pair_max_patch_blocks"] = int(max_capacity)
    routing_state["cosa_value_risk_pair_selected_indices"] = (
        selected_indices.detach()
    )
    # Candidate admission and traversal solve different problems. Value risk
    # plus debt selects the support; pure pooled QK relevance orders exact QK
    # visits so the online-softmax running maximum becomes informative early.
    routing_state["cosa_value_risk_pair_qk_order_priority"] = score.detach()

    if stats_store is not None:
        unreflected_error = (
            previous
            - previous_before_service
            - settled_arrival
            + consumed_service
            - reflection
        ).abs().amax()
        denominator = capacity.squeeze(-1).clamp_min(1).float()
        overlap = (selected & base_mask).sum(dim=-1).float()
        stats = {
            "cosa_pair_debt_enabled": torch.tensor(1.0, device=score.device),
            "cosa_pair_service_useful_repayment": useful_repayment.detach(),
            "cosa_pair_service_wasted": wasted_service.detach(),
            "cosa_pair_service_utilization": service_utilization.detach(),
            "cosa_pair_service_gate_debt": (
                service_gate_uses_debt.float().detach()
            ),
            "cosa_pair_qk_capacity": capacity.float().mean().detach(),
            "cosa_pair_arrival": arrival.sum(dim=-1).mean().detach(),
            "cosa_pair_innovation": innovation.mean().detach(),
            "cosa_pair_debt_before_service": (
                previous_before_service.mean().detach()
            ),
            "cosa_pair_settled_arrival": (
                settled_arrival.sum(dim=-1).mean().detach()
            ),
            "cosa_pair_consumed_pv_service": (
                consumed_service.sum(dim=-1).mean().detach()
            ),
            "cosa_pair_admitted_debt": (
                admitted_pair.sum(dim=-1).mean().detach()
            ),
            "cosa_pair_backlog": previous.sum(dim=-1).mean().detach(),
            "cosa_pair_qk_overlap": (overlap / denominator).mean().detach(),
            "cosa_pair_qk_swap_fraction": (
                1.0 - overlap / denominator
            ).mean().detach(),
            "cosa_pair_budget_error": (
                selected.sum(dim=-1, keepdim=True) - capacity
            ).abs().amax().float().detach(),
            "cosa_pair_conservation_error": unreflected_error.detach(),
        }
        if current_selected is not None and debt_selected is not None:
            current_debt_overlap = (
                current_selected & debt_selected
            ).sum(dim=-1).float()
            stats.update({
                "cosa_pair_current_debt_overlap": (
                    current_debt_overlap / denominator
                ).mean().detach(),
                "cosa_pair_history_swap_fraction": (
                    1.0 - current_debt_overlap / denominator
                ).mean().detach(),
            })
        stats_store.update(stats)
        if risk_safe_observer:
            assert risk is not None
            stats_store.update(
                _observe_cosa_endpoint_omission_risk(
                    risk,
                    current_selected,
                    debt_selected,
                    protected_mask=protected_mask,
                )
            )
    return selected


def finalize_cosa_value_risk_pair_debt(
    routing_state: dict | None,
    *,
    stats_store: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Settle the final exact PV service in the CoSA pair ledger."""
    if routing_state is None:
        return {}
    state_key = "cosa_value_risk_pair_debt"
    pending_key = "cosa_value_risk_pair_pending_arrival"
    pending_layer_key = "cosa_value_risk_pair_pending_layer"
    service_key = "qk_pv_service_observation"
    pending_arrival = routing_state.pop(pending_key, None)
    pending_layer = routing_state.pop(pending_layer_key, None)
    if pending_arrival is None:
        return {}
    service_observation = routing_state.pop(service_key, None)
    if service_observation is None or not service_observation.get(
        "identity_observed", False
    ):
        raise RuntimeError("terminal CoSA settlement requires exact PV service")
    if service_observation.get("layer_idx") != pending_layer:
        raise RuntimeError("terminal PV service does not match CoSA layer")

    if "pair_service_fraction" in service_observation:
        consumed_service = service_observation[
            "pair_service_fraction"
        ].to(pending_arrival)
    else:
        key_blocks = int(service_observation["service_fraction"].shape[-1])
        pair_ids = torch.arange(key_blocks, device=pending_arrival.device) // 2
        pair_size = torch.bincount(
            pair_ids, minlength=(key_blocks + 1) // 2
        ).to(pending_arrival)
        consumed_service = _reduce_pv_service_to_pair(
            service_observation["service_fraction"].to(pending_arrival),
            pair_ids=pair_ids,
            pair_size=pair_size,
        )
    previous = routing_state.get(state_key)
    if previous is None or tuple(previous.shape) != tuple(pending_arrival.shape):
        previous = torch.zeros_like(pending_arrival)
    else:
        previous = previous.to(pending_arrival).clamp_min(0.0)

    unreflected = previous + pending_arrival - consumed_service
    reflection = (-unreflected).clamp_min(0.0)
    terminal_debt = unreflected.clamp_min(0.0)
    conservation_error = (
        terminal_debt
        - previous
        - pending_arrival
        + consumed_service
        - reflection
    ).abs().amax()
    routing_state[state_key] = terminal_debt.detach()
    stats = {
        "cosa_pair_terminal_settled": torch.tensor(
            1.0, device=terminal_debt.device
        ),
        "cosa_pair_terminal_layer": torch.tensor(
            float(pending_layer), device=terminal_debt.device
        ),
        "cosa_pair_terminal_arrival": pending_arrival.sum(
            dim=-1
        ).mean().detach(),
        "cosa_pair_terminal_service": consumed_service.sum(
            dim=-1
        ).mean().detach(),
        "cosa_pair_terminal_backlog": terminal_debt.sum(
            dim=-1
        ).mean().detach(),
        "cosa_pair_terminal_conservation_error": conservation_error.detach(),
    }
    if stats_store is not None:
        stats_store.update(stats)
    return stats


def get_preview_adaptive_block_mask(
    pooled_score: torch.Tensor,
    pooled_value: torch.Tensor,
    sparse_ratio: float,
    redistribution_fraction: float = 0.05,
    activation_threshold: float = 0.60,
    num_frames: int | None = None,
    tokens_per_frame: int | None = None,
    q_block_size: int = 128,
    k_block_size: int = 64,
    local_frame_radius: int = 1,
    local_fraction: float = 0.0,
    geometry_weight: float = 0.0,
    value_detail: torch.Tensor | None = None,
    head_protection: str = "none",
    protected_head_fraction: float = 0.25,
    donor_retention_threshold: float = 0.0,
    donor_exchange_scope: str = "head",
    receiver_gain_threshold: float = 0.0,
    exchange_gain_cost_ratio: float = 0.0,
    head_exchange_cap_fraction: float = 1.0,
    layer_confidence_threshold: float = 0.0,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    force_last_block: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Redistribute a strict block budget using confidence-aware output preview."""
    if pooled_score.ndim != 4 or pooled_value.ndim != 4:
        raise ValueError("pooled_score and pooled_value must be four-dimensional")
    if pooled_score.shape[:2] != pooled_value.shape[:2]:
        raise ValueError("pooled score/value batch and head dimensions must match")
    if pooled_score.shape[-1] != pooled_value.shape[-2]:
        raise ValueError("pooled score/value key block dimensions must match")
    if not 0.0 <= sparse_ratio < 1.0:
        raise ValueError("sparse_ratio must be in [0, 1)")
    if not 0.0 <= redistribution_fraction < 1.0:
        raise ValueError("redistribution_fraction must be in [0, 1)")
    if activation_threshold < 0:
        raise ValueError("activation_threshold must be non-negative")
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if geometry_weight < 0:
        raise ValueError("geometry_weight must be non-negative")
    if head_protection not in {"none", "value_detail", "locality"}:
        raise ValueError(
            "head_protection must be none, value_detail, or locality"
        )
    if not 0.0 <= protected_head_fraction < 1.0:
        raise ValueError("protected_head_fraction must be in [0, 1)")
    if not 0.0 <= donor_retention_threshold <= 1.0:
        raise ValueError("donor_retention_threshold must be in [0, 1]")
    if donor_exchange_scope not in {"head", "layer"}:
        raise ValueError("donor_exchange_scope must be head or layer")
    if donor_exchange_scope == "layer" and donor_retention_threshold <= 0:
        raise ValueError(
            "layer donor exchange requires a positive retention threshold"
        )
    if not 0.0 <= receiver_gain_threshold <= 1.0:
        raise ValueError("receiver_gain_threshold must be in [0, 1]")
    if exchange_gain_cost_ratio < 0:
        raise ValueError("exchange_gain_cost_ratio must be non-negative")
    if not 0.0 <= head_exchange_cap_fraction <= 1.0:
        raise ValueError("head_exchange_cap_fraction must be in [0, 1]")
    if layer_confidence_threshold < 0:
        raise ValueError("layer_confidence_threshold must be non-negative")
    utility_certification = (
        receiver_gain_threshold > 0 or exchange_gain_cost_ratio > 0
    )
    if (
        utility_certification or head_exchange_cap_fraction < 1.0
        or layer_confidence_threshold > 0
    ) and donor_exchange_scope != "layer":
        raise ValueError(
            "receiver utility certification, head exchange caps, and layer "
            "confidence gating require layer donor exchange"
        )
    if head_protection == "value_detail":
        if value_detail is None:
            raise ValueError("value_detail is required for value-detail protection")
        if value_detail.shape != pooled_score.shape[:2] + pooled_score.shape[-1:]:
            raise ValueError("value_detail must have shape [B, H, K_blocks]")
    if not 0.0 <= local_fraction < 1.0:
        raise ValueError("local_fraction must be in [0, 1)")
    if (
        local_fraction > 0
        or geometry_weight > 0
        or head_protection == "locality"
    ) and (
        num_frames is None or tokens_per_frame is None
    ):
        raise ValueError(
            "frame metadata is required for geometry-aware preview routing"
        )

    B, heads, query_blocks, key_blocks = pooled_score.shape
    base_budget = max(1, int(key_blocks * (1.0 - sparse_ratio)))
    score = pooled_score.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    probability = score / score.sum(dim=-1, keepdim=True).clamp_min(eps)
    local_region = None
    if local_fraction > 0 or head_protection == "locality":
        min_frame_distance = _block_min_frame_distance(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            q_blocks=query_blocks,
            k_blocks=key_blocks,
            device=pooled_score.device,
        )
        local_region = min_frame_distance <= local_frame_radius
    selection_score = probability
    if geometry_weight > 0:
        geometry_prior, _ = build_soft_geometry_prior(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            q_blocks=query_blocks,
            k_blocks=key_blocks,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_decay="linear",
            decay_gamma=0.25,
            geometry_sigma=None,
            device=pooled_score.device,
        )
        selection_score = torch.softmax(
            probability.clamp_min(eps).log()
            + geometry_weight
            * geometry_prior.view(1, 1, query_blocks, key_blocks),
            dim=-1,
        )
    if force_last_block:
        selection_score = selection_score.clone()
        selection_score[..., -1] = float("inf")

    fixed_count = torch.full(
        (B, heads, query_blocks),
        base_budget,
        dtype=torch.long,
        device=pooled_score.device,
    )
    fixed_mask = _select_variable_topk(selection_score, fixed_count)
    coarse_dense = torch.matmul(
        probability.to(pooled_value.dtype), pooled_value
    ).float()
    sparse_probability = probability.masked_fill(~fixed_mask, 0.0)
    sparse_probability = sparse_probability / sparse_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(eps)
    coarse_sparse = torch.matmul(
        sparse_probability.to(pooled_value.dtype), pooled_value
    ).float()
    preview_error = (coarse_sparse - coarse_dense).norm(dim=-1) / (
        coarse_dense.norm(dim=-1).clamp_min(eps)
    )
    layer_confidence = preview_error.std(
        dim=(1, 2), unbiased=False
    ) / preview_error.mean(dim=(1, 2)).clamp_min(eps)

    protected_head = torch.zeros(
        (B, heads), dtype=torch.bool, device=pooled_score.device
    )
    expected_value_detail = None
    head_value_detail = None
    head_local_affinity = None
    head_protection_score = None
    if head_protection == "value_detail":
        detail = value_detail.float().nan_to_num(
            nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        expected_value_detail = torch.matmul(
            probability, detail.unsqueeze(-1)
        ).squeeze(-1)
        head_value_detail = expected_value_detail.mean(dim=-1)
        head_protection_score = head_value_detail
    elif head_protection == "locality":
        head_local_affinity = (
            probability
            * local_region.view(1, 1, query_blocks, key_blocks)
        ).sum(dim=-1).mean(dim=-1)
        head_protection_score = head_local_affinity
    if head_protection_score is not None and protected_head_fraction > 0:
        protected_count = min(
            heads - 1,
            max(1, math.ceil(heads * protected_head_fraction)),
        )
        protected_indices = head_protection_score.topk(
            protected_count, dim=-1
        ).indices
        protected_head.scatter_(-1, protected_indices, True)

    delta = min(
        max(0, round(base_budget * redistribution_fraction)),
        base_budget - 1,
        key_blocks - base_budget,
    )
    low_budget = base_budget - delta
    high_budget = base_budget + delta
    adaptive_count = fixed_count.clone()
    active = torch.zeros(
        (B, heads), dtype=torch.bool, device=pooled_score.device
    )
    donor_retention = None
    donor_eligible = None
    donor_selected = None
    receiver_selected = None
    donor_cost = None
    receiver_gain = None
    receiver_gain_fraction = None
    half = query_blocks // 2
    if delta > 0 and half > 0:
        candidate_count = fixed_count.clone()
        if donor_retention_threshold > 0:
            low_count = torch.full_like(fixed_count, low_budget)
            low_mask = _select_variable_topk(selection_score, low_count)
            fixed_mass = probability.masked_fill(~fixed_mask, 0.0).sum(dim=-1)
            low_mass = probability.masked_fill(~low_mask, 0.0).sum(dim=-1)
            donor_retention = low_mass / fixed_mass.clamp_min(eps)
            donor_eligible = donor_retention >= donor_retention_threshold
            if donor_exchange_scope == "layer":
                if utility_certification:
                    low_probability = probability.masked_fill(~low_mask, 0.0)
                    low_probability = low_probability / low_probability.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(eps)
                    low_preview = torch.matmul(
                        low_probability.to(pooled_value.dtype), pooled_value
                    ).float()
                    low_error = (low_preview - coarse_dense).norm(dim=-1) / (
                        coarse_dense.norm(dim=-1).clamp_min(eps)
                    )
                    donor_cost = (low_error - preview_error).clamp_min(0.0)

                    high_count = torch.full_like(fixed_count, high_budget)
                    high_mask = _select_variable_topk(
                        selection_score, high_count
                    )
                    high_probability = probability.masked_fill(
                        ~high_mask, 0.0
                    )
                    high_probability = high_probability / high_probability.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(eps)
                    high_preview = torch.matmul(
                        high_probability.to(pooled_value.dtype), pooled_value
                    ).float()
                    high_error = (high_preview - coarse_dense).norm(dim=-1) / (
                        coarse_dense.norm(dim=-1).clamp_min(eps)
                    )
                    receiver_gain = (
                        preview_error - high_error
                    ).clamp_min(0.0)
                    receiver_gain_fraction = receiver_gain / preview_error.clamp_min(
                        eps
                    )

                row_count = heads * query_blocks
                max_exchange = row_count // 2
                unprotected = (~protected_head).unsqueeze(-1).expand_as(
                    donor_eligible
                )
                donor_pool = donor_eligible & unprotected
                receiver_pool = (
                    (preview_error >= activation_threshold)
                    & ~donor_eligible
                    & unprotected
                )
                receiver_pool &= (
                    layer_confidence >= layer_confidence_threshold
                ).view(B, 1, 1)
                if utility_certification:
                    receiver_pool &= (
                        receiver_gain_fraction >= receiver_gain_threshold
                    )
                    donor_priority = -donor_cost
                    receiver_priority = receiver_gain
                else:
                    donor_priority = donor_retention
                    receiver_priority = preview_error

                if head_exchange_cap_fraction < 1.0:
                    head_cap = min(
                        query_blocks,
                        math.ceil(
                            query_blocks * head_exchange_cap_fraction
                        ),
                    )
                    if head_cap == 0:
                        donor_pool = torch.zeros_like(donor_pool)
                        receiver_pool = torch.zeros_like(receiver_pool)
                    else:
                        donor_cap_indices = donor_priority.masked_fill(
                            ~donor_pool, float("-inf")
                        ).topk(head_cap, dim=-1).indices
                        donor_cap_valid = (
                            torch.arange(
                                head_cap, device=pooled_score.device
                            ).view(1, 1, head_cap)
                            < donor_pool.sum(dim=-1).clamp(max=head_cap).unsqueeze(-1)
                        )
                        capped_donor_pool = torch.zeros_like(donor_pool)
                        capped_donor_pool.scatter_(
                            -1, donor_cap_indices, donor_cap_valid
                        )
                        donor_pool = capped_donor_pool

                        receiver_cap_indices = receiver_priority.masked_fill(
                            ~receiver_pool, float("-inf")
                        ).topk(head_cap, dim=-1).indices
                        receiver_cap_valid = (
                            torch.arange(
                                head_cap, device=pooled_score.device
                            ).view(1, 1, head_cap)
                            < receiver_pool.sum(dim=-1).clamp(
                                max=head_cap
                            ).unsqueeze(-1)
                        )
                        capped_receiver_pool = torch.zeros_like(receiver_pool)
                        capped_receiver_pool.scatter_(
                            -1, receiver_cap_indices, receiver_cap_valid
                        )
                        receiver_pool = capped_receiver_pool

                flat_donor_pool = donor_pool.reshape(B, row_count)
                flat_receiver_pool = receiver_pool.reshape(B, row_count)
                exchange_count = torch.minimum(
                    flat_donor_pool.sum(dim=-1),
                    flat_receiver_pool.sum(dim=-1),
                ).clamp(max=max_exchange)
                flat_donor_priority = donor_priority.reshape(B, row_count)
                flat_donor_priority = flat_donor_priority.masked_fill(
                    ~flat_donor_pool, float("-inf")
                )
                donor_indices = flat_donor_priority.topk(
                    max_exchange,
                    dim=-1,
                    largest=True,
                    sorted=utility_certification,
                ).indices
                exchange_valid = (
                    torch.arange(max_exchange, device=pooled_score.device)
                    .view(1, max_exchange)
                    < exchange_count.unsqueeze(-1)
                )
                flat_receiver_priority = receiver_priority.reshape(B, row_count)
                flat_receiver_priority = flat_receiver_priority.masked_fill(
                    ~flat_receiver_pool, float("-inf")
                )
                receiver_indices = flat_receiver_priority.topk(
                    max_exchange,
                    dim=-1,
                    largest=True,
                    sorted=utility_certification,
                ).indices
                if exchange_gain_cost_ratio > 0:
                    paired_cost = torch.gather(
                        donor_cost.reshape(B, row_count), -1, donor_indices
                    )
                    paired_gain = torch.gather(
                        receiver_gain.reshape(B, row_count),
                        -1,
                        receiver_indices,
                    )
                    exchange_valid &= paired_gain >= (
                        exchange_gain_cost_ratio * paired_cost
                    )

                flat_donor_selected = torch.zeros_like(flat_donor_pool)
                flat_donor_selected.scatter_(
                    -1, donor_indices, exchange_valid
                )
                flat_receiver_selected = torch.zeros_like(flat_receiver_pool)
                flat_receiver_selected.scatter_(
                    -1, receiver_indices, exchange_valid
                )
                donor_selected = flat_donor_selected.reshape(
                    B, heads, query_blocks
                )
                receiver_selected = flat_receiver_selected.reshape(
                    B, heads, query_blocks
                )
            else:
                donor_count = donor_eligible.sum(dim=-1).clamp(max=half)
                donor_priority = donor_retention.masked_fill(
                    ~donor_eligible, float("-inf")
                )
                donor_indices = donor_priority.topk(
                    half, dim=-1, largest=True, sorted=False
                ).indices
                donor_valid = (
                    torch.arange(half, device=pooled_score.device)
                    .view(1, 1, half)
                    < donor_count.unsqueeze(-1)
                )
                donor_selected = torch.zeros_like(donor_eligible)
                donor_selected.scatter_(-1, donor_indices, donor_valid)

                receiver_priority = preview_error.masked_fill(
                    donor_selected, float("-inf")
                )
                receiver_indices = receiver_priority.topk(
                    half, dim=-1, largest=True, sorted=False
                ).indices
                receiver_selected = torch.zeros_like(donor_eligible)
                receiver_selected.scatter_(-1, receiver_indices, donor_valid)
            candidate_count = torch.where(
                donor_selected,
                torch.full_like(candidate_count, low_budget),
                candidate_count,
            )
            candidate_count = torch.where(
                receiver_selected,
                torch.full_like(candidate_count, high_budget),
                candidate_count,
            )
        else:
            order = preview_error.argsort(dim=-1)
            low_indices = order[..., :half]
            high_indices = order[..., -half:]
            candidate_count.scatter_(
                -1, low_indices, torch.full_like(low_indices, low_budget)
            )
            candidate_count.scatter_(
                -1, high_indices, torch.full_like(high_indices, high_budget)
            )
        if donor_exchange_scope == "layer":
            adaptive_count = candidate_count
            active = (donor_selected | receiver_selected).any(dim=-1)
        else:
            active = preview_error.mean(dim=-1) >= activation_threshold
            active &= ~protected_head
            adaptive_count = torch.where(
                active.unsqueeze(-1), candidate_count, fixed_count
            )

    if local_fraction > 0:
        requested_local = torch.ceil(
            adaptive_count.float() * local_fraction
        ).long()
        local_score = selection_score.masked_fill(
            ~local_region.view(1, 1, query_blocks, key_blocks),
            float("-inf"),
        )
        available_local = torch.isfinite(local_score).sum(dim=-1)
        local_count = torch.minimum(requested_local, available_local)
        block_mask = _select_variable_topk_in_region(
            selection_score,
            local_count,
            local_region,
        )
        remaining_count = adaptive_count - block_mask.sum(dim=-1)
        remaining_score = selection_score.masked_fill(
            block_mask, float("-inf")
        )
        block_mask |= _select_variable_topk(
            remaining_score, remaining_count
        )
    else:
        block_mask = _select_variable_topk(selection_score, adaptive_count)
    if stats_store is not None:
        stats_store.update(
            {
                "preview_adaptive_patch_sparsity": (
                    1.0 - block_mask.float().mean()
                ).detach(),
                "preview_adaptive_error_mean": preview_error.mean().detach(),
                "preview_adaptive_error_std": preview_error.std(
                    unbiased=False
                ).detach(),
                "preview_adaptive_active_fraction": (
                    active.float().mean().detach()
                ),
                "preview_adaptive_protected_head_fraction": (
                    protected_head.float().mean().detach()
                ),
                "preview_adaptive_value_detail_mean": (
                    expected_value_detail.mean().detach()
                    if expected_value_detail is not None
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_protected_value_detail": (
                    head_value_detail.masked_select(protected_head).mean().detach()
                    if head_value_detail is not None and protected_head.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_protected_local_affinity": (
                    head_local_affinity.masked_select(protected_head)
                    .mean()
                    .detach()
                    if head_local_affinity is not None and protected_head.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_donor_eligible_fraction": (
                    donor_eligible.float().mean().detach()
                    if donor_eligible is not None
                    else torch.tensor(
                        0.5 if delta > 0 else 0.0,
                        device=pooled_score.device,
                    )
                ),
                "preview_adaptive_exchange_fraction": (
                    (
                        donor_selected
                        & active.unsqueeze(-1)
                    ).float().mean().detach()
                    if donor_selected is not None
                    else (
                        active.float().mean() * (half / query_blocks)
                    ).detach()
                ),
                "preview_adaptive_donor_retention": (
                    donor_retention.masked_select(donor_selected)
                    .mean()
                    .detach()
                    if donor_retention is not None and donor_selected.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_donor_retention_threshold": torch.tensor(
                    donor_retention_threshold,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
                "preview_adaptive_donor_cost": (
                    donor_cost.masked_select(donor_selected).mean().detach()
                    if donor_cost is not None and donor_selected.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_receiver_gain": (
                    receiver_gain.masked_select(receiver_selected).mean().detach()
                    if receiver_gain is not None and receiver_selected.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_receiver_gain_fraction": (
                    receiver_gain_fraction.masked_select(
                        receiver_selected
                    ).mean().detach()
                    if receiver_gain_fraction is not None
                    and receiver_selected.any()
                    else torch.tensor(
                        float("nan"), device=pooled_score.device
                    )
                ),
                "preview_adaptive_receiver_gain_threshold": torch.tensor(
                    receiver_gain_threshold,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
                "preview_adaptive_exchange_gain_cost_ratio": torch.tensor(
                    exchange_gain_cost_ratio,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
                "preview_adaptive_head_exchange_cap_fraction": torch.tensor(
                    head_exchange_cap_fraction,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
                "preview_adaptive_layer_confidence": (
                    layer_confidence.mean().detach()
                ),
                "preview_adaptive_layer_confidence_threshold": torch.tensor(
                    layer_confidence_threshold,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
                "preview_adaptive_max_head_exchange_fraction": (
                    torch.maximum(
                        donor_selected.sum(dim=-1),
                        receiver_selected.sum(dim=-1),
                    ).float().max()
                    / max(query_blocks, 1)
                    if donor_selected is not None
                    else torch.tensor(0.0, device=pooled_score.device)
                ).detach(),
                "preview_adaptive_donor_fraction_per_head": (
                    donor_selected.float().mean(dim=-1).detach()
                    if donor_selected is not None
                    else torch.zeros(
                        (B, heads), device=pooled_score.device
                    )
                ),
                "preview_adaptive_receiver_fraction_per_head": (
                    receiver_selected.float().mean(dim=-1).detach()
                    if receiver_selected is not None
                    else torch.zeros(
                        (B, heads), device=pooled_score.device
                    )
                ),
                "preview_adaptive_net_blocks_per_head": (
                    (
                        receiver_selected.sum(dim=-1)
                        - donor_selected.sum(dim=-1)
                    ).float()
                    * delta
                    if donor_selected is not None
                    else torch.zeros(
                        (B, heads), device=pooled_score.device
                    )
                ).detach(),
                "preview_adaptive_layer_exchange": torch.tensor(
                    float(donor_exchange_scope == "layer"),
                    device=pooled_score.device,
                ),
                "preview_adaptive_low_budget": torch.tensor(
                    low_budget, device=pooled_score.device, dtype=torch.float32
                ),
                "preview_adaptive_high_budget": torch.tensor(
                    high_budget, device=pooled_score.device, dtype=torch.float32
                ),
                "preview_adaptive_mean_budget": adaptive_count.float()
                .mean()
                .detach(),
                "preview_adaptive_budget_error": (
                    (
                        adaptive_count.sum(dim=(1, 2))
                        - base_budget * heads * query_blocks
                    )
                    if donor_exchange_scope == "layer"
                    else (
                        adaptive_count.sum(dim=-1)
                        - base_budget * query_blocks
                    )
                ).abs().max().float().detach(),
                "preview_adaptive_head_budget_std": (
                    adaptive_count.sum(dim=-1)
                    .float()
                    .std(dim=-1, unbiased=False)
                    .mean()
                    .detach()
                ),
                "preview_adaptive_local_fraction": (
                    (
                        block_mask
                        & local_region.view(
                            1, 1, query_blocks, key_blocks
                        )
                    )
                    .sum()
                    .float()
                    / block_mask.sum().clamp_min(1).float()
                    if local_fraction > 0
                    else torch.tensor(0.0, device=pooled_score.device)
                ).detach(),
                "preview_adaptive_geometry_weight": torch.tensor(
                    geometry_weight,
                    device=pooled_score.device,
                    dtype=torch.float32,
                ),
            }
        )
    return block_mask


def _block_segments(
    num_frames: int,
    tokens_per_frame: int,
    block_size: int,
    num_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Represent each flattened block as at most two frame-local segments."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if block_size > tokens_per_frame:
        raise ValueError(
            "block_size must not exceed tokens_per_frame for two-segment routing"
        )

    total_tokens = num_frames * tokens_per_frame
    block_idx = torch.arange(num_blocks, device=device, dtype=torch.long)
    token_start = block_idx * block_size
    token_end = torch.clamp(token_start + block_size, max=total_tokens)

    first_frame = token_start // tokens_per_frame
    last_frame = (token_end - 1).clamp_min(0) // tokens_per_frame
    crosses_frame = last_frame != first_frame

    frames = torch.stack((first_frame, last_frame), dim=-1)
    starts = torch.stack(
        (
            token_start - first_frame * tokens_per_frame,
            torch.zeros_like(token_start),
        ),
        dim=-1,
    )
    ends = torch.stack(
        (
            torch.where(
                crosses_frame,
                torch.full_like(token_end, tokens_per_frame),
                token_end - first_frame * tokens_per_frame,
            ),
            token_end - last_frame * tokens_per_frame,
        ),
        dim=-1,
    )
    valid = torch.stack((torch.ones_like(crosses_frame), crosses_frame), dim=-1)
    return frames, starts, ends, valid


def _block_min_frame_distance(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int,
    k_block_size: int,
    q_blocks: int,
    k_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    """Minimum frame distance between flattened Q/K blocks."""
    q_frames, _, _, q_valid = _block_segments(
        num_frames, tokens_per_frame, q_block_size, q_blocks, device
    )
    k_frames, _, _, k_valid = _block_segments(
        num_frames, tokens_per_frame, k_block_size, k_blocks, device
    )
    frame_distance = torch.full(
        (q_blocks, k_blocks),
        fill_value=num_frames,
        dtype=torch.long,
        device=device,
    )
    for q_segment in range(2):
        for k_segment in range(2):
            valid_pair = q_valid[:, q_segment, None] & k_valid[None, :, k_segment]
            distance = (
                q_frames[:, q_segment, None] - k_frames[None, :, k_segment]
            ).abs()
            frame_distance = torch.where(
                valid_pair, torch.minimum(frame_distance, distance), frame_distance
            )
    return frame_distance


def _record_importance_block_analysis(
    stats_store: dict | None,
    final_map: torch.Tensor,
    orig_k_blocks: int,
    num_frames: int | None,
    tokens_per_frame: int | None,
    q_block_size: int,
    k_block_size: int,
    decay_factor: float,
    dense_neighbor: int,
    has_mixed_last_block: bool,
) -> None:
    """Store scalar diagnostics for the plain importance-only block mask."""
    if stats_store is None or num_frames is None or tokens_per_frame is None:
        return
    if orig_k_blocks <= 0:
        return

    selected = final_map[..., :orig_k_blocks]
    if selected.numel() == 0:
        return

    B, nh, q_blocks, k_blocks = selected.shape
    device = selected.device
    selected_f = selected.float()
    selected_total = selected_f.sum().clamp_min(1.0)
    frame_distance = _block_min_frame_distance(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        device=device,
    )
    distance = frame_distance.view(1, 1, q_blocks, k_blocks)

    def selected_fraction(mask: torch.Tensor) -> torch.Tensor:
        return (selected_f * mask.view(1, 1, q_blocks, k_blocks).float()).sum() / selected_total

    total_tokens = num_frames * tokens_per_frame
    k_start = torch.arange(k_blocks, device=device, dtype=torch.long) * k_block_size
    k_end = torch.clamp(k_start + k_block_size, max=total_tokens)
    k_cross_frame = (k_start // tokens_per_frame) != (
        (k_end - 1).clamp_min(0) // tokens_per_frame
    )

    key_centers = torch.clamp(k_start + k_block_size // 2, max=total_tokens - 1)
    key_frames = torch.clamp(key_centers // tokens_per_frame, max=num_frames - 1)
    frame_index = key_frames.view(1, 1, 1, k_blocks).expand(B, nh, q_blocks, k_blocks)
    frame_counts = torch.zeros(
        (B, nh, q_blocks, num_frames),
        dtype=selected_f.dtype,
        device=device,
    )
    frame_counts.scatter_add_(-1, frame_index, selected_f)
    row_total = frame_counts.sum(dim=-1)
    valid_rows = row_total > 0
    valid_rows_f = valid_rows.float()
    valid_den = valid_rows_f.sum().clamp_min(1.0)
    prob = frame_counts / row_total.clamp_min(1.0).unsqueeze(-1)
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1)
    entropy = entropy / math.log(max(num_frames, 2))
    unique_frames = (frame_counts > 0).sum(dim=-1).float()

    radial_mask = _get_cached_radial_block_mask(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        decay_factor=decay_factor,
        dense_neighbor=dense_neighbor,
        device=device,
    )

    stats_store.update(
        {
            "importance_patch_sparsity": 1.0 - selected_f.mean().detach(),
            "importance_selected_same_frame_fraction": selected_fraction(frame_distance == 0).detach(),
            "importance_selected_adjacent_frame_fraction": selected_fraction(frame_distance == 1).detach(),
            "importance_selected_near_2_4_fraction": selected_fraction(
                (frame_distance >= 2) & (frame_distance <= 4)
            ).detach(),
            "importance_selected_mid_5_12_fraction": selected_fraction(
                (frame_distance >= 5) & (frame_distance <= 12)
            ).detach(),
            "importance_selected_far_gt12_fraction": selected_fraction(frame_distance > 12).detach(),
            "importance_selected_mean_frame_distance": (
                selected_f * distance.float()
            ).sum().div(selected_total).detach(),
            "importance_key_frame_entropy": (
                entropy * valid_rows_f
            ).sum().div(valid_den).detach(),
            "importance_unique_key_frames_per_query": (
                unique_frames * valid_rows_f
            ).sum().div(valid_den).detach(),
            "importance_key_frame_coverage_fraction": (
                unique_frames * valid_rows_f
            ).sum().div(valid_den * max(num_frames, 1)).detach(),
            "importance_selected_cross_frame_k_fraction": (
                selected_f * k_cross_frame.view(1, 1, 1, k_blocks).float()
            ).sum().div(selected_total).detach(),
            "importance_selected_radial_fraction": (
                selected_f * radial_mask.view(1, 1, q_blocks, k_blocks).float()
            ).sum().div(selected_total).detach(),
            "importance_radial_available_fraction": radial_mask.float().mean().detach(),
        }
    )
    if has_mixed_last_block:
        stats_store["importance_selected_special_mixed_k_fraction"] = (
            selected_f[..., -1].sum() / selected_total
        ).detach()
    else:
        stats_store["importance_selected_special_mixed_k_fraction"] = torch.zeros(
            (), device=device
        )


def _temporal_geometry_weight(
    frame_distance: torch.Tensor,
    core_frame_radius: int,
    transition_frame_radius: int,
    geometry_decay: str,
    decay_gamma: float,
) -> torch.Tensor:
    """Return a continuous Core-Transition-Far temporal prior."""
    if geometry_decay not in {"linear", "exponential"}:
        raise ValueError(
            "geometry_decay must be 'linear' or 'exponential', "
            f"got {geometry_decay!r}"
        )

    weight = torch.zeros_like(frame_distance, dtype=torch.float32)
    core = frame_distance <= core_frame_radius
    weight[core] = 1.0

    transition = (frame_distance > core_frame_radius) & (
        frame_distance <= transition_frame_radius
    )
    if transition.any():
        transition_distance = frame_distance[transition] - core_frame_radius
        if geometry_decay == "linear":
            span = max(transition_frame_radius - core_frame_radius, 1)
            weight[transition] = 1.0 - transition_distance.float() / span
        elif geometry_decay == "exponential":
            weight[transition] = torch.exp(
                -decay_gamma * transition_distance.float()
            )
    return weight


def build_soft_geometry_prior(
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int,
    k_block_size: int,
    q_blocks: int,
    k_blocks: int,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build block-pair geometry priors without assigning cross-frame blocks to one frame.

    Each block is split into at most two frame-local token intervals. The prior
    for a Q/K block pair is the strongest valid segment-pair prior.
    """
    if num_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError("num_frames and tokens_per_frame must be positive")
    if core_frame_radius < 0:
        raise ValueError("core_frame_radius must be non-negative")
    if transition_frame_radius < core_frame_radius:
        raise ValueError(
            "transition_frame_radius must be >= core_frame_radius"
        )
    if decay_gamma < 0:
        raise ValueError("decay_gamma must be non-negative")

    normalized_device = torch.device(device or "cpu")
    sigma = float(
        max(q_block_size, k_block_size)
        if geometry_sigma is None
        else geometry_sigma
    )
    if sigma <= 0:
        raise ValueError("geometry_sigma must be positive")

    cache_key = (
        num_frames,
        tokens_per_frame,
        q_block_size,
        k_block_size,
        q_blocks,
        k_blocks,
        core_frame_radius,
        transition_frame_radius,
        geometry_decay,
        float(decay_gamma),
        sigma,
        normalized_device.type,
        normalized_device.index,
    )
    cached = _SOFT_GEOMETRY_CACHE.get(cache_key)
    if cached is not None:
        return cached

    q_frames, q_starts, q_ends, q_valid = _block_segments(
        num_frames, tokens_per_frame, q_block_size, q_blocks, normalized_device
    )
    k_frames, k_starts, k_ends, k_valid = _block_segments(
        num_frames, tokens_per_frame, k_block_size, k_blocks, normalized_device
    )

    prior = torch.zeros(
        (q_blocks, k_blocks), dtype=torch.float32, device=normalized_device
    )
    min_frame_distance = torch.full(
        (q_blocks, k_blocks),
        fill_value=num_frames,
        dtype=torch.long,
        device=normalized_device,
    )

    for q_segment in range(2):
        for k_segment in range(2):
            valid_pair = q_valid[:, q_segment, None] & k_valid[None, :, k_segment]
            frame_distance = (
                q_frames[:, q_segment, None] - k_frames[None, :, k_segment]
            ).abs()
            min_frame_distance = torch.where(
                valid_pair,
                torch.minimum(min_frame_distance, frame_distance),
                min_frame_distance,
            )

            temporal_weight = _temporal_geometry_weight(
                frame_distance,
                core_frame_radius,
                transition_frame_radius,
                geometry_decay,
                decay_gamma,
            )
            interval_distance = torch.maximum(
                torch.maximum(
                    k_starts[None, :, k_segment] - q_ends[:, q_segment, None],
                    q_starts[:, q_segment, None] - k_ends[None, :, k_segment],
                ),
                torch.zeros((), dtype=torch.long, device=normalized_device),
            )
            spatial_weight = torch.exp(
                -torch.square(interval_distance.float() / sigma)
            )
            segment_prior = temporal_weight * spatial_weight
            segment_prior.masked_fill_(~valid_pair, 0.0)
            prior = torch.maximum(prior, segment_prior)

    if len(_SOFT_GEOMETRY_CACHE) >= _SOFT_GEOMETRY_CACHE_SIZE:
        _SOFT_GEOMETRY_CACHE.pop(next(iter(_SOFT_GEOMETRY_CACHE)))
    _SOFT_GEOMETRY_CACHE[cache_key] = (prior, min_frame_distance)
    return prior, min_frame_distance


def _frame_normalize_scores(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    k_block_size: int,
    eps: float,
) -> torch.Tensor:
    """Normalize content mass per key frame using each K block's center frame."""
    k_blocks = pooled_score.shape[-1]
    centers = (
        torch.arange(k_blocks, device=pooled_score.device, dtype=torch.long)
        * k_block_size
        + k_block_size // 2
    )
    key_frames = torch.clamp(centers // tokens_per_frame, max=num_frames - 1)
    frame_index = key_frames.view(1, 1, 1, k_blocks).expand_as(pooled_score)
    frame_sums = torch.zeros(
        (*pooled_score.shape[:-1], num_frames),
        dtype=pooled_score.dtype,
        device=pooled_score.device,
    )
    frame_sums.scatter_add_(-1, frame_index, pooled_score)
    return pooled_score / frame_sums.gather(-1, frame_index).clamp_min(eps)


def _distance_calibrate_scores(
    pooled_score: torch.Tensor,
    frame_distance: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize importance by the mean score at each frame distance."""
    B, nh, q_blocks, k_blocks = pooled_score.shape
    num_distance_bins = int(frame_distance.max().item()) + 1

    distance_idx = frame_distance.view(1, 1, q_blocks, k_blocks)
    distance_idx = distance_idx.expand(B, nh, -1, -1)
    distance_sums = torch.zeros(
        (B, nh, q_blocks, num_distance_bins),
        dtype=pooled_score.dtype,
        device=pooled_score.device,
    )
    distance_counts = torch.zeros_like(distance_sums)
    distance_sums.scatter_add_(-1, distance_idx, pooled_score)
    distance_counts.scatter_add_(-1, distance_idx, torch.ones_like(pooled_score))

    expected = distance_sums.gather(-1, distance_idx)
    expected = expected / distance_counts.gather(-1, distance_idx).clamp_min(1.0)
    return pooled_score / expected.clamp_min(eps)


def _distance_calibrate_scores_reference(
    pooled_score: torch.Tensor,
    frame_distance: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Slow reference implementation used by unit tests."""
    q_blocks, k_blocks = frame_distance.shape
    distance = frame_distance.view(1, 1, q_blocks, k_blocks)
    calibrated = torch.zeros_like(pooled_score)

    for dist in torch.unique(frame_distance):
        mask = distance == dist
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
        expected = (pooled_score * mask).sum(dim=-1, keepdim=True) / count
        calibrated = torch.where(mask, pooled_score / expected.clamp_min(eps), calibrated)

    return calibrated


def _normalized_entropy(pooled_score: torch.Tensor, eps: float) -> torch.Tensor:
    """Return row-wise normalized entropy in [0, 1] for each query block."""
    denom = pooled_score.sum(dim=-1, keepdim=True).clamp_min(eps)
    prob = pooled_score / denom
    entropy = -(prob * prob.clamp_min(eps).log()).sum(dim=-1, keepdim=True)
    return entropy / math.log(max(pooled_score.shape[-1], 2))


def get_soft_geometry_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sparse_ratio: float = 0.7,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 1.0,
    routing_bias: torch.Tensor | None = None,
    routing_bias_weight: float = 0.0,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    head_adaptive_geometry: bool = False,
    force_last_block: bool = False,
    protect_reference_frame: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Select a strict Top-K mask using content importance plus geometry prior."""
    if not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError(f"sparse_ratio must be in [0, 1], got {sparse_ratio}")
    if geometry_weight < 0:
        raise ValueError("geometry_weight must be non-negative")
    if routing_bias_weight < 0:
        raise ValueError("routing_bias_weight must be non-negative")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    geometry_prior, frame_distance = build_soft_geometry_prior(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        core_frame_radius=core_frame_radius,
        transition_frame_radius=transition_frame_radius,
        geometry_decay=geometry_decay,
        decay_gamma=decay_gamma,
        geometry_sigma=geometry_sigma,
        device=pooled_score.device,
    )

    content_score = pooled_score.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    content_score = content_score.clamp_min(0.0)
    if frame_normalize_importance:
        content_score = _frame_normalize_scores(
            content_score,
            num_frames,
            tokens_per_frame,
            k_block_size,
            eps,
        )
    entropy_source = content_score
    if distance_calibrate_importance:
        content_score = _distance_calibrate_scores(content_score, frame_distance, eps)

    if entropy_adaptive_geometry and head_adaptive_geometry:
        raise ValueError(
            "entropy_adaptive_geometry and head_adaptive_geometry are mutually exclusive"
        )

    effective_geometry_weight = geometry_weight
    entropy = None
    head_geometry_gate = None
    if entropy_adaptive_geometry:
        entropy = _normalized_entropy(entropy_source, eps)
        effective_geometry_weight = geometry_weight * entropy
    elif head_adaptive_geometry:
        k_frame = _block_center_frame_indices(
            k_blocks,
            k_block_size,
            num_frames,
            tokens_per_frame,
            pooled_score.device,
        )
        head_frame_mass = torch.zeros(
            (B, nh, q_blocks, num_frames),
            dtype=content_score.dtype,
            device=content_score.device,
        )
        head_frame_mass.scatter_add_(
            -1,
            k_frame.view(1, 1, 1, -1).expand(B, nh, q_blocks, -1),
            content_score,
        )
        head_frame_prob = head_frame_mass / head_frame_mass.sum(
            dim=-1, keepdim=True
        ).clamp_min(eps)
        head_frame_entropy = -(
            head_frame_prob * head_frame_prob.clamp_min(eps).log()
        ).sum(dim=-1)
        head_frame_entropy = head_frame_entropy / math.log(max(num_frames, 2))
        head_geometry_gate = (1.0 - head_frame_entropy).clamp_min(0.0)
        head_geometry_gate = head_geometry_gate / head_geometry_gate.amax(
            dim=1, keepdim=True
        ).clamp_min(eps)
        head_geometry_gate = head_geometry_gate.square()
        effective_geometry_weight = (
            geometry_weight * head_geometry_gate.unsqueeze(-1)
        )

    routing_score = torch.log(content_score.clamp_min(eps))
    routing_score = routing_score + effective_geometry_weight * geometry_prior.view(
        1, 1, q_blocks, k_blocks
    )
    if routing_bias is not None and routing_bias_weight > 0:
        if routing_bias.shape[-2:] != (q_blocks, k_blocks):
            raise ValueError(
                "routing_bias must match pooled_score query/key block dimensions"
            )
        routing_score = routing_score + routing_bias_weight * routing_bias

    k_total = int(k_blocks * (1.0 - sparse_ratio))
    reference_key_mask = None
    if protect_reference_frame:
        total_tokens = num_frames * tokens_per_frame
        k_start = torch.arange(k_blocks, device=pooled_score.device) * k_block_size
        k_end = torch.clamp(k_start + k_block_size, max=total_tokens)
        reference_key_mask = (k_start // tokens_per_frame == 0) | (
            (k_end - 1).clamp_min(0) // tokens_per_frame == 0
        )
        k_total = max(k_total, int(reference_key_mask.sum().item()))
        routing_score = routing_score.clone()
        routing_score[..., reference_key_mask] = torch.inf
    if force_last_block and k_total > 0:
        if reference_key_mask is None:
            routing_score = routing_score.clone()
        routing_score[..., -1] = torch.inf

    if k_total <= 0:
        block_mask = torch.zeros_like(pooled_score, dtype=torch.bool)
    elif k_total >= k_blocks:
        block_mask = torch.ones_like(pooled_score, dtype=torch.bool)
    else:
        topk_indices = torch.topk(
            routing_score, k=k_total, dim=-1, largest=True, sorted=False
        ).indices
        block_mask = torch.zeros_like(pooled_score, dtype=torch.bool)
        block_mask.scatter_(-1, topk_indices, True)

    if stats_store is not None:
        selected = block_mask.float()
        selected_total = selected.sum().clamp_min(1.0)
        core = (frame_distance <= core_frame_radius).view(
            1, 1, q_blocks, k_blocks
        )
        transition = (
            (frame_distance > core_frame_radius)
            & (frame_distance <= transition_frame_radius)
        ).view(1, 1, q_blocks, k_blocks)
        far = (frame_distance > transition_frame_radius).view(
            1, 1, q_blocks, k_blocks
        )
        stats_store.update(
            {
                "soft_geometry_k_total": k_total,
                "soft_geometry_patch_sparsity": 1.0 - block_mask.float().mean(),
                "soft_geometry_core_fraction": (selected * core).sum()
                / selected_total,
                "soft_geometry_transition_fraction": (selected * transition).sum()
                / selected_total,
                "soft_geometry_far_fraction": (selected * far).sum()
                / selected_total,
                "soft_geometry_mean_prior": geometry_prior.mean(),
                "soft_geometry_mean_entropy": (
                    entropy.mean()
                    if entropy is not None
                    else torch.tensor(float("nan"), device=pooled_score.device)
                ),
                "soft_geometry_mean_geometry_weight": (
                    effective_geometry_weight.mean()
                    if torch.is_tensor(effective_geometry_weight)
                    else torch.tensor(
                        float(effective_geometry_weight),
                        device=pooled_score.device,
                    )
                ),
                "soft_geometry_head_gate_mean": (
                    head_geometry_gate.mean()
                    if head_geometry_gate is not None
                    else torch.tensor(float("nan"), device=pooled_score.device)
                ),
            }
        )
        if reference_key_mask is not None:
            stats_store["soft_geometry_reference_fraction"] = (
                selected
                * reference_key_mask.view(1, 1, 1, k_blocks).float()
            ).sum() / selected_total
    return block_mask


def _select_variable_topk(
    score: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    """Select a different number of keys for each batch/head/query row."""
    selected = torch.zeros_like(score, dtype=torch.bool)
    max_count = int(count.max().item())
    if max_count <= 0:
        return selected

    indices = torch.topk(
        score,
        k=max_count,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    positions = torch.arange(max_count, device=score.device).view(1, 1, 1, -1)
    valid = positions < count.unsqueeze(-1)
    selected.scatter_(-1, indices, valid)
    return selected


def _select_variable_topk_in_region(
    score: torch.Tensor,
    count: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """Select variable Top-K values after compacting a shared Q/K region."""
    B, nh, q_blocks, k_blocks = score.shape
    region = region.reshape(q_blocks, k_blocks)
    max_candidates = int(region.sum(dim=-1).max().item())
    if max_candidates <= 0 or int(count.max().item()) <= 0:
        return torch.zeros_like(score, dtype=torch.bool)

    candidate_indices = torch.topk(
        region.to(torch.uint8),
        k=max_candidates,
        dim=-1,
        largest=True,
        sorted=False,
    ).indices
    candidate_valid = region.gather(-1, candidate_indices)
    expanded_indices = candidate_indices.view(
        1, 1, q_blocks, max_candidates
    ).expand(B, nh, -1, -1)
    compact_score = score.gather(-1, expanded_indices)
    compact_score = compact_score.masked_fill(
        ~candidate_valid.view(1, 1, q_blocks, max_candidates),
        float("-inf"),
    )
    compact_selected = _select_variable_topk(compact_score, count)
    selected = torch.zeros_like(score, dtype=torch.bool)
    selected.scatter_(-1, expanded_indices, compact_selected)
    return selected


def get_role_adaptive_dual_path_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    sparse_ratio: float = 0.70,
    layer_idx: int = 0,
    num_layers: int = 24,
    local_frame_radius: int = 2,
    min_local_fraction: float = 0.10,
    max_local_fraction: float = 0.50,
    layer_schedule: str = "early_decay",
    geometry_weight: float = 0.5,
    context_geometry_weight: float = 0.0,
    context_schedule: str = "flat",
    context_gate: str = "none",
    context_alignment_threshold: float = 1.2,
    context_alignment_temperature: float = 0.1,
    context_core_frame_radius: int = 4,
    context_transition_frame_radius: int = 12,
    force_last_block: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Split a strict block budget between local support and global retrieval."""
    if not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError("sparse_ratio must be in [0, 1]")
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if not 0.0 <= min_local_fraction <= max_local_fraction <= 1.0:
        raise ValueError(
            "local fractions must satisfy 0 <= min <= max <= 1"
        )
    if num_layers < 1 or not 0 <= layer_idx < num_layers:
        raise ValueError("layer_idx must be in [0, num_layers)")
    if layer_schedule not in {"flat", "early_decay", "middle_peak"}:
        raise ValueError(
            "layer_schedule must be flat, early_decay, or middle_peak"
        )
    if geometry_weight < 0:
        raise ValueError("geometry_weight must be non-negative")
    if context_geometry_weight < 0:
        raise ValueError("context_geometry_weight must be non-negative")
    if context_schedule not in {"flat", "late_ramp", "late_only"}:
        raise ValueError(
            "context_schedule must be flat, late_ramp, or late_only"
        )
    if context_gate not in {"none", "disagreement"}:
        raise ValueError("context_gate must be none or disagreement")
    if context_alignment_temperature <= 0:
        raise ValueError("context_alignment_temperature must be positive")
    if context_core_frame_radius < 0:
        raise ValueError("context_core_frame_radius must be non-negative")
    if context_transition_frame_radius < context_core_frame_radius:
        raise ValueError(
            "context_transition_frame_radius must be >= context_core_frame_radius"
        )

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    content = pooled_score.float().nan_to_num(
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    log_content = torch.log(content.clamp_min(eps))

    frame_distance = _block_min_frame_distance(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        device=device,
    )
    local_region = (frame_distance <= local_frame_radius).view(
        1, 1, q_blocks, k_blocks
    )

    k_frame = _block_center_frame_indices(
        k_blocks,
        k_block_size,
        num_frames,
        tokens_per_frame,
        device,
    )
    frame_mass = torch.zeros(
        (B, nh, q_blocks, num_frames),
        dtype=content.dtype,
        device=device,
    )
    frame_mass.scatter_add_(
        -1,
        k_frame.view(1, 1, 1, -1).expand(B, nh, q_blocks, -1),
        content,
    )
    total_mass = frame_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
    frame_prob = frame_mass / total_mass
    frame_entropy = -(
        frame_prob * frame_prob.clamp_min(eps).log()
    ).sum(dim=-1)
    frame_entropy = frame_entropy / math.log(max(num_frames, 2))

    local_mass = (content * local_region).sum(dim=-1)
    local_affinity = local_mass / content.sum(dim=-1).clamp_min(eps)
    concentration = (1.0 - frame_entropy).clamp(0.0, 1.0)
    head_role = local_affinity * (0.5 + 0.5 * concentration)

    progress = layer_idx / max(num_layers - 1, 1)
    if layer_schedule == "flat":
        layer_gate = 0.5
    elif layer_schedule == "early_decay":
        layer_gate = 1.0 - progress
    else:
        layer_gate = math.sin(math.pi * progress)
    allocation_gate = 0.5 * head_role + 0.5 * layer_gate
    local_fraction = min_local_fraction + (
        max_local_fraction - min_local_fraction
    ) * allocation_gate

    budget = int(k_blocks * (1.0 - sparse_ratio))
    if budget <= 0:
        block_mask = torch.zeros_like(pooled_score, dtype=torch.bool)
        if stats_store is not None:
            stats_store["dual_path_patch_sparsity"] = torch.tensor(
                1.0, device=device
            )
        return block_mask
    if budget >= k_blocks:
        block_mask = torch.ones_like(pooled_score, dtype=torch.bool)
        if stats_store is not None:
            stats_store["dual_path_patch_sparsity"] = torch.tensor(
                0.0, device=device
            )
        return block_mask

    target_local = torch.round(local_fraction * budget).long().clamp(
        min=0, max=budget
    )
    selected = torch.zeros_like(pooled_score, dtype=torch.bool)
    if force_last_block:
        selected[..., -1] = True

    geometry_prior, _ = build_soft_geometry_prior(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        core_frame_radius=local_frame_radius,
        transition_frame_radius=local_frame_radius,
        geometry_decay="linear",
        decay_gamma=0.25,
        geometry_sigma=None,
        device=device,
    )
    local_score = log_content + geometry_weight * geometry_prior.view(
        1, 1, q_blocks, k_blocks
    )
    local_score = local_score.masked_fill(~local_region | selected, float("-inf"))
    selected_local = (selected & local_region).sum(dim=-1)
    available_local = torch.isfinite(local_score).sum(dim=-1)
    local_count = torch.minimum(
        (target_local - selected_local).clamp_min(0),
        available_local,
    )
    selected |= _select_variable_topk_in_region(
        local_score,
        local_count,
        local_region,
    )

    if context_schedule == "flat":
        context_layer_gate = 1.0
    elif context_schedule == "late_ramp":
        context_layer_gate = progress * progress
    else:
        late_start = 2.0 / 3.0
        context_layer_gate = max(
            0.0,
            (progress - late_start) / (1.0 - late_start),
        )
    effective_context_geometry_weight = (
        context_geometry_weight * context_layer_gate
    )

    context_score = log_content
    context_alignment = None
    context_gate_value = None
    routed_context_weight = effective_context_geometry_weight
    if effective_context_geometry_weight > 0:
        context_prior, _ = build_soft_geometry_prior(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            q_blocks=q_blocks,
            k_blocks=k_blocks,
            core_frame_radius=context_core_frame_radius,
            transition_frame_radius=context_transition_frame_radius,
            geometry_decay="linear",
            decay_gamma=0.25,
            geometry_sigma=None,
            device=device,
        )
        content_prob = content / content.sum(dim=-1, keepdim=True).clamp_min(eps)
        prior_view = context_prior.view(1, 1, q_blocks, k_blocks)
        context_alignment = (content_prob * prior_view).sum(dim=-1)
        context_alignment = context_alignment / prior_view.mean(
            dim=-1
        ).clamp_min(eps)
        if context_gate == "disagreement":
            context_gate_value = torch.sigmoid(
                (context_alignment_threshold - context_alignment)
                / context_alignment_temperature
            )
            routed_context_weight = (
                effective_context_geometry_weight
                * context_gate_value.unsqueeze(-1)
            )
        context_score = context_score + routed_context_weight * prior_view

    remaining = (budget - selected.sum(dim=-1)).clamp_min(0)
    global_score = context_score.masked_fill(
        selected | local_region, float("-inf")
    )
    available_global = torch.isfinite(global_score).sum(dim=-1)
    global_count = torch.minimum(remaining, available_global)
    selected |= _select_variable_topk(global_score, global_count)

    fallback_count = (budget - selected.sum(dim=-1)).clamp_min(0)
    fallback_score = context_score.masked_fill(selected, float("-inf"))
    selected |= _select_variable_topk(fallback_score, fallback_count)

    if stats_store is not None:
        selected_float = selected.float()
        selected_total = selected_float.sum().clamp_min(1.0)
        actual_local = (selected_float * local_region).sum() / selected_total
        stats_store.update(
            {
                "dual_path_patch_sparsity": 1.0 - selected_float.mean(),
                "dual_path_local_fraction": actual_local,
                "dual_path_remote_fraction": 1.0 - actual_local,
                "dual_path_requested_local_fraction": (
                    target_local.float() / budget
                ).mean(),
                "dual_path_local_blocks_per_query": (
                    selected & local_region
                ).sum(dim=-1).float().mean(),
                "dual_path_frame_entropy": frame_entropy.mean(),
                "dual_path_frame_entropy_head_std": frame_entropy.std(
                    dim=1, unbiased=False
                ).mean(),
                "dual_path_local_affinity": local_affinity.mean(),
                "dual_path_local_affinity_head_std": local_affinity.std(
                    dim=1, unbiased=False
                ).mean(),
                "dual_path_head_role": head_role.mean(),
                "dual_path_head_role_std": head_role.std(
                    dim=1, unbiased=False
                ).mean(),
                "dual_path_layer_gate": torch.tensor(layer_gate, device=device),
                "dual_path_context_geometry_weight": torch.tensor(
                    context_geometry_weight, device=device
                ),
                "dual_path_effective_context_geometry_weight": torch.tensor(
                    effective_context_geometry_weight, device=device
                ),
                "dual_path_routed_context_geometry_weight": (
                    routed_context_weight.mean()
                    if torch.is_tensor(routed_context_weight)
                    else torch.tensor(routed_context_weight, device=device)
                ),
                "dual_path_context_gate_mean": (
                    context_gate_value.mean()
                    if context_gate_value is not None
                    else torch.tensor(float("nan"), device=device)
                ),
                "dual_path_context_alignment": (
                    context_alignment.mean()
                    if context_alignment is not None
                    else torch.tensor(float("nan"), device=device)
                ),
                "dual_path_context_alignment_std": (
                    context_alignment.std(unbiased=False)
                    if context_alignment is not None
                    else torch.tensor(float("nan"), device=device)
                ),
                "dual_path_budget_error": (
                    selected.sum(dim=-1) - budget
                ).abs().float().max(),
            }
        )
    return selected


def _get_frame_key_table(
    num_frames: int,
    tokens_per_frame: int,
    k_block_size: int,
    k_blocks: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached frame-to-key-block indices and their padding mask."""
    cache_key = (
        num_frames,
        tokens_per_frame,
        k_block_size,
        k_blocks,
        device.type,
        device.index,
    )
    cached = _FRAME_KEY_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    total_tokens = num_frames * tokens_per_frame
    k_start = torch.arange(k_blocks, device=device) * k_block_size
    k_center = torch.clamp(k_start + k_block_size // 2, max=total_tokens - 1)
    key_frame = torch.clamp(k_center // tokens_per_frame, max=num_frames - 1)
    frame_counts = torch.bincount(key_frame, minlength=num_frames)
    max_blocks_per_frame = int(frame_counts.max().item())
    frame_offsets = torch.cumsum(frame_counts, dim=0) - frame_counts
    position_in_frame = torch.arange(k_blocks, device=device) - frame_offsets[key_frame]
    frame_key_table = torch.full(
        (num_frames, max_blocks_per_frame),
        k_blocks,
        dtype=torch.long,
        device=device,
    )
    frame_key_table[key_frame, position_in_frame] = torch.arange(
        k_blocks, device=device
    )
    padding_mask = frame_key_table == k_blocks
    safe_table = frame_key_table.clamp_max(k_blocks - 1)

    if len(_FRAME_KEY_TABLE_CACHE) >= _FRAME_KEY_TABLE_CACHE_SIZE:
        _FRAME_KEY_TABLE_CACHE.pop(next(iter(_FRAME_KEY_TABLE_CACHE)))
    _FRAME_KEY_TABLE_CACHE[cache_key] = (safe_table, padding_mask)
    return safe_table, padding_mask


def _select_strict_budget_with_frame_coverage(
    pooled_score: torch.Tensor,
    protected_mask: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    k_block_size: int,
    sparse_ratio: float,
    frame_coverage_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill a strict per-query budget while reserving one anchor per chosen frame."""
    if not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError("sparse_ratio must be in [0, 1]")
    if not 0.0 <= frame_coverage_ratio <= 1.0:
        raise ValueError("frame_coverage_ratio must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    protected = protected_mask.view(1, 1, q_blocks, k_blocks).expand(
        B, nh, -1, -1
    )
    selected = protected.clone()

    base_budget = int(k_blocks * (1.0 - sparse_ratio))
    protected_count = protected.sum(dim=-1)
    budget = torch.maximum(
        protected_count,
        torch.full_like(protected_count, base_budget),
    ).clamp_max(k_blocks)
    remaining = budget - protected_count
    if int(remaining.max().item()) <= 0:
        return selected, torch.zeros_like(remaining)

    score = pooled_score.float().nan_to_num(
        nan=float("-inf"), posinf=float("-inf"), neginf=float("-inf")
    )
    far_score = score.masked_fill(protected, float("-inf"))

    safe_table, padding_mask = _get_frame_key_table(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        k_block_size=k_block_size,
        k_blocks=k_blocks,
        device=device,
    )
    max_blocks_per_frame = safe_table.shape[-1]
    grouped_score = far_score[..., safe_table]
    grouped_score = grouped_score.masked_fill(
        padding_mask.view(1, 1, 1, num_frames, max_blocks_per_frame),
        float("-inf"),
    )
    frame_score, best_position = grouped_score.max(dim=-1)
    candidate_indices = safe_table.view(
        1, 1, 1, num_frames, max_blocks_per_frame
    ).expand(B, nh, q_blocks, -1, -1)
    frame_anchor_indices = candidate_indices.gather(
        -1, best_position.unsqueeze(-1)
    ).squeeze(-1)

    available_frames = torch.isfinite(frame_score).sum(dim=-1)
    target_frames = int(math.ceil(num_frames * frame_coverage_ratio))
    coverage_count = torch.minimum(
        remaining,
        torch.minimum(
            available_frames,
            torch.full_like(available_frames, target_frames),
        ),
    )
    max_coverage = int(coverage_count.max().item())
    anchor_mask = torch.zeros_like(selected)
    if max_coverage > 0:
        selected_frames = torch.topk(
            frame_score,
            k=max_coverage,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        anchor_indices = frame_anchor_indices.gather(-1, selected_frames)
        positions = torch.arange(max_coverage, device=device).view(1, 1, 1, -1)
        valid_anchor = positions < coverage_count.unsqueeze(-1)
        anchor_mask.scatter_(
            -1, anchor_indices.clamp_max(k_blocks - 1), valid_anchor
        )
        selected |= anchor_mask

    fill_count = (budget - selected.sum(dim=-1)).clamp_min(0)
    max_fill = int(fill_count.max().item())
    if max_fill > 0:
        fill_score = score.masked_fill(selected, float("-inf"))
        fill_indices = torch.topk(
            fill_score,
            k=max_fill,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        positions = torch.arange(max_fill, device=device).view(1, 1, 1, -1)
        valid_fill = positions < fill_count.unsqueeze(-1)
        fill_mask = torch.zeros_like(selected)
        fill_mask.scatter_(-1, fill_indices, valid_fill)
        selected |= fill_mask

    return selected, anchor_mask.sum(dim=-1)


def _block_center_frame_indices(
    block_count: int,
    block_size: int,
    num_frames: int,
    tokens_per_frame: int,
    device: torch.device,
) -> torch.Tensor:
    total_tokens = num_frames * tokens_per_frame
    block_start = torch.arange(block_count, device=device) * block_size
    block_center = torch.clamp(
        block_start + block_size // 2,
        max=total_tokens - 1,
    )
    return torch.clamp(block_center // tokens_per_frame, max=num_frames - 1)


def _compute_head_consensus_view_scores(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int,
    k_block_size: int,
    bidirectional: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate block probabilities into a compact query-view/key-view graph."""
    B, _, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    q_frame = _block_center_frame_indices(
        q_blocks,
        q_block_size,
        num_frames,
        tokens_per_frame,
        device,
    )
    k_frame = _block_center_frame_indices(
        k_blocks,
        k_block_size,
        num_frames,
        tokens_per_frame,
        device,
    )

    consensus = pooled_score.float().mean(dim=1).nan_to_num(
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    key_frame_mass = torch.zeros(
        (B, q_blocks, num_frames),
        dtype=consensus.dtype,
        device=device,
    )
    key_frame_mass.scatter_add_(
        -1,
        k_frame.view(1, 1, -1).expand(B, q_blocks, -1),
        consensus,
    )

    view_score = torch.zeros(
        (B, num_frames, num_frames),
        dtype=consensus.dtype,
        device=device,
    )
    view_score.scatter_add_(
        1,
        q_frame.view(1, -1, 1).expand(B, -1, num_frames),
        key_frame_mass,
    )
    q_frame_count = torch.bincount(q_frame, minlength=num_frames).clamp_min(1)
    view_score = view_score / q_frame_count.view(1, -1, 1)
    if bidirectional:
        view_score = 0.5 * (view_score + view_score.transpose(-1, -2))
    return view_score, q_frame, k_frame


def _select_persistent_view_graph(
    view_score: torch.Tensor,
    local_radius: int,
    remote_topk: int,
    remote_ratio: float,
    protect_reference_frame: bool,
) -> torch.Tensor:
    """Select a shared per-batch view graph from head-consensus scores."""
    B, num_frames, _ = view_score.shape
    device = view_score.device
    frame = torch.arange(num_frames, device=device)
    local_graph = (
        (frame[:, None] - frame[None, :]).abs() <= local_radius
    )
    graph = local_graph.view(1, num_frames, num_frames).expand(
        B, -1, -1
    ).clone()
    if protect_reference_frame:
        graph[:, :, 0] = True

    target_remote = max(remote_topk, int(math.ceil(num_frames * remote_ratio)))
    candidate_score = view_score.masked_fill(graph, float("-inf"))
    available = torch.isfinite(candidate_score).sum(dim=-1)
    selected_count = torch.minimum(
        available,
        torch.full_like(available, target_remote),
    )
    max_selected = int(selected_count.max().item())
    if max_selected > 0:
        selected_frames = torch.topk(
            candidate_score,
            k=max_selected,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        positions = torch.arange(max_selected, device=device).view(1, 1, -1)
        valid = positions < selected_count.unsqueeze(-1)
        graph.scatter_(-1, selected_frames, valid)
    return graph


def _strict_budget_with_view_graph(
    pooled_score: torch.Tensor,
    view_graph: torch.Tensor,
    q_frame: torch.Tensor,
    k_frame: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    k_block_size: int,
    sparse_ratio: float,
    force_frame_anchors: bool = False,
    single_pass: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match blocks inside selected view pairs while keeping a strict budget."""
    if not 0.0 <= sparse_ratio <= 1.0:
        raise ValueError("sparse_ratio must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    score = pooled_score.float().nan_to_num(
        nan=float("-inf"),
        posinf=float("-inf"),
        neginf=float("-inf"),
    )
    graph_for_query = view_graph[:, q_frame, :]
    allowed = graph_for_query[:, :, k_frame]
    allowed = allowed.unsqueeze(1).expand(B, nh, -1, -1)

    own_frame = (q_frame[:, None] == k_frame[None, :]).view(
        1, 1, q_blocks, k_blocks
    )
    base_budget = int(k_blocks * (1.0 - sparse_ratio))
    if not force_frame_anchors and single_pass:
        routing_score = score.masked_fill(~allowed, float("-inf")).clone()
        routing_score.masked_fill_(own_frame, torch.inf)
        topk_indices = torch.topk(
            routing_score,
            k=base_budget,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        selected = torch.zeros_like(allowed)
        selected.scatter_(-1, topk_indices, True)
        anchor_count = torch.zeros(
            (B, nh, q_blocks),
            dtype=torch.long,
            device=device,
        )
        return selected, anchor_count

    selected = own_frame.expand(B, nh, -1, -1).clone()
    selected_count = selected.sum(dim=-1)
    budget = torch.maximum(
        selected_count,
        torch.full_like(selected_count, base_budget),
    ).clamp_max(k_blocks)

    anchor_count = torch.zeros_like(selected_count)
    if force_frame_anchors:
        safe_table, padding_mask = _get_frame_key_table(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            k_block_size=k_block_size,
            k_blocks=k_blocks,
            device=device,
        )
        max_blocks_per_frame = safe_table.shape[-1]
        anchor_score = score.masked_fill(~allowed | selected, float("-inf"))
        grouped_score = anchor_score[..., safe_table]
        grouped_score = grouped_score.masked_fill(
            padding_mask.view(1, 1, 1, num_frames, max_blocks_per_frame),
            float("-inf"),
        )
        frame_score, best_position = grouped_score.max(dim=-1)
        candidate_indices = safe_table.view(
            1, 1, 1, num_frames, max_blocks_per_frame
        ).expand(B, nh, q_blocks, -1, -1)
        frame_anchor = candidate_indices.gather(
            -1,
            best_position.unsqueeze(-1),
        ).squeeze(-1)

        remaining = budget - selected_count
        available_frames = torch.isfinite(frame_score).sum(dim=-1)
        anchor_count = torch.minimum(remaining, available_frames)
        max_anchors = int(anchor_count.max().item())
        if max_anchors > 0:
            anchor_frames = torch.topk(
                frame_score,
                k=max_anchors,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            anchor_indices = frame_anchor.gather(-1, anchor_frames)
            positions = torch.arange(max_anchors, device=device).view(
                1, 1, 1, -1
            )
            valid_anchor = positions < anchor_count.unsqueeze(-1)
            anchor_mask = torch.zeros_like(selected)
            anchor_mask.scatter_(-1, anchor_indices, valid_anchor)
            selected |= anchor_mask

    fill_count = (budget - selected.sum(dim=-1)).clamp_min(0)
    max_fill = int(fill_count.max().item())
    if max_fill > 0:
        fill_score = score.masked_fill(~allowed | selected, float("-inf"))
        available_blocks = torch.isfinite(fill_score).sum(dim=-1)
        graph_fill_count = torch.minimum(fill_count, available_blocks)
        graph_max_fill = int(graph_fill_count.max().item())
        if graph_max_fill > 0:
            fill_indices = torch.topk(
                fill_score,
                k=graph_max_fill,
                dim=-1,
                largest=True,
                sorted=False,
            ).indices
            positions = torch.arange(graph_max_fill, device=device).view(
                1, 1, 1, -1
            )
            valid_fill = positions < graph_fill_count.unsqueeze(-1)
            fill_mask = torch.zeros_like(selected)
            fill_mask.scatter_(-1, fill_indices, valid_fill)
            selected |= fill_mask

    # A tiny-sequence graph can expose fewer blocks than the requested budget.
    # Fill the remainder globally so sparsity remains comparable across methods.
    fallback_count = (budget - selected.sum(dim=-1)).clamp_min(0)
    max_fallback = int(fallback_count.max().item())
    if max_fallback > 0:
        fallback_score = score.masked_fill(selected, float("-inf"))
        fallback_indices = torch.topk(
            fallback_score,
            k=max_fallback,
            dim=-1,
            largest=True,
            sorted=False,
        ).indices
        positions = torch.arange(max_fallback, device=device).view(1, 1, 1, -1)
        valid_fallback = positions < fallback_count.unsqueeze(-1)
        fallback_mask = torch.zeros_like(selected)
        fallback_mask.scatter_(-1, fallback_indices, valid_fallback)
        selected |= fallback_mask

    return selected, anchor_count


def get_coverage_layerwise_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    layer_idx: int = 0,
    hybrid_early_end: int = 8,
    hybrid_mid_end: int = 16,
    hybrid_mid_sparse_ratio: float = 0.55,
    hybrid_late_sparse_ratio: float = 0.70,
    hybrid_early_local_radius: int = 2,
    mid_frame_coverage_ratio: float = 0.25,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 0.5,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    force_last_block: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Regime-aware local, coverage-constrained, and anchor-protected routing."""
    if hybrid_early_end < 0 or hybrid_mid_end < hybrid_early_end:
        raise ValueError(
            "hybrid layer boundaries must satisfy 0 <= early_end <= mid_end"
        )
    if hybrid_early_local_radius < 0:
        raise ValueError("hybrid_early_local_radius must be non-negative")
    if not 0.0 <= mid_frame_coverage_ratio <= 1.0:
        raise ValueError("mid_frame_coverage_ratio must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    frame_distance = _block_min_frame_distance(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        device=device,
    )
    local_region = frame_distance <= hybrid_early_local_radius
    anchor_count = None

    if layer_idx < hybrid_early_end:
        stage_id = 0
        block_mask = local_region.view(1, 1, q_blocks, k_blocks).expand(
            B, nh, -1, -1
        ).clone()
    elif layer_idx < hybrid_mid_end:
        stage_id = 1
        block_mask, anchor_count = _select_strict_budget_with_frame_coverage(
            pooled_score=pooled_score,
            protected_mask=local_region,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            k_block_size=k_block_size,
            sparse_ratio=hybrid_mid_sparse_ratio,
            frame_coverage_ratio=mid_frame_coverage_ratio,
        )
    else:
        stage_id = 2
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            sparse_ratio=hybrid_late_sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=force_last_block,
            protect_reference_frame=True,
            stats_store=stats_store,
            eps=eps,
        )

    if stats_store is not None:
        selected = block_mask.float()
        selected_total = selected.sum().clamp_min(1.0)
        stats_store.update(
            {
                "coverage_hybrid_stage_id": torch.tensor(
                    float(stage_id), device=device
                ),
                "coverage_hybrid_patch_sparsity": 1.0 - selected.mean(),
                "coverage_hybrid_local_fraction": (
                    selected
                    * local_region.view(1, 1, q_blocks, k_blocks).float()
                ).sum()
                / selected_total,
            }
        )
        if anchor_count is not None:
            stats_store["coverage_hybrid_mid_anchors_per_query"] = (
                anchor_count.float().mean()
            )
            stats_store["coverage_hybrid_mid_anchor_fraction"] = (
                anchor_count.float().sum() / selected_total
            )
    return block_mask


def get_persistent_view_graph_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    layer_idx: int = 0,
    view_graph_early_end: int = 6,
    view_graph_mid_end: int = 18,
    view_graph_mid_sparse_ratio: float = 0.70,
    view_graph_local_radius: int = 1,
    view_graph_remote_topk: int = 3,
    view_graph_remote_ratio: float = 0.30,
    view_graph_refresh_interval: int = 3,
    view_graph_momentum: float = 0.60,
    view_graph_bidirectional: bool = True,
    view_graph_protect_reference: bool = True,
    view_graph_force_anchors: bool = False,
    view_graph_hard_routing: bool = False,
    view_graph_weight: float = 0.15,
    view_graph_head_adaptive: bool = True,
    sparse_ratio: float = 0.70,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 0.5,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    force_last_block: bool = False,
    routing_state: dict | None = None,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Use baseline soft routing around a persistent middle-layer view graph."""
    if view_graph_early_end < 0 or view_graph_mid_end < view_graph_early_end:
        raise ValueError(
            "view graph boundaries must satisfy 0 <= early_end <= mid_end"
        )
    if view_graph_local_radius < 0:
        raise ValueError("view_graph_local_radius must be non-negative")
    if view_graph_remote_topk < 0:
        raise ValueError("view_graph_remote_topk must be non-negative")
    if not 0.0 <= view_graph_remote_ratio <= 1.0:
        raise ValueError("view_graph_remote_ratio must be in [0, 1]")
    if view_graph_refresh_interval < 1:
        raise ValueError("view_graph_refresh_interval must be >= 1")
    if not 0.0 <= view_graph_momentum < 1.0:
        raise ValueError("view_graph_momentum must be in [0, 1)")
    if view_graph_weight < 0:
        raise ValueError("view_graph_weight must be non-negative")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    in_graph_stage = view_graph_early_end <= layer_idx < view_graph_mid_end
    if not in_graph_stage:
        stage_id = 0 if layer_idx < view_graph_early_end else 2
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            sparse_ratio=sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=force_last_block,
            stats_store=stats_store,
            eps=eps,
        )
        if stats_store is not None:
            stats_store["persistent_view_graph_stage_id"] = torch.tensor(
                float(stage_id), device=device
            )
        return block_mask

    stage_id = 1
    q_frame = _block_center_frame_indices(
        q_blocks,
        q_block_size,
        num_frames,
        tokens_per_frame,
        device,
    )
    k_frame = _block_center_frame_indices(
        k_blocks,
        k_block_size,
        num_frames,
        tokens_per_frame,
        device,
    )
    state = routing_state if routing_state is not None else {}
    shape_key = (B, num_frames, q_blocks, k_blocks, device.type, device.index)
    cached_shape = state.get("view_graph_shape")
    cached_graph = state.get("view_graph_mask")
    cached_score = state.get("view_graph_score")
    last_refresh = state.get("view_graph_last_refresh", -view_graph_refresh_interval)
    should_refresh = (
        (cached_graph is None if view_graph_hard_routing else cached_score is None)
        or cached_shape != shape_key
        or layer_idx - int(last_refresh) >= view_graph_refresh_interval
    )

    if should_refresh:
        current_score, _, _ = _compute_head_consensus_view_scores(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            bidirectional=view_graph_bidirectional,
        )
        previous_score = state.get("view_graph_score")
        if previous_score is not None and cached_shape == shape_key:
            graph_score = (
                view_graph_momentum * previous_score
                + (1.0 - view_graph_momentum) * current_score
            )
        else:
            graph_score = current_score
        view_graph = None
        if view_graph_hard_routing:
            view_graph = _select_persistent_view_graph(
                view_score=graph_score,
                local_radius=view_graph_local_radius,
                remote_topk=view_graph_remote_topk,
                remote_ratio=view_graph_remote_ratio,
                protect_reference_frame=view_graph_protect_reference,
            )
        state["view_graph_shape"] = shape_key
        state["view_graph_score"] = graph_score.detach()
        state["view_graph_mask"] = (
            view_graph.detach() if view_graph is not None else None
        )
        state["view_graph_last_refresh"] = layer_idx
    else:
        graph_score = state["view_graph_score"]
        view_graph = cached_graph

    view_bias = None
    head_gate = None
    if view_graph_hard_routing:
        target_remote = max(
            view_graph_remote_topk,
            int(math.ceil(num_frames * view_graph_remote_ratio)),
        )
        minimum_candidate_blocks = min(
            num_frames,
            target_remote + 1,
        ) * max(k_blocks // num_frames, 1)
        required_blocks = int(k_blocks * (1.0 - view_graph_mid_sparse_ratio))
        block_mask, anchor_count = _strict_budget_with_view_graph(
            pooled_score=pooled_score,
            view_graph=view_graph,
            q_frame=q_frame,
            k_frame=k_frame,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            k_block_size=k_block_size,
            sparse_ratio=view_graph_mid_sparse_ratio,
            force_frame_anchors=view_graph_force_anchors,
            single_pass=minimum_candidate_blocks >= required_blocks,
        )
    else:
        view_prior = graph_score[:, q_frame, :][:, :, k_frame]
        view_prior_mean = view_prior.mean(dim=-1, keepdim=True).clamp_min(eps)
        view_bias = torch.log(view_prior.clamp_min(eps) / view_prior_mean)
        view_bias = view_bias.clamp(min=-4.0, max=4.0).unsqueeze(1)
        if view_graph_head_adaptive:
            head_frame_mass = torch.zeros(
                (B, nh, q_blocks, num_frames),
                dtype=pooled_score.dtype,
                device=device,
            )
            head_frame_mass.scatter_add_(
                -1,
                k_frame.view(1, 1, 1, -1).expand(B, nh, q_blocks, -1),
                pooled_score,
            )
            head_frame_prob = head_frame_mass / head_frame_mass.sum(
                dim=-1, keepdim=True
            ).clamp_min(eps)
            head_entropy = -(
                head_frame_prob
                * head_frame_prob.clamp_min(eps).log()
            ).sum(dim=-1)
            head_entropy = head_entropy / math.log(max(num_frames, 2))
            head_gate = (1.0 - head_entropy).clamp_min(0.0)
            head_gate = head_gate / head_gate.amax(
                dim=1, keepdim=True
            ).clamp_min(eps)
            head_gate = head_gate.square()
            view_bias = view_bias * head_gate.unsqueeze(-1)
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            sparse_ratio=view_graph_mid_sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            routing_bias=view_bias,
            routing_bias_weight=view_graph_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=force_last_block,
            stats_store=stats_store,
            eps=eps,
        )
        anchor_count = torch.zeros(
            (B, nh, q_blocks), dtype=torch.long, device=device
        )

    if stats_store is not None:
        selected = block_mask.float()
        frame = torch.arange(num_frames, device=device)
        local_graph = (
            (frame[:, None] - frame[None, :]).abs()
            <= view_graph_local_radius
        )
        selected_graph_edges = (
            view_graph.float().sum(dim=-1).mean()
            if view_graph is not None
            else torch.tensor(float(num_frames), device=device)
        )
        stats_store.update(
            {
                "persistent_view_graph_stage_id": torch.tensor(
                    float(stage_id), device=device
                ),
                "persistent_view_graph_patch_sparsity": 1.0 - selected.mean(),
                "persistent_view_graph_refresh": torch.tensor(
                    float(should_refresh), device=device
                ),
                "persistent_view_graph_views_per_query": selected_graph_edges,
                "persistent_view_graph_remote_views_per_query": (
                    (
                        view_graph
                        & ~local_graph.view(1, num_frames, num_frames)
                    ).float().sum(dim=-1).mean()
                    if view_graph is not None
                    else (~local_graph).float().sum(dim=-1).mean()
                ),
                "persistent_view_graph_anchors_per_query": (
                    anchor_count.float().mean()
                ),
                "persistent_view_graph_score_mean": graph_score.mean(),
                "persistent_view_graph_bias_std": (
                    view_bias.float().std()
                    if view_bias is not None
                    else torch.tensor(0.0, device=device)
                ),
                "persistent_view_graph_head_gate_mean": (
                    head_gate.float().mean()
                    if head_gate is not None
                    else torch.tensor(1.0, device=device)
                ),
            }
        )
    return block_mask


def get_layerwise_hybrid_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    layer_idx: int = 0,
    hybrid_early_end: int = 8,
    hybrid_mid_end: int = 16,
    hybrid_mid_sparse_ratio: float = 0.50,
    hybrid_late_sparse_ratio: float = 0.70,
    hybrid_early_dense_neighbor: int = 4,
    cdf_threshold: float | None = None,
    decay_factor: float = 1.0,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 1.0,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    force_last_block: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Route early layers radially, middle layers by importance, late layers by soft geometry."""
    if hybrid_early_end < 0 or hybrid_mid_end < hybrid_early_end:
        raise ValueError(
            "hybrid layer boundaries must satisfy 0 <= early_end <= mid_end"
        )
    if hybrid_early_dense_neighbor < 0:
        raise ValueError("hybrid_early_dense_neighbor must be non-negative")
    if not 0.0 <= hybrid_mid_sparse_ratio <= 1.0:
        raise ValueError("hybrid_mid_sparse_ratio must be in [0, 1]")
    if not 0.0 <= hybrid_late_sparse_ratio <= 1.0:
        raise ValueError("hybrid_late_sparse_ratio must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    config_sparse_ratio = None

    if layer_idx < hybrid_early_end:
        stage_id = 0
        radial_mask = _get_cached_radial_block_mask(
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            decay_factor=decay_factor,
            dense_neighbor=hybrid_early_dense_neighbor,
            device=device,
        )
        if radial_mask.shape != (q_blocks, k_blocks):
            raise ValueError(
                f"Radial mask shape {radial_mask.shape} does not match "
                f"pooled score blocks {(q_blocks, k_blocks)}"
            )
        block_mask = radial_mask.view(1, 1, q_blocks, k_blocks).expand(
            B, nh, -1, -1
        ).clone()
    elif layer_idx < hybrid_mid_end:
        stage_id = 1
        config_sparse_ratio = hybrid_mid_sparse_ratio
        allowed_mask = torch.ones(
            (q_blocks, k_blocks), dtype=torch.bool, device=device
        )
        block_mask = _select_importance_in_region(
            pooled_score=pooled_score,
            allowed_mask=allowed_mask,
            sparse_ratio=hybrid_mid_sparse_ratio,
            cdf_threshold=cdf_threshold,
            eps=eps,
        )
    else:
        stage_id = 2
        config_sparse_ratio = hybrid_late_sparse_ratio
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            sparse_ratio=hybrid_late_sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=force_last_block,
            stats_store=stats_store,
            eps=eps,
        )

    if stats_store is not None:
        stats_store.update(
            {
                "layerwise_hybrid_stage_id": torch.tensor(
                    float(stage_id), device=device
                ),
                "layerwise_hybrid_patch_sparsity": (
                    1.0 - block_mask.float().mean()
                ).detach(),
            }
        )
        if config_sparse_ratio is not None:
            stats_store["layerwise_hybrid_config_sparse_ratio"] = torch.tensor(
                float(config_sparse_ratio), device=device
            )
    return block_mask


def get_local_layerwise_hybrid_block_mask(
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    q_block_size: int = 128,
    k_block_size: int = 64,
    layer_idx: int = 0,
    hybrid_early_end: int = 8,
    hybrid_mid_end: int = 16,
    hybrid_mid_sparse_ratio: float = 0.50,
    hybrid_late_sparse_ratio: float = 0.70,
    hybrid_early_local_radius: int = 2,
    cdf_threshold: float | None = None,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 1.0,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    force_last_block: bool = False,
    stats_store: dict | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Keep only local frames early; keep local plus far importance in the middle."""
    if hybrid_early_end < 0 or hybrid_mid_end < hybrid_early_end:
        raise ValueError(
            "hybrid layer boundaries must satisfy 0 <= early_end <= mid_end"
        )
    if hybrid_early_local_radius < 0:
        raise ValueError("hybrid_early_local_radius must be non-negative")
    if not 0.0 <= hybrid_mid_sparse_ratio <= 1.0:
        raise ValueError("hybrid_mid_sparse_ratio must be in [0, 1]")
    if not 0.0 <= hybrid_late_sparse_ratio <= 1.0:
        raise ValueError("hybrid_late_sparse_ratio must be in [0, 1]")

    B, nh, q_blocks, k_blocks = pooled_score.shape
    device = pooled_score.device
    frame_distance = _block_min_frame_distance(
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        q_block_size=q_block_size,
        k_block_size=k_block_size,
        q_blocks=q_blocks,
        k_blocks=k_blocks,
        device=device,
    )
    local_region = frame_distance <= hybrid_early_local_radius
    config_sparse_ratio = None

    if layer_idx < hybrid_early_end:
        stage_id = 0
        block_mask = local_region.view(1, 1, q_blocks, k_blocks).expand(
            B, nh, -1, -1
        ).clone()
    elif layer_idx < hybrid_mid_end:
        stage_id = 1
        config_sparse_ratio = hybrid_mid_sparse_ratio
        far_importance = _select_importance_in_region(
            pooled_score=pooled_score,
            allowed_mask=~local_region,
            sparse_ratio=hybrid_mid_sparse_ratio,
            cdf_threshold=cdf_threshold,
            eps=eps,
        )
        local_mask = local_region.view(1, 1, q_blocks, k_blocks).expand(
            B, nh, -1, -1
        )
        block_mask = local_mask | far_importance
    else:
        stage_id = 2
        config_sparse_ratio = hybrid_late_sparse_ratio
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=q_block_size,
            k_block_size=k_block_size,
            sparse_ratio=hybrid_late_sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=force_last_block,
            stats_store=stats_store,
            eps=eps,
        )

    if stats_store is not None:
        stats_store.update(
            {
                "layerwise_hybrid_stage_id": torch.tensor(
                    float(stage_id), device=device
                ),
                "layerwise_hybrid_patch_sparsity": (
                    1.0 - block_mask.float().mean()
                ).detach(),
                "layerwise_hybrid_local_fraction": (
                    block_mask.float()
                    * local_region.view(1, 1, q_blocks, k_blocks).float()
                ).sum().div(block_mask.float().sum().clamp_min(1.0)).detach(),
            }
        )
        if config_sparse_ratio is not None:
            stats_store["layerwise_hybrid_config_sparse_ratio"] = torch.tensor(
                float(config_sparse_ratio), device=device
            )
    return block_mask


def block_sparse_attn_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    topk: int | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    return_sparsity: bool = False,
    dtype: torch.dtype = torch.float16,
    # Radial + layer-wise fusion parameters
    use_radial_layerwise: bool = False,
    num_frames: int | None = None,
    tokens_per_frame: int | None = None,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    layer_idx: int = 0,
    layer_sparsity_ratios: list | None = None,
    use_distance_routed: bool = False,
    route_frame_threshold: int = 4,
    use_covariance_aware_importance: bool = False,
    covariance_weight: float = 0.5,
    covariance_eps: float = 1e-8,
    key_block_variance: torch.Tensor | None = None,
    use_adaptive_slit_routing: bool = False,
    adaptive_slit_temporal_window: int = 10,
    adaptive_slit_stable_quantile: float = 0.6,
    adaptive_slit_change_quantile: float = 0.7,
    adaptive_slit_narrow_width: int = 1,
    adaptive_slit_base_width: int = 2,
    adaptive_slit_expand_width: int = 4,
    # Soft geometry-content routing parameters
    use_soft_geometry_routing: bool = False,
    use_preview_adaptive_routing: bool = False,
    preview_redistribution_fraction: float = 0.05,
    preview_activation_threshold: float = 0.60,
    preview_local_radius: int = 1,
    preview_local_fraction: float = 0.0,
    preview_geometry_weight: float = 0.0,
    preview_head_protection: str = "none",
    preview_protected_head_fraction: float = 0.25,
    preview_donor_retention_threshold: float = 0.0,
    preview_donor_exchange_scope: str = "head",
    preview_receiver_gain_threshold: float = 0.0,
    preview_exchange_gain_cost_ratio: float = 0.0,
    preview_head_exchange_cap_fraction: float = 1.0,
    preview_layer_confidence_threshold: float = 0.0,
    preview_exchange_layer_start: int = 0,
    preview_exchange_layer_end: int = -1,
    # Role-adaptive local/global dual-path routing parameters
    use_dual_path_routing: bool = False,
    dual_path_local_radius: int = 2,
    dual_path_min_local_fraction: float = 0.10,
    dual_path_max_local_fraction: float = 0.50,
    dual_path_layer_schedule: str = "early_decay",
    dual_path_num_layers: int = 24,
    dual_path_context_geometry_weight: float = 0.0,
    dual_path_context_schedule: str = "flat",
    dual_path_context_gate: str = "none",
    dual_path_context_alignment_threshold: float = 1.2,
    dual_path_context_alignment_temperature: float = 0.1,
    # Layer-wise radial/importance/soft hybrid routing parameters
    use_layerwise_hybrid_routing: bool = False,
    use_local_layerwise_hybrid_routing: bool = False,
    use_coverage_layerwise_routing: bool = False,
    use_persistent_view_graph_routing: bool = False,
    hybrid_early_end: int = 8,
    hybrid_mid_end: int = 16,
    hybrid_mid_sparse_ratio: float = 0.50,
    hybrid_late_sparse_ratio: float = 0.70,
    hybrid_early_dense_neighbor: int = 4,
    hybrid_early_local_radius: int = 2,
    mid_frame_coverage_ratio: float = 0.25,
    view_graph_early_end: int = 6,
    view_graph_mid_end: int = 18,
    view_graph_mid_sparse_ratio: float = 0.70,
    view_graph_local_radius: int = 1,
    view_graph_remote_topk: int = 3,
    view_graph_remote_ratio: float = 0.30,
    view_graph_refresh_interval: int = 3,
    view_graph_momentum: float = 0.60,
    view_graph_bidirectional: bool = True,
    view_graph_protect_reference: bool = True,
    view_graph_force_anchors: bool = False,
    view_graph_hard_routing: bool = False,
    view_graph_weight: float = 0.15,
    view_graph_head_adaptive: bool = True,
    core_frame_radius: int = 4,
    transition_frame_radius: int = 12,
    geometry_weight: float = 1.0,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    head_adaptive_geometry: bool = False,
    routing_stats_store: dict | None = None,
    routing_state: dict | None = None,
    analyze_importance_blocks: bool = False,
    analyze_block_selection: bool = False,
    # Special token handling
    num_patch_tokens: int | None = None,
):
    """Block sparse attention using SpargeAttn kernels

    Args:
        query (torch.Tensor): (B, nheads, Tq, head_dim)
        key (torch.Tensor): (B, nheads, Tk, head_dim)
        value (torch.Tensor): (B, nheads, Tk, head_dim)
            sink tokens are appended to the end of key and value.
        pooled_score (torch.Tensor): (B, nheads, q_blk, k_blk)
            where q_blk and k_blk are the number of query and key blocks.
            The score here *doesn't* contain the sink tokens.
        topk, sparse_ratio, cdf_threshold: the mode of sparse attention
            - topk: choose the top-k key blocks for each query block
            - sparse_ratio: choose a ratio of top blocks
            - cdf_threshold: choose blocks that accumulate to a certain threshold
        use_radial_layerwise: If True, use fused radial+layer-wise+importance mask
        num_frames: Number of frames (required if use_radial_layerwise=True)
        tokens_per_frame: Number of patch tokens per frame (required if use_radial_layerwise=True)
        decay_factor: Radial decay factor (default 1.0)
        dense_neighbor: Number of neighboring frames with full attention (default 1)
        layer_idx: Current layer index for layer-wise sparsity (default 0)
        layer_sparsity_ratios: List of per-layer sparsity ratios (optional)
        num_patch_tokens: Actual number of patch tokens (for special-token handling)

    Returns:
        out: Attention output of shape (B, nheads, T, head_dim)
    """
    out_dtype = query.dtype
    native_sparse_output = os.environ.get(
        "SPARSE_VGGT_NATIVE_SPARSE_OUTPUT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    native_output_dtype = (
        value.dtype
        if value.dtype in {torch.float16, torch.bfloat16}
        else dtype
    )

    # Hardcode some arguments for using SpargeAttn kernels
    _is_causal = 0
    KBLK = 64
    Tk = key.shape[-2]
    pv_threshold_value = _resolve_qk_pv_threshold(
        os.environ.get("SPARSE_VGGT_QK_PV_THRESHOLD", "1e10"), Tk
    )
    pv_threshold_active = _pv_threshold_active_for_layer(
        os.environ.get("SPARSE_VGGT_QK_PV_ACTIVE_LAYERS", ""),
        layer_idx,
    )
    if not pv_threshold_active:
        pv_threshold_value = 1e10
    pvthreshd = (
        hyperparameter_check(
            pv_threshold_value, query.size(-3), query.device
        )
        if pv_threshold_active
        else None
    )

    # Get block mask
    orig_Kblk = pooled_score.shape[-1]
    total_Kblk = math.ceil(Tk / KBLK)
    sink_blocks = total_Kblk - orig_Kblk

    # IMPORTANT: A区(patch-to-patch)的key包含patch tokens + special tokens
    # 当special tokens与最后一个patch块混合时，必须保护该混合块
    # 检测条件：有special tokens且最后一个pooled块不是完整的patch块
    has_mixed_last_block = False
    if num_patch_tokens is not None and num_patch_tokens < Tk:
        # Special tokens存在，检查最后一个pooled块是否为混合块
        if num_patch_tokens % KBLK != 0:
            has_mixed_last_block = True

    routed_modes = (
        use_radial_layerwise,
        use_distance_routed,
        use_soft_geometry_routing,
        use_preview_adaptive_routing,
        use_dual_path_routing,
        use_layerwise_hybrid_routing,
        use_local_layerwise_hybrid_routing,
        use_coverage_layerwise_routing,
        use_persistent_view_graph_routing,
    )
    if sum(routed_modes) > 1:
        raise ValueError(
            "use_radial_layerwise, use_distance_routed, and "
            "use_soft_geometry_routing/use_dual_path_routing/"
            "use_preview_adaptive_routing/"
            "use_layerwise_hybrid_routing/"
            "use_local_layerwise_hybrid_routing/"
            "use_coverage_layerwise_routing/"
            "use_persistent_view_graph_routing are mutually exclusive"
        )

    if use_preview_adaptive_routing:
        if sparse_ratio is None or cdf_threshold is not None or topk is not None:
            raise ValueError(
                "preview-adaptive routing currently supports ratio-only selection"
            )
        if num_patch_tokens is None:
            raise ValueError(
                "num_patch_tokens is required for preview-adaptive routing"
            )
        if preview_local_fraction > 0 and (
            num_frames is None or tokens_per_frame is None
        ):
            raise ValueError(
                "frame metadata is required for preview local protection"
            )
        patch_value = value[..., :num_patch_tokens, :]
        B, heads, _, head_dim = patch_value.shape
        pooled_value = _mean_pool_contiguous_value_blocks(patch_value, KBLK)
        value_detail = None
        if preview_head_protection == "value_detail":
            pooled_value_second_moment = F.avg_pool1d(
                patch_value.float().square().permute(0, 1, 3, 2).reshape(
                    B * heads, head_dim, num_patch_tokens
                ),
                kernel_size=KBLK,
                stride=KBLK,
                ceil_mode=True,
            ).reshape(B, heads, head_dim, orig_Kblk).permute(0, 1, 3, 2)
            pooled_value_float = pooled_value.float()
            residual_energy = (
                pooled_value_second_moment - pooled_value_float.square()
            ).clamp_min(0.0).mean(dim=-1)
            total_energy = pooled_value_second_moment.mean(dim=-1).clamp_min(1e-8)
            value_detail = (residual_energy / total_energy).sqrt()
        preview_exchange_enabled = (
            layer_idx >= preview_exchange_layer_start
            and (
                preview_exchange_layer_end < 0
                or layer_idx < preview_exchange_layer_end
            )
        )
        effective_preview_redistribution = (
            preview_redistribution_fraction
            if preview_exchange_enabled
            else 0.0
        )
        block_mask = get_preview_adaptive_block_mask(
            pooled_score=pooled_score,
            pooled_value=pooled_value,
            sparse_ratio=sparse_ratio,
            redistribution_fraction=effective_preview_redistribution,
            activation_threshold=preview_activation_threshold,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            local_frame_radius=preview_local_radius,
            local_fraction=preview_local_fraction,
            geometry_weight=preview_geometry_weight,
            value_detail=value_detail,
            head_protection=preview_head_protection,
            protected_head_fraction=preview_protected_head_fraction,
            donor_retention_threshold=preview_donor_retention_threshold,
            donor_exchange_scope=preview_donor_exchange_scope,
            receiver_gain_threshold=preview_receiver_gain_threshold,
            exchange_gain_cost_ratio=preview_exchange_gain_cost_ratio,
            head_exchange_cap_fraction=(
                preview_head_exchange_cap_fraction
            ),
            layer_confidence_threshold=(
                preview_layer_confidence_threshold
            ),
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if routing_stats_store is not None:
            routing_stats_store["preview_adaptive_exchange_enabled"] = (
                torch.tensor(
                    float(preview_exchange_enabled),
                    device=pooled_score.device,
                )
            )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

    elif use_persistent_view_graph_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for persistent view graph routing"
            )
        if sparse_ratio is None or cdf_threshold is not None or topk is not None:
            raise ValueError(
                "persistent view graph routing currently supports ratio-only selection"
            )
        block_mask = get_persistent_view_graph_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            layer_idx=layer_idx,
            view_graph_early_end=view_graph_early_end,
            view_graph_mid_end=view_graph_mid_end,
            view_graph_mid_sparse_ratio=view_graph_mid_sparse_ratio,
            view_graph_local_radius=view_graph_local_radius,
            view_graph_remote_topk=view_graph_remote_topk,
            view_graph_remote_ratio=view_graph_remote_ratio,
            view_graph_refresh_interval=view_graph_refresh_interval,
            view_graph_momentum=view_graph_momentum,
            view_graph_bidirectional=view_graph_bidirectional,
            view_graph_protect_reference=view_graph_protect_reference,
            view_graph_force_anchors=view_graph_force_anchors,
            view_graph_hard_routing=view_graph_hard_routing,
            view_graph_weight=view_graph_weight,
            view_graph_head_adaptive=view_graph_head_adaptive,
            sparse_ratio=sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=has_mixed_last_block,
            routing_state=routing_state,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True

    elif use_coverage_layerwise_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for coverage layerwise routing"
            )
        block_mask = get_coverage_layerwise_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            layer_idx=layer_idx,
            hybrid_early_end=hybrid_early_end,
            hybrid_mid_end=hybrid_mid_end,
            hybrid_mid_sparse_ratio=hybrid_mid_sparse_ratio,
            hybrid_late_sparse_ratio=hybrid_late_sparse_ratio,
            hybrid_early_local_radius=hybrid_early_local_radius,
            mid_frame_coverage_ratio=mid_frame_coverage_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True

    elif use_local_layerwise_hybrid_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for local layerwise hybrid routing"
            )
        block_mask = get_local_layerwise_hybrid_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            layer_idx=layer_idx,
            hybrid_early_end=hybrid_early_end,
            hybrid_mid_end=hybrid_mid_end,
            hybrid_mid_sparse_ratio=hybrid_mid_sparse_ratio,
            hybrid_late_sparse_ratio=hybrid_late_sparse_ratio,
            hybrid_early_local_radius=hybrid_early_local_radius,
            cdf_threshold=cdf_threshold,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True

    elif use_layerwise_hybrid_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for layerwise hybrid routing"
            )
        block_mask = get_layerwise_hybrid_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            layer_idx=layer_idx,
            hybrid_early_end=hybrid_early_end,
            hybrid_mid_end=hybrid_mid_end,
            hybrid_mid_sparse_ratio=hybrid_mid_sparse_ratio,
            hybrid_late_sparse_ratio=hybrid_late_sparse_ratio,
            hybrid_early_dense_neighbor=hybrid_early_dense_neighbor,
            cdf_threshold=cdf_threshold,
            decay_factor=decay_factor,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True

    elif use_dual_path_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for dual-path routing"
            )
        if sparse_ratio is None or cdf_threshold is not None or topk is not None:
            raise ValueError(
                "dual-path routing currently supports ratio-only selection"
            )
        block_mask = get_role_adaptive_dual_path_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            sparse_ratio=sparse_ratio,
            layer_idx=layer_idx,
            num_layers=dual_path_num_layers,
            local_frame_radius=dual_path_local_radius,
            min_local_fraction=dual_path_min_local_fraction,
            max_local_fraction=dual_path_max_local_fraction,
            layer_schedule=dual_path_layer_schedule,
            geometry_weight=geometry_weight,
            context_geometry_weight=dual_path_context_geometry_weight,
            context_schedule=dual_path_context_schedule,
            context_gate=dual_path_context_gate,
            context_alignment_threshold=dual_path_context_alignment_threshold,
            context_alignment_temperature=dual_path_context_alignment_temperature,
            context_core_frame_radius=core_frame_radius,
            context_transition_frame_radius=transition_frame_radius,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

    # Use strict-budget soft geometry routing if enabled.
    elif use_soft_geometry_routing:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame are required for soft geometry routing"
            )
        if sparse_ratio is None or cdf_threshold is not None or topk is not None:
            raise ValueError(
                "soft geometry routing currently supports ratio-only selection"
            )
        block_mask = get_soft_geometry_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            sparse_ratio=sparse_ratio,
            core_frame_radius=core_frame_radius,
            transition_frame_radius=transition_frame_radius,
            geometry_weight=geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            head_adaptive_geometry=head_adaptive_geometry,
            force_last_block=has_mixed_last_block,
            stats_store=routing_stats_store,
        )
        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask
    elif use_distance_routed:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError(
                "num_frames and tokens_per_frame must be provided "
                "when use_distance_routed=True"
            )
        block_mask = get_distance_routed_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            route_frame_threshold=route_frame_threshold,
            use_covariance_aware_importance=use_covariance_aware_importance,
            covariance_weight=covariance_weight,
            covariance_eps=covariance_eps,
            key_block_variance=key_block_variance,
            use_adaptive_slit_routing=use_adaptive_slit_routing,
            adaptive_slit_temporal_window=adaptive_slit_temporal_window,
            adaptive_slit_stable_quantile=adaptive_slit_stable_quantile,
            adaptive_slit_change_quantile=adaptive_slit_change_quantile,
            adaptive_slit_narrow_width=adaptive_slit_narrow_width,
            adaptive_slit_base_width=adaptive_slit_base_width,
            adaptive_slit_expand_width=adaptive_slit_expand_width,
            stats_store=routing_stats_store,
        )

        if sink_blocks > 0:
            sink_mask = torch.ones(
                (*block_mask.shape[:-1], sink_blocks),
                dtype=torch.bool,
                device=block_mask.device,
            )
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True

    # Use radial+layer-wise fusion if enabled
    elif use_radial_layerwise:
        if num_frames is None or tokens_per_frame is None:
            raise ValueError("num_frames and tokens_per_frame must be provided when use_radial_layerwise=True")

        # Get fused mask (radial + layer-wise + importance)
        import time
        t0 = time.perf_counter()
        block_mask = get_radial_layerwise_block_mask(
            pooled_score=pooled_score,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,  # Default query block size
            k_block_size=KBLK,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            layer_idx=layer_idx,
            layer_sparsity_ratios=layer_sparsity_ratios,
        )
        t1 = time.perf_counter()

        # Add sink blocks (always attend to special tokens)
        if sink_blocks > 0:
            B, nh, q_blk, k_blk = block_mask.shape
            sink_mask = torch.ones(B, nh, q_blk, sink_blocks, dtype=torch.bool, device=block_mask.device)
            final_map = torch.cat([block_mask, sink_mask], dim=-1)
        else:
            final_map = block_mask

        # A区：保护包含special tokens的混合块（最后一个pooled块）
        if has_mixed_last_block:
            final_map[:, :, :, orig_Kblk - 1] = True
    else:
        use_block_debt = os.environ.get(
            "SPARSE_VGGT_BLOCK_DEBT", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if use_block_debt:
            if (
                sparse_ratio is None
                or cdf_threshold is not None
                or topk is not None
            ):
                raise ValueError(
                    "block-debt routing supports ratio-only selection"
                )
            if num_frames is None or tokens_per_frame is None:
                raise ValueError(
                    "block-debt routing requires frame metadata"
                )
            block_mask = get_block_debt_mask(
                pooled_score,
                sparse_ratio=sparse_ratio,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
                layer_idx=layer_idx,
                routing_state=routing_state,
                momentum=float(
                    os.environ.get(
                        "SPARSE_VGGT_BLOCK_DEBT_MOMENTUM", "0.75"
                    )
                ),
                repayment=float(
                    os.environ.get(
                        "SPARSE_VGGT_BLOCK_DEBT_REPAYMENT", "1.0"
                    )
                ),
                service_credit_scale=float(
                    os.environ.get(
                        "SPARSE_VGGT_BLOCK_DEBT_CREDIT", "1.0"
                    )
                ),
                frame_balance_fraction=float(
                    os.environ.get(
                        "SPARSE_VGGT_BLOCK_DEBT_FRAME_BALANCE", "0.2"
                    )
                ),
                stats_store=routing_stats_store,
            )
            if sink_blocks > 0:
                sink_mask = torch.ones(
                    (*block_mask.shape[:-1], sink_blocks),
                    dtype=torch.bool,
                    device=block_mask.device,
                )
                final_map = torch.cat([block_mask, sink_mask], dim=-1)
            else:
                final_map = block_mask
        else:
            # Keep an exact upstream QK sparse attention path for baselines.
            exact_upstream_mask = os.environ.get(
                "SPARSE_VGGT_OFFICIAL_MASK", "0"
            ).lower() in {"1", "true", "yes", "on"}
            budget_only_upstream = os.environ.get(
                "SPARSE_VGGT_COSA_BUDGET_ONLY_UPSTREAM", "0"
            ).lower() in {"1", "true", "yes", "on"}
            use_cosa_early = os.environ.get(
                "SPARSE_VGGT_COSA_VALUE_RISK_PAIR_DEBT", "0"
            ).lower() in {"1", "true", "yes", "on"}
            if budget_only_upstream:
                if not use_cosa_early:
                    raise ValueError(
                        "budget-only upstream requires the CoSA pair selector"
                    )
                if (
                    sparse_ratio is None
                    or cdf_threshold is not None
                    or topk is not None
                ):
                    raise ValueError(
                        "budget-only upstream supports ratio selection only"
                    )
                final_map = _fixed_ratio_capacity_mask(
                    pooled_score,
                    sparse_ratio=sparse_ratio,
                    sink_blocks=sink_blocks,
                    force_last_patch=(
                        has_mixed_last_block and not exact_upstream_mask
                    ),
                )
            else:
                final_map = get_block_mask(
                    pooled_score,
                    sink_blocks=sink_blocks,
                    topk=topk,
                    sparse_ratio=sparse_ratio,
                    cdf_threshold=cdf_threshold,
                )
            if (
                has_mixed_last_block
                and not exact_upstream_mask
                and not budget_only_upstream
            ):
                final_map[:, :, :, orig_Kblk - 1] = True

            use_projected_qk_debt = os.environ.get(
                "SPARSE_VGGT_PROJECTED_QK_DEBT", "0"
            ).lower() in {"1", "true", "yes", "on"}
            use_projected_qk_pv_debt = os.environ.get(
                "SPARSE_VGGT_PROJECTED_QK_PV_DEBT", "0"
            ).lower() in {"1", "true", "yes", "on"}
            use_cosa_value_risk_pair_debt = use_cosa_early
            cosa_production_fast = os.environ.get(
                "SPARSE_VGGT_COSA_PRODUCTION_FAST", "0"
            ).lower() in {"1", "true", "yes", "on"}
            cosa_fp32_projection = os.environ.get(
                "SPARSE_VGGT_COSA_FP32_PROJECTION", "0"
            ).lower() in {"1", "true", "yes", "on"}
            cosa_risk_safe_observer = os.environ.get(
                "SPARSE_VGGT_COSA_RISK_SAFE_OBSERVER", "0"
            ).lower() in {"1", "true", "yes", "on"}
            observe_qk_bundles = os.environ.get(
                "SPARSE_VGGT_QK_BUNDLE_OBSERVER", "0"
            ).lower() in {"1", "true", "yes", "on"}
            observe_qk_value_risk = os.environ.get(
                "SPARSE_VGGT_QK_VALUE_RISK_OBSERVER", "0"
            ).lower() in {"1", "true", "yes", "on"}
            value_risk_execution = os.environ.get(
                "SPARSE_VGGT_QK_VALUE_RISK_EXECUTION", "observe"
            ).strip().lower()
            value_risk_definition = os.environ.get(
                "SPARSE_VGGT_QK_VALUE_RISK_DEFINITION", "innovation"
            ).strip().lower()
            value_risk_production = os.environ.get(
                "SPARSE_VGGT_QK_VALUE_RISK_PRODUCTION", "0"
            ).lower() in {"1", "true", "yes", "on"}
            value_risk_map_dir = os.environ.get(
                "SPARSE_VGGT_QK_VALUE_RISK_MAP_DIR", ""
            ).strip() or None
            value_risk_map_layers = {
                int(token.strip())
                for token in os.environ.get(
                    "SPARSE_VGGT_QK_VALUE_RISK_MAP_LAYERS", "0,7,15,23"
                ).split(",")
                if token.strip()
            }
            if sum((
                use_projected_qk_debt,
                use_projected_qk_pv_debt,
                use_cosa_value_risk_pair_debt,
                observe_qk_value_risk and value_risk_execution != "observe",
            )) > 1:
                raise ValueError("choose one QK debt execution mechanism")
            protected_patch_mask = (
                torch.nn.functional.one_hot(
                    torch.tensor(orig_Kblk - 1, device=final_map.device),
                    num_classes=orig_Kblk,
                ).to(dtype=torch.bool).view(1, 1, 1, orig_Kblk)
                .expand_as(final_map[..., :orig_Kblk])
                if has_mixed_last_block and not exact_upstream_mask
                else None
            )
            if observe_qk_bundles:
                bundle_mode = os.environ.get(
                    "SPARSE_VGGT_QK_BUNDLE_MODE", "pair"
                ).strip().lower()
                if bundle_mode == "pair":
                    bundle_size = 2
                elif bundle_mode == "frame":
                    if tokens_per_frame is None:
                        raise ValueError("frame bundles require frame metadata")
                    bundle_size = 1
                    child_centers = (
                        torch.arange(orig_Kblk, device=final_map.device)
                        * KBLK
                        + KBLK // 2
                    )
                    child_to_bundle = torch.clamp(
                        child_centers // tokens_per_frame,
                        max=max(num_frames - 1, 0),
                    )
                else:
                    raise ValueError("QK bundle mode must be pair or frame")
                if bundle_mode == "pair":
                    child_to_bundle = None
                observed_mask = observe_qk_bundle_debt(
                    pooled_score,
                    final_map[..., :orig_Kblk],
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                    stats_store=routing_stats_store,
                    protected_mask=protected_patch_mask,
                    bundle_size=bundle_size,
                    child_to_bundle=child_to_bundle,
                )
                if not torch.equal(observed_mask, final_map[..., :orig_Kblk]):
                    raise RuntimeError("QK bundle observer changed execution")
            if observe_qk_value_risk:
                if num_patch_tokens is None:
                    raise ValueError("value-risk observer requires patch metadata")
                patch_value = value[..., :num_patch_tokens, :]
                B, heads, _, head_dim = patch_value.shape
                pooled_value = _mean_pool_contiguous_value_blocks(
                    patch_value, KBLK
                )
                observed_mask = observe_qk_value_risk_debt(
                    pooled_score,
                    pooled_value,
                    final_map[..., :orig_Kblk],
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                    stats_store=routing_stats_store,
                    protected_mask=protected_patch_mask,
                    execution_mode=value_risk_execution,
                    risk_definition=value_risk_definition,
                    production_fast_path=value_risk_production,
                    attention_map_dir=value_risk_map_dir,
                    attention_map_layers=value_risk_map_layers,
                    num_frames=num_frames,
                    tokens_per_frame=tokens_per_frame,
                )
                if (
                    value_risk_execution == "observe"
                    and not torch.equal(observed_mask, final_map[..., :orig_Kblk])
                ):
                    raise RuntimeError("QK value-risk observer changed execution")
                final_map[..., :orig_Kblk] = observed_mask
            if use_cosa_value_risk_pair_debt:
                if num_patch_tokens is None:
                    raise ValueError("CoSA pair debt requires patch metadata")
                patch_value = value[..., :num_patch_tokens, :]
                B, heads, _, head_dim = patch_value.shape
                cosa_fused_routing = os.environ.get(
                    "SPARSE_VGGT_COSA_FUSED_ROUTING", "0"
                ).lower() in {"1", "true", "yes", "on"}
                pooled_value = (
                    None
                    if cosa_fused_routing
                    else _mean_pool_contiguous_value_blocks(patch_value, KBLK)
                )
                cosa_capacity_hint = None
                if (
                    sparse_ratio is not None
                    and cdf_threshold is None
                    and topk is None
                ):
                    cosa_capacity_hint = int(
                        orig_Kblk * (1.0 - sparse_ratio)
                    ) + int(protected_patch_mask is not None)
                cosa_patch_mask = get_cosa_value_risk_pair_debt_mask(
                    pooled_score,
                    pooled_value,
                    final_map[..., :orig_Kblk],
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                    stats_store=routing_stats_store,
                    protected_mask=protected_patch_mask,
                    max_capacity_hint=cosa_capacity_hint,
                    fp32_projection=cosa_fp32_projection,
                    risk_safe_observer=cosa_risk_safe_observer,
                    raw_value=patch_value if cosa_fused_routing else None,
                    value_block_size=KBLK,
                )
                if sink_blocks > 0:
                    final_map = torch.cat(
                        [cosa_patch_mask, final_map[..., orig_Kblk:]],
                        dim=-1,
                    )
                else:
                    final_map = cosa_patch_mask
            if use_projected_qk_debt or use_projected_qk_pv_debt:
                debt_selector = (
                    get_projected_qk_pv_debt_mask
                    if use_projected_qk_pv_debt
                    else get_projected_qk_debt_mask
                )
                projected_patch_mask = debt_selector(
                    pooled_score,
                    final_map[..., :orig_Kblk],
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                    stats_store=routing_stats_store,
                    protected_mask=protected_patch_mask,
                )
                if sink_blocks > 0:
                    final_map = torch.cat(
                        [projected_patch_mask, final_map[..., orig_Kblk:]],
                        dim=-1,
                    )
                else:
                    final_map = projected_patch_mask

    if routing_stats_store is not None:
        routing_stats_store["qk_pv_threshold"] = torch.tensor(
            pv_threshold_value,
            device=pooled_score.device,
        )

    if analyze_importance_blocks or analyze_block_selection:
        _record_importance_block_analysis(
            stats_store=routing_stats_store,
            final_map=final_map,
            orig_k_blocks=orig_Kblk,
            num_frames=num_frames,
            tokens_per_frame=tokens_per_frame,
            q_block_size=128,
            k_block_size=KBLK,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            has_mixed_last_block=has_mixed_last_block,
        )
    use_cosa_ordered_lut = os.environ.get(
        "SPARSE_VGGT_COSA_ORDERED_LUT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    ordered_lut_active_layers = os.environ.get(
        "SPARSE_VGGT_COSA_ORDERED_LUT_ACTIVE_LAYERS", ""
    ).strip()
    if (
        use_cosa_ordered_lut
        and ordered_lut_active_layers
        and not _pv_threshold_active_for_layer(
            ordered_lut_active_layers,
            layer_idx,
        )
    ):
        # Pooled-QK traversal exists to expose an early online-softmax bound
        # to the PV post-cut. Without a post-cut, physical traversal computes
        # the same support with better K/V locality.
        use_cosa_ordered_lut = False
    if use_cosa_ordered_lut:
        cosa_pair_enabled = os.environ.get(
            "SPARSE_VGGT_COSA_VALUE_RISK_PAIR_DEBT", "0"
        ).lower() in {"1", "true", "yes", "on"}
        topk_ordered_lut = os.environ.get(
            "SPARSE_VGGT_COSA_TOPK_ORDERED_LUT", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if not cosa_pair_enabled and not topk_ordered_lut:
            raise ValueError(
                "ordered LUT requires CoSA pair routing or debt-free Top-K order"
            )
        if cosa_pair_enabled and topk_ordered_lut:
            raise ValueError("choose one ordered-LUT priority source")
        use_hrm_order = os.environ.get(
            "SPARSE_VGGT_COSA_HRM_ORDERED_LUT", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if use_hrm_order:
            if num_patch_tokens is None:
                raise ValueError("HRM ordered LUT requires patch metadata")
            ordered_priority = _cosa_hrm_order_priority(
                query[..., :num_patch_tokens, :],
                key[..., :num_patch_tokens, :],
                final_map[..., :orig_Kblk],
                queries_per_block=int(
                    os.environ.get("SPARSE_VGGT_COSA_HRM_QUERIES_PER_BLOCK", "4")
                ),
                proxy_key_stride=int(
                    os.environ.get("SPARSE_VGGT_COSA_HRM_PROXY_KEY_STRIDE", "8")
                ),
            )
        elif cosa_pair_enabled:
            ordered_priority = routing_state.get(
                "cosa_value_risk_pair_qk_order_priority"
            ) if routing_state is not None else None
            if ordered_priority is None:
                raise RuntimeError("CoSA ordered LUT is missing routing priority")
        else:
            ordered_priority = pooled_score
        max_patch_blocks = routing_state.get(
            "cosa_value_risk_pair_max_patch_blocks"
        ) if routing_state is not None else None
        max_selected_blocks = (
            int(max_patch_blocks) + int(sink_blocks)
            if max_patch_blocks is not None
            else None
        )
        use_compact_ordered_lut = os.environ.get(
            "SPARSE_VGGT_COSA_COMPACT_ORDERED_LUT", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if topk_ordered_lut and use_compact_ordered_lut:
            raise ValueError("debt-free Top-K order does not use pair compact indices")
        use_frontier_order = os.environ.get(
            "SPARSE_VGGT_COSA_QK_FRONTIER_ORDER", "0"
        ).lower() in {"1", "true", "yes", "on"}
        use_frontier_anchor_order = os.environ.get(
            "SPARSE_VGGT_COSA_QK_FRONTIER_ANCHOR_ORDER", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if use_frontier_order and use_frontier_anchor_order:
            raise ValueError("choose one CoSA QK frontier traversal")
        selected_patch_indices = None
        if use_compact_ordered_lut:
            selected_patch_indices = routing_state.get(
                "cosa_value_risk_pair_selected_indices"
            ) if routing_state is not None else None
            if selected_patch_indices is None:
                raise RuntimeError(
                    "compact CoSA ordered LUT is missing selected indices"
                )
        lut, valid_block_num = _priority_ordered_block_lut(
            final_map,
            ordered_priority.to(pooled_score),
            max_selected_blocks=max_selected_blocks,
            selected_patch_indices=selected_patch_indices,
            frontier_only=use_frontier_order,
            frontier_anchor_only=use_frontier_anchor_order,
        )
        if (
            max_selected_blocks is not None
            and routing_stats_store is not None
            and bool((valid_block_num > max_selected_blocks).any())
        ):
            raise RuntimeError("ordered LUT selection bound excluded service")
        if routing_stats_store is not None:
            valid_jump = (
                torch.arange(lut.shape[-1], device=lut.device)
                .view(1, 1, 1, -1)
                < valid_block_num.unsqueeze(-1)
            )
            jump_count = valid_jump.sum().clamp_min(1)
            routing_stats_store.update({
                "cosa_pair_ordered_lut_enabled": torch.tensor(
                    1.0, device=lut.device
                ),
                "cosa_pair_hrm_order_enabled": torch.tensor(
                    float(use_hrm_order), device=lut.device
                ),
                "cosa_pair_ordered_negative_jump_fraction": (
                    ((lut < 0) & valid_jump).sum().float() / jump_count
                ).detach(),
                "cosa_pair_ordered_mean_abs_jump": (
                    lut.abs().masked_fill(~valid_jump, 0).sum().float()
                    / jump_count
                ).detach(),
            })
    else:
        lut, valid_block_num = block_map_lut_triton(final_map)

    # Quantization
    fused_centered_quant = os.environ.get(
        "SPARSE_VGGT_FUSED_CENTERED_K_QUANT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    fused_source_quant = os.environ.get(
        "SPARSE_VGGT_FUSED_SOURCE_QUANT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    skip_global_k_centering = os.environ.get(
        "SPARSE_VGGT_SKIP_GLOBAL_K_CENTERING", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if fused_source_quant and not fused_centered_quant:
        raise ValueError("source-fused quantization requires fused centered quantization")
    if skip_global_k_centering and not fused_centered_quant:
        raise ValueError("skipping K centering requires fused centered quantization")
    if fused_source_quant:
        query = query.contiguous()
        key = key.contiguous().to(dtype)
        value = value.contiguous().to(dtype)
        km = (
            key[..., :1, :]
            if skip_global_k_centering
            else key.mean(dim=-2, keepdim=True)
        )
    else:
        query, key, value = (
            query.contiguous().to(dtype),
            key.contiguous().to(dtype),
            value.contiguous().to(dtype),
        )
        km = (
            key[..., :1, :]
            if skip_global_k_centering
            else key.mean(dim=-2, keepdim=True)
        )
    if fused_centered_quant:
        from sparse_vggt.kernels.centered_quant import (
            per_block_int8_fused_centered,
        )

        q_int8, q_scale, k_int8, k_scale = per_block_int8_fused_centered(
            query,
            key,
            km,
            center_key=not skip_global_k_centering,
            emulate_fp16_query_source=fused_source_quant,
        )
    else:
        q_int8, q_scale, k_int8, k_scale = per_block_int8(query, key - km)
    q_scale = q_scale.squeeze(-1)
    k_scale = k_scale.squeeze(-1)

    # Get softmax scale
    hd = query.shape[-1]
    scale = 1.0 / (hd**0.5)

    collect_qk_pv_service = os.environ.get(
        "SPARSE_VGGT_COLLECT_QK_PV_SERVICE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    cosa_pair_debt_enabled = os.environ.get(
        "SPARSE_VGGT_COSA_VALUE_RISK_PAIR_DEBT", "0"
    ).lower() in {"1", "true", "yes", "on"}
    cosa_qk_admission_service = os.environ.get(
        "SPARSE_VGGT_COSA_DEBT_SERVICE_DOMAIN", "pv_execution"
    ).lower() == "qk_admission"
    collect_qk_pv_service = (
        collect_qk_pv_service
        or (cosa_pair_debt_enabled and not cosa_qk_admission_service)
    )
    cosa_kernel_pair_service = os.environ.get(
        "SPARSE_VGGT_COSA_KERNEL_PAIR_SERVICE", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if cosa_kernel_pair_service and not collect_qk_pv_service:
        raise RuntimeError("kernel pair service requires PV service collection")

    # SpargeAttn attention kernel
    o = torch.empty(
        query.shape,
        device=query.device,
        dtype=native_output_dtype if native_sparse_output else dtype,
    )
    if pv_threshold_active:
        assert pvthreshd is not None
        pv_count = qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8,
            k_int8,
            value,
            o,
            lut,
            valid_block_num,
            pvthreshd,
            q_scale,
            k_scale,
            1,
            _is_causal,
            1,
            scale,
            int(collect_qk_pv_service),
        )
    else:
        if collect_qk_pv_service:
            raise RuntimeError(
                "layer-gated PV service collection requires the threshold kernel"
            )
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf(
            q_int8,
            k_int8,
            value,
            o,
            lut,
            valid_block_num,
            q_scale,
            k_scale,
            1,
            _is_causal,
            1,
            scale,
        )
        pv_count = None
    cosa_production_fast = os.environ.get(
        "SPARSE_VGGT_COSA_PRODUCTION_FAST", "0"
    ).lower() in {"1", "true", "yes", "on"}
    cosa_pv_count_stats = os.environ.get(
        "SPARSE_VGGT_COSA_PV_COUNT_STATS", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if collect_qk_pv_service:
        q_blocks = valid_block_num.shape[-1]
        if q_blocks <= 0:
            raise RuntimeError("PV service requires at least one Q block")
        if cosa_kernel_pair_service:
            if pv_count.ndim != 4 or pv_count.dtype != torch.int32:
                raise RuntimeError(
                    "compressed kernel pair service must be int32 [B,H,Q,pair]"
                )
            if pv_count.shape[-2] != q_blocks:
                raise RuntimeError("compressed PV service changed the Q layout")
            query_warps_per_block = 4 if hd == 64 else 8
            pair_service_fraction = _reduce_kernel_pair_service_counts(
                pv_count,
                patch_blocks=orig_Kblk,
                query_warps_per_block=query_warps_per_block,
            )
            identity_observed = True
            execution_map = None
            executed = None
            service_fraction = None
            if cosa_pv_count_stats:
                raise RuntimeError(
                    "per-warp PV statistics are unavailable with compressed pair service"
                )
        else:
            pv_query_axis = -2 if pv_count.ndim == 4 else -1
            if pv_count.shape[pv_query_axis] % q_blocks != 0:
                raise RuntimeError(
                    "PV count does not align with QK support domains"
                )
            query_warps = (
                pv_count.shape[-2] if pv_count.ndim == 4 else pv_count.shape[-1]
            )
            warps_per_query_block = query_warps // q_blocks
            identity_observed = pv_count.ndim == 4
            pair_service_fraction = None
            if identity_observed:
                execution_map = pv_count.view(
                    *pv_count.shape[:-2], q_blocks, warps_per_query_block,
                    pv_count.shape[-1]
                )
                if cosa_production_fast and cosa_pair_debt_enabled:
                    pair_service_fraction = _reduce_exact_pv_execution_to_pair(
                        execution_map,
                        patch_blocks=orig_Kblk,
                    )
                    executed = None
                    service_fraction = None
                else:
                    if bool(((execution_map != 0) & (execution_map != 1)).any()):
                        raise RuntimeError("PV execution map must be binary")
                    support = final_map[..., None, :].expand_as(execution_map)
                    if bool((execution_map.bool() & ~support).any()):
                        raise RuntimeError("PV execution escaped the QK support")
                    executed = execution_map.sum(dim=-1).float()
                    service_fraction = execution_map.float().mean(dim=-2)
            elif pv_count.ndim == 3:
                if cosa_production_fast and cosa_pair_debt_enabled:
                    raise RuntimeError("fast CoSA debt requires exact PV identity")
                execution_map = None
                service_fraction = None
                executed = pv_count.view(
                    *pv_count.shape[:-1], q_blocks, warps_per_query_block
                ).float()
            else:
                raise RuntimeError("unexpected SpargeAttn PV observation shape")
        if cosa_production_fast and cosa_pair_debt_enabled:
            if cosa_pv_count_stats and routing_stats_store is not None:
                executed_per_warp = execution_map.sum(dim=-1).float()
                candidate_per_warp = valid_block_num[..., None].float().expand_as(
                    executed_per_warp
                )
                total_candidates = candidate_per_warp.sum().clamp_min(1.0)
                total_executed = executed_per_warp.sum()
                skipped_per_warp = candidate_per_warp - executed_per_warp
                routing_stats_store.update({
                    "qk_pv_service_observed": torch.tensor(
                        1.0, device=execution_map.device
                    ),
                    "qk_pv_service_identity_observed": torch.tensor(
                        1.0, device=execution_map.device
                    ),
                    "qk_pv_warps_per_query_block": torch.tensor(
                        float(warps_per_query_block), device=execution_map.device
                    ),
                    "qk_pv_candidate_slots": total_candidates.detach(),
                    "qk_pv_executed_slots": total_executed.detach(),
                    "qk_pv_skipped_slots": skipped_per_warp.sum().detach(),
                    "qk_pv_executed_fraction": (
                        total_executed / total_candidates
                    ).detach(),
                    "qk_pv_domain_executed_min": (
                        executed_per_warp.min().detach()
                    ),
                    "qk_pv_domain_executed_max": (
                        executed_per_warp.max().detach()
                    ),
                    "qk_pv_domain_skipped_mean": (
                        skipped_per_warp.mean().detach()
                    ),
                    "qk_pv_domain_skipped_max": (
                        skipped_per_warp.max().detach()
                    ),
                })
            if routing_state is not None:
                routing_state["qk_pv_service_observation"] = {
                    "layer_idx": int(layer_idx),
                    "identity_observed": True,
                    "pair_service_fraction": pair_service_fraction.detach(),
                }
        else:
            capacity = valid_block_num[..., None].float()
            if bool((executed < 0).any()) or bool((executed > capacity).any()):
                raise RuntimeError("PV execution count exceeds QK support capacity")
            candidate_slots = capacity.expand_as(executed)
            skipped = candidate_slots - executed
            total_candidates = candidate_slots.sum().clamp_min(1.0)
            total_executed = executed.sum()
            if routing_stats_store is not None:
                routing_stats_store.update({
                    "qk_pv_service_observed": torch.tensor(
                        1.0, device=executed.device
                    ),
                    "qk_pv_service_identity_observed": torch.tensor(
                        float(identity_observed), device=executed.device
                    ),
                    "qk_pv_warps_per_query_block": torch.tensor(
                        float(warps_per_query_block), device=executed.device
                    ),
                    "qk_pv_candidate_slots": total_candidates.detach(),
                    "qk_pv_executed_slots": total_executed.detach(),
                    "qk_pv_skipped_slots": skipped.sum().detach(),
                    "qk_pv_executed_fraction": (
                        total_executed / total_candidates
                    ).detach(),
                    "qk_pv_domain_executed_min": executed.min().detach(),
                    "qk_pv_domain_executed_max": executed.max().detach(),
                    "qk_pv_domain_skipped_mean": skipped.mean().detach(),
                    "qk_pv_domain_skipped_max": skipped.max().detach(),
                })
        if routing_state is not None and not (
            cosa_production_fast and cosa_pair_debt_enabled
        ):
            routing_state["qk_pv_service_observation"] = {
                "layer_idx": int(layer_idx),
                "identity_observed": identity_observed,
                "executed": executed.detach(),
                "capacity": candidate_slots.detach(),
                "skipped": skipped.detach(),
                "support": final_map.detach(),
            }
            if identity_observed:
                routing_state["qk_pv_service_observation"].update({
                    "execution_map": execution_map.detach(),
                    "service_fraction": service_fraction[..., :orig_Kblk].detach(),
                })
    if not native_sparse_output:
        o = o.to(out_dtype)
    if return_sparsity:
        fast_sparsity_accounting = os.environ.get(
            "SPARSE_VGGT_FAST_SPARSITY_ACCOUNTING", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if fast_sparsity_accounting:
            budget_only = os.environ.get(
                "SPARSE_VGGT_COSA_BUDGET_ONLY_UPSTREAM", "0"
            ).lower() in {"1", "true", "yes", "on"}
            if not budget_only or sparse_ratio is None:
                raise ValueError(
                    "fast sparsity accounting requires ratio-only budget routing"
                )
            exact_upstream = os.environ.get(
                "SPARSE_VGGT_OFFICIAL_MASK", "0"
            ).lower() in {"1", "true", "yes", "on"}
            sparsity = _fixed_ratio_capacity_sparsity(
                patch_blocks=orig_Kblk,
                sink_blocks=sink_blocks,
                sparse_ratio=sparse_ratio,
                force_last_patch=has_mixed_last_block and not exact_upstream,
            )
        else:
            sparsity = 1 - final_map.float().mean()
        return o, sparsity
    else:
        return o
