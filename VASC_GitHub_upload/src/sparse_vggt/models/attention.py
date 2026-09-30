import math
import os

import torch
import torch.nn.functional as F
from einops import rearrange

from sparse_vggt.analysis.scout_oracle import (
    analyze_cosa_exact_pair_risk,
    analyze_cosa_postcut_orders,
    analyze_scout_oracle,
)
from sparse_vggt.analysis.service_observation import (
    analyze_grouped_service_observation,
    analyze_transported_representation_debt,
    service_observation_enabled,
)
from sparse_vggt.utils.sparse_wrapper import block_sparse_attn_cuda
from sparse_vggt.kernels.qkv_layout import pack_qkv_patch_then_special
from sparse_vggt.utils.tokens import (
    combine_patch_and_special_frame_major,
    get_patch_tokens,
    get_special_tokens,
    pack_patch_then_special_direct_with_patch,
    reorder_to_patch_then_special,
    reorder_to_patch_then_special_with_patch,
    restore_to_frame_major,
)


def predict_attention(
    query,
    key,
    ks_q=128,
    ks_k=64,
    pool_mode="avg",
    return_key_block_variance=False,
):
    """
    Args:
        query: (B, nh, Tq, C)
        key: (B, nh, Tk, C)

    Return:
        pooled_prob: (B, nh, Tq, Tk)
        key_block_variance: Optional (B, nh, Tk) scalar variance proxy
    """
    assert pool_mode in ["max", "avg"], f"{pool_mode=}"

    pooling_fn = {
        "max": F.max_pool1d,
        "avg": F.avg_pool1d,
    }[pool_mode]

    assert query.ndim == 4, f"{query.shape=}"
    assert key.ndim == 4, f"{key.shape=}"

    B, nh, Tq, C = query.shape
    _, _, Tk, _ = key.shape

    # Query Pooling
    query = rearrange(query, "B nh Tq C -> (B nh) C Tq")
    pooled_query = pooling_fn(query, kernel_size=ks_q, ceil_mode=True)
    pooled_query = rearrange(pooled_query, "(B nh) C Tq -> B nh Tq C", B=B, nh=nh)

    # Key Pooling
    key = rearrange(key, "B nh Tk C -> (B nh) C Tk")
    pooled_key_channels = pooling_fn(key, kernel_size=ks_k, ceil_mode=True)

    key_block_variance = None
    if return_key_block_variance:
        if pool_mode == "avg":
            key_block_mean = pooled_key_channels
        else:
            key_block_mean = F.avg_pool1d(
                key, kernel_size=ks_k, ceil_mode=True
            )
        token_squared_norm = (key * key).sum(
            dim=1, keepdim=True, dtype=torch.float32
        )
        mean_squared_norm = F.avg_pool1d(
            token_squared_norm, kernel_size=ks_k, ceil_mode=True
        ).squeeze(1)
        squared_mean_norm = key_block_mean.float().square().sum(dim=1)
        key_block_variance = (mean_squared_norm - squared_mean_norm).clamp_min_(0.0)
        key_block_variance = rearrange(
            key_block_variance,
            "(B nh) Tk -> B nh Tk",
            B=B,
            nh=nh,
        )

    pooled_key = rearrange(
        pooled_key_channels, "(B nh) C Tk -> B nh Tk C", B=B, nh=nh
    )

    # Dot Product
    scale = 1 / math.sqrt(C)
    pooled_score = pooled_query @ pooled_key.transpose(-1, -2) * scale  # (B, nh, Tq, Tk)
    pooled_prob = F.softmax(pooled_score, dim=-1)

    if return_key_block_variance:
        return pooled_prob, key_block_variance
    return pooled_prob


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def resolve_hps_complement_correction(
    mode: str,
    selected_cells: int,
    candidate_cells: int,
) -> bool:
    """Choose the cheaper exact MC correction side."""
    normalized = mode.strip().lower()
    if normalized in {"1", "true", "yes", "on", "complement"}:
        return True
    if normalized in {"0", "false", "no", "off", "selected"}:
        return False
    if normalized != "auto":
        raise ValueError(
            "HPS complement correction must be auto, selected, or complement"
        )
    if not 0 <= selected_cells <= candidate_cells:
        raise ValueError(
            "HPS selected cells must fit the correction candidate set"
        )
    return selected_cells * 2 > candidate_cells


def resolve_direct_attention_tile(
    num_frames: int,
    profile: str,
    block_m: int,
    block_n: int,
) -> tuple[int, int]:
    """Resolve a manual, frozen HPS, or measured speed tile profile."""
    if profile == "manual":
        return block_m, block_n
    if profile == "hps_v02":
        return 128, 64 if num_frames <= 112 else 32
    if profile == "speed":
        return 128, 32
    raise ValueError(
        "direct tile profile must be manual, hps_v02, or speed"
    )


def resolve_direct_attention_stages(
    num_frames: int,
    profile: str,
    requested_stages: int,
) -> int | None:
    """Resolve the profile stage count unless explicitly overridden."""
    if requested_stages:
        return requested_stages
    if profile == "hps_v02":
        return 2 if num_frames <= 112 else 3
    if profile == "speed":
        return 3
    return None


def hierarchical_refinement_counts(
    requested_residual_tokens: int,
    max_remote_cells: int,
) -> tuple[int, int, int]:
    """Maximize +3 spatial coverage, then spend surplus on +15 service."""
    if requested_residual_tokens < 0 or max_remote_cells < 0:
        raise ValueError("hierarchical refinement budgets must be non-negative")
    quadrant_cells = min(
        max_remote_cells,
        requested_residual_tokens // 3,
    )
    remaining_tokens = requested_residual_tokens - 3 * quadrant_cells
    dense_cells = min(quadrant_cells, remaining_tokens // 12)
    quadrant_cells -= dense_cells
    used_tokens = 15 * dense_cells + 3 * quadrant_cells
    return dense_cells, quadrant_cells, used_tokens


def radial_width(
    frame_i: int,
    frame_j: int,
    token_per_frame: int,
    block_size: int = 128,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
) -> int:
    """Lightweight VGGT version of Radial Attention's frame-pair window width."""
    dist = abs(frame_i - frame_j)
    if dist <= dense_neighbor:
        return token_per_frame

    group = dist.bit_length()
    width = (2 ** token_per_frame.bit_length()) / (2**group) * decay_factor
    return int(max(block_size, min(token_per_frame, width)))


def build_radial_a_mask(
    num_frames: int,
    patch_tokens_per_frame: int,
    device,
    block_size: int = 128,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
):
    patch_total = num_frames * patch_tokens_per_frame
    mask = torch.zeros((patch_total, patch_total), device=device, dtype=torch.bool)
    idx = torch.arange(patch_tokens_per_frame, device=device)
    distance = (idx[:, None] - idx[None, :]).abs()

    for query_frame in range(num_frames):
        query_start = query_frame * patch_tokens_per_frame
        query_end = query_start + patch_tokens_per_frame
        for key_frame in range(num_frames):
            key_start = key_frame * patch_tokens_per_frame
            key_end = key_start + patch_tokens_per_frame
            width = radial_width(
                query_frame,
                key_frame,
                token_per_frame=patch_tokens_per_frame,
                block_size=block_size,
                decay_factor=decay_factor,
                dense_neighbor=dense_neighbor,
            )
            mask[query_start:query_end, key_start:key_end] = distance <= width
    return mask


def hilbert_patch_reorder(x, H: int, W: int, inverse: bool = False):
    """Reorder patch tokens inside each frame with a Hilbert curve."""
    from sparse_vggt.utils.hilbert import make_hilbert_gather_idx

    B, nh, patch_total, hd = x.shape
    patch_tokens_per_frame = H * W
    assert patch_total % patch_tokens_per_frame == 0, f"{patch_total=}, {H=}, {W=}"
    N = patch_total // patch_tokens_per_frame

    idx = make_hilbert_gather_idx(W, H, inverse=inverse)
    idx = torch.tensor(idx, dtype=torch.long, device=x.device)
    x = x.view(B, nh, N, patch_tokens_per_frame, hd)
    x = x.index_select(-2, idx)
    return x.reshape(B, nh, patch_total, hd).contiguous()


def pool_spatial_patch_tokens(
    x: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    pool_size: int,
) -> torch.Tensor:
    """Pool per-frame patch K/V without mixing frames or attention heads."""
    if pool_size < 1:
        raise ValueError("pool_size must be at least 1")

    batch, heads, token_count, head_dim = x.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "patch token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )
    if pool_size == 1:
        return x.view(batch, heads, num_frames, tokens_per_frame, head_dim)

    spatial = x.view(
        batch, heads, num_frames, height, width, head_dim
    ).permute(0, 1, 2, 5, 3, 4)
    spatial = spatial.reshape(
        batch * heads * num_frames, head_dim, height, width
    )
    pooled = F.avg_pool2d(
        spatial,
        kernel_size=pool_size,
        stride=pool_size,
        ceil_mode=True,
    )
    pooled_height = math.ceil(height / pool_size)
    pooled_width = math.ceil(width / pool_size)
    pooled = pooled.view(
        batch,
        heads,
        num_frames,
        head_dim,
        pooled_height,
        pooled_width,
    ).permute(0, 1, 2, 4, 5, 3)
    return pooled.reshape(
        batch,
        heads,
        num_frames,
        pooled_height * pooled_width,
        head_dim,
    ).contiguous()


def spatial_pool_multiplicity(
    height: int,
    width: int,
    pool_size: int,
    *,
    device,
    dtype,
) -> torch.Tensor:
    """Return how many original patches each ceil-mode pool represents."""
    cache_key = (
        height,
        width,
        pool_size,
        _device_cache_key(torch.device(device)),
        dtype,
    )
    if _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False):
        cached = _SPATIAL_POOL_MULTIPLICITY_CACHE.get(cache_key)
        if cached is not None:
            return cached
    row_starts = torch.arange(0, height, pool_size, device=device)
    col_starts = torch.arange(0, width, pool_size, device=device)
    row_counts = (height - row_starts).clamp_max(pool_size)
    col_counts = (width - col_starts).clamp_max(pool_size)
    result = (row_counts[:, None] * col_counts[None, :]).to(
        dtype=dtype
    ).flatten()
    if _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False):
        _SPATIAL_POOL_MULTIPLICITY_CACHE[cache_key] = result
    return result


def stride_spatial_patch_tokens(
    x: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    stride: int,
    phase: int | tuple[int, ...] | list[int],
) -> torch.Tensor:
    """Pick real tokens from each spatial cell at one or more phases."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    batch, heads, token_count, head_dim = x.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "patch token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )
    phases = (phase,) if isinstance(phase, int) else tuple(phase)
    if not phases:
        raise ValueError("at least one phase is required")
    if len(phases) > stride * stride:
        raise ValueError("phase count cannot exceed stride squared")

    row_starts = torch.arange(0, height, stride, device=x.device)
    col_starts = torch.arange(0, width, stride, device=x.device)
    selected_indices = []
    for current_phase in phases:
        current_phase %= stride * stride
        row_offset, col_offset = divmod(current_phase, stride)
        rows = (row_starts + row_offset).clamp_max(height - 1)
        cols = (col_starts + col_offset).clamp_max(width - 1)
        selected_indices.append((rows[:, None] * width + cols[None, :]).flatten())
    # Partial border cells can map multiple phases to the same real patch.
    # unique() prevents those tokens from receiving accidental extra weight.
    selected_indices = torch.unique(torch.cat(selected_indices), sorted=True)
    spatial = x.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    return spatial.index_select(3, selected_indices).contiguous()


def select_detail_spatial_representatives(
    key: torch.Tensor,
    value: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    cell_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one real K/V per cell using within-cell V detail residual."""
    if key.shape != value.shape:
        raise ValueError("key and value must have identical shapes")
    if cell_size < 1:
        raise ValueError("cell_size must be at least 1")
    batch, heads, token_count, head_dim = key.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "patch token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )

    cell_rows = math.ceil(height / cell_size)
    cell_cols = math.ceil(width / cell_size)
    padded_height = cell_rows * cell_size
    padded_width = cell_cols * cell_size
    key_spatial = key.view(
        batch, heads, num_frames, height, width, head_dim
    )
    value_spatial = value.view(
        batch, heads, num_frames, height, width, head_dim
    )
    padded_key = key.new_zeros(
        batch,
        heads,
        num_frames,
        padded_height,
        padded_width,
        head_dim,
    )
    padded_value = value.new_zeros(padded_key.shape)
    padded_key[..., :height, :width, :] = key_spatial
    padded_value[..., :height, :width, :] = value_spatial

    def to_cells(x: torch.Tensor) -> torch.Tensor:
        return x.view(
            batch,
            heads,
            num_frames,
            cell_rows,
            cell_size,
            cell_cols,
            cell_size,
            head_dim,
        ).permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(
            batch,
            heads,
            num_frames,
            cell_rows * cell_cols,
            cell_size * cell_size,
            head_dim,
        )

    key_cells = to_cells(padded_key)
    value_cells = to_cells(padded_value)
    valid = torch.zeros(
        (padded_height, padded_width),
        device=key.device,
        dtype=torch.bool,
    )
    valid[:height, :width] = True
    valid_cells = valid.view(
        cell_rows, cell_size, cell_cols, cell_size
    ).permute(0, 2, 1, 3).reshape(
        cell_rows * cell_cols, cell_size * cell_size
    )
    valid_weights = valid_cells.view(
        1, 1, 1, cell_rows * cell_cols, cell_size * cell_size, 1
    ).to(dtype=value.dtype)
    cell_mean = (value_cells * valid_weights).sum(dim=-2, keepdim=True)
    cell_mean = cell_mean / valid_weights.sum(dim=-2, keepdim=True).clamp_min(1)
    detail_score = (value_cells - cell_mean).float().square().mean(dim=-1)
    detail_score = detail_score.masked_fill(
        ~valid_cells.view(1, 1, 1, cell_rows * cell_cols, -1),
        float("-inf"),
    )
    selected_phase = detail_score.argmax(dim=-1)
    gather_index = selected_phase[..., None, None].expand(
        batch,
        heads,
        num_frames,
        cell_rows * cell_cols,
        1,
        head_dim,
    )
    selected_key = key_cells.gather(-2, gather_index).squeeze(-2)
    selected_value = value_cells.gather(-2, gather_index).squeeze(-2)
    return selected_key.contiguous(), selected_value.contiguous()


_FOUR_BY_FOUR_PHASE_ORDER = (
    0, 15, 3, 12, 5, 10, 6, 9,
    1, 14, 2, 13, 4, 11, 7, 8,
)

_HEAD_SHARDED_INDEX_CACHE = {}
_HEAD_ROTATING_INDEX_CACHE = {}
_PARENT_CHILD_MAP_CACHE = {}
_CHILD_PARENT_REVERSE_CACHE = {}
_SPATIAL_POOL_MULTIPLICITY_CACHE = {}
_ALL_PARENT_MASS_CACHE = {}
_REMOTE_PARENT_INDEX_CACHE = {}


def _device_cache_key(device: torch.device) -> tuple[str, int | None]:
    return device.type, device.index


def remote_parent_index_grid(
    num_frames: int,
    parents_per_frame: int,
    *,
    device,
) -> torch.Tensor:
    """Cache global parent indices for every query frame's remote frames."""
    if num_frames < 1 or parents_per_frame < 1:
        raise ValueError("remote parent dimensions must be positive")
    device = torch.device(device)
    cache_key = (
        num_frames,
        parents_per_frame,
        _device_cache_key(device),
    )
    cached = _REMOTE_PARENT_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parent_total = num_frames * parents_per_frame
    all_parents = torch.arange(parent_total, device=device)
    source_frames = torch.div(
        all_parents,
        parents_per_frame,
        rounding_mode="floor",
    )
    query_frames = torch.arange(num_frames, device=device)
    remote = all_parents.view(1, -1).expand(
        num_frames, -1
    )[source_frames.view(1, -1) != query_frames.view(-1, 1)].view(
        num_frames,
        (num_frames - 1) * parents_per_frame,
    ).contiguous()
    _REMOTE_PARENT_INDEX_CACHE[cache_key] = remote
    return remote


def rotating_spatial_phases(
    stride: int,
    samples_per_cell: int,
    layer_phase: int,
    phase_mode: str = "default",
) -> tuple[int, ...]:
    """Choose complementary real-token phases and rotate them by layer."""
    phase_count = stride * stride
    if not 1 <= samples_per_cell <= phase_count:
        raise ValueError("samples_per_cell must be in [1, stride squared]")
    if phase_mode not in {"default", "quadrant_balanced"}:
        raise ValueError("phase_mode must be default or quadrant_balanced")
    if phase_mode == "quadrant_balanced":
        if stride != 4:
            raise ValueError("quadrant-balanced phases require stride=4")
        phase_order = _FOUR_BY_FOUR_PHASE_ORDER
    elif stride == 2:
        # Diagonal pairs avoid a per-layer horizontal or vertical bias.
        phase_order = (0, 3, 1, 2)
    else:
        phase_order = tuple(range(phase_count))
    start = (layer_phase * samples_per_cell) % phase_count
    return tuple(
        phase_order[(start + offset) % phase_count]
        for offset in range(samples_per_cell)
    )


def head_sharded_spatial_phases(
    stride: int,
    num_heads: int,
    layer_idx: int,
    mode: str = "rotating",
) -> tuple[int, ...]:
    """Assign one spatial phase to each head, optionally rotating by depth."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if num_heads < 1:
        raise ValueError("num_heads must be at least 1")
    if mode not in {"rotating", "fixed", "anchored", "anchored_heads"}:
        raise ValueError(
            "phase mode must be rotating, fixed, anchored, "
            "or anchored_heads"
        )
    phase_count = stride * stride
    phase_order = (
        _FOUR_BY_FOUR_PHASE_ORDER
        if stride == 4
        else tuple(range(phase_count))
    )
    if mode in {"anchored", "anchored_heads"} and phase_count > 1:
        anchor_count = min(
            num_heads,
            max(1, phase_count // 4),
            phase_count - 1,
        )
        anchor_phases = phase_order[:anchor_count]
        rotating_phases = phase_order[anchor_count:]
        return tuple(
            anchor_phases[head_idx]
            if head_idx < anchor_count
            else rotating_phases[
                (head_idx - anchor_count + layer_idx)
                % len(rotating_phases)
            ]
            for head_idx in range(num_heads)
        )
    layer_offset = layer_idx if mode == "rotating" else 0
    return tuple(
        phase_order[(head_idx + layer_offset) % phase_count]
        for head_idx in range(num_heads)
    )


def head_sharded_spatial_patch_tokens(
    x: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    stride: int,
    layer_idx: int,
    phase_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather one phase per head and report valid heads for border cells."""
    batch, heads, token_count, head_dim = x.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError("patch token count does not match the frame grid")
    phase_count = stride * stride
    if phase_mode == "rotating":
        phase_key = layer_idx % phase_count
    elif (
        phase_mode in {"anchored", "anchored_heads"}
        and phase_count > 1
    ):
        anchor_count = min(
            heads,
            max(1, phase_count // 4),
            phase_count - 1,
        )
        phase_key = layer_idx % (phase_count - anchor_count)
    else:
        phase_key = 0
    cache_key = (
        height,
        width,
        stride,
        heads,
        phase_key,
        phase_mode,
        _device_cache_key(x.device),
    )
    cached = _HEAD_SHARDED_INDEX_CACHE.get(cache_key)
    if cached is None:
        phases = head_sharded_spatial_phases(
            stride, heads, layer_idx, phase_mode
        )
        row_starts = torch.arange(0, height, stride, device=x.device)
        col_starts = torch.arange(0, width, stride, device=x.device)
        head_indices = []
        head_valid = []
        for phase in phases:
            row_offset, col_offset = divmod(phase, stride)
            raw_rows = row_starts + row_offset
            raw_cols = col_starts + col_offset
            valid = (
                (raw_rows[:, None] < height)
                & (raw_cols[None, :] < width)
            ).flatten()
            rows = raw_rows.clamp_max(height - 1)
            cols = raw_cols.clamp_max(width - 1)
            head_indices.append(
                (rows[:, None] * width + cols[None, :]).flatten()
            )
            head_valid.append(valid)
        cached = torch.stack(head_indices), torch.stack(head_valid)
        _HEAD_SHARDED_INDEX_CACHE[cache_key] = cached
    base_indices, head_valid = cached
    gather_indices = base_indices.view(
        1, heads, 1, -1, 1
    ).expand(batch, heads, num_frames, -1, head_dim)
    spatial = x.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    gathered = spatial.gather(3, gather_indices).contiguous()
    return gathered, head_valid


def spatial_parent_child_map(
    height: int,
    width: int,
    child_stride: int,
    parent_stride: int,
    *,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map coarse parent cells to their spatially contained child cells."""
    if parent_stride < child_stride:
        raise ValueError("parent stride must be at least the child stride")
    if parent_stride % child_stride != 0:
        raise ValueError("parent stride must be divisible by child stride")
    cache_key = (
        height,
        width,
        child_stride,
        parent_stride,
        _device_cache_key(torch.device(device)),
    )
    cached = _PARENT_CHILD_MAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    child_rows = math.ceil(height / child_stride)
    child_cols = math.ceil(width / child_stride)
    parent_rows = math.ceil(height / parent_stride)
    parent_cols = math.ceil(width / parent_stride)
    side = parent_stride // child_stride
    parent_to_children = []
    parent_child_valid = []
    for parent_row in range(parent_rows):
        for parent_col in range(parent_cols):
            indices = []
            valid = []
            for row_offset in range(side):
                for col_offset in range(side):
                    child_row = parent_row * side + row_offset
                    child_col = parent_col * side + col_offset
                    is_valid = (
                        child_row < child_rows and child_col < child_cols
                    )
                    clamped_row = min(child_row, child_rows - 1)
                    clamped_col = min(child_col, child_cols - 1)
                    indices.append(clamped_row * child_cols + clamped_col)
                    valid.append(is_valid)
            valid_indices = [
                index for index, is_valid in zip(indices, valid) if is_valid
            ]
            packed_indices = valid_indices + [
                valid_indices[-1]
            ] * (len(indices) - len(valid_indices))
            parent_to_children.append(packed_indices)
            parent_child_valid.append(
                [True] * len(valid_indices)
                + [False] * (len(indices) - len(valid_indices))
            )
    result = (
        torch.tensor(parent_to_children, device=device, dtype=torch.long),
        torch.tensor(parent_child_valid, device=device, dtype=torch.bool),
    )
    _PARENT_CHILD_MAP_CACHE[cache_key] = result
    return result


def spatial_child_parent_map(
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    children_per_frame: int,
) -> torch.Tensor:
    """Return the static reverse map without rebuilding it every layer."""
    if not _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False):
        child_to_parent = torch.empty(
            children_per_frame,
            device=parent_to_children.device,
            dtype=torch.long,
        )
        for parent_idx in range(parent_to_children.shape[0]):
            valid_children = parent_to_children[
                parent_idx, parent_child_valid[parent_idx]
            ]
            child_to_parent[valid_children] = parent_idx
        return child_to_parent
    cache_key = (
        parent_to_children.data_ptr(),
        parent_child_valid.data_ptr(),
        children_per_frame,
        _device_cache_key(parent_to_children.device),
    )
    cached = _CHILD_PARENT_REVERSE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parent_ids = torch.arange(
        parent_to_children.shape[0],
        device=parent_to_children.device,
        dtype=torch.long,
    )[:, None].expand_as(parent_to_children)
    child_to_parent = torch.empty(
        children_per_frame,
        device=parent_to_children.device,
        dtype=torch.long,
    )
    child_to_parent[parent_to_children[parent_child_valid]] = parent_ids[
        parent_child_valid
    ]
    _CHILD_PARENT_REVERSE_CACHE[cache_key] = child_to_parent
    return child_to_parent


def spatial_all_parent_mass(
    child_masses: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    num_frames: int,
) -> torch.Tensor:
    """Cache static per-frame parent masses for additive service."""
    if not _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False):
        parent_mass = (
            child_masses[parent_to_children]
            * parent_child_valid.to(dtype=torch.float32)
        ).sum(dim=-1)
        return parent_mass.repeat(num_frames)
    cache_key = (
        child_masses.data_ptr(),
        parent_to_children.data_ptr(),
        parent_child_valid.data_ptr(),
        num_frames,
        _device_cache_key(parent_to_children.device),
    )
    cached = _ALL_PARENT_MASS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    parent_mass = (
        child_masses[parent_to_children]
        * parent_child_valid.to(dtype=torch.float32)
    ).sum(dim=-1)
    all_parent_mass = parent_mass.repeat(num_frames)
    _ALL_PARENT_MASS_CACHE[cache_key] = all_parent_mass
    return all_parent_mass


def ordered_parent_child_services(
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    layer_idx: int,
    phase_mode: str,
) -> torch.Tensor:
    """Order child services with optional persistent first-child anchors."""
    if phase_mode not in {
        "rotating",
        "fixed",
        "anchored",
        "anchored_heads",
    }:
        raise ValueError("unsupported parent-child phase mode")
    max_children = parent_to_children.shape[-1]
    ordered_rows = []
    for parent_idx in range(parent_to_children.shape[0]):
        valid_count = int(parent_child_valid[parent_idx].sum().item())
        valid_children = parent_to_children[parent_idx, :valid_count]
        if phase_mode == "anchored" and valid_count > 1:
            anchor = valid_children[:1]
            rotating = valid_children[1:]
            shift = layer_idx % rotating.numel()
            ordered = torch.cat(
                (anchor, torch.roll(rotating, shifts=-shift, dims=0))
            )
        else:
            # Preserve the existing child rotation for both legacy parent
            # modes. Only anchored mode reserves a persistent child slot.
            shift = layer_idx % valid_count
            ordered = torch.roll(valid_children, shifts=-shift, dims=0)
        if valid_count < max_children:
            ordered = torch.cat(
                (
                    ordered,
                    ordered[-1:].expand(max_children - valid_count),
                )
            )
        ordered_rows.append(ordered)
    return torch.stack(ordered_rows)


def aggregate_child_cells_to_parents(
    child_values: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
) -> torch.Tensor:
    """Average child-cell values inside each parent without border repeats."""
    gathered = child_values[..., parent_to_children]
    weights = parent_child_valid.to(gathered.dtype)
    return (gathered * weights).sum(dim=-1) / weights.sum(
        dim=-1
    ).clamp_min(1.0)


def head_rotating_spatial_patch_tokens(
    x: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    stride: int,
    samples_per_cell: int,
    layer_phase: int,
    phase_mode: str = "default",
) -> torch.Tensor:
    """Pick complementary spatial phases across attention heads."""
    if stride < 1:
        raise ValueError("stride must be at least 1")
    batch, heads, token_count, head_dim = x.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "patch token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )
    cache_key = (
        height,
        width,
        stride,
        samples_per_cell,
        heads,
        layer_phase % (stride * stride),
        phase_mode,
        _device_cache_key(x.device),
    )
    base_indices = _HEAD_ROTATING_INDEX_CACHE.get(cache_key)
    if base_indices is None:
        row_starts = torch.arange(0, height, stride, device=x.device)
        col_starts = torch.arange(0, width, stride, device=x.device)
        head_indices = []
        for head_id in range(heads):
            phases = rotating_spatial_phases(
                stride,
                samples_per_cell,
                layer_phase + head_id,
                phase_mode,
            )
            selected_indices = []
            for phase in phases:
                row_offset, col_offset = divmod(phase, stride)
                rows = (row_starts + row_offset).clamp_max(height - 1)
                cols = (col_starts + col_offset).clamp_max(width - 1)
                selected_indices.append(
                    (rows[:, None] * width + cols[None, :]).flatten()
                )
            # Partial border cells keep one entry per cell and phase.
            head_indices.append(torch.cat(selected_indices))
        base_indices = torch.stack(head_indices)
        _HEAD_ROTATING_INDEX_CACHE[cache_key] = base_indices
    gather_indices = base_indices.view(
        1, heads, 1, -1, 1
    ).expand(batch, heads, num_frames, -1, head_dim)
    spatial = x.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    return spatial.gather(3, gather_indices).contiguous()


def spatial_phase_patch_indices(
    height: int,
    width: int,
    stride: int,
    phase: int,
    *,
    device,
) -> torch.Tensor:
    """Flattened query-patch indices belonging to one spatial phase."""
    phase %= stride * stride
    row_offset, col_offset = divmod(phase, stride)
    rows = torch.arange(row_offset, height, stride, device=device)
    cols = torch.arange(col_offset, width, stride, device=device)
    return (rows[:, None] * width + cols[None, :]).flatten()


def query_aligned_multiresolution_patch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    local_frame_radius: int,
    remote_stride: int,
    query_frame_chunk: int,
    special_key: torch.Tensor | None,
    special_value: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use each query patch's spatial phase for remote real-token routing."""
    batch, heads, token_count, head_dim = query.shape
    tokens_per_frame = height * width
    query_frames = query.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    key_frames = key.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    value_frames = value.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    frame_ids = torch.arange(num_frames, device=query.device)
    output = None

    for phase in range(remote_stride * remote_stride):
        query_indices = spatial_phase_patch_indices(
            height,
            width,
            remote_stride,
            phase,
            device=query.device,
        )
        if query_indices.numel() == 0:
            continue
        remote_key_frames = stride_spatial_patch_tokens(
            key,
            num_frames,
            height,
            width,
            remote_stride,
            phase,
        )
        remote_value_frames = stride_spatial_patch_tokens(
            value,
            num_frames,
            height,
            width,
            remote_stride,
            phase,
        )
        remote_tokens_per_frame = remote_key_frames.shape[-2]

        for chunk_start in range(0, num_frames, query_frame_chunk):
            chunk_end = min(chunk_start + query_frame_chunk, num_frames)
            chunk_centers = frame_ids[chunk_start:chunk_end]
            is_local = (
                chunk_centers[:, None] - frame_ids[None, :]
            ).abs() <= local_frame_radius
            local_counts = is_local.sum(dim=-1)
            for local_count_tensor in torch.unique(local_counts):
                local_count = int(local_count_tensor.item())
                centers = chunk_centers[local_counts == local_count_tensor]
                center_mask = is_local[local_counts == local_count_tensor]
                group_size = centers.numel()
                expanded_frames = frame_ids.expand(group_size, num_frames)
                local_indices = expanded_frames[center_mask].view(
                    group_size, local_count
                )
                remote_count = num_frames - local_count
                remote_indices = expanded_frames[~center_mask].view(
                    group_size, remote_count
                )

                exact_key = key_frames[:, :, local_indices, :, :].reshape(
                    batch,
                    heads,
                    group_size,
                    local_count * tokens_per_frame,
                    head_dim,
                )
                exact_value = value_frames[:, :, local_indices, :, :].reshape(
                    batch,
                    heads,
                    group_size,
                    local_count * tokens_per_frame,
                    head_dim,
                )
                key_parts = [exact_key]
                value_parts = [exact_value]
                if remote_count > 0:
                    key_parts.append(
                        remote_key_frames[:, :, remote_indices, :, :].reshape(
                            batch,
                            heads,
                            group_size,
                            remote_count * remote_tokens_per_frame,
                            head_dim,
                        )
                    )
                    value_parts.append(
                        remote_value_frames[:, :, remote_indices, :, :].reshape(
                            batch,
                            heads,
                            group_size,
                            remote_count * remote_tokens_per_frame,
                            head_dim,
                        )
                    )
                if special_key is not None and special_key.shape[-2] > 0:
                    special_tokens = special_key.shape[-2]
                    key_parts.append(
                        special_key.unsqueeze(2).expand(
                            batch, heads, group_size, special_tokens, head_dim
                        )
                    )
                    value_parts.append(
                        special_value.unsqueeze(2).expand(
                            batch, heads, group_size, special_tokens, head_dim
                        )
                    )

                selected_query = query_frames[:, :, centers, :, :].index_select(
                    3, query_indices
                )
                group_output = F.scaled_dot_product_attention(
                    selected_query,
                    torch.cat(key_parts, dim=-2),
                    torch.cat(value_parts, dim=-2),
                )
                if output is None:
                    output = torch.empty(
                        query_frames.shape,
                        device=query.device,
                        dtype=group_output.dtype,
                    )
                for group_index, center in enumerate(centers.tolist()):
                    output[:, :, center, :, :].index_copy_(
                        2,
                        query_indices,
                        group_output[:, :, group_index, :, :],
                    )

    if output is None:
        raise ValueError("num_frames and the spatial grid must be positive")

    remote_tokens_per_frame = math.ceil(height / remote_stride) * math.ceil(
        width / remote_stride
    )
    local_counts = torch.stack(
        [
            ((frame_ids - center).abs() <= local_frame_radius).sum()
            for center in frame_ids
        ]
    ).sum().item()
    total_local_keys = local_counts * tokens_per_frame
    total_remote_keys = (
        num_frames * num_frames - local_counts
    ) * remote_tokens_per_frame
    selected_patch_keys = total_local_keys + total_remote_keys
    dense_patch_keys = num_frames * num_frames * tokens_per_frame
    selected_key_fraction = selected_patch_keys / dense_patch_keys
    special_tokens = 0 if special_key is None else special_key.shape[-2]
    mean_patch_keys = selected_patch_keys / num_frames
    effective_key_fraction = (
        mean_patch_keys + special_tokens
    ) / (num_frames * tokens_per_frame + special_tokens)
    stats = {
        "multiresolution_patch_sparsity": 1.0 - selected_key_fraction,
        "multiresolution_effective_sparsity": 1.0 - effective_key_fraction,
        "multiresolution_local_exact_fraction": (
            total_local_keys / selected_patch_keys
        ),
        "multiresolution_remote_pooled_fraction": (
            total_remote_keys / selected_patch_keys
        ),
        "multiresolution_mean_patch_keys": mean_patch_keys,
        "multiresolution_pooled_tokens_per_frame": float(
            remote_tokens_per_frame
        ),
        "multiresolution_carrier_tokens": float(special_tokens),
        "multiresolution_local_radius": float(local_frame_radius),
        "multiresolution_remote_pool_size": float(remote_stride),
        "multiresolution_remote_mode_id": 1.0,
        "multiresolution_remote_phase": -1.0,
        "multiresolution_remote_samples_per_cell": 1.0,
        "multiresolution_area_bias": 0.0,
        "multiresolution_query_aligned": 1.0,
    }
    return output.reshape(batch, heads, token_count, head_dim), stats


def multiresolution_patch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    local_frame_radius: int = 2,
    remote_pool_size: int = 2,
    query_frame_chunk: int = 4,
    remote_mode: str = "strided",
    remote_phase: int = 0,
    remote_samples_per_cell: int = 1,
    remote_phase_mode: str = "layer",
    use_area_bias: bool = False,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Attend to exact nearby frames and pooled remote frames.

    Every query keeps all frames connected. Nearby K/V remain at patch
    resolution, while remote K/V are represented by non-overlapping spatial
    averages. Existing special tokens are appended as global carrier tokens.
    """
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if query_frame_chunk < 1:
        raise ValueError("query_frame_chunk must be at least 1")
    if remote_mode not in {"avg", "strided", "detail"}:
        raise ValueError("remote_mode must be 'avg', 'strided', or 'detail'")
    if remote_phase_mode not in {"layer", "query", "head"}:
        raise ValueError(
            "remote_phase_mode must be 'layer', 'query', or 'head'"
        )
    if not 1 <= remote_samples_per_cell <= remote_pool_size**2:
        raise ValueError(
            "remote_samples_per_cell must be in [1, remote_pool_size squared]"
        )
    if remote_mode == "avg" and remote_samples_per_cell != 1:
        raise ValueError("avg remote mode supports exactly one sample per cell")
    if remote_mode == "detail" and remote_samples_per_cell != 1:
        raise ValueError("detail remote mode supports exactly one sample per cell")
    if use_area_bias and remote_samples_per_cell != 1:
        raise ValueError("area bias currently requires one remote sample per cell")
    if remote_phase_mode == "query":
        if remote_mode != "strided":
            raise ValueError("query phase mode requires strided remote tokens")
        if remote_samples_per_cell != 1:
            raise ValueError("query phase mode currently requires one sample per cell")
        if use_area_bias:
            raise ValueError("query phase mode does not use area bias")
        return query_aligned_multiresolution_patch_attention(
            query,
            key,
            value,
            num_frames=num_frames,
            height=height,
            width=width,
            local_frame_radius=local_frame_radius,
            remote_stride=remote_pool_size,
            query_frame_chunk=query_frame_chunk,
            special_key=special_key,
            special_value=special_value,
        )
    if remote_phase_mode == "head" and remote_mode != "strided":
        raise ValueError("head phase mode requires strided remote tokens")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if (special_key is None) != (special_value is None):
        raise ValueError("special_key and special_value must be provided together")
    if special_key is not None and special_key.shape != special_value.shape:
        raise ValueError("special_key and special_value must have identical shapes")

    batch, heads, token_count, head_dim = query.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "query token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )

    query_frames = query.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    key_frames = key.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    value_frames = value.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    if remote_mode == "avg":
        pooled_key = pool_spatial_patch_tokens(
            key, num_frames, height, width, remote_pool_size
        )
        pooled_value = pool_spatial_patch_tokens(
            value, num_frames, height, width, remote_pool_size
        )
    elif remote_mode == "strided":
        if remote_phase_mode == "head":
            pooled_key = head_rotating_spatial_patch_tokens(
                key,
                num_frames,
                height,
                width,
                remote_pool_size,
                remote_samples_per_cell,
                remote_phase,
            )
            pooled_value = head_rotating_spatial_patch_tokens(
                value,
                num_frames,
                height,
                width,
                remote_pool_size,
                remote_samples_per_cell,
                remote_phase,
            )
        else:
            remote_phases = rotating_spatial_phases(
                remote_pool_size,
                remote_samples_per_cell,
                remote_phase,
            )
            pooled_key = stride_spatial_patch_tokens(
                key,
                num_frames,
                height,
                width,
                remote_pool_size,
                remote_phases,
            )
            pooled_value = stride_spatial_patch_tokens(
                value,
                num_frames,
                height,
                width,
                remote_pool_size,
                remote_phases,
            )
    else:
        pooled_key, pooled_value = select_detail_spatial_representatives(
            key,
            value,
            num_frames,
            height,
            width,
            remote_pool_size,
        )
    pooled_tokens_per_frame = pooled_key.shape[-2]
    pooled_multiplicity = spatial_pool_multiplicity(
        height,
        width,
        remote_pool_size,
        device=query.device,
        dtype=query.dtype,
    )
    output = None
    frame_ids = torch.arange(num_frames, device=query.device)

    total_local_keys = 0
    total_remote_keys = 0
    for chunk_start in range(0, num_frames, query_frame_chunk):
        chunk_end = min(chunk_start + query_frame_chunk, num_frames)
        chunk_centers = frame_ids[chunk_start:chunk_end]
        is_local = (
            chunk_centers[:, None] - frame_ids[None, :]
        ).abs() <= local_frame_radius
        local_counts = is_local.sum(dim=-1)

        # Edge frames have fewer valid neighbors. Group equal-length contexts
        # so each SDPA call remains mask-free and kernel-friendly.
        for local_count_tensor in torch.unique(local_counts):
            local_count = int(local_count_tensor.item())
            centers = chunk_centers[local_counts == local_count_tensor]
            center_mask = is_local[local_counts == local_count_tensor]
            group_size = centers.numel()

            expanded_frames = frame_ids.expand(group_size, num_frames)
            local_indices = expanded_frames[center_mask].view(
                group_size, local_count
            )
            remote_count = num_frames - local_count
            remote_indices = expanded_frames[~center_mask].view(
                group_size, remote_count
            )

            exact_key = key_frames[:, :, local_indices, :, :].reshape(
                batch,
                heads,
                group_size,
                local_count * tokens_per_frame,
                head_dim,
            )
            exact_value = value_frames[:, :, local_indices, :, :].reshape(
                batch,
                heads,
                group_size,
                local_count * tokens_per_frame,
                head_dim,
            )
            if remote_count > 0:
                remote_key = pooled_key[:, :, remote_indices, :, :].reshape(
                    batch,
                    heads,
                    group_size,
                    remote_count * pooled_tokens_per_frame,
                    head_dim,
                )
                remote_value = pooled_value[:, :, remote_indices, :, :].reshape(
                    batch,
                    heads,
                    group_size,
                    remote_count * pooled_tokens_per_frame,
                    head_dim,
                )
                key_parts = [exact_key, remote_key]
                value_parts = [exact_value, remote_value]
                if use_area_bias:
                    exact_bias = torch.zeros(
                        (1, 1, group_size, 1, exact_key.shape[-2]),
                        device=query.device,
                        dtype=query.dtype,
                    )
                    remote_bias = pooled_multiplicity.log().view(
                        1, 1, 1, 1, pooled_tokens_per_frame
                    ).expand(
                        1,
                        1,
                        group_size,
                        remote_count,
                        pooled_tokens_per_frame,
                    ).reshape(
                        1,
                        1,
                        group_size,
                        1,
                        remote_count * pooled_tokens_per_frame,
                    )
                    bias_parts = [exact_bias, remote_bias]
                else:
                    bias_parts = []
            else:
                key_parts = [exact_key]
                value_parts = [exact_value]
                bias_parts = []

            if special_key is not None and special_key.shape[-2] > 0:
                special_tokens = special_key.shape[-2]
                key_parts.append(
                    special_key.unsqueeze(2).expand(
                        batch, heads, group_size, special_tokens, head_dim
                    )
                )
                value_parts.append(
                    special_value.unsqueeze(2).expand(
                        batch, heads, group_size, special_tokens, head_dim
                    )
                )
                if bias_parts:
                    bias_parts.append(
                        torch.zeros(
                            (1, 1, group_size, 1, special_tokens),
                            device=query.device,
                            dtype=query.dtype,
                        )
                    )

            selected_key = torch.cat(key_parts, dim=-2)
            selected_value = torch.cat(value_parts, dim=-2)
            selected_query = query_frames[:, :, centers, :, :]
            attention_bias = (
                torch.cat(bias_parts, dim=-1) if bias_parts else None
            )
            group_output = F.scaled_dot_product_attention(
                selected_query,
                selected_key,
                selected_value,
                attn_mask=attention_bias,
            )
            if output is None:
                output = torch.empty(
                    query_frames.shape,
                    device=query.device,
                    dtype=group_output.dtype,
                )
            output[:, :, centers, :, :] = group_output

            total_local_keys += (
                group_size * local_count * tokens_per_frame
            )
            total_remote_keys += (
                group_size * remote_count * pooled_tokens_per_frame
            )

    if output is None:
        raise ValueError("num_frames must be positive")

    selected_patch_keys = total_local_keys + total_remote_keys
    dense_patch_keys = num_frames * num_frames * tokens_per_frame
    selected_key_fraction = selected_patch_keys / dense_patch_keys
    special_tokens = 0 if special_key is None else special_key.shape[-2]
    mean_patch_keys = selected_patch_keys / num_frames
    dense_keys_per_query = num_frames * tokens_per_frame + special_tokens
    effective_key_fraction = (
        mean_patch_keys + special_tokens
    ) / dense_keys_per_query
    stats = {
        "multiresolution_patch_sparsity": 1.0 - selected_key_fraction,
        "multiresolution_effective_sparsity": 1.0 - effective_key_fraction,
        "multiresolution_local_exact_fraction": (
            total_local_keys / selected_patch_keys
        ),
        "multiresolution_remote_pooled_fraction": (
            total_remote_keys / selected_patch_keys
        ),
        "multiresolution_mean_patch_keys": mean_patch_keys,
        "multiresolution_pooled_tokens_per_frame": float(
            pooled_tokens_per_frame
        ),
        "multiresolution_carrier_tokens": float(special_tokens),
        "multiresolution_local_radius": float(local_frame_radius),
        "multiresolution_remote_pool_size": float(remote_pool_size),
        "multiresolution_remote_mode_id": float(
            {"avg": 0, "strided": 1, "detail": 2}[remote_mode]
        ),
        "multiresolution_remote_phase": float(
            -2
            if remote_phase_mode == "head"
            else remote_phase % (remote_pool_size * remote_pool_size)
        ),
        "multiresolution_remote_samples_per_cell": float(
            remote_samples_per_cell
        ),
        "multiresolution_area_bias": float(use_area_bias),
    }
    return output.reshape(batch, heads, token_count, head_dim), stats


def select_residual_debt_frame_pairs(
    query_carrier: torch.Tensor,
    key_carrier: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_value: torch.Tensor,
    *,
    local_frame_radius: int,
    extra_frames_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    temperature: float,
    surface_query_carrier: torch.Tensor | None = None,
    surface_key_carrier: torch.Tensor | None = None,
    surface_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Allocate residual phases using persistent pose and surface debt."""
    if query_carrier.shape != key_carrier.shape:
        raise ValueError("query and key carriers must have identical shapes")
    if coarse_value.shape != residual_value.shape:
        raise ValueError("coarse and residual values must have identical shapes")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("residual debt momentum must be in [0, 1]")
    if not 0.0 <= repayment <= 1.0:
        raise ValueError("residual debt repayment must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("residual debt temperature must be positive")
    if not 0.0 <= surface_weight <= 1.0:
        raise ValueError("surface debt weight must be in [0, 1]")
    if (surface_query_carrier is None) != (surface_key_carrier is None):
        raise ValueError("surface query and key carriers must be provided together")
    if surface_query_carrier is not None:
        if surface_query_carrier.shape != query_carrier.shape:
            raise ValueError("surface query carrier shape must match pose carrier")
        if surface_key_carrier.shape != key_carrier.shape:
            raise ValueError("surface key carrier shape must match pose carrier")

    batch, heads, num_frames, _ = query_carrier.shape
    max_local_count = min(num_frames, 2 * local_frame_radius + 1)
    max_extra_frames = max(0, num_frames - max_local_count)
    extra_frames_per_query = min(extra_frames_per_query, max_extra_frames)
    if extra_frames_per_query < 0:
        raise ValueError("extra_frames_per_query must be non-negative")

    query_unit = F.normalize(query_carrier.float(), dim=-1)
    key_unit = F.normalize(key_carrier.float(), dim=-1)
    carrier_logits = torch.einsum(
        "bhqd,bhkd->bhqk", query_unit, key_unit
    ) / temperature
    carrier_relevance = carrier_logits.softmax(dim=-1).mean(dim=1)

    frame_detail = (
        (residual_value.float() - coarse_value.float())
        .square()
        .mean(dim=(1, 3, 4))
    )
    frame_detail = frame_detail / frame_detail.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    pose_need = carrier_relevance * frame_detail[:, None, :]
    pose_need = pose_need / pose_need.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)

    if surface_weight > 0.0 and surface_query_carrier is not None:
        surface_query_unit = F.normalize(
            surface_query_carrier.float(), dim=-1
        )
        surface_key_unit = F.normalize(surface_key_carrier.float(), dim=-1)
        surface_logits = torch.einsum(
            "bhqd,bhkd->bhqk", surface_query_unit, surface_key_unit
        ) / temperature
        surface_relevance = surface_logits.softmax(dim=-1).mean(dim=1)
        surface_need = surface_relevance * frame_detail[:, None, :]
        surface_need = surface_need / surface_need.mean(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    else:
        surface_need = pose_need

    state_key = "residual_budget_debt"
    previous_debt = None if routing_state is None else routing_state.get(state_key)
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != (batch, num_frames, num_frames)
    ):
        previous_debt = torch.zeros_like(pose_need)
    else:
        previous_debt = previous_debt.to(
            device=pose_need.device, dtype=pose_need.dtype
        )
    pose_debt = momentum * previous_debt + pose_need

    surface_state_key = "residual_budget_surface_debt"
    previous_surface_debt = (
        None if routing_state is None else routing_state.get(surface_state_key)
    )
    if (
        layer_idx == 0
        or previous_surface_debt is None
        or tuple(previous_surface_debt.shape)
        != (batch, num_frames, num_frames)
    ):
        previous_surface_debt = torch.zeros_like(surface_need)
    else:
        previous_surface_debt = previous_surface_debt.to(
            device=surface_need.device, dtype=surface_need.dtype
        )
    surface_debt = momentum * previous_surface_debt + surface_need
    debt = torch.lerp(pose_debt, surface_debt, surface_weight)

    frame_ids = torch.arange(num_frames, device=debt.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    routing_score = debt.masked_fill(local_mask.unsqueeze(0), float("-inf"))
    if extra_frames_per_query > 0:
        selected_indices = routing_score.topk(
            extra_frames_per_query, dim=-1
        ).indices
        selected_pose_debt = pose_debt.gather(-1, selected_indices)
        next_pose_debt = pose_debt.scatter(
            -1,
            selected_indices,
            selected_pose_debt * (1.0 - repayment),
        )
        selected_surface_debt = surface_debt.gather(-1, selected_indices)
        next_surface_debt = surface_debt.scatter(
            -1,
            selected_indices,
            selected_surface_debt * (1.0 - repayment),
        )
        selected_pose_need_mean = float(
            pose_need.gather(-1, selected_indices).mean().item()
        )
        selected_surface_need_mean = float(
            surface_need.gather(-1, selected_indices).mean().item()
        )
    else:
        selected_indices = torch.empty(
            batch,
            num_frames,
            0,
            device=debt.device,
            dtype=torch.long,
        )
        next_pose_debt = pose_debt
        next_surface_debt = surface_debt
        selected_pose_need_mean = 0.0
        selected_surface_need_mean = 0.0

    if routing_state is not None:
        routing_state[state_key] = next_pose_debt.detach()
        routing_state[surface_state_key] = next_surface_debt.detach()
        coverage_key = "residual_budget_coverage"
        coverage = routing_state.get(coverage_key)
        if (
            layer_idx == 0
            or coverage is None
            or tuple(coverage.shape) != (batch, num_frames, num_frames)
        ):
            coverage = torch.zeros_like(debt, dtype=torch.bool)
        if extra_frames_per_query > 0:
            coverage = coverage.scatter(
                -1,
                selected_indices,
                torch.ones_like(selected_indices, dtype=torch.bool),
            )
        routing_state[coverage_key] = coverage.detach()
        available_pairs = (~local_mask).sum().item() * batch
        coverage_fraction = (
            float(coverage.sum().item() / available_pairs)
            if available_pairs > 0
            else 0.0
        )
    else:
        coverage_fraction = 0.0

    stats = {
        "residual_budget_extra_frames_per_query": float(
            extra_frames_per_query
        ),
        "residual_budget_frame_detail": float(frame_detail.mean().item()),
        "residual_budget_debt": float(
            debt.masked_fill(local_mask.unsqueeze(0), 0.0).mean().item()
        ),
        "residual_budget_selected_need": selected_pose_need_mean,
        "residual_budget_selected_surface_need": (
            selected_surface_need_mean
        ),
        "residual_budget_surface_weight": float(surface_weight),
        "residual_budget_coverage_fraction": coverage_fraction,
    }
    return selected_indices, stats


def schedule_residual_debt_cells(
    current_need: torch.Tensor,
    *,
    local_frame_radius: int,
    extra_tokens_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    repayment_mode: str = "reset",
    service_credit_scale: float = 1.0,
    effective_service_credit: torch.Tensor | None = None,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Allocate residual cells from scorer-provided need and persistent debt."""
    if current_need.ndim != 4:
        raise ValueError(
            "current_need must have shape [batch, query_frames, key_frames, cells]"
        )
    if current_need.shape[1] != current_need.shape[2]:
        raise ValueError("query and key frame counts must match")
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("residual debt momentum must be in [0, 1]")
    if not 0.0 <= repayment <= 1.0:
        raise ValueError("residual debt repayment must be in [0, 1]")
    if repayment_mode not in {
        "reset",
        "deficit",
        "bounded_deficit",
        "effective_bounded_deficit",
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }:
        raise ValueError(
            "repayment_mode must be reset, deficit, bounded_deficit, "
            "effective_bounded_deficit, age_bounded_deficit, or "
            "frontier_age_bounded_deficit"
        )
    if service_credit_scale < 0.0:
        raise ValueError("service_credit_scale must be non-negative")
    if extra_tokens_per_query < 0:
        raise ValueError("extra_tokens_per_query must be non-negative")

    batch, num_frames, _, cells_per_frame = current_need.shape
    max_local_count = min(num_frames, 2 * local_frame_radius + 1)
    max_remote_tokens = max(
        0, (num_frames - max_local_count) * cells_per_frame
    )
    extra_tokens_per_query = min(extra_tokens_per_query, max_remote_tokens)
    current_need = current_need.float().clamp_min(0.0)
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)

    frame_ids = torch.arange(num_frames, device=current_need.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    state_key = "residual_budget_cell_debt"
    previous_debt = None if routing_state is None else routing_state.get(state_key)
    expected_shape = (batch, num_frames, num_frames, cells_per_frame)
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != expected_shape
    ):
        previous_debt = torch.zeros_like(current_need)
    else:
        previous_debt = previous_debt.to(
            device=current_need.device, dtype=current_need.dtype
        )
    debt = momentum * previous_debt + current_need

    remote_mask = (~local_mask)[None, :, :, None].expand_as(debt)
    effective_service_mode = repayment_mode == "effective_bounded_deficit"
    effective_credit = None
    if effective_service_mode:
        if effective_service_credit is None:
            raise ValueError(
                "effective_bounded_deficit requires effective service credit"
            )
        if tuple(effective_service_credit.shape) != expected_shape:
            raise ValueError(
                "effective service credit must match current_need shape"
            )
        effective_credit = (
            effective_service_credit.float().clamp_min(0.0)
        )
        effective_credit = effective_credit.masked_fill(
            ~remote_mask, 0.0
        )
    service_age = torch.zeros_like(debt)
    service_cycle_layers = 0
    age_service_mode = repayment_mode in {
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }
    if age_service_mode:
        age_key = "residual_budget_cell_service_age"
        previous_age = (
            None if routing_state is None else routing_state.get(age_key)
        )
        if (
            layer_idx == 0
            or previous_age is None
            or tuple(previous_age.shape) != expected_shape
        ):
            service_age = torch.zeros_like(debt)
        else:
            service_age = previous_age.to(
                device=debt.device, dtype=debt.dtype
            )
        if extra_tokens_per_query > 0:
            service_cycle_layers = max(
                1,
                math.ceil(max_remote_tokens / extra_tokens_per_query),
            )
            if repayment_mode == "age_bounded_deficit":
                routing_priority = debt * (
                    1.0 + service_age / float(service_cycle_layers)
                )
            else:
                routing_priority = debt
        else:
            routing_priority = debt
    else:
        age_key = None
        routing_priority = debt

    routing_score = routing_priority.masked_fill(
        local_mask[None, :, :, None], float("-inf")
    )
    flat_score = routing_score.flatten(start_dim=-2)
    flat_debt = debt.flatten(start_dim=-2)
    flat_need = current_need.flatten(start_dim=-2)
    service_credit = torch.zeros(
        batch,
        num_frames,
        device=debt.device,
        dtype=debt.dtype,
    )
    service_credit_conservation_error = 0.0
    frontier_service_fraction = 0.0
    frontier_overdue_selected_fraction = 0.0
    if extra_tokens_per_query > 0:
        if repayment_mode == "frontier_age_bounded_deficit":
            frontier_slots = max(
                1,
                math.ceil(
                    extra_tokens_per_query / service_cycle_layers
                ),
            )
            protected_count = extra_tokens_per_query - frontier_slots
            candidate_count = min(
                max_remote_tokens,
                extra_tokens_per_query + frontier_slots,
            )
            candidate = flat_score.topk(candidate_count, dim=-1)
            protected_indices = candidate.indices[..., :protected_count]
            frontier_indices = candidate.indices[..., protected_count:]
            flat_service_age = service_age.flatten(start_dim=-2)
            frontier_age = flat_service_age.gather(-1, frontier_indices)
            frontier_overdue = frontier_age >= float(service_cycle_layers)
            frontier_count = frontier_indices.shape[-1]
            rank_preference = torch.linspace(
                1.0,
                0.0,
                frontier_count + 1,
                device=debt.device,
                dtype=debt.dtype,
            )[:-1]
            frontier_priority = (
                2.0 * frontier_overdue.to(debt.dtype)
                + rank_preference[None, None, :]
            )
            frontier_choice = frontier_priority.topk(
                frontier_slots, dim=-1
            ).indices
            selected_frontier_indices = frontier_indices.gather(
                -1, frontier_choice
            )
            selected_indices = torch.cat(
                (protected_indices, selected_frontier_indices), dim=-1
            )
            frontier_service_fraction = float(
                frontier_slots / extra_tokens_per_query
            )
            if collect_stats:
                frontier_overdue_selected_fraction = float(
                    frontier_overdue.gather(-1, frontier_choice)
                    .float()
                    .mean()
                    .item()
                )
        else:
            selected_indices = flat_score.topk(
                extra_tokens_per_query, dim=-1
            ).indices
        selected_debt = flat_debt.gather(-1, selected_indices)
        if repayment_mode in {
            "deficit",
            "bounded_deficit",
            "effective_bounded_deficit",
            "age_bounded_deficit",
            "frontier_age_bounded_deficit",
        }:
            remote_need = current_need.masked_fill(
                local_mask[None, :, :, None], 0.0
            )
            base_service_credit = remote_need.sum(
                dim=(-2, -1)
            ) / float(extra_tokens_per_query)
            if effective_service_mode:
                flat_effective_credit = effective_credit.flatten(
                    start_dim=-2
                )
                selected_effective_credit = flat_effective_credit.gather(
                    -1, selected_indices
                )
                selected_credit_mean = selected_effective_credit.mean(
                    dim=-1, keepdim=True
                )
                selected_credit_weight = torch.where(
                    selected_credit_mean > 1e-8,
                    selected_effective_credit
                    / selected_credit_mean.clamp_min(1e-8),
                    torch.ones_like(selected_effective_credit),
                )
                credit_to_apply = (
                    base_service_credit[..., None]
                    * selected_credit_weight
                )
                service_credit = credit_to_apply.mean(dim=-1)
                remote_demand = remote_need.sum(dim=(-2, -1))
                if collect_stats:
                    service_credit_conservation_error = float(
                        (
                            (
                                credit_to_apply.sum(dim=-1)
                                - remote_demand
                            ).abs()
                            / remote_demand.clamp_min(1e-8)
                        ).mean().item()
                    )
            else:
                service_credit = base_service_credit
                credit_to_apply = service_credit[..., None]
            selected_next_debt = selected_debt - (
                repayment
                * service_credit_scale
                * credit_to_apply
            )
            if repayment_mode in {
                "bounded_deficit",
                "effective_bounded_deficit",
                "age_bounded_deficit",
                "frontier_age_bounded_deficit",
            }:
                selected_next_debt = selected_next_debt.clamp_min(0.0)
        else:
            selected_next_debt = selected_debt * (1.0 - repayment)
        next_debt = flat_debt.scatter(
            -1,
            selected_indices,
            selected_next_debt,
        ).view_as(debt)
        selected_need_mean = (
            float(flat_need.gather(-1, selected_indices).mean().item())
            if collect_stats
            else 0.0
        )
    else:
        selected_indices = torch.empty(
            batch,
            num_frames,
            0,
            device=debt.device,
            dtype=torch.long,
        )
        next_debt = debt
        selected_need_mean = 0.0

    selected_service_age_mean = 0.0
    overdue_service_fraction = 0.0
    if age_service_mode:
        next_service_age = torch.where(
            remote_mask,
            service_age + 1.0,
            torch.zeros_like(service_age),
        )
        if extra_tokens_per_query > 0:
            flat_service_age = service_age.flatten(start_dim=-2)
            if collect_stats:
                selected_service_age_mean = float(
                    flat_service_age.gather(
                        -1, selected_indices
                    ).mean().item()
                )
            flat_next_service_age = next_service_age.flatten(start_dim=-2)
            flat_next_service_age = flat_next_service_age.scatter(
                -1,
                selected_indices,
                torch.zeros_like(selected_indices, dtype=service_age.dtype),
            )
            next_service_age = flat_next_service_age.view_as(service_age)
        if collect_stats:
            remote_service_age = service_age.masked_select(remote_mask)
            if remote_service_age.numel() > 0 and service_cycle_layers > 0:
                overdue_service_fraction = float(
                    (remote_service_age >= float(service_cycle_layers))
                    .float()
                    .mean()
                    .item()
                )
        if routing_state is not None:
            routing_state[age_key] = next_service_age.detach()

    coverage_fraction = 0.0
    coverage_gain = 0.0
    selection_new_fraction = 0.0
    effective_local_radius = min(local_frame_radius, num_frames - 1)
    local_pair_count = (
        num_frames * (2 * effective_local_radius + 1)
        - effective_local_radius * (effective_local_radius + 1)
    )
    available_cells = (
        (num_frames * num_frames - local_pair_count)
        * cells_per_frame
        * batch
    )
    if routing_state is not None:
        routing_state[state_key] = next_debt.detach()
        if collect_stats:
            coverage_key = "residual_budget_cell_coverage"
            coverage = routing_state.get(coverage_key)
            if (
                layer_idx == 0
                or coverage is None
                or tuple(coverage.shape) != expected_shape
            ):
                coverage = torch.zeros_like(debt, dtype=torch.bool)
            previous_coverage_fraction = (
                float(coverage.sum().item() / available_cells)
                if available_cells > 0
                else 0.0
            )
            if extra_tokens_per_query > 0:
                flat_coverage = coverage.flatten(start_dim=-2)
                already_selected = flat_coverage.gather(
                    -1, selected_indices
                )
                selection_new_fraction = float(
                    (~already_selected).float().mean().item()
                )
                flat_coverage = flat_coverage.scatter(
                    -1,
                    selected_indices,
                    torch.ones_like(selected_indices, dtype=torch.bool),
                )
                coverage = flat_coverage.view_as(coverage)
            routing_state[coverage_key] = coverage.detach()
            coverage_fraction = (
                float(coverage.sum().item() / available_cells)
                if available_cells > 0
                else 0.0
            )
            coverage_gain = coverage_fraction - previous_coverage_fraction

    negative_credit_fraction = 0.0
    debt_before_mean = 0.0
    debt_after_mean = 0.0
    debt_repaid_fraction = 0.0
    if collect_stats and _env_flag(
        "SPARSE_VGGT_FINE_REFRESH_PROBE", default=False
    ):
        debt_before_repayment = debt.masked_select(remote_mask)
        debt_after_repayment = next_debt.masked_select(remote_mask)
        if debt_before_repayment.numel() > 0:
            debt_before_mean = float(debt_before_repayment.mean().item())
            debt_after_mean = float(debt_after_repayment.mean().item())
            debt_repaid_fraction = float(
                (
                    (
                        debt_before_repayment.sum()
                        - debt_after_repayment.sum()
                    )
                    / debt_before_repayment.sum().clamp_min(1e-8)
                ).item()
            )
            negative_credit_fraction = float(
                (debt_after_repayment < 0.0).float().mean().item()
            )
    stats = {
        "residual_budget_extra_frames_per_query": float(
            extra_tokens_per_query / cells_per_frame
        ),
        "residual_budget_debt": debt_before_mean,
        "residual_budget_debt_after_repayment": debt_after_mean,
        "residual_budget_debt_repaid_fraction": debt_repaid_fraction,
        "residual_budget_repayment_mode_id": float(
            {
                "reset": 0,
                "deficit": 1,
                "bounded_deficit": 2,
                "effective_bounded_deficit": 5,
                "age_bounded_deficit": 3,
                "frontier_age_bounded_deficit": 4,
            }[repayment_mode]
        ),
        "residual_budget_service_credit_scale": float(
            service_credit_scale
        ),
        "residual_budget_service_credit": (
            float(service_credit.mean().item()) if collect_stats else 0.0
        ),
        "residual_budget_effective_service_credit": float(
            effective_service_mode
        ),
        "residual_budget_service_credit_conservation_error": (
            service_credit_conservation_error
        ),
        "residual_budget_negative_credit_fraction": (
            negative_credit_fraction
        ),
        "residual_budget_service_cycle_layers": float(
            service_cycle_layers
        ),
        "residual_budget_service_age": (
            float(service_age.masked_select(remote_mask).mean().item())
            if collect_stats and available_cells > 0
            else 0.0
        ),
        "residual_budget_selected_service_age": selected_service_age_mean,
        "residual_budget_overdue_service_fraction": (
            overdue_service_fraction
        ),
        "residual_budget_frontier_service_fraction": (
            frontier_service_fraction
        ),
        "residual_budget_frontier_overdue_selected_fraction": (
            frontier_overdue_selected_fraction
        ),
        "residual_budget_selected_need": selected_need_mean,
        "residual_budget_selected_surface_need": selected_need_mean,
        "residual_budget_coverage_fraction": coverage_fraction,
        "residual_budget_cell_coverage_fraction": coverage_fraction,
        "residual_budget_coverage_gain": coverage_gain,
        "residual_budget_selection_new_fraction": selection_new_fraction,
        "residual_budget_uncovered_fraction": (
            1.0 - coverage_fraction if available_cells > 0 else 0.0
        ),
    }
    return selected_indices, stats


def _project_capped_service_liability(
    liability: torch.Tensor,
    upper_bound: torch.Tensor,
    total_capacity: torch.Tensor,
) -> torch.Tensor:
    """Project non-negative work liability onto a capped service simplex."""
    liability = liability.float().clamp_min(0.0)
    upper_bound = upper_bound.to(liability).clamp_min(0.0)
    total_capacity = total_capacity.to(liability).clamp_min(0.0)
    clipped = torch.minimum(liability, upper_bound)
    needs_projection = clipped.sum(dim=(-2, -1), keepdim=True) > total_capacity
    if not bool(needs_projection.any()):
        return clipped

    lower = torch.zeros_like(total_capacity)
    upper = liability.amax(dim=(-2, -1), keepdim=True)
    for _ in range(40):
        threshold = (lower + upper) * 0.5
        candidate = torch.minimum(
            (liability - threshold).clamp_min(0.0), upper_bound
        )
        mass = candidate.sum(dim=(-2, -1), keepdim=True)
        lower = torch.where(mass > total_capacity, threshold, lower)
        upper = torch.where(mass > total_capacity, upper, threshold)
    projected = torch.minimum(
        (liability - upper).clamp_min(0.0), upper_bound
    )
    return torch.where(needs_projection, projected, clipped)


def _measure_parent_debt_transportability(
    previous_need_distribution: torch.Tensor,
    current_need: torch.Tensor,
    previous_debt: torch.Tensor,
    valid_parent: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Measure how much old parent liability remains supported now."""
    current_mass = current_need.float().clamp_min(0.0).masked_fill(
        ~valid_parent, 0.0
    )
    previous_distribution = previous_need_distribution.float().clamp_min(0.0)
    previous_distribution = previous_distribution.masked_fill(
        ~valid_parent, 0.0
    )
    current_distribution = current_mass / current_mass.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    previous_distribution = previous_distribution / previous_distribution.sum(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)

    common_mass = torch.minimum(
        previous_distribution, current_distribution
    )
    overlap = common_mass.sum(dim=(-2, -1))
    parent_survival = torch.where(
        previous_distribution > 1e-8,
        torch.minimum(
            torch.ones_like(current_distribution),
            current_distribution / previous_distribution.clamp_min(1e-8),
        ),
        torch.zeros_like(current_distribution),
    ).masked_fill(~valid_parent, 0.0)
    debt_mass = previous_debt.float().clamp_min(0.0).masked_fill(
        ~valid_parent, 0.0
    )
    debt_total = debt_mass.sum(dim=(-2, -1))
    debt_weighted_survival = torch.where(
        debt_total > 1e-8,
        (debt_mass * parent_survival).sum(dim=(-2, -1))
        / debt_total.clamp_min(1e-8),
        torch.ones_like(debt_total),
    )
    supported_debt = debt_mass * parent_survival
    return current_distribution, {
        "overlap": overlap,
        "debt_weighted_survival": debt_weighted_survival,
        "parent_survival": parent_survival,
        "supported_debt_fraction": torch.where(
            debt_total > 1e-8,
            supported_debt.sum(dim=(-2, -1)) / debt_total.clamp_min(1e-8),
            torch.ones_like(debt_total),
        ),
    }






def _schedule_unified_incremental_service(
    current_need: torch.Tensor,
    child_counts: torch.Tensor,
    incremental_child_need: torch.Tensor,
    *,
    local_frame_radius: int,
    child_budget_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    repayment_mode: str,
    service_credit_scale: float,
    service_conditioned_momentum: bool,
    bounded_wait_reobservation: bool,
    effective_service_credit: torch.Tensor | None,
    target_sparsity: float | None,
    total_layers: int,
    parent_carrier_execution: bool,
    collect_stats: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Allocate one fixed budget over parent admission and child refinement.

    Child actions are sorted by decreasing marginal need inside each parent.
    A global Top-K over those actions therefore preserves the prefix constraint:
    a parent can receive its r-th unit of service only after all more valuable
    units have been admitted. This turns parent coverage and fine refinement
    into two outcomes of one budget allocation rather than two controllers.
    """
    if current_need.ndim != 4:
        raise ValueError("current_need must have shape [B, Q, K, parents]")
    if incremental_child_need.ndim != 5:
        raise ValueError(
            "incremental child need must have shape [B, Q, K, parents, C]"
        )
    if incremental_child_need.shape[:-1] != current_need.shape:
        raise ValueError("incremental child need must match current parent need")
    if child_counts.ndim != 1 or child_counts.shape[0] != current_need.shape[-1]:
        raise ValueError("child_counts must have one entry per parent cell")
    if repayment_mode not in {
        "reset",
        "deficit",
        "bounded_deficit",
        "effective_bounded_deficit",
    }:
        raise ValueError("unsupported unified-service repayment mode")

    batch, num_frames, _, parents_per_frame = current_need.shape
    child_slots = incremental_child_need.shape[-1]
    child_counts = child_counts.to(device=current_need.device, dtype=torch.long)
    if bool((child_counts < 0).any()) or bool((child_counts > child_slots).any()):
        raise ValueError("child service capacities must fit incremental slots")
    if not bool((child_counts > 0).any()):
        raise ValueError("at least one parent must have positive capacity")
    if total_layers <= layer_idx:
        raise ValueError("total_layers must exceed the current layer index")

    current_need = current_need.float().clamp_min(0.0)
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    child_need = incremental_child_need.float().clamp_min(0.0)
    slot_ids = torch.arange(child_slots, device=current_need.device)
    valid_child = slot_ids.view(1, 1, 1, 1, -1) < child_counts.view(
        1, 1, 1, -1, 1
    )
    child_mean = (
        (child_need * valid_child).sum(dim=-1, keepdim=True)
        / child_counts.clamp_min(1).view(1, 1, 1, -1, 1)
    ).clamp_min(1e-8)
    child_need = (child_need / child_mean).masked_fill(~valid_child, 0.0)

    state_key = "residual_budget_grouped_parent_debt"
    relevance_bounded_debt = _env_flag(
        "SPARSE_VGGT_RELEVANCE_BOUNDED_DEBT", default=False
    )
    action_level_debt = _env_flag(
        "SPARSE_VGGT_ACTION_LEVEL_DEBT", default=False
    )
    if relevance_bounded_debt and action_level_debt:
        raise ValueError("relevance-bounded and action-level debt are exclusive")
    pv_service_debt_feedback = _env_flag(
        "SPARSE_VGGT_PV_SERVICE_DEBT_FEEDBACK", default=False
    )
    pv_importance_alignment_observer = _env_flag(
        "SPARSE_VGGT_PV_IMPORTANCE_ALIGNMENT_OBSERVER", default=False
    )
    if pv_service_debt_feedback or pv_importance_alignment_observer:
        if routing_state is None:
            raise ValueError("PV execution/debt observation requires routing state")
    if pv_service_debt_feedback:
        if repayment_mode not in {
            "deficit",
            "bounded_deficit",
            "effective_bounded_deficit",
        }:
            raise ValueError(
                "PV-service debt feedback requires deficit repayment"
            )
    if action_level_debt and (
        pv_service_debt_feedback or pv_importance_alignment_observer
    ):
        raise ValueError("action-level debt does not use PV debt feedback")
    previous_debt = None if routing_state is None else routing_state.get(state_key)
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != tuple(current_need.shape)
    ):
        previous_debt = torch.zeros_like(current_need)
    else:
        previous_debt = previous_debt.to(current_need)

    transport_observer = _env_flag(
        "SPARSE_VGGT_DEBT_TRANSPORTABILITY_OBSERVER", default=False
    )
    transport_active = transport_observer
    transport_state_key = "residual_budget_previous_parent_need_distribution"
    previous_need_distribution = (
        None if routing_state is None else routing_state.get(transport_state_key)
    )
    frame_ids = torch.arange(num_frames, device=current_need.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    transport_valid_parent = (~local_mask)[None, :, :, None].expand_as(
        current_need
    )
    current_need_distribution = current_need.masked_fill(
        ~transport_valid_parent, 0.0
    )
    current_need_distribution = current_need_distribution / (
        current_need_distribution.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    )
    transport_stats = {
        "overlap": current_need.new_ones(batch, num_frames),
        "debt_weighted_survival": current_need.new_ones(batch, num_frames),
        "parent_survival": current_need.new_ones(current_need.shape),
        "supported_debt_fraction": current_need.new_ones(batch, num_frames),
    }
    transport_reference = None
    if (
        previous_need_distribution is not None
        and tuple(previous_need_distribution.shape) == tuple(current_need.shape)
    ):
        transport_reference = previous_need_distribution.to(current_need)
    if transport_active and layer_idx != 0 and transport_reference is not None:
        current_need_distribution, transport_stats = (
            _measure_parent_debt_transportability(
                transport_reference,
                current_need,
                previous_debt,
                transport_valid_parent,
            )
        )

    action_state_key = "residual_budget_grouped_action_debt"
    current_action_need = current_need[..., None] * child_need
    previous_action_debt = (
        None if routing_state is None else routing_state.get(action_state_key)
    )
    if (
        layer_idx == 0
        or previous_action_debt is None
        or tuple(previous_action_debt.shape) != tuple(current_action_need.shape)
    ):
        previous_action_debt = torch.zeros_like(current_action_need)
    else:
        previous_action_debt = previous_action_debt.to(current_action_need)
    transported_action_debt = torch.minimum(
        previous_action_debt, current_action_need
    ).masked_fill(~valid_child, 0.0)
    rejected_action_debt = (
        previous_action_debt - transported_action_debt
    ).clamp_min(0.0).masked_fill(~valid_child, 0.0)
    child_denominator = child_counts.clamp_min(1).view(1, 1, 1, -1)
    if action_level_debt:
        transported_previous_debt = (
            transported_action_debt.sum(dim=-1) / child_denominator
        )
        rejected_previous_debt = (
            rejected_action_debt.sum(dim=-1) / child_denominator
        )
    else:
        transported_previous_debt = torch.minimum(previous_debt, current_need)
        rejected_previous_debt = (
            previous_debt - transported_previous_debt
        ).clamp_min(0.0)
    action_liability = current_action_need + transported_action_debt
    momentum_key = "residual_budget_grouped_parent_momentum"
    effective_momentum = None
    if action_level_debt:
        debt = action_liability.sum(dim=-1) / child_denominator
    elif relevance_bounded_debt:
        debt = current_need + transported_previous_debt
    elif service_conditioned_momentum:
        effective_momentum = (
            None if routing_state is None else routing_state.get(momentum_key)
        )
        if (
            layer_idx == 0
            or effective_momentum is None
            or tuple(effective_momentum.shape) != (batch, num_frames)
        ):
            effective_momentum = current_need.new_full(
                (batch, num_frames), momentum
            )
        else:
            effective_momentum = effective_momentum.to(current_need)
        history_survival = (
            1.0
        )
        debt = (
            effective_momentum[..., None, None]
            * previous_debt
            * history_survival
            + current_need
        )
    else:
        history_survival = (
            1.0
        )
        debt = momentum * previous_debt * history_survival + current_need

    remote_parent = (~local_mask)[None, :, :, None].expand_as(debt)
    remote_action = remote_parent[..., None] & valid_child
    current_action_priority = current_need[..., None] * child_need
    historical_parent_priority = (debt - current_need).clamp_min(0.0)

    historical_action_priority = (
        historical_parent_priority[..., None] * child_need
    )
    base_action_priority = (
        action_liability
        if action_level_debt
        else current_action_priority + historical_action_priority
    )
    max_budget = (
        int(remote_action.sum(dim=(-3, -2, -1)).min().item())
        if num_frames else 0
    )
    child_budget_per_query = min(max(child_budget_per_query, 0), max_budget)






    service_cycle_layers = 0
    parent_service_age = torch.zeros_like(debt)
    overdue_parent = torch.zeros_like(debt, dtype=torch.bool)
    if bounded_wait_reobservation:
        age_key = "residual_budget_grouped_parent_service_age"
        previous_age = (
            None if routing_state is None else routing_state.get(age_key)
        )
        if (
            layer_idx != 0
            and previous_age is not None
            and tuple(previous_age.shape) == tuple(debt.shape)
        ):
            parent_service_age = previous_age.to(debt)
        if child_budget_per_query > 0:
            service_cycle_layers = max(
                1, math.ceil(max_budget / child_budget_per_query)
            )
            overdue_parent = (
                parent_service_age >= float(service_cycle_layers)
            ) & remote_parent

        # Reobservation is the smallest complete service quantum: at most one
        # overdue parent per query is fully refreshed. The rest of the budget
        # stays on the original debt-detail ranking, avoiding a second tuned
        # budget split while preserving child-prefix ordering.
        base_scale = base_action_priority.masked_fill(
            ~remote_action, 0.0
        ).amax(dim=(-3, -2, -1), keepdim=True).clamp_min(1e-8)
        normalized_priority = base_action_priority / base_scale
        parent_priority = (
            2.0 * parent_service_age
            + normalized_priority.max(dim=-1).values
        ).masked_fill(~overdue_parent, float("-inf"))
        flat_parent_priority = parent_priority.flatten(start_dim=-2)
        forced_parent_indices = flat_parent_priority.argmax(dim=-1)
        has_overdue_parent = overdue_parent.flatten(start_dim=-2).any(
            dim=-1
        )
        forced_parent = torch.zeros_like(overdue_parent)
        forced_parent.flatten(start_dim=-2).scatter_(
            -1, forced_parent_indices[..., None], True
        )
        forced_parent &= has_overdue_parent[..., None, None]
        action_priority = torch.where(
            forced_parent[..., None],
            2.0 + normalized_priority,
            normalized_priority,
        )
    else:
        age_key = None
        forced_parent = torch.zeros_like(debt, dtype=torch.bool)
        action_priority = base_action_priority
    action_priority = action_priority.masked_fill(
        ~remote_action, float("-inf")
    )

    selected_actions = torch.zeros_like(action_priority, dtype=torch.bool)
    current_selected_actions = torch.zeros_like(
        current_action_priority, dtype=torch.bool
    )
    if child_budget_per_query > 0:
        flat_priority = action_priority.flatten(start_dim=-3)
        action_indices = flat_priority.topk(
            child_budget_per_query, dim=-1
        ).indices
        selected_actions.flatten(start_dim=-3).scatter_(
            -1, action_indices, True
        )
        flat_current_priority = current_action_priority.masked_fill(
            ~remote_action, float("-inf")
        ).flatten(start_dim=-3)
        current_indices = flat_current_priority.topk(
            child_budget_per_query, dim=-1
        ).indices
        current_selected_actions.flatten(start_dim=-3).scatter_(
            -1, current_indices, True
        )
    served_dense = selected_actions.sum(dim=-1)
    if not bool((served_dense.sum(dim=(-2, -1)) == child_budget_per_query).all()):
        raise RuntimeError("unified incremental service did not fill child budget")

    active_parent = served_dense > 0
    active_count = active_parent.sum(dim=(-2, -1))
    max_active = int(active_count.max().item()) if active_count.numel() else 0
    if max_active > 0:
        parent_score = action_priority.max(dim=-1).values.masked_fill(
            ~active_parent, float("-inf")
        )
        flat_parent_score = parent_score.flatten(start_dim=-2)
        parent_indices = flat_parent_score.topk(max_active, dim=-1).indices
        served_counts = served_dense.flatten(start_dim=-2).gather(
            -1, parent_indices
        )
    else:
        parent_indices = torch.empty(
            batch, num_frames, 0, device=debt.device, dtype=torch.long
        )
        served_counts = torch.empty_like(parent_indices)

    service_fraction_dense = served_dense.float() / child_counts.clamp_min(1).view(
        1, 1, 1, -1
    )
    next_action_debt = None
    if action_level_debt:
        next_action_debt = action_liability.masked_fill(
            selected_actions | ~remote_action, 0.0
        )
        next_debt = next_action_debt.sum(dim=-1) / child_denominator
        service_credit_dense = (debt - next_debt).clamp_min(0.0)
    elif relevance_bounded_debt:
        service_credit_dense = debt * service_fraction_dense
        next_debt = debt * (1.0 - service_fraction_dense)
    elif repayment_mode == "reset":
        service_credit_dense = served_dense.float()
        next_debt = debt * (1.0 - service_fraction_dense)
    else:
        remote_demand = current_need.masked_fill(~remote_parent, 0.0).sum(
            dim=(-2, -1)
        )
        credit_per_child = remote_demand / max(child_budget_per_query, 1)
        service_credit_dense = credit_per_child[..., None, None] * served_dense.float()
        if repayment_mode == "effective_bounded_deficit":
            if effective_service_credit is None:
                raise ValueError("effective unified repayment requires service credit")
            effective = effective_service_credit.float().clamp_min(0.0)
            active = served_dense > 0
            active_mean = (
                (effective * active).sum(dim=(-2, -1), keepdim=True)
                / active.sum(dim=(-2, -1), keepdim=True).clamp_min(1)
            )
            service_credit_dense = service_credit_dense * torch.where(
                active_mean > 1e-8,
                effective / active_mean.clamp_min(1e-8),
                torch.ones_like(effective),
            )
        next_debt = debt - repayment * service_credit_scale * service_credit_dense
        if repayment_mode in {"bounded_deficit", "effective_bounded_deficit"}:
            next_debt = next_debt.clamp_min(0.0)
        next_debt = torch.where(active_parent, next_debt, debt)

    next_effective_momentum = None
    repayment_utilization = None
    if routing_state is not None:
        routing_state[state_key] = next_debt.detach()
        if transport_active:
            routing_state[transport_state_key] = (
                current_need_distribution.detach()
            )
        if action_level_debt:
            routing_state[action_state_key] = next_action_debt.detach()
        if bounded_wait_reobservation:
            full_parent_service = served_dense >= child_counts.view(
                1, 1, 1, -1
            )
            next_parent_service_age = torch.where(
                remote_parent,
                parent_service_age + 1.0,
                torch.zeros_like(parent_service_age),
            )
            next_parent_service_age = torch.where(
                full_parent_service,
                torch.zeros_like(next_parent_service_age),
                next_parent_service_age,
            )
            routing_state[age_key] = next_parent_service_age.detach()
        if service_conditioned_momentum and not action_level_debt:
            if repayment_mode == "reset":
                repayment_utilization = current_need.new_zeros(batch, num_frames)
            else:
                requested = (
                    repayment * service_credit_scale * service_credit_dense
                ).sum(dim=(-2, -1))
                actual = (debt - next_debt).clamp_min(0.0).sum(dim=(-2, -1))
                repayment_utilization = torch.where(
                    requested > 1e-8,
                    actual / requested.clamp_min(1e-8),
                    torch.zeros_like(requested),
                ).clamp_(0.0, 1.0)
            next_effective_momentum = momentum * repayment_utilization
            routing_state[momentum_key] = next_effective_momentum.detach()

    selected_capacity = (
        child_counts[parent_indices.remainder(parents_per_frame)]
        if max_active > 0 else served_counts
    )
    selected_fraction = served_counts.float() / selected_capacity.clamp_min(1).float()
    frame_service = served_dense.sum(dim=-1).float()
    remote_frames = (~local_mask)[None].expand(batch, -1, -1)
    remote_frame_count = remote_frames.sum(dim=-1).clamp_min(1)
    frame_coverage = ((frame_service > 0) & remote_frames).sum(dim=-1).float()
    frame_coverage = frame_coverage / remote_frame_count.float()
    remote_frame_service = frame_service.masked_fill(~remote_frames, 0.0)
    mean_frame_service = remote_frame_service.sum(dim=-1) / remote_frame_count.float()
    frame_variance = (
        (remote_frame_service - mean_frame_service[..., None]).square()
        * remote_frames
    ).sum(dim=-1) / remote_frame_count.float()
    frame_cv = frame_variance.sqrt() / mean_frame_service.clamp_min(1e-8)


    stats = {
        "residual_budget_debt": float(
            debt.masked_select(remote_parent).mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_debt_after_repayment": float(
            next_debt.masked_select(remote_parent).mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_service_credit": float(
            service_credit_dense.mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_effective_service_credit": float(
            repayment_mode == "effective_bounded_deficit"
        ),
        "residual_budget_negative_credit_fraction": float(
            (next_debt < 0).float().mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_grouped_parent_candidates": float(max_active),
        "residual_budget_grouped_active_parent_fraction": float(
            (served_counts > 0).float().mean().item()
        ) if collect_stats and served_counts.numel() else 0.0,
        "residual_budget_grouped_partial_parent_fraction": float(
            ((selected_fraction > 0) & (selected_fraction < 1)).float().mean().item()
        ) if collect_stats and selected_fraction.numel() else 0.0,
        "residual_budget_grouped_child_budget": float(child_budget_per_query),
        "residual_budget_grouped_cost_aware": 1.0,
        "residual_budget_grouped_cost_power": 0.0,
        "residual_budget_grouped_mean_service_cost": float(
            served_counts.float().mean().item()
        ) if collect_stats and served_counts.numel() else 0.0,
        "residual_budget_grouped_capacity_promotion_fraction": 0.0,
        "residual_budget_grouped_candidate_override": -1.0,
        "residual_budget_frame_balance_fraction": 0.0,
        "residual_budget_frame_balance_budget": 0.0,
        "residual_budget_frame_balance_service": 0.0,
        "residual_budget_frame_balance_candidates": 0.0,
        "residual_budget_remote_frame_coverage": float(
            frame_coverage.mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_zero_service_frame_fraction": float(
            (1.0 - frame_coverage).mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_frame_service_cv": float(
            frame_cv.mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_service_conditioned_momentum": float(
            service_conditioned_momentum
        ),
        "residual_budget_relevance_bounded_debt": float(
            relevance_bounded_debt
        ),
        "residual_budget_action_level_debt": float(action_level_debt),
        "residual_budget_transportable_previous_debt": float(
            transported_previous_debt.masked_select(remote_parent).mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_rejected_previous_debt": float(
            rejected_previous_debt.masked_select(remote_parent).mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_effective_momentum": float(
            effective_momentum.mean().item()
        ) if collect_stats and effective_momentum is not None else float(momentum),
        "residual_budget_next_effective_momentum": float(
            next_effective_momentum.mean().item()
        ) if collect_stats and next_effective_momentum is not None else float(momentum),
        "residual_budget_unified_incremental_service": 1.0,
        "residual_budget_bounded_wait_reobservation": float(
            bounded_wait_reobservation
        ),
        "residual_budget_reobservation_service_cycle_layers": float(
            service_cycle_layers
        ),
        "residual_budget_reobservation_parent_age": float(
            parent_service_age.masked_select(remote_parent).mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_reobservation_max_parent_age": float(
            parent_service_age.masked_select(remote_parent).max().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_reobservation_overdue_parent_fraction": float(
            overdue_parent.masked_select(remote_parent).float().mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_reobservation_selected_overdue_fraction": float(
            (overdue_parent & active_parent)[active_parent].float().mean().item()
        ) if collect_stats and bool(active_parent.any()) else 0.0,
        "residual_budget_reobservation_forced_parent_fraction": float(
            forced_parent.masked_select(remote_parent).float().mean().item()
        ) if collect_stats and bool(remote_parent.any()) else 0.0,
        "residual_budget_reobservation_full_parent_fraction": float(
            (
                served_dense
                >= child_counts.view(1, 1, 1, -1)
            )[active_parent].float().mean().item()
        ) if collect_stats and bool(active_parent.any()) else 0.0,
        "residual_budget_unified_active_parents": float(
            active_count.float().mean().item()
        ) if collect_stats else 0.0,
        "residual_budget_unified_mean_service_depth": float(
            served_dense[active_parent].float().mean().item()
        ) if collect_stats and bool(active_parent.any()) else 0.0,
    }
    if pv_service_debt_feedback or pv_importance_alignment_observer:
        selected_action_values = {}
        if pv_importance_alignment_observer:
            selected_slot_ids = torch.arange(
                child_slots, device=served_counts.device
            )
            selected_slot_mask = (
                selected_slot_ids.view(1, 1, 1, -1)
                < served_counts[..., None]
            )

            def gather_selected_actions(source: torch.Tensor) -> torch.Tensor:
                flat_source = source.flatten(start_dim=-3, end_dim=-2)
                gathered = flat_source.gather(
                    -2,
                    parent_indices[..., None].expand(
                        *parent_indices.shape, child_slots
                    ),
                )
                return gathered[selected_slot_mask].view(
                    batch, num_frames, child_budget_per_query
                )

            historical_debt = (debt - current_need).clamp_min(0.0)
            selected_action_values = {
                "selected_action_priority": gather_selected_actions(
                    debt[..., None] * child_need
                ).detach(),
                "selected_current_action_priority": gather_selected_actions(
                    current_need[..., None] * child_need
                ).detach(),
                "selected_history_action_priority": gather_selected_actions(
                    historical_debt[..., None] * child_need
                ).detach(),
            }
        stats["_pv_service_debt_snapshot"] = {
            "state_key": state_key,
            "momentum_key": momentum_key,
            "age_key": age_key,
            "debt": debt.detach(),
            "current_need": current_need.detach(),
            "historical_debt": (debt - current_need).clamp_min(0.0).detach(),
            "nominal_next_debt": next_debt.detach(),
            "service_credit": service_credit_dense.detach(),
            "served_dense": served_dense.detach(),
            "parent_indices": parent_indices.detach(),
            "served_counts": served_counts.detach(),
            "child_counts": child_counts.detach(),
            "remote_parent": remote_parent.detach(),
            "parent_service_age": parent_service_age.detach(),
            "repayment": float(repayment),
            "service_credit_scale": float(service_credit_scale),
            "repayment_mode": repayment_mode,
            "momentum": float(momentum),
            "service_conditioned_momentum": bool(
                service_conditioned_momentum
            ),
            "bounded_wait_reobservation": bool(
                bounded_wait_reobservation
            ),
            **selected_action_values,
        }
    if service_observation_enabled(layer_idx):
        stats["_service_observation_snapshot"] = {
            "current_need": current_need.detach(),
            "debt": debt.detach(),
            "next_debt": next_debt.detach(),
            "remote_mask": remote_parent.detach(),
            "parent_indices": parent_indices.detach(),
            "served_counts": served_counts.detach(),
            "service_credit": service_credit_dense.detach(),
            "action_priority": action_priority.detach(),
            "current_action_priority": (
                current_need[..., None] * child_need
            ).detach(),
            "selected_actions": selected_actions.detach(),
            "current_selected_actions": current_selected_actions.detach(),
            "historical_action_priority": historical_action_priority.detach(),
            "transport_overlap": transport_stats["overlap"].detach(),
            "transport_debt_survival": (
                transport_stats["debt_weighted_survival"].detach()
            ),
            "transport_parent_survival": (
                transport_stats["parent_survival"].detach()
            ),
            "valid_child_actions": valid_child.detach(),
            "child_budget_per_query": int(child_budget_per_query),
            "momentum": float(momentum),
            "repayment": float(repayment),
            "service_credit_scale": float(service_credit_scale),
        }
        if action_level_debt:
            stats["_service_observation_snapshot"].update({
                "action_debt": action_liability.detach(),
                "next_action_debt": next_action_debt.detach(),
            })
    return parent_indices, served_counts, stats


def apply_pv_execution_service_feedback(
    execution_fraction_by_child_block: torch.Tensor,
    snapshot: dict,
    *,
    child_budget_per_query: int,
    child_block_size: int,
    routing_state: dict,
) -> dict[str, float]:
    """Recompute debt repayment from PV work units that actually executed.

    A work unit is one selected-child block for one query block and head. The
    kernel returns its binary execution decision; averaging those decisions
    yields the realized service fraction for every selected-child block. The
    scheduler credit is then repaid only by that realized fraction.
    """
    served_dense = snapshot["served_dense"]
    parent_indices = snapshot["parent_indices"]
    served_counts = snapshot["served_counts"]
    if child_budget_per_query <= 0:
        raise ValueError("PV-service feedback requires positive child budget")
    if child_block_size <= 0:
        raise ValueError("PV-service feedback requires positive block size")
    expected_blocks = math.ceil(child_budget_per_query / child_block_size)
    expected_shape = (
        served_dense.shape[0],
        served_dense.shape[1],
        expected_blocks,
    )
    if tuple(execution_fraction_by_child_block.shape) != expected_shape:
        raise ValueError(
            "PV execution map does not match grouped-service descriptors"
        )
    per_query_service = served_counts.sum(dim=-1)
    if not bool((per_query_service == child_budget_per_query).all()):
        raise RuntimeError("grouped service does not match PV child budget")

    child_slots = torch.arange(
        int(snapshot["child_counts"].max().item()),
        device=served_counts.device,
    )
    served_mask = child_slots.view(1, 1, 1, -1) < served_counts[..., None]
    expanded_parents = parent_indices[..., None].expand(
        *parent_indices.shape, child_slots.numel()
    )[served_mask].view(
        served_counts.shape[0], served_counts.shape[1], child_budget_per_query
    )
    child_positions = torch.arange(
        child_budget_per_query, device=served_counts.device
    )
    child_block_ids = torch.div(
        child_positions, child_block_size, rounding_mode="floor"
    )
    position_execution = execution_fraction_by_child_block.float().clamp(
        0.0, 1.0
    ).gather(
        -1,
        child_block_ids.view(1, 1, -1).expand(
            served_counts.shape[0], served_counts.shape[1], -1
        ),
    )
    missed_service = 1.0 - position_execution
    missed_dense = torch.zeros_like(served_dense, dtype=torch.float32)
    missed_dense.flatten(start_dim=-2).scatter_add_(
        -1, expanded_parents, missed_service
    )
    actual_fraction = torch.where(
        served_dense > 0,
        1.0 - missed_dense / served_dense.float().clamp_min(1.0),
        torch.zeros_like(missed_dense),
    ).clamp_(0.0, 1.0)

    debt = snapshot["debt"]
    service_credit = snapshot["service_credit"]
    requested_credit = (
        snapshot["repayment"]
        * snapshot["service_credit_scale"]
        * service_credit
    )
    realized_credit = requested_credit * actual_fraction
    corrected_next_debt = debt - realized_credit
    if snapshot["repayment_mode"] in {
        "bounded_deficit",
        "effective_bounded_deficit",
    }:
        corrected_next_debt = corrected_next_debt.clamp_min(0.0)
    corrected_next_debt = torch.where(
        served_dense > 0, corrected_next_debt, debt
    )
    routing_state[snapshot["state_key"]] = corrected_next_debt.detach()

    if snapshot["bounded_wait_reobservation"]:
        child_counts = snapshot["child_counts"].view(1, 1, 1, -1)
        fully_executed = (
            (served_dense >= child_counts)
            & (actual_fraction >= 1.0 - 1e-6)
        )
        next_age = torch.where(
            snapshot["remote_parent"],
            snapshot["parent_service_age"] + 1.0,
            torch.zeros_like(snapshot["parent_service_age"]),
        )
        next_age = torch.where(
            fully_executed, torch.zeros_like(next_age), next_age
        )
        routing_state[snapshot["age_key"]] = next_age.detach()

    if snapshot["service_conditioned_momentum"]:
        requested = requested_credit.sum(dim=(-2, -1))
        realized = (debt - corrected_next_debt).clamp_min(0.0).sum(
            dim=(-2, -1)
        )
        utilization = torch.where(
            requested > 1e-8,
            realized / requested.clamp_min(1e-8),
            torch.zeros_like(requested),
        ).clamp_(0.0, 1.0)
        next_momentum = snapshot["momentum"] * utilization
        routing_state[snapshot["momentum_key"]] = next_momentum.detach()
    else:
        next_momentum = None

    active = served_dense > 0
    total_service = served_dense.float().sum().clamp_min(1.0)
    actual_service_fraction = (
        (served_dense.float() * actual_fraction).sum() / total_service
    )
    restored_debt = (
        corrected_next_debt - snapshot["nominal_next_debt"]
    ).clamp_min(0.0)
    return {
        "residual_budget_pv_service_debt_feedback": 1.0,
        "residual_budget_pv_actual_service_fraction": float(
            actual_service_fraction.item()
        ),
        "residual_budget_pv_unserved_service_fraction": float(
            (1.0 - actual_service_fraction).item()
        ),
        "residual_budget_pv_restored_debt": float(
            restored_debt[active].mean().item()
        ) if bool(active.any()) else 0.0,
        "residual_budget_debt_after_pv_feedback": float(
            corrected_next_debt[
                snapshot["remote_parent"]
            ].mean().item()
        ) if bool(snapshot["remote_parent"].any()) else 0.0,
        "residual_budget_pv_next_effective_momentum": float(
            next_momentum.mean().item()
        ) if next_momentum is not None else float(snapshot["momentum"]),
    }


def measure_pv_importance_alignment(
    execution_fraction_by_child_block: torch.Tensor,
    snapshot: dict,
    *,
    child_budget_per_query: int,
    child_block_size: int,
) -> dict[str, float]:
    """Measure whether debt-selected work survives online-softmax skipping."""
    served_dense = snapshot["served_dense"]
    served_counts = snapshot["served_counts"]
    if child_budget_per_query <= 0 or child_block_size <= 0:
        raise ValueError("PV alignment observation requires positive budgets")
    expected_blocks = math.ceil(child_budget_per_query / child_block_size)
    expected_shape = (
        served_dense.shape[0],
        served_dense.shape[1],
        expected_blocks,
    )
    if tuple(execution_fraction_by_child_block.shape) != expected_shape:
        raise ValueError("PV execution map does not match alignment descriptors")
    if not bool(
        (served_counts.sum(dim=-1) == child_budget_per_query).all()
    ):
        raise RuntimeError("grouped service does not match PV child budget")

    child_slots = torch.arange(
        int(snapshot["child_counts"].max().item()),
        device=served_counts.device,
    )
    served_mask = child_slots.view(1, 1, 1, -1) < served_counts[..., None]
    expanded_parents = snapshot["parent_indices"][..., None].expand(
        *snapshot["parent_indices"].shape, child_slots.numel()
    )[served_mask].view(
        served_counts.shape[0], served_counts.shape[1], child_budget_per_query
    )
    block_ids = torch.div(
        torch.arange(child_budget_per_query, device=served_counts.device),
        child_block_size,
        rounding_mode="floor",
    )
    execution = execution_fraction_by_child_block.float().clamp(0.0, 1.0)
    execution = execution.gather(
        -1,
        block_ids.view(1, 1, -1).expand(
            served_counts.shape[0], served_counts.shape[1], -1
        ),
    )
    skipped = 1.0 - execution

    def selected_parent_values(name: str) -> torch.Tensor:
        return snapshot[name].flatten(start_dim=-2).gather(
            -1, expanded_parents
        ).float().clamp_min(0.0)

    def weighted_skip(weights: torch.Tensor) -> torch.Tensor:
        denominator = weights.sum()
        if float(denominator.item()) <= 1e-8:
            return skipped.mean()
        return (weights * skipped).sum() / denominator

    accumulated = selected_parent_values("debt")
    current = selected_parent_values("current_need")
    history = selected_parent_values("historical_debt")
    unweighted_skip = skipped.mean()
    accumulated_skip = weighted_skip(accumulated)
    current_skip = weighted_skip(current)
    history_skip = weighted_skip(history)

    flat_debt = accumulated.reshape(-1)
    flat_execution = execution.reshape(-1)
    debt_centered = flat_debt - flat_debt.mean()
    execution_centered = flat_execution - flat_execution.mean()
    correlation_denominator = (
        debt_centered.square().sum().sqrt()
        * execution_centered.square().sum().sqrt()
    )
    correlation = torch.where(
        correlation_denominator > 1e-8,
        (debt_centered * execution_centered).sum()
        / correlation_denominator.clamp_min(1e-8),
        flat_debt.new_zeros(()),
    )
    debt_quartile = torch.quantile(flat_debt, 0.75)
    top_debt = accumulated >= debt_quartile
    top_debt_skip = (
        skipped[top_debt].mean() if bool(top_debt.any()) else unweighted_skip
    )

    action_priority = snapshot.get("selected_action_priority")
    current_action = snapshot.get("selected_current_action_priority")
    history_action = snapshot.get("selected_history_action_priority")
    if action_priority is None or current_action is None or history_action is None:
        raise RuntimeError("PV alignment snapshot is missing child priorities")
    action_priority = action_priority.float().clamp_min(0.0)
    current_action = current_action.float().clamp_min(0.0)
    history_action = history_action.float().clamp_min(0.0)
    action_skip = weighted_skip(action_priority)
    current_action_skip = weighted_skip(current_action)
    history_action_skip = weighted_skip(history_action)
    flat_action = action_priority.reshape(-1)
    action_centered = flat_action - flat_action.mean()
    action_correlation_denominator = (
        action_centered.square().sum().sqrt()
        * execution_centered.square().sum().sqrt()
    )
    action_correlation = torch.where(
        action_correlation_denominator > 1e-8,
        (action_centered * execution_centered).sum()
        / action_correlation_denominator.clamp_min(1e-8),
        flat_action.new_zeros(()),
    )
    action_quartile = torch.quantile(flat_action, 0.75)
    top_action = action_priority >= action_quartile
    top_action_skip = (
        skipped[top_action].mean() if bool(top_action.any()) else unweighted_skip
    )

    return {
        "residual_budget_pv_alignment_observer": 1.0,
        "residual_budget_pv_alignment_unweighted_skip_fraction": float(
            unweighted_skip.item()
        ),
        "residual_budget_pv_alignment_accumulated_debt_weighted_skip": float(
            accumulated_skip.item()
        ),
        "residual_budget_pv_alignment_current_need_weighted_skip": float(
            current_skip.item()
        ),
        "residual_budget_pv_alignment_history_weighted_skip": float(
            history_skip.item()
        ),
        "residual_budget_pv_alignment_debt_skip_excess": float(
            (accumulated_skip - unweighted_skip).item()
        ),
        "residual_budget_pv_alignment_history_skip_excess": float(
            (history_skip - unweighted_skip).item()
        ),
        "residual_budget_pv_alignment_debt_execution_correlation": float(
            correlation.item()
        ),
        "residual_budget_pv_alignment_top_debt_quartile_skip": float(
            top_debt_skip.item()
        ),
        "residual_budget_pv_alignment_action_weighted_skip": float(
            action_skip.item()
        ),
        "residual_budget_pv_alignment_current_action_weighted_skip": float(
            current_action_skip.item()
        ),
        "residual_budget_pv_alignment_history_action_weighted_skip": float(
            history_action_skip.item()
        ),
        "residual_budget_pv_alignment_action_skip_excess": float(
            (action_skip - unweighted_skip).item()
        ),
        "residual_budget_pv_alignment_action_execution_correlation": float(
            action_correlation.item()
        ),
        "residual_budget_pv_alignment_top_action_quartile_skip": float(
            top_action_skip.item()
        ),
    }


def schedule_grouped_residual_debt_cells(
    current_need: torch.Tensor,
    child_counts: torch.Tensor,
    *,
    local_frame_radius: int,
    child_budget_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    repayment_mode: str,
    service_credit_scale: float,
    service_conditioned_momentum: bool = False,
    bounded_wait_reobservation: bool = False,
    effective_service_credit: torch.Tensor | None = None,
    service_costs: torch.Tensor | None = None,
    candidate_count_override: int | None = None,
    cost_power: float = 0.0,
    frame_balance_fraction: float = 0.0,
    incremental_child_need: torch.Tensor | None = None,
    target_sparsity: float | None = None,
    total_layers: int = 24,
    parent_carrier_execution: bool = False,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Schedule variable-cost parent cells and repay only served children."""
    if current_need.ndim != 4:
        raise ValueError("current_need must have shape [B, Q, K, parents]")
    if child_counts.ndim != 1 or child_counts.shape[0] != current_need.shape[-1]:
        raise ValueError("child_counts must have one entry per parent cell")
    if repayment_mode in {
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }:
        raise ValueError("age-based repayment is not implemented for grouped parents")
    if repayment_mode not in {
        "reset",
        "deficit",
        "bounded_deficit",
        "effective_bounded_deficit",
    }:
        raise ValueError("unsupported grouped-parent repayment mode")
    batch, num_frames, _, parents_per_frame = current_need.shape
    current_need = current_need.float().clamp_min(0.0)
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    child_counts = child_counts.to(
        device=current_need.device, dtype=torch.long
    )
    if bool((child_counts < 0).any()):
        raise ValueError("parent service capacities must be non-negative")
    if not bool((child_counts > 0).any()):
        raise ValueError("at least one parent must have positive capacity")
    if cost_power < 0.0:
        raise ValueError("grouped parent cost power must be non-negative")
    if not 0.0 <= frame_balance_fraction <= 1.0:
        raise ValueError("frame balance fraction must be in [0, 1]")
    if candidate_count_override is not None and candidate_count_override < 0:
        raise ValueError("candidate count override must be non-negative")
    if service_costs is not None:
        if tuple(service_costs.shape) != tuple(current_need.shape):
            raise ValueError("service costs must match current need shape")
        service_costs = service_costs.to(
            device=current_need.device, dtype=torch.long
        )
        max_child_counts = child_counts.view(1, 1, 1, -1)
        if bool(
            ((service_costs < 1) | (service_costs > max_child_counts)).any()
        ):
            raise ValueError("service costs must lie within each parent size")
    if incremental_child_need is not None:
        if service_costs is not None or candidate_count_override is not None:
            raise ValueError(
                "unified incremental service does not use parent cost buckets"
            )
        if frame_balance_fraction != 0.0:
            raise ValueError(
                "unified incremental service owns frame coverage allocation"
            )
        return _schedule_unified_incremental_service(
            current_need,
            child_counts,
            incremental_child_need,
            local_frame_radius=local_frame_radius,
            child_budget_per_query=child_budget_per_query,
            layer_idx=layer_idx,
            routing_state=routing_state,
            momentum=momentum,
            repayment=repayment,
            repayment_mode=repayment_mode,
            service_credit_scale=service_credit_scale,
            service_conditioned_momentum=service_conditioned_momentum,
            bounded_wait_reobservation=bounded_wait_reobservation,
            effective_service_credit=effective_service_credit,
            target_sparsity=target_sparsity,
            total_layers=total_layers,
            parent_carrier_execution=parent_carrier_execution,
            collect_stats=collect_stats,
        )

    frame_ids = torch.arange(num_frames, device=current_need.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    max_local_count = min(num_frames, 2 * local_frame_radius + 1)
    max_remote_frames = max(0, num_frames - max_local_count)
    max_child_budget = max_remote_frames * int(child_counts.sum().item())
    child_budget_per_query = min(child_budget_per_query, max_child_budget)

    state_key = "residual_budget_grouped_parent_debt"
    previous_debt = None if routing_state is None else routing_state.get(state_key)
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != tuple(current_need.shape)
    ):
        previous_debt = torch.zeros_like(current_need)
    else:
        previous_debt = previous_debt.to(current_need)
    momentum_key = "residual_budget_grouped_parent_momentum"
    effective_momentum = None
    if service_conditioned_momentum:
        effective_momentum = (
            None if routing_state is None else routing_state.get(momentum_key)
        )
        if (
            layer_idx == 0
            or effective_momentum is None
            or tuple(effective_momentum.shape) != (batch, num_frames)
        ):
            effective_momentum = current_need.new_full(
                (batch, num_frames), momentum
            )
        else:
            effective_momentum = effective_momentum.to(current_need)
        debt = (
            effective_momentum[..., None, None] * previous_debt
            + current_need
        )
    else:
        debt = momentum * previous_debt + current_need
    positive_capacity = child_counts > 0
    remote_mask = (
        (~local_mask)[None, :, :, None]
        & positive_capacity.view(1, 1, 1, -1)
    ).expand_as(debt)
    if service_costs is None:
        candidate_service_costs = child_counts.view(1, 1, 1, -1).expand_as(
            debt
        )
    else:
        candidate_service_costs = service_costs
    cost_adjusted_priority = debt / candidate_service_costs.float().pow(
        cost_power
    )
    flat_score = cost_adjusted_priority.masked_fill(
        ~remote_mask, float("-inf")
    ).flatten(start_dim=-2)

    max_parent_cost = int(child_counts.max().item())
    deficit_per_frame = int((max_parent_cost - child_counts).sum().item())
    max_candidates = (
        max_remote_frames * int(positive_capacity.sum().item())
    )
    if candidate_count_override is not None:
        candidate_count = min(max_candidates, candidate_count_override)
    elif service_costs is None:
        candidate_count = min(
            max_candidates,
            math.ceil(
                (
                    child_budget_per_query
                    + max_remote_frames * deficit_per_frame
                )
                / max_parent_cost
            ) if child_budget_per_query > 0 else 0,
        )
    else:
        candidate_count = min(max_candidates, child_budget_per_query)
    balanced_flat_mask = None
    balanced_flat_costs = None
    balanced_budget_per_query = None
    balanced_candidate_count = 0
    if candidate_count > 0:
        if frame_balance_fraction > 0.0:
            requested_balanced_budget = math.floor(
                child_budget_per_query * frame_balance_fraction
            )
            remote_frame_mask = ~local_mask
            remote_frame_count = remote_frame_mask.sum(dim=-1)
            per_frame_quota = torch.where(
                remote_frame_count > 0,
                torch.div(
                    requested_balanced_budget,
                    remote_frame_count.clamp_min(1),
                    rounding_mode="floor",
                ),
                torch.zeros_like(remote_frame_count),
            )
            balanced_budget_per_query = (
                per_frame_quota * remote_frame_count
            )

            frame_score = cost_adjusted_priority.masked_fill(
                ~remote_mask, float("-inf")
            )
            frame_order = frame_score.argsort(dim=-1, descending=True)
            ordered_costs = candidate_service_costs.gather(
                -1, frame_order
            )
            ordered_cost_before = (
                ordered_costs.cumsum(dim=-1) - ordered_costs
            )
            frame_quota = per_frame_quota.view(
                1, num_frames, 1, 1
            )
            ordered_balanced_mask = (
                (ordered_cost_before < frame_quota)
                & remote_mask.gather(-1, frame_order)
            )
            ordered_balanced_costs = torch.minimum(
                ordered_costs,
                (frame_quota - ordered_cost_before).clamp_min(0),
            )
            balanced_mask = torch.zeros_like(remote_mask)
            balanced_mask.scatter_(
                -1, frame_order, ordered_balanced_mask
            )
            balanced_costs = torch.zeros_like(candidate_service_costs)
            balanced_costs.scatter_(
                -1, frame_order, ordered_balanced_costs
            )
            balanced_flat_mask = balanced_mask.flatten(start_dim=-2)
            balanced_flat_costs = balanced_costs.flatten(start_dim=-2)

            global_order = flat_score.argsort(dim=-1, descending=True)
            global_rank = torch.empty_like(global_order)
            rank_values = torch.arange(
                global_order.shape[-1],
                device=debt.device,
                dtype=global_order.dtype,
            ).view(1, 1, -1)
            global_rank.scatter_(
                -1, global_order, rank_values.expand_as(global_order)
            )
            frame_balanced_order_key = global_rank + (
                (~balanced_flat_mask).to(global_rank.dtype)
                * global_order.shape[-1]
            )
            balanced_candidate_count = int(
                balanced_flat_mask.sum(dim=-1).max().item()
            )
            candidate_count = min(
                max_candidates,
                candidate_count + balanced_candidate_count,
            )
            parent_indices = frame_balanced_order_key.topk(
                candidate_count, dim=-1, largest=False
            ).indices
        else:
            parent_indices = flat_score.topk(
                candidate_count, dim=-1
            ).indices
        if service_costs is not None:
            while candidate_count < max_candidates:
                parent_ids = parent_indices.remainder(parents_per_frame)
                full_capacity = child_counts[parent_ids].sum(dim=-1)
                if bool((full_capacity >= child_budget_per_query).all()):
                    break
                shortfall = int(
                    (child_budget_per_query - full_capacity)
                    .clamp_min(0)
                    .max()
                    .item()
                )
                candidate_count = min(
                    max_candidates,
                    candidate_count
                    + math.ceil(shortfall / max_parent_cost),
                )
                if frame_balance_fraction > 0.0:
                    parent_indices = frame_balanced_order_key.topk(
                        candidate_count, dim=-1, largest=False
                    ).indices
                else:
                    parent_indices = flat_score.topk(
                        candidate_count, dim=-1
                    ).indices
        flat_service_costs = candidate_service_costs.flatten(start_dim=-2)
        candidate_costs = flat_service_costs.gather(-1, parent_indices)
        if frame_balance_fraction > 0.0:
            selected_balanced = balanced_flat_mask.gather(
                -1, parent_indices
            )
            selected_balanced_costs = balanced_flat_costs.gather(
                -1, parent_indices
            )
            candidate_costs = torch.where(
                selected_balanced,
                selected_balanced_costs,
                candidate_costs,
            )
        else:
            selected_balanced = None
        promoted_service = torch.zeros_like(candidate_costs)
        if service_costs is not None:
            parent_ids = parent_indices.remainder(parents_per_frame)
            full_costs = child_counts[parent_ids]
            capacity_shortfall = (
                child_budget_per_query - candidate_costs.sum(dim=-1)
            ).clamp_min(0)
            if selected_balanced is None:
                extra_capacity = full_costs - candidate_costs
            else:
                extra_capacity = torch.where(
                    selected_balanced,
                    torch.zeros_like(full_costs),
                    full_costs - candidate_costs,
                )
            extra_consumed_before = (
                extra_capacity.cumsum(dim=-1) - extra_capacity
            )
            promoted_service = torch.minimum(
                (capacity_shortfall[..., None] - extra_consumed_before)
                .clamp_min(0),
                extra_capacity,
            )
            candidate_costs = candidate_costs + promoted_service
        consumed_before = candidate_costs.cumsum(dim=-1) - candidate_costs
        served_counts = (
            child_budget_per_query - consumed_before
        ).clamp(min=0)
        served_counts = torch.minimum(served_counts, candidate_costs)
        if not bool(
            (served_counts.sum(dim=-1) == child_budget_per_query).all()
        ):
            raise RuntimeError("grouped parent candidates did not fill child budget")
        service_fraction = served_counts.float() / candidate_costs.float()

        flat_debt = debt.flatten(start_dim=-2)
        selected_debt = flat_debt.gather(-1, parent_indices)
        if repayment_mode == "reset":
            selected_next_debt = selected_debt * (1.0 - service_fraction)
            selected_service_credit = served_counts.float()
            mean_credit = 0.0
        else:
            remote_demand = current_need.masked_fill(
                ~remote_mask, 0.0
            ).sum(dim=(-2, -1))
            credit_per_child = remote_demand / max(child_budget_per_query, 1)
            credit = credit_per_child[..., None] * served_counts.float()
            if repayment_mode == "effective_bounded_deficit":
                if effective_service_credit is None:
                    raise ValueError(
                        "effective grouped repayment requires service credit"
                    )
                effective = effective_service_credit.float().clamp_min(0.0)
                selected_effective = effective.flatten(start_dim=-2).gather(
                    -1, parent_indices
                )
                active = served_counts > 0
                active_mean = (
                    (selected_effective * active).sum(dim=-1, keepdim=True)
                    / active.sum(dim=-1, keepdim=True).clamp_min(1)
                )
                credit = credit * torch.where(
                    active_mean > 1e-8,
                    selected_effective / active_mean.clamp_min(1e-8),
                    torch.ones_like(selected_effective),
                )
            selected_service_credit = credit
            selected_next_debt = selected_debt - (
                repayment * service_credit_scale * credit
            )
            if repayment_mode in {
                "bounded_deficit",
                "effective_bounded_deficit",
            }:
                selected_next_debt = selected_next_debt.clamp_min(0.0)
            selected_next_debt = torch.where(
                served_counts > 0, selected_next_debt, selected_debt
            )
            mean_credit = float(credit.mean().item()) if collect_stats else 0.0
        next_debt = flat_debt.scatter(
            -1, parent_indices, selected_next_debt
        ).view_as(debt)
    else:
        parent_indices = torch.empty(
            batch, num_frames, 0, device=debt.device, dtype=torch.long
        )
        served_counts = torch.empty_like(parent_indices)
        service_fraction = torch.empty_like(parent_indices, dtype=debt.dtype)
        selected_balanced = None
        selected_service_credit = torch.empty_like(
            parent_indices, dtype=debt.dtype
        )
        next_debt = debt
        mean_credit = 0.0

    next_effective_momentum = None
    repayment_utilization = None
    if routing_state is not None:
        routing_state[state_key] = next_debt.detach()
        if service_conditioned_momentum:
            if candidate_count > 0 and repayment_mode != "reset":
                requested_repayment = (
                    repayment
                    * service_credit_scale
                    * selected_service_credit
                )
                actual_repayment = (
                    selected_debt - selected_next_debt
                ).clamp_min(0.0)
                requested_mass = requested_repayment.sum(dim=-1)
                repaid_mass = actual_repayment.sum(dim=-1)
                utilization = torch.where(
                    requested_mass > 1e-8,
                    repaid_mass / requested_mass.clamp_min(1e-8),
                    torch.zeros_like(requested_mass),
                ).clamp_(0.0, 1.0)
            else:
                utilization = current_need.new_zeros(batch, num_frames)
            repayment_utilization = utilization
            next_effective_momentum = momentum * utilization
            routing_state[momentum_key] = next_effective_momentum.detach()

    momentum_diagnostics = {}
    if collect_stats and service_conditioned_momentum:
        for prefix, tensor in (
            ("effective", effective_momentum),
            ("next_effective", next_effective_momentum),
            ("repayment_utilization", repayment_utilization),
        ):
            if tensor is None or tensor.numel() == 0:
                continue
            flat = tensor.detach().float().flatten()
            quantiles = torch.quantile(
                flat,
                flat.new_tensor((0.10, 0.50, 0.90)),
            )
            key = f"residual_budget_{prefix}_momentum"
            if prefix == "repayment_utilization":
                key = "residual_budget_repayment_utilization"
            momentum_diagnostics.update(
                {
                    f"{key}_mean": float(flat.mean().item()),
                    f"{key}_std": float(flat.std(unbiased=False).item()),
                    f"{key}_min": float(flat.min().item()),
                    f"{key}_p10": float(quantiles[0].item()),
                    f"{key}_p50": float(quantiles[1].item()),
                    f"{key}_p90": float(quantiles[2].item()),
                    f"{key}_max": float(flat.max().item()),
                }
            )
        if next_effective_momentum is not None:
            flat_next = next_effective_momentum.detach().float().flatten()
            momentum_diagnostics.update(
                {
                    "residual_budget_next_effective_momentum_zero_fraction": float(
                        (flat_next <= 1e-6).float().mean().item()
                    ),
                    "residual_budget_next_effective_momentum_cap_fraction": float(
                        (flat_next >= momentum - 1e-6).float().mean().item()
                    ),
                }
            )
    active_parent_fraction = (
        float((served_counts > 0).float().mean().item())
        if collect_stats and served_counts.numel() > 0
        else 0.0
    )
    partial_parent_fraction = (
        float(
            ((service_fraction > 0) & (service_fraction < 1))
            .float().mean().item()
        )
        if collect_stats and service_fraction.numel() > 0
        else 0.0
    )
    balanced_service_mean = 0.0
    frame_coverage_mean = 0.0
    zero_service_frame_fraction = 0.0
    frame_service_cv_mean = 0.0
    if collect_stats:
        if selected_balanced is not None:
            balanced_service_mean = float(
                (
                    served_counts
                    * selected_balanced.to(served_counts.dtype)
                ).sum(dim=-1).float().mean().item()
            )
        selected_key_frames = torch.div(
            parent_indices,
            parents_per_frame,
            rounding_mode="floor",
        )
        frame_service = torch.zeros(
            batch,
            num_frames,
            num_frames,
            device=debt.device,
            dtype=served_counts.dtype,
        )
        if served_counts.shape[-1] > 0:
            frame_service.scatter_add_(
                -1, selected_key_frames, served_counts
            )
        remote_frame_mask = (~local_mask)[None].expand(
            batch, -1, -1
        )
        remote_frame_count = remote_frame_mask.sum(dim=-1).clamp_min(1)
        served_frame_count = (
            (frame_service > 0) & remote_frame_mask
        ).sum(dim=-1)
        frame_coverage = (
            served_frame_count.float() / remote_frame_count.float()
        )
        remote_frame_service = frame_service.float().masked_fill(
            ~remote_frame_mask, 0.0
        )
        mean_frame_service = (
            remote_frame_service.sum(dim=-1)
            / remote_frame_count.float()
        )
        frame_service_variance = (
            (
                (
                    remote_frame_service
                    - mean_frame_service[..., None]
                ).square()
                * remote_frame_mask
            ).sum(dim=-1)
            / remote_frame_count.float()
        )
        frame_service_cv = (
            frame_service_variance.sqrt()
            / mean_frame_service.clamp_min(1e-8)
        )
        frame_coverage_mean = float(frame_coverage.mean().item())
        zero_service_frame_fraction = float(
            (1.0 - frame_coverage).mean().item()
        )
        frame_service_cv_mean = float(frame_service_cv.mean().item())
    stats = {
        "residual_budget_debt": (
            float(debt.masked_select(remote_mask).mean().item())
            if collect_stats and bool(remote_mask.any()) else 0.0
        ),
        "residual_budget_debt_after_repayment": (
            float(next_debt.masked_select(remote_mask).mean().item())
            if collect_stats and bool(remote_mask.any()) else 0.0
        ),
        "residual_budget_service_credit": mean_credit,
        "residual_budget_effective_service_credit": float(
            repayment_mode == "effective_bounded_deficit"
        ),
        "residual_budget_negative_credit_fraction": (
            float((next_debt < 0).float().mean().item())
            if collect_stats else 0.0
        ),
        "residual_budget_grouped_parent_candidates": float(candidate_count),
        "residual_budget_grouped_active_parent_fraction": active_parent_fraction,
        "residual_budget_grouped_partial_parent_fraction": partial_parent_fraction,
        "residual_budget_grouped_child_budget": float(child_budget_per_query),
        "residual_budget_grouped_cost_aware": float(service_costs is not None),
        "residual_budget_grouped_cost_power": float(cost_power),
        "residual_budget_grouped_mean_service_cost": (
            float(candidate_costs.float().mean().item())
            if collect_stats and candidate_count > 0 else 0.0
        ),
        "residual_budget_grouped_capacity_promotion_fraction": (
            float((promoted_service > 0).float().mean().item())
            if collect_stats and candidate_count > 0 else 0.0
        ),
        "residual_budget_grouped_candidate_override": float(
            candidate_count_override
            if candidate_count_override is not None else -1
        ),
        "residual_budget_frame_balance_fraction": float(
            frame_balance_fraction
        ),
        "residual_budget_frame_balance_budget": (
            float(balanced_budget_per_query.float().mean().item())
            if collect_stats and balanced_budget_per_query is not None
            else 0.0
        ),
        "residual_budget_frame_balance_service": (
            balanced_service_mean
        ),
        "residual_budget_frame_balance_candidates": float(
            balanced_candidate_count
        ),
        "residual_budget_remote_frame_coverage": (
            frame_coverage_mean
        ),
        "residual_budget_zero_service_frame_fraction": (
            zero_service_frame_fraction
        ),
        "residual_budget_frame_service_cv": (
            frame_service_cv_mean
        ),
        "residual_budget_service_conditioned_momentum": float(
            service_conditioned_momentum
        ),
        "residual_budget_effective_momentum": (
            float(effective_momentum.mean().item())
            if collect_stats and effective_momentum is not None
            else float(momentum)
        ),
        "residual_budget_next_effective_momentum": (
            float(next_effective_momentum.mean().item())
            if collect_stats and next_effective_momentum is not None
            else float(momentum)
        ),
        **momentum_diagnostics,
    }
    if service_observation_enabled(layer_idx):
        service_credit = torch.zeros_like(debt).flatten(start_dim=-2)
        if parent_indices.shape[-1] > 0:
            service_credit.scatter_add_(
                -1,
                parent_indices,
                selected_service_credit.to(service_credit.dtype),
            )
        service_credit = service_credit.view_as(debt)
        stats["_service_observation_snapshot"] = {
            "current_need": current_need.detach(),
            "debt": debt.detach(),
            "next_debt": next_debt.detach(),
            "remote_mask": remote_mask.detach(),
            "parent_indices": parent_indices.detach(),
            "served_counts": served_counts.detach(),
            "service_credit": service_credit.detach(),
            "child_budget_per_query": int(child_budget_per_query),
            "momentum": float(momentum),
            "repayment": float(repayment),
            "service_credit_scale": float(service_credit_scale),
        }
    return parent_indices, served_counts, stats


def schedule_mixed_parent_debt_cells(
    current_need: torch.Tensor,
    *,
    local_frame_radius: int,
    extra_budget_per_query: int,
    children_per_parent: int,
    residuals_per_parent: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    repayment_mode: str,
    service_credit_scale: float,
    effective_service_credit: torch.Tensor | None = None,
    cost_power: float = 0.0,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Spend a fixed token budget on parent promotion and child residuals."""
    if current_need.ndim != 4:
        raise ValueError("current_need must have shape [B, Q, K, parents]")
    if children_per_parent < 2:
        raise ValueError("mixed parent execution requires at least two children")
    if not 0 <= residuals_per_parent <= children_per_parent:
        raise ValueError("mixed residual cap must lie within the parent size")
    if repayment_mode not in {
        "reset", "deficit", "bounded_deficit", "effective_bounded_deficit"
    }:
        raise ValueError("unsupported mixed-parent repayment mode")
    if cost_power < 0.0:
        raise ValueError("mixed parent cost power must be non-negative")

    batch, num_frames, _, parents_per_frame = current_need.shape
    current_need = current_need.float().clamp_min(0.0)
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)
    frame_ids = torch.arange(num_frames, device=current_need.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    remote_mask = (~local_mask)[None, :, :, None].expand_as(current_need)
    max_remote_parents = int(remote_mask[0, 0].sum().item())

    promotion_cost = children_per_parent - 1
    full_service_cost = promotion_cost + residuals_per_parent
    extra_budget_per_query = max(0, int(extra_budget_per_query))
    hard_parents_per_query = min(
        max_remote_parents,
        math.ceil(extra_budget_per_query / full_service_cost)
        if extra_budget_per_query >= promotion_cost else 0,
    )
    promotion_budget = hard_parents_per_query * promotion_cost
    residual_budget = min(
        hard_parents_per_query * residuals_per_parent,
        max(0, extra_budget_per_query - promotion_budget),
    )
    realized_extra_budget = promotion_budget + residual_budget

    state_key = "residual_budget_mixed_parent_debt"
    previous_debt = None if routing_state is None else routing_state.get(state_key)
    if (
        layer_idx == 0
        or previous_debt is None
        or tuple(previous_debt.shape) != tuple(current_need.shape)
    ):
        previous_debt = torch.zeros_like(current_need)
    else:
        previous_debt = previous_debt.to(current_need)
    debt = momentum * previous_debt + current_need
    priority = debt / float(full_service_cost) ** cost_power
    flat_priority = priority.masked_fill(
        ~remote_mask, float("-inf")
    ).flatten(start_dim=-2)

    if hard_parents_per_query > 0:
        parent_indices = flat_priority.topk(
            hard_parents_per_query, dim=-1
        ).indices
        served_counts = torch.zeros_like(parent_indices)
        if residual_budget > 0:
            full_parents, partial_children = divmod(
                residual_budget, residuals_per_parent
            )
            served_counts[..., :full_parents] = residuals_per_parent
            if partial_children > 0:
                served_counts[..., full_parents] = partial_children
        flat_debt = debt.flatten(start_dim=-2)
        selected_debt = flat_debt.gather(-1, parent_indices)
        service_fraction = (
            promotion_cost + served_counts.float()
        ) / full_service_cost
        if repayment_mode == "reset":
            selected_next_debt = selected_debt * (1.0 - service_fraction)
            mean_credit = 0.0
        else:
            remote_demand = current_need.masked_fill(
                ~remote_mask, 0.0
            ).sum(dim=(-2, -1))
            credit_per_unit = remote_demand / max(realized_extra_budget, 1)
            credit = credit_per_unit[..., None] * (
                promotion_cost + served_counts.float()
            )
            if repayment_mode == "effective_bounded_deficit":
                if effective_service_credit is None:
                    raise ValueError(
                        "effective mixed repayment requires service credit"
                    )
                selected_effective = effective_service_credit.float().flatten(
                    start_dim=-2
                ).gather(-1, parent_indices)
                effective_mean = selected_effective.mean(
                    dim=-1, keepdim=True
                )
                credit = credit * torch.where(
                    effective_mean > 1e-8,
                    selected_effective / effective_mean.clamp_min(1e-8),
                    torch.ones_like(selected_effective),
                )
            selected_next_debt = selected_debt - (
                repayment * service_credit_scale * credit
            )
            if repayment_mode in {
                "bounded_deficit", "effective_bounded_deficit"
            }:
                selected_next_debt = selected_next_debt.clamp_min(0.0)
            mean_credit = float(credit.mean().item()) if collect_stats else 0.0
        next_debt = flat_debt.scatter(
            -1, parent_indices, selected_next_debt
        ).view_as(debt)
    else:
        parent_indices = torch.empty(
            batch, num_frames, 0, device=debt.device, dtype=torch.long
        )
        served_counts = torch.empty_like(parent_indices)
        next_debt = debt
        mean_credit = 0.0

    if routing_state is not None:
        routing_state[state_key] = next_debt.detach()
    stats = {
        "residual_budget_debt": (
            float(debt.masked_select(remote_mask).mean().item())
            if collect_stats and bool(remote_mask.any()) else 0.0
        ),
        "residual_budget_debt_after_repayment": (
            float(next_debt.masked_select(remote_mask).mean().item())
            if collect_stats and bool(remote_mask.any()) else 0.0
        ),
        "residual_budget_service_credit": mean_credit,
        "residual_budget_negative_credit_fraction": (
            float((next_debt < 0).float().mean().item())
            if collect_stats else 0.0
        ),
        "residual_budget_mixed_parent_execution": 1.0,
        "residual_budget_mixed_hard_parents_per_query": float(
            hard_parents_per_query
        ),
        "residual_budget_mixed_promotion_budget": float(promotion_budget),
        "residual_budget_mixed_residual_budget": float(residual_budget),
        "residual_budget_mixed_residuals_per_parent": float(
            residuals_per_parent
        ),
        "residual_budget_mixed_realized_extra_budget": float(
            realized_extra_budget
        ),
        "residual_budget_mixed_unspent_budget": float(
            extra_budget_per_query - realized_extra_budget
        ),
    }
    return parent_indices, served_counts, stats


def expand_grouped_parent_selection(
    parent_indices: torch.Tensor,
    served_counts: torch.Tensor,
    parent_to_children: torch.Tensor,
    *,
    parents_per_frame: int,
    children_per_frame: int,
    child_budget_per_query: int,
    ordered_children: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand grouped parent service into exact flattened child indices."""
    if parent_indices.shape != served_counts.shape:
        raise ValueError("parent indices and served counts must match")
    if child_budget_per_query == 0:
        return parent_indices[..., :0]
    parent_ids = parent_indices.remainder(parents_per_frame)
    key_frames = torch.div(
        parent_indices, parents_per_frame, rounding_mode="floor"
    )
    if ordered_children is None:
        child_ids = parent_to_children[parent_ids]
    else:
        if ordered_children.ndim != 4:
            raise ValueError("ordered children must have shape [B, K, P, C]")
        if (
            ordered_children.shape[0] != parent_indices.shape[0]
            or ordered_children.shape[2] != parents_per_frame
            or ordered_children.shape[3] != parent_to_children.shape[-1]
        ):
            raise ValueError("ordered children do not match parent selection")
        batch_ids = torch.arange(
            parent_indices.shape[0], device=parent_indices.device
        ).view(-1, 1, 1).expand_as(parent_ids)
        child_ids = ordered_children[batch_ids, key_frames, parent_ids]
    flat_child_indices = key_frames[..., None] * children_per_frame + child_ids
    slot_ids = torch.arange(
        parent_to_children.shape[-1], device=parent_indices.device
    )
    served_mask = slot_ids < served_counts[..., None]
    expanded = flat_child_indices[served_mask].view(
        *parent_indices.shape[:-1], child_budget_per_query
    )
    return expanded


def encode_grouped_parent_service_masks(
    parent_indices: torch.Tensor,
    served_counts: torch.Tensor,
    ordered_children: torch.Tensor,
    substituted_parent_indices: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    parents_per_frame: int,
) -> torch.Tensor:
    """Encode active 4x4 parents as canonical four-child service masks.

    The scheduler may rotate child service order across layers. The mask is
    deliberately expressed in the canonical ``parent_to_children`` order so
    execution can recover child indices without carrying a flat descriptor
    for every selected child. A substituted carrier serves every valid child.
    """
    if parent_indices.shape != served_counts.shape:
        raise ValueError("parent indices and served counts must match")
    if parent_indices.ndim != 3:
        raise ValueError("parent indices must have shape [B, Q, K]")
    if ordered_children.ndim != 4:
        raise ValueError("ordered children must have shape [B, F, P, C]")
    if parent_to_children.shape != parent_child_valid.shape:
        raise ValueError("parent child indices and validity must match")
    if parent_to_children.shape[0] != parents_per_frame:
        raise ValueError("parent map does not match parents per frame")
    child_slots = parent_to_children.shape[-1]
    if child_slots != 4:
        raise ValueError("parent service masks require exactly four child slots")
    if (
        ordered_children.shape[0] != parent_indices.shape[0]
        or ordered_children.shape[2:] != parent_to_children.shape
    ):
        raise ValueError("ordered children do not match parent selection")
    if substituted_parent_indices.shape[:2] != parent_indices.shape[:2]:
        raise ValueError("substituted parents must match batch and queries")
    if bool((served_counts < 0).any()) or bool((served_counts > child_slots).any()):
        raise ValueError("served counts must fit the four-child mask")

    parent_ids = parent_indices.remainder(parents_per_frame)
    key_frames = torch.div(
        parent_indices, parents_per_frame, rounding_mode="floor"
    )
    batch_ids = torch.arange(
        parent_indices.shape[0], device=parent_indices.device
    ).view(-1, 1, 1).expand_as(parent_ids)
    selected_order = ordered_children[
        batch_ids, key_frames, parent_ids
    ]
    canonical_children = parent_to_children[parent_ids]
    canonical_valid = parent_child_valid[parent_ids]
    served_slots = (
        torch.arange(child_slots, device=parent_indices.device)
        < served_counts[..., None]
    )
    selected_canonical = (
        (
            selected_order[..., :, None]
            == canonical_children[..., None, :]
        )
        & served_slots[..., :, None]
        & canonical_valid[..., None, :]
    ).any(dim=-2)

    bit_values = 1 << torch.arange(
        child_slots, device=parent_indices.device, dtype=torch.int32
    )
    service_masks = (
        selected_canonical.to(torch.int32) * bit_values
    ).sum(dim=-1, dtype=torch.int32)
    if substituted_parent_indices.shape[-1] > 0:
        substituted = (
            parent_indices[..., None]
            == substituted_parent_indices[..., None, :]
        ).any(dim=-1)
        valid_masks = (
            canonical_valid.to(torch.int32) * bit_values
        ).sum(dim=-1, dtype=torch.int32)
        service_masks = torch.where(
            substituted, valid_masks, service_masks
        )
    return service_masks.contiguous()


def expand_grouped_parent_service_masks(
    parent_indices: torch.Tensor,
    service_masks: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    parents_per_frame: int,
    children_per_frame: int,
) -> torch.Tensor:
    """Expand canonical parent masks for validation and generic fallbacks."""
    if parent_indices.shape != service_masks.shape:
        raise ValueError("parent indices and service masks must match")
    if parent_to_children.shape != parent_child_valid.shape:
        raise ValueError("parent child indices and validity must match")
    child_slots = parent_to_children.shape[-1]
    if child_slots != 4:
        raise ValueError("parent service masks require exactly four child slots")
    parent_ids = parent_indices.remainder(parents_per_frame)
    key_frames = torch.div(
        parent_indices, parents_per_frame, rounding_mode="floor"
    )
    child_ids = parent_to_children[parent_ids]
    slot_bits = 1 << torch.arange(
        child_slots, device=parent_indices.device, dtype=torch.int32
    )
    selected = (
        (service_masks[..., None].to(torch.int32) & slot_bits) != 0
    ) & parent_child_valid[parent_ids]
    per_query_counts = selected.sum(dim=(-2, -1))
    if int(per_query_counts.min().item()) != int(per_query_counts.max().item()):
        raise ValueError("service masks must have equal query budgets")
    flat_children = key_frames[..., None] * children_per_frame + child_ids
    return flat_children[selected].view(
        *parent_indices.shape[:2], int(per_query_counts.min().item())
    )


def parent_service_mask_tail_descriptors(
    parent_indices: torch.Tensor,
    service_masks: torch.Tensor,
    parent_to_children: torch.Tensor,
    *,
    parents_per_frame: int,
    children_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack the fourth child of fully upgraded parents into a compact tail."""
    if parent_indices.shape != service_masks.shape:
        raise ValueError("parent indices and service masks must match")
    if parent_to_children.shape != (parents_per_frame, 4):
        raise ValueError("parent child map must have four canonical slots")
    slot_bits = 1 << torch.arange(
        4, device=service_masks.device, dtype=torch.int32
    )
    selected = (
        service_masks[..., None].to(torch.int32) & slot_bits
    ) != 0
    has_tail = selected.sum(dim=-1) == 4
    tail_counts = has_tail.sum(dim=-1)
    max_tail = int(tail_counts.max().item()) if tail_counts.numel() else 0
    tail_indices = torch.zeros(
        *parent_indices.shape[:2],
        max_tail,
        device=parent_indices.device,
        dtype=parent_indices.dtype,
    )
    tail_valid = torch.zeros_like(tail_indices, dtype=torch.bool)
    if max_tail == 0:
        return tail_indices, tail_valid
    parent_ids = parent_indices.remainder(parents_per_frame)
    key_frames = torch.div(
        parent_indices, parents_per_frame, rounding_mode="floor"
    )
    fourth_children = (
        key_frames * children_per_frame
        + parent_to_children[parent_ids, 3]
    )
    batch_ids, query_ids, parent_slots = has_tail.nonzero(as_tuple=True)
    tail_slots = has_tail.cumsum(dim=-1)[
        batch_ids, query_ids, parent_slots
    ] - 1
    tail_indices[batch_ids, query_ids, tail_slots] = fourth_children[
        batch_ids, query_ids, parent_slots
    ]
    tail_valid[batch_ids, query_ids, tail_slots] = True
    return tail_indices.contiguous(), tail_valid.contiguous()


def select_additive_parent_substitutions(
    parent_indices: torch.Tensor,
    served_counts: torch.Tensor,
    child_counts: torch.Tensor,
    ordered_children: torch.Tensor,
    *,
    parents_per_frame: int,
    children_per_frame: int,
    substitution_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap selected parent carriers for their one missing child at equal cost."""
    if parent_indices.shape != served_counts.shape:
        raise ValueError("parent indices and served counts must match")
    if not 0.0 <= substitution_fraction <= 1.0:
        raise ValueError("substitution fraction must be in [0, 1]")
    if ordered_children.ndim != 4:
        raise ValueError("ordered children must have shape [B, K, P, C]")
    if substitution_fraction == 0.0 or parent_indices.shape[-1] == 0:
        empty = parent_indices[..., :0]
        return empty, empty

    parent_ids = parent_indices.remainder(parents_per_frame)
    key_frames = torch.div(
        parent_indices, parents_per_frame, rounding_mode="floor"
    )
    capacities = child_counts.to(parent_indices.device)[parent_ids]
    eligible = (capacities > 1) & (served_counts == capacities - 1)
    available = int(eligible.sum(dim=-1).min().item())
    substitution_count = min(
        available,
        round(available * substitution_fraction),
    )
    if substitution_count == 0:
        empty = parent_indices[..., :0]
        return empty, empty

    candidate_rank = torch.arange(
        parent_indices.shape[-1], device=parent_indices.device
    ).view(1, 1, -1)
    ranked_eligible = (-candidate_rank).expand_as(
        parent_indices
    ).masked_fill(~eligible, torch.iinfo(torch.long).min)
    substitution_positions = ranked_eligible.topk(
        substitution_count, dim=-1
    ).indices
    substituted_parents = parent_indices.gather(
        -1, substitution_positions
    )
    substituted_parent_ids = parent_ids.gather(
        -1, substitution_positions
    )
    substituted_frames = key_frames.gather(
        -1, substitution_positions
    )
    missing_slots = served_counts.gather(
        -1, substitution_positions
    )
    batch_ids = torch.arange(
        parent_indices.shape[0], device=parent_indices.device
    ).view(-1, 1, 1).expand_as(substituted_parent_ids)
    missing_child_ids = ordered_children[
        batch_ids,
        substituted_frames,
        substituted_parent_ids,
        missing_slots,
    ]
    missing_children = (
        substituted_frames * children_per_frame + missing_child_ids
    )
    return substituted_parents, missing_children


def select_residual_debt_cells(
    query_carrier: torch.Tensor,
    key_carrier: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_value: torch.Tensor,
    *,
    local_frame_radius: int,
    extra_tokens_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    repayment: float,
    repayment_mode: str = "reset",
    service_credit_scale: float = 1.0,
    temperature: float,
    frame_detail_power: float,
    spatial_detail_power: float,
    cell_scorer: str = "carrier_residual",
    carrier_residual_balance: float = 0.5,
    cell_importance: torch.Tensor | None = None,
    cell_importance_weight: float = 0.0,
    cell_importance_floor: float = 0.0,
    cell_importance_gate_threshold: float = 0.0,
    cell_importance_gate_temperature: float = 0.05,
    collect_stats: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Repay fine-grained spatial debt without dropping coarse frame links."""
    if query_carrier.shape != key_carrier.shape:
        raise ValueError("query and key carriers must have identical shapes")
    if coarse_value.shape != residual_value.shape:
        raise ValueError("coarse and residual values must have identical shapes")
    if not 0.0 <= momentum <= 1.0:
        raise ValueError("residual debt momentum must be in [0, 1]")
    if not 0.0 <= repayment <= 1.0:
        raise ValueError("residual debt repayment must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("residual debt temperature must be positive")
    if not 0.0 <= frame_detail_power <= 1.0:
        raise ValueError("frame_detail_power must be in [0, 1]")
    if not 0.0 <= spatial_detail_power <= 1.0:
        raise ValueError("spatial_detail_power must be in [0, 1]")
    if not 0.0 <= carrier_residual_balance <= 1.0:
        raise ValueError("carrier_residual_balance must be in [0, 1]")
    scorer_ids = {
        "uniform": 0,
        "residual": 1,
        "carrier": 2,
        "carrier_residual": 3,
        "patch_qk": 4,
    }
    if cell_scorer not in scorer_ids:
        raise ValueError(
            "cell_scorer must be uniform, residual, carrier, "
            "carrier_residual, or patch_qk"
        )
    if not 0.0 <= cell_importance_weight <= 1.0:
        raise ValueError("cell_importance_weight must be in [0, 1]")
    if not 0.0 <= cell_importance_floor <= cell_importance_weight:
        raise ValueError(
            "cell_importance_floor must be in [0, cell_importance_weight]"
        )
    if cell_importance_gate_threshold < 0.0:
        raise ValueError("cell_importance_gate_threshold must be non-negative")
    if cell_importance_gate_temperature <= 0.0:
        raise ValueError("cell_importance_gate_temperature must be positive")

    batch, heads, num_frames, _ = query_carrier.shape
    cells_per_frame = coarse_value.shape[-2]
    max_local_count = min(num_frames, 2 * local_frame_radius + 1)
    max_remote_tokens = max(
        0, (num_frames - max_local_count) * cells_per_frame
    )
    extra_tokens_per_query = min(extra_tokens_per_query, max_remote_tokens)
    if extra_tokens_per_query < 0:
        raise ValueError("extra_tokens_per_query must be non-negative")

    query_unit = F.normalize(query_carrier.float(), dim=-1)
    key_unit = F.normalize(key_carrier.float(), dim=-1)
    carrier_logits = torch.einsum(
        "bhqd,bhkd->bhqk", query_unit, key_unit
    ) / temperature
    carrier_relevance = carrier_logits.softmax(dim=-1).mean(dim=1)

    raw_cell_detail = (
        (residual_value.float() - coarse_value.float())
        .square()
        .mean(dim=(1, 4))
    )
    frame_detail = raw_cell_detail.mean(dim=-1, keepdim=True)
    spatial_detail = (
        raw_cell_detail / frame_detail.clamp_min(1e-8)
    ).pow(spatial_detail_power)
    spatial_detail = spatial_detail / spatial_detail.mean(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    relative_frame_detail = frame_detail / frame_detail.mean(
        dim=-2, keepdim=True
    ).clamp_min(1e-8)
    cell_detail = spatial_detail * relative_frame_detail.pow(
        frame_detail_power
    )
    carrier_need = carrier_relevance[..., None].expand(
        batch, num_frames, num_frames, cells_per_frame
    )
    residual_need = cell_detail[:, None, :, :].expand(
        batch, num_frames, num_frames, cells_per_frame
    )
    expected_importance_shape = (
        batch,
        num_frames,
        num_frames,
        cells_per_frame,
    )
    importance_scale = None
    if cell_scorer == "patch_qk" or cell_importance_weight > 0.0:
        if cell_importance is None:
            raise ValueError(
                "patch-QK scoring or positive importance weight requires "
                "cell_importance"
            )
        if tuple(cell_importance.shape) != expected_importance_shape:
            raise ValueError(
                "cell_importance must have shape "
                "[batch, query_frames, key_frames, cells]"
            )
        importance_scale = cell_importance.float()
        importance_scale = importance_scale / importance_scale.mean(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)

    if cell_scorer == "uniform":
        current_need = torch.ones_like(carrier_need)
    elif cell_scorer == "residual":
        current_need = residual_need
    elif cell_scorer == "carrier":
        current_need = carrier_need
    elif cell_scorer == "patch_qk":
        current_need = importance_scale
    else:
        carrier_power = 2.0 * carrier_residual_balance
        residual_power = 2.0 * (1.0 - carrier_residual_balance)
        current_need = carrier_need.pow(carrier_power) * residual_need.pow(
            residual_power
        )
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)

    frame_ids = torch.arange(num_frames, device=current_need.device)
    local_mask = (
        frame_ids[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    importance_gate = torch.ones(
        batch, num_frames, device=current_need.device, dtype=current_need.dtype
    )
    effective_importance_weight = torch.full_like(
        importance_gate, cell_importance_weight
    )
    base_need_concentration = torch.zeros_like(importance_gate)
    if cell_importance_weight > 0.0:
        if cell_importance_gate_threshold > 0.0 and extra_tokens_per_query > 0:
            base_score = current_need.masked_fill(
                local_mask[None, :, :, None], float("-inf")
            ).flatten(start_dim=-2)
            concentration_k = min(extra_tokens_per_query, base_score.shape[-1])
            base_need_concentration = base_score.topk(
                concentration_k, dim=-1
            ).values.mean(dim=-1)
            layer_concentration = base_need_concentration.mean(
                dim=-1, keepdim=True
            )
            layer_gate = torch.sigmoid(
                (
                    cell_importance_gate_threshold
                    - layer_concentration
                )
                / cell_importance_gate_temperature
            )
            importance_gate = layer_gate.expand_as(base_need_concentration)
            effective_importance_weight = cell_importance_floor + (
                cell_importance_weight - cell_importance_floor
            ) * importance_gate
        current_need = current_need * importance_scale.clamp_min(1e-4).pow(
            effective_importance_weight[..., None, None]
        )
    selected_indices, stats = schedule_residual_debt_cells(
        current_need,
        local_frame_radius=local_frame_radius,
        extra_tokens_per_query=extra_tokens_per_query,
        layer_idx=layer_idx,
        routing_state=routing_state,
        momentum=momentum,
        repayment=repayment,
        repayment_mode=repayment_mode,
        service_credit_scale=service_credit_scale,
        effective_service_credit=(carrier_need * residual_need),
        collect_stats=collect_stats,
    )
    stats.update({
        "residual_budget_surface_weight": 0.0,
        "residual_budget_frame_detail_power": float(frame_detail_power),
        "residual_budget_spatial_detail_power": float(
            spatial_detail_power
        ),
        "residual_budget_cell_scorer_id": float(scorer_ids[cell_scorer]),
        "residual_budget_carrier_residual_balance": float(
            carrier_residual_balance
        ),
        "residual_budget_cell_importance_weight": float(
            cell_importance_weight
        ),
        "residual_budget_cell_importance_floor": float(
            cell_importance_floor
        ),
        "residual_budget_cell_importance_gate_threshold": float(
            cell_importance_gate_threshold
        ),
    })
    if collect_stats:
        stats.update({
            "residual_budget_frame_detail": float(
                cell_detail.mean().item()
            ),
            "residual_budget_cell_importance_gate": float(
                importance_gate.mean().item()
            ),
            "residual_budget_cell_importance_effective_weight": float(
                effective_importance_weight.mean().item()
            ),
            "residual_budget_base_need_concentration": float(
                base_need_concentration.mean().item()
            ),
        })
    return selected_indices, stats


def select_grouped_residual_debt_cells(
    query_carrier: torch.Tensor,
    parent_key: torch.Tensor,
    parent_head_valid: torch.Tensor,
    coarse_value: torch.Tensor,
    residual_value: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    *,
    local_frame_radius: int,
    child_budget_per_query: int,
    layer_idx: int,
    routing_state: dict | None,
    momentum: float,
    service_conditioned_momentum: bool,
    repayment: float,
    repayment_mode: str,
    service_credit_scale: float,
    temperature: float,
    frame_detail_power: float,
    spatial_detail_power: float,
    cell_scorer: str,
    carrier_residual_balance: float,
    unified_incremental_service: bool,
    bounded_wait_reobservation: bool,
    adaptive_parent_service: bool,
    mixed_parent_execution: bool,
    additive_parent_service: bool,
    additive_full_upgrade_fraction: float,
    additive_parent_substitution_fraction: float,
    routing_phase_mode: str,
    mixed_extra_budget_per_query: int,
    mixed_residuals_per_parent: int,
    parent_hard_fraction: float,
    parent_cost_power: float,
    frame_balance_fraction: float,
    target_sparsity: float,
    total_layers: int,
    collect_stats: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    """Score 4x4 parent cells and allocate their 2x2 child refinements."""
    batch, heads, num_frames, head_dim = query_carrier.shape
    if parent_key.shape[:3] != (batch, heads, num_frames):
        raise ValueError("parent key shape must match query carriers")
    parents_per_frame = parent_key.shape[-2]
    if parent_head_valid.shape != (heads, parents_per_frame):
        raise ValueError("parent validity mask must match heads and parents")
    if cell_scorer not in {
        "uniform",
        "residual",
        "carrier",
        "carrier_residual",
        "patch_qk",
        "scout_service",
    }:
        raise ValueError("unsupported grouped-parent scorer")
    if additive_parent_service and mixed_parent_execution:
        raise ValueError(
            "additive and mixed parent execution are mutually exclusive"
        )
    if unified_incremental_service and adaptive_parent_service:
        raise ValueError(
            "unified incremental service replaces adaptive parent buckets"
        )
    if unified_incremental_service and mixed_parent_execution:
        raise ValueError(
            "unified incremental service replaces mixed parent execution"
        )
    if unified_incremental_service and frame_balance_fraction != 0.0:
        raise ValueError(
            "unified incremental service owns frame coverage allocation"
        )
    if unified_incremental_service and additive_full_upgrade_fraction != 0.0:
        raise ValueError(
            "unified incremental service does not use fixed full upgrades"
        )
    if (
        unified_incremental_service
        and additive_parent_substitution_fraction != 0.0
    ):
        raise ValueError(
            "unified incremental service does not use fixed substitutions"
        )
    if not 0.0 <= additive_full_upgrade_fraction <= 0.25:
        raise ValueError(
            "additive full-upgrade fraction must be in [0, 0.25]"
        )
    if not 0.0 <= additive_parent_substitution_fraction <= 1.0:
        raise ValueError(
            "additive parent substitution fraction must be in [0, 1]"
        )
    if (
        additive_parent_substitution_fraction > 0.0
        and not additive_parent_service
    ):
        raise ValueError(
            "parent substitution requires additive parent service"
        )
    if routing_phase_mode not in {
        "rotating",
        "fixed",
        "anchored",
        "anchored_heads",
    }:
        raise ValueError("unsupported grouped-parent phase mode")
    if not 0.0 <= parent_hard_fraction <= 1.0:
        raise ValueError("parent hard fraction must be in [0, 1]")
    if parent_cost_power < 0.0:
        raise ValueError("parent cost power must be non-negative")
    if not 0.0 <= frame_balance_fraction <= 1.0:
        raise ValueError("frame balance fraction must be in [0, 1]")

    query_unit = F.normalize(query_carrier.float(), dim=-1)
    parent_key_unit = F.normalize(parent_key.float(), dim=-1)
    head_logits = torch.einsum(
        "bhqd,bhkpd->bhqkp", query_unit, parent_key_unit
    ) / temperature
    valid = parent_head_valid.to(head_logits.dtype).view(
        1, heads, 1, 1, parents_per_frame
    )
    parent_logits = (head_logits * valid).sum(dim=1) / valid.sum(
        dim=1
    ).clamp_min(1.0)
    parent_relevance = parent_logits.flatten(start_dim=-2).softmax(
        dim=-1
    ).view_as(parent_logits)
    if collect_stats:
        flat_parent_relevance = parent_relevance.flatten(start_dim=-2)
        parent_relevance_entropy = (
            -(
                flat_parent_relevance
                * flat_parent_relevance.clamp_min(1e-12).log()
            ).sum(dim=-1)
            / math.log(max(flat_parent_relevance.shape[-1], 2))
        ).mean()
    else:
        parent_relevance_entropy = None

    if cell_scorer == "scout_service":
        raw_child_detail = torch.ones(
            batch,
            num_frames,
            coarse_value.shape[-2],
            device=coarse_value.device,
            dtype=torch.float32,
        )
        raw_parent_detail = torch.ones(
            batch,
            num_frames,
            parents_per_frame,
            device=coarse_value.device,
            dtype=torch.float32,
        )
        parent_detail = raw_parent_detail
    else:
        raw_child_detail = (
            (residual_value.float() - coarse_value.float())
            .square()
            .mean(dim=(1, 4))
        )
        raw_parent_detail = aggregate_child_cells_to_parents(
            raw_child_detail, parent_to_children, parent_child_valid
        )
        frame_detail = raw_parent_detail.mean(dim=-1, keepdim=True)
        spatial_detail = (
            raw_parent_detail / frame_detail.clamp_min(1e-8)
        ).pow(spatial_detail_power)
        spatial_detail = spatial_detail / spatial_detail.mean(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        relative_frame_detail = frame_detail / frame_detail.mean(
            dim=-2, keepdim=True
        ).clamp_min(1e-8)
        parent_detail = spatial_detail * relative_frame_detail.pow(
            frame_detail_power
        )
    residual_need = parent_detail[:, None].expand(
        batch, num_frames, num_frames, parents_per_frame
    )
    carrier_need = parent_relevance
    if cell_scorer == "uniform":
        current_need = torch.ones_like(carrier_need)
    elif cell_scorer == "residual":
        current_need = residual_need
    elif cell_scorer in {"carrier", "patch_qk", "scout_service"}:
        current_need = carrier_need
    else:
        carrier_power = 2.0 * carrier_residual_balance
        residual_power = 2.0 * (1.0 - carrier_residual_balance)
        current_need = carrier_need.pow(carrier_power) * residual_need.pow(
            residual_power
        )
    current_need = current_need / current_need.mean(
        dim=(-2, -1), keepdim=True
    ).clamp_min(1e-8)

    child_counts = parent_child_valid.sum(dim=-1)
    service_costs = None
    candidate_count_override = None
    incremental_child_need = None
    ordered_children = None
    parent_heterogeneity = torch.zeros_like(raw_parent_detail)
    hard_parent_fraction = 0.0
    if additive_parent_service:
        base_service_capacities = (child_counts - 1).clamp_min(0)
        if additive_full_upgrade_fraction > 0.0:
            service_capacities = child_counts
            base_costs = base_service_capacities.clamp_min(1)
            service_costs = base_costs.view(
                1, 1, 1, parents_per_frame
            ).expand(batch, num_frames, num_frames, -1)
            base_budget = round(
                child_budget_per_query
                * (1.0 - additive_full_upgrade_fraction)
            )
            candidate_count_override = (
                math.ceil(
                    base_budget / max(int(base_costs.max().item()), 1)
                )
                if base_budget > 0 else 0
            )
        else:
            service_capacities = base_service_capacities
        child_order = ordered_parent_child_services(
            parent_to_children,
            parent_child_valid,
            layer_idx=layer_idx,
            phase_mode=routing_phase_mode,
        )
        ordered_children = child_order.view(
            1, 1, parents_per_frame, parent_to_children.shape[-1]
        ).expand(batch, num_frames, -1, -1)
    else:
        service_capacities = child_counts
    if unified_incremental_service:
        child_detail = raw_child_detail[..., parent_to_children]
        valid_children = parent_child_valid.view(
            1, 1, parents_per_frame, -1
        )
        ranked_child_slots = child_detail.masked_fill(
            ~valid_children, float("-inf")
        ).argsort(dim=-1, descending=True)
        child_map = parent_to_children.view(
            1, 1, parents_per_frame, -1
        ).expand(batch, num_frames, -1, -1)
        ordered_children = child_map.gather(-1, ranked_child_slots)
        ranked_child_need = child_detail.gather(-1, ranked_child_slots)
        incremental_child_need = ranked_child_need[:, None].expand(
            batch,
            num_frames,
            num_frames,
            parents_per_frame,
            ranked_child_need.shape[-1],
        )
    if adaptive_parent_service:
        child_detail = raw_child_detail[..., parent_to_children]
        valid_children = parent_child_valid.view(
            1, 1, parents_per_frame, -1
        )
        valid_weight = valid_children.to(child_detail.dtype)
        child_mean = (
            (child_detail * valid_weight).sum(dim=-1)
            / child_counts.to(child_detail.dtype).view(1, 1, -1)
        )
        child_variance = (
            (
                (child_detail - child_mean[..., None]).square()
                * valid_weight
            ).sum(dim=-1)
            / child_counts.to(child_detail.dtype).view(1, 1, -1)
        )
        parent_heterogeneity = child_variance / child_mean.square().clamp_min(
            1e-8
        )
        hard_count = min(
            parents_per_frame,
            max(0, round(parents_per_frame * parent_hard_fraction)),
        )
        hard_parent = torch.zeros_like(parent_heterogeneity, dtype=torch.bool)
        if hard_count > 0:
            hard_indices = parent_heterogeneity.topk(
                hard_count, dim=-1
            ).indices
            hard_parent.scatter_(-1, hard_indices, True)
        per_key_cost = torch.where(
            hard_parent,
            child_counts.view(1, 1, -1),
            torch.ones_like(child_counts).view(1, 1, -1),
        )
        service_costs = per_key_cost[:, None].expand(
            batch, num_frames, num_frames, parents_per_frame
        )
        ranked_child_slots = child_detail.masked_fill(
            ~valid_children, float("-inf")
        ).argsort(dim=-1, descending=True)
        child_map = parent_to_children.view(
            1, 1, parents_per_frame, -1
        ).expand(batch, num_frames, -1, -1)
        ordered_children = child_map.gather(-1, ranked_child_slots)
        hard_parent_fraction = (
            float(hard_parent.float().mean().item()) if collect_stats else 0.0
        )
    if mixed_parent_execution:
        parent_indices, served_counts, stats = (
            schedule_mixed_parent_debt_cells(
                current_need,
                local_frame_radius=local_frame_radius,
                extra_budget_per_query=mixed_extra_budget_per_query,
                children_per_parent=parent_to_children.shape[-1],
                residuals_per_parent=mixed_residuals_per_parent,
                layer_idx=layer_idx,
                routing_state=routing_state,
                momentum=momentum,
                repayment=repayment,
                repayment_mode=repayment_mode,
                service_credit_scale=service_credit_scale,
                effective_service_credit=carrier_need * residual_need,
                cost_power=parent_cost_power,
                collect_stats=collect_stats,
            )
        )
    else:
        parent_indices, served_counts, stats = (
            schedule_grouped_residual_debt_cells(
                current_need,
                service_capacities,
                local_frame_radius=local_frame_radius,
                child_budget_per_query=child_budget_per_query,
                layer_idx=layer_idx,
                routing_state=routing_state,
                momentum=momentum,
                service_conditioned_momentum=service_conditioned_momentum,
                bounded_wait_reobservation=bounded_wait_reobservation,
                repayment=repayment,
                repayment_mode=repayment_mode,
                service_credit_scale=service_credit_scale,
                effective_service_credit=(
                    carrier_need
                    if cell_scorer == "scout_service"
                    else carrier_need * residual_need
                ),
                service_costs=service_costs,
                candidate_count_override=candidate_count_override,
                cost_power=parent_cost_power,
                frame_balance_fraction=frame_balance_fraction,
                incremental_child_need=incremental_child_need,
                target_sparsity=target_sparsity,
                total_layers=total_layers,
                parent_carrier_execution=additive_parent_service,
                collect_stats=collect_stats,
            )
        )
    if additive_parent_substitution_fraction > 0.0:
        (
            substituted_parent_indices,
            substitution_child_indices,
        ) = select_additive_parent_substitutions(
            parent_indices,
            served_counts,
            child_counts,
            ordered_children,
            parents_per_frame=parents_per_frame,
            children_per_frame=coarse_value.shape[-2],
            substitution_fraction=(
                additive_parent_substitution_fraction
            ),
        )
    else:
        substituted_parent_indices = parent_indices[..., :0]
        substitution_child_indices = parent_indices[..., :0]
    stats.update({
        "residual_budget_grouped_parent_relevance": (
            float(parent_relevance.max(dim=-1).values.mean().item())
            if collect_stats else 0.0
        ),
        "residual_budget_parent_relevance_entropy": (
            float(parent_relevance_entropy.item())
            if parent_relevance_entropy is not None else 0.0
        ),
        "residual_budget_grouped_parent_detail": (
            float(parent_detail.mean().item()) if collect_stats else 0.0
        ),
        "residual_budget_carrier_residual_balance": float(
            carrier_residual_balance
        ),
        "residual_budget_adaptive_parent_service": float(
            adaptive_parent_service
        ),
        "residual_budget_unified_incremental_service": float(
            unified_incremental_service
        ),
        "residual_budget_additive_parent_service": float(
            additive_parent_service
        ),
        "residual_budget_additive_full_upgrade_fraction": float(
            additive_full_upgrade_fraction
        ),
        "residual_budget_additive_parent_substitution_fraction": float(
            additive_parent_substitution_fraction
        ),
        "residual_budget_additive_parent_substitutions_per_query": float(
            substituted_parent_indices.shape[-1]
        ),
        "residual_budget_fixed_anchor_head_fraction": float(
            routing_phase_mode in {"anchored", "anchored_heads"}
        ) * min(
            1.0,
            max(1, parent_to_children.shape[-1]) / max(heads, 1),
        ),
        "residual_budget_fixed_child_anchor": float(
            routing_phase_mode == "anchored" and additive_parent_service
        ),
        "residual_budget_cell_scorer_id": float({
            "uniform": 0,
            "residual": 1,
            "carrier": 2,
            "carrier_residual": 3,
            "patch_qk": 4,
            "scout_service": 5,
        }[cell_scorer]),
        "residual_budget_parent_hard_fraction": hard_parent_fraction,
        "residual_budget_parent_heterogeneity": (
            float(parent_heterogeneity.mean().item())
            if collect_stats and adaptive_parent_service else 0.0
        ),
    })
    observation_snapshot = stats.get("_service_observation_snapshot")
    if isinstance(observation_snapshot, dict):
        observation_child_detail = (
            (residual_value.float() - coarse_value.float())
            .square()
            .mean(dim=(1, 4))
        )
        observation_parent_detail = aggregate_child_cells_to_parents(
            observation_child_detail,
            parent_to_children,
            parent_child_valid,
        )
        observation_snapshot.update({
            "parent_relevance": parent_relevance.detach(),
            "parent_detail": observation_parent_detail.detach(),
            "raw_child_detail": observation_child_detail.detach(),
            "parent_to_children": parent_to_children.detach(),
            "parent_child_valid": parent_child_valid.detach(),
            "query_carrier": query_carrier.detach(),
            "parent_key": parent_key.detach(),
            "parent_head_valid": parent_head_valid.detach(),
            "temperature": float(temperature),
        })
    return (
        parent_indices,
        served_counts,
        ordered_children,
        substituted_parent_indices,
        substitution_child_indices,
        stats,
    )


def gather_frame_tokens(
    frame_tokens: torch.Tensor,
    frame_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather per-batch frame-token rows for one or more query groups."""
    batch, heads, num_frames, tokens_per_frame, head_dim = frame_tokens.shape
    if frame_indices.ndim != 3 or frame_indices.shape[0] != batch:
        raise ValueError("frame_indices must have shape [batch, groups, frames]")
    use_triton = os.environ.get(
        "SPARSE_VGGT_TRITON_GATHER", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if use_triton and frame_tokens.is_cuda and not torch.is_grad_enabled():
        from sparse_vggt.kernels.gather import triton_gather_frame_tokens

        return triton_gather_frame_tokens(frame_tokens, frame_indices)
    group_size, selected_frames = frame_indices.shape[1:]
    gather_index = frame_indices[:, None, :, :, None, None].expand(
        batch,
        heads,
        group_size,
        selected_frames,
        tokens_per_frame,
        head_dim,
    )
    expanded_tokens = frame_tokens.unsqueeze(2).expand(
        batch,
        heads,
        group_size,
        num_frames,
        tokens_per_frame,
        head_dim,
    )
    return expanded_tokens.gather(3, gather_index)


def gather_flat_frame_tokens(
    frame_tokens: torch.Tensor,
    flat_token_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather arbitrary spatial cells for each batch and query group."""
    batch, heads, num_frames, tokens_per_frame, head_dim = frame_tokens.shape
    if flat_token_indices.ndim != 3 or flat_token_indices.shape[0] != batch:
        raise ValueError(
            "flat_token_indices must have shape [batch, groups, tokens]"
        )
    use_triton = os.environ.get(
        "SPARSE_VGGT_TRITON_GATHER", "0"
    ).lower() in {"1", "true", "yes", "on"}
    if use_triton and frame_tokens.is_cuda and not torch.is_grad_enabled():
        from sparse_vggt.kernels.gather import (
            triton_gather_flat_frame_tokens,
        )

        return triton_gather_flat_frame_tokens(
            frame_tokens, flat_token_indices
        )
    group_size, selected_tokens = flat_token_indices.shape[1:]
    flattened_tokens = frame_tokens.reshape(
        batch, heads, num_frames * tokens_per_frame, head_dim
    )
    gather_index = flat_token_indices[:, None, :, :, None].expand(
        batch,
        heads,
        group_size,
        selected_tokens,
        head_dim,
    )
    expanded_tokens = flattened_tokens.unsqueeze(2).expand(
        batch,
        heads,
        group_size,
        num_frames * tokens_per_frame,
        head_dim,
    )
    return expanded_tokens.gather(3, gather_index)


def grouped_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run grouped attention through the standard four-dimensional SDPA API."""
    if query.ndim != 5 or key.ndim != 5 or value.ndim != 5:
        raise ValueError("grouped query, key, and value must be five-dimensional")
    if key.shape != value.shape:
        raise ValueError("grouped key and value must have identical shapes")
    if query.shape[:3] != key.shape[:3]:
        raise ValueError("grouped query and key batch dimensions must match")
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("grouped query and key head dimensions must match")

    batch, heads, groups, query_tokens, head_dim = query.shape
    key_tokens = key.shape[-2]
    flat_query = query.reshape(
        batch * heads, groups, query_tokens, head_dim
    )
    flat_key = key.reshape(batch * heads, groups, key_tokens, head_dim)
    flat_value = value.reshape_as(flat_key)
    flat_attention_bias = None
    if attention_bias is not None:
        if attention_bias.ndim != 5:
            raise ValueError("grouped attention bias must be five-dimensional")
        if attention_bias.shape[0] != batch:
            raise ValueError("grouped attention bias batch dimension must match")
        if attention_bias.shape[1] not in {1, heads}:
            raise ValueError("grouped attention bias head dimension must broadcast")
        if attention_bias.shape[2] != groups:
            raise ValueError("grouped attention bias group dimension must match")
        if attention_bias.shape[-2] not in {1, query_tokens}:
            raise ValueError("grouped attention bias query dimension must broadcast")
        if attention_bias.shape[-1] != key_tokens:
            raise ValueError("grouped attention bias key dimension must match")
        expanded_bias = attention_bias.expand(
            batch,
            heads,
            groups,
            attention_bias.shape[-2],
            key_tokens,
        )
        flat_attention_bias = expanded_bias.reshape(
            batch * heads,
            groups,
            attention_bias.shape[-2],
            key_tokens,
        )
    flat_output = F.scaled_dot_product_attention(
        flat_query,
        flat_key,
        flat_value,
        attn_mask=flat_attention_bias,
    )
    return flat_output.reshape(
        batch, heads, groups, query_tokens, head_dim
    )


def refinement_multiplicity_log_bias(
    coarse_cell_indices: torch.Tensor,
    refinement_cell_indices: list[torch.Tensor],
    *,
    total_cells: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Split one carrier's softmax mass across its refinement tokens."""
    if coarse_cell_indices.ndim != 3:
        raise ValueError("coarse cell indices must be three-dimensional")
    if total_cells < 1:
        raise ValueError("total_cells must be positive")
    if not refinement_cell_indices:
        return (
            torch.zeros(
                coarse_cell_indices.shape,
                device=coarse_cell_indices.device,
                dtype=dtype,
            ),
            [],
        )

    batch, groups, _ = coarse_cell_indices.shape
    counts = torch.zeros(
        batch,
        groups,
        total_cells,
        device=coarse_cell_indices.device,
        dtype=torch.float32,
    )
    for indices in refinement_cell_indices:
        if indices.ndim != 3 or indices.shape[:2] != (batch, groups):
            raise ValueError(
                "refinement cell indices must match coarse batch and groups"
            )
        counts.scatter_add_(
            -1,
            indices,
            torch.ones(
                indices.shape,
                device=indices.device,
                dtype=counts.dtype,
            ),
        )

    all_bias = -torch.log1p(counts)
    coarse_bias = all_bias.gather(-1, coarse_cell_indices).to(dtype=dtype)
    refinement_bias = [
        all_bias.gather(-1, indices).to(dtype=dtype)
        for indices in refinement_cell_indices
    ]
    return coarse_bias, refinement_bias


def encode_multiphase_residual_indices(
    phase_cell_indices: list[torch.Tensor],
    *,
    coarse_cells_per_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Encode phase-major residual sources and retain their coarse cell ids."""
    if not phase_cell_indices:
        raise ValueError("at least one residual phase is required")
    reference_shape = phase_cell_indices[0].shape[:-1]
    if any(
        indices.shape[:-1] != reference_shape
        for indices in phase_cell_indices
    ):
        raise ValueError("residual phase index prefixes must match")
    if coarse_cells_per_frame < 1:
        raise ValueError("coarse_cells_per_frame must be positive")

    residual_cells_per_frame = (
        coarse_cells_per_frame * len(phase_cell_indices)
    )
    encoded_indices = []
    base_indices = []
    for phase_idx, indices in enumerate(phase_cell_indices):
        source_frames = torch.div(
            indices, coarse_cells_per_frame, rounding_mode="floor"
        )
        source_cells = indices.remainder(coarse_cells_per_frame)
        encoded_indices.append(
            source_frames * residual_cells_per_frame
            + phase_idx * coarse_cells_per_frame
            + source_cells
        )
        base_indices.append(indices)
    return (
        torch.cat(encoded_indices, dim=-1),
        torch.cat(base_indices, dim=-1),
        residual_cells_per_frame,
    )


def mixed_parent_execution_kv(
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    parent_head_valid: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    hard_parent_indices: torch.Tensor,
    residual_cell_indices: torch.Tensor,
    parent_to_children: torch.Tensor,
    centers: torch.Tensor,
    *,
    bias_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack easy parent carriers and promoted 2x2 child carriers."""
    batch, heads, num_frames, parents_per_frame, head_dim = parent_key.shape
    group_size = centers.numel()
    hard_indices = hard_parent_indices[:, centers, :]
    hard_count = hard_indices.shape[-1]
    all_parent_indices = torch.arange(
        num_frames * parents_per_frame, device=parent_key.device
    ).view(1, 1, -1).expand(batch, group_size, -1)
    parent_frames = torch.div(
        all_parent_indices, parents_per_frame, rounding_mode="floor"
    )
    keep_parent = parent_frames != centers.view(1, -1, 1)
    if hard_count > 0:
        keep_parent.scatter_(2, hard_indices, False)
    easy_parent_indices = all_parent_indices[keep_parent].view(
        batch, group_size, -1
    )
    easy_key = gather_flat_frame_tokens(parent_key, easy_parent_indices)
    easy_value = gather_flat_frame_tokens(parent_value, easy_parent_indices)

    del parent_head_valid
    easy_bias = torch.full(
        easy_parent_indices.shape,
        math.log(parent_to_children.shape[-1]),
        device=parent_key.device,
        dtype=bias_dtype,
    )

    hard_parent_ids = hard_indices.remainder(parents_per_frame)
    hard_frames = torch.div(
        hard_indices, parents_per_frame, rounding_mode="floor"
    )
    hard_child_ids = parent_to_children[hard_parent_ids]
    hard_child_indices = (
        hard_frames[..., None] * child_key.shape[-2] + hard_child_ids
    ).flatten(start_dim=-2)
    hard_key = gather_flat_frame_tokens(child_key, hard_child_indices)
    hard_value = gather_flat_frame_tokens(child_value, hard_child_indices)
    hard_bias, residual_bias = refinement_multiplicity_log_bias(
        hard_child_indices,
        [residual_cell_indices[:, centers, :]],
        total_cells=num_frames * child_key.shape[-2],
        dtype=bias_dtype,
    )
    remote_key = torch.cat((easy_key, hard_key), dim=-2)
    remote_value = torch.cat((easy_value, hard_value), dim=-2)
    remote_bias = torch.cat((easy_bias, hard_bias), dim=-1)
    return (
        remote_key,
        remote_value,
        remote_bias,
        residual_bias[0],
        remote_key.shape[-2],
    )


def additive_parent_service_descriptors(
    parent_key: torch.Tensor,
    child_key: torch.Tensor,
    selected_child_indices: torch.Tensor,
    substituted_parent_indices: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    centers: torch.Tensor,
    *,
    child_masses: torch.Tensor,
    bias_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe mass-conserving parent scouts and selected child services.

    A parent carrier keeps the mass of its unserved children. Each selected
    child receives one child cell's mass, so the total mass of every parent
    remains unchanged.
    """
    batch, _, num_frames, parents_per_frame, _ = parent_key.shape
    group_size = centers.numel()
    children_per_frame = child_key.shape[-2]
    if child_masses.shape != (children_per_frame,):
        raise ValueError("child masses must have one entry per child cell")
    child_masses = child_masses.to(
        device=parent_key.device, dtype=torch.float32
    )

    if substituted_parent_indices.shape[:2] != (batch, num_frames):
        raise ValueError(
            "substituted parent indices must match batch and query frames"
        )
    if _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False):
        base_remote_parents = remote_parent_index_grid(
            num_frames,
            parents_per_frame,
            device=parent_key.device,
        ).index_select(0, centers)
        remote_parent_indices = base_remote_parents.unsqueeze(0).expand(
            batch, -1, -1
        )
        if substituted_parent_indices.shape[-1] > 0:
            substitutions = substituted_parent_indices[:, centers, :]
            substitution_frames = torch.div(
                substitutions,
                parents_per_frame,
                rounding_mode="floor",
            )
            substitution_ids = substitutions.remainder(parents_per_frame)
            remote_positions = (
                substitution_frames
                - (
                    substitution_frames > centers.view(1, -1, 1)
                ).to(substitution_frames.dtype)
            ) * parents_per_frame + substitution_ids
            keep_remote = torch.ones(
                remote_parent_indices.shape,
                device=parent_key.device,
                dtype=torch.bool,
            )
            keep_remote.scatter_(2, remote_positions, False)
            remote_parent_indices = remote_parent_indices[
                keep_remote
            ].view(batch, group_size, -1)
    else:
        all_parent_indices = torch.arange(
            num_frames * parents_per_frame, device=parent_key.device
        ).view(1, 1, -1).expand(batch, group_size, -1)
        parent_frames = torch.div(
            all_parent_indices, parents_per_frame, rounding_mode="floor"
        )
        remote_mask = parent_frames != centers.view(1, -1, 1)
        if substituted_parent_indices.shape[-1] > 0:
            substitutions = substituted_parent_indices[:, centers, :]
            remote_mask.scatter_(2, substitutions, False)
        remote_parent_indices = all_parent_indices[remote_mask].view(
            batch, group_size, -1
        )

    child_to_parent = spatial_child_parent_map(
        parent_to_children,
        parent_child_valid,
        children_per_frame=children_per_frame,
    )

    selected_children = selected_child_indices[:, centers, :]
    selected_frames = torch.div(
        selected_children, children_per_frame, rounding_mode="floor"
    )
    selected_child_ids = selected_children.remainder(children_per_frame)
    selected_parent_indices = (
        selected_frames * parents_per_frame
        + child_to_parent[selected_child_ids]
    )
    service_mass = torch.zeros(
        batch,
        group_size,
        num_frames * parents_per_frame,
        device=parent_key.device,
        dtype=torch.float32,
    )
    selected_child_mass = child_masses[selected_child_ids]
    service_mass.scatter_add_(
        -1,
        selected_parent_indices,
        selected_child_mass,
    )
    remote_service_mass = service_mass.gather(
        -1, remote_parent_indices
    )
    all_parent_mass = spatial_all_parent_mass(
        child_masses,
        parent_to_children,
        parent_child_valid,
        num_frames=num_frames,
    )
    remaining_mass = (
        all_parent_mass[remote_parent_indices] - remote_service_mass
    )
    if bool((remaining_mass < 0).any()):
        raise RuntimeError(
            "additive service cannot exceed a parent's child mass"
        )
    parent_bias = torch.where(
        remaining_mass > 0,
        torch.log(remaining_mass.clamp_min(1e-8)),
        torch.full_like(remaining_mass, float("-inf")),
    ).to(dtype=bias_dtype)
    child_bias = torch.log(selected_child_mass).to(dtype=bias_dtype)
    return (
        remote_parent_indices,
        parent_bias,
        selected_children,
        child_bias,
    )


def additive_parent_service_kv(
    parent_key: torch.Tensor,
    parent_value: torch.Tensor,
    child_key: torch.Tensor,
    child_value: torch.Tensor,
    selected_child_indices: torch.Tensor,
    substituted_parent_indices: torch.Tensor,
    parent_to_children: torch.Tensor,
    parent_child_valid: torch.Tensor,
    centers: torch.Tensor,
    *,
    child_masses: torch.Tensor,
    bias_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Pack additive service descriptors for the generic SDPA fallback."""
    (
        remote_parent_indices,
        parent_bias,
        selected_children,
        child_bias,
    ) = additive_parent_service_descriptors(
        parent_key,
        child_key,
        selected_child_indices,
        substituted_parent_indices,
        parent_to_children,
        parent_child_valid,
        centers,
        child_masses=child_masses,
        bias_dtype=bias_dtype,
    )
    remote_parent_key = gather_flat_frame_tokens(
        parent_key, remote_parent_indices
    )
    remote_parent_value = gather_flat_frame_tokens(
        parent_value, remote_parent_indices
    )
    selected_child_key = gather_flat_frame_tokens(
        child_key, selected_children
    )
    selected_child_value = gather_flat_frame_tokens(
        child_value, selected_children
    )
    packed_key = torch.cat(
        (remote_parent_key, selected_child_key), dim=-2
    )
    packed_value = torch.cat(
        (remote_parent_value, selected_child_value), dim=-2
    )
    packed_bias = torch.cat((parent_bias, child_bias), dim=-1)
    return (
        packed_key,
        packed_value,
        packed_bias,
        remote_parent_key.shape[-2],
        selected_child_key.shape[-2],
    )


def residual_budget_patch_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    target_sparsity: float,
    local_frame_radius: int,
    remote_pool_size: int,
    routing_parent_size: int = 2,
    routing_phase_mode: str = "rotating",
    unified_incremental_service: bool = False,
    bounded_wait_reobservation: bool = False,
    adaptive_parent_service: bool = False,
    mixed_parent_execution: bool = False,
    additive_parent_service: bool = False,
    additive_full_upgrade_fraction: float = 0.0,
    additive_parent_substitution_fraction: float = 0.0,
    mixed_residuals_per_parent: int = 4,
    parent_hard_fraction: float = 0.25,
    parent_cost_power: float = 1.0,
    frame_balance_fraction: float = 0.0,
    query_frame_chunk: int,
    layer_idx: int,
    momentum: float,
    repayment: float,
    repayment_mode: str = "reset",
    service_credit_scale: float = 1.0,
    temperature: float,
    exact_budget_fraction: float,
    routing_state: dict | None,
    service_conditioned_momentum: bool = False,
    surface_weight: float = 0.0,
    selection_granularity: str = "frame",
    frame_detail_power: float = 1.0,
    spatial_detail_power: float = 1.0,
    cell_scorer: str = "carrier_residual",
    carrier_residual_balance: float = 0.5,
    cell_refinement_phases: int = 1,
    cell_precision_fraction: float = 0.0,
    mass_conserving_refinement: bool = False,
    exact_mass_conserving_refinement: bool = False,
    cell_importance_weight: float = 0.0,
    cell_importance_floor: float = 0.0,
    cell_importance_gate_threshold: float = 0.0,
    cell_importance_gate_temperature: float = 0.05,
    total_layers: int = 24,
    special_query: torch.Tensor | None = None,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep coarse frame connectivity and repay selected pairwise detail debt."""
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be in [0, 1)")
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if remote_pool_size < 2:
        raise ValueError("residual budget routing requires remote_pool_size >= 2")
    if routing_parent_size < remote_pool_size:
        raise ValueError("routing_parent_size must be at least remote_pool_size")
    if routing_parent_size % remote_pool_size != 0:
        raise ValueError("routing_parent_size must divide into execution cells")
    if unified_incremental_service and routing_parent_size == remote_pool_size:
        raise ValueError(
            "unified incremental service requires grouped parent routing"
        )
    if bounded_wait_reobservation and not unified_incremental_service:
        raise ValueError(
            "bounded-wait reobservation requires unified incremental service"
        )
    if routing_phase_mode not in {
        "rotating",
        "fixed",
        "anchored",
        "anchored_heads",
    }:
        raise ValueError(
            "routing_phase_mode must be rotating, fixed, anchored, "
            "or anchored_heads"
        )
    if not 0.0 <= parent_hard_fraction <= 1.0:
        raise ValueError("parent_hard_fraction must be in [0, 1]")
    if parent_cost_power < 0.0:
        raise ValueError("parent_cost_power must be non-negative")
    if not 0.0 <= frame_balance_fraction <= 1.0:
        raise ValueError("frame_balance_fraction must be in [0, 1]")
    if query_frame_chunk < 1:
        raise ValueError("query_frame_chunk must be at least 1")
    if not 0.0 <= exact_budget_fraction <= 1.0:
        raise ValueError("exact_budget_fraction must be in [0, 1]")
    if not 0.0 <= surface_weight <= 1.0:
        raise ValueError("surface_weight must be in [0, 1]")
    if repayment_mode not in {
        "reset",
        "deficit",
        "bounded_deficit",
        "effective_bounded_deficit",
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }:
        raise ValueError(
            "repayment_mode must be reset, deficit, bounded_deficit, "
            "effective_bounded_deficit, age_bounded_deficit, or "
            "frontier_age_bounded_deficit"
        )
    if service_credit_scale < 0.0:
        raise ValueError("service_credit_scale must be non-negative")
    if selection_granularity not in {"frame", "cell"}:
        raise ValueError("selection_granularity must be 'frame' or 'cell'")
    if selection_granularity == "cell" and exact_budget_fraction > 0.0:
        raise ValueError("cell selection currently requires exact_fraction=0")
    if exact_mass_conserving_refinement and not mass_conserving_refinement:
        raise ValueError(
            "exact MC requires mass_conserving_refinement=True"
        )
    if exact_mass_conserving_refinement and (
        selection_granularity != "cell"
        or local_frame_radius != 0
        or cell_refinement_phases != 1
        or cell_precision_fraction != 0.0
    ):
        raise ValueError(
            "exact MC currently requires the direct one-phase cell layout"
        )
    if not 0.0 <= frame_detail_power <= 1.0:
        raise ValueError("frame_detail_power must be in [0, 1]")
    if not 0.0 <= spatial_detail_power <= 1.0:
        raise ValueError("spatial_detail_power must be in [0, 1]")
    if not 0.0 <= carrier_residual_balance <= 1.0:
        raise ValueError("carrier_residual_balance must be in [0, 1]")
    if cell_scorer not in {
        "uniform",
        "residual",
        "carrier",
        "carrier_residual",
        "patch_qk",
        "scout_service",
    }:
        raise ValueError(
            "cell_scorer must be uniform, residual, carrier, "
            "carrier_residual, patch_qk, or scout_service"
        )
    if selection_granularity == "frame" and cell_scorer != "carrier_residual":
        raise ValueError("non-default cell scorers require cell selection")
    if selection_granularity == "frame" and repayment_mode != "reset":
        raise ValueError("deficit repayment requires cell selection")
    if cell_refinement_phases not in {1, 2, 3, 15}:
        raise ValueError("cell_refinement_phases must be 1, 2, 3, or 15")
    hierarchical_refinement = cell_refinement_phases == 15
    if hierarchical_refinement and (
        remote_pool_size != 4
        or selection_granularity != "cell"
        or cell_precision_fraction != 0.0
    ):
        raise ValueError(
            "15-phase hierarchical refinement requires 4x4 cell selection "
            "without precision cells"
        )
    if not 0.0 <= cell_precision_fraction <= 0.5:
        raise ValueError("cell_precision_fraction must be in [0, 0.5]")
    if not 0.0 <= cell_importance_weight <= 1.0:
        raise ValueError("cell_importance_weight must be in [0, 1]")
    if not 0.0 <= cell_importance_floor <= cell_importance_weight:
        raise ValueError(
            "cell_importance_floor must be in [0, cell_importance_weight]"
        )
    if cell_importance_gate_threshold < 0.0:
        raise ValueError("cell_importance_gate_threshold must be non-negative")
    if cell_importance_gate_temperature <= 0.0:
        raise ValueError("cell_importance_gate_temperature must be positive")
    if selection_granularity == "frame" and cell_refinement_phases != 1:
        raise ValueError("frame selection requires cell_refinement_phases=1")
    if selection_granularity == "frame" and cell_precision_fraction > 0.0:
        raise ValueError("frame selection requires cell_precision_fraction=0")
    if cell_refinement_phases != 1 and cell_precision_fraction > 0.0:
        raise ValueError(
            "cell_precision_fraction requires cell_refinement_phases=1"
        )
    if (special_query is None) != (special_key is None):
        raise ValueError("special query and key must be provided together")
    if (special_key is None) != (special_value is None):
        raise ValueError("special key and value must be provided together")
    use_grouped_parent_routing = routing_parent_size > remote_pool_size
    if use_grouped_parent_routing and selection_granularity != "cell":
        raise ValueError("grouped parent routing requires cell selection")
    if use_grouped_parent_routing and hierarchical_refinement:
        raise ValueError(
            "grouped parent routing supports one, two, or three child phases"
        )
    if use_grouped_parent_routing and cell_precision_fraction > 0.0:
        raise ValueError("grouped parent routing does not support precision cells")
    if use_grouped_parent_routing and cell_importance_weight > 0.0:
        raise ValueError("grouped parent routing uses its own parent relevance")
    if adaptive_parent_service and not use_grouped_parent_routing:
        raise ValueError("adaptive parent service requires grouped routing")
    if mixed_parent_execution and not use_grouped_parent_routing:
        raise ValueError("mixed parent execution requires grouped routing")
    if additive_parent_service and not use_grouped_parent_routing:
        raise ValueError("additive parent service requires grouped routing")
    if additive_parent_service and mixed_parent_execution:
        raise ValueError(
            "additive and mixed parent execution are mutually exclusive"
        )
    if additive_parent_service and adaptive_parent_service:
        raise ValueError(
            "additive parent service uses its own variable service capacity"
        )
    if additive_parent_service and not mass_conserving_refinement:
        raise ValueError("additive parent service requires mass conservation")
    if additive_parent_service and exact_mass_conserving_refinement:
        raise ValueError("additive parent service does not support exact MC")
    if additive_parent_service and local_frame_radius != 0:
        raise ValueError("additive parent service currently requires radius 0")
    if additive_parent_service and cell_refinement_phases != 1:
        raise ValueError(
            "additive parent service currently uses one child service phase"
        )
    if not additive_parent_service and additive_full_upgrade_fraction > 0.0:
        raise ValueError(
            "additive full upgrades require additive parent service"
        )
    if not 0.0 <= additive_full_upgrade_fraction <= 0.25:
        raise ValueError(
            "additive full-upgrade fraction must be in [0, 0.25]"
        )
    if (
        not additive_parent_service
        and additive_parent_substitution_fraction > 0.0
    ):
        raise ValueError(
            "additive parent substitution requires additive parent service"
        )
    if not 0.0 <= additive_parent_substitution_fraction <= 1.0:
        raise ValueError(
            "additive parent substitution fraction must be in [0, 1]"
        )
    if mixed_parent_execution and not mass_conserving_refinement:
        raise ValueError("mixed parent execution requires mass conservation")
    if mixed_parent_execution and exact_mass_conserving_refinement:
        raise ValueError("mixed parent execution does not support exact MC")
    if mixed_parent_execution and local_frame_radius != 0:
        raise ValueError("mixed parent execution currently requires radius 0")
    if mixed_parent_execution and not 0 <= mixed_residuals_per_parent <= (
        routing_parent_size // remote_pool_size
    ) ** 2:
        raise ValueError("mixed residual cap must fit one routing parent")

    batch, heads, token_count, head_dim = query.shape
    tokens_per_frame = height * width
    if token_count != num_frames * tokens_per_frame:
        raise ValueError(
            "patch token count does not match the frame grid: "
            f"{token_count} != {num_frames} * {height} * {width}"
        )
    query_frames = query.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    key_frames = key.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )
    value_frames = value.view(
        batch, heads, num_frames, tokens_per_frame, head_dim
    )

    spatial_phase_mode = (
        "quadrant_balanced" if hierarchical_refinement else "default"
    )
    coarse_key = head_rotating_spatial_patch_tokens(
        key,
        num_frames,
        height,
        width,
        remote_pool_size,
        1,
        layer_idx,
        spatial_phase_mode,
    )
    coarse_value = head_rotating_spatial_patch_tokens(
        value,
        num_frames,
        height,
        width,
        remote_pool_size,
        1,
        layer_idx,
        spatial_phase_mode,
    )
    residual_phase_count = max(
        cell_refinement_phases,
        2 if cell_precision_fraction > 0.0 else 1,
    )
    residual_keys = [
        head_rotating_spatial_patch_tokens(
            key,
            num_frames,
            height,
            width,
            remote_pool_size,
            1,
            layer_idx + phase_offset,
            spatial_phase_mode,
        )
        for phase_offset in range(1, residual_phase_count + 1)
    ]
    residual_values = [
        head_rotating_spatial_patch_tokens(
            value,
            num_frames,
            height,
            width,
            remote_pool_size,
            1,
            layer_idx + phase_offset,
            spatial_phase_mode,
        )
        for phase_offset in range(1, residual_phase_count + 1)
    ]
    residual_key = residual_keys[0]
    residual_value = residual_values[0]
    coarse_tokens_per_frame = coarse_key.shape[-2]
    if use_grouped_parent_routing:
        routing_parent_key, routing_parent_valid = (
            head_sharded_spatial_patch_tokens(
                key,
                num_frames,
                height,
                width,
                routing_parent_size,
                layer_idx,
                routing_phase_mode,
            )
        )
        routing_parent_value, _ = head_sharded_spatial_patch_tokens(
            value,
            num_frames,
            height,
            width,
            routing_parent_size,
            layer_idx,
            routing_phase_mode,
        )
        parent_to_children, parent_child_valid = spatial_parent_child_map(
            height,
            width,
            remote_pool_size,
            routing_parent_size,
            device=query.device,
        )
    else:
        routing_parent_key = None
        routing_parent_value = None
        routing_parent_valid = None
        parent_to_children = None
        parent_child_valid = None

    effective_local_radius = min(local_frame_radius, num_frames - 1)
    local_pair_count = (
        num_frames * (2 * effective_local_radius + 1)
        - effective_local_radius * (effective_local_radius + 1)
    )
    dense_patch_keys = num_frames * num_frames * tokens_per_frame
    remote_pair_count = num_frames * num_frames - local_pair_count
    if mixed_parent_execution or additive_parent_service:
        parent_tokens_per_frame = routing_parent_key.shape[-2]
        base_patch_keys = (
            local_pair_count * tokens_per_frame
            + remote_pair_count * parent_tokens_per_frame
        )
    else:
        parent_tokens_per_frame = 0
        base_patch_keys = (
            local_pair_count * tokens_per_frame
            + remote_pair_count * coarse_tokens_per_frame
        )
    target_patch_keys = (1.0 - target_sparsity) * dense_patch_keys
    budget_floor_retained_fraction = base_patch_keys / dense_patch_keys
    budget_feasible = target_patch_keys + 1e-6 >= base_patch_keys
    requested_extra_patch_keys = max(
        0.0,
        target_patch_keys - base_patch_keys,
    )
    requested_extra_keys_per_query = (
        requested_extra_patch_keys / num_frames
    )
    max_local_count = min(num_frames, 2 * local_frame_radius + 1)
    max_remote_frames = max(0, num_frames - max_local_count)
    max_remote_cells = max_remote_frames * coarse_tokens_per_frame
    precision_cells_per_query = 0
    dense_cells_per_query = 0
    quadrant_cells_per_query = 0
    if selection_granularity == "frame":
        exact_upgrade_cost = tokens_per_frame - coarse_tokens_per_frame
        exact_frames_per_query = int(
            math.floor(
                requested_extra_keys_per_query
                * exact_budget_fraction
                / exact_upgrade_cost
            )
        )
        remaining_keys_per_query = max(
            0.0,
            requested_extra_keys_per_query
            - exact_frames_per_query * exact_upgrade_cost,
        )
        phase_frames_per_query = int(
            round(remaining_keys_per_query / coarse_tokens_per_frame)
        )
        exact_frames_per_query = min(
            exact_frames_per_query, max_remote_frames
        )
        phase_frames_per_query = min(
            phase_frames_per_query,
            max_remote_frames - exact_frames_per_query,
        )
        residual_tokens_per_query = (
            phase_frames_per_query * coarse_tokens_per_frame
        )
    else:
        exact_frames_per_query = 0
        phase_frames_per_query = 0
        requested_residual_tokens = round(requested_extra_keys_per_query)
        if additive_parent_service:
            service_capacity_per_frame = int(
                (
                    parent_child_valid.sum(dim=-1) - 1
                ).clamp_min(0).sum().item()
            )
            max_additive_service = (
                max_remote_frames * service_capacity_per_frame
            )
            residual_cells_per_query = min(
                requested_residual_tokens, max_additive_service
            )
            residual_tokens_per_query = residual_cells_per_query
            budget_feasible = (
                budget_feasible
                and requested_residual_tokens <= max_additive_service
            )
        elif mixed_parent_execution:
            residual_cells_per_query = requested_residual_tokens
            residual_tokens_per_query = requested_residual_tokens
        elif hierarchical_refinement:
            (
                dense_cells_per_query,
                quadrant_cells_per_query,
                residual_tokens_per_query,
            ) = hierarchical_refinement_counts(
                requested_residual_tokens,
                max_remote_cells,
            )
            residual_cells_per_query = (
                dense_cells_per_query + quadrant_cells_per_query
            )
        else:
            precision_cells_per_query = round(
                requested_residual_tokens * cell_precision_fraction
            )
            residual_cells_per_query = min(
                requested_residual_tokens - precision_cells_per_query,
                max_remote_cells,
            )
            precision_cells_per_query = min(
                precision_cells_per_query,
                residual_cells_per_query,
            )
            if cell_precision_fraction == 0.0:
                residual_cells_per_query = min(
                    round(
                        requested_extra_keys_per_query
                        / cell_refinement_phases
                    ),
                    max_remote_cells,
                )
            residual_tokens_per_query = (
                residual_cells_per_query * cell_refinement_phases
                + precision_cells_per_query
            )

    if special_query is not None and special_query.shape[-2] > 0:
        special_tokens = special_query.shape[-2]
        if special_tokens % num_frames != 0:
            raise ValueError("special tokens must divide evenly across frames")
        specials_per_frame = special_tokens // num_frames
        query_carrier = special_query.view(
            batch, heads, num_frames, specials_per_frame, head_dim
        ).mean(dim=3)
        key_carrier = special_key.view(
            batch, heads, num_frames, specials_per_frame, head_dim
        ).mean(dim=3)
    else:
        query_carrier = query_frames.mean(dim=3)
        key_carrier = key_frames.mean(dim=3)

    surface_query_carrier = None
    surface_key_carrier = None
    if surface_weight > 0.0:
        surface_query_carrier = query_frames.mean(dim=3)
        surface_key_carrier = key_frames.mean(dim=3)

    cell_importance = None
    if selection_granularity == "cell" and not use_grouped_parent_routing and (
        cell_scorer == "patch_qk"
        or cell_importance_weight > 0.0
        or cell_precision_fraction > 0.0
    ):
        importance_query = F.normalize(
            query_frames.float().mean(dim=3), dim=-1
        )
        importance_key = F.normalize(residual_key.float(), dim=-1)
        cell_logits = torch.zeros(
            batch,
            num_frames,
            num_frames,
            coarse_tokens_per_frame,
            device=query.device,
            dtype=torch.float32,
        )
        for head_idx in range(heads):
            cell_logits.add_(
                torch.einsum(
                    "bqd,bkcd->bqkc",
                    importance_query[:, head_idx],
                    importance_key[:, head_idx],
                )
            )
        cell_logits.div_(heads * temperature)
        cell_importance = cell_logits.flatten(start_dim=-2).softmax(
            dim=-1
        ).view_as(cell_logits)

    if selection_granularity == "frame":
        extra_frame_indices, debt_stats = select_residual_debt_frame_pairs(
            query_carrier,
            key_carrier,
            coarse_value,
            residual_value,
            local_frame_radius=local_frame_radius,
            extra_frames_per_query=(
                exact_frames_per_query + phase_frames_per_query
            ),
            layer_idx=layer_idx,
            routing_state=routing_state,
            momentum=momentum,
            repayment=repayment,
            temperature=temperature,
            surface_query_carrier=surface_query_carrier,
            surface_key_carrier=surface_key_carrier,
            surface_weight=surface_weight,
        )
        extra_frames_per_query = extra_frame_indices.shape[-1]
        exact_frames_per_query = min(
            exact_frames_per_query, extra_frames_per_query
        )
        phase_frames_per_query = (
            extra_frames_per_query - exact_frames_per_query
        )
        residual_tokens_per_query = (
            phase_frames_per_query * coarse_tokens_per_frame
        )
        residual_cell_indices = None
    else:
        collect_routing_stats = not _env_flag(
            "SPARSE_VGGT_SKIP_ROUTING_STATS", default=False
        )
        mixed_parent_indices = None
        substituted_parent_indices = None
        if use_grouped_parent_routing:
            (
                parent_indices,
                served_counts,
                ordered_children,
                substituted_parent_indices,
                substitution_child_indices,
                debt_stats,
            ) = (
                select_grouped_residual_debt_cells(
                    query_carrier,
                    routing_parent_key,
                    routing_parent_valid,
                    coarse_value,
                    residual_value,
                    parent_to_children,
                    parent_child_valid,
                    local_frame_radius=local_frame_radius,
                    child_budget_per_query=residual_cells_per_query,
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                    momentum=momentum,
                    service_conditioned_momentum=(
                        service_conditioned_momentum
                    ),
                    repayment=repayment,
                    repayment_mode=repayment_mode,
                    service_credit_scale=service_credit_scale,
                    temperature=temperature,
                    frame_detail_power=frame_detail_power,
                    spatial_detail_power=spatial_detail_power,
                    cell_scorer=cell_scorer,
                    carrier_residual_balance=carrier_residual_balance,
                    unified_incremental_service=(
                        unified_incremental_service
                    ),
                    bounded_wait_reobservation=(
                        bounded_wait_reobservation
                    ),
                    adaptive_parent_service=adaptive_parent_service,
                    mixed_parent_execution=mixed_parent_execution,
                    additive_parent_service=additive_parent_service,
                    additive_full_upgrade_fraction=(
                        additive_full_upgrade_fraction
                    ),
                    additive_parent_substitution_fraction=(
                        additive_parent_substitution_fraction
                    ),
                    routing_phase_mode=routing_phase_mode,
                    mixed_extra_budget_per_query=(
                        residual_cells_per_query
                    ),
                    mixed_residuals_per_parent=(
                        mixed_residuals_per_parent
                    ),
                    parent_hard_fraction=parent_hard_fraction,
                    parent_cost_power=parent_cost_power,
                    frame_balance_fraction=frame_balance_fraction,
                    target_sparsity=target_sparsity,
                    total_layers=total_layers,
                    collect_stats=collect_routing_stats,
                )
            )
            observation_snapshot = debt_stats.pop(
                "_service_observation_snapshot", None
            )
            if isinstance(observation_snapshot, dict):
                debt_stats.update(
                    analyze_grouped_service_observation(
                        query_frames,
                        key_frames,
                        observation_snapshot,
                        value_frames=value_frames,
                        height=height,
                        width=width,
                        parent_size=routing_parent_size,
                        local_frame_radius=local_frame_radius,
                        layer_idx=layer_idx,
                        routing_state=routing_state,
                        special_key=special_key,
                        special_value=special_value,
                    )
                )
            if mixed_parent_execution:
                mixed_parent_indices = parent_indices
                residual_cells_per_query = int(
                    debt_stats["residual_budget_mixed_residual_budget"]
                )
            residual_cell_indices = expand_grouped_parent_selection(
                parent_indices,
                served_counts,
                parent_to_children,
                parents_per_frame=parent_to_children.shape[0],
                children_per_frame=coarse_tokens_per_frame,
                child_budget_per_query=residual_cells_per_query,
                ordered_children=ordered_children,
            )
            if substitution_child_indices.shape[-1] > 0:
                residual_cell_indices = torch.cat(
                    (
                        residual_cell_indices,
                        substitution_child_indices,
                    ),
                    dim=-1,
                )
                residual_tokens_per_query += (
                    substitution_child_indices.shape[-1]
                )
        else:
            residual_cell_indices, debt_stats = select_residual_debt_cells(
                query_carrier,
                key_carrier,
                coarse_value,
                residual_value,
                local_frame_radius=local_frame_radius,
                extra_tokens_per_query=residual_cells_per_query,
                layer_idx=layer_idx,
                routing_state=routing_state,
                momentum=momentum,
                repayment=repayment,
                repayment_mode=repayment_mode,
                service_credit_scale=service_credit_scale,
                temperature=temperature,
                frame_detail_power=frame_detail_power,
                spatial_detail_power=spatial_detail_power,
                cell_scorer=cell_scorer,
                carrier_residual_balance=carrier_residual_balance,
                cell_importance=cell_importance,
                cell_importance_weight=cell_importance_weight,
                cell_importance_floor=cell_importance_floor,
                cell_importance_gate_threshold=(
                    cell_importance_gate_threshold
                ),
                cell_importance_gate_temperature=(
                    cell_importance_gate_temperature
                ),
                collect_stats=collect_routing_stats,
            )
        if precision_cells_per_query > 0:
            flat_importance = cell_importance.flatten(start_dim=-2)
            selected_importance = flat_importance.gather(
                -1, residual_cell_indices
            )
            precision_positions = selected_importance.topk(
                precision_cells_per_query, dim=-1
            ).indices
            precision_cell_indices = residual_cell_indices.gather(
                -1, precision_positions
            )
            precision_importance = float(
                selected_importance.gather(-1, precision_positions)
                .mean()
                .item()
            )
        else:
            precision_cell_indices = residual_cell_indices[..., :0]
            precision_importance = 0.0
        residual_cells_per_query = residual_cell_indices.shape[-1]
        if hierarchical_refinement:
            dense_cell_indices = residual_cell_indices[
                ..., :dense_cells_per_query
            ]
            residual_phase_cell_indices = (
                [residual_cell_indices] * 3
                + [dense_cell_indices] * 12
            )
            residual_tokens_per_query = sum(
                indices.shape[-1]
                for indices in residual_phase_cell_indices
            )
            debt_stats.update({
                "residual_budget_hierarchical_refinement": 1.0,
                "residual_budget_dense_cells_per_query": float(
                    dense_cells_per_query
                ),
                "residual_budget_quadrant_cells_per_query": float(
                    quadrant_cells_per_query
                ),
            })
        else:
            residual_phase_cell_indices = (
                [residual_cell_indices] * cell_refinement_phases
            )
            residual_tokens_per_query = (
                residual_cells_per_query * cell_refinement_phases
                + precision_cells_per_query
            )
        debt_stats["residual_budget_extra_frames_per_query"] = float(
            residual_tokens_per_query / coarse_tokens_per_frame
        )
        extra_frame_indices = None

    use_grouped_sdpa = _env_flag(
        "SPARSE_VGGT_GROUPED_SDPA", default=False
    )
    use_mixed_direct_attention = (
        mixed_parent_execution
        and _env_flag("SPARSE_VGGT_DIRECT_ATTENTION", default=False)
        and query.is_cuda
        and num_frames > 1
        and not torch.is_grad_enabled()
        and selection_granularity == "cell"
        and local_frame_radius == 0
        and exact_frames_per_query == 0
        and cell_refinement_phases == 1
        and precision_cells_per_query == 0
        and query_frames.is_contiguous()
        and key_frames.is_contiguous()
        and value_frames.is_contiguous()
        and routing_parent_key.is_contiguous()
        and routing_parent_value.is_contiguous()
        and coarse_key.is_contiguous()
        and coarse_value.is_contiguous()
        and residual_key.is_contiguous()
        and residual_value.is_contiguous()
    )
    use_direct_attention = (
        _env_flag("SPARSE_VGGT_DIRECT_ATTENTION", default=False)
        and query.is_cuda
        and num_frames > 1
        and residual_tokens_per_query > 0
        and not torch.is_grad_enabled()
        and selection_granularity == "cell"
        and not mixed_parent_execution
        and not additive_parent_service
        and local_frame_radius == 0
        and exact_frames_per_query == 0
        and precision_cells_per_query == 0
        and query_frames.is_contiguous()
        and key_frames.is_contiguous()
        and value_frames.is_contiguous()
        and coarse_key.is_contiguous()
        and coarse_value.is_contiguous()
        and residual_key.is_contiguous()
        and residual_value.is_contiguous()
    )
    use_additive_direct_attention = (
        additive_parent_service
        and _env_flag("SPARSE_VGGT_DIRECT_ATTENTION", default=False)
        and _env_flag(
            "SPARSE_VGGT_ADDITIVE_DIRECT_ATTENTION", default=False
        )
        and query.is_cuda
        and num_frames > 1
        and not torch.is_grad_enabled()
        and selection_granularity == "cell"
        and not mixed_parent_execution
        and local_frame_radius == 0
        and exact_frames_per_query == 0
        and cell_refinement_phases == 1
        and precision_cells_per_query == 0
        and query_frames.is_contiguous()
        and key_frames.is_contiguous()
        and value_frames.is_contiguous()
        and routing_parent_key.is_contiguous()
        and routing_parent_value.is_contiguous()
        and coarse_key.is_contiguous()
        and coarse_value.is_contiguous()
    )
    any_direct_attention = (
        use_direct_attention
        or use_mixed_direct_attention
        or use_additive_direct_attention
    )
    block_aligned_residual = (
        (use_direct_attention or use_mixed_direct_attention)
        and _env_flag(
            "SPARSE_VGGT_BLOCK_ALIGNED_RESIDUAL", default=False
        )
    )
    direct_qk_bf16 = (
        any_direct_attention
        and _env_flag("SPARSE_VGGT_DIRECT_QK_BF16", default=False)
    )
    direct_qk_bf16_fused = (
        direct_qk_bf16
        and _env_flag(
            "SPARSE_VGGT_DIRECT_QK_BF16_FUSED", default=False
        )
    )
    direct_tile_profile = os.environ.get(
        "SPARSE_VGGT_DIRECT_TILE_PROFILE", "manual"
    )
    direct_block_m, direct_block_n = resolve_direct_attention_tile(
        num_frames,
        direct_tile_profile,
        int(os.environ.get("SPARSE_VGGT_DIRECT_BLOCK_M", "128")),
        int(os.environ.get("SPARSE_VGGT_DIRECT_BLOCK_N", "128")),
    )
    direct_num_warps = int(
        os.environ.get("SPARSE_VGGT_DIRECT_NUM_WARPS", "0")
    ) or None
    direct_num_stages = resolve_direct_attention_stages(
        num_frames,
        direct_tile_profile,
        int(os.environ.get("SPARSE_VGGT_DIRECT_NUM_STAGES", "0")),
    )
    direct_int32_indices = (
        any_direct_attention
        and _env_flag(
            "SPARSE_VGGT_DIRECT_INT32_INDICES", default=False
        )
    )
    direct_native_output = (
        any_direct_attention
        and _env_flag(
            "SPARSE_VGGT_DIRECT_NATIVE_OUTPUT", default=False
        )
    )
    phase_compiled_layout = (
        any_direct_attention
        and _env_flag(
            "SPARSE_VGGT_PHASE_COMPILED_LAYOUT", default=False
        )
    )
    use_parent_mask_additive_attention = (
        use_additive_direct_attention
        and _env_flag(
            "SPARSE_VGGT_PARENT_MASK_ATTENTION", default=False
        )
        and parent_to_children.shape[-1] == 4
    )
    use_compiled_carrier_attention = (
        use_additive_direct_attention
        and not use_parent_mask_additive_attention
        and _env_flag(
            "SPARSE_VGGT_COMPILED_CARRIER_ATTENTION", default=False
        )
    )
    compiled_carrier_dense_block_n = int(
        os.environ.get("SPARSE_VGGT_COMPILED_CARRIER_DENSE_BLOCK_N", "32")
    )
    compiled_carrier_parent_block_n = int(
        os.environ.get("SPARSE_VGGT_COMPILED_CARRIER_PARENT_BLOCK_N", "32")
    )
    compiled_carrier_child_block_n = int(
        os.environ.get("SPARSE_VGGT_COMPILED_CARRIER_CHILD_BLOCK_N", "32")
    )
    compiled_carrier_value_bf16 = _env_flag(
        "SPARSE_VGGT_COMPILED_CARRIER_VALUE_BF16", default=False
    )
    compiled_carrier_precast_value_bf16 = _env_flag(
        "SPARSE_VGGT_COMPILED_CARRIER_PRECAST_VALUE_BF16", default=False
    )
    use_additive_flash_carrier = (
        use_additive_direct_attention
        and not use_parent_mask_additive_attention
        and not use_compiled_carrier_attention
        and _env_flag(
            "SPARSE_VGGT_ADDITIVE_FLASH_CARRIER",
            default=False,
        )
        and (
            query_frames.dtype in {torch.float16, torch.bfloat16}
            or (direct_qk_bf16 and not direct_qk_bf16_fused)
        )
    )
    use_additive_flash_positive_groups = (
        use_additive_flash_carrier
        and _env_flag(
            "SPARSE_VGGT_ADDITIVE_FLASH_POSITIVE_GROUPS",
            default=False,
        )
    )
    two_stage_qk_pv_requested = _env_flag(
        "SPARSE_VGGT_TWO_STAGE_QK_PV", default=False
    )
    collect_two_stage_qk_pv = _env_flag(
        "SPARSE_VGGT_COLLECT_TWO_STAGE_QK_PV", default=False
    )
    pv_service_debt_feedback = _env_flag(
        "SPARSE_VGGT_PV_SERVICE_DEBT_FEEDBACK", default=False
    )
    pv_importance_alignment_observer = _env_flag(
        "SPARSE_VGGT_PV_IMPORTANCE_ALIGNMENT_OBSERVER", default=False
    )
    carrier_compensation_residual_observer = _env_flag(
        "SPARSE_VGGT_CARRIER_COMPENSATION_RESIDUAL_OBSERVER", default=False
    )
    carrier_compensated_pv = _env_flag(
        "SPARSE_VGGT_CARRIER_COMPENSATED_PV", default=False
    )
    two_stage_pv_log2_threshold = float(
        os.environ.get("SPARSE_VGGT_PV_LOG2_THRESHOLD", "-4.0")
    )
    use_two_stage_qk_pv = (
        two_stage_qk_pv_requested
        and use_additive_direct_attention
        and not use_parent_mask_additive_attention
        and not use_compiled_carrier_attention
        and not use_additive_flash_carrier
    )
    if two_stage_qk_pv_requested and not use_two_stage_qk_pv:
        raise RuntimeError(
            "two-stage QK/PV currently requires additive descriptor-direct "
            "parent-carrier execution"
        )
    if collect_two_stage_qk_pv and not use_two_stage_qk_pv:
        raise RuntimeError(
            "two-stage QK/PV statistics require the two-stage backend"
        )
    if pv_service_debt_feedback and not (
        use_two_stage_qk_pv
        and unified_incremental_service
        and additive_parent_service
    ):
        raise RuntimeError(
            "PV-service debt feedback requires unified additive two-stage "
            "execution"
        )
    if pv_importance_alignment_observer and not (
        use_two_stage_qk_pv
        and unified_incremental_service
        and additive_parent_service
    ):
        raise RuntimeError(
            "PV importance alignment observation requires unified additive "
            "two-stage execution"
        )
    if carrier_compensation_residual_observer and not (
        use_two_stage_qk_pv
        and collect_two_stage_qk_pv
        and unified_incremental_service
        and additive_parent_service
    ):
        raise RuntimeError(
            "carrier compensation observation requires unified additive "
            "two-stage execution with statistics"
        )
    if carrier_compensated_pv and not (
        use_two_stage_qk_pv
        and unified_incremental_service
        and additive_parent_service
    ):
        raise RuntimeError(
            "carrier-compensated PV requires unified additive two-stage "
            "execution"
        )
    mixed_flash_dtype_supported = (
        query_frames.dtype in {torch.float16, torch.bfloat16}
        or direct_qk_bf16
    )
    use_mixed_flash_self = (
        use_mixed_direct_attention
        and _env_flag(
            "SPARSE_VGGT_MIXED_FLASH_SELF", default=False
        )
        and mixed_flash_dtype_supported
    )
    use_mixed_flash_parent = (
        use_mixed_direct_attention
        and not use_mixed_flash_self
        and _env_flag(
            "SPARSE_VGGT_MIXED_FLASH_PARENT", default=False
        )
        and mixed_flash_dtype_supported
    )
    use_hps_flash_carrier = (
        use_direct_attention
        and _env_flag(
            "SPARSE_VGGT_HPS_FLASH_CARRIER", default=False
        )
        and mass_conserving_refinement
        and not exact_mass_conserving_refinement
        and len(residual_keys) == 1
        and (
            query_frames.dtype in {torch.float16, torch.bfloat16}
            or (direct_qk_bf16 and not direct_qk_bf16_fused)
        )
    )
    hps_correction_mode = os.environ.get(
        "SPARSE_VGGT_HPS_COMPLEMENT_CORRECTION", "selected"
    )
    use_hps_complement_correction = (
        use_hps_flash_carrier
        and resolve_hps_complement_correction(
            hps_correction_mode,
            residual_tokens_per_query,
            max_remote_cells,
        )
    )
    use_triton_pack = (
        _env_flag("SPARSE_VGGT_TRITON_PACK", default=False)
        and not use_direct_attention
        and use_grouped_sdpa
        and query.is_cuda
        and num_frames > 1
        and residual_tokens_per_query > 0
        and not torch.is_grad_enabled()
        and selection_granularity == "cell"
        and not mixed_parent_execution
        and not additive_parent_service
        and local_frame_radius == 0
        and exact_frames_per_query == 0
        and cell_refinement_phases == 1
        and precision_cells_per_query == 0
        and key_frames.is_contiguous()
        and value_frames.is_contiguous()
        and coarse_key.is_contiguous()
        and coarse_value.is_contiguous()
        and residual_key.is_contiguous()
        and residual_value.is_contiguous()
        and (special_key is None or special_key.is_contiguous())
        and (special_value is None or special_value.is_contiguous())
    )
    if exact_mass_conserving_refinement and not use_direct_attention:
        raise RuntimeError(
            "exact MC requires SPARSE_VGGT_DIRECT_ATTENTION=1 and the "
            "descriptor-direct inference layout"
        )
    if use_mixed_direct_attention:
        parents_per_frame = routing_parent_key.shape[-2]
        mixed_residual_indices = residual_cell_indices
        if block_aligned_residual:
            mixed_residual_indices = mixed_residual_indices.sort(
                dim=-1
            ).values
        kernel_residual_indices = (
            mixed_residual_indices.to(dtype=torch.int32)
            if direct_int32_indices
            else mixed_residual_indices
        )
        direct_query = (
            query_frames.to(torch.bfloat16)
            if direct_qk_bf16
            and (
                use_mixed_flash_self
                or use_mixed_flash_parent
                or not direct_qk_bf16_fused
            )
            else query_frames
        )

        def direct_key(source):
            return (
                source.to(torch.bfloat16)
                if direct_qk_bf16
                and (
                    use_mixed_flash_self
                    or use_mixed_flash_parent
                    or not direct_qk_bf16_fused
                )
                else source
            )

        direct_special_key = (
            None
            if special_key is None
            else special_key.to(direct_query.dtype).contiguous()
        )
        if not use_mixed_flash_parent:
            mixed_parent_ids = mixed_parent_indices.remainder(
                parents_per_frame
            )
            mixed_parent_frames = torch.div(
                mixed_parent_indices,
                parents_per_frame,
                rounding_mode="floor",
            )
            query_frame_ids = torch.arange(
                num_frames,
                device=query.device,
                dtype=mixed_parent_frames.dtype,
            ).view(1, -1, 1)
            hard_parent_positions = (
                mixed_parent_frames
                - (mixed_parent_frames > query_frame_ids).to(
                    dtype=mixed_parent_frames.dtype
                )
            ) * parents_per_frame + mixed_parent_ids
            remote_parent_indices = remote_parent_index_grid(
                num_frames,
                parents_per_frame,
                device=query.device,
            ).view(1, num_frames, -1).expand(batch, -1, -1)
            keep_easy_parent = torch.ones(
                remote_parent_indices.shape,
                device=query.device,
                dtype=torch.bool,
            )
            keep_easy_parent.scatter_(2, hard_parent_positions, False)
            easy_parent_indices = remote_parent_indices[
                keep_easy_parent
            ].view(
                batch,
                num_frames,
                -1,
            )
            kernel_easy_parent_indices = (
                easy_parent_indices.to(dtype=torch.int32)
                if direct_int32_indices
                else easy_parent_indices
            )

        if use_mixed_flash_parent:
            from sparse_vggt.kernels.mixed_flash_parent import (
                mixed_flash_parent_attention,
            )

            kernel_hard_parent_indices = (
                mixed_parent_indices.to(dtype=torch.int32)
                if direct_int32_indices
                else mixed_parent_indices
            )
            kernel_parent_to_children = parent_to_children.to(
                device=query.device,
                dtype=(
                    torch.int32
                    if direct_int32_indices
                    else parent_to_children.dtype
                ),
            ).contiguous()
            output = mixed_flash_parent_attention(
                direct_query,
                direct_key(key_frames),
                value_frames,
                direct_key(routing_parent_key),
                routing_parent_value,
                direct_key(coarse_key),
                coarse_value,
                direct_key(residual_key),
                residual_value,
                kernel_hard_parent_indices,
                kernel_residual_indices,
                kernel_parent_to_children,
                direct_special_key,
                (
                    None
                    if special_value is None
                    else special_value.contiguous()
                ),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        elif use_mixed_flash_self:
            from sparse_vggt.kernels.mixed_flash_self import (
                mixed_flash_self_attention,
            )

            kernel_hard_parent_indices = (
                mixed_parent_indices.to(dtype=torch.int32)
                if direct_int32_indices
                else mixed_parent_indices
            )
            kernel_parent_to_children = parent_to_children.to(
                device=query.device,
                dtype=(
                    torch.int32
                    if direct_int32_indices
                    else parent_to_children.dtype
                ),
            ).contiguous()
            output = mixed_flash_self_attention(
                direct_query,
                direct_key(key_frames),
                value_frames,
                direct_key(routing_parent_key),
                routing_parent_value,
                direct_key(coarse_key),
                coarse_value,
                direct_key(residual_key),
                residual_value,
                kernel_easy_parent_indices,
                kernel_hard_parent_indices,
                kernel_residual_indices,
                kernel_parent_to_children,
                direct_special_key,
                (
                    None
                    if special_value is None
                    else special_value.contiguous()
                ),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        else:
            from sparse_vggt.kernels.mixed_direct_attention import (
                mixed_descriptor_direct_attention,
            )

            mixed_child_ids = parent_to_children[mixed_parent_ids]
            mixed_hard_child_indices = (
                mixed_parent_frames[..., None] * coarse_tokens_per_frame
                + mixed_child_ids
            ).flatten(start_dim=-2)
            refined_child_map = torch.zeros(
                batch,
                num_frames,
                num_frames * coarse_tokens_per_frame,
                device=query.device,
                dtype=torch.bool,
            )
            if mixed_residual_indices.shape[-1] > 0:
                refined_child_map.scatter_(
                    2, mixed_residual_indices, True
                )
            kernel_hard_child_indices = (
                mixed_hard_child_indices.to(dtype=torch.int32)
                if direct_int32_indices
                else mixed_hard_child_indices
            )
            output = mixed_descriptor_direct_attention(
                direct_query,
                direct_key(key_frames),
                value_frames,
                direct_key(routing_parent_key),
                routing_parent_value,
                direct_key(coarse_key),
                coarse_value,
                direct_key(residual_key),
                residual_value,
                kernel_easy_parent_indices,
                kernel_hard_child_indices,
                kernel_residual_indices,
                refined_child_map,
                direct_special_key,
                (
                    None
                    if special_value is None
                    else special_value.contiguous()
                ),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                dot_qk_bf16=direct_qk_bf16_fused,
                prevalidated_remote_layout=phase_compiled_layout,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        if output.dtype != query_frames.dtype:
            output = output.to(query_frames.dtype)
        mixed_remote_keys_per_query = (
            (num_frames - 1) * parents_per_frame
            + mixed_parent_indices.shape[-1]
            * (parent_to_children.shape[-1] - 1)
        )
        total_local_keys = local_pair_count * tokens_per_frame
        total_remote_keys = num_frames * mixed_remote_keys_per_query
        total_residual_keys = num_frames * residual_tokens_per_query
        total_exact_remote_keys = 0
        chunk_starts = ()
    elif use_additive_direct_attention:
        all_centers = torch.arange(num_frames, device=query.device)
        child_masses = spatial_pool_multiplicity(
            height,
            width,
            remote_pool_size,
            device=query.device,
            dtype=torch.float32,
        )
        (
            additive_parent_indices,
            additive_parent_bias,
            additive_child_indices,
            additive_child_bias,
        ) = additive_parent_service_descriptors(
            routing_parent_key,
            coarse_key,
            residual_cell_indices,
            substituted_parent_indices,
            parent_to_children,
            parent_child_valid,
            all_centers,
            child_masses=child_masses,
            bias_dtype=query_frames.dtype,
        )
        if use_parent_mask_additive_attention:
            service_child_masks = encode_grouped_parent_service_masks(
                parent_indices,
                served_counts,
                ordered_children,
                substituted_parent_indices,
                parent_to_children,
                parent_child_valid,
                parents_per_frame=parent_to_children.shape[0],
            )
            service_parent_indices = parent_indices
            (
                tail_child_indices,
                tail_child_valid,
            ) = parent_service_mask_tail_descriptors(
                parent_indices,
                service_child_masks,
                parent_to_children,
                parents_per_frame=parent_to_children.shape[0],
                children_per_frame=coarse_tokens_per_frame,
            )
            tail_child_ids = tail_child_indices.remainder(
                coarse_tokens_per_frame
            )
            tail_child_log_bias = torch.where(
                tail_child_valid,
                torch.log(child_masses[tail_child_ids]),
                torch.full(
                    (),
                    -float("inf"),
                    device=query.device,
                    dtype=torch.float32,
                ),
            ).to(dtype=query_frames.dtype)
            kernel_parent_to_children = parent_to_children.to(
                dtype=torch.int32
            ).contiguous()
            child_log_masses = torch.log(child_masses).contiguous()
        if use_additive_flash_carrier:
            parent_mass = (
                child_masses[parent_to_children]
                * parent_child_valid.to(dtype=torch.float32)
            ).sum(dim=-1)
            base_parent_mass = int(parent_mass.max().item())
            parent_correction_mass = torch.full(
                (
                    batch,
                    num_frames,
                    num_frames * routing_parent_key.shape[-2],
                ),
                float(base_parent_mass),
                device=query.device,
                dtype=torch.float32,
            )
            target_parent_mass = torch.exp(
                additive_parent_bias.to(dtype=torch.float32)
            )
            parent_correction_mass.scatter_(
                -1,
                additive_parent_indices,
                float(base_parent_mass) - target_parent_mass,
            )
        if direct_int32_indices:
            additive_parent_indices = additive_parent_indices.to(
                dtype=torch.int32
            )
            additive_child_indices = additive_child_indices.to(
                dtype=torch.int32
            )
            if use_parent_mask_additive_attention:
                service_parent_indices = service_parent_indices.to(
                    dtype=torch.int32
                )
                tail_child_indices = tail_child_indices.to(
                    dtype=torch.int32
                )
        direct_query = (
            query_frames.to(torch.bfloat16)
            if direct_qk_bf16 and not direct_qk_bf16_fused
            else query_frames
        )

        def additive_direct_key(source):
            return (
                source.to(torch.bfloat16)
                if direct_qk_bf16 and not direct_qk_bf16_fused
                else source
            )

        direct_special_key = (
            None
            if special_key is None
            else special_key.to(direct_query.dtype).contiguous()
        )
        if use_parent_mask_additive_attention:
            from sparse_vggt.kernels.mixed_direct_attention import (
                parent_mask_additive_attention,
            )

            output = parent_mask_additive_attention(
                direct_query,
                additive_direct_key(key_frames),
                value_frames,
                additive_direct_key(routing_parent_key),
                routing_parent_value,
                additive_direct_key(coarse_key),
                coarse_value,
                additive_parent_indices,
                additive_parent_bias,
                service_parent_indices,
                service_child_masks,
                kernel_parent_to_children,
                child_log_masses,
                tail_child_indices,
                tail_child_log_bias,
                direct_special_key,
                None if special_value is None else special_value.contiguous(),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                dot_qk_bf16=direct_qk_bf16_fused,
                prevalidated_remote_layout=phase_compiled_layout,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        elif use_compiled_carrier_attention:
            from sparse_vggt.kernels.mixed_direct_attention import (
                compiled_carrier_additive_attention,
            )

            def compiled_carrier_value(source):
                if compiled_carrier_precast_value_bf16:
                    return source.to(torch.bfloat16).contiguous()
                return source.contiguous()

            if (
                layer_idx == 0
                and _env_flag(
                    "SPARSE_VGGT_COMPILED_CARRIER_DEBUG", default=False
                )
            ):
                debug_sources = {
                    "query": direct_query,
                    "dense_key": additive_direct_key(key_frames),
                    "dense_value": value_frames,
                    "parent_key": additive_direct_key(routing_parent_key),
                    "parent_value": routing_parent_value,
                    "child_key": additive_direct_key(coarse_key),
                    "child_value": coarse_value,
                    "parent_indices": additive_parent_indices,
                    "child_indices": additive_child_indices,
                }
                if direct_special_key is not None:
                    debug_sources["special_key"] = direct_special_key
                    debug_sources["special_value"] = compiled_carrier_value(
                        special_value.to(value_frames.dtype)
                    )
                print(
                    "compiled_carrier_debug ",
                    {
                        name: {
                            "shape": tuple(source.shape),
                            "dtype": str(source.dtype),
                            "stride": tuple(source.stride()),
                            "contiguous": source.is_contiguous(),
                        }
                        for name, source in debug_sources.items()
                    },
                    flush=True,
                )

            output = compiled_carrier_additive_attention(
                direct_query,
                additive_direct_key(key_frames),
                compiled_carrier_value(value_frames),
                additive_direct_key(routing_parent_key),
                compiled_carrier_value(routing_parent_value),
                additive_direct_key(coarse_key),
                compiled_carrier_value(coarse_value),
                additive_parent_indices.contiguous(),
                additive_parent_bias.contiguous(),
                additive_child_indices.contiguous(),
                additive_child_bias.contiguous(),
                direct_special_key,
                (
                    None
                    if special_value is None
                    else compiled_carrier_value(
                        special_value.to(value_frames.dtype)
                    )
                ),
                block_m=direct_block_m,
                dense_block_n=compiled_carrier_dense_block_n,
                parent_block_n=compiled_carrier_parent_block_n,
                child_block_n=compiled_carrier_child_block_n,
                num_warps=direct_num_warps or 4,
                num_stages=direct_num_stages or 3,
                prevalidated_remote_layout=phase_compiled_layout,
                value_bf16=compiled_carrier_value_bf16,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        elif use_additive_flash_carrier:
            from sparse_vggt.kernels.additive_flash_carrier import (
                additive_flash_carrier_attention,
            )

            output = additive_flash_carrier_attention(
                direct_query,
                additive_direct_key(key_frames),
                value_frames,
                additive_direct_key(routing_parent_key),
                routing_parent_value,
                additive_direct_key(coarse_key),
                coarse_value,
                parent_correction_mass,
                additive_child_indices,
                additive_child_bias,
                direct_special_key,
                None if special_value is None else special_value.contiguous(),
                base_parent_mass=base_parent_mass,
                base_child_mass=int(child_masses.max().item()),
                positive_flash_groups=(
                    use_additive_flash_positive_groups
                ),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps or 8,
                num_stages=direct_num_stages or 2,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        else:
            from sparse_vggt.kernels.mixed_direct_attention import (
                additive_descriptor_direct_attention,
            )

            kernel_child_to_parent = None
            if (
                carrier_compensation_residual_observer
                or carrier_compensated_pv
            ):
                kernel_child_to_parent = spatial_child_parent_map(
                    parent_to_children,
                    parent_child_valid,
                    children_per_frame=coarse_tokens_per_frame,
                ).to(dtype=torch.int32).contiguous()

            additive_output = additive_descriptor_direct_attention(
                direct_query,
                additive_direct_key(key_frames),
                value_frames,
                additive_direct_key(routing_parent_key),
                routing_parent_value,
                additive_direct_key(coarse_key),
                coarse_value,
                additive_parent_indices,
                additive_parent_bias,
                additive_child_indices,
                additive_child_bias,
                direct_special_key,
                None if special_value is None else special_value.contiguous(),
                child_to_parent=kernel_child_to_parent,
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                dot_qk_bf16=direct_qk_bf16_fused,
                prevalidated_remote_layout=phase_compiled_layout,
                two_stage_qk_pv=use_two_stage_qk_pv,
                pv_log2_threshold=two_stage_pv_log2_threshold,
                collect_two_stage_stats=collect_two_stage_qk_pv,
                return_pv_execution_map=(
                    pv_service_debt_feedback
                    or pv_importance_alignment_observer
                ),
                observe_carrier_compensation=(
                    carrier_compensation_residual_observer
                ),
                carrier_compensated_pv=carrier_compensated_pv,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
            pv_execution_fraction = None
            need_pv_execution_map = (
                pv_service_debt_feedback
                or pv_importance_alignment_observer
            )
            if collect_two_stage_qk_pv and need_pv_execution_map:
                output, two_stage_stats, pv_execution_fraction = additive_output
                debt_stats.update({
                    f"residual_budget_two_stage_{key}": value
                    for key, value in two_stage_stats.items()
                })
            elif collect_two_stage_qk_pv:
                output, two_stage_stats = additive_output
                debt_stats.update({
                    f"residual_budget_two_stage_{key}": value
                    for key, value in two_stage_stats.items()
                })
            elif need_pv_execution_map:
                output, pv_execution_fraction = additive_output
            else:
                output = additive_output
            if need_pv_execution_map:
                feedback_snapshot = debt_stats.pop(
                    "_pv_service_debt_snapshot", None
                )
                if not isinstance(feedback_snapshot, dict):
                    raise RuntimeError(
                        "PV-service debt snapshot is missing"
                    )
                if additive_child_indices.shape[-1] != residual_cell_indices.shape[-1]:
                    raise RuntimeError(
                        "PV-service feedback requires one descriptor per "
                        "scheduled child"
                    )
                if pv_importance_alignment_observer:
                    debt_stats.update(
                        measure_pv_importance_alignment(
                            pv_execution_fraction,
                            feedback_snapshot,
                            child_budget_per_query=(
                                residual_cell_indices.shape[-1]
                            ),
                            child_block_size=direct_block_n,
                        )
                    )
                if pv_service_debt_feedback:
                    debt_stats.update(
                        apply_pv_execution_service_feedback(
                            pv_execution_fraction,
                            feedback_snapshot,
                            child_budget_per_query=(
                                residual_cell_indices.shape[-1]
                            ),
                            child_block_size=direct_block_n,
                            routing_state=routing_state,
                        )
                    )
        if output.dtype != query_frames.dtype:
            output = output.to(query_frames.dtype)
        total_local_keys = local_pair_count * tokens_per_frame
        total_remote_keys = (
            num_frames * additive_parent_indices.shape[-1]
        )
        total_residual_keys = (
            num_frames * additive_child_indices.shape[-1]
        )
        total_exact_remote_keys = 0
        chunk_starts = ()
    elif use_direct_attention:
        (
            direct_residual_indices,
            all_base_refinement_indices,
            _,
        ) = encode_multiphase_residual_indices(
            residual_phase_cell_indices,
            coarse_cells_per_frame=coarse_tokens_per_frame,
        )
        if block_aligned_residual:
            direct_residual_indices = direct_residual_indices.sort(
                dim=-1
            ).values
        if len(residual_phase_cell_indices) == 1:
            refined_map = torch.zeros(
                batch,
                num_frames,
                num_frames * coarse_tokens_per_frame,
                device=query.device,
                dtype=torch.bool,
            )
            refined_map.scatter_(2, all_base_refinement_indices, True)
        else:
            refined_map = torch.zeros(
                batch,
                num_frames,
                num_frames * coarse_tokens_per_frame,
                device=query.device,
                dtype=torch.int32,
            )
            refined_map.scatter_add_(
                2,
                all_base_refinement_indices,
                torch.ones_like(
                    all_base_refinement_indices, dtype=refined_map.dtype
                ),
            )
        kernel_residual_indices = (
            direct_residual_indices.to(dtype=torch.int32)
            if direct_int32_indices
            else direct_residual_indices
        )
        direct_query = (
            query_frames.to(torch.bfloat16)
            if direct_qk_bf16 and not direct_qk_bf16_fused
            else query_frames
        )
        direct_dense_key = (
            key_frames.to(torch.bfloat16)
            if direct_qk_bf16 and not direct_qk_bf16_fused
            else key_frames
        )
        direct_coarse_key = (
            coarse_key.to(torch.bfloat16)
            if direct_qk_bf16 and not direct_qk_bf16_fused
            else coarse_key
        )
        if len(residual_keys) == 1:
            packed_residual_key = residual_key
            packed_residual_value = residual_value
        else:
            packed_residual_key = torch.cat(
                residual_keys, dim=-2
            ).contiguous()
            packed_residual_value = torch.cat(
                residual_values, dim=-2
            ).contiguous()
        direct_residual_key = (
            packed_residual_key.to(torch.bfloat16)
            if direct_qk_bf16 and not direct_qk_bf16_fused
            else packed_residual_key
        )
        direct_special_key = (
            None
            if special_key is None
            else special_key.to(direct_query.dtype).contiguous()
        )
        if use_hps_flash_carrier:
            from sparse_vggt.kernels.hps_flash_carrier import (
                hps_flash_carrier_attention,
            )

            complement_indices = None
            if use_hps_complement_correction:
                remote_cell_indices = remote_parent_index_grid(
                    num_frames,
                    coarse_tokens_per_frame,
                    device=query.device,
                ).view(1, num_frames, -1).expand(batch, -1, -1)
                complement_mask = ~refined_map.gather(
                    2, remote_cell_indices
                )
                complement_indices = remote_cell_indices[
                    complement_mask
                ].view(batch, num_frames, -1)
                if direct_int32_indices:
                    complement_indices = complement_indices.to(
                        dtype=torch.int32
                    )
            output = hps_flash_carrier_attention(
                direct_query,
                direct_dense_key,
                value_frames,
                direct_coarse_key,
                coarse_value,
                direct_residual_key,
                packed_residual_value,
                kernel_residual_indices,
                complement_indices,
                direct_special_key,
                None if special_value is None else special_value.contiguous(),
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps or 4,
                num_stages=direct_num_stages or 3,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
                complement_correction=use_hps_complement_correction,
            )
        else:
            from sparse_vggt.kernels.direct_attention import (
                descriptor_direct_debt_attention,
            )

            output = descriptor_direct_debt_attention(
                direct_query,
                direct_dense_key,
                value_frames,
                direct_coarse_key,
                coarse_value,
                direct_residual_key,
                packed_residual_value,
                kernel_residual_indices,
                refined_map,
                direct_special_key,
                None if special_value is None else special_value.contiguous(),
                mass_conserving=mass_conserving_refinement,
                exact_mass_conserving=exact_mass_conserving_refinement,
                single_phase_layout=phase_compiled_layout,
                block_m=direct_block_m,
                block_n=direct_block_n,
                num_warps=direct_num_warps,
                num_stages=direct_num_stages,
                dot_qk_bf16=direct_qk_bf16_fused,
                output_dtype=(
                    query_frames.dtype if direct_native_output else None
                ),
            )
        if output.dtype != query_frames.dtype:
            output = output.to(query_frames.dtype)
        total_local_keys = local_pair_count * tokens_per_frame
        total_remote_keys = remote_pair_count * coarse_tokens_per_frame
        total_residual_keys = num_frames * residual_tokens_per_query
        total_exact_remote_keys = 0
        chunk_starts = ()
    else:
        output = None
        total_local_keys = 0
        total_remote_keys = 0
        total_residual_keys = 0
        total_exact_remote_keys = 0
        frame_ids = torch.arange(num_frames, device=query.device)
        chunk_starts = range(0, num_frames, query_frame_chunk)
    for chunk_start in chunk_starts:
        chunk_end = min(chunk_start + query_frame_chunk, num_frames)
        chunk_centers = frame_ids[chunk_start:chunk_end]
        is_local = (
            chunk_centers[:, None] - frame_ids[None, :]
        ).abs() <= local_frame_radius
        local_counts = is_local.sum(dim=-1)

        for local_count_tensor in torch.unique(local_counts):
            local_count = int(local_count_tensor.item())
            centers = chunk_centers[local_counts == local_count_tensor]
            center_mask = is_local[local_counts == local_count_tensor]
            group_size = centers.numel()
            expanded_frames = frame_ids.expand(group_size, num_frames)
            local_indices = expanded_frames[center_mask].view(
                group_size, local_count
            )
            remote_count = num_frames - local_count
            remote_indices = expanded_frames[~center_mask].view(
                group_size, remote_count
            )

            exact_key = key_frames[:, :, local_indices, :, :].reshape(
                batch,
                heads,
                group_size,
                local_count * tokens_per_frame,
                head_dim,
            )
            exact_value = value_frames[:, :, local_indices, :, :].reshape(
                batch,
                heads,
                group_size,
                local_count * tokens_per_frame,
                head_dim,
            )
            key_parts = [exact_key]
            value_parts = [exact_value]
            attention_bias_parts = []
            if mass_conserving_refinement:
                attention_bias_parts.append(
                    torch.zeros(
                        batch,
                        group_size,
                        local_count * tokens_per_frame,
                        device=query.device,
                        dtype=query.dtype,
                    )
                )
            if selection_granularity == "frame":
                selected_extra = extra_frame_indices[:, centers, :]
                selected_exact = selected_extra[
                    :, :, :exact_frames_per_query
                ]
                selected_phase = selected_extra[
                    :, :, exact_frames_per_query:
                ]
            else:
                selected_exact = None
                selected_phase = residual_cell_indices[:, centers, :]
                selected_phase_indices = [
                    indices[:, centers, :]
                    for indices in residual_phase_cell_indices
                ]
                selected_precision = precision_cell_indices[:, centers, :]
            mixed_remote_slots = 0
            additive_parent_slots = 0
            additive_child_slots = 0
            if remote_count > 0 and additive_parent_service:
                (
                    additive_remote_key,
                    additive_remote_value,
                    additive_remote_bias,
                    additive_parent_slots,
                    additive_child_slots,
                ) = additive_parent_service_kv(
                    routing_parent_key,
                    routing_parent_value,
                    coarse_key,
                    coarse_value,
                    residual_cell_indices,
                    substituted_parent_indices,
                    parent_to_children,
                    parent_child_valid,
                    centers,
                    child_masses=spatial_pool_multiplicity(
                        height,
                        width,
                        remote_pool_size,
                        device=query.device,
                        dtype=torch.float32,
                    ),
                    bias_dtype=query.dtype,
                )
                key_parts.append(additive_remote_key)
                value_parts.append(additive_remote_value)
                attention_bias_parts.append(additive_remote_bias)
            if remote_count > 0 and mixed_parent_execution:
                (
                    mixed_remote_key,
                    mixed_remote_value,
                    mixed_remote_bias,
                    mixed_residual_bias,
                    mixed_remote_slots,
                ) = mixed_parent_execution_kv(
                    routing_parent_key,
                    routing_parent_value,
                    routing_parent_valid,
                    coarse_key,
                    coarse_value,
                    mixed_parent_indices,
                    residual_cell_indices,
                    parent_to_children,
                    centers,
                    bias_dtype=query.dtype,
                )
                key_parts.append(mixed_remote_key)
                value_parts.append(mixed_remote_value)
                attention_bias_parts.append(mixed_remote_bias)
                refinement_bias_parts = [mixed_residual_bias]
            if (
                remote_count > 0
                and not mixed_parent_execution
                and not additive_parent_service
            ):
                coarse_mask = (~center_mask).unsqueeze(0).expand(
                    batch, -1, -1
                ).clone()
                if exact_frames_per_query > 0:
                    coarse_mask.scatter_(2, selected_exact, False)
                frame_grid = frame_ids.view(1, 1, num_frames).expand(
                    batch, group_size, -1
                )
                coarse_indices = frame_grid[coarse_mask].view(
                    batch,
                    group_size,
                    remote_count - exact_frames_per_query,
                )
                refinement_bias_parts = []
                if mass_conserving_refinement:
                    cell_offsets = torch.arange(
                        coarse_tokens_per_frame,
                        device=query.device,
                    ).view(1, 1, -1)
                    coarse_cell_indices = (
                        coarse_indices[..., None] * coarse_tokens_per_frame
                        + cell_offsets
                    ).reshape(batch, group_size, -1)
                    if selection_granularity == "frame":
                        selected_phase_cells = (
                            selected_phase[..., None]
                            * coarse_tokens_per_frame
                            + cell_offsets
                        ).reshape(batch, group_size, -1)
                        refinement_cell_indices = (
                            [selected_phase_cells]
                            if residual_tokens_per_query > 0
                            else []
                        )
                    else:
                        refinement_cell_indices = list(
                            selected_phase_indices
                        )
                        if precision_cells_per_query > 0:
                            refinement_cell_indices.append(
                                selected_precision
                            )
                    coarse_bias, refinement_bias_parts = (
                        refinement_multiplicity_log_bias(
                            coarse_cell_indices,
                            refinement_cell_indices,
                            total_cells=(
                                num_frames * coarse_tokens_per_frame
                            ),
                            dtype=query.dtype,
                        )
                    )
                if not use_triton_pack:
                    remote_key = gather_frame_tokens(
                        coarse_key, coarse_indices
                    ).reshape(
                        batch,
                        heads,
                        group_size,
                        (remote_count - exact_frames_per_query)
                        * coarse_tokens_per_frame,
                        head_dim,
                    )
                    remote_value = gather_frame_tokens(
                        coarse_value, coarse_indices
                    ).reshape_as(remote_key)
                    key_parts.append(remote_key)
                    value_parts.append(remote_value)
                if mass_conserving_refinement:
                    attention_bias_parts.append(coarse_bias)
            if exact_frames_per_query > 0:
                selected_exact_key = gather_frame_tokens(
                    key_frames, selected_exact
                ).reshape(
                    batch,
                    heads,
                    group_size,
                    exact_frames_per_query * tokens_per_frame,
                    head_dim,
                )
                selected_exact_value = gather_frame_tokens(
                    value_frames, selected_exact
                ).reshape_as(selected_exact_key)
                key_parts.append(selected_exact_key)
                value_parts.append(selected_exact_value)
                if mass_conserving_refinement:
                    attention_bias_parts.append(
                        torch.zeros(
                            batch,
                            group_size,
                            exact_frames_per_query * tokens_per_frame,
                            device=query.device,
                            dtype=query.dtype,
                        )
                    )
            if residual_tokens_per_query > 0 and not additive_parent_service:
                if not use_triton_pack:
                    if selection_granularity == "frame":
                        selected_residual_key = gather_frame_tokens(
                            residual_key, selected_phase
                        ).reshape(
                            batch,
                            heads,
                            group_size,
                            residual_tokens_per_query,
                            head_dim,
                        )
                        selected_residual_value = gather_frame_tokens(
                            residual_value, selected_phase
                        ).reshape_as(selected_residual_key)
                    else:
                        phase_indices = list(selected_phase_indices)
                        if precision_cells_per_query > 0:
                            phase_indices.append(selected_precision)
                        selected_residual_key = torch.cat(
                            [
                                gather_flat_frame_tokens(
                                    phase_key, phase_index
                                )
                                for phase_key, phase_index in zip(
                                    residual_keys, phase_indices
                                )
                            ],
                            dim=-2,
                        )
                        selected_residual_value = torch.cat(
                            [
                                gather_flat_frame_tokens(
                                    phase_value, phase_index
                                )
                                for phase_value, phase_index in zip(
                                    residual_values, phase_indices
                                )
                            ],
                            dim=-2,
                        )
                    key_parts.append(selected_residual_key)
                    value_parts.append(selected_residual_value)
                if mass_conserving_refinement:
                    attention_bias_parts.append(
                        torch.cat(refinement_bias_parts, dim=-1)
                    )
            if special_key is not None and special_key.shape[-2] > 0:
                special_tokens = special_key.shape[-2]
                if not use_triton_pack:
                    key_parts.append(
                        special_key.unsqueeze(2).expand(
                            batch,
                            heads,
                            group_size,
                            special_tokens,
                            head_dim,
                        )
                    )
                    value_parts.append(
                        special_value.unsqueeze(2).expand(
                            batch,
                            heads,
                            group_size,
                            special_tokens,
                            head_dim,
                        )
                    )
                if mass_conserving_refinement:
                    attention_bias_parts.append(
                        torch.zeros(
                            batch,
                            group_size,
                            special_tokens,
                            device=query.device,
                            dtype=query.dtype,
                        )
                    )

            selected_query = query_frames[:, :, centers, :, :]
            if use_triton_pack:
                from sparse_vggt.kernels.pack import triton_pack_debt_kv

                compact_key, compact_value = triton_pack_debt_kv(
                    key_frames,
                    value_frames,
                    coarse_key,
                    coarse_value,
                    residual_key,
                    residual_value,
                    centers,
                    coarse_indices,
                    selected_phase,
                    special_key,
                    special_value,
                )
            else:
                compact_key = torch.cat(key_parts, dim=-2)
                compact_value = torch.cat(value_parts, dim=-2)
            attention_bias = None
            if mass_conserving_refinement:
                attention_bias = torch.cat(
                    attention_bias_parts, dim=-1
                )[:, None, :, None, :]
            if use_grouped_sdpa:
                group_output = grouped_scaled_dot_product_attention(
                    selected_query,
                    compact_key,
                    compact_value,
                    attention_bias,
                )
            else:
                group_output = F.scaled_dot_product_attention(
                    selected_query,
                    compact_key,
                    compact_value,
                    attn_mask=attention_bias,
                )
            if output is None:
                output = torch.empty(
                    query_frames.shape,
                    device=query.device,
                    dtype=group_output.dtype,
                )
            output[:, :, centers, :, :] = group_output
            total_local_keys += group_size * local_count * tokens_per_frame
            if mixed_parent_execution:
                total_remote_keys += group_size * mixed_remote_slots
            elif additive_parent_service:
                total_remote_keys += group_size * additive_parent_slots
                total_residual_keys += group_size * additive_child_slots
            else:
                total_remote_keys += (
                    group_size
                    * (remote_count - exact_frames_per_query)
                    * coarse_tokens_per_frame
                )
            if not additive_parent_service:
                total_residual_keys += group_size * residual_tokens_per_query
            total_exact_remote_keys += (
                group_size * exact_frames_per_query * tokens_per_frame
            )

    if output is None:
        raise ValueError("num_frames must be positive")
    selected_patch_keys = (
        total_local_keys
        + total_remote_keys
        + total_residual_keys
        + total_exact_remote_keys
    )
    actual_sparsity = 1.0 - selected_patch_keys / dense_patch_keys
    special_tokens = 0 if special_key is None else special_key.shape[-2]
    mean_patch_keys = selected_patch_keys / num_frames
    effective_fraction = (
        mean_patch_keys + special_tokens
    ) / (num_frames * tokens_per_frame + special_tokens)
    stats = {
        "residual_budget_patch_sparsity": actual_sparsity,
        "residual_budget_effective_sparsity": 1.0 - effective_fraction,
        "residual_budget_target_sparsity": float(target_sparsity),
        "residual_budget_grouped_sdpa": float(use_grouped_sdpa),
        "residual_budget_direct_attention": float(any_direct_attention),
        "residual_budget_additive_direct_attention": float(
            use_additive_direct_attention
        ),
        "residual_budget_parent_mask_attention": float(
            use_parent_mask_additive_attention
        ),
        "residual_budget_compiled_carrier_attention": float(
            use_compiled_carrier_attention
        ),
        "residual_budget_compiled_carrier_value_bf16": float(
            compiled_carrier_value_bf16
        ),
        "residual_budget_compiled_carrier_precast_value_bf16": float(
            compiled_carrier_precast_value_bf16
        ),
        "residual_budget_compiled_carrier_dense_block_n": float(
            compiled_carrier_dense_block_n
        ),
        "residual_budget_compiled_carrier_parent_block_n": float(
            compiled_carrier_parent_block_n
        ),
        "residual_budget_compiled_carrier_child_block_n": float(
            compiled_carrier_child_block_n
        ),
        "residual_budget_static_geometry_cache": float(
            _env_flag("SPARSE_VGGT_STATIC_GEOMETRY_CACHE", default=False)
        ),
        "residual_budget_additive_flash_carrier": float(
            use_additive_flash_carrier
        ),
        "residual_budget_additive_flash_positive_groups": float(
            use_additive_flash_positive_groups
        ),
        "residual_budget_two_stage_qk_pv": float(use_two_stage_qk_pv),
        "residual_budget_two_stage_pv_log2_threshold": float(
            two_stage_pv_log2_threshold
        ),
        "residual_budget_collect_two_stage_qk_pv": float(
            collect_two_stage_qk_pv
        ),
        "residual_budget_pv_service_debt_feedback": float(
            pv_service_debt_feedback
        ),
        "residual_budget_pv_alignment_observer": float(
            pv_importance_alignment_observer
        ),
        "residual_budget_carrier_compensation_residual_observer": float(
            carrier_compensation_residual_observer
        ),
        "residual_budget_carrier_compensated_pv": float(
            carrier_compensated_pv
        ),
        "residual_budget_hps_complement_correction": float(
            use_hps_complement_correction
        ),
        "residual_budget_hps_selected_cell_fraction": float(
            residual_tokens_per_query / max(max_remote_cells, 1)
        ),
        "residual_budget_mixed_direct_attention": float(
            use_mixed_direct_attention
        ),
        "residual_budget_mixed_flash_parent": float(
            use_mixed_flash_parent
        ),
        "residual_budget_mixed_flash_self": float(
            use_mixed_flash_self
        ),
        "residual_budget_direct_int32_indices": float(
            direct_int32_indices
        ),
        "residual_budget_direct_tile_profile_id": float(
            direct_tile_profile == "speed"
        ),
        "residual_budget_block_aligned_residual": float(
            block_aligned_residual
        ),
        "residual_budget_direct_qk_bf16": float(direct_qk_bf16),
        "residual_budget_direct_qk_bf16_fused": float(
            direct_qk_bf16_fused
        ),
        "residual_budget_phase_compiled_layout": float(
            phase_compiled_layout
        ),
        "residual_budget_hps_flash_carrier": float(
            use_hps_flash_carrier
        ),
        "residual_budget_direct_block_m": float(direct_block_m),
        "residual_budget_direct_block_n": float(direct_block_n),
        "residual_budget_triton_pack": float(use_triton_pack),
        "residual_budget_mass_conserving_refinement": float(
            mass_conserving_refinement
        ),
        "residual_budget_exact_mass_conserving_refinement": float(
            exact_mass_conserving_refinement
        ),
        "residual_budget_budget_error": actual_sparsity - target_sparsity,
        "residual_budget_budget_feasible": float(budget_feasible),
        "residual_budget_budget_floor_retained_fraction": (
            budget_floor_retained_fraction
        ),
        "residual_budget_budget_floor_sparsity": (
            1.0 - budget_floor_retained_fraction
        ),
        "residual_budget_protected_pair_fraction": (
            local_pair_count / (num_frames * num_frames)
        ),
        "residual_budget_protected_dense_fraction_of_dense": (
            total_local_keys / dense_patch_keys
        ),
        "residual_budget_remote_coarse_fraction_of_dense": (
            total_remote_keys / dense_patch_keys
        ),
        "residual_budget_residual_fraction_of_dense": (
            total_residual_keys / dense_patch_keys
        ),
        "residual_budget_local_exact_fraction": (
            total_local_keys / selected_patch_keys
        ),
        "residual_budget_coarse_fraction": (
            total_remote_keys / selected_patch_keys
        ),
        "residual_budget_exact_fraction": (
            total_exact_remote_keys / selected_patch_keys
        ),
        "residual_budget_residual_fraction": (
            total_residual_keys / selected_patch_keys
        ),
        "residual_budget_mean_patch_keys": mean_patch_keys,
        "residual_budget_coarse_tokens_per_frame": float(
            coarse_tokens_per_frame
        ),
        "residual_budget_routing_parent_size": float(routing_parent_size),
        "residual_budget_grouped_parent_routing": float(
            use_grouped_parent_routing
        ),
        "residual_budget_mixed_parent_execution": float(
            mixed_parent_execution
        ),
        "residual_budget_additive_parent_service": float(
            additive_parent_service
        ),
        "residual_budget_additive_full_upgrade_fraction": float(
            additive_full_upgrade_fraction
        ),
        "residual_budget_mixed_residuals_per_parent_config": float(
            mixed_residuals_per_parent
        ),
        "residual_budget_adaptive_parent_service": float(
            adaptive_parent_service
        ),
        "residual_budget_parent_hard_fraction_config": float(
            parent_hard_fraction
        ),
        "residual_budget_parent_cost_power": float(parent_cost_power),
        "residual_budget_frame_balance_fraction": float(
            frame_balance_fraction
        ),
        "residual_budget_routing_phase_mode_id": float({
            "fixed": 0,
            "rotating": 1,
            "anchored": 2,
            "anchored_heads": 3,
        }[routing_phase_mode]),
        "residual_budget_local_radius": float(local_frame_radius),
        "residual_budget_exact_frames_per_query": float(
            exact_frames_per_query
        ),
        "residual_budget_phase_frames_per_query": float(
            phase_frames_per_query
        ),
        "residual_budget_residual_tokens_per_query": float(
            residual_tokens_per_query
        ),
        "residual_budget_selection_granularity_id": float(
            selection_granularity == "cell"
        ),
        "residual_budget_frame_detail_power": float(frame_detail_power),
        "residual_budget_spatial_detail_power": float(
            spatial_detail_power
        ),
        "residual_budget_cell_refinement_phases": float(
            cell_refinement_phases
        ),
        "residual_budget_cell_precision_fraction": float(
            cell_precision_fraction
        ),
        "residual_budget_precision_cells_per_query": float(
            precision_cells_per_query
        ),
        "residual_budget_selected_precision_importance": float(
            precision_importance if selection_granularity == "cell" else 0.0
        ),
        "residual_budget_cell_importance_weight": float(
            cell_importance_weight
        ),
        "residual_budget_exact_budget_fraction": float(
            exact_budget_fraction
        ),
        **debt_stats,
    }
    return output.reshape(batch, heads, token_count, head_dim), stats


def dense_radial_a_attention_forward(
    self,
    q_reordered,
    k_reordered,
    v_reordered,
    B,
    N,
    P,
    S,
    H,
    W,
    nh,
    hd,
    hidden_dim,
    aux_output_store=None,
):
    """Dense validation path: radial mask on A, dense B/C/D.

    This is intentionally not an acceleration path. It lets us test whether an
    A-only radial mask hurts downstream quality before touching sparse kernels.
    """
    patch_tokens_per_frame = H * W
    patch_total = N * patch_tokens_per_frame
    num_tokens = q_reordered.shape[-2]

    block_size = int(os.environ.get("SPARSE_VGGT_RADIAL_BLOCK_SIZE", "128"))
    decay_factor = float(os.environ.get("SPARSE_VGGT_RADIAL_DECAY_FACTOR", "1.0"))
    dense_neighbor = int(os.environ.get("SPARSE_VGGT_RADIAL_DENSE_NEIGHBOR", "1"))

    score = (q_reordered * self.scale) @ k_reordered.transpose(-2, -1)

    a_mask = build_radial_a_mask(
        num_frames=N,
        patch_tokens_per_frame=patch_tokens_per_frame,
        device=score.device,
        block_size=block_size,
        decay_factor=decay_factor,
        dense_neighbor=dense_neighbor,
    )
    full_mask = torch.ones((num_tokens, num_tokens), device=score.device, dtype=torch.bool)
    full_mask[:patch_total, :patch_total] = a_mask
    score = score.masked_fill(~full_mask, torch.finfo(score.dtype).min)
    attn = score.softmax(dim=-1)

    out = attn @ v_reordered
    out = restore_to_frame_major(out, N, P, S)
    out = out.view(B, nh, N * P, hd)
    out = out.transpose(1, 2).reshape(B, N * P, hidden_dim)
    out = self.proj(out)
    out = self.proj_drop(out)

    if aux_output_store is not None:
        aux_output_store["radial_a_dense"] = {
            "block_size": block_size,
            "decay_factor": decay_factor,
            "dense_neighbor": dense_neighbor,
            "a_keep_ratio": a_mask.float().mean().item(),
        }

    return out


def chunked_radial_a_attention_forward(
    self,
    q_reordered,
    k_reordered,
    v_reordered,
    B,
    N,
    P,
    S,
    H,
    W,
    nh,
    hd,
    hidden_dim,
    aux_output_store=None,
):
    """Chunked validation path that avoids materializing the full A matrix.

    Patch queries attend to radial patch-key windows plus all special keys.
    Special queries still use dense attention over all keys.
    """
    patch_tokens_per_frame = H * W
    patch_total = N * patch_tokens_per_frame

    block_size = int(os.environ.get("SPARSE_VGGT_RADIAL_BLOCK_SIZE", "128"))
    decay_factor = float(os.environ.get("SPARSE_VGGT_RADIAL_DECAY_FACTOR", "1.0"))
    dense_neighbor = int(os.environ.get("SPARSE_VGGT_RADIAL_DENSE_NEIGHBOR", "1"))
    query_chunk = int(os.environ.get("SPARSE_VGGT_RADIAL_QUERY_CHUNK", "128"))
    use_hilbert = _env_flag("SPARSE_VGGT_RADIAL_USE_HILBERT", default=False)

    q_patch = q_reordered[..., :patch_total, :]
    k_patch = k_reordered[..., :patch_total, :]
    v_patch = v_reordered[..., :patch_total, :]
    k_special = k_reordered[..., patch_total:, :]
    v_special = v_reordered[..., patch_total:, :]

    if use_hilbert:
        q_patch = hilbert_patch_reorder(q_patch, H, W)
        k_patch = hilbert_patch_reorder(k_patch, H, W)
        v_patch = hilbert_patch_reorder(v_patch, H, W)

    out_patch = torch.empty_like(q_patch)
    scale = self.scale
    kept_entries = 0
    total_a_entries = patch_total * patch_total

    for query_frame in range(N):
        query_frame_start = query_frame * patch_tokens_per_frame
        query_frame_end = query_frame_start + patch_tokens_per_frame

        for query_start_local in range(0, patch_tokens_per_frame, query_chunk):
            query_end_local = min(query_start_local + query_chunk, patch_tokens_per_frame)
            query_start = query_frame_start + query_start_local
            query_end = query_frame_start + query_end_local
            q_chunk = q_patch[..., query_start:query_end, :]

            score_parts = []
            value_parts = []
            mask_parts = []
            local_query = torch.arange(
                query_start_local,
                query_end_local,
                device=q_reordered.device,
            )

            for key_frame in range(N):
                width = radial_width(
                    query_frame,
                    key_frame,
                    token_per_frame=patch_tokens_per_frame,
                    block_size=block_size,
                    decay_factor=decay_factor,
                    dense_neighbor=dense_neighbor,
                )

                key_start_local = max(0, query_start_local - width)
                key_end_local = min(patch_tokens_per_frame, query_end_local + width)
                key_start = key_frame * patch_tokens_per_frame + key_start_local
                key_end = key_frame * patch_tokens_per_frame + key_end_local

                k_local = k_patch[..., key_start:key_end, :]
                v_local = v_patch[..., key_start:key_end, :]
                score = (q_chunk * scale) @ k_local.transpose(-2, -1)

                local_key = torch.arange(
                    key_start_local,
                    key_end_local,
                    device=q_reordered.device,
                )
                mask = (local_query[:, None] - local_key[None, :]).abs() <= width
                kept_entries += int(mask.sum().item())

                score_parts.append(score)
                value_parts.append(v_local)
                mask_parts.append(mask)

            if S > 0:
                score_parts.append((q_chunk * scale) @ k_special.transpose(-2, -1))
                value_parts.append(v_special)
                mask_parts.append(
                    torch.ones(
                        (query_end - query_start, N * S),
                        device=q_reordered.device,
                        dtype=torch.bool,
                    )
                )

            score = torch.cat(score_parts, dim=-1)
            value = torch.cat(value_parts, dim=-2)
            mask = torch.cat(mask_parts, dim=-1)
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            attn = score.softmax(dim=-1)
            out_patch[..., query_start:query_end, :] = attn @ value

    if use_hilbert:
        out_patch = hilbert_patch_reorder(out_patch, H, W, inverse=True)

    if S > 0:
        q_special = q_reordered[..., patch_total:, :]
        out_special = F.scaled_dot_product_attention(q_special, k_reordered, v_reordered)
        out_reordered = torch.cat([out_patch, out_special], dim=-2)
    else:
        out_reordered = out_patch

    out = restore_to_frame_major(out_reordered, N, P, S)
    out = out.view(B, nh, N * P, hd)
    out = out.transpose(1, 2).reshape(B, N * P, hidden_dim)
    out = self.proj(out)
    out = self.proj_drop(out)

    if aux_output_store is not None:
        aux_output_store["radial_a_chunked"] = {
            "block_size": block_size,
            "decay_factor": decay_factor,
            "dense_neighbor": dense_neighbor,
            "query_chunk": query_chunk,
            "use_hilbert": use_hilbert,
            "a_keep_ratio": kept_entries / total_a_entries,
        }

    return out


def adaptive_sparse_attention_forward(
    self,
    x,
    pos,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    pool_mode: str = "avg",
    aux_sparsity_only: bool = True,
    aux_output_store: dict | None = None,
    num_special_tokens: int = 5,
    num_heads: int = 16,
    # Radial + layer-wise fusion parameters
    use_radial_layerwise: bool = False,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
    layer_idx: int = 0,
    layer_sparsity_ratios: list | None = None,
    use_distance_routed: bool = False,
    route_frame_threshold: int = 4,
    use_covariance_aware_importance: bool = False,
    covariance_weight: float = 0.5,
    covariance_eps: float = 1e-8,
    use_adaptive_slit_routing: bool = False,
    adaptive_slit_temporal_window: int = 10,
    adaptive_slit_stable_quantile: float = 0.6,
    adaptive_slit_change_quantile: float = 0.7,
    adaptive_slit_narrow_width: int = 1,
    adaptive_slit_base_width: int = 2,
    adaptive_slit_expand_width: int = 4,
    use_soft_geometry_routing: bool = False,
    use_preview_adaptive_routing: bool = False,
    use_multiresolution_routing: bool = False,
    use_residual_budget_routing: bool = False,
    multiresolution_local_radius: int = 2,
    multiresolution_remote_pool_size: int = 2,
    multiresolution_query_frame_chunk: int = 4,
    multiresolution_remote_mode: str = "strided",
    multiresolution_remote_samples_per_cell: int = 1,
    multiresolution_remote_phase_mode: str = "layer",
    multiresolution_area_bias: bool = False,
    residual_budget_target_sparsity: float = 0.70,
    residual_budget_local_radius: int = 0,
    residual_budget_remote_pool_size: int = 2,
    residual_budget_routing_parent_size: int = 2,
    residual_budget_routing_phase_mode: str = "rotating",
    residual_budget_unified_incremental_service: bool = False,
    residual_budget_bounded_wait_reobservation: bool = False,
    residual_budget_adaptive_parent_service: bool = False,
    residual_budget_mixed_parent_execution: bool = False,
    residual_budget_additive_parent_service: bool = False,
    residual_budget_additive_full_upgrade_fraction: float = 0.0,
    residual_budget_additive_parent_substitution_fraction: float = 0.0,
    residual_budget_mixed_residuals_per_parent: int = 4,
    residual_budget_parent_hard_fraction: float = 0.25,
    residual_budget_parent_cost_power: float = 1.0,
    residual_budget_frame_balance_fraction: float = 0.0,
    residual_budget_fine_refresh: bool = False,
    residual_budget_query_frame_chunk: int = 4,
    residual_budget_momentum: float = 0.75,
    residual_budget_service_conditioned_momentum: bool = False,
    residual_budget_repayment: float = 1.0,
    residual_budget_repayment_mode: str = "reset",
    residual_budget_service_credit_scale: float = 1.0,
    residual_budget_temperature: float = 1.0,
    residual_budget_exact_fraction: float = 0.0,
    residual_budget_surface_weight: float = 0.0,
    residual_budget_selection_granularity: str = "frame",
    residual_budget_frame_detail_power: float = 1.0,
    residual_budget_spatial_detail_power: float = 1.0,
    residual_budget_cell_scorer: str = "carrier_residual",
    residual_budget_carrier_residual_balance: float = 0.5,
    residual_budget_cell_refinement_phases: int = 1,
    residual_budget_cell_precision_fraction: float = 0.0,
    residual_budget_mass_conserving_refinement: bool = False,
    residual_budget_exact_mass_conserving_refinement: bool = False,
    residual_budget_cell_importance_weight: float = 0.0,
    residual_budget_cell_importance_floor: float = 0.0,
    residual_budget_cell_importance_gate_threshold: float = 0.0,
    residual_budget_cell_importance_gate_temperature: float = 0.05,
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
    analyze_scout_oracle_routing: bool = False,
    scout_oracle_layers: tuple[int, ...] = (0, 8, 15, 23),
    scout_oracle_query_blocks: int = 32,
    scout_oracle_queries_per_block: int = 4,
    scout_oracle_local_radius: int = 1,
    scout_oracle_coarse_group_blocks: int = 4,
    scout_oracle_candidate_multiplier: float = 2.0,
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
    analyze_importance_blocks: bool = False,
    analyze_block_selection: bool = False,
    routing_state: dict | None = None,
):
    """Adaptive Block Sparse Attention forward pass to replace the original attention forward function.

    Args:
        x: (B, N * P, hidden_dim) == (B, N * (H * W + S), hidden_dim)
        pos: Position embeddings
        sparse_ratio: Sparsity ratio for block selection
        cdf_threshold: CDF threshold for block selection
        pool_mode: Pooling mode for attention prediction ("avg" or "max")
        aux_sparsity_only: If True, only store sparsity in aux_output_store
        aux_output_store: Dictionary to store auxiliary outputs
        num_special_tokens: Number of special tokens per frame
        num_heads: Number of attention heads
        use_radial_layerwise: If True, use fused radial+layer-wise+importance mask
        decay_factor: Radial decay factor (for radial constraint)
        dense_neighbor: Number of neighboring frames with full attention
        layer_idx: Current layer index (for layer-wise sparsity)
        layer_sparsity_ratios: List of per-layer sparsity ratios (optional)
        use_distance_routed: Route near frame pairs to radial and far pairs to importance
        route_frame_threshold: Maximum frame distance handled by radial attention
        use_covariance_aware_importance: Add K-block variance to far importance scores
        use_adaptive_slit_routing: Replace near radial routing with adaptive local slit blocks

    Return:
        x: (B, N * P, hidden_dim)
    """

    if routing_state is None:
        routing_state = getattr(self, "_sparse_routing_state", None)

    B, NP, hidden_dim = x.shape
    S = num_special_tokens
    nh = num_heads
    hd = hidden_dim // num_heads

    # Infer H, W and N from pos
    H = int(pos[0].max(0).values[0])
    W = int(pos[0].max(0).values[1])
    N = NP // (H * W + S)
    P = H * W + S

    # Sanity check
    assert N * (H * W + S) == NP, f"{N=}, {H=}, {W=}, {S=}, {NP=}"

    qkv = self.qkv(x)
    three = 3
    qkv = rearrange(qkv, "B N (three nh hd)-> B N three nh hd", three=three, nh=nh, hd=hd)
    qkv = rearrange(qkv, "B N three nh hd -> three B nh N hd")

    q, k, v = qkv.unbind(0)  # (B, num_heads, N * P, head_dim)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # Make the global-attention layout explicit for experimentation:
    # [all patch tokens, all special tokens].
    radial_a_active = _env_flag(
        "SPARSE_VGGT_RADIAL_A_DENSE", default=False
    ) or _env_flag("SPARSE_VGGT_RADIAL_A_CHUNKED", default=False)
    reuse_patch_extraction = _env_flag(
        "SPARSE_VGGT_REUSE_PATCH_EXTRACTION", default=False
    )
    direct_patch_special_pack = _env_flag(
        "SPARSE_VGGT_DIRECT_PATCH_SPECIAL_PACK", default=False
    )
    direct_value_pack = _env_flag(
        "SPARSE_VGGT_DIRECT_VALUE_PACK", default=False
    )
    fused_qkv_layout = _env_flag(
        "SPARSE_VGGT_FUSED_QKV_LAYOUT", default=False
    )
    skip_query_reorder = (
        _env_flag("SPARSE_VGGT_SKIP_QUERY_REORDER", default=False)
        and not radial_a_active
    )
    q_special_packed = None
    if fused_qkv_layout:
        if not reuse_patch_extraction or radial_a_active:
            raise ValueError(
                "fused QKV layout requires patch reuse and non-radial attention"
            )
        (
            q_patch,
            q_special_packed,
            k_patch,
            k_reordered,
            v_patch,
            v_reordered,
        ) = pack_qkv_patch_then_special(q, k, v, N, P, S)
        q_reordered = None
    elif reuse_patch_extraction:
        if skip_query_reorder:
            q_patch = get_patch_tokens(q, N, P, S)
            q_reordered = None
        else:
            q_patch, q_reordered = reorder_to_patch_then_special_with_patch(
                q, N, P, S
            )
        pack_key_tokens = (
            pack_patch_then_special_direct_with_patch
            if direct_patch_special_pack
            else reorder_to_patch_then_special_with_patch
        )
        pack_value_tokens = (
            pack_patch_then_special_direct_with_patch
            if direct_patch_special_pack or direct_value_pack
            else reorder_to_patch_then_special_with_patch
        )
        k_patch, k_reordered = pack_key_tokens(k, N, P, S)
        v_patch, v_reordered = pack_value_tokens(v, N, P, S)
    else:
        q_reordered = (
            None
            if skip_query_reorder
            else reorder_to_patch_then_special(q, N, P, S)
        )
        k_reordered = reorder_to_patch_then_special(k, N, P, S)
        v_reordered = reorder_to_patch_then_special(v, N, P, S)

    if _env_flag("SPARSE_VGGT_RADIAL_A_DENSE", default=False):
        return dense_radial_a_attention_forward(
            self,
            q_reordered=q_reordered,
            k_reordered=k_reordered,
            v_reordered=v_reordered,
            B=B,
            N=N,
            P=P,
            S=S,
            H=H,
            W=W,
            nh=nh,
            hd=hd,
            hidden_dim=hidden_dim,
            aux_output_store=aux_output_store,
        )

    if _env_flag("SPARSE_VGGT_RADIAL_A_CHUNKED", default=False):
        return chunked_radial_a_attention_forward(
            self,
            q_reordered=q_reordered,
            k_reordered=k_reordered,
            v_reordered=v_reordered,
            B=B,
            N=N,
            P=P,
            S=S,
            H=H,
            W=W,
            nh=nh,
            hd=hd,
            hidden_dim=hidden_dim,
            aux_output_store=aux_output_store,
        )

    # separate patch and special tokens
    patch_token_count = N * (P - S)
    if not reuse_patch_extraction:
        q_patch = get_patch_tokens(q, N, P, S)
        k_patch = get_patch_tokens(k, N, P, S)
        v_patch = get_patch_tokens(v, N, P, S)
    q_special = (
        q_special_packed
        if q_special_packed is not None
        else (
            get_special_tokens(q, N, P, S)
            if skip_query_reorder
            else q_reordered[..., patch_token_count:, :]
        )
    )

    # special tokens attend to all tokens
    if q_special is not None:
        x_special = F.scaled_dot_product_attention(q_special, k_reordered, v_reordered)
    else:
        x_special = None
    # release memory unless needed
    if aux_sparsity_only:
        del q, k, v, qkv

    # In reordered layout, the keys/values already follow:
    # [all patch tokens, all special tokens].
    key = k_reordered
    value = v_reordered

    if self.training:
        raise NotImplementedError("This is currently only training-free. Use .eval()")

    if use_residual_budget_routing:
        special_key = (
            k_reordered[..., patch_token_count:, :] if S > 0 else None
        )
        special_value = (
            v_reordered[..., patch_token_count:, :] if S > 0 else None
        )
        x_patch, residual_budget_stats = residual_budget_patch_attention(
            q_patch,
            k_patch,
            v_patch,
            num_frames=N,
            height=H,
            width=W,
            target_sparsity=residual_budget_target_sparsity,
            local_frame_radius=residual_budget_local_radius,
            remote_pool_size=residual_budget_remote_pool_size,
            routing_parent_size=residual_budget_routing_parent_size,
            routing_phase_mode=residual_budget_routing_phase_mode,
            unified_incremental_service=(
                residual_budget_unified_incremental_service
            ),
            bounded_wait_reobservation=(
                residual_budget_bounded_wait_reobservation
            ),
            adaptive_parent_service=(
                residual_budget_adaptive_parent_service
            ),
            mixed_parent_execution=(
                residual_budget_mixed_parent_execution
            ),
            additive_parent_service=(
                residual_budget_additive_parent_service
            ),
            additive_full_upgrade_fraction=(
                residual_budget_additive_full_upgrade_fraction
            ),
            additive_parent_substitution_fraction=(
                residual_budget_additive_parent_substitution_fraction
            ),
            mixed_residuals_per_parent=(
                residual_budget_mixed_residuals_per_parent
            ),
            parent_hard_fraction=(
                residual_budget_parent_hard_fraction
            ),
            parent_cost_power=residual_budget_parent_cost_power,
            frame_balance_fraction=(
                residual_budget_frame_balance_fraction
            ),
            query_frame_chunk=residual_budget_query_frame_chunk,
            layer_idx=layer_idx,
            total_layers=dual_path_num_layers,
            momentum=residual_budget_momentum,
            service_conditioned_momentum=(
                residual_budget_service_conditioned_momentum
            ),
            repayment=residual_budget_repayment,
            repayment_mode=residual_budget_repayment_mode,
            service_credit_scale=(
                residual_budget_service_credit_scale
            ),
            temperature=residual_budget_temperature,
            exact_budget_fraction=residual_budget_exact_fraction,
            routing_state=routing_state,
            surface_weight=residual_budget_surface_weight,
            selection_granularity=(
                residual_budget_selection_granularity
            ),
            frame_detail_power=residual_budget_frame_detail_power,
            spatial_detail_power=residual_budget_spatial_detail_power,
            cell_scorer=residual_budget_cell_scorer,
            carrier_residual_balance=(
                residual_budget_carrier_residual_balance
            ),
            cell_refinement_phases=(
                residual_budget_cell_refinement_phases
            ),
            cell_precision_fraction=(
                residual_budget_cell_precision_fraction
            ),
            mass_conserving_refinement=(
                residual_budget_mass_conserving_refinement
            ),
            exact_mass_conserving_refinement=(
                residual_budget_exact_mass_conserving_refinement
            ),
            cell_importance_weight=(
                residual_budget_cell_importance_weight
            ),
            cell_importance_floor=(
                residual_budget_cell_importance_floor
            ),
            cell_importance_gate_threshold=(
                residual_budget_cell_importance_gate_threshold
            ),
            cell_importance_gate_temperature=(
                residual_budget_cell_importance_gate_temperature
            ),
            special_query=q_special,
            special_key=special_key,
            special_value=special_value,
        )
        transported_snapshot = residual_budget_stats.pop(
            "_transported_debt_observation_snapshot", None
        )
        if isinstance(transported_snapshot, dict):
            frame_ids = transported_snapshot["query_frame_ids"]
            patch_ids = transported_snapshot["query_patch_ids"]
            sparse_sample = (
                x_patch.view(B, nh, N, H * W, hd)
                .index_select(2, frame_ids)
                .index_select(3, patch_ids)
            )
            dense_sample = transported_snapshot["dense_output"].to(
                sparse_sample
            )
            residual = (dense_sample - sparse_sample).permute(
                0, 2, 3, 1, 4
            ).reshape(B, frame_ids.numel(), patch_ids.numel(), nh * hd)
            projected_residual = F.linear(
                residual.to(self.proj.weight.dtype),
                self.proj.weight,
                bias=None,
            ).float()
            residual_budget_stats.update(
                analyze_transported_representation_debt(
                    projected_residual,
                    transported_snapshot,
                    layer_idx=layer_idx,
                    routing_state=routing_state,
                )
            )
        if aux_output_store is not None:
            residual_budget_stats["residual_budget_fine_refresh"] = float(
                residual_budget_fine_refresh
            )
            aux_output_store.update(residual_budget_stats)
            aux_output_store["sparsity"] = residual_budget_stats[
                "residual_budget_patch_sparsity"
            ]

        x_patch = rearrange(
            x_patch,
            "B nh (N H W) hd -> B nh N (H W) hd",
            N=N,
            H=H,
            W=W,
        )
        if x_special is not None:
            x_special = x_special.view(B, nh, N * S, hd)
            x_reordered = torch.cat(
                [x_patch.reshape(B, nh, N * H * W, hd), x_special],
                dim=-2,
            )
        else:
            x_reordered = x_patch.reshape(B, nh, N * H * W, hd)
        x = restore_to_frame_major(x_reordered, N, P, S)
        x = x.view(B, nh, N * P, hd)
        x = x.transpose(1, 2).reshape(B, N * P, nh * hd)
        x = self.proj(x)
        return self.proj_drop(x)

    if use_multiresolution_routing:
        special_key = (
            k_reordered[..., patch_token_count:, :] if S > 0 else None
        )
        special_value = (
            v_reordered[..., patch_token_count:, :] if S > 0 else None
        )
        x_patch, multiresolution_stats = multiresolution_patch_attention(
            q_patch,
            k_patch,
            v_patch,
            num_frames=N,
            height=H,
            width=W,
            local_frame_radius=multiresolution_local_radius,
            remote_pool_size=multiresolution_remote_pool_size,
            query_frame_chunk=multiresolution_query_frame_chunk,
            remote_mode=multiresolution_remote_mode,
            remote_phase=layer_idx,
            remote_samples_per_cell=(
                multiresolution_remote_samples_per_cell
            ),
            remote_phase_mode=multiresolution_remote_phase_mode,
            use_area_bias=multiresolution_area_bias,
            special_key=special_key,
            special_value=special_value,
        )
        if aux_output_store is not None:
            aux_output_store.update(multiresolution_stats)
            aux_output_store["sparsity"] = multiresolution_stats[
                "multiresolution_patch_sparsity"
            ]

        x_patch = rearrange(
            x_patch,
            "B nh (N H W) hd -> B nh N (H W) hd",
            N=N,
            H=H,
            W=W,
        )
        if x_special is not None:
            x_special = x_special.view(B, nh, N * S, hd)
            x_reordered = torch.cat(
                [x_patch.reshape(B, nh, N * H * W, hd), x_special],
                dim=-2,
            )
        else:
            x_reordered = x_patch.reshape(B, nh, N * H * W, hd)
        x = restore_to_frame_major(x_reordered, N, P, S)
        x = x.view(B, nh, N * P, hd)
        x = x.transpose(1, 2).reshape(B, N * P, nh * hd)
        x = self.proj(x)
        return self.proj_drop(x)

    else:
        if use_covariance_aware_importance:
            attn_pooled, key_block_variance = predict_attention(
                query=q_patch,
                key=k_patch,
                pool_mode=pool_mode,
                return_key_block_variance=True,
            )
        else:
            attn_pooled = predict_attention(
                query=q_patch, key=k_patch, pool_mode=pool_mode
            )
            key_block_variance = None
        if (
            analyze_scout_oracle_routing
            and layer_idx in scout_oracle_layers
            and sparse_ratio is not None
        ):
            scout_metrics = analyze_scout_oracle(
                query=q_patch,
                key=k_patch,
                value=v_patch,
                pooled_score=attn_pooled,
                num_frames=N,
                tokens_per_frame=H * W,
                sparse_ratio=sparse_ratio,
                local_frame_radius=scout_oracle_local_radius,
                max_query_blocks=scout_oracle_query_blocks,
                queries_per_block=scout_oracle_queries_per_block,
                coarse_group_blocks=scout_oracle_coarse_group_blocks,
                candidate_multiplier=scout_oracle_candidate_multiplier,
            )
            if aux_output_store is not None:
                aux_output_store.update(scout_metrics)
        exact_risk_oracle_layers = tuple(
            int(item)
            for item in os.environ.get(
                "SPARSE_VGGT_COSA_EXACT_RISK_ORACLE_LAYERS",
                "0,7,15,23",
            ).split(",")
            if item.strip()
        )
        exact_risk_oracle_active = (
            _env_flag("SPARSE_VGGT_COSA_EXACT_RISK_ORACLE")
            and layer_idx in exact_risk_oracle_layers
        )
        postcut_order_layers = tuple(
            int(item)
            for item in os.environ.get(
                "SPARSE_VGGT_COSA_POSTCUT_ORDER_OBSERVER_LAYERS",
                "0,7,15,23",
            ).split(",")
            if item.strip()
        )
        postcut_order_active = (
            _env_flag("SPARSE_VGGT_COSA_POSTCUT_ORDER_OBSERVER")
            and layer_idx in postcut_order_layers
        )
        if not exact_risk_oracle_active and not postcut_order_active:
            # The diagnostic retains patch K/V through sparse execution only
            # on sampled layers; the production path keeps its old lifetime.
            del k_patch, v_patch

        # patch attention
        x_patch, sparsity = block_sparse_attn_cuda(
            query=q_patch,
            key=key,
            value=value,
            pooled_score=attn_pooled,
            sparse_ratio=sparse_ratio,
            cdf_threshold=cdf_threshold,
            return_sparsity=True,
            use_radial_layerwise=use_radial_layerwise,
            num_frames=N,
            tokens_per_frame=H * W,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            layer_idx=layer_idx,
            layer_sparsity_ratios=layer_sparsity_ratios,
            use_distance_routed=use_distance_routed,
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
            use_soft_geometry_routing=use_soft_geometry_routing,
            use_preview_adaptive_routing=use_preview_adaptive_routing,
            preview_redistribution_fraction=preview_redistribution_fraction,
            preview_activation_threshold=preview_activation_threshold,
            preview_local_radius=preview_local_radius,
            preview_local_fraction=preview_local_fraction,
            preview_geometry_weight=preview_geometry_weight,
            preview_head_protection=preview_head_protection,
            preview_protected_head_fraction=preview_protected_head_fraction,
            preview_donor_retention_threshold=(
                preview_donor_retention_threshold
            ),
            preview_donor_exchange_scope=preview_donor_exchange_scope,
            preview_receiver_gain_threshold=(
                preview_receiver_gain_threshold
            ),
            preview_exchange_gain_cost_ratio=(
                preview_exchange_gain_cost_ratio
            ),
            preview_head_exchange_cap_fraction=(
                preview_head_exchange_cap_fraction
            ),
            preview_layer_confidence_threshold=(
                preview_layer_confidence_threshold
            ),
            preview_exchange_layer_start=preview_exchange_layer_start,
            preview_exchange_layer_end=preview_exchange_layer_end,
            use_dual_path_routing=use_dual_path_routing,
            dual_path_local_radius=dual_path_local_radius,
            dual_path_min_local_fraction=dual_path_min_local_fraction,
            dual_path_max_local_fraction=dual_path_max_local_fraction,
            dual_path_layer_schedule=dual_path_layer_schedule,
            dual_path_num_layers=dual_path_num_layers,
            dual_path_context_geometry_weight=dual_path_context_geometry_weight,
            dual_path_context_schedule=dual_path_context_schedule,
            dual_path_context_gate=dual_path_context_gate,
            dual_path_context_alignment_threshold=(
                dual_path_context_alignment_threshold
            ),
            dual_path_context_alignment_temperature=(
                dual_path_context_alignment_temperature
            ),
            use_layerwise_hybrid_routing=use_layerwise_hybrid_routing,
            use_local_layerwise_hybrid_routing=use_local_layerwise_hybrid_routing,
            use_coverage_layerwise_routing=use_coverage_layerwise_routing,
            use_persistent_view_graph_routing=use_persistent_view_graph_routing,
            hybrid_early_end=hybrid_early_end,
            hybrid_mid_end=hybrid_mid_end,
            hybrid_mid_sparse_ratio=hybrid_mid_sparse_ratio,
            hybrid_late_sparse_ratio=hybrid_late_sparse_ratio,
            hybrid_early_dense_neighbor=hybrid_early_dense_neighbor,
            hybrid_early_local_radius=hybrid_early_local_radius,
            mid_frame_coverage_ratio=mid_frame_coverage_ratio,
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
            routing_stats_store=(
                None
                if _env_flag("SPARSE_VGGT_SKIP_ROUTING_STATS", default=False)
                else aux_output_store
            ),
            routing_state=routing_state,
            analyze_importance_blocks=analyze_importance_blocks,
            analyze_block_selection=analyze_block_selection,
            num_patch_tokens=patch_token_count,
        )

        endpoint_key = "cosa_exact_pair_risk_endpoint_supports"
        endpoint_supports = (
            routing_state.pop(endpoint_key, None)
            if routing_state is not None
            else None
        )
        postcut_key = "cosa_postcut_order_observer_snapshot"
        postcut_snapshot = (
            routing_state.pop(postcut_key, None)
            if routing_state is not None
            else None
        )
        if exact_risk_oracle_active:
            if endpoint_supports is None:
                raise RuntimeError(
                    "exact CoSA risk oracle requires Current/Debt supports"
                )
            if endpoint_supports["layer_idx"] != layer_idx:
                raise RuntimeError("stale CoSA exact-risk endpoint supports")
            exact_risk_metrics = analyze_cosa_exact_pair_risk(
                query=q_patch,
                key=k_patch,
                value=v_patch,
                current_support=endpoint_supports["current"],
                debt_support=endpoint_supports["debt"],
                max_query_blocks=int(
                    os.environ.get(
                        "SPARSE_VGGT_COSA_EXACT_RISK_ORACLE_QUERY_BLOCKS",
                        "16",
                    )
                ),
                queries_per_block=int(
                    os.environ.get(
                        "SPARSE_VGGT_COSA_EXACT_RISK_ORACLE_QUERIES_PER_BLOCK",
                        "2",
                    )
                ),
            )
            if aux_output_store is not None:
                aux_output_store.update(exact_risk_metrics)
        if postcut_order_active:
            if postcut_snapshot is None:
                raise RuntimeError(
                    "CoSA post-cut observer requires the frozen support snapshot"
                )
            if postcut_snapshot["layer_idx"] != layer_idx:
                raise RuntimeError("stale CoSA post-cut observer snapshot")
            postcut_metrics = analyze_cosa_postcut_orders(
                query=q_patch,
                key=k_patch,
                value=v_patch,
                support=postcut_snapshot["support"],
                pooled_score=postcut_snapshot["pooled_score"],
                debt_priority=postcut_snapshot["debt_priority"],
                max_query_blocks=int(
                    os.environ.get(
                        "SPARSE_VGGT_COSA_POSTCUT_ORDER_QUERY_BLOCKS", "8"
                    )
                ),
                queries_per_block=int(
                    os.environ.get(
                        "SPARSE_VGGT_COSA_POSTCUT_ORDER_QUERIES_PER_BLOCK", "4"
                    )
                ),
                proxy_key_stride=int(
                    os.environ.get(
                        "SPARSE_VGGT_COSA_POSTCUT_ORDER_PROXY_KEY_STRIDE", "8"
                    )
                ),
            )
            if aux_output_store is not None:
                aux_output_store.update(postcut_metrics)
        if exact_risk_oracle_active or postcut_order_active:
            del k_patch
        if exact_risk_oracle_active or postcut_order_active:
            del v_patch

    if aux_output_store is not None:
        aux_output_store["sparsity"] = sparsity

        if not aux_sparsity_only:
            aux_output_store.update(
                {
                    "attn_pooled": attn_pooled,
                    "query": q,
                    "key": k,
                    "shape": {
                        "B": B,
                        "N": N,
                        "P": P,
                        "head_dim": hd,
                        "num_heads": nh,
                        "H": H,
                        "W": W,
                    },
                }
            )

    x_patch = rearrange(
        x_patch,
        "B nh (N H W) hd -> B nh N (H W) hd",
        N=N,
        H=H,
        W=W,
    )

    # combine patch and special tokens
    if _env_flag("SPARSE_VGGT_DIRECT_FRAME_MAJOR_RESTORE", default=False):
        if x_special is not None:
            x_special = x_special.view(B, nh, N * S, hd)
        x = combine_patch_and_special_frame_major(
            x_patch,
            x_special,
            N,
            P,
            S,
        )
    else:
        if x_special is not None:
            x_special = x_special.view(B, nh, N * S, hd)
            x_reordered = torch.cat(
                [x_patch.reshape(B, nh, N * H * W, hd), x_special],
                dim=-2,
            )
            x = restore_to_frame_major(x_reordered, N, P, S)
        else:
            x = restore_to_frame_major(
                x_patch.reshape(B, nh, N * H * W, hd),
                N,
                P,
                S,
            )

    x = x.view(B, nh, N * P, hd)
    x = x.transpose(1, 2).reshape(B, N * P, nh * hd)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x
