"""Sampled dense-oracle diagnostics for hierarchical debt routing.

The observation path is deliberately read-only. It compares the routing
signals and the service decision against dense attention mass on a fixed,
deterministic sample of query frames and spatial queries. Nothing returned by
this module is fed back into attention.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

import torch
import torch.nn.functional as F


_TRUE_VALUES = {"1", "true", "yes", "on"}


def service_observation_enabled(layer_idx: int | None = None) -> bool:
    enabled = os.environ.get(
        "SPARSE_VGGT_SERVICE_OBSERVATION", "0"
    ).strip().lower()
    if enabled not in _TRUE_VALUES:
        return False
    if layer_idx is None:
        return True
    layer_spec = os.environ.get(
        "SPARSE_VGGT_SERVICE_OBSERVATION_LAYERS", "all"
    ).strip().lower()
    if layer_spec in {"", "all", "*"}:
        return True
    try:
        return layer_idx in {
            int(value.strip())
            for value in layer_spec.split(",")
            if value.strip()
        }
    except ValueError as error:
        raise ValueError(
            "SPARSE_VGGT_SERVICE_OBSERVATION_LAYERS must be 'all' or "
            "a comma-separated integer list"
        ) from error


def transported_debt_observation_enabled(
    layer_idx: int | None = None,
) -> bool:
    enabled = os.environ.get(
        "SPARSE_VGGT_TRANSPORTED_DEBT_OBSERVER", "0"
    ).strip().lower()
    return enabled in _TRUE_VALUES and service_observation_enabled(layer_idx)


def service_observation_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _even_indices(count: int, requested: int, device: torch.device) -> torch.Tensor:
    sample_count = min(count, requested)
    if sample_count == count:
        return torch.arange(count, device=device)
    return torch.linspace(
        0, count - 1, steps=sample_count, device=device
    ).round().long().unique()


def _spatial_grid_indices(
    height: int,
    width: int,
    requested: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a deterministic 2D-stratified patch sample."""
    requested = min(height * width, requested)
    row_count = min(
        height,
        max(1, round(math.sqrt(requested * height / max(width, 1)))),
    )
    column_count = min(width, max(1, math.ceil(requested / row_count)))
    while row_count * column_count < requested:
        if column_count < width:
            column_count += 1
        elif row_count < height:
            row_count += 1
        else:
            break
    rows = _even_indices(height, row_count, device)
    columns = _even_indices(width, column_count, device)
    indices = (rows[:, None] * width + columns[None, :]).flatten()
    if indices.numel() > requested:
        keep = _even_indices(indices.numel(), requested, device)
        indices = indices.index_select(0, keep)
    return indices.unique(sorted=True)


def _aggregate_parent_mass(
    patch_mass: torch.Tensor,
    *,
    height: int,
    width: int,
    parent_size: int,
) -> torch.Tensor:
    """Sum [B, K, H*W] patch mass into row-major parent cells."""
    batch, key_frames, patch_count = patch_mass.shape
    if patch_count != height * width:
        raise ValueError("patch mass does not match the spatial grid")
    parent_rows = math.ceil(height / parent_size)
    parent_cols = math.ceil(width / parent_size)
    padded_height = parent_rows * parent_size
    padded_width = parent_cols * parent_size
    grid = patch_mass.new_zeros(
        batch, key_frames, padded_height, padded_width
    )
    grid[:, :, :height, :width] = patch_mass.view(
        batch, key_frames, height, width
    )
    return (
        grid.view(
            batch,
            key_frames,
            parent_rows,
            parent_size,
            parent_cols,
            parent_size,
        )
        .sum(dim=(3, 5))
        .flatten(start_dim=-2)
    )


def _spatial_parent_patch_map(
    height: int,
    width: int,
    parent_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map row-major parent cells to their fine patch indices."""
    parent_rows = math.ceil(height / parent_size)
    parent_cols = math.ceil(width / parent_size)
    children = []
    valid = []
    for parent_row in range(parent_rows):
        for parent_col in range(parent_cols):
            parent_children = []
            parent_valid = []
            for row_offset in range(parent_size):
                for column_offset in range(parent_size):
                    row = parent_row * parent_size + row_offset
                    column = parent_col * parent_size + column_offset
                    is_valid = row < height and column < width
                    parent_children.append(
                        row * width + column if is_valid else 0
                    )
                    parent_valid.append(is_valid)
            children.append(parent_children)
            valid.append(parent_valid)
    return (
        torch.tensor(children, device=device, dtype=torch.long),
        torch.tensor(valid, device=device, dtype=torch.bool),
    )


def _softmax_probability_interval(
    lower_logits: torch.Tensor,
    upper_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate independent logit intervals through softmax monotonically."""
    if lower_logits.shape != upper_logits.shape:
        raise ValueError("softmax interval bounds must share a shape")
    if lower_logits.shape[-1] < 2:
        ones = torch.ones_like(lower_logits)
        return ones, ones

    upper_lse = torch.logsumexp(upper_logits, dim=-1, keepdim=True)
    lower_lse = torch.logsumexp(lower_logits, dim=-1, keepdim=True)
    upper_share = (upper_logits - upper_lse).exp().clamp_max(1.0 - 1e-7)
    lower_share = (lower_logits - lower_lse).exp().clamp_max(1.0 - 1e-7)
    other_upper_lse = upper_lse + torch.log1p(-upper_share)
    other_lower_lse = lower_lse + torch.log1p(-lower_share)
    return (
        torch.sigmoid(lower_logits - other_upper_lse),
        torch.sigmoid(upper_logits - other_lower_lse),
    )


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().flatten()
    right = right.float().flatten()
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.square().sum().sqrt() * right.square().sum().sqrt()
    if float(denominator.item()) <= 1e-12:
        return 0.0
    return float((left * right).sum().div(denominator).item())


def _ranks(value: torch.Tensor) -> torch.Tensor:
    """Average ranks with exact tie handling."""
    value = value.flatten()
    if value.numel() == 0:
        return value.float()
    order = value.argsort(stable=True)
    sorted_value = value.index_select(0, order)
    _, counts = torch.unique_consecutive(
        sorted_value, return_counts=True
    )
    starts = counts.cumsum(0) - counts
    average = starts.float() + (counts.float() - 1.0) / 2.0
    sorted_ranks = torch.repeat_interleave(average, counts)
    ranks = torch.empty_like(value, dtype=torch.float32)
    ranks.scatter_(0, order, sorted_ranks)
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() < 2:
        return 0.0
    return _pearson(_ranks(left.float().flatten()), _ranks(right.float().flatten()))


def _masked_pair_metric(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
    metric,
) -> float:
    valid = valid.expand_as(left)
    if int(valid.sum().item()) < 2:
        return 0.0
    return metric(left.masked_select(valid), right.masked_select(valid))


def _topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(valid)
    k = min(k, int(valid.sum(dim=-1).min().item()))
    if k <= 0:
        return torch.zeros_like(valid)
    indices = scores.masked_fill(~valid, float("-inf")).topk(k, dim=-1).indices
    result = torch.zeros_like(valid)
    result.scatter_(-1, indices, True)
    return result


def _topk_mask_counts(
    scores: torch.Tensor,
    valid: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Top-K mask with a separate exact K for every leading row."""
    if scores.shape != valid.shape or scores.shape[:-1] != counts.shape:
        raise ValueError("scores, validity, and row counts must align")
    counts = counts.to(device=scores.device, dtype=torch.long).clamp_min(0)
    valid_counts = valid.sum(dim=-1)
    if bool((counts > valid_counts).any()):
        raise ValueError("row Top-K count exceeds valid candidates")
    order = scores.masked_fill(~valid, float("-inf")).argsort(
        dim=-1, descending=True, stable=True
    )
    selected_rank = (
        torch.arange(scores.shape[-1], device=scores.device)
        < counts[..., None]
    )
    result = torch.zeros_like(valid)
    result.scatter_(-1, order, selected_rank)
    return result


def _rowwise_metric_values(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
    metric,
) -> torch.Tensor:
    """Evaluate a scalar metric independently for every leading row."""
    if left.shape != right.shape or left.shape != valid.shape:
        raise ValueError("rowwise metric tensors must share one shape")
    width = left.shape[-1]
    values = []
    for left_row, right_row, valid_row in zip(
        left.reshape(-1, width),
        right.reshape(-1, width),
        valid.reshape(-1, width),
    ):
        if int(valid_row.sum().item()) >= 2:
            values.append(metric(left_row[valid_row], right_row[valid_row]))
    return left.new_tensor(values, dtype=torch.float32)


def _distribution_stats(value: torch.Tensor) -> dict[str, float]:
    """Deterministic row-distribution summary for uncertainty reporting."""
    value = value.float().flatten()
    if value.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "p10": 0.0,
            "min": 0.0,
        }
    mean_value = value.mean()
    std_value = value.std(unbiased=False)
    se_value = std_value / math.sqrt(value.numel())
    half_width = 1.96 * se_value
    return {
        "mean": float(mean_value.item()),
        "std": float(std_value.item()),
        "se": float(se_value.item()),
        "ci95_low": float((mean_value - half_width).item()),
        "ci95_high": float((mean_value + half_width).item()),
        "p10": float(torch.quantile(value, 0.10).item()),
        "min": float(value.min().item()),
    }


def _mean_fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = numerator.float() / denominator.float().clamp_min(1.0)
    return float(value.mean().item())


def _normalize_over_valid(
    value: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    valid_weight = valid.to(value.dtype)
    count = valid_weight.sum(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    mean_value = (
        (value * valid_weight).sum(dim=(-2, -1), keepdim=True) / count
    )
    return (value / mean_value.clamp_min(1e-12)).masked_fill(~valid, 0.0)


@torch.no_grad()
def analyze_grouped_service_observation(
    query_frames: torch.Tensor,
    key_frames: torch.Tensor,
    snapshot: dict[str, Any],
    *,
    value_frames: torch.Tensor | None = None,
    height: int,
    width: int,
    parent_size: int,
    local_frame_radius: int,
    layer_idx: int,
    routing_state: dict | None,
    special_key: torch.Tensor | None = None,
    special_value: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compare need, debt, and service with sampled dense attention mass.

    Dense attention probability mass aggregated into each physical parent cell
    is the oracle. Multiplying that mass by the observed parent value-detail
    proxy yields an output-repair gain proxy. Both use the same fixed query
    sample at every layer so cross-layer persistence is measurable.
    """
    started = time.perf_counter()
    if query_frames.shape != key_frames.shape or query_frames.ndim != 5:
        raise ValueError("query/key frames must share [B, H, F, P, D] shape")
    batch, heads, num_frames, patches_per_frame, head_dim = query_frames.shape
    parents_per_frame = snapshot["current_need"].shape[-1]
    expected_parents = math.ceil(height / parent_size) * math.ceil(
        width / parent_size
    )
    if parents_per_frame != expected_parents:
        raise ValueError("scheduler parent count does not match the spatial grid")

    max_query_frames = service_observation_int(
        "SPARSE_VGGT_SERVICE_OBSERVATION_QUERY_FRAMES", 100
    )
    queries_per_frame = service_observation_int(
        "SPARSE_VGGT_SERVICE_OBSERVATION_QUERIES_PER_FRAME", 8
    )
    query_frame_ids = _even_indices(
        num_frames, max_query_frames, query_frames.device
    )
    query_patch_ids = _spatial_grid_indices(
        height, width, queries_per_frame, query_frames.device
    )

    observe_transported_debt = transported_debt_observation_enabled(layer_idx)
    if observe_transported_debt:
        if value_frames is None or value_frames.shape != query_frames.shape:
            raise ValueError(
                "transported debt observation requires matching value frames"
            )
        if (special_key is None) != (special_value is None):
            raise ValueError("special key/value must either both be set or omitted")

    oracle_mass_rows = []
    oracle_child_mass_rows = []
    dense_output_rows = []
    if parent_size % 2 != 0:
        raise ValueError("service observation requires an even parent size")
    child_size = parent_size // 2
    scale = head_dim ** -0.5
    flat_key = key_frames.flatten(start_dim=2, end_dim=3).float()
    flat_value = (
        value_frames.flatten(start_dim=2, end_dim=3).float()
        if observe_transported_debt else None
    )
    dense_key = flat_key
    dense_value = flat_value
    if observe_transported_debt and special_key is not None:
        dense_key = torch.cat((dense_key, special_key.float()), dim=-2)
        dense_value = torch.cat(
            (dense_value, special_value.float()), dim=-2
        )
    for query_frame_id in query_frame_ids.tolist():
        sampled_query = query_frames[
            :, :, query_frame_id, query_patch_ids
        ].float()
        logits = torch.matmul(
            sampled_query, flat_key.transpose(-1, -2)
        ) * scale
        probability = logits.softmax(dim=-1).view(
            batch,
            heads,
            query_patch_ids.numel(),
            num_frames,
            patches_per_frame,
        )
        if observe_transported_debt:
            dense_logits = torch.matmul(
                sampled_query, dense_key.transpose(-1, -2)
            ) * scale
            dense_probability = dense_logits.softmax(dim=-1)
            dense_output_rows.append(
                torch.matmul(dense_probability, dense_value)
            )
        patch_mass = probability.mean(dim=(1, 2))
        oracle_mass_rows.append(
            _aggregate_parent_mass(
                patch_mass,
                height=height,
                width=width,
                parent_size=parent_size,
            )
        )
        oracle_child_mass_rows.append(
            _aggregate_parent_mass(
                patch_mass,
                height=height,
                width=width,
                parent_size=child_size,
            )
        )
    oracle_mass = torch.stack(oracle_mass_rows, dim=1)
    oracle_child_mass = torch.stack(oracle_child_mass_rows, dim=1)

    current_need = snapshot["current_need"].index_select(
        1, query_frame_ids
    ).float()
    debt = snapshot["debt"].index_select(1, query_frame_ids).float()
    parent_detail = snapshot["parent_detail"].float()
    oracle_gain = oracle_mass * parent_detail[:, None]

    selected_service = torch.zeros_like(snapshot["debt"])
    parent_indices = snapshot["parent_indices"].long()
    served_counts = snapshot["served_counts"].float()
    if parent_indices.shape[-1] > 0:
        selected_service = selected_service.flatten(start_dim=-2)
        selected_service.scatter_add_(-1, parent_indices, served_counts)
        selected_service = selected_service.view_as(snapshot["debt"])
    selected_service = selected_service.index_select(1, query_frame_ids)
    selected = selected_service > 0

    key_frame_ids = torch.arange(num_frames, device=query_frames.device)
    remote_frame = (
        query_frame_ids[:, None] - key_frame_ids[None, :]
    ).abs() > local_frame_radius
    valid = remote_frame.view(
        1, query_frame_ids.numel(), num_frames, 1
    ).expand(batch, -1, -1, parents_per_frame)
    flat_valid = valid.flatten(start_dim=-2)
    flat_mass = oracle_mass.flatten(start_dim=-2)
    flat_gain = oracle_gain.flatten(start_dim=-2)
    flat_current = current_need.flatten(start_dim=-2)
    flat_debt = debt.flatten(start_dim=-2)
    flat_selected = selected.flatten(start_dim=-2)

    active_counts = flat_selected.sum(dim=-1)
    topk = max(1, round(float(active_counts.float().mean().item())))
    oracle_top = _topk_mask(flat_gain, flat_valid, topk)
    current_top = _topk_mask(flat_current, flat_valid, topk)
    debt_top = _topk_mask(flat_debt, flat_valid, topk)

    oracle_mass_total = flat_mass.masked_fill(~flat_valid, 0.0).sum(dim=-1)
    oracle_gain_total = flat_gain.masked_fill(~flat_valid, 0.0).sum(dim=-1)
    selected_mass = flat_mass.masked_fill(~flat_selected, 0.0).sum(dim=-1)
    selected_gain = flat_gain.masked_fill(~flat_selected, 0.0).sum(dim=-1)
    oracle_top_count = oracle_top.sum(dim=-1)

    state = routing_state if routing_state is not None else {}
    normalized_oracle_gain = _normalize_over_valid(oracle_gain, valid)
    oracle_backlog_key = "_service_observation_oracle_backlog"
    previous_oracle_backlog = state.get(oracle_backlog_key)
    if (
        previous_oracle_backlog is None
        or previous_oracle_backlog.shape != normalized_oracle_gain.shape
    ):
        previous_oracle_backlog = torch.zeros_like(normalized_oracle_gain)
    else:
        previous_oracle_backlog = previous_oracle_backlog.to(
            normalized_oracle_gain
        )
    momentum = float(snapshot.get("momentum", 0.75))
    repayment = float(snapshot.get("repayment", 1.0))
    service_credit_scale = float(
        snapshot.get("service_credit_scale", 1.0)
    )
    oracle_backlog = (
        momentum * previous_oracle_backlog + normalized_oracle_gain
    ).masked_fill(~valid, 0.0)
    current_oracle_demand = normalized_oracle_gain.sum(dim=(-2, -1))
    service_credit = snapshot.get("service_credit")
    if service_credit is None:
        sampled_service_credit = selected_service
    else:
        sampled_service_credit = service_credit.index_select(
            1, query_frame_ids
        ).float()
    sampled_service_credit = sampled_service_credit.masked_fill(~valid, 0.0)
    credit_total = sampled_service_credit.sum(
        dim=(-2, -1), keepdim=True
    )
    fallback_credit = selected_service.masked_fill(~valid, 0.0)
    fallback_total = fallback_credit.sum(
        dim=(-2, -1), keepdim=True
    )
    credit_pattern = torch.where(
        credit_total > 1e-12,
        sampled_service_credit / credit_total.clamp_min(1e-12),
        fallback_credit / fallback_total.clamp_min(1e-12),
    )
    oracle_credit = current_oracle_demand[..., None, None] * credit_pattern
    next_oracle_backlog = (
        oracle_backlog
        - repayment * service_credit_scale * oracle_credit
    ).clamp_min(0.0).masked_fill(~valid, 0.0)
    state[oracle_backlog_key] = next_oracle_backlog.detach()

    flat_oracle_backlog = oracle_backlog.flatten(start_dim=-2)
    oracle_backlog_top = _topk_mask(
        flat_oracle_backlog, flat_valid, topk
    )
    oracle_backlog_top_count = oracle_backlog_top.sum(dim=-1)
    current_backlog_spearman = _masked_pair_metric(
        flat_current, flat_oracle_backlog, flat_valid, _spearman
    )
    debt_backlog_spearman = _masked_pair_metric(
        flat_debt, flat_oracle_backlog, flat_valid, _spearman
    )
    oracle_backlog_total = flat_oracle_backlog.sum(dim=-1)
    selected_backlog = flat_oracle_backlog.masked_fill(
        ~flat_selected, 0.0
    ).sum(dim=-1)

    current_policy_gain = flat_gain.masked_fill(
        ~current_top, 0.0
    ).sum(dim=-1).mean()
    actual_policy_gain = selected_gain.mean()
    cumulative_key = "_service_observation_cumulative_gain"
    cumulative = state.get(cumulative_key, {})
    cumulative_actual = float(cumulative.get("actual", 0.0)) + float(
        actual_policy_gain.item()
    )
    cumulative_current = float(cumulative.get("current", 0.0)) + float(
        current_policy_gain.item()
    )
    state[cumulative_key] = {
        "actual": cumulative_actual,
        "current": cumulative_current,
    }

    age_key = "_service_observation_wait_age"
    previous_age = state.get(age_key)
    full_remote = snapshot["remote_mask"].bool()
    full_selected = torch.zeros_like(full_remote)
    if parent_indices.shape[-1] > 0:
        full_selected = full_selected.flatten(start_dim=-2)
        full_selected.scatter_(
            -1, parent_indices, served_counts > 0
        )
        full_selected = full_selected.view_as(full_remote)

    # Keep an event-time ledger over the exact parent cells that were served.
    # This state is observer-only: it never feeds back into routing or kernels.
    event_count_key = "_service_observation_event_count"
    event_count = int(state.get(event_count_key, 0)) + 1
    state[event_count_key] = event_count
    service_count_key = "_service_observation_service_count"
    previous_service_count = state.get(service_count_key)
    if (
        previous_service_count is None
        or previous_service_count.shape != full_remote.shape
    ):
        previous_service_count = torch.zeros_like(
            snapshot["debt"], dtype=torch.float32
        )
    else:
        previous_service_count = previous_service_count.to(
            snapshot["debt"]
        ).float()
    service_count = previous_service_count + full_selected.float()
    state[service_count_key] = service_count.detach()

    sampled_previous_service_count = previous_service_count.index_select(
        1, query_frame_ids
    ).flatten(start_dim=-2)
    sampled_service_count = service_count.index_select(
        1, query_frame_ids
    ).flatten(start_dim=-2)
    repeated_selected = flat_selected & (sampled_previous_service_count > 0)
    first_selected = flat_selected & ~repeated_selected
    unique_served = sampled_service_count > 0

    valid_service_count = sampled_service_count.masked_fill(~flat_valid, 0.0)
    service_total = valid_service_count.sum(dim=-1)
    service_square_total = valid_service_count.square().sum(dim=-1)
    valid_parent_count = flat_valid.sum(dim=-1).float().clamp_min(1.0)
    service_hhi = torch.where(
        service_total > 0,
        service_square_total / service_total.square().clamp_min(1e-12),
        torch.zeros_like(service_total),
    )
    normalized_service_hhi = service_hhi * valid_parent_count
    effective_service_fraction = torch.where(
        normalized_service_hhi > 0,
        normalized_service_hhi.reciprocal(),
        torch.zeros_like(normalized_service_hhi),
    )

    selected_count = flat_selected.sum(dim=-1).float()
    repeated_count = repeated_selected.sum(dim=-1).float()
    first_count = first_selected.sum(dim=-1).float()
    repeated_gain = flat_gain.masked_fill(~repeated_selected, 0.0).sum(dim=-1)
    first_gain = flat_gain.masked_fill(~first_selected, 0.0).sum(dim=-1)
    repeated_backlog = flat_oracle_backlog.masked_fill(
        ~repeated_selected, 0.0
    ).sum(dim=-1)
    first_backlog = flat_oracle_backlog.masked_fill(
        ~first_selected, 0.0
    ).sum(dim=-1)

    debt_only_selected = flat_selected & debt_top & ~current_top
    current_only_selected = flat_selected & current_top & ~debt_top
    shared_endpoint_selected = flat_selected & current_top & debt_top
    debt_only_gain = flat_gain.masked_fill(
        ~debt_only_selected, 0.0
    ).sum(dim=-1)
    selected_gain_safe = selected_gain.clamp_min(1e-12)
    debt_only_count = debt_only_selected.sum(dim=-1).float()
    if previous_age is None or previous_age.shape != full_remote.shape:
        previous_age = torch.zeros_like(snapshot["debt"], dtype=torch.float32)
    else:
        previous_age = previous_age.to(snapshot["debt"]).float()
    wait_age = torch.where(full_remote, previous_age + 1.0, 0.0)
    wait_age = wait_age.masked_fill(full_selected, 0.0)
    state[age_key] = wait_age.detach()
    sampled_wait = wait_age.index_select(1, query_frame_ids).flatten(start_dim=-2)

    coverage_key = "_service_observation_coverage"
    previous_coverage = state.get(coverage_key)
    if previous_coverage is None or previous_coverage.shape != full_remote.shape:
        previous_coverage = torch.zeros_like(full_remote)
    coverage = previous_coverage | full_selected
    state[coverage_key] = coverage.detach()
    sampled_coverage = coverage.index_select(1, query_frame_ids).flatten(
        start_dim=-2
    )

    current_oracle_spearman = _masked_pair_metric(
        flat_current, flat_gain, flat_valid, _spearman
    )
    debt_oracle_spearman = _masked_pair_metric(
        flat_debt, flat_gain, flat_valid, _spearman
    )
    gain_recall_by_frame = (
        selected_gain / oracle_gain_total.clamp_min(1e-12)
    )
    backlog_recall_by_frame = (
        selected_backlog / oracle_backlog_total.clamp_min(1e-12)
    )
    starvation_by_frame = (
        (oracle_top & ~flat_selected).sum(dim=-1).float()
        / oracle_top_count.float().clamp_min(1.0)
    )
    coverage_by_frame = (
        sampled_coverage.masked_fill(~flat_valid, False).float().sum(dim=-1)
        / flat_valid.sum(dim=-1).float().clamp_min(1.0)
    )
    oracle_top_wait = sampled_wait.masked_select(oracle_top)

    metrics = {
        "service_obs_schema_version": 6.0,
        "service_obs_layer": float(layer_idx),
        "service_obs_total_frames": float(num_frames),
        "service_obs_sampled_query_frames": float(query_frame_ids.numel()),
        "service_obs_query_frame_coverage": float(
            query_frame_ids.numel() / max(num_frames, 1)
        ),
        "service_obs_queries_per_frame": float(query_patch_ids.numel()),
        "service_obs_oracle_mass_mean": float(
            flat_mass.masked_select(flat_valid).mean().item()
        ),
        "service_obs_oracle_gain_mean": float(
            flat_gain.masked_select(flat_valid).mean().item()
        ),
        "service_obs_current_oracle_pearson": _masked_pair_metric(
            flat_current, flat_gain, flat_valid, _pearson
        ),
        "service_obs_current_oracle_spearman": current_oracle_spearman,
        "service_obs_debt_oracle_pearson": _masked_pair_metric(
            flat_debt, flat_gain, flat_valid, _pearson
        ),
        "service_obs_debt_oracle_spearman": debt_oracle_spearman,
        "service_obs_debt_rank_gain": (
            debt_oracle_spearman - current_oracle_spearman
        ),
        "service_obs_current_backlog_spearman": (
            current_backlog_spearman
        ),
        "service_obs_debt_backlog_spearman": debt_backlog_spearman,
        "service_obs_debt_backlog_rank_gain": (
            debt_backlog_spearman - current_backlog_spearman
        ),
        "service_obs_current_backlog_topk_recall": _mean_fraction(
            (current_top & oracle_backlog_top).sum(dim=-1),
            oracle_backlog_top_count,
        ),
        "service_obs_debt_backlog_topk_recall": _mean_fraction(
            (debt_top & oracle_backlog_top).sum(dim=-1),
            oracle_backlog_top_count,
        ),
        "service_obs_selection_backlog_topk_recall": _mean_fraction(
            (flat_selected & oracle_backlog_top).sum(dim=-1),
            oracle_backlog_top_count,
        ),
        "service_obs_selection_oracle_backlog_recall": float(
            (
                selected_backlog
                / oracle_backlog_total.clamp_min(1e-12)
            ).mean().item()
        ),
        "service_obs_oracle_backlog_repaid_fraction": float(
            (
                (
                    oracle_backlog.sum(dim=(-2, -1))
                    - next_oracle_backlog.sum(dim=(-2, -1))
                )
                / oracle_backlog.sum(dim=(-2, -1)).clamp_min(1e-12)
            ).mean().item()
        ),
        "service_obs_cumulative_actual_gain": cumulative_actual,
        "service_obs_cumulative_current_policy_gain": cumulative_current,
        "service_obs_cumulative_regret_vs_current": (
            (cumulative_current - cumulative_actual)
            / max(cumulative_current, 1e-12)
        ),
        "service_obs_current_topk_recall": _mean_fraction(
            (current_top & oracle_top).sum(dim=-1), oracle_top_count
        ),
        "service_obs_debt_topk_recall": _mean_fraction(
            (debt_top & oracle_top).sum(dim=-1), oracle_top_count
        ),
        "service_obs_selection_topk_recall": _mean_fraction(
            (flat_selected & oracle_top).sum(dim=-1), oracle_top_count
        ),
        "service_obs_selection_oracle_mass_recall": float(
            (selected_mass / oracle_mass_total.clamp_min(1e-12)).mean().item()
        ),
        "service_obs_selection_oracle_gain_recall": float(
            gain_recall_by_frame.mean().item()
        ),
        "service_obs_frame_gain_recall_p10": float(
            torch.quantile(gain_recall_by_frame, 0.10).item()
        ),
        "service_obs_frame_gain_recall_min": float(
            gain_recall_by_frame.min().item()
        ),
        "service_obs_frame_gain_recall_cv": float(
            (
                gain_recall_by_frame.std(unbiased=False)
                / gain_recall_by_frame.mean().clamp_min(1e-12)
            ).item()
        ),
        "service_obs_frame_backlog_recall_p10": float(
            torch.quantile(backlog_recall_by_frame, 0.10).item()
        ),
        "service_obs_oracle_top_starvation": _mean_fraction(
            (oracle_top & ~flat_selected).sum(dim=-1), oracle_top_count
        ),
        "service_obs_oracle_top_starvation_p90": float(
            torch.quantile(starvation_by_frame, 0.90).item()
        ),
        "service_obs_oracle_top_wait_mean": float(
            oracle_top_wait.mean().item()
        ),
        "service_obs_oracle_top_wait_p95": float(
            torch.quantile(oracle_top_wait, 0.95).item()
        ),
        "service_obs_oracle_top_wait_max": float(
            oracle_top_wait.max().item()
        ),
        "service_obs_selected_wait_before_service": float(
            previous_age.index_select(1, query_frame_ids)
            .flatten(start_dim=-2)
            .masked_select(flat_selected)
            .mean()
            .item()
        ) if bool(flat_selected.any()) else 0.0,
        "service_obs_cumulative_parent_coverage": float(
            sampled_coverage.masked_select(flat_valid).float().mean().item()
        ),
        "service_obs_cumulative_frame_coverage_p10": float(
            torch.quantile(coverage_by_frame, 0.10).item()
        ),
        "service_obs_cumulative_frame_coverage_min": float(
            coverage_by_frame.min().item()
        ),
        "service_obs_ledger_event_count": float(event_count),
        "service_obs_ledger_repeat_selection_fraction": float(
            (repeated_count / selected_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_first_selection_fraction": float(
            (first_count / selected_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_unique_parent_coverage": float(
            (
                unique_served.masked_fill(~flat_valid, False).sum(dim=-1).float()
                / valid_parent_count
            ).mean().item()
        ),
        "service_obs_ledger_redundant_service_share": float(
            (
                (valid_service_count - unique_served.float())
                .clamp_min(0.0)
                .sum(dim=-1)
                / service_total.clamp_min(1.0)
            ).mean().item()
        ),
        "service_obs_ledger_normalized_hhi": float(
            normalized_service_hhi.mean().item()
        ),
        "service_obs_ledger_effective_service_fraction": float(
            effective_service_fraction.mean().item()
        ),
        "service_obs_ledger_service_count_p90": float(
            torch.quantile(
                valid_service_count.masked_select(flat_valid), 0.90
            ).item()
        ),
        "service_obs_ledger_service_count_max": float(
            valid_service_count.max().item()
        ),
        "service_obs_ledger_repeated_gain_per_action": float(
            (repeated_gain / repeated_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_first_gain_per_action": float(
            (first_gain / first_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_repeated_backlog_per_action": float(
            (repeated_backlog / repeated_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_first_backlog_per_action": float(
            (first_backlog / first_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_debt_only_selection_fraction": float(
            (debt_only_count / selected_count.clamp_min(1.0)).mean().item()
        ),
        "service_obs_ledger_current_only_selection_fraction": float(
            (
                current_only_selected.sum(dim=-1).float()
                / selected_count.clamp_min(1.0)
            ).mean().item()
        ),
        "service_obs_ledger_shared_endpoint_fraction": float(
            (
                shared_endpoint_selected.sum(dim=-1).float()
                / selected_count.clamp_min(1.0)
            ).mean().item()
        ),
        "service_obs_ledger_debt_only_oracle_gain_share": float(
            (debt_only_gain / selected_gain_safe).mean().item()
        ),
        "service_obs_ledger_debt_only_oracle_top_hit_rate": float(
            (
                (debt_only_selected & oracle_top).sum(dim=-1).float()
                / debt_only_count.clamp_min(1.0)
            ).mean().item()
        ),
    }
    raw_child_detail = snapshot.get("raw_child_detail")
    parent_to_children = snapshot.get("parent_to_children")
    parent_child_valid = snapshot.get("parent_child_valid")
    parent_relevance = snapshot.get("parent_relevance")
    action_priority = snapshot.get("action_priority")
    current_action_priority = snapshot.get("current_action_priority")
    selected_actions = snapshot.get("selected_actions")
    valid_child_actions = snapshot.get("valid_child_actions")
    query_carrier = snapshot.get("query_carrier")
    parent_key = snapshot.get("parent_key")
    parent_head_valid = snapshot.get("parent_head_valid")
    proxy_omitted_bound = None
    oracle_omitted_bound = None
    if all(
        value is not None
        for value in (
            raw_child_detail,
            parent_to_children,
            parent_child_valid,
            parent_relevance,
            action_priority,
            current_action_priority,
            selected_actions,
            valid_child_actions,
        )
    ):
        parent_to_children = parent_to_children.long()
        parent_child_valid = parent_child_valid.bool()
        grouped_child_mass = oracle_child_mass[..., parent_to_children]
        grouped_child_detail = raw_child_detail[..., parent_to_children]
        canonical_valid = parent_child_valid.view(
            1, 1, parents_per_frame, -1
        )
        ranked_slots = grouped_child_detail.masked_fill(
            ~canonical_valid, float("-inf")
        ).argsort(dim=-1, descending=True, stable=True)
        ranked_mass = grouped_child_mass.gather(
            -1,
            ranked_slots[:, None].expand(
                -1, query_frame_ids.numel(), -1, -1, -1
            ),
        )
        ranked_detail = grouped_child_detail.gather(-1, ranked_slots)
        oracle_action_gain = ranked_mass * ranked_detail[:, None]

        # Triangle-inequality certificates use value-residual magnitude rather
        # than the squared detail score used for ranking. The oracle version
        # uses dense child attention mass. The deployable proxy splits coarse
        # parent relevance uniformly over valid children and therefore adds no
        # extra QK work.
        child_magnitude = ranked_detail.clamp_min(0.0).sqrt()
        oracle_action_bound = ranked_mass * child_magnitude[:, None]
        child_count = canonical_valid.sum(dim=-1, keepdim=True).clamp_min(1)
        sampled_parent_relevance = parent_relevance.index_select(
            1, query_frame_ids
        ).float()
        proxy_action_bound = (
            sampled_parent_relevance[..., None]
            * child_magnitude[:, None]
            / child_count[:, None]
        )

        sampled_action_priority = action_priority.index_select(
            1, query_frame_ids
        ).float()
        sampled_current_action_priority = current_action_priority.index_select(
            1, query_frame_ids
        ).float()
        sampled_selected_actions = selected_actions.index_select(
            1, query_frame_ids
        ).bool()
        full_action_valid = valid_child_actions.expand_as(selected_actions)
        sampled_action_valid = full_action_valid.index_select(
            1, query_frame_ids
        ) & valid[..., None]

        flat_action_valid = sampled_action_valid.flatten(start_dim=-3)
        flat_action_oracle = oracle_action_gain.flatten(start_dim=-3)
        flat_action_priority = sampled_action_priority.flatten(start_dim=-3)
        flat_current_action_priority = (
            sampled_current_action_priority.flatten(start_dim=-3)
        )
        flat_selected_actions = sampled_selected_actions.flatten(start_dim=-3)
        flat_oracle_action_bound = oracle_action_bound.flatten(start_dim=-3)
        flat_proxy_action_bound = proxy_action_bound.flatten(start_dim=-3)
        omitted_actions = flat_action_valid & ~flat_selected_actions
        oracle_omitted_bound = flat_oracle_action_bound.masked_fill(
            ~omitted_actions, 0.0
        ).sum(dim=-1)
        proxy_omitted_bound = flat_proxy_action_bound.masked_fill(
            ~omitted_actions, 0.0
        ).sum(dim=-1)
        action_counts = flat_selected_actions.sum(dim=-1)
        oracle_action_top = _topk_mask_counts(
            flat_action_oracle, flat_action_valid, action_counts
        )
        current_action_top = _topk_mask_counts(
            flat_current_action_priority, flat_action_valid, action_counts
        )
        debt_action_top = _topk_mask_counts(
            flat_action_priority, flat_action_valid, action_counts
        )

        oracle_optimal_gain = flat_action_oracle.masked_fill(
            ~oracle_action_top, 0.0
        ).sum(dim=-1)
        selected_action_gain = flat_action_oracle.masked_fill(
            ~flat_selected_actions, 0.0
        ).sum(dim=-1)
        current_action_gain = flat_action_oracle.masked_fill(
            ~current_action_top, 0.0
        ).sum(dim=-1)
        selected_efficiency = (
            selected_action_gain / oracle_optimal_gain.clamp_min(1e-12)
        )
        current_efficiency = (
            current_action_gain / oracle_optimal_gain.clamp_min(1e-12)
        )
        selected_recall = (
            (flat_selected_actions & oracle_action_top).sum(dim=-1).float()
            / action_counts.float().clamp_min(1.0)
        )
        current_recall = (
            (current_action_top & oracle_action_top).sum(dim=-1).float()
            / action_counts.float().clamp_min(1.0)
        )
        debt_recall = (
            (debt_action_top & oracle_action_top).sum(dim=-1).float()
            / action_counts.float().clamp_min(1.0)
        )
        debt_spearman = _rowwise_metric_values(
            flat_action_priority,
            flat_action_oracle,
            flat_action_valid,
            _spearman,
        )
        current_spearman = _rowwise_metric_values(
            flat_current_action_priority,
            flat_action_oracle,
            flat_action_valid,
            _spearman,
        )
        efficiency_stats = _distribution_stats(selected_efficiency)
        spearman_stats = _distribution_stats(debt_spearman)
        target_budget = int(snapshot.get("child_budget_per_query", 0))
        budget_error = (action_counts - target_budget).abs().float()
        metrics.update({
            "service_obs_schema_version": 6.0,
            "service_obs_action_budget": float(target_budget),
            "service_obs_action_budget_error_mean": float(
                budget_error.mean().item()
            ),
            "service_obs_action_budget_error_max": float(
                budget_error.max().item()
            ),
            "service_obs_action_current_oracle_spearman": float(
                current_spearman.mean().item()
            ),
            "service_obs_action_debt_oracle_spearman": (
                spearman_stats["mean"]
            ),
            "service_obs_action_debt_rank_gain": float(
                debt_spearman.mean().item()
                - current_spearman.mean().item()
            ),
            "service_obs_action_debt_spearman_se": spearman_stats["se"],
            "service_obs_action_current_topk_recall": float(
                current_recall.mean().item()
            ),
            "service_obs_action_debt_topk_recall": float(
                debt_recall.mean().item()
            ),
            "service_obs_action_selection_topk_recall": float(
                selected_recall.mean().item()
            ),
            "service_obs_action_oracle_gain_efficiency": (
                efficiency_stats["mean"]
            ),
            "service_obs_action_oracle_gain_efficiency_std": (
                efficiency_stats["std"]
            ),
            "service_obs_action_oracle_gain_efficiency_se": (
                efficiency_stats["se"]
            ),
            "service_obs_action_oracle_gain_efficiency_ci95_low": (
                efficiency_stats["ci95_low"]
            ),
            "service_obs_action_oracle_gain_efficiency_ci95_high": (
                efficiency_stats["ci95_high"]
            ),
            "service_obs_action_oracle_gain_efficiency_p10": (
                efficiency_stats["p10"]
            ),
            "service_obs_action_oracle_gain_efficiency_min": (
                efficiency_stats["min"]
            ),
            "service_obs_action_current_policy_efficiency": float(
                current_efficiency.mean().item()
            ),
            "service_obs_action_gain_delta_vs_current": float(
                (selected_efficiency - current_efficiency).mean().item()
            ),
        })
        if all(
            value is not None
            for value in (query_carrier, parent_key, parent_head_valid)
        ):
            sampled_query_carrier = query_carrier.index_select(
                2, query_frame_ids
            ).float()
            query_unit = F.normalize(sampled_query_carrier, dim=-1)
            parent_key_unit = F.normalize(parent_key.float(), dim=-1)
            head_valid = parent_head_valid.bool()
            if parent_key_unit.shape[-2] != parents_per_frame:
                raise ValueError("observer parent key count does not match routing")
            if head_valid.shape != (heads, parents_per_frame):
                raise ValueError("observer parent head mask has an invalid shape")

            temperature = float(snapshot.get("temperature", 1.0))
            if temperature <= 0.0:
                raise ValueError("observer routing temperature must be positive")
            head_logits = torch.einsum(
                "bhqd,bhkpd->bhqkp", query_unit, parent_key_unit
            ) / temperature
            head_weight = head_valid.to(head_logits.dtype).view(
                1, heads, 1, 1, parents_per_frame
            )
            valid_head_count = head_weight.sum(dim=1).clamp_min(1.0)
            approximate_logits = (
                (head_logits * head_weight).sum(dim=1) / valid_head_count
            )

            raw_parent_map, raw_parent_valid = _spatial_parent_patch_map(
                height,
                width,
                parent_size,
                key_frames.device,
            )
            if raw_parent_map.shape[0] != parents_per_frame:
                raise ValueError("fine key map does not match routing parents")
            per_head_error = parent_key_unit.new_zeros(
                batch, heads, num_frames, parents_per_frame
            )
            per_head_directional_error = parent_key_unit.new_zeros(
                batch,
                heads,
                query_frame_ids.numel(),
                num_frames,
                parents_per_frame,
            )
            for parent_idx in range(parents_per_frame):
                child_ids = raw_parent_map[parent_idx]
                child_valid = raw_parent_valid[parent_idx]
                child_key = key_frames.index_select(3, child_ids).float()
                child_key = F.normalize(child_key, dim=-1)
                representative = parent_key_unit[
                    :, :, :, parent_idx, None, :
                ]
                key_delta = child_key - representative
                innovation = key_delta.norm(dim=-1)
                innovation = innovation.masked_fill(~child_valid, 0.0)
                per_head_error[..., parent_idx] = innovation.max(dim=-1).values
                directional_error = torch.einsum(
                    "bhqd,bhkcd->bhqkc", query_unit, key_delta
                ).abs()
                directional_error = directional_error.masked_fill(
                    ~child_valid, 0.0
                )
                per_head_directional_error[..., parent_idx] = (
                    directional_error.max(dim=-1).values
                )
            parent_error = (
                per_head_error[:, :, None]
                * head_weight
            ).sum(dim=1) / valid_head_count
            parent_error = parent_error / temperature
            directional_parent_error = (
                per_head_directional_error * head_weight
            ).sum(dim=1) / valid_head_count
            directional_parent_error = directional_parent_error / temperature
            sampled_current_need = current_need
            sampled_parent_relevance = parent_relevance.index_select(
                1, query_frame_ids
            ).float()
            parent_factor = (
                sampled_current_need
                / sampled_parent_relevance.clamp_min(1e-12)
            )
            child_factor = (
                sampled_current_action_priority
                / sampled_current_need[..., None].clamp_min(1e-12)
            )

            def summarize_interval(
                parent_radius: torch.Tensor,
                prefix: str,
            ) -> dict[str, float]:
                flat_approximate = approximate_logits.flatten(start_dim=-2)
                flat_radius = parent_radius.flatten(start_dim=-2)
                lower_probability, upper_probability = (
                    _softmax_probability_interval(
                        flat_approximate - flat_radius,
                        flat_approximate + flat_radius,
                    )
                )
                lower_probability = lower_probability.view_as(
                    approximate_logits
                )
                upper_probability = upper_probability.view_as(
                    approximate_logits
                )
                lower_action = (
                    lower_probability * parent_factor
                )[..., None] * child_factor
                upper_action = (
                    upper_probability * parent_factor
                )[..., None] * child_factor
                flat_lower = lower_action.flatten(start_dim=-3)
                flat_upper = upper_action.flatten(start_dim=-3)
                current_top = current_action_top
                omitted = flat_action_valid & ~current_top
                selected_lower = flat_lower.masked_fill(
                    ~current_top, float("inf")
                ).min(dim=-1).values
                omitted_upper = flat_upper.masked_fill(
                    ~omitted, float("-inf")
                ).max(dim=-1).values
                certified_margin = selected_lower - omitted_upper
                ambiguous = flat_action_valid & (
                    (current_top & (flat_lower <= omitted_upper[..., None]))
                    | (omitted & (flat_upper >= selected_lower[..., None]))
                )
                disagreement = current_top ^ oracle_action_top
                disagreement_count = disagreement.sum(dim=-1)
                disagreement_rows = disagreement_count > 0
                if disagreement_rows.any():
                    disagreement_coverage = (
                        (ambiguous & disagreement).sum(dim=-1).float()
                        / disagreement_count.float().clamp_min(1.0)
                    )[disagreement_rows].mean()
                else:
                    disagreement_coverage = certified_margin.new_tensor(1.0)
                return {
                    f"{prefix}_certified_topk_fraction": float(
                        (certified_margin > 0.0).float().mean().item()
                    ),
                    f"{prefix}_frontier_fraction": _mean_fraction(
                        ambiguous.sum(dim=-1), flat_action_valid.sum(dim=-1)
                    ),
                    f"{prefix}_oracle_disagreement_row_fraction": float(
                        disagreement_rows.float().mean().item()
                    ),
                    f"{prefix}_oracle_disagreement_coverage": float(
                        disagreement_coverage.item()
                    ),
                    f"{prefix}_margin_mean": float(
                        certified_margin.mean().item()
                    ),
                    f"{prefix}_parent_logit_radius_mean": float(
                        parent_radius.mean().item()
                    ),
                }

            metrics.update(summarize_interval(
                parent_error, "service_obs_qk_interval"
            ))
            metrics.update(summarize_interval(
                directional_parent_error,
                "service_obs_qk_directional_interval",
            ))
    if (
        raw_child_detail is not None
        and parent_to_children is not None
        and parent_child_valid is not None
    ):
        grouped_detail = raw_child_detail[..., parent_to_children]
        child_valid = parent_child_valid.view(
            1, 1, parents_per_frame, -1
        )
        valid_weight = child_valid.to(grouped_detail.dtype)
        child_count = valid_weight.sum(dim=-1).clamp_min(1.0)
        detail_mean = (
            grouped_detail * valid_weight
        ).sum(dim=-1) / child_count
        detail_variance = (
            (grouped_detail - detail_mean[..., None]).square()
            * valid_weight
        ).sum(dim=-1) / child_count
        detail_sum = (grouped_detail * valid_weight).sum(dim=-1)
        detail_max = grouped_detail.masked_fill(
            ~child_valid, float("-inf")
        ).max(dim=-1).values
        metrics.update({
            "service_obs_parent_child_detail_cv": float(
                (
                    detail_variance.sqrt()
                    / detail_mean.clamp_min(1e-8)
                ).mean().item()
            ),
            "service_obs_parent_child_max_share": float(
                (detail_max / detail_sum.clamp_min(1e-8)).mean().item()
            ),
        })

    previous_key = "_service_observation_previous"
    previous = state.get(previous_key)
    previous_query_frame_ids = (
        previous.get("query_frame_ids")
        if isinstance(previous, dict) else None
    )
    if (
        isinstance(previous, dict)
        and torch.is_tensor(previous_query_frame_ids)
        and torch.equal(previous_query_frame_ids, query_frame_ids)
        and previous.get("oracle_gain") is not None
        and previous["oracle_gain"].shape == oracle_gain.shape
    ):
        previous_gain = previous["oracle_gain"].to(oracle_gain)
        previous_current = previous["current_need"].to(current_need)
        previous_selected = previous["selected"].to(selected)
        oracle_layer = _masked_pair_metric(
            previous_gain.flatten(start_dim=-2),
            flat_gain,
            flat_valid,
            _spearman,
        )
        current_layer = _masked_pair_metric(
            previous_current.flatten(start_dim=-2),
            flat_current,
            flat_valid,
            _spearman,
        )
        selection_churn = (
            previous_selected.flatten(start_dim=-2) ^ flat_selected
        ).masked_select(flat_valid).float().mean()
        previous_oracle_top = _topk_mask(
            previous_gain.flatten(start_dim=-2), flat_valid, topk
        )
        oracle_top_intersection = (
            previous_oracle_top & oracle_top
        ).sum(dim=-1)
        oracle_top_union = (
            previous_oracle_top | oracle_top
        ).sum(dim=-1)
        metrics.update({
            "service_obs_oracle_layer_spearman": oracle_layer,
            "service_obs_current_layer_spearman": current_layer,
            "service_obs_oracle_persistence_advantage": (
                oracle_layer - current_layer
            ),
            "service_obs_selection_churn": float(selection_churn.item()),
            "service_obs_oracle_topk_jaccard": _mean_fraction(
                oracle_top_intersection, oracle_top_union
            ),
        })
    else:
        metrics.update({
            "service_obs_oracle_layer_spearman": 0.0,
            "service_obs_current_layer_spearman": 0.0,
            "service_obs_oracle_persistence_advantage": 0.0,
            "service_obs_selection_churn": 0.0,
            "service_obs_oracle_topk_jaccard": 0.0,
        })
    state[previous_key] = {
        "query_frame_ids": query_frame_ids.detach(),
        "oracle_gain": oracle_gain.detach(),
        "current_need": current_need.detach(),
        "selected": selected.detach(),
    }
    transport_overlap = snapshot.get("transport_overlap")
    transport_debt_survival = snapshot.get("transport_debt_survival")
    transport_parent_survival = snapshot.get("transport_parent_survival")
    if transport_overlap is not None and transport_debt_survival is not None:
        overlap = transport_overlap.float().flatten()
        survival = transport_debt_survival.float().flatten()
        parent_survival = (
            transport_parent_survival.float().flatten()
            if transport_parent_survival is not None
            else survival
        )
        metrics.update({
            "service_obs_transport_overlap_mean": float(overlap.mean().item()),
            "service_obs_transport_overlap_p10": float(
                torch.quantile(overlap, 0.10).item()
            ),
            "service_obs_transport_overlap_p50": float(
                torch.quantile(overlap, 0.50).item()
            ),
            "service_obs_transport_overlap_p90": float(
                torch.quantile(overlap, 0.90).item()
            ),
            "service_obs_transport_debt_survival_mean": float(
                survival.mean().item()
            ),
            "service_obs_transport_debt_survival_p10": float(
                torch.quantile(survival, 0.10).item()
            ),
            "service_obs_transport_debt_survival_p50": float(
                torch.quantile(survival, 0.50).item()
            ),
            "service_obs_transport_debt_survival_p90": float(
                torch.quantile(survival, 0.90).item()
            ),
            "service_obs_transport_parent_survival_p10": float(
                torch.quantile(parent_survival, 0.10).item()
            ),
            "service_obs_transport_parent_survival_p50": float(
                torch.quantile(parent_survival, 0.50).item()
            ),
            "service_obs_transport_parent_survival_p90": float(
                torch.quantile(parent_survival, 0.90).item()
            ),
        })
    metrics["service_obs_runtime_ms"] = (
        time.perf_counter() - started
    ) * 1000.0
    if observe_transported_debt:
        sampled_credit = snapshot.get("service_credit")
        if sampled_credit is None:
            sampled_credit = selected_service
        else:
            sampled_credit = sampled_credit.index_select(
                1, query_frame_ids
            ).float()
        metrics["_transported_debt_observation_snapshot"] = {
            "query_frame_ids": query_frame_ids.detach(),
            "query_patch_ids": query_patch_ids.detach(),
            "dense_output": torch.stack(dense_output_rows, dim=2).detach(),
            "scalar_debt": debt.sum(dim=(-2, -1)).detach(),
            "service_credit": sampled_credit.sum(dim=(-2, -1)).detach(),
            "proxy_omitted_bound": (
                None
                if proxy_omitted_bound is None
                else proxy_omitted_bound.detach()
            ),
            "oracle_omitted_bound": (
                None
                if oracle_omitted_bound is None
                else oracle_omitted_bound.detach()
            ),
        }
    return metrics


@torch.no_grad()
def analyze_transported_representation_debt(
    projected_residual: torch.Tensor,
    snapshot: dict[str, Any],
    *,
    layer_idx: int,
    routing_state: dict | None,
) -> dict[str, float]:
    """Measure post-service approximation debt in residual-stream space."""
    if projected_residual.ndim != 4:
        raise ValueError(
            "projected residual must have [B, sampled frames, queries, D] shape"
        )
    state = routing_state if routing_state is not None else {}
    current = projected_residual.float()
    current_norm = current.norm(dim=-1)
    query_frame_ids = snapshot["query_frame_ids"]

    state_key = "_transported_representation_debt"
    previous_state = state.get(state_key)
    compatible = (
        layer_idx > 0
        and isinstance(previous_state, dict)
        and torch.is_tensor(previous_state.get("query_frame_ids"))
        and torch.equal(previous_state["query_frame_ids"], query_frame_ids)
        and previous_state.get("residual") is not None
        and previous_state["residual"].shape == current.shape
    )
    if compatible:
        previous = previous_state["residual"].to(current)
        cumulative = previous_state["cumulative"].to(current)
        path_length = previous_state["path_length"].to(current_norm)
    else:
        previous = torch.zeros_like(current)
        cumulative = torch.zeros_like(current)
        path_length = torch.zeros_like(current_norm)

    previous_norm = previous.norm(dim=-1)
    cumulative_norm = cumulative.norm(dim=-1)
    eps = 1e-8
    previous_cosine = torch.where(
        (previous_norm > eps) & (current_norm > eps),
        (previous * current).sum(dim=-1)
        / (previous_norm * current_norm).clamp_min(eps),
        torch.zeros_like(current_norm),
    )
    cumulative_cosine = torch.where(
        (cumulative_norm > eps) & (current_norm > eps),
        (cumulative * current).sum(dim=-1)
        / (cumulative_norm * current_norm).clamp_min(eps),
        torch.zeros_like(current_norm),
    )
    next_cumulative = cumulative + current
    next_path_length = path_length + current_norm
    coherence = next_cumulative.norm(dim=-1) / next_path_length.clamp_min(eps)

    scalar_debt = snapshot["scalar_debt"].float()
    service_credit = snapshot["service_credit"].float()
    proxy_omitted_bound = snapshot.get("proxy_omitted_bound")
    oracle_omitted_bound = snapshot.get("oracle_omitted_bound")
    frame_residual_norm = current_norm.mean(dim=-1)
    previous_frame_norm = previous_norm.mean(dim=-1)
    residual_reduction = previous_frame_norm - frame_residual_norm

    state[state_key] = {
        "query_frame_ids": query_frame_ids.detach(),
        "residual": current.detach(),
        "cumulative": next_cumulative.detach(),
        "path_length": next_path_length.detach(),
    }
    metrics = {
        "transported_debt_schema_version": 2.0,
        "transported_debt_layer": float(layer_idx),
        "transported_debt_sample_count": float(current_norm.numel()),
        "transported_debt_residual_norm_mean": float(
            current_norm.mean().item()
        ),
        "transported_debt_residual_norm_std": float(
            current_norm.std(unbiased=False).item()
        ),
        "transported_debt_previous_cosine_mean": float(
            previous_cosine.mean().item()
        ),
        "transported_debt_previous_positive_alignment": float(
            (previous_cosine > 0).float().mean().item()
        ),
        "transported_debt_cumulative_cosine_mean": float(
            cumulative_cosine.mean().item()
        ),
        "transported_debt_transportable_fraction": float(
            cumulative_cosine.clamp_min(0).mean().item()
        ),
        "transported_debt_coherence_ratio": float(coherence.mean().item()),
        "transported_debt_cancellation_fraction": float(
            (1.0 - coherence).clamp(0.0, 1.0).mean().item()
        ),
        "transported_debt_scalar_norm_pearson": _pearson(
            scalar_debt, frame_residual_norm
        ),
        "transported_debt_scalar_norm_spearman": _spearman(
            scalar_debt, frame_residual_norm
        ),
        "transported_debt_service_reduction_pearson": _pearson(
            service_credit, residual_reduction
        ),
        "transported_debt_residual_reduction_mean": float(
            residual_reduction.mean().item()
        ),
    }
    if proxy_omitted_bound is not None:
        proxy_omitted_bound = proxy_omitted_bound.float()
        metrics.update({
            "transported_debt_proxy_omitted_bound_mean": float(
                proxy_omitted_bound.mean().item()
            ),
            "transported_debt_proxy_bound_pearson": _pearson(
                proxy_omitted_bound, frame_residual_norm
            ),
            "transported_debt_proxy_bound_spearman": _spearman(
                proxy_omitted_bound, frame_residual_norm
            ),
        })
    if oracle_omitted_bound is not None:
        oracle_omitted_bound = oracle_omitted_bound.float()
        metrics.update({
            "transported_debt_oracle_omitted_bound_mean": float(
                oracle_omitted_bound.mean().item()
            ),
            "transported_debt_oracle_bound_pearson": _pearson(
                oracle_omitted_bound, frame_residual_norm
            ),
            "transported_debt_oracle_bound_spearman": _spearman(
                oracle_omitted_bound, frame_residual_norm
            ),
        })
    if proxy_omitted_bound is not None and oracle_omitted_bound is not None:
        metrics.update({
            "transported_debt_proxy_oracle_bound_pearson": _pearson(
                proxy_omitted_bound, oracle_omitted_bound
            ),
            "transported_debt_proxy_oracle_bound_spearman": _spearman(
                proxy_omitted_bound, oracle_omitted_bound
            ),
        })
    return metrics
