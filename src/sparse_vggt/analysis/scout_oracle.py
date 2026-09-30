import math

import torch
import torch.nn.functional as F


def _correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() < 2:
        return x.new_tensor(float("nan"))
    x = x - x.mean()
    y = y - y.mean()
    denom = x.square().sum().sqrt() * y.square().sum().sqrt()
    if denom <= torch.finfo(x.dtype).eps:
        return x.new_tensor(float("nan"))
    return (x * y).sum() / denom


def _rank(x: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(x, dim=-1)
    ranks = torch.empty_like(order)
    values = torch.arange(x.shape[-1], device=x.device, dtype=order.dtype)
    values = values.view(*([1] * (x.ndim - 1)), -1).expand_as(order)
    ranks.scatter_(-1, order, values)
    return ranks.float()


def _top_fraction_recall(
    predicted: torch.Tensor,
    target: torch.Tensor,
    fraction: float = 0.25,
) -> torch.Tensor:
    count = max(1, math.ceil(target.shape[-1] * fraction))
    predicted_top = predicted.topk(count, dim=-1).indices
    target_top = target.topk(count, dim=-1).indices
    overlap = (
        predicted_top.unsqueeze(-1) == target_top.unsqueeze(-2)
    ).any(dim=-1)
    return overlap.float().mean()


def _variable_topk_mask(
    score: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    sorted_indices = score.argsort(dim=-1, descending=True)
    positions = torch.arange(score.shape[-1], device=score.device)
    selected_sorted = positions.view(*([1] * (score.ndim - 1)), -1) < count.unsqueeze(-1)
    selected = torch.zeros_like(score, dtype=torch.bool)
    selected.scatter_(-1, sorted_indices, selected_sorted)
    return selected


def _mass_with_variable_budget(
    score: torch.Tensor,
    oracle_mass: torch.Tensor,
    count: torch.Tensor,
) -> torch.Tensor:
    selected = _variable_topk_mask(score, count)
    return oracle_mass.masked_fill(~selected, 0.0).sum(dim=-1).mean()


def _sparse_output_error(
    dense_logits: torch.Tensor,
    value: torch.Tensor,
    dense_output: torch.Tensor,
    block_mask: torch.Tensor,
    queries_per_block: int,
    key_block_size: int,
    query_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    if query_blocks is None:
        sampled_mask = block_mask.repeat_interleave(
            queries_per_block, dim=-2
        )
    else:
        sampled_mask = block_mask.index_select(-2, query_blocks)
    token_blocks = torch.arange(
        dense_logits.shape[-1], device=dense_logits.device
    ) // key_block_size
    token_mask = sampled_mask.index_select(-1, token_blocks)
    sparse_probability = dense_logits.masked_fill(
        ~token_mask, float("-inf")
    ).softmax(dim=-1)
    sparse_output = torch.matmul(
        sparse_probability.to(value.dtype), value
    ).float()
    return (sparse_output - dense_output).norm(dim=-1) / dense_output.norm(
        dim=-1
    ).clamp_min(1e-6)


def _two_level_budget(
    need_score: torch.Tensor,
    base_budget: int,
    max_budget: int,
    redistribution_fraction: float = 0.25,
) -> torch.Tensor:
    high_budget = min(
        max_budget,
        math.ceil(base_budget * (1.0 + redistribution_fraction)),
    )
    low_budget = max(1, 2 * base_budget - high_budget)
    ranks = _rank(need_score)
    hard = ranks >= need_score.shape[-1] // 2
    return torch.where(
        hard,
        torch.full_like(ranks, high_budget, dtype=torch.long),
        torch.full_like(ranks, low_budget, dtype=torch.long),
    )


def _sample_query_indices(
    total_tokens: int,
    query_block_size: int,
    max_query_blocks: int,
    queries_per_block: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = math.ceil(total_tokens / query_block_size)
    sampled_blocks = min(num_blocks, max_query_blocks)
    block_indices = torch.linspace(
        0,
        num_blocks - 1,
        steps=sampled_blocks,
        device=device,
    ).round().long().unique()

    query_indices = []
    query_blocks = []
    for block_idx in block_indices.tolist():
        start = block_idx * query_block_size
        end = min(start + query_block_size, total_tokens)
        count = min(queries_per_block, end - start)
        offsets = torch.linspace(
            0,
            end - start - 1,
            steps=count,
            device=device,
        ).round().long()
        query_indices.append(offsets + start)
        query_blocks.append(
            torch.full((count,), block_idx, device=device, dtype=torch.long)
        )
    return torch.cat(query_indices), torch.cat(query_blocks)


def _ordered_postcut_statistics(
    block_max: torch.Tensor,
    support: torch.Tensor,
    priority: torch.Tensor,
    thresholds: tuple[float, ...],
    block_contribution: torch.Tensor | None = None,
    full_output: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Measure running-max convergence for one candidate traversal order."""
    if block_max.ndim != 4:
        raise ValueError("block maxima must be [B, H, sampled queries, Kblk]")
    if support.shape != priority.shape or support.shape != block_max.shape[:2] + (
        block_max.shape[-1],
    ):
        raise ValueError("support and priority must match block-max rows")
    if support.dtype != torch.bool:
        raise ValueError("post-cut support must be boolean")
    if (block_contribution is None) != (full_output is None):
        raise ValueError("post-cut contribution and output must be paired")
    if block_contribution is not None:
        expected = block_max.shape + (block_contribution.shape[-1],)
        if block_contribution.shape != expected:
            raise ValueError("block contribution must match sampled QK rows")
        if full_output.shape != block_contribution.shape[:3] + (
            block_contribution.shape[-1],
        ):
            raise ValueError("full output must match sampled contribution rows")

    key_blocks = support.shape[-1]
    count = support.sum(dim=-1).clamp_min(1)
    ordered_blocks = priority.float().masked_fill(
        ~support, float("-inf")
    ).argsort(dim=-1, descending=True)
    ordered_max = block_max.gather(
        -1,
        ordered_blocks.unsqueeze(-2).expand(
            *block_max.shape[:-1], key_blocks
        ),
    )
    positions = torch.arange(key_blocks, device=block_max.device).view(
        1, 1, 1, key_blocks
    )
    valid = positions < count.unsqueeze(-1).unsqueeze(-1)
    ordered_max = ordered_max.masked_fill(~valid, float("-inf"))
    running_after = ordered_max.cummax(dim=-1).values
    running_before = torch.cat(
        [
            torch.full_like(running_after[..., :1], float("-inf")),
            running_after[..., :-1],
        ],
        dim=-1,
    )
    block_gap = (ordered_max - running_before).amax(dim=-2)
    valid_blocks = valid[..., 0, :]
    denominator = valid_blocks.sum().clamp_min(1)

    global_max = block_max.masked_fill(
        ~support.unsqueeze(-2), float("-inf")
    ).amax(dim=-1, keepdim=True)
    stable = (running_after >= global_max - 1e-5) & valid
    first_stable = stable.to(torch.int64).argmax(dim=-1) + 1
    normalized_first = first_stable.float() / count.unsqueeze(-1).float()
    result = {
        "stable_rank_mean": normalized_first.mean(),
        "stable_rank_p90": torch.quantile(
            normalized_first.reshape(-1), 0.9
        ),
    }
    for fraction, label in ((0.25, "p025"), (0.50, "p050"), (0.75, "p075")):
        prefix_index = (
            torch.ceil(count.float() * fraction).to(torch.long) - 1
        ).clamp(min=0, max=key_blocks - 1)
        prefix_max = running_after.gather(
            -1,
            prefix_index.unsqueeze(-1).unsqueeze(-1).expand(
                *running_after.shape[:-1], 1
            ),
        )
        result[f"stable_fraction_{label}"] = (
            prefix_max >= global_max - 1e-5
        ).float().mean()
    for threshold in thresholds:
        label = f"m{abs(int(threshold))}" if float(threshold).is_integer() else str(
            threshold
        ).replace("-", "m").replace(".", "p")
        skipped = (block_gap < threshold) & valid_blocks
        result[f"skip_fraction_{label}"] = skipped.sum().float() / denominator
        if block_contribution is not None:
            omitted_output = (
                block_contribution
                * skipped.unsqueeze(-2).unsqueeze(-1).to(block_contribution)
            ).sum(dim=-2)
            relative_error = omitted_output.norm(dim=-1) / full_output.norm(
                dim=-1
            ).clamp_min(1e-8)
            result[f"relative_output_error_{label}"] = relative_error.mean()
    return result


@torch.no_grad()
def analyze_cosa_postcut_orders(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    support: torch.Tensor,
    pooled_score: torch.Tensor,
    debt_priority: torch.Tensor | None = None,
    *,
    query_block_size: int = 128,
    key_block_size: int = 64,
    max_query_blocks: int = 8,
    queries_per_block: int = 4,
    proxy_key_stride: int = 8,
    thresholds: tuple[float, ...] = (-1.0, -2.0, -3.0, -4.0, -6.0, -8.0),
) -> dict[str, torch.Tensor]:
    """Compare CoSA-inspired visit orders without changing sparse execution.

    Exact sampled QK supplies the evaluation target. The HRM order itself uses
    only strided-key proxy logits: blocks estimated to contain a row maximum
    are visited first, followed by descending proxy probability mass.
    """
    if query.shape != key.shape or key.shape != value.shape or query.ndim != 4:
        raise ValueError(
            "query, key, and value must be matching [B, H, T, D] tensors"
        )
    if support.ndim != 4 or support.dtype != torch.bool:
        raise ValueError("post-cut support must be boolean [B, H, Qblk, Kblk]")
    if pooled_score.shape != support.shape:
        raise ValueError("pooled score must match post-cut support")
    if debt_priority is not None and debt_priority.shape != support.shape:
        raise ValueError("debt priority must match post-cut support")
    if min(
        query_block_size,
        key_block_size,
        max_query_blocks,
        queries_per_block,
        proxy_key_stride,
    ) <= 0:
        raise ValueError("post-cut block and sampling parameters must be positive")

    batch, heads, total_tokens, head_dim = query.shape
    query_blocks_total = math.ceil(total_tokens / query_block_size)
    key_blocks = math.ceil(total_tokens / key_block_size)
    if tuple(support.shape) != (batch, heads, query_blocks_total, key_blocks):
        raise ValueError("post-cut support layout does not match patch tokens")

    query_indices, query_blocks = _sample_query_indices(
        total_tokens=total_tokens,
        query_block_size=query_block_size,
        max_query_blocks=max_query_blocks,
        queries_per_block=queries_per_block,
        device=query.device,
    )
    sampled_query = query.index_select(-2, query_indices)
    logits = torch.matmul(sampled_query, key.transpose(-1, -2)).float()
    logits = logits * (head_dim**-0.5)
    padding = key_blocks * key_block_size - total_tokens
    block_logits = F.pad(logits, (0, padding), value=float("-inf")).reshape(
        batch,
        heads,
        query_indices.numel(),
        key_blocks,
        key_block_size,
    )
    exact_block_max = block_logits.amax(dim=-1)
    proxy_logits = block_logits[..., ::proxy_key_stride]
    proxy_probability = proxy_logits.flatten(-2).softmax(dim=-1).reshape_as(
        proxy_logits
    )
    proxy_block_mass = proxy_probability.sum(dim=-1)
    proxy_block_max = proxy_logits.amax(dim=-1)
    block_value = F.pad(value.float(), (0, 0, 0, padding)).reshape(
        batch,
        heads,
        key_blocks,
        key_block_size,
        head_dim,
    )

    unique_blocks, sampled_to_unique = query_blocks.unique(
        sorted=True, return_inverse=True
    )
    sampled_support = support.index_select(-2, unique_blocks)
    sampled_score = pooled_score.index_select(-2, unique_blocks).float()
    sampled_debt = (
        debt_priority.index_select(-2, unique_blocks).float()
        if debt_priority is not None
        else sampled_score
    )
    block_ids = torch.arange(key_blocks, device=query.device, dtype=torch.float32)
    natural_priority = -block_ids.view(1, 1, 1, key_blocks).expand_as(
        sampled_score
    )

    order_results: dict[str, list[torch.Tensor]] = {
        order: [] for order in ("natural", "qk", "hrm", "debt")
    }
    hrm_recall = []
    for group_index in range(unique_blocks.numel()):
        rows = sampled_to_unique == group_index
        exact_group = exact_block_max[..., rows, :]
        logits_group = block_logits[..., rows, :, :]
        proxy_max_group = proxy_block_max[..., rows, :]
        proxy_mass_group = proxy_block_mass[..., rows, :].sum(dim=-2)
        support_group = sampled_support[..., group_index, :]
        probability_group = logits_group.masked_fill(
            ~support_group.unsqueeze(-2).unsqueeze(-1), float("-inf")
        )
        probability_group = probability_group.flatten(-2).softmax(
            dim=-1
        ).reshape_as(logits_group)
        block_contribution = torch.einsum(
            "bhrks,bhksd->bhrkd", probability_group, block_value
        )
        full_output = block_contribution.sum(dim=-2)

        masked_proxy_max = proxy_max_group.masked_fill(
            ~support_group.unsqueeze(-2), float("-inf")
        )
        estimated_rowmax = masked_proxy_max.argmax(dim=-1)
        has_estimated_rowmax = torch.zeros_like(support_group)
        has_estimated_rowmax.scatter_(-1, estimated_rowmax, True)
        exact_rowmax = exact_group.masked_fill(
            ~support_group.unsqueeze(-2), float("-inf")
        ).argmax(dim=-1)
        hrm_recall.append(
            has_estimated_rowmax.gather(-1, exact_rowmax).float().mean()
        )
        priority_span = proxy_mass_group.sum(dim=-1, keepdim=True) + 1.0
        hrm_priority = proxy_mass_group + (
            priority_span * has_estimated_rowmax.float()
        )
        priorities = {
            "natural": natural_priority[..., group_index, :],
            "qk": sampled_score[..., group_index, :],
            "hrm": hrm_priority,
            "debt": sampled_debt[..., group_index, :],
        }
        for order, priority in priorities.items():
            measured = _ordered_postcut_statistics(
                exact_group,
                support_group,
                priority,
                thresholds,
                block_contribution=block_contribution,
                full_output=full_output,
            )
            for metric, value in measured.items():
                order_results[order].append((metric, value))

    result = {
        "cosa_postcut_order_observer_enabled": query.new_tensor(1.0),
        "cosa_postcut_order_sampled_queries": query.new_tensor(
            float(query_indices.numel())
        ),
        "cosa_postcut_order_sampled_query_blocks": query.new_tensor(
            float(unique_blocks.numel())
        ),
        "cosa_postcut_order_proxy_hrm_recall": torch.stack(hrm_recall).mean(),
    }
    for order, entries in order_results.items():
        metric_names = {name for name, _ in entries}
        for metric in metric_names:
            values = [value for name, value in entries if name == metric]
            result[f"cosa_postcut_order_{order}_{metric}"] = torch.stack(
                values
            ).mean()
    return result


@torch.no_grad()
def analyze_cosa_exact_pair_risk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    current_support: torch.Tensor,
    debt_support: torch.Tensor,
    *,
    query_block_size: int = 128,
    key_block_size: int = 64,
    max_query_blocks: int = 16,
    queries_per_block: int = 2,
) -> dict[str, torch.Tensor]:
    """Evaluate Current and Debt with sampled exact QK and pair residuals.

    This oracle is diagnostic only. It computes dense softmax on sampled
    queries, then compares the frozen endpoint supports under an independent
    pair-carrier residual bound and the actual sparse-output error.
    """
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if query.ndim != 4:
        raise ValueError("query, key, and value must be [B, H, T, D]")
    if current_support.shape != debt_support.shape:
        raise ValueError("Current and Debt supports must have matching shapes")
    if current_support.ndim != 4:
        raise ValueError("endpoint supports must be [B, H, Qblk, Kblk]")
    if current_support.dtype != torch.bool or debt_support.dtype != torch.bool:
        raise ValueError("endpoint supports must be boolean")
    if min(
        query_block_size,
        key_block_size,
        max_query_blocks,
        queries_per_block,
    ) <= 0:
        raise ValueError("block and sampling parameters must be positive")

    batch, heads, total_tokens, head_dim = query.shape
    key_blocks = math.ceil(total_tokens / key_block_size)
    query_blocks_total = math.ceil(total_tokens / query_block_size)
    expected_support_shape = (batch, heads, query_blocks_total, key_blocks)
    if tuple(current_support.shape) != expected_support_shape:
        raise ValueError("endpoint support layout does not match patch tokens")

    query_indices, query_blocks = _sample_query_indices(
        total_tokens=total_tokens,
        query_block_size=query_block_size,
        max_query_blocks=max_query_blocks,
        queries_per_block=queries_per_block,
        device=query.device,
    )
    sampled_query = query.index_select(-2, query_indices)
    logits = torch.matmul(sampled_query, key.transpose(-1, -2)).float()
    scaled_logits = logits * (head_dim**-0.5)
    probability = scaled_logits.softmax(dim=-1)
    dense_output = torch.matmul(probability.to(value.dtype), value).float()

    padding = key_blocks * key_block_size - total_tokens
    padded_value = F.pad(value.float(), (0, 0, 0, padding))
    block_value = padded_value.reshape(
        batch, heads, key_blocks, key_block_size, head_dim
    ).sum(dim=-2)
    valid_counts = torch.full(
        (key_blocks,),
        float(key_block_size),
        device=value.device,
        dtype=torch.float32,
    )
    if padding:
        valid_counts[-1] -= float(padding)
    block_value = block_value / valid_counts.view(1, 1, key_blocks, 1)

    left = block_value[..., 0::2, :]
    right = block_value[..., 1::2, :]
    if right.shape[-2] != left.shape[-2]:
        right = torch.cat([right, left[..., -1:, :]], dim=-2)
    pair_carrier = ((left + right) * 0.5).repeat_interleave(2, dim=-2)
    pair_carrier = pair_carrier[..., :key_blocks, :]
    token_blocks = (
        torch.arange(total_tokens, device=value.device) // key_block_size
    )
    token_carrier = pair_carrier.index_select(-2, token_blocks)
    token_residual = (value.float() - token_carrier).norm(dim=-1)
    weighted_residual = probability * token_residual.unsqueeze(-2)

    sampled_current = current_support.index_select(-2, query_blocks)
    sampled_debt = debt_support.index_select(-2, query_blocks)
    dense_norm = dense_output.norm(dim=-1).clamp_min(1e-6)

    # Current and Debt are equal-budget endpoint proposals. Their union is a
    # parameter-free candidate envelope: it expands only where the endpoints
    # disagree. Exact QK mass times pair innovation then contracts that
    # envelope back to the original PV budget.
    padded_probability = F.pad(probability, (0, padding))
    block_probability = padded_probability.reshape(
        batch,
        heads,
        probability.shape[-2],
        key_blocks,
        key_block_size,
    ).sum(dim=-1)
    block_innovation = (
        (block_value - pair_carrier).square().mean(dim=-1).sqrt()
    )
    exact_value_risk = block_probability * block_innovation.unsqueeze(-2)
    centered_value_risk = block_probability * (
        block_value.unsqueeze(-3) - dense_output.unsqueeze(-2)
    ).square().mean(dim=-1).sqrt()
    unique_query_blocks, sampled_to_unique = query_blocks.unique(
        sorted=True, return_inverse=True
    )
    grouped_exact_risk = []
    for group_idx in range(unique_query_blocks.numel()):
        grouped_exact_risk.append(
            exact_value_risk[..., sampled_to_unique == group_idx, :].amax(
                dim=-2
            )
        )
    grouped_exact_risk = torch.stack(grouped_exact_risk, dim=-2)
    grouped_centered_risk = []
    for group_idx in range(unique_query_blocks.numel()):
        grouped_centered_risk.append(
            centered_value_risk[
                ..., sampled_to_unique == group_idx, :
            ].amax(dim=-2)
        )
    grouped_centered_risk = torch.stack(grouped_centered_risk, dim=-2)
    grouped_current = current_support.index_select(-2, unique_query_blocks)
    grouped_debt = debt_support.index_select(-2, unique_query_blocks)
    grouped_union = grouped_current | grouped_debt
    final_budget = grouped_current.sum(dim=-1)
    if not torch.equal(final_budget, grouped_debt.sum(dim=-1)):
        raise ValueError("Current and Debt must use the same final budget")
    refined_score = grouped_exact_risk.masked_fill(
        ~grouped_union, float("-inf")
    )
    grouped_refined = _variable_topk_mask(refined_score, final_budget)
    centered_refined_score = grouped_centered_risk.masked_fill(
        ~grouped_union, float("-inf")
    )
    grouped_centered_refined = _variable_topk_mask(
        centered_refined_score, final_budget
    )
    sampled_refined = grouped_refined.index_select(-2, sampled_to_unique)
    sampled_centered_refined = grouped_centered_refined.index_select(
        -2, sampled_to_unique
    )
    union_count = grouped_union.sum(dim=-1).clamp_min(1)
    candidate_ratio = union_count.float() / final_budget.clamp_min(1).float()
    pv_cut_fraction = (
        (union_count - final_budget).clamp_min(0).float()
        / union_count.float()
    )

    def endpoint_bound(
        sampled_support: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_support = sampled_support.index_select(-1, token_blocks)
        selected_probability = probability.masked_fill(~token_support, 0.0)
        selected_mass = selected_probability.sum(dim=-1).clamp_min(1e-8)
        selected_output = torch.matmul(
            selected_probability.to(value.dtype), value
        ).float() / selected_mass.unsqueeze(-1)

        omitted_probability = probability.masked_fill(token_support, 0.0)
        residual_term = weighted_residual.masked_fill(
            token_support, 0.0
        ).sum(dim=-1)
        omitted_mass = omitted_probability.sum(dim=-1)
        omitted_carrier = torch.matmul(
            omitted_probability, token_carrier
        )
        carrier_drift = (
            omitted_carrier
            - omitted_mass.unsqueeze(-1) * selected_output
        ).norm(dim=-1)
        residual_term = residual_term / dense_norm
        carrier_drift = carrier_drift / dense_norm
        return residual_term, carrier_drift, residual_term + carrier_drift

    current_residual, current_drift, current_bound = endpoint_bound(
        sampled_current
    )
    debt_residual, debt_drift, debt_bound = endpoint_bound(sampled_debt)
    refined_residual, refined_drift, refined_bound = endpoint_bound(
        sampled_refined
    )
    bound_scale = current_bound.abs().clamp_min(1.0)
    bound_tolerance = 8.0 * torch.finfo(torch.float32).eps * bound_scale

    current_error = _sparse_output_error(
        dense_logits=scaled_logits,
        value=value,
        dense_output=dense_output,
        block_mask=current_support,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
        query_blocks=query_blocks,
    )
    debt_error = _sparse_output_error(
        dense_logits=scaled_logits,
        value=value,
        dense_output=dense_output,
        block_mask=debt_support,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
        query_blocks=query_blocks,
    )

    refined_token_support = sampled_refined.index_select(-1, token_blocks)
    refined_probability = scaled_logits.masked_fill(
        ~refined_token_support, float("-inf")
    ).softmax(dim=-1)
    refined_output = torch.matmul(
        refined_probability.to(value.dtype), value
    ).float()
    refined_error = (refined_output - dense_output).norm(
        dim=-1
    ) / dense_norm
    centered_refined_token_support = sampled_centered_refined.index_select(
        -1, token_blocks
    )
    centered_refined_probability = scaled_logits.masked_fill(
        ~centered_refined_token_support, float("-inf")
    ).softmax(dim=-1)
    centered_refined_output = torch.matmul(
        centered_refined_probability.to(value.dtype), value
    ).float()
    centered_refined_error = (
        (centered_refined_output - dense_output).norm(dim=-1) / dense_norm
    )
    error_scale = current_error.abs().clamp_min(1.0)
    error_tolerance = 8.0 * torch.finfo(torch.float32).eps * error_scale

    return {
        "cosa_pair_exact_oracle_enabled": query.new_tensor(1.0),
        "cosa_pair_exact_oracle_sampled_queries": query.new_tensor(
            float(query_indices.numel())
        ),
        "cosa_pair_exact_current_carrier_residual": current_residual.mean(),
        "cosa_pair_exact_debt_carrier_residual": debt_residual.mean(),
        "cosa_pair_exact_current_carrier_drift": current_drift.mean(),
        "cosa_pair_exact_debt_carrier_drift": debt_drift.mean(),
        "cosa_pair_exact_current_bound": current_bound.mean(),
        "cosa_pair_exact_debt_bound": debt_bound.mean(),
        "cosa_pair_exact_debt_bound_delta": (
            debt_bound - current_bound
        ).mean(),
        "cosa_pair_exact_debt_bound_safe_fraction": (
            (debt_bound <= current_bound + bound_tolerance).float().mean()
        ),
        "cosa_pair_exact_current_output_error": current_error.mean(),
        "cosa_pair_exact_debt_output_error": debt_error.mean(),
        "cosa_pair_exact_debt_output_error_delta": (
            debt_error - current_error
        ).mean(),
        "cosa_pair_exact_debt_output_better_fraction": (
            (debt_error <= current_error + error_tolerance).float().mean()
        ),
        "cosa_pair_exact_union_candidate_ratio": candidate_ratio.mean(),
        "cosa_pair_exact_union_pv_cut_fraction": pv_cut_fraction.mean(),
        "cosa_pair_exact_refined_current_overlap": (
            (grouped_refined & grouped_current).sum(dim=-1).float()
            / final_budget.clamp_min(1).float()
        ).mean(),
        "cosa_pair_exact_refined_debt_overlap": (
            (grouped_refined & grouped_debt).sum(dim=-1).float()
            / final_budget.clamp_min(1).float()
        ).mean(),
        "cosa_pair_exact_refined_carrier_residual": refined_residual.mean(),
        "cosa_pair_exact_refined_carrier_drift": refined_drift.mean(),
        "cosa_pair_exact_refined_bound": refined_bound.mean(),
        "cosa_pair_exact_refined_output_error": refined_error.mean(),
        "cosa_pair_exact_refined_vs_current_error": (
            refined_error - current_error
        ).mean(),
        "cosa_pair_exact_refined_vs_debt_error": (
            refined_error - debt_error
        ).mean(),
        "cosa_pair_exact_refined_better_both_fraction": (
            (
                (refined_error <= current_error + error_tolerance)
                & (refined_error <= debt_error + error_tolerance)
            ).float().mean()
        ),
        "cosa_pair_exact_centered_refined_current_overlap": (
            (grouped_centered_refined & grouped_current).sum(dim=-1).float()
            / final_budget.clamp_min(1).float()
        ).mean(),
        "cosa_pair_exact_centered_refined_debt_overlap": (
            (grouped_centered_refined & grouped_debt).sum(dim=-1).float()
            / final_budget.clamp_min(1).float()
        ).mean(),
        "cosa_pair_exact_centered_refined_output_error": (
            centered_refined_error.mean()
        ),
        "cosa_pair_exact_centered_refined_vs_current_error": (
            centered_refined_error - current_error
        ).mean(),
        "cosa_pair_exact_centered_refined_vs_debt_error": (
            centered_refined_error - debt_error
        ).mean(),
        "cosa_pair_exact_centered_refined_better_both_fraction": (
            (
                (centered_refined_error <= current_error + error_tolerance)
                & (centered_refined_error <= debt_error + error_tolerance)
            ).float().mean()
        ),
    }


@torch.no_grad()
def analyze_scout_oracle(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    pooled_score: torch.Tensor,
    num_frames: int,
    tokens_per_frame: int,
    sparse_ratio: float,
    local_frame_radius: int = 1,
    query_block_size: int = 128,
    key_block_size: int = 64,
    max_query_blocks: int = 32,
    queries_per_block: int = 4,
    coarse_group_blocks: int = 4,
    candidate_multiplier: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Compare cheap routing signals against sampled dense-token attention.

    The routine is diagnostic only. It samples query tokens, computes their
    exact patch attention, and measures both block retrieval and the ability of
    a frame-level scout to identify queries that need remote communication.
    """
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("query, key, and value must have identical shapes")
    if query.ndim != 4:
        raise ValueError("query, key, and value must be [B, H, T, D]")
    if num_frames <= 0 or tokens_per_frame <= 0:
        raise ValueError("num_frames and tokens_per_frame must be positive")
    if query.shape[-2] != num_frames * tokens_per_frame:
        raise ValueError("patch token count does not match frame metadata")
    if not 0.0 <= sparse_ratio < 1.0:
        raise ValueError("sparse_ratio must be in [0, 1)")
    if local_frame_radius < 0:
        raise ValueError("local_frame_radius must be non-negative")
    if min(
        query_block_size,
        key_block_size,
        max_query_blocks,
        queries_per_block,
        coarse_group_blocks,
    ) <= 0:
        raise ValueError("block and sampling parameters must be positive")
    if candidate_multiplier < 1.0:
        raise ValueError("candidate_multiplier must be at least 1")

    B, heads, total_tokens, head_dim = query.shape
    key_blocks = math.ceil(total_tokens / key_block_size)
    if pooled_score.shape[:2] != (B, heads):
        raise ValueError("pooled_score batch/head dimensions do not match")
    if pooled_score.shape[-1] != key_blocks:
        raise ValueError("pooled_score key block count does not match key")

    query_indices, query_blocks = _sample_query_indices(
        total_tokens=total_tokens,
        query_block_size=query_block_size,
        max_query_blocks=max_query_blocks,
        queries_per_block=queries_per_block,
        device=query.device,
    )
    sampled_query = query.index_select(-2, query_indices)
    scale = head_dim**-0.5
    dense_logits = torch.matmul(
        sampled_query, key.transpose(-1, -2)
    ).float() * scale
    dense_probability = dense_logits.softmax(dim=-1)

    padding = key_blocks * key_block_size - total_tokens
    oracle_block_mass = F.pad(dense_probability, (0, padding)).reshape(
        B,
        heads,
        query_indices.numel(),
        key_blocks,
        key_block_size,
    ).sum(dim=-1)

    block_budget = max(1, int(key_blocks * (1.0 - sparse_ratio)))
    block_budget = min(block_budget, key_blocks)
    sampled_importance = pooled_score.index_select(-2, query_blocks).float()
    importance_indices = sampled_importance.topk(block_budget, dim=-1).indices
    importance_mass_recall = oracle_block_mass.gather(
        -1, importance_indices
    ).sum(dim=-1).mean()
    oracle_indices = oracle_block_mass.topk(block_budget, dim=-1).indices
    importance_topk_recall = (
        importance_indices.unsqueeze(-1) == oracle_indices.unsqueeze(-2)
    ).any(dim=-1).float().mean()

    coarse_tokens = coarse_group_blocks * key_block_size
    key_channels = key.permute(0, 1, 3, 2).reshape(
        B * heads, head_dim, total_tokens
    )
    coarse_key = F.avg_pool1d(
        key_channels,
        kernel_size=coarse_tokens,
        stride=coarse_tokens,
        ceil_mode=True,
    ).reshape(B, heads, head_dim, -1).permute(0, 1, 3, 2)
    coarse_score = torch.matmul(
        sampled_query, coarse_key.transpose(-1, -2)
    ).float() * scale
    coarse_groups = coarse_score.shape[-1]
    candidate_blocks = math.ceil(block_budget * candidate_multiplier)
    candidate_groups = min(
        coarse_groups,
        math.ceil(candidate_blocks / coarse_group_blocks),
    )
    selected_groups = coarse_score.topk(candidate_groups, dim=-1).indices
    block_groups = (
        torch.arange(key_blocks, device=query.device) // coarse_group_blocks
    )
    candidate_mask = (
        selected_groups.unsqueeze(-1) == block_groups.view(1, 1, 1, 1, -1)
    ).any(dim=-2)
    refined_score = sampled_importance.masked_fill(~candidate_mask, float("-inf"))
    refined_indices = refined_score.topk(block_budget, dim=-1).indices
    refined_mass_recall = oracle_block_mass.gather(
        -1, refined_indices
    ).sum(dim=-1).mean()
    refined_topk_recall = (
        refined_indices.unsqueeze(-1) == oracle_indices.unsqueeze(-2)
    ).any(dim=-1).float().mean()

    query_frames = query_indices // tokens_per_frame
    key_frames = torch.arange(total_tokens, device=query.device) // tokens_per_frame
    local_token_mask = (
        query_frames[:, None] - key_frames[None, :]
    ).abs() <= local_frame_radius
    local_probability = dense_logits.masked_fill(
        ~local_token_mask.view(1, 1, query_indices.numel(), total_tokens),
        float("-inf"),
    ).softmax(dim=-1)
    dense_output = torch.matmul(dense_probability.to(value.dtype), value).float()
    local_output = torch.matmul(local_probability.to(value.dtype), value).float()
    oracle_need = (dense_output - local_output).norm(dim=-1) / dense_output.norm(
        dim=-1
    ).clamp_min(1e-6)
    oracle_remote_mass = dense_probability.masked_fill(
        local_token_mask.view(1, 1, query_indices.numel(), total_tokens), 0.0
    ).sum(dim=-1)

    frame_key = key.reshape(
        B, heads, num_frames, tokens_per_frame, head_dim
    ).mean(dim=-2)
    frame_value = value.reshape(
        B, heads, num_frames, tokens_per_frame, head_dim
    ).mean(dim=-2)
    frame_logits = torch.matmul(
        sampled_query, frame_key.transpose(-1, -2)
    ).float() * scale
    frame_probability = frame_logits.softmax(dim=-1)
    frame_ids = torch.arange(num_frames, device=query.device)
    local_frame_mask = (
        query_frames[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    local_frame_probability = frame_logits.masked_fill(
        ~local_frame_mask.view(1, 1, query_indices.numel(), num_frames),
        float("-inf"),
    ).softmax(dim=-1)
    scout_output = torch.matmul(
        frame_probability.to(frame_value.dtype), frame_value
    ).float()
    scout_local_output = torch.matmul(
        local_frame_probability.to(frame_value.dtype), frame_value
    ).float()
    scout_value_need = (scout_output - scout_local_output).norm(
        dim=-1
    ) / scout_output.norm(dim=-1).clamp_min(1e-6)
    scout_remote_mass = frame_probability.masked_fill(
        local_frame_mask.view(1, 1, query_indices.numel(), num_frames), 0.0
    ).sum(dim=-1)

    pooled_query_channels = F.avg_pool1d(
        query.permute(0, 1, 3, 2).reshape(
            B * heads, head_dim, total_tokens
        ),
        kernel_size=query_block_size,
        stride=query_block_size,
        ceil_mode=True,
    )
    pooled_query = pooled_query_channels.reshape(
        B, heads, head_dim, -1
    ).permute(0, 1, 3, 2)
    unique_query_blocks = query_blocks[::queries_per_block]
    block_query = pooled_query.index_select(-2, unique_query_blocks)
    block_frame_logits = torch.matmul(
        block_query, frame_key.transpose(-1, -2)
    ).float() * scale
    block_frame_probability = block_frame_logits.softmax(dim=-1)
    block_query_frames = (
        unique_query_blocks * query_block_size + query_block_size // 2
    ).clamp_max(total_tokens - 1) // tokens_per_frame
    block_local_frame_mask = (
        block_query_frames[:, None] - frame_ids[None, :]
    ).abs() <= local_frame_radius
    block_local_frame_probability = block_frame_logits.masked_fill(
        ~block_local_frame_mask.view(
            1, 1, unique_query_blocks.numel(), num_frames
        ),
        float("-inf"),
    ).softmax(dim=-1)
    block_scout_output = torch.matmul(
        block_frame_probability.to(frame_value.dtype), frame_value
    ).float()
    block_scout_local_output = torch.matmul(
        block_local_frame_probability.to(frame_value.dtype), frame_value
    ).float()
    block_scout_need = (block_scout_output - block_scout_local_output).norm(
        dim=-1
    ) / block_scout_output.norm(dim=-1).clamp_min(1e-6)

    sampled_block_count = unique_query_blocks.numel()
    oracle_need_by_block = oracle_need.reshape(
        B, heads, sampled_block_count, queries_per_block
    ).mean(dim=-1)
    oracle_mass_by_block = oracle_block_mass.reshape(
        B,
        heads,
        sampled_block_count,
        queries_per_block,
        key_blocks,
    ).mean(dim=-2)
    importance_by_block = pooled_score.index_select(
        -2, unique_query_blocks
    ).float()
    scout_budget = _two_level_budget(
        block_scout_need, block_budget, key_blocks
    )
    oracle_budget = _two_level_budget(
        oracle_need_by_block, block_budget, key_blocks
    )
    scout_adaptive_mass = _mass_with_variable_budget(
        importance_by_block, oracle_mass_by_block, scout_budget
    )
    oracle_adaptive_mass = _mass_with_variable_budget(
        importance_by_block, oracle_mass_by_block, oracle_budget
    )
    fixed_block_mass = _mass_with_variable_budget(
        importance_by_block,
        oracle_mass_by_block,
        torch.full_like(scout_budget, block_budget),
    )
    fixed_budget = torch.full_like(scout_budget, block_budget)
    fixed_block_mask = _variable_topk_mask(
        importance_by_block, fixed_budget
    )
    scout_block_mask = _variable_topk_mask(
        importance_by_block, scout_budget
    )
    fixed_output_error = _sparse_output_error(
        dense_logits=dense_logits,
        value=value,
        dense_output=dense_output,
        block_mask=fixed_block_mask,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
    )
    scout_output_error = _sparse_output_error(
        dense_logits=dense_logits,
        value=value,
        dense_output=dense_output,
        block_mask=scout_block_mask,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
    )
    fixed_error_by_block = fixed_output_error.reshape(
        B, heads, sampled_block_count, queries_per_block
    ).mean(dim=-1)
    pooled_value_channels = F.avg_pool1d(
        value.permute(0, 1, 3, 2).reshape(
            B * heads, head_dim, total_tokens
        ),
        kernel_size=key_block_size,
        stride=key_block_size,
        ceil_mode=True,
    )
    pooled_value = pooled_value_channels.reshape(
        B, heads, head_dim, key_blocks
    ).permute(0, 1, 3, 2)
    pooled_value_second_moment = F.avg_pool1d(
        value.float().square().permute(0, 1, 3, 2).reshape(
            B * heads, head_dim, total_tokens
        ),
        kernel_size=key_block_size,
        stride=key_block_size,
        ceil_mode=True,
    ).reshape(B, heads, head_dim, key_blocks).permute(0, 1, 3, 2)
    value_residual_energy = (
        pooled_value_second_moment - pooled_value.float().square()
    ).clamp_min(0.0).mean(dim=-1)
    value_total_energy = pooled_value_second_moment.mean(dim=-1).clamp_min(1e-8)
    value_detail = (value_residual_energy / value_total_energy).sqrt()
    coarse_dense_output = torch.matmul(
        importance_by_block.to(pooled_value.dtype), pooled_value
    ).float()
    fixed_coarse_probability = importance_by_block.masked_fill(
        ~fixed_block_mask, 0.0
    )
    fixed_coarse_probability = fixed_coarse_probability / (
        fixed_coarse_probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    )
    coarse_sparse_output = torch.matmul(
        fixed_coarse_probability.to(pooled_value.dtype), pooled_value
    ).float()
    coarse_preview_error = (coarse_sparse_output - coarse_dense_output).norm(
        dim=-1
    ) / coarse_dense_output.norm(dim=-1).clamp_min(1e-6)
    expected_value_detail = torch.matmul(
        importance_by_block, value_detail.unsqueeze(-1)
    ).squeeze(-1)
    head_value_detail = expected_value_detail.mean(dim=-1)
    preview_budget = _two_level_budget(
        coarse_preview_error, block_budget, key_blocks
    )
    preview_block_mask = _variable_topk_mask(
        importance_by_block, preview_budget
    )
    preview_output_error = _sparse_output_error(
        dense_logits=dense_logits,
        value=value,
        dense_output=dense_output,
        block_mask=preview_block_mask,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
    )
    preview_sweep_errors = {}
    for redistribution_fraction in (0.10, 0.15):
        sweep_budget = _two_level_budget(
            coarse_preview_error,
            block_budget,
            key_blocks,
            redistribution_fraction=redistribution_fraction,
        )
        sweep_mask = _variable_topk_mask(importance_by_block, sweep_budget)
        sweep_error = _sparse_output_error(
            dense_logits=dense_logits,
            value=value,
            dense_output=dense_output,
            block_mask=sweep_mask,
            queries_per_block=queries_per_block,
            key_block_size=key_block_size,
        ).mean()
        preview_sweep_errors[redistribution_fraction] = sweep_error
    protected_head_count = min(heads - 1, max(1, math.ceil(heads * 0.25)))
    protected_head = torch.zeros(
        (B, heads), dtype=torch.bool, device=query.device
    )
    protected_head.scatter_(
        -1,
        head_value_detail.topk(protected_head_count, dim=-1).indices,
        True,
    )
    protected_preview_budget = torch.where(
        protected_head.unsqueeze(-1),
        torch.full_like(preview_budget, block_budget),
        preview_budget,
    )
    protected_preview_mask = _variable_topk_mask(
        importance_by_block, protected_preview_budget
    )
    protected_preview_output_error = _sparse_output_error(
        dense_logits=dense_logits,
        value=value,
        dense_output=dense_output,
        block_mask=protected_preview_mask,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
    )
    fixed_captured_mass = importance_by_block.masked_fill(
        ~fixed_block_mask, 0.0
    ).sum(dim=-1)
    importance_tail_mass = 1.0 - fixed_captured_mass
    importance_probability = importance_by_block / importance_by_block.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    importance_entropy = -(
        importance_probability
        * importance_probability.clamp_min(1e-8).log()
    ).sum(dim=-1) / math.log(max(key_blocks, 2))
    sparse_error_oracle_budget = _two_level_budget(
        fixed_error_by_block, block_budget, key_blocks
    )
    sparse_error_oracle_mask = _variable_topk_mask(
        importance_by_block, sparse_error_oracle_budget
    )
    sparse_error_oracle_output_error = _sparse_output_error(
        dense_logits=dense_logits,
        value=value,
        dense_output=dense_output,
        block_mask=sparse_error_oracle_mask,
        queries_per_block=queries_per_block,
        key_block_size=key_block_size,
    )

    return {
        "scout_oracle_sampled_queries": query_indices.new_tensor(
            query_indices.numel(), dtype=torch.float32
        ),
        "scout_oracle_block_budget": query_indices.new_tensor(
            block_budget, dtype=torch.float32
        ),
        "scout_oracle_importance_mass_recall": importance_mass_recall,
        "scout_oracle_importance_topk_recall": importance_topk_recall,
        "scout_oracle_refined_mass_recall": refined_mass_recall,
        "scout_oracle_refined_topk_recall": refined_topk_recall,
        "scout_oracle_candidate_fraction": candidate_mask.float().mean(),
        "scout_oracle_need_mean": oracle_need.mean(),
        "scout_oracle_remote_mass_mean": oracle_remote_mass.mean(),
        "scout_value_need_correlation": _correlation(
            scout_value_need, oracle_need
        ),
        "scout_value_need_spearman": _correlation(
            _rank(scout_value_need), _rank(oracle_need)
        ),
        "scout_remote_mass_correlation": _correlation(
            scout_remote_mass, oracle_need
        ),
        "scout_hard_query_recall": _top_fraction_recall(
            scout_value_need, oracle_need
        ),
        "scout_block_value_need_correlation": _correlation(
            block_scout_need, oracle_need_by_block
        ),
        "scout_block_value_need_spearman": _correlation(
            _rank(block_scout_need), _rank(oracle_need_by_block)
        ),
        "scout_block_hard_query_recall": _top_fraction_recall(
            block_scout_need, oracle_need_by_block
        ),
        "scout_fixed_block_mass_recall": fixed_block_mass,
        "scout_adaptive_block_mass_recall": scout_adaptive_mass,
        "scout_oracle_adaptive_block_mass_recall": oracle_adaptive_mass,
        "scout_adaptive_block_mass_gain": (
            scout_adaptive_mass - fixed_block_mass
        ),
        "scout_adaptive_mean_block_budget": scout_budget.float().mean(),
        "scout_block_sparse_error_correlation": _correlation(
            block_scout_need, fixed_error_by_block
        ),
        "scout_block_sparse_error_spearman": _correlation(
            _rank(block_scout_need), _rank(fixed_error_by_block)
        ),
        "scout_block_sparse_error_hard_recall": _top_fraction_recall(
            block_scout_need, fixed_error_by_block
        ),
        "scout_fixed_output_error": fixed_output_error.mean(),
        "scout_adaptive_output_error": scout_output_error.mean(),
        "scout_oracle_adaptive_output_error": (
            sparse_error_oracle_output_error.mean()
        ),
        "scout_adaptive_output_error_gain": (
            fixed_output_error.mean() - scout_output_error.mean()
        ),
        "scout_preview_error_correlation": _correlation(
            coarse_preview_error, fixed_error_by_block
        ),
        "scout_preview_error_spearman": _correlation(
            _rank(coarse_preview_error), _rank(fixed_error_by_block)
        ),
        "scout_preview_hard_query_recall": _top_fraction_recall(
            coarse_preview_error, fixed_error_by_block
        ),
        "scout_tail_mass_error_correlation": _correlation(
            importance_tail_mass, fixed_error_by_block
        ),
        "scout_entropy_error_correlation": _correlation(
            importance_entropy, fixed_error_by_block
        ),
        "scout_preview_adaptive_output_error": preview_output_error.mean(),
        "scout_preview_output_error_gain": (
            fixed_output_error.mean() - preview_output_error.mean()
        ),
        "scout_preview_mean_block_budget": preview_budget.float().mean(),
        "scout_preview_error_mean": coarse_preview_error.mean(),
        "scout_preview_error_std": coarse_preview_error.std(unbiased=False),
        "scout_value_detail_mean": expected_value_detail.mean(),
        "scout_value_detail_preview_gap_correlation": _correlation(
            expected_value_detail,
            (coarse_preview_error - fixed_error_by_block).abs(),
        ),
        "scout_preview_protected_output_error": (
            protected_preview_output_error.mean()
        ),
        "scout_preview_protected_output_error_gain": (
            fixed_output_error.mean() - protected_preview_output_error.mean()
        ),
        "scout_preview_adaptive_output_error_r010": (
            preview_sweep_errors[0.10]
        ),
        "scout_preview_adaptive_output_error_r015": (
            preview_sweep_errors[0.15]
        ),
        "scout_preview_output_error_gain_r010": (
            fixed_output_error.mean() - preview_sweep_errors[0.10]
        ),
        "scout_preview_output_error_gain_r015": (
            fixed_output_error.mean() - preview_sweep_errors[0.15]
        ),
    }
