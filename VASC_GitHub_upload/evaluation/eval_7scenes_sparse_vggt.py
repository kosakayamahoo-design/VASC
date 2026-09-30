import os
import sys
import argparse
import csv
import hashlib
from collections import defaultdict
from copy import deepcopy
import json
import re
import time

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


DEBT_MEMORY_DIAGNOSTIC_METRICS = (
    "residual_budget_effective_momentum",
    "residual_budget_next_effective_momentum",
    "residual_budget_effective_momentum_std",
    "residual_budget_effective_momentum_min",
    "residual_budget_effective_momentum_p10",
    "residual_budget_effective_momentum_p50",
    "residual_budget_effective_momentum_p90",
    "residual_budget_effective_momentum_max",
    "residual_budget_next_effective_momentum_std",
    "residual_budget_next_effective_momentum_min",
    "residual_budget_next_effective_momentum_p10",
    "residual_budget_next_effective_momentum_p50",
    "residual_budget_next_effective_momentum_p90",
    "residual_budget_next_effective_momentum_max",
    "residual_budget_next_effective_momentum_zero_fraction",
    "residual_budget_next_effective_momentum_cap_fraction",
    "residual_budget_repayment_utilization_mean",
    "residual_budget_repayment_utilization_std",
    "residual_budget_repayment_utilization_min",
    "residual_budget_repayment_utilization_p10",
    "residual_budget_repayment_utilization_p50",
    "residual_budget_repayment_utilization_p90",
    "residual_budget_repayment_utilization_max",
)


PV_IMPORTANCE_ALIGNMENT_METRICS = (
    "residual_budget_pv_alignment_observer",
    "residual_budget_pv_alignment_unweighted_skip_fraction",
    "residual_budget_pv_alignment_accumulated_debt_weighted_skip",
    "residual_budget_pv_alignment_current_need_weighted_skip",
    "residual_budget_pv_alignment_history_weighted_skip",
    "residual_budget_pv_alignment_debt_skip_excess",
    "residual_budget_pv_alignment_history_skip_excess",
    "residual_budget_pv_alignment_debt_execution_correlation",
    "residual_budget_pv_alignment_top_debt_quartile_skip",
    "residual_budget_pv_alignment_action_weighted_skip",
    "residual_budget_pv_alignment_current_action_weighted_skip",
    "residual_budget_pv_alignment_history_action_weighted_skip",
    "residual_budget_pv_alignment_action_skip_excess",
    "residual_budget_pv_alignment_action_execution_correlation",
    "residual_budget_pv_alignment_top_action_quartile_skip",
)

CARRIER_COMPENSATION_OBSERVER_METRICS = (
    "residual_budget_carrier_compensation_residual_observer",
    "residual_budget_carrier_compensated_pv",
    "residual_budget_two_stage_carrier_observer_skipped_tiles",
    "residual_budget_two_stage_carrier_observer_skipped_query_rows",
    "residual_budget_two_stage_carrier_observer_probability_mass",
    "residual_budget_two_stage_carrier_observer_zero_residual_l2",
    "residual_budget_two_stage_carrier_observer_compensated_residual_l2",
    "residual_budget_two_stage_carrier_observer_residual_ratio",
    "residual_budget_two_stage_carrier_observer_relative_error_reduction",
    "residual_budget_two_stage_carrier_observer_value_innovation_rms",
    "residual_budget_two_stage_carrier_observer_unsafe_tile_fraction",
    "residual_budget_two_stage_carrier_observer_residual_bound_mean",
    "residual_budget_two_stage_carrier_observer_bound_carrier_residual_correlation",
    "residual_budget_two_stage_carrier_observer_bound_zero_residual_correlation",
)

COSA_POSTCUT_ORDER_METRICS = (
    "cosa_postcut_order_observer_enabled",
    "cosa_postcut_order_sampled_queries",
    "cosa_postcut_order_sampled_query_blocks",
    "cosa_postcut_order_proxy_hrm_recall",
) + tuple(
    f"cosa_postcut_order_{order}_{metric}"
    for order in ("natural", "qk", "hrm", "debt")
    for metric in (
        "stable_rank_mean",
        "stable_rank_p90",
        "stable_fraction_p025",
        "stable_fraction_p050",
        "stable_fraction_p075",
        "skip_fraction_m1",
        "skip_fraction_m2",
        "skip_fraction_m3",
        "skip_fraction_m4",
        "skip_fraction_m6",
        "skip_fraction_m8",
        "relative_output_error_m1",
        "relative_output_error_m2",
        "relative_output_error_m3",
        "relative_output_error_m4",
        "relative_output_error_m6",
        "relative_output_error_m8",
    )
)

PROJECTED_QK_METRICS = COSA_POSTCUT_ORDER_METRICS + (
    "cosa_pair_debt_enabled",
    "cosa_pair_service_useful_repayment",
    "cosa_pair_service_wasted",
    "cosa_pair_service_utilization",
    "cosa_pair_service_gate_debt",
    "cosa_pair_qk_capacity",
    "cosa_pair_arrival",
    "cosa_pair_innovation",
    "cosa_pair_debt_before_service",
    "cosa_pair_settled_arrival",
    "cosa_pair_consumed_pv_service",
    "cosa_pair_admitted_debt",
    "cosa_pair_backlog",
    "cosa_pair_qk_overlap",
    "cosa_pair_qk_swap_fraction",
    "cosa_pair_current_debt_overlap",
    "cosa_pair_history_swap_fraction",
    "cosa_pair_budget_error",
    "cosa_pair_conservation_error",
    "cosa_pair_risk_observer_enabled",
    "cosa_pair_risk_current_omission",
    "cosa_pair_risk_debt_omission",
    "cosa_pair_risk_debt_delta",
    "cosa_pair_risk_debt_safe_fraction",
    "cosa_pair_risk_support_changed_fraction",
    "cosa_pair_exact_oracle_enabled",
    "cosa_pair_exact_oracle_sampled_queries",
    "cosa_pair_exact_current_carrier_residual",
    "cosa_pair_exact_debt_carrier_residual",
    "cosa_pair_exact_current_carrier_drift",
    "cosa_pair_exact_debt_carrier_drift",
    "cosa_pair_exact_current_bound",
    "cosa_pair_exact_debt_bound",
    "cosa_pair_exact_debt_bound_delta",
    "cosa_pair_exact_debt_bound_safe_fraction",
    "cosa_pair_exact_current_output_error",
    "cosa_pair_exact_debt_output_error",
    "cosa_pair_exact_debt_output_error_delta",
    "cosa_pair_exact_debt_output_better_fraction",
    "cosa_pair_exact_union_candidate_ratio",
    "cosa_pair_exact_union_pv_cut_fraction",
    "cosa_pair_exact_refined_current_overlap",
    "cosa_pair_exact_refined_debt_overlap",
    "cosa_pair_exact_refined_carrier_residual",
    "cosa_pair_exact_refined_carrier_drift",
    "cosa_pair_exact_refined_bound",
    "cosa_pair_exact_refined_output_error",
    "cosa_pair_exact_refined_vs_current_error",
    "cosa_pair_exact_refined_vs_debt_error",
    "cosa_pair_exact_refined_better_both_fraction",
    "cosa_pair_exact_centered_refined_current_overlap",
    "cosa_pair_exact_centered_refined_debt_overlap",
    "cosa_pair_exact_centered_refined_output_error",
    "cosa_pair_exact_centered_refined_vs_current_error",
    "cosa_pair_exact_centered_refined_vs_debt_error",
    "cosa_pair_exact_centered_refined_better_both_fraction",
    "cosa_pair_ordered_lut_enabled",
    "cosa_pair_hrm_order_enabled",
    "cosa_pair_ordered_negative_jump_fraction",
    "cosa_pair_ordered_mean_abs_jump",
    "cosa_pair_terminal_settled",
    "cosa_pair_terminal_layer",
    "cosa_pair_terminal_arrival",
    "cosa_pair_terminal_service",
    "cosa_pair_terminal_backlog",
    "cosa_pair_terminal_conservation_error",
    "projected_qk_debt_enabled",
    "projected_qk_debt_capacity",
    "projected_qk_debt_arrival",
    "projected_qk_debt_admitted",
    "projected_qk_debt_before",
    "projected_qk_debt_after",
    "projected_qk_debt_projection_scale",
    "projected_qk_debt_base_overlap",
    "projected_qk_debt_swap_fraction",
    "projected_qk_debt_protected_retention",
    "projected_qk_debt_reflection",
    "projected_qk_debt_conservation_error",
    "projected_qk_pv_debt_enabled",
    "projected_qk_pv_debt_mean",
    "projected_qk_pv_debt_max",
    "projected_qk_pv_debt_before_service",
    "projected_qk_pv_settled_arrival",
    "projected_qk_pv_arrival_max",
    "projected_qk_pv_arrival_budget_error",
    "projected_qk_pv_reflection",
    "projected_qk_pv_service_reduction",
    "projected_qk_pv_debt_admission_scale",
    "projected_qk_pv_debt_admitted",
    "projected_qk_pv_debt_admitted_max",
    "projected_qk_pv_service_fraction",
    "projected_qk_pv_base_overlap",
    "projected_qk_pv_swap_fraction",
    "projected_qk_pv_protected_retention",
    "projected_qk_pv_budget_error",
    "projected_qk_pv_queue_conservation_error",
    "projected_qk_pv_terminal_settled",
    "projected_qk_pv_terminal_layer",
    "projected_qk_pv_terminal_arrival",
    "projected_qk_pv_terminal_service",
    "projected_qk_pv_terminal_service_reduction",
    "projected_qk_pv_terminal_reflection",
    "projected_qk_pv_terminal_backlog",
    "projected_qk_pv_terminal_backlog_max",
    "projected_qk_pv_terminal_conservation_error",
    "qk_pv_service_observed",
    "qk_pv_service_identity_observed",
    "qk_pv_warps_per_query_block",
    "qk_pv_candidate_slots",
    "qk_pv_executed_slots",
    "qk_pv_skipped_slots",
    "qk_pv_executed_fraction",
    "qk_pv_domain_executed_min",
    "qk_pv_domain_executed_max",
    "qk_pv_domain_skipped_mean",
    "qk_pv_domain_skipped_max",
    "qk_bundle_observer_enabled",
    "qk_bundle_observer_size",
    "qk_bundle_observer_count",
    "qk_bundle_observer_arrival",
    "qk_bundle_observer_service",
    "qk_bundle_observer_debt_before",
    "qk_bundle_observer_debt_after",
    "qk_bundle_observer_debt_max",
    "qk_bundle_observer_admitted",
    "qk_bundle_observer_reflection",
    "qk_bundle_observer_conservation_error",
    "qk_bundle_observer_base_bundle_fill",
    "qk_bundle_observer_current_base_overlap",
    "qk_bundle_observer_debt_base_overlap",
    "qk_bundle_observer_debt_current_overlap",
    "qk_bundle_observer_current_swap_fraction",
    "qk_bundle_observer_debt_swap_fraction",
    "qk_bundle_observer_history_swap_fraction",
    "qk_bundle_observer_current_score_retention",
    "qk_bundle_observer_debt_score_retention",
    "qk_bundle_observer_future_base_score_ratio",
    "qk_bundle_observer_future_current_score_ratio",
    "qk_bundle_observer_future_debt_score_ratio",
    "qk_bundle_observer_budget_error",
    "qk_bundle_observer_protected_retention",
    "qk_value_risk_observer_enabled",
    "qk_value_risk_observer_execution_mode",
    "qk_value_risk_observer_definition",
    "qk_value_risk_observer_state_present",
    "qk_value_risk_observer_state_shape_match",
    "qk_value_risk_observer_stored_checksum",
    "qk_value_risk_observer_key_blocks",
    "qk_value_risk_observer_pair_blocks",
    "qk_value_risk_observer_query_rows",
    "qk_value_risk_observer_innovation_mean",
    "qk_value_risk_observer_innovation_max",
    "qk_value_risk_observer_relative_innovation_mean",
    "qk_value_risk_observer_relative_innovation_max",
    "qk_value_risk_observer_arrival",
    "qk_value_risk_observer_debt_before",
    "qk_value_risk_observer_debt_after",
    "qk_value_risk_observer_debt_max",
    "qk_value_risk_observer_admitted",
    "qk_value_risk_observer_reflection",
    "qk_value_risk_observer_conservation_error",
    "qk_value_risk_observer_current_base_overlap",
    "qk_value_risk_observer_debt_base_overlap",
    "qk_value_risk_observer_debt_current_overlap",
    "qk_value_risk_observer_current_swap_fraction",
    "qk_value_risk_observer_debt_swap_fraction",
    "qk_value_risk_observer_history_swap_fraction",
    "qk_value_risk_observer_current_risk_retention",
    "qk_value_risk_observer_debt_risk_retention",
    "qk_value_risk_observer_future_base_risk_ratio",
    "qk_value_risk_observer_future_current_risk_ratio",
    "qk_value_risk_observer_future_debt_risk_ratio",
    "qk_value_risk_observer_budget_error",
    "qk_value_risk_observer_protected_retention",
)


def optional_float(value):
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def get_args_parser():
    parser = argparse.ArgumentParser("Indoor reconstruction evaluation for sparse-VGGT", add_help=True)
    parser.add_argument("--weights", type=str, required=True, help="Path to VGGT checkpoint")
    parser.add_argument("--streamvggt_root", type=str, required=True, help="Path to StreamVGGT repository root")
    parser.add_argument(
        "--vggt_root",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "external", "vggt"),
        help="Path to the VGGT repository root to import from",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="7scenes",
        choices=["7scenes", "nrgbd", "eth3d"],
        help="Indoor reconstruction benchmark to evaluate",
    )
    parser.add_argument(
        "--vanilla_vggt",
        action="store_true",
        help="Evaluate the imported VGGT model without a sparse wrapper",
    )
    parser.add_argument("--data_root", type=str, required=True, help="Path to the selected dataset root")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--conf_thresh", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse_ratio", type=float, default=None)
    parser.add_argument("--cdf_threshold", type=optional_float, default=0.97,
                        help="CDF block-selection threshold, or 'none' for ratio-only evaluation")
    parser.add_argument("--layer_sparsity", type=str, default=None,
                        help="Comma-separated list of sparsity ratios per layer, e.g., '0.95,0.90,0.85,0.80'")
    parser.add_argument("--pool_mode", type=str, default="avg", choices=["avg", "max"])
    parser.add_argument("--use_hilbert", action="store_true")
    parser.add_argument("--aux_output", action="store_true")
    parser.add_argument("--disable_aux_sparsity_only", action="store_true")
    parser.add_argument(
        "--skip_block_sparsity_collect",
        action="store_true",
        help="Skip per-sequence block sparsity extraction to avoid extra CUDA sync",
    )
    parser.add_argument(
        "--analyze_importance_blocks",
        action="store_true",
        help="Backward-compatible alias for --analyze_block_selection",
    )
    parser.add_argument(
        "--analyze_block_selection",
        action="store_true",
        help="Collect per-layer block-selection diagnostics for the active routing mode",
    )
    parser.add_argument(
        "--importance_block_analysis_csv",
        type=str,
        default=None,
        help="Backward-compatible alias for --block_selection_analysis_csv",
    )
    parser.add_argument(
        "--block_selection_analysis_csv",
        type=str,
        default=None,
        help="Optional CSV path for per-layer block-selection diagnostics",
    )
    parser.add_argument("--use_proj", action="store_true", help="Reconstruct points from depth + camera instead of point head")
    parser.add_argument("--scene", type=str, default=None,
                        help="Restrict evaluation to one scene, e.g. chess or whiteroom")
    parser.add_argument("--seq_id", type=str, default=None,
                        help="Restrict 7-Scenes evaluation to a normalized sequence id, e.g. seq-03")
    parser.add_argument("--kf_every", type=int, default=200,
                        help="Evaluate every Nth video frame in each sequence")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Evaluate exactly the first N sampled frames from each selected sequence",
    )
    parser.add_argument(
        "--num_frames_is_cap",
        action="store_true",
        help="Treat --num_frames as a per-sequence maximum and keep shorter sequences",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=None,
        help="Deterministically cap predicted and GT point clouds before ICP/metrics",
    )
    parser.add_argument("--pose_only", action="store_true",
                        help="Compute ATE/RPE and skip point-cloud reconstruction metrics")
    parser.add_argument("--metrics_json", type=str, default=None,
                        help="Write pose accuracy and runtime metrics as JSON")
    parser.add_argument("--timing_warmup", type=int, default=0,
                        help="Untimed forward passes per sequence before measurement")
    parser.add_argument("--timing_repeats", type=int, default=1,
                        help="Timed forward passes per sequence; median latency is reported")
    parser.add_argument(
        "--profile_attention",
        action="store_true",
        help="Measure global sparse-attention latency in separate CUDA-timed forwards",
    )
    parser.add_argument(
        "--torch_profiler_trace",
        type=str,
        default=None,
        help="Export one additional model forward as a PyTorch profiler trace",
    )
    parser.add_argument(
        "--analyze_scout_oracle",
        action="store_true",
        help="Compare sampled dense attention with scout/refine routing signals",
    )
    parser.add_argument(
        "--scout_oracle_layers", type=str, default="0,8,15,23"
    )
    parser.add_argument("--scout_oracle_query_blocks", type=int, default=32)
    parser.add_argument(
        "--scout_oracle_queries_per_block", type=int, default=4
    )
    parser.add_argument("--scout_oracle_local_radius", type=int, default=1)
    parser.add_argument(
        "--scout_oracle_coarse_group_blocks", type=int, default=4
    )
    parser.add_argument(
        "--scout_oracle_candidate_multiplier", type=float, default=2.0
    )
    # Radial + layer-wise fusion parameters
    parser.add_argument("--use_radial_layerwise", action="store_true",
                        help="Enable fused radial+layer-wise+importance mask")
    parser.add_argument("--decay_factor", type=float, default=1.0,
                        help="Radial decay factor for spatial constraint (default: 1.0)")
    parser.add_argument("--dense_neighbor", type=int, default=1,
                        help="Number of neighboring frames with full attention (default: 1)")
    parser.add_argument(
        "--use_distance_routed",
        action="store_true",
        help="Use radial attention for near frame pairs and importance for far pairs",
    )
    parser.add_argument(
        "--route_frame_threshold",
        type=int,
        default=4,
        help="Maximum |frame_i-frame_j| routed to radial attention (default: 4)",
    )
    parser.add_argument(
        "--use_covariance_aware_importance",
        action="store_true",
        help="Calibrate far-frame importance using K-block variance",
    )
    parser.add_argument(
        "--covariance_weight",
        type=float,
        default=0.5,
        help="Weight of log K-block variance in far importance (default: 0.5)",
    )
    parser.add_argument(
        "--covariance_eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for covariance-aware importance",
    )
    parser.add_argument(
        "--use_adaptive_slit_routing",
        action="store_true",
        help="Within distance routing, use adaptive local slit blocks for near frame pairs",
    )
    parser.add_argument("--adaptive_slit_temporal_window", type=int, default=10,
                        help="Temporal window used for adaptive slit stability statistics")
    parser.add_argument("--adaptive_slit_stable_quantile", type=float, default=0.6,
                        help="High-mean/low-variance quantile for narrow stable slits")
    parser.add_argument("--adaptive_slit_change_quantile", type=float, default=0.7,
                        help="High-variance/low-mean quantile for expanded changing slits")
    parser.add_argument("--adaptive_slit_narrow_width", type=int, default=1,
                        help="Local key-block half-width for stable near-frame slits")
    parser.add_argument("--adaptive_slit_base_width", type=int, default=2,
                        help="Local key-block half-width for default near-frame slits")
    parser.add_argument("--adaptive_slit_expand_width", type=int, default=4,
                        help="Local key-block half-width for changing near-frame slits")
    parser.add_argument("--use_soft_geometry_routing", action="store_true",
                        help="Enable strict-budget geometry-content routing")
    parser.add_argument(
        "--use_preview_adaptive_routing",
        action="store_true",
        help="Redistribute a strict block budget using pooled output error",
    )
    parser.add_argument(
        "--use_multiresolution_routing",
        action="store_true",
        help=(
            "Use exact nearby K/V, pooled remote K/V, and all-frame carrier "
            "tokens without importance ranking"
        ),
    )
    parser.add_argument("--multiresolution_local_radius", type=int, default=2)
    parser.add_argument(
        "--multiresolution_remote_pool_size", type=int, default=2
    )
    parser.add_argument(
        "--multiresolution_query_frame_chunk", type=int, default=4
    )
    parser.add_argument(
        "--multiresolution_remote_mode",
        choices=["avg", "strided", "detail"],
        default="strided",
    )
    parser.add_argument(
        "--multiresolution_remote_samples_per_cell", type=int, default=1
    )
    parser.add_argument(
        "--multiresolution_layer_samples",
        type=str,
        default=None,
        help="Comma-separated per-layer remote samples per spatial cell",
    )
    parser.add_argument(
        "--multiresolution_layer_pool_sizes",
        type=str,
        default=None,
        help="Comma-separated per-layer remote spatial pool sizes",
    )
    parser.add_argument(
        "--multiresolution_layer_local_radii",
        type=str,
        default=None,
        help="Comma-separated per-layer exact local frame radii",
    )
    parser.add_argument(
        "--multiresolution_remote_phase_mode",
        choices=["layer", "query", "head"],
        default="layer",
    )
    parser.add_argument(
        "--multiresolution_area_bias", action="store_true"
    )
    parser.add_argument(
        "--use_residual_budget_routing",
        action="store_true",
        help="Allocate complementary remote phases using persistent pair debt",
    )
    parser.add_argument(
        "--residual_budget_target_sparsity", type=float, default=0.70
    )
    parser.add_argument(
        "--residual_budget_local_radius", type=int, default=0
    )
    parser.add_argument(
        "--residual_budget_remote_pool_size", type=int, default=2
    )
    parser.add_argument(
        "--residual_budget_routing_parent_size", type=int, default=2
    )
    parser.add_argument(
        "--residual_budget_routing_phase_mode",
        choices=["rotating", "fixed", "anchored", "anchored_heads"],
        default="rotating",
    )
    parser.add_argument(
        "--residual_budget_unified_incremental_service",
        action="store_true",
        help=(
            "Allocate parent coverage and child refinement from one debt "
            "queue, with execution selected by the structural budget floor"
        ),
    )
    parser.add_argument(
        "--residual_budget_bounded_wait_reobservation",
        action="store_true",
        help=(
            "Prioritize unified 4x4 parents that have waited longer than the "
            "budget-derived full-service cycle, without adding service budget"
        ),
    )
    parser.add_argument(
        "--residual_budget_adaptive_parent_service",
        action="store_true",
        help="Use child heterogeneity for cost-aware 1/4 parent service",
    )
    parser.add_argument(
        "--residual_budget_mixed_parent_execution",
        action="store_true",
        help="Execute easy 4x4 parents and promote debt-selected 2x2 children",
    )
    parser.add_argument(
        "--residual_budget_additive_parent_service",
        action="store_true",
        help=(
            "Keep 4x4 parent scouts as attention service and add "
            "debt-selected 2x2 child carriers"
        ),
    )
    parser.add_argument(
        "--residual_budget_additive_full_upgrade_fraction",
        type=float,
        default=0.0,
        help=(
            "Reserve this budget fraction for fourth-child upgrades on "
            "highest-debt additive parents"
        ),
    )
    parser.add_argument(
        "--residual_budget_additive_parent_substitution_fraction",
        type=float,
        default=0.0,
        help=(
            "Replace this fraction of selected parent+three-child services "
            "with equal-cost four-child services"
        ),
    )
    parser.add_argument(
        "--residual_budget_mixed_residuals_per_parent", type=int, default=4
    )
    parser.add_argument(
        "--residual_budget_parent_hard_fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--residual_budget_parent_cost_power", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_frame_balance_fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_fine_refresh_layers",
        type=str,
        default="",
        help=(
            "Comma-separated global layers that replace 4x4 parent routing "
            "with a budget-matched 2x2 debt refresh"
        ),
    )
    parser.add_argument(
        "--residual_budget_fine_refresh_sparsity",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--residual_budget_budget_neutral_fine_refresh",
        action="store_true",
        help=(
            "Compensate fixed fine-refresh layers on all ordinary layers so "
            "the layer-average target sparsity is unchanged"
        ),
    )
    parser.add_argument(
        "--residual_budget_query_frame_chunk", type=int, default=4
    )
    parser.add_argument(
        "--residual_budget_momentum", type=float, default=0.75
    )
    parser.add_argument(
        "--residual_budget_service_conditioned_momentum",
        action="store_true",
        help=(
            "adapt the next grouped-parent debt momentum below the configured "
            "maximum using actual repayment-credit utilization"
        ),
    )
    parser.add_argument(
        "--residual_budget_repayment", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_repayment_mode",
        choices=[
            "reset",
            "deficit",
            "bounded_deficit",
            "effective_bounded_deficit",
            "age_bounded_deficit",
            "frontier_age_bounded_deficit",
        ],
        default="reset",
    )
    parser.add_argument(
        "--residual_budget_service_credit_scale", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_temperature", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_exact_fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_surface_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_selection_granularity",
        choices=["frame", "cell"],
        default="frame",
    )
    parser.add_argument(
        "--residual_budget_frame_detail_power", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_spatial_detail_power", type=float, default=1.0
    )
    parser.add_argument(
        "--residual_budget_cell_scorer",
        choices=[
            "uniform",
            "residual",
            "carrier",
            "carrier_residual",
            "patch_qk",
            "scout_service",
        ],
        default="carrier_residual",
    )
    parser.add_argument(
        "--residual_budget_carrier_residual_balance",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--residual_budget_cell_refinement_phases", type=int, default=1
    )
    parser.add_argument(
        "--residual_budget_cell_precision_fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_mass_conserving_refinement",
        action="store_true",
    )
    parser.add_argument(
        "--residual_budget_exact_mass_conserving_refinement",
        action="store_true",
    )
    parser.add_argument(
        "--residual_budget_cell_importance_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_cell_importance_floor", type=float, default=0.0
    )
    parser.add_argument(
        "--residual_budget_cell_importance_gate_threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--residual_budget_cell_importance_gate_temperature",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--preview_redistribution_fraction", type=float, default=0.05
    )
    parser.add_argument(
        "--preview_activation_threshold", type=float, default=0.60
    )
    parser.add_argument("--preview_local_radius", type=int, default=1)
    parser.add_argument("--preview_local_fraction", type=float, default=0.0)
    parser.add_argument("--preview_geometry_weight", type=float, default=0.0)
    parser.add_argument(
        "--preview_head_protection",
        choices=["none", "value_detail", "locality"],
        default="none",
    )
    parser.add_argument(
        "--preview_protected_head_fraction", type=float, default=0.25
    )
    parser.add_argument(
        "--preview_donor_retention_threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--preview_donor_exchange_scope",
        choices=["head", "layer"],
        default="head",
    )
    parser.add_argument(
        "--preview_receiver_gain_threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--preview_exchange_gain_cost_ratio", type=float, default=0.0
    )
    parser.add_argument(
        "--preview_head_exchange_cap_fraction", type=float, default=1.0
    )
    parser.add_argument(
        "--preview_layer_confidence_threshold", type=float, default=0.0
    )
    parser.add_argument(
        "--preview_exchange_layer_start", type=int, default=0
    )
    parser.add_argument(
        "--preview_exchange_layer_end", type=int, default=-1
    )
    parser.add_argument(
        "--use_dual_path_routing",
        action="store_true",
        help="Split a strict budget between local support and global importance",
    )
    parser.add_argument("--dual_path_local_radius", type=int, default=2)
    parser.add_argument(
        "--dual_path_min_local_fraction", type=float, default=0.10
    )
    parser.add_argument(
        "--dual_path_max_local_fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--dual_path_layer_schedule",
        choices=["flat", "early_decay", "middle_peak"],
        default="early_decay",
    )
    parser.add_argument(
        "--dual_path_context_geometry_weight", type=float, default=0.0
    )
    parser.add_argument(
        "--dual_path_context_schedule",
        choices=["flat", "late_ramp", "late_only"],
        default="flat",
    )
    parser.add_argument(
        "--dual_path_context_gate",
        choices=["none", "disagreement"],
        default="none",
    )
    parser.add_argument(
        "--dual_path_context_alignment_threshold", type=float, default=1.2
    )
    parser.add_argument(
        "--dual_path_context_alignment_temperature", type=float, default=0.1
    )
    parser.add_argument(
        "--use_layerwise_hybrid_routing",
        action="store_true",
        help="Use layer-wise radial/importance/soft hybrid routing",
    )
    parser.add_argument(
        "--use_local_layerwise_hybrid_routing",
        action="store_true",
        help="Use local early + far-importance middle + soft late hybrid routing",
    )
    parser.add_argument(
        "--use_coverage_layerwise_routing",
        action="store_true",
        help="Use local early + coverage-constrained middle + reference-soft late routing",
    )
    parser.add_argument(
        "--use_persistent_view_graph_routing",
        action="store_true",
        help="Use soft routing around a persistent middle-layer view graph",
    )
    parser.add_argument("--hybrid_early_end", type=int, default=8,
                        help="First layer index after the early radial stage")
    parser.add_argument("--hybrid_mid_end", type=int, default=16,
                        help="First layer index after the middle importance stage")
    parser.add_argument("--hybrid_mid_sparse_ratio", type=float, default=0.50,
                        help="Sparse ratio for middle-stage importance routing")
    parser.add_argument("--hybrid_late_sparse_ratio", type=float, default=0.70,
                        help="Sparse ratio for late-stage soft-geometry routing")
    parser.add_argument(
        "--hybrid_early_dense_neighbor",
        type=int,
        default=4,
        help="Frame distance receiving full radial width in the early stage",
    )
    parser.add_argument(
        "--hybrid_early_local_radius",
        type=int,
        default=2,
        help="Frame radius kept exactly in local early hybrid routing",
    )
    parser.add_argument(
        "--mid_frame_coverage_ratio",
        type=float,
        default=0.25,
        help="Fraction of key frames receiving a middle-stage coverage anchor",
    )
    parser.add_argument("--view_graph_early_end", type=int, default=6)
    parser.add_argument("--view_graph_mid_end", type=int, default=18)
    parser.add_argument("--view_graph_mid_sparse_ratio", type=float, default=0.70)
    parser.add_argument("--view_graph_local_radius", type=int, default=1)
    parser.add_argument("--view_graph_remote_topk", type=int, default=3)
    parser.add_argument("--view_graph_remote_ratio", type=float, default=0.30)
    parser.add_argument("--view_graph_refresh_interval", type=int, default=3)
    parser.add_argument("--view_graph_momentum", type=float, default=0.60)
    parser.add_argument(
        "--view_graph_directed",
        action="store_false",
        dest="view_graph_bidirectional",
        help="Use directed rather than mutual view scores",
    )
    parser.add_argument(
        "--no_view_graph_reference",
        action="store_false",
        dest="view_graph_protect_reference",
        help="Do not force frame zero into the middle-layer view graph",
    )
    parser.add_argument(
        "--view_graph_hard_routing",
        action="store_true",
        help="Restrict middle-layer blocks to selected graph views",
    )
    parser.add_argument("--view_graph_weight", type=float, default=0.15)
    parser.add_argument(
        "--no_view_graph_head_adaptive",
        action="store_false",
        dest="view_graph_head_adaptive",
    )
    parser.add_argument("--core_frame_radius", type=int, default=4,
                        help="Frame radius receiving the full geometry prior")
    parser.add_argument("--transition_frame_radius", type=int, default=12,
                        help="Frame radius where the geometry prior decays to zero")
    parser.add_argument("--geometry_weight", type=float, default=1.0,
                        help="Weight applied to the geometry prior")
    parser.add_argument(
        "--layer_geometry_weights",
        type=str,
        default=None,
        help=(
            "Comma-separated geometry weights for each global layer; "
            "overrides --geometry_weight_schedule"
        ),
    )
    parser.add_argument(
        "--geometry_weight_schedule",
        choices=["none", "early_low_late_high", "early_high_late_low", "middle_high"],
        default="none",
        help="Preset layer-wise geometry-weight schedule for soft geometry routing",
    )
    parser.add_argument("--geometry_decay", choices=["linear", "exponential"],
                        default="linear")
    parser.add_argument("--decay_gamma", type=float, default=0.25,
                        help="Exponential transition decay rate")
    parser.add_argument("--geometry_sigma", type=optional_float, default=None,
                        help="Frame-local token distance scale, or none for block-size default")
    parser.add_argument("--frame_normalize_importance", action="store_true",
                        help="Normalize pooled importance within each key frame")
    parser.add_argument("--distance_calibrate_importance", action="store_true",
                        help="Normalize importance by same-distance expected mass")
    parser.add_argument("--entropy_adaptive_geometry", action="store_true",
                        help="Scale geometry weight by per-query routing entropy")
    parser.add_argument(
        "--head_adaptive_geometry",
        action="store_true",
        help="Apply geometry prior mainly to low frame-entropy attention heads",
    )
    return parser


def set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def collect_runtime_provenance(vanilla_vggt=False):
    import sparse_vggt
    import vggt.models.vggt as vggt_model_module
    from sparse_vggt.utils import sparse_wrapper

    def describe_file(path):
        resolved = os.path.realpath(path)
        digest = hashlib.sha256()
        with open(resolved, "rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": resolved, "sha256": digest.hexdigest()}

    return {
        "eval_script": describe_file(__file__),
        "model_implementation": (
            "vanilla_vggt" if vanilla_vggt else "sparse_vggt"
        ),
        "vggt_model": describe_file(vggt_model_module.__file__),
        "sparse_vggt_package": describe_file(sparse_vggt.__file__),
        "sparse_wrapper": describe_file(sparse_wrapper.__file__),
        "official_qk_sparse_mask": os.environ.get(
            "SPARSE_VGGT_OFFICIAL_MASK", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "grouped_sdpa": os.environ.get(
            "SPARSE_VGGT_GROUPED_SDPA", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "triton_gather": os.environ.get(
            "SPARSE_VGGT_TRITON_GATHER", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "triton_pack": os.environ.get(
            "SPARSE_VGGT_TRITON_PACK", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "direct_attention": os.environ.get(
            "SPARSE_VGGT_DIRECT_ATTENTION", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "additive_direct_attention": os.environ.get(
            "SPARSE_VGGT_ADDITIVE_DIRECT_ATTENTION", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "additive_flash_carrier": os.environ.get(
            "SPARSE_VGGT_ADDITIVE_FLASH_CARRIER", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "additive_flash_positive_groups": os.environ.get(
            "SPARSE_VGGT_ADDITIVE_FLASH_POSITIVE_GROUPS", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "skip_routing_stats": os.environ.get(
            "SPARSE_VGGT_SKIP_ROUTING_STATS", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "direct_tile_profile": os.environ.get(
            "SPARSE_VGGT_DIRECT_TILE_PROFILE", "manual"
        ),
        "block_aligned_residual": os.environ.get(
            "SPARSE_VGGT_BLOCK_ALIGNED_RESIDUAL", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "direct_block_m": int(
            os.environ.get("SPARSE_VGGT_DIRECT_BLOCK_M", "128")
        ),
        "direct_block_n": int(
            os.environ.get("SPARSE_VGGT_DIRECT_BLOCK_N", "128")
        ),
        "direct_num_warps": int(
            os.environ.get("SPARSE_VGGT_DIRECT_NUM_WARPS", "0")
        ),
        "direct_num_stages": int(
            os.environ.get("SPARSE_VGGT_DIRECT_NUM_STAGES", "0")
        ),
        "direct_int32_indices": os.environ.get(
            "SPARSE_VGGT_DIRECT_INT32_INDICES", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "direct_qk_bf16": os.environ.get(
            "SPARSE_VGGT_DIRECT_QK_BF16", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "direct_qk_bf16_fused": os.environ.get(
            "SPARSE_VGGT_DIRECT_QK_BF16_FUSED", "0"
        ).lower() in {"1", "true", "yes", "on"},
        "hps_complement_correction": os.environ.get(
            "SPARSE_VGGT_HPS_COMPLEMENT_CORRECTION", "0"
        ).lower() in {"1", "true", "yes", "on"},
    }


def add_repo_paths(streamvggt_root, vggt_root):
    if vggt_root not in sys.path:
        sys.path.insert(0, vggt_root)

    stream_src = os.path.join(streamvggt_root, "src")
    sparse_src = os.path.join(os.path.dirname(__file__), "..", "src")
    if sparse_src not in sys.path:
        sys.path.insert(0, sparse_src)

    if stream_src not in sys.path:
        sys.path.append(stream_src)


def collect_block_sparsity(model):
    """Collect the last forward's actual final-map sparsity from each sparse layer."""
    stores = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(stores, dict):
        return None, None

    per_layer = {}
    for layer_idx, store in stores.items():
        if not isinstance(store, dict) or "sparsity" not in store:
            continue
        value = store["sparsity"]
        if torch.is_tensor(value):
            value = value.detach().float().item()
        per_layer[str(layer_idx)] = float(value)

    if not per_layer:
        return None, None
    return float(np.mean(list(per_layer.values()))), per_layer


def parse_layer_float_list(value, name, expected_len):
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != expected_len:
        raise ValueError(
            f"--{name} expects {expected_len} comma-separated values, got {len(parts)}"
        )
    try:
        values = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"--{name} contains a non-float value: {value}") from exc
    if any(not np.isfinite(item) or item < 0 for item in values):
        raise ValueError(f"--{name} values must be finite and non-negative")
    return values


def make_layer_geometry_weights(args, num_layers=24):
    explicit = parse_layer_float_list(
        args.layer_geometry_weights,
        "layer_geometry_weights",
        num_layers,
    )
    if explicit is not None:
        return explicit

    schedule = args.geometry_weight_schedule
    if schedule == "none":
        return None

    splits = (num_layers // 3, 2 * num_layers // 3)
    early_end, middle_end = splits
    if schedule == "early_low_late_high":
        multipliers = [0.5] * early_end
        multipliers += [1.0] * (middle_end - early_end)
        multipliers += [1.5] * (num_layers - middle_end)
    elif schedule == "early_high_late_low":
        multipliers = [1.5] * early_end
        multipliers += [1.0] * (middle_end - early_end)
        multipliers += [0.5] * (num_layers - middle_end)
    elif schedule == "middle_high":
        multipliers = [0.5] * early_end
        multipliers += [1.5] * (middle_end - early_end)
        multipliers += [0.5] * (num_layers - middle_end)
    else:
        raise ValueError(f"Unknown geometry weight schedule: {schedule}")

    return [args.geometry_weight * multiplier for multiplier in multipliers]


def build_model(args, device):
    from vggt.models.vggt import VGGT

    model = VGGT()
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt, strict=True)
    del ckpt

    if args.vanilla_vggt:
        model.sparse_aux_output_store = {}
        model.eval()
        return model.to(device)

    from sparse_vggt.models.vggt import sparse_aggregator_from_vggt
    from sparse_vggt.models.vggt_layerwise import (
        sparse_aggregator_from_vggt_layerwise,
    )

    # Parse layer_sparsity if provided
    layer_sparsity_ratios = None
    if args.layer_sparsity is not None:
        try:
            layer_sparsity_ratios = [float(x.strip()) for x in args.layer_sparsity.split(',')]
            print(f"Using layer-wise sparsity: {layer_sparsity_ratios}")
        except ValueError as e:
            print(f"Warning: Failed to parse --layer_sparsity '{args.layer_sparsity}': {e}")
            print("Falling back to uniform sparsity")
            layer_sparsity_ratios = None

    layer_geometry_weights = make_layer_geometry_weights(args)
    multiresolution_layer_samples = None
    if args.multiresolution_layer_samples:
        try:
            multiresolution_layer_samples = [
                int(value.strip())
                for value in args.multiresolution_layer_samples.split(",")
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "--multiresolution_layer_samples must contain integers"
            ) from error
    multiresolution_layer_pool_sizes = None
    if args.multiresolution_layer_pool_sizes:
        try:
            multiresolution_layer_pool_sizes = [
                int(value.strip())
                for value in args.multiresolution_layer_pool_sizes.split(",")
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "--multiresolution_layer_pool_sizes must contain integers"
            ) from error
    multiresolution_layer_local_radii = None
    if args.multiresolution_layer_local_radii:
        try:
            multiresolution_layer_local_radii = [
                int(value.strip())
                for value in args.multiresolution_layer_local_radii.split(",")
                if value.strip()
            ]
        except ValueError as error:
            raise ValueError(
                "--multiresolution_layer_local_radii must contain integers"
            ) from error
    try:
        scout_oracle_layers = tuple(
            int(value.strip())
            for value in args.scout_oracle_layers.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError("--scout_oracle_layers must be comma-separated integers") from error
    if args.analyze_scout_oracle and not scout_oracle_layers:
        raise ValueError("--scout_oracle_layers must not be empty")
    try:
        residual_budget_fine_refresh_layers = tuple(
            int(value.strip())
            for value in args.residual_budget_fine_refresh_layers.split(",")
            if value.strip()
        )
    except ValueError as error:
        raise ValueError(
            "--residual_budget_fine_refresh_layers must be "
            "comma-separated integers"
        ) from error
    if layer_geometry_weights is not None and not args.use_soft_geometry_routing:
        raise ValueError(
            "--layer_geometry_weights and --geometry_weight_schedule require "
            "--use_soft_geometry_routing"
        )
    if layer_geometry_weights is not None:
        print(f"Using layer-wise geometry weights: {layer_geometry_weights}")

    routed_modes = (
        args.use_radial_layerwise,
        args.use_distance_routed,
        args.use_soft_geometry_routing,
        args.use_preview_adaptive_routing,
        args.use_multiresolution_routing,
        args.use_residual_budget_routing,
        args.use_dual_path_routing,
        args.use_layerwise_hybrid_routing,
        args.use_local_layerwise_hybrid_routing,
        args.use_coverage_layerwise_routing,
        args.use_persistent_view_graph_routing,
    )
    analyze_block_selection = (
        args.analyze_block_selection or args.analyze_importance_blocks
    )
    if sum(routed_modes) > 1:
        raise ValueError(
            "--use_radial_layerwise, --use_distance_routed, and "
            "--use_soft_geometry_routing/--use_dual_path_routing/"
            "--use_preview_adaptive_routing/"
            "--use_multiresolution_routing/"
            "--use_residual_budget_routing/"
            "--use_layerwise_hybrid_routing/"
            "--use_local_layerwise_hybrid_routing/"
            "--use_coverage_layerwise_routing/"
            "--use_persistent_view_graph_routing are mutually exclusive"
        )
    if args.use_adaptive_slit_routing and not args.use_distance_routed:
        raise ValueError("--use_adaptive_slit_routing requires --use_distance_routed")
    if args.use_covariance_aware_importance and not args.use_distance_routed:
        raise ValueError(
            "--use_covariance_aware_importance requires --use_distance_routed"
        )
    if args.head_adaptive_geometry and not args.use_soft_geometry_routing:
        raise ValueError(
            "--head_adaptive_geometry requires --use_soft_geometry_routing"
        )

    # Use the extended aggregator for all routed and layer-wise modes.
    if (
        layer_sparsity_ratios is not None
        or args.use_radial_layerwise
        or args.use_distance_routed
        or args.use_soft_geometry_routing
        or args.use_preview_adaptive_routing
        or args.use_multiresolution_routing
        or args.use_residual_budget_routing
        or args.use_dual_path_routing
        or args.use_layerwise_hybrid_routing
        or args.use_local_layerwise_hybrid_routing
        or args.use_coverage_layerwise_routing
        or args.use_persistent_view_graph_routing
        or args.analyze_scout_oracle
    ):
        model.aggregator, aux_output_store = sparse_aggregator_from_vggt_layerwise(
            model.aggregator,
            layer_sparsity_ratios=layer_sparsity_ratios,
            sparse_ratio=args.sparse_ratio,
            cdf_threshold=args.cdf_threshold,
            pool_mode=args.pool_mode,
            use_hilbert=args.use_hilbert,
            aux_output=args.aux_output or analyze_block_selection,
            aux_sparsity_only=not args.disable_aux_sparsity_only,
            use_radial_layerwise=args.use_radial_layerwise,
            decay_factor=args.decay_factor,
            dense_neighbor=args.dense_neighbor,
            use_distance_routed=args.use_distance_routed,
            route_frame_threshold=args.route_frame_threshold,
            use_covariance_aware_importance=args.use_covariance_aware_importance,
            covariance_weight=args.covariance_weight,
            covariance_eps=args.covariance_eps,
            use_adaptive_slit_routing=args.use_adaptive_slit_routing,
            adaptive_slit_temporal_window=args.adaptive_slit_temporal_window,
            adaptive_slit_stable_quantile=args.adaptive_slit_stable_quantile,
            adaptive_slit_change_quantile=args.adaptive_slit_change_quantile,
            adaptive_slit_narrow_width=args.adaptive_slit_narrow_width,
            adaptive_slit_base_width=args.adaptive_slit_base_width,
            adaptive_slit_expand_width=args.adaptive_slit_expand_width,
            use_soft_geometry_routing=args.use_soft_geometry_routing,
            use_preview_adaptive_routing=args.use_preview_adaptive_routing,
            use_multiresolution_routing=args.use_multiresolution_routing,
            use_residual_budget_routing=(
                args.use_residual_budget_routing
            ),
            multiresolution_local_radius=(
                args.multiresolution_local_radius
            ),
            multiresolution_remote_pool_size=(
                args.multiresolution_remote_pool_size
            ),
            multiresolution_query_frame_chunk=(
                args.multiresolution_query_frame_chunk
            ),
            multiresolution_remote_mode=args.multiresolution_remote_mode,
            multiresolution_remote_samples_per_cell=(
                args.multiresolution_remote_samples_per_cell
            ),
            multiresolution_layer_samples=multiresolution_layer_samples,
            multiresolution_layer_pool_sizes=(
                multiresolution_layer_pool_sizes
            ),
            multiresolution_layer_local_radii=(
                multiresolution_layer_local_radii
            ),
            multiresolution_remote_phase_mode=(
                args.multiresolution_remote_phase_mode
            ),
            multiresolution_area_bias=args.multiresolution_area_bias,
            residual_budget_target_sparsity=(
                args.residual_budget_target_sparsity
            ),
            residual_budget_local_radius=(
                args.residual_budget_local_radius
            ),
            residual_budget_remote_pool_size=(
                args.residual_budget_remote_pool_size
            ),
            residual_budget_routing_parent_size=(
                args.residual_budget_routing_parent_size
            ),
            residual_budget_routing_phase_mode=(
                args.residual_budget_routing_phase_mode
            ),
            residual_budget_unified_incremental_service=(
                args.residual_budget_unified_incremental_service
            ),
            residual_budget_bounded_wait_reobservation=(
                args.residual_budget_bounded_wait_reobservation
            ),
            residual_budget_adaptive_parent_service=(
                args.residual_budget_adaptive_parent_service
            ),
            residual_budget_mixed_parent_execution=(
                args.residual_budget_mixed_parent_execution
            ),
            residual_budget_additive_parent_service=(
                args.residual_budget_additive_parent_service
            ),
            residual_budget_additive_full_upgrade_fraction=(
                args.residual_budget_additive_full_upgrade_fraction
            ),
            residual_budget_additive_parent_substitution_fraction=(
                args.residual_budget_additive_parent_substitution_fraction
            ),
            residual_budget_mixed_residuals_per_parent=(
                args.residual_budget_mixed_residuals_per_parent
            ),
            residual_budget_parent_hard_fraction=(
                args.residual_budget_parent_hard_fraction
            ),
            residual_budget_parent_cost_power=(
                args.residual_budget_parent_cost_power
            ),
            residual_budget_frame_balance_fraction=(
                args.residual_budget_frame_balance_fraction
            ),
            residual_budget_fine_refresh_layers=(
                residual_budget_fine_refresh_layers
            ),
            residual_budget_fine_refresh_sparsity=(
                args.residual_budget_fine_refresh_sparsity
            ),
            residual_budget_budget_neutral_fine_refresh=(
                args.residual_budget_budget_neutral_fine_refresh
            ),
            residual_budget_query_frame_chunk=(
                args.residual_budget_query_frame_chunk
            ),
            residual_budget_momentum=args.residual_budget_momentum,
            residual_budget_service_conditioned_momentum=(
                args.residual_budget_service_conditioned_momentum
            ),
            residual_budget_repayment=args.residual_budget_repayment,
            residual_budget_repayment_mode=(
                args.residual_budget_repayment_mode
            ),
            residual_budget_service_credit_scale=(
                args.residual_budget_service_credit_scale
            ),
            residual_budget_temperature=args.residual_budget_temperature,
            residual_budget_exact_fraction=(
                args.residual_budget_exact_fraction
            ),
            residual_budget_surface_weight=(
                args.residual_budget_surface_weight
            ),
            residual_budget_selection_granularity=(
                args.residual_budget_selection_granularity
            ),
            residual_budget_frame_detail_power=(
                args.residual_budget_frame_detail_power
            ),
            residual_budget_spatial_detail_power=(
                args.residual_budget_spatial_detail_power
            ),
            residual_budget_cell_scorer=args.residual_budget_cell_scorer,
            residual_budget_carrier_residual_balance=(
                args.residual_budget_carrier_residual_balance
            ),
            residual_budget_cell_refinement_phases=(
                args.residual_budget_cell_refinement_phases
            ),
            residual_budget_cell_precision_fraction=(
                args.residual_budget_cell_precision_fraction
            ),
            residual_budget_mass_conserving_refinement=(
                args.residual_budget_mass_conserving_refinement
            ),
            residual_budget_exact_mass_conserving_refinement=(
                args.residual_budget_exact_mass_conserving_refinement
            ),
            residual_budget_cell_importance_weight=(
                args.residual_budget_cell_importance_weight
            ),
            residual_budget_cell_importance_floor=(
                args.residual_budget_cell_importance_floor
            ),
            residual_budget_cell_importance_gate_threshold=(
                args.residual_budget_cell_importance_gate_threshold
            ),
            residual_budget_cell_importance_gate_temperature=(
                args.residual_budget_cell_importance_gate_temperature
            ),
            preview_redistribution_fraction=(
                args.preview_redistribution_fraction
            ),
            preview_activation_threshold=args.preview_activation_threshold,
            preview_local_radius=args.preview_local_radius,
            preview_local_fraction=args.preview_local_fraction,
            preview_geometry_weight=args.preview_geometry_weight,
            preview_head_protection=args.preview_head_protection,
            preview_protected_head_fraction=(
                args.preview_protected_head_fraction
            ),
            preview_donor_retention_threshold=(
                args.preview_donor_retention_threshold
            ),
            preview_donor_exchange_scope=args.preview_donor_exchange_scope,
            preview_receiver_gain_threshold=(
                args.preview_receiver_gain_threshold
            ),
            preview_exchange_gain_cost_ratio=(
                args.preview_exchange_gain_cost_ratio
            ),
            preview_head_exchange_cap_fraction=(
                args.preview_head_exchange_cap_fraction
            ),
            preview_layer_confidence_threshold=(
                args.preview_layer_confidence_threshold
            ),
            preview_exchange_layer_start=args.preview_exchange_layer_start,
            preview_exchange_layer_end=args.preview_exchange_layer_end,
            use_dual_path_routing=args.use_dual_path_routing,
            dual_path_local_radius=args.dual_path_local_radius,
            dual_path_min_local_fraction=args.dual_path_min_local_fraction,
            dual_path_max_local_fraction=args.dual_path_max_local_fraction,
            dual_path_layer_schedule=args.dual_path_layer_schedule,
            dual_path_context_geometry_weight=(
                args.dual_path_context_geometry_weight
            ),
            dual_path_context_schedule=args.dual_path_context_schedule,
            dual_path_context_gate=args.dual_path_context_gate,
            dual_path_context_alignment_threshold=(
                args.dual_path_context_alignment_threshold
            ),
            dual_path_context_alignment_temperature=(
                args.dual_path_context_alignment_temperature
            ),
            analyze_scout_oracle_routing=args.analyze_scout_oracle,
            scout_oracle_layers=scout_oracle_layers,
            scout_oracle_query_blocks=args.scout_oracle_query_blocks,
            scout_oracle_queries_per_block=(
                args.scout_oracle_queries_per_block
            ),
            scout_oracle_local_radius=args.scout_oracle_local_radius,
            scout_oracle_coarse_group_blocks=(
                args.scout_oracle_coarse_group_blocks
            ),
            scout_oracle_candidate_multiplier=(
                args.scout_oracle_candidate_multiplier
            ),
            use_layerwise_hybrid_routing=args.use_layerwise_hybrid_routing,
            use_local_layerwise_hybrid_routing=args.use_local_layerwise_hybrid_routing,
            use_coverage_layerwise_routing=args.use_coverage_layerwise_routing,
            use_persistent_view_graph_routing=args.use_persistent_view_graph_routing,
            hybrid_early_end=args.hybrid_early_end,
            hybrid_mid_end=args.hybrid_mid_end,
            hybrid_mid_sparse_ratio=args.hybrid_mid_sparse_ratio,
            hybrid_late_sparse_ratio=args.hybrid_late_sparse_ratio,
            hybrid_early_dense_neighbor=args.hybrid_early_dense_neighbor,
            hybrid_early_local_radius=args.hybrid_early_local_radius,
            mid_frame_coverage_ratio=args.mid_frame_coverage_ratio,
            view_graph_early_end=args.view_graph_early_end,
            view_graph_mid_end=args.view_graph_mid_end,
            view_graph_mid_sparse_ratio=args.view_graph_mid_sparse_ratio,
            view_graph_local_radius=args.view_graph_local_radius,
            view_graph_remote_topk=args.view_graph_remote_topk,
            view_graph_remote_ratio=args.view_graph_remote_ratio,
            view_graph_refresh_interval=args.view_graph_refresh_interval,
            view_graph_momentum=args.view_graph_momentum,
            view_graph_bidirectional=args.view_graph_bidirectional,
            view_graph_protect_reference=args.view_graph_protect_reference,
            view_graph_force_anchors=False,
            view_graph_hard_routing=args.view_graph_hard_routing,
            view_graph_weight=args.view_graph_weight,
            view_graph_head_adaptive=args.view_graph_head_adaptive,
            core_frame_radius=args.core_frame_radius,
            transition_frame_radius=args.transition_frame_radius,
            geometry_weight=args.geometry_weight,
            layer_geometry_weights=layer_geometry_weights,
            geometry_decay=args.geometry_decay,
            decay_gamma=args.decay_gamma,
            geometry_sigma=args.geometry_sigma,
            frame_normalize_importance=args.frame_normalize_importance,
            distance_calibrate_importance=args.distance_calibrate_importance,
            entropy_adaptive_geometry=args.entropy_adaptive_geometry,
            head_adaptive_geometry=args.head_adaptive_geometry,
            analyze_block_selection=analyze_block_selection,
        )
    else:
        model.aggregator, aux_output_store = sparse_aggregator_from_vggt(
            model.aggregator,
            sparse_ratio=args.sparse_ratio,
            cdf_threshold=args.cdf_threshold,
            pool_mode=args.pool_mode,
            use_hilbert=args.use_hilbert,
            aux_output=args.aux_output or analyze_block_selection,
            aux_sparsity_only=not args.disable_aux_sparsity_only,
            analyze_block_selection=analyze_block_selection,
        )
    model.sparse_aux_output_store = aux_output_store
    model.eval()
    return model.to(device)


class CudaAttentionTimer:
    """Time global attention module execution with CUDA events."""

    def __init__(self, model):
        self.enabled = False
        self.event_pairs = []
        self.active_starts = []
        self.handles = []
        for block in model.aggregator.global_blocks:
            self.handles.append(block.attn.register_forward_pre_hook(self._pre_forward))
            self.handles.append(block.attn.register_forward_hook(self._post_forward))

    def _pre_forward(self, _module, _inputs):
        if not self.enabled:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self.active_starts.append(start)

    def _post_forward(self, _module, _inputs, _output):
        if not self.enabled:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.event_pairs.append((self.active_starts.pop(), end))

    def begin(self):
        self.event_pairs = []
        self.active_starts = []
        self.enabled = True

    def finish(self):
        self.enabled = False
        if self.active_starts:
            raise RuntimeError("Attention timing hooks did not observe balanced forward calls")
        return float(sum(start.elapsed_time(end) for start, end in self.event_pairs))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def batch_views_to_images(batch):
    images = []
    for view in batch:
        # StreamVGGT eval stores images in [-1, 1]; VGGT expects [0, 1].
        img = (view["img"] + 1.0) / 2.0
        images.append(img)
    return torch.cat(images, dim=0)


def predictions_to_per_view_preds(predictions):
    pose_enc = predictions["pose_enc"]           # [1, S, 9]
    world_points = predictions["world_points"]   # [1, S, H, W, 3]
    world_points_conf = predictions["world_points_conf"]  # [1, S, H, W]
    depth = predictions["depth"]                 # [1, S, H, W, 1]
    depth_conf = predictions["depth_conf"]       # [1, S, H, W]

    preds = []
    num_views = pose_enc.shape[1]
    for s in range(num_views):
        preds.append(
            {
                "camera_pose": pose_enc[0, s],
                "pts3d": world_points[:, s],
                "conf": world_points_conf[:, s],
                "depth": depth[:, s],
                "depth_conf": depth_conf[:, s],
            }
        )
    return preds


def to_homogeneous(extrinsics):
    """Convert batched world-to-camera matrices from [N, 3, 4] to [N, 4, 4]."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], extrinsics.shape[0], axis=0)
    poses[:, :3, :4] = extrinsics
    return poses


def align_camera_poses_sim3(pred_c2w, gt_c2w):
    """Align predicted camera-to-world poses to ground truth using camera centers."""
    pred_centers = pred_c2w[:, :3, 3]
    gt_centers = gt_c2w[:, :3, 3]
    pred_mean = pred_centers.mean(axis=0)
    gt_mean = gt_centers.mean(axis=0)
    pred_zero = pred_centers - pred_mean
    gt_zero = gt_centers - gt_mean
    variance = np.mean(np.sum(pred_zero * pred_zero, axis=1))

    if variance < 1e-12:
        raise ValueError("Cannot align a degenerate predicted camera trajectory")

    covariance = (gt_zero.T @ pred_zero) / len(pred_centers)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        reflection[-1, -1] = -1
    rotation = u @ reflection @ vt
    scale = float(np.sum(singular_values * np.diag(reflection)) / variance)
    translation = gt_mean - scale * (rotation @ pred_mean)

    aligned = pred_c2w.copy()
    aligned[:, :3, :3] = rotation[None] @ pred_c2w[:, :3, :3]
    aligned[:, :3, 3] = (scale * (rotation @ pred_centers.T)).T + translation
    return aligned, scale


def rotation_error_degrees(rotation):
    cosine = (np.trace(rotation) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def trajectory_metrics(pred_extrinsics, gt_c2w):
    """Return Sim(3)-aligned trajectory metrics for one sequence."""
    pred_c2w = np.linalg.inv(to_homogeneous(pred_extrinsics))
    gt_c2w = np.asarray(gt_c2w, dtype=np.float64)
    aligned_pred, scale = align_camera_poses_sim3(pred_c2w, gt_c2w)

    translation_errors = np.linalg.norm(
        aligned_pred[:, :3, 3] - gt_c2w[:, :3, 3], axis=1
    )
    result = {
        "ate_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "alignment_scale": scale,
        "num_frames": int(len(gt_c2w)),
    }

    if len(gt_c2w) < 2:
        result.update({"rpe_trans_rmse_m": None, "rpe_rot_rmse_deg": None})
        return result

    rpe_translation = []
    rpe_rotation = []
    for idx in range(len(gt_c2w) - 1):
        gt_relative = np.linalg.inv(gt_c2w[idx]) @ gt_c2w[idx + 1]
        pred_relative = np.linalg.inv(aligned_pred[idx]) @ aligned_pred[idx + 1]
        error = np.linalg.inv(gt_relative) @ pred_relative
        rpe_translation.append(np.linalg.norm(error[:3, 3]))
        rpe_rotation.append(rotation_error_degrees(error[:3, :3]))

    result.update(
        {
            "rpe_trans_rmse_m": float(np.sqrt(np.mean(np.square(rpe_translation)))),
            "rpe_rot_rmse_deg": float(np.sqrt(np.mean(np.square(rpe_rotation)))),
        }
    )
    return result


BLOCK_SELECTION_ANALYSIS_METRICS = (
    "soft_geometry_mean_geometry_weight",
    "soft_geometry_reference_fraction",
    "layerwise_hybrid_stage_id",
    "layerwise_hybrid_config_sparse_ratio",
    "layerwise_hybrid_patch_sparsity",
    "layerwise_hybrid_local_fraction",
    "coverage_hybrid_stage_id",
    "coverage_hybrid_patch_sparsity",
    "coverage_hybrid_local_fraction",
    "coverage_hybrid_mid_anchors_per_query",
    "coverage_hybrid_mid_anchor_fraction",
    "persistent_view_graph_stage_id",
    "persistent_view_graph_patch_sparsity",
    "persistent_view_graph_refresh",
    "persistent_view_graph_views_per_query",
    "persistent_view_graph_remote_views_per_query",
    "persistent_view_graph_anchors_per_query",
    "persistent_view_graph_score_mean",
    "persistent_view_graph_bias_std",
    "persistent_view_graph_head_gate_mean",
    "importance_patch_sparsity",
    "importance_selected_same_frame_fraction",
    "importance_selected_adjacent_frame_fraction",
    "importance_selected_near_2_4_fraction",
    "importance_selected_mid_5_12_fraction",
    "importance_selected_far_gt12_fraction",
    "importance_selected_mean_frame_distance",
    "importance_key_frame_entropy",
    "importance_unique_key_frames_per_query",
    "importance_key_frame_coverage_fraction",
    "importance_selected_cross_frame_k_fraction",
    "importance_selected_radial_fraction",
    "importance_radial_available_fraction",
    "importance_selected_special_mixed_k_fraction",
)
IMPORTANCE_BLOCK_ANALYSIS_METRICS = BLOCK_SELECTION_ANALYSIS_METRICS


def get_block_selection_mode(args):
    if args.use_residual_budget_routing:
        return "residual_debt_budget"
    if args.use_multiresolution_routing:
        return "multiresolution_interaction"
    if args.use_dual_path_routing:
        return "role_adaptive_dual_path"
    if args.use_soft_geometry_routing:
        return "soft_geometry"
    if args.use_distance_routed:
        return "distance_routed"
    if args.use_radial_layerwise:
        return "radial_layerwise"
    if args.use_layerwise_hybrid_routing:
        return "layerwise_hybrid"
    if args.use_local_layerwise_hybrid_routing:
        return "local_layerwise_hybrid"
    if args.use_coverage_layerwise_routing:
        return "coverage_layerwise"
    if args.use_persistent_view_graph_routing:
        return "persistent_view_graph"
    return "importance"


def block_selection_analysis_enabled(args):
    return args.analyze_block_selection or args.analyze_importance_blocks


def _metric_to_float(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu().item())
    return float(value)


def collect_block_selection_analysis(model, selection_mode=None):
    """Return averaged block-selection diagnostics and per-layer rows."""
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return {}, []

    values = defaultdict(list)
    layer_rows = []
    for layer_idx, layer_store in store.items():
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer_idx)}
        if selection_mode is not None:
            row["block_selection_mode"] = selection_mode
        for metric in BLOCK_SELECTION_ANALYSIS_METRICS:
            if metric not in layer_store:
                continue
            value = _metric_to_float(layer_store.get(metric))
            if value is None:
                continue
            row[metric] = value
            values[metric].append(value)
        if len(row) > 1:
            layer_rows.append(row)

    summary = {
        metric: float(np.mean(metric_values))
        for metric, metric_values in values.items()
        if metric_values
    }
    return summary, layer_rows


def collect_importance_block_analysis(model):
    """Backward-compatible wrapper for older callers."""
    return collect_block_selection_analysis(model, selection_mode="importance")


def write_block_selection_analysis_csv(pose_records, csv_path):
    rows = []
    for record_idx, record in enumerate(pose_records):
        for layer_row in record.get("block_selection_analysis_per_layer", []):
            row = {
                "record_idx": record_idx,
                "scene_id": record.get("scene_id", ""),
                "block_selection_mode": record.get("block_selection_mode", ""),
            }
            row.update(layer_row)
            rows.append(row)

    if not rows:
        return

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = [
        "record_idx",
        "scene_id",
        "block_selection_mode",
        "layer",
        *BLOCK_SELECTION_ANALYSIS_METRICS,
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_importance_block_analysis_csv(pose_records, csv_path):
    """Backward-compatible wrapper for older callers."""
    write_block_selection_analysis_csv(pose_records, csv_path)


def summarize_pose_records(records):
    if not records:
        raise ValueError("No sequences were evaluated; check --scene and --seq_id filters")
    summary = {}
    for metric in (
        "ate_rmse_m",
        "rpe_trans_rmse_m",
        "rpe_rot_rmse_deg",
        "inference_ms",
        "attention_ms",
        "block_sparsity",
        "soft_geometry_patch_sparsity",
        "soft_geometry_core_fraction",
        "soft_geometry_transition_fraction",
        "soft_geometry_far_fraction",
        "soft_geometry_mean_prior",
        "soft_geometry_mean_entropy",
        "soft_geometry_mean_geometry_weight",
        "soft_geometry_head_gate_mean",
        "dual_path_patch_sparsity",
        "dual_path_local_fraction",
        "dual_path_remote_fraction",
        "dual_path_requested_local_fraction",
        "dual_path_local_blocks_per_query",
        "dual_path_frame_entropy",
        "dual_path_frame_entropy_head_std",
        "dual_path_local_affinity",
        "dual_path_local_affinity_head_std",
        "dual_path_head_role",
        "dual_path_head_role_std",
        "dual_path_layer_gate",
        "dual_path_context_geometry_weight",
        "dual_path_effective_context_geometry_weight",
        "dual_path_routed_context_geometry_weight",
        "dual_path_context_gate_mean",
        "dual_path_context_alignment",
        "dual_path_context_alignment_std",
        "dual_path_budget_error",
        "scout_oracle_sampled_queries",
        "scout_oracle_block_budget",
        "scout_oracle_importance_mass_recall",
        "scout_oracle_importance_topk_recall",
        "scout_oracle_refined_mass_recall",
        "scout_oracle_refined_topk_recall",
        "scout_oracle_candidate_fraction",
        "scout_oracle_need_mean",
        "scout_oracle_remote_mass_mean",
        "scout_value_need_correlation",
        "scout_value_need_spearman",
        "scout_remote_mass_correlation",
        "scout_hard_query_recall",
        "scout_block_value_need_correlation",
        "scout_block_value_need_spearman",
        "scout_block_hard_query_recall",
        "scout_fixed_block_mass_recall",
        "scout_adaptive_block_mass_recall",
        "scout_oracle_adaptive_block_mass_recall",
        "scout_adaptive_block_mass_gain",
        "scout_adaptive_mean_block_budget",
        "scout_block_sparse_error_correlation",
        "scout_block_sparse_error_spearman",
        "scout_block_sparse_error_hard_recall",
        "scout_fixed_output_error",
        "scout_adaptive_output_error",
        "scout_oracle_adaptive_output_error",
        "scout_adaptive_output_error_gain",
        "scout_preview_error_correlation",
        "scout_preview_error_spearman",
        "scout_preview_hard_query_recall",
        "scout_tail_mass_error_correlation",
        "scout_entropy_error_correlation",
        "scout_preview_adaptive_output_error",
        "scout_preview_output_error_gain",
        "scout_preview_mean_block_budget",
        "scout_preview_error_mean",
        "scout_preview_error_std",
        "scout_value_detail_mean",
        "scout_value_detail_preview_gap_correlation",
        "scout_preview_protected_output_error",
        "scout_preview_protected_output_error_gain",
        "scout_preview_adaptive_output_error_r010",
        "scout_preview_adaptive_output_error_r015",
        "scout_preview_output_error_gain_r010",
        "scout_preview_output_error_gain_r015",
        "preview_adaptive_patch_sparsity",
        "preview_adaptive_error_mean",
        "preview_adaptive_error_std",
        "preview_adaptive_active_fraction",
        "preview_adaptive_low_budget",
        "preview_adaptive_high_budget",
        "preview_adaptive_mean_budget",
        "preview_adaptive_budget_error",
        "preview_adaptive_local_fraction",
        "preview_adaptive_geometry_weight",
        "preview_adaptive_protected_head_fraction",
        "preview_adaptive_value_detail_mean",
        "preview_adaptive_protected_value_detail",
        "preview_adaptive_protected_local_affinity",
        "preview_adaptive_donor_eligible_fraction",
        "preview_adaptive_exchange_fraction",
        "preview_adaptive_donor_retention",
        "preview_adaptive_donor_retention_threshold",
        "preview_adaptive_donor_cost",
        "preview_adaptive_receiver_gain",
        "preview_adaptive_receiver_gain_fraction",
        "preview_adaptive_receiver_gain_threshold",
        "preview_adaptive_exchange_gain_cost_ratio",
        "preview_adaptive_head_exchange_cap_fraction",
        "preview_adaptive_layer_confidence",
        "preview_adaptive_layer_confidence_threshold",
        "preview_adaptive_max_head_exchange_fraction",
        "preview_adaptive_exchange_enabled",
        "preview_adaptive_layer_exchange",
        "preview_adaptive_head_budget_std",
        "multiresolution_patch_sparsity",
        "multiresolution_effective_sparsity",
        "multiresolution_local_exact_fraction",
        "multiresolution_remote_pooled_fraction",
        "multiresolution_mean_patch_keys",
        "multiresolution_pooled_tokens_per_frame",
        "multiresolution_carrier_tokens",
        "multiresolution_local_radius",
        "multiresolution_remote_pool_size",
        "multiresolution_remote_mode_id",
        "multiresolution_remote_phase",
        "multiresolution_remote_samples_per_cell",
        "multiresolution_area_bias",
        "multiresolution_query_aligned",
        "residual_budget_patch_sparsity",
        "residual_budget_effective_sparsity",
        "residual_budget_target_sparsity",
        "residual_budget_budget_error",
        "residual_budget_budget_feasible",
        "residual_budget_budget_floor_retained_fraction",
        "residual_budget_budget_floor_sparsity",
        "residual_budget_protected_pair_fraction",
        "residual_budget_protected_dense_fraction_of_dense",
        "residual_budget_remote_coarse_fraction_of_dense",
        "residual_budget_residual_fraction_of_dense",
        "residual_budget_local_exact_fraction",
        "residual_budget_coarse_fraction",
        "residual_budget_exact_fraction",
        "residual_budget_residual_fraction",
        "residual_budget_mean_patch_keys",
        "residual_budget_coarse_tokens_per_frame",
        "residual_budget_grouped_parent_routing",
        "residual_budget_unified_incremental_service",
        "residual_budget_bounded_wait_reobservation",
        "residual_budget_reobservation_service_cycle_layers",
        "residual_budget_reobservation_parent_age",
        "residual_budget_reobservation_max_parent_age",
        "residual_budget_reobservation_overdue_parent_fraction",
        "residual_budget_reobservation_selected_overdue_fraction",
        "residual_budget_reobservation_forced_parent_fraction",
        "residual_budget_reobservation_full_parent_fraction",
        "residual_budget_unified_active_parents",
        "residual_budget_unified_mean_service_depth",
        "residual_budget_adaptive_parent_service",
        "residual_budget_mixed_parent_execution",
        "residual_budget_additive_parent_service",
        "residual_budget_additive_full_upgrade_fraction",
        "residual_budget_additive_parent_substitution_fraction",
        "residual_budget_additive_parent_substitutions_per_query",
        "residual_budget_fixed_anchor_head_fraction",
        "residual_budget_fixed_child_anchor",
        "residual_budget_routing_phase_mode_id",
        "residual_budget_mixed_residuals_per_parent_config",
        "residual_budget_mixed_hard_parents_per_query",
        "residual_budget_mixed_promotion_budget",
        "residual_budget_mixed_residual_budget",
        "residual_budget_mixed_residuals_per_parent",
        "residual_budget_mixed_realized_extra_budget",
        "residual_budget_mixed_unspent_budget",
        "residual_budget_parent_hard_fraction_config",
        "residual_budget_parent_cost_power",
        "residual_budget_frame_balance_fraction",
        "residual_budget_fine_refresh",
        "residual_budget_frame_balance_budget",
        "residual_budget_frame_balance_service",
        "residual_budget_frame_balance_candidates",
        "residual_budget_remote_frame_coverage",
        "residual_budget_zero_service_frame_fraction",
        "residual_budget_frame_service_cv",
        "residual_budget_service_conditioned_momentum",
        "residual_budget_effective_momentum",
        "residual_budget_next_effective_momentum",
        *DEBT_MEMORY_DIAGNOSTIC_METRICS,
        *PV_IMPORTANCE_ALIGNMENT_METRICS,
        *CARRIER_COMPENSATION_OBSERVER_METRICS,
        *PROJECTED_QK_METRICS,
        "residual_budget_grouped_cost_aware",
        "residual_budget_grouped_mean_service_cost",
        "residual_budget_grouped_capacity_promotion_fraction",
        "residual_budget_grouped_parent_relevance",
        "residual_budget_parent_relevance_entropy",
        "residual_budget_parent_hard_fraction",
        "residual_budget_parent_heterogeneity",
        "residual_budget_local_radius",
        "residual_budget_exact_frames_per_query",
        "residual_budget_phase_frames_per_query",
        "residual_budget_exact_budget_fraction",
        "residual_budget_extra_frames_per_query",
        "residual_budget_frame_detail",
        "residual_budget_debt",
        "residual_budget_debt_after_repayment",
        "residual_budget_debt_repaid_fraction",
        "residual_budget_repayment_mode_id",
        "residual_budget_service_credit_scale",
        "residual_budget_service_credit",
        "residual_budget_negative_credit_fraction",
        "residual_budget_service_cycle_layers",
        "residual_budget_service_age",
        "residual_budget_selected_service_age",
        "residual_budget_overdue_service_fraction",
        "residual_budget_frontier_service_fraction",
        "residual_budget_frontier_overdue_selected_fraction",
        "residual_budget_selected_need",
        "residual_budget_selected_surface_need",
        "residual_budget_surface_weight",
        "residual_budget_cell_coverage_fraction",
        "residual_budget_residual_tokens_per_query",
        "residual_budget_selection_granularity_id",
        "residual_budget_frame_detail_power",
        "residual_budget_spatial_detail_power",
        "residual_budget_cell_scorer_id",
        "residual_budget_carrier_residual_balance",
        "residual_budget_cell_refinement_phases",
        "residual_budget_cell_precision_fraction",
        "residual_budget_mass_conserving_refinement",
        "residual_budget_exact_mass_conserving_refinement",
        "residual_budget_effective_service_credit",
        "residual_budget_service_credit_conservation_error",
        "residual_budget_direct_attention",
        "residual_budget_additive_direct_attention",
        "residual_budget_compiled_carrier_attention",
        "residual_budget_compiled_carrier_value_bf16",
        "residual_budget_compiled_carrier_precast_value_bf16",
        "residual_budget_compiled_carrier_dense_block_n",
        "residual_budget_compiled_carrier_parent_block_n",
        "residual_budget_compiled_carrier_child_block_n",
        "residual_budget_static_geometry_cache",
        "residual_budget_phase_compiled_layout",
        "residual_budget_additive_flash_carrier",
        "residual_budget_additive_flash_positive_groups",
        "residual_budget_hps_complement_correction",
        "residual_budget_hps_selected_cell_fraction",
        "residual_budget_mixed_direct_attention",
        "residual_budget_direct_tile_profile_id",
        "residual_budget_block_aligned_residual",
        "residual_budget_direct_block_m",
        "residual_budget_direct_block_n",
        "residual_budget_direct_qk_bf16",
        "residual_budget_precision_cells_per_query",
        "residual_budget_selected_precision_importance",
        "residual_budget_cell_importance_weight",
        "residual_budget_cell_importance_floor",
        "residual_budget_cell_importance_gate_threshold",
        "residual_budget_cell_importance_gate",
        "residual_budget_cell_importance_effective_weight",
        "residual_budget_base_need_concentration",
        "residual_budget_coverage_fraction",
        "residual_budget_coverage_gain",
        "residual_budget_selection_new_fraction",
        "residual_budget_uncovered_fraction",
        "adaptive_slit_stable_fraction",
        "adaptive_slit_change_fraction",
        "adaptive_slit_selected_near_fraction",
        "adaptive_slit_dense_core_fraction",
        "covariance_k_variance_mean",
        "covariance_k_variance_std",
        "covariance_score_shift_std",
        *IMPORTANCE_BLOCK_ANALYSIS_METRICS,
    ):
        values = [record[metric] for record in records if record.get(metric) is not None]
        summary[metric] = float(np.mean(values)) if values else None
    summary["total_inference_ms"] = float(sum(record["inference_ms"] for record in records))
    attention_values = [
        record["attention_ms"] for record in records if record.get("attention_ms") is not None
    ]
    summary["total_attention_ms"] = (
        float(sum(attention_values)) if len(attention_values) == len(records) else None
    )
    summary["peak_memory_mb"] = float(max(record["peak_memory_mb"] for record in records))
    summary["num_sequences"] = len(records)
    summary["num_frames"] = int(sum(record["num_frames"] for record in records))
    return summary


def write_metrics_checkpoint(
    path,
    args,
    runtime_provenance,
    pose_records,
    reconstruction_records,
):
    """Atomically preserve completed scenes while a long evaluation runs."""
    payload = {
        "config": vars(args),
        "runtime_provenance": runtime_provenance,
        "records": pose_records,
        "summary": summarize_pose_records(pose_records),
        "reconstruction_records": reconstruction_records,
        "completed_scene_ids": [
            record["scene_id"] for record in pose_records
        ],
    }
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(temporary_path, path)


def collect_sparse_aux_metrics(model):
    """Average scalar routing diagnostics across modified global layers."""
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return {}

    output_names = {
        "sparsity": "block_sparsity",
        "soft_geometry_patch_sparsity": "soft_geometry_patch_sparsity",
        "soft_geometry_core_fraction": "soft_geometry_core_fraction",
        "soft_geometry_transition_fraction": "soft_geometry_transition_fraction",
        "soft_geometry_far_fraction": "soft_geometry_far_fraction",
        "soft_geometry_mean_prior": "soft_geometry_mean_prior",
        "soft_geometry_mean_entropy": "soft_geometry_mean_entropy",
        "soft_geometry_mean_geometry_weight": "soft_geometry_mean_geometry_weight",
        "soft_geometry_head_gate_mean": "soft_geometry_head_gate_mean",
        "dual_path_patch_sparsity": "dual_path_patch_sparsity",
        "dual_path_local_fraction": "dual_path_local_fraction",
        "dual_path_remote_fraction": "dual_path_remote_fraction",
        "dual_path_requested_local_fraction": (
            "dual_path_requested_local_fraction"
        ),
        "dual_path_local_blocks_per_query": "dual_path_local_blocks_per_query",
        "dual_path_frame_entropy": "dual_path_frame_entropy",
        "dual_path_frame_entropy_head_std": (
            "dual_path_frame_entropy_head_std"
        ),
        "dual_path_local_affinity": "dual_path_local_affinity",
        "dual_path_local_affinity_head_std": (
            "dual_path_local_affinity_head_std"
        ),
        "dual_path_head_role": "dual_path_head_role",
        "dual_path_head_role_std": "dual_path_head_role_std",
        "dual_path_layer_gate": "dual_path_layer_gate",
        "dual_path_context_geometry_weight": (
            "dual_path_context_geometry_weight"
        ),
        "dual_path_effective_context_geometry_weight": (
            "dual_path_effective_context_geometry_weight"
        ),
        "dual_path_routed_context_geometry_weight": (
            "dual_path_routed_context_geometry_weight"
        ),
        "dual_path_context_gate_mean": "dual_path_context_gate_mean",
        "dual_path_context_alignment": "dual_path_context_alignment",
        "dual_path_context_alignment_std": "dual_path_context_alignment_std",
        "dual_path_budget_error": "dual_path_budget_error",
        "scout_oracle_sampled_queries": "scout_oracle_sampled_queries",
        "scout_oracle_block_budget": "scout_oracle_block_budget",
        "scout_oracle_importance_mass_recall": (
            "scout_oracle_importance_mass_recall"
        ),
        "scout_oracle_importance_topk_recall": (
            "scout_oracle_importance_topk_recall"
        ),
        "scout_oracle_refined_mass_recall": (
            "scout_oracle_refined_mass_recall"
        ),
        "scout_oracle_refined_topk_recall": (
            "scout_oracle_refined_topk_recall"
        ),
        "scout_oracle_candidate_fraction": "scout_oracle_candidate_fraction",
        "scout_oracle_need_mean": "scout_oracle_need_mean",
        "scout_oracle_remote_mass_mean": "scout_oracle_remote_mass_mean",
        "scout_value_need_correlation": "scout_value_need_correlation",
        "scout_value_need_spearman": "scout_value_need_spearman",
        "scout_remote_mass_correlation": "scout_remote_mass_correlation",
        "scout_hard_query_recall": "scout_hard_query_recall",
        "scout_block_value_need_correlation": (
            "scout_block_value_need_correlation"
        ),
        "scout_block_value_need_spearman": (
            "scout_block_value_need_spearman"
        ),
        "scout_block_hard_query_recall": "scout_block_hard_query_recall",
        "scout_fixed_block_mass_recall": "scout_fixed_block_mass_recall",
        "scout_adaptive_block_mass_recall": (
            "scout_adaptive_block_mass_recall"
        ),
        "scout_oracle_adaptive_block_mass_recall": (
            "scout_oracle_adaptive_block_mass_recall"
        ),
        "scout_adaptive_block_mass_gain": "scout_adaptive_block_mass_gain",
        "scout_adaptive_mean_block_budget": (
            "scout_adaptive_mean_block_budget"
        ),
        "scout_block_sparse_error_correlation": (
            "scout_block_sparse_error_correlation"
        ),
        "scout_block_sparse_error_spearman": (
            "scout_block_sparse_error_spearman"
        ),
        "scout_block_sparse_error_hard_recall": (
            "scout_block_sparse_error_hard_recall"
        ),
        "scout_fixed_output_error": "scout_fixed_output_error",
        "scout_adaptive_output_error": "scout_adaptive_output_error",
        "scout_oracle_adaptive_output_error": (
            "scout_oracle_adaptive_output_error"
        ),
        "scout_adaptive_output_error_gain": (
            "scout_adaptive_output_error_gain"
        ),
        "scout_preview_error_correlation": "scout_preview_error_correlation",
        "scout_preview_error_spearman": "scout_preview_error_spearman",
        "scout_preview_hard_query_recall": "scout_preview_hard_query_recall",
        "scout_tail_mass_error_correlation": (
            "scout_tail_mass_error_correlation"
        ),
        "scout_entropy_error_correlation": (
            "scout_entropy_error_correlation"
        ),
        "scout_preview_adaptive_output_error": (
            "scout_preview_adaptive_output_error"
        ),
        "scout_preview_output_error_gain": "scout_preview_output_error_gain",
        "scout_preview_mean_block_budget": "scout_preview_mean_block_budget",
        "scout_preview_error_mean": "scout_preview_error_mean",
        "scout_preview_error_std": "scout_preview_error_std",
        "scout_value_detail_mean": "scout_value_detail_mean",
        "scout_value_detail_preview_gap_correlation": (
            "scout_value_detail_preview_gap_correlation"
        ),
        "scout_preview_protected_output_error": (
            "scout_preview_protected_output_error"
        ),
        "scout_preview_protected_output_error_gain": (
            "scout_preview_protected_output_error_gain"
        ),
        "scout_preview_adaptive_output_error_r010": (
            "scout_preview_adaptive_output_error_r010"
        ),
        "scout_preview_adaptive_output_error_r015": (
            "scout_preview_adaptive_output_error_r015"
        ),
        "scout_preview_output_error_gain_r010": (
            "scout_preview_output_error_gain_r010"
        ),
        "scout_preview_output_error_gain_r015": (
            "scout_preview_output_error_gain_r015"
        ),
        "preview_adaptive_patch_sparsity": "preview_adaptive_patch_sparsity",
        "preview_adaptive_error_mean": "preview_adaptive_error_mean",
        "preview_adaptive_error_std": "preview_adaptive_error_std",
        "preview_adaptive_active_fraction": "preview_adaptive_active_fraction",
        "preview_adaptive_low_budget": "preview_adaptive_low_budget",
        "preview_adaptive_high_budget": "preview_adaptive_high_budget",
        "preview_adaptive_mean_budget": "preview_adaptive_mean_budget",
        "preview_adaptive_budget_error": "preview_adaptive_budget_error",
        "preview_adaptive_local_fraction": "preview_adaptive_local_fraction",
        "preview_adaptive_geometry_weight": (
            "preview_adaptive_geometry_weight"
        ),
        "preview_adaptive_protected_head_fraction": (
            "preview_adaptive_protected_head_fraction"
        ),
        "preview_adaptive_value_detail_mean": (
            "preview_adaptive_value_detail_mean"
        ),
        "preview_adaptive_protected_value_detail": (
            "preview_adaptive_protected_value_detail"
        ),
        "preview_adaptive_protected_local_affinity": (
            "preview_adaptive_protected_local_affinity"
        ),
        "preview_adaptive_donor_eligible_fraction": (
            "preview_adaptive_donor_eligible_fraction"
        ),
        "preview_adaptive_exchange_fraction": (
            "preview_adaptive_exchange_fraction"
        ),
        "preview_adaptive_donor_retention": (
            "preview_adaptive_donor_retention"
        ),
        "preview_adaptive_donor_retention_threshold": (
            "preview_adaptive_donor_retention_threshold"
        ),
        "preview_adaptive_donor_cost": "preview_adaptive_donor_cost",
        "preview_adaptive_receiver_gain": "preview_adaptive_receiver_gain",
        "preview_adaptive_receiver_gain_fraction": (
            "preview_adaptive_receiver_gain_fraction"
        ),
        "preview_adaptive_receiver_gain_threshold": (
            "preview_adaptive_receiver_gain_threshold"
        ),
        "preview_adaptive_exchange_gain_cost_ratio": (
            "preview_adaptive_exchange_gain_cost_ratio"
        ),
        "preview_adaptive_head_exchange_cap_fraction": (
            "preview_adaptive_head_exchange_cap_fraction"
        ),
        "preview_adaptive_layer_confidence": (
            "preview_adaptive_layer_confidence"
        ),
        "preview_adaptive_layer_confidence_threshold": (
            "preview_adaptive_layer_confidence_threshold"
        ),
        "preview_adaptive_max_head_exchange_fraction": (
            "preview_adaptive_max_head_exchange_fraction"
        ),
        "preview_adaptive_exchange_enabled": (
            "preview_adaptive_exchange_enabled"
        ),
        "preview_adaptive_layer_exchange": (
            "preview_adaptive_layer_exchange"
        ),
        "preview_adaptive_head_budget_std": (
            "preview_adaptive_head_budget_std"
        ),
        "multiresolution_patch_sparsity": "multiresolution_patch_sparsity",
        "multiresolution_effective_sparsity": (
            "multiresolution_effective_sparsity"
        ),
        "multiresolution_local_exact_fraction": (
            "multiresolution_local_exact_fraction"
        ),
        "multiresolution_remote_pooled_fraction": (
            "multiresolution_remote_pooled_fraction"
        ),
        "multiresolution_mean_patch_keys": "multiresolution_mean_patch_keys",
        "multiresolution_pooled_tokens_per_frame": (
            "multiresolution_pooled_tokens_per_frame"
        ),
        "multiresolution_carrier_tokens": "multiresolution_carrier_tokens",
        "multiresolution_local_radius": "multiresolution_local_radius",
        "multiresolution_remote_pool_size": (
            "multiresolution_remote_pool_size"
        ),
        "multiresolution_remote_mode_id": "multiresolution_remote_mode_id",
        "multiresolution_remote_phase": "multiresolution_remote_phase",
        "multiresolution_remote_samples_per_cell": (
            "multiresolution_remote_samples_per_cell"
        ),
        "multiresolution_area_bias": "multiresolution_area_bias",
        "multiresolution_query_aligned": "multiresolution_query_aligned",
        "residual_budget_patch_sparsity": "residual_budget_patch_sparsity",
        "residual_budget_effective_sparsity": (
            "residual_budget_effective_sparsity"
        ),
        "residual_budget_target_sparsity": (
            "residual_budget_target_sparsity"
        ),
        "residual_budget_budget_error": "residual_budget_budget_error",
        "residual_budget_budget_feasible": "residual_budget_budget_feasible",
        "residual_budget_budget_floor_retained_fraction": (
            "residual_budget_budget_floor_retained_fraction"
        ),
        "residual_budget_budget_floor_sparsity": (
            "residual_budget_budget_floor_sparsity"
        ),
        "residual_budget_protected_pair_fraction": (
            "residual_budget_protected_pair_fraction"
        ),
        "residual_budget_protected_dense_fraction_of_dense": (
            "residual_budget_protected_dense_fraction_of_dense"
        ),
        "residual_budget_remote_coarse_fraction_of_dense": (
            "residual_budget_remote_coarse_fraction_of_dense"
        ),
        "residual_budget_residual_fraction_of_dense": (
            "residual_budget_residual_fraction_of_dense"
        ),
        "residual_budget_local_exact_fraction": (
            "residual_budget_local_exact_fraction"
        ),
        "residual_budget_coarse_fraction": "residual_budget_coarse_fraction",
        "residual_budget_exact_fraction": "residual_budget_exact_fraction",
        "residual_budget_residual_fraction": (
            "residual_budget_residual_fraction"
        ),
        "residual_budget_mean_patch_keys": "residual_budget_mean_patch_keys",
        "residual_budget_coarse_tokens_per_frame": (
            "residual_budget_coarse_tokens_per_frame"
        ),
        "residual_budget_grouped_parent_routing": (
            "residual_budget_grouped_parent_routing"
        ),
        "residual_budget_unified_incremental_service": (
            "residual_budget_unified_incremental_service"
        ),
        "residual_budget_bounded_wait_reobservation": (
            "residual_budget_bounded_wait_reobservation"
        ),
        "residual_budget_reobservation_service_cycle_layers": (
            "residual_budget_reobservation_service_cycle_layers"
        ),
        "residual_budget_reobservation_parent_age": (
            "residual_budget_reobservation_parent_age"
        ),
        "residual_budget_reobservation_max_parent_age": (
            "residual_budget_reobservation_max_parent_age"
        ),
        "residual_budget_reobservation_overdue_parent_fraction": (
            "residual_budget_reobservation_overdue_parent_fraction"
        ),
        "residual_budget_reobservation_selected_overdue_fraction": (
            "residual_budget_reobservation_selected_overdue_fraction"
        ),
        "residual_budget_reobservation_forced_parent_fraction": (
            "residual_budget_reobservation_forced_parent_fraction"
        ),
        "residual_budget_reobservation_full_parent_fraction": (
            "residual_budget_reobservation_full_parent_fraction"
        ),
        "residual_budget_unified_active_parents": (
            "residual_budget_unified_active_parents"
        ),
        "residual_budget_unified_mean_service_depth": (
            "residual_budget_unified_mean_service_depth"
        ),
        "residual_budget_adaptive_parent_service": (
            "residual_budget_adaptive_parent_service"
        ),
        "residual_budget_mixed_parent_execution": (
            "residual_budget_mixed_parent_execution"
        ),
        "residual_budget_additive_parent_service": (
            "residual_budget_additive_parent_service"
        ),
        "residual_budget_additive_full_upgrade_fraction": (
            "residual_budget_additive_full_upgrade_fraction"
        ),
        "residual_budget_additive_parent_substitution_fraction": (
            "residual_budget_additive_parent_substitution_fraction"
        ),
        "residual_budget_additive_parent_substitutions_per_query": (
            "residual_budget_additive_parent_substitutions_per_query"
        ),
        "residual_budget_fixed_anchor_head_fraction": (
            "residual_budget_fixed_anchor_head_fraction"
        ),
        "residual_budget_fixed_child_anchor": (
            "residual_budget_fixed_child_anchor"
        ),
        "residual_budget_routing_phase_mode_id": (
            "residual_budget_routing_phase_mode_id"
        ),
        "residual_budget_mixed_residuals_per_parent_config": (
            "residual_budget_mixed_residuals_per_parent_config"
        ),
        "residual_budget_mixed_hard_parents_per_query": (
            "residual_budget_mixed_hard_parents_per_query"
        ),
        "residual_budget_mixed_promotion_budget": (
            "residual_budget_mixed_promotion_budget"
        ),
        "residual_budget_mixed_residual_budget": (
            "residual_budget_mixed_residual_budget"
        ),
        "residual_budget_mixed_residuals_per_parent": (
            "residual_budget_mixed_residuals_per_parent"
        ),
        "residual_budget_mixed_realized_extra_budget": (
            "residual_budget_mixed_realized_extra_budget"
        ),
        "residual_budget_mixed_unspent_budget": (
            "residual_budget_mixed_unspent_budget"
        ),
        "residual_budget_parent_hard_fraction_config": (
            "residual_budget_parent_hard_fraction_config"
        ),
        "residual_budget_parent_cost_power": (
            "residual_budget_parent_cost_power"
        ),
        "residual_budget_frame_balance_fraction": (
            "residual_budget_frame_balance_fraction"
        ),
        "residual_budget_fine_refresh": (
            "residual_budget_fine_refresh"
        ),
        "residual_budget_frame_balance_budget": (
            "residual_budget_frame_balance_budget"
        ),
        "residual_budget_frame_balance_service": (
            "residual_budget_frame_balance_service"
        ),
        "residual_budget_frame_balance_candidates": (
            "residual_budget_frame_balance_candidates"
        ),
        "residual_budget_remote_frame_coverage": (
            "residual_budget_remote_frame_coverage"
        ),
        "residual_budget_zero_service_frame_fraction": (
            "residual_budget_zero_service_frame_fraction"
        ),
        "residual_budget_frame_service_cv": (
            "residual_budget_frame_service_cv"
        ),
        "residual_budget_service_conditioned_momentum": (
            "residual_budget_service_conditioned_momentum"
        ),
        "residual_budget_effective_momentum": (
            "residual_budget_effective_momentum"
        ),
        "residual_budget_next_effective_momentum": (
            "residual_budget_next_effective_momentum"
        ),
        "residual_budget_grouped_cost_aware": (
            "residual_budget_grouped_cost_aware"
        ),
        "residual_budget_grouped_mean_service_cost": (
            "residual_budget_grouped_mean_service_cost"
        ),
        "residual_budget_grouped_capacity_promotion_fraction": (
            "residual_budget_grouped_capacity_promotion_fraction"
        ),
        "residual_budget_grouped_parent_relevance": (
            "residual_budget_grouped_parent_relevance"
        ),
        "residual_budget_parent_relevance_entropy": (
            "residual_budget_parent_relevance_entropy"
        ),
        "residual_budget_parent_hard_fraction": (
            "residual_budget_parent_hard_fraction"
        ),
        "residual_budget_parent_heterogeneity": (
            "residual_budget_parent_heterogeneity"
        ),
        "residual_budget_local_radius": "residual_budget_local_radius",
        "residual_budget_exact_frames_per_query": (
            "residual_budget_exact_frames_per_query"
        ),
        "residual_budget_phase_frames_per_query": (
            "residual_budget_phase_frames_per_query"
        ),
        "residual_budget_exact_budget_fraction": (
            "residual_budget_exact_budget_fraction"
        ),
        "residual_budget_extra_frames_per_query": (
            "residual_budget_extra_frames_per_query"
        ),
        "residual_budget_frame_detail": "residual_budget_frame_detail",
        "residual_budget_debt": "residual_budget_debt",
        "residual_budget_debt_after_repayment": (
            "residual_budget_debt_after_repayment"
        ),
        "residual_budget_debt_repaid_fraction": (
            "residual_budget_debt_repaid_fraction"
        ),
        "residual_budget_repayment_mode_id": (
            "residual_budget_repayment_mode_id"
        ),
        "residual_budget_service_credit_scale": (
            "residual_budget_service_credit_scale"
        ),
        "residual_budget_service_credit": (
            "residual_budget_service_credit"
        ),
        "residual_budget_negative_credit_fraction": (
            "residual_budget_negative_credit_fraction"
        ),
        "residual_budget_service_cycle_layers": (
            "residual_budget_service_cycle_layers"
        ),
        "residual_budget_service_age": "residual_budget_service_age",
        "residual_budget_selected_service_age": (
            "residual_budget_selected_service_age"
        ),
        "residual_budget_overdue_service_fraction": (
            "residual_budget_overdue_service_fraction"
        ),
        "residual_budget_frontier_service_fraction": (
            "residual_budget_frontier_service_fraction"
        ),
        "residual_budget_frontier_overdue_selected_fraction": (
            "residual_budget_frontier_overdue_selected_fraction"
        ),
        "residual_budget_selected_need": "residual_budget_selected_need",
        "residual_budget_selected_surface_need": (
            "residual_budget_selected_surface_need"
        ),
        "residual_budget_surface_weight": (
            "residual_budget_surface_weight"
        ),
        "residual_budget_cell_coverage_fraction": (
            "residual_budget_cell_coverage_fraction"
        ),
        "residual_budget_residual_tokens_per_query": (
            "residual_budget_residual_tokens_per_query"
        ),
        "residual_budget_selection_granularity_id": (
            "residual_budget_selection_granularity_id"
        ),
        "residual_budget_frame_detail_power": (
            "residual_budget_frame_detail_power"
        ),
        "residual_budget_spatial_detail_power": (
            "residual_budget_spatial_detail_power"
        ),
        "residual_budget_cell_scorer_id": (
            "residual_budget_cell_scorer_id"
        ),
        "residual_budget_carrier_residual_balance": (
            "residual_budget_carrier_residual_balance"
        ),
        "residual_budget_cell_refinement_phases": (
            "residual_budget_cell_refinement_phases"
        ),
        "residual_budget_cell_precision_fraction": (
            "residual_budget_cell_precision_fraction"
        ),
        "residual_budget_mass_conserving_refinement": (
            "residual_budget_mass_conserving_refinement"
        ),
        "residual_budget_exact_mass_conserving_refinement": (
            "residual_budget_exact_mass_conserving_refinement"
        ),
        "residual_budget_effective_service_credit": (
            "residual_budget_effective_service_credit"
        ),
        "residual_budget_service_credit_conservation_error": (
            "residual_budget_service_credit_conservation_error"
        ),
        "residual_budget_direct_attention": (
            "residual_budget_direct_attention"
        ),
        "residual_budget_additive_direct_attention": (
            "residual_budget_additive_direct_attention"
        ),
        "residual_budget_compiled_carrier_attention": (
            "residual_budget_compiled_carrier_attention"
        ),
        "residual_budget_compiled_carrier_value_bf16": (
            "residual_budget_compiled_carrier_value_bf16"
        ),
        "residual_budget_compiled_carrier_precast_value_bf16": (
            "residual_budget_compiled_carrier_precast_value_bf16"
        ),
        "residual_budget_compiled_carrier_dense_block_n": (
            "residual_budget_compiled_carrier_dense_block_n"
        ),
        "residual_budget_compiled_carrier_parent_block_n": (
            "residual_budget_compiled_carrier_parent_block_n"
        ),
        "residual_budget_compiled_carrier_child_block_n": (
            "residual_budget_compiled_carrier_child_block_n"
        ),
        "residual_budget_static_geometry_cache": (
            "residual_budget_static_geometry_cache"
        ),
        "residual_budget_phase_compiled_layout": (
            "residual_budget_phase_compiled_layout"
        ),
        "residual_budget_additive_flash_carrier": (
            "residual_budget_additive_flash_carrier"
        ),
        "residual_budget_additive_flash_positive_groups": (
            "residual_budget_additive_flash_positive_groups"
        ),
        "residual_budget_hps_complement_correction": (
            "residual_budget_hps_complement_correction"
        ),
        "residual_budget_hps_selected_cell_fraction": (
            "residual_budget_hps_selected_cell_fraction"
        ),
        "residual_budget_mixed_direct_attention": (
            "residual_budget_mixed_direct_attention"
        ),
        "residual_budget_direct_tile_profile_id": (
            "residual_budget_direct_tile_profile_id"
        ),
        "residual_budget_block_aligned_residual": (
            "residual_budget_block_aligned_residual"
        ),
        "residual_budget_direct_block_m": (
            "residual_budget_direct_block_m"
        ),
        "residual_budget_direct_block_n": (
            "residual_budget_direct_block_n"
        ),
        "residual_budget_direct_qk_bf16": (
            "residual_budget_direct_qk_bf16"
        ),
        "residual_budget_precision_cells_per_query": (
            "residual_budget_precision_cells_per_query"
        ),
        "residual_budget_selected_precision_importance": (
            "residual_budget_selected_precision_importance"
        ),
        "residual_budget_cell_importance_weight": (
            "residual_budget_cell_importance_weight"
        ),
        "residual_budget_cell_importance_floor": (
            "residual_budget_cell_importance_floor"
        ),
        "residual_budget_cell_importance_gate_threshold": (
            "residual_budget_cell_importance_gate_threshold"
        ),
        "residual_budget_cell_importance_gate": (
            "residual_budget_cell_importance_gate"
        ),
        "residual_budget_cell_importance_effective_weight": (
            "residual_budget_cell_importance_effective_weight"
        ),
        "residual_budget_base_need_concentration": (
            "residual_budget_base_need_concentration"
        ),
        "residual_budget_coverage_fraction": (
            "residual_budget_coverage_fraction"
        ),
        "residual_budget_coverage_gain": "residual_budget_coverage_gain",
        "residual_budget_selection_new_fraction": (
            "residual_budget_selection_new_fraction"
        ),
        "residual_budget_uncovered_fraction": (
            "residual_budget_uncovered_fraction"
        ),
        "adaptive_slit_stable_fraction": "adaptive_slit_stable_fraction",
        "adaptive_slit_change_fraction": "adaptive_slit_change_fraction",
        "adaptive_slit_selected_near_fraction": "adaptive_slit_selected_near_fraction",
        "adaptive_slit_dense_core_fraction": "adaptive_slit_dense_core_fraction",
        "covariance_k_variance_mean": "covariance_k_variance_mean",
        "covariance_k_variance_std": "covariance_k_variance_std",
        "covariance_score_shift_std": "covariance_score_shift_std",
        "layerwise_hybrid_stage_id": "layerwise_hybrid_stage_id",
        "layerwise_hybrid_config_sparse_ratio": "layerwise_hybrid_config_sparse_ratio",
        "layerwise_hybrid_patch_sparsity": "layerwise_hybrid_patch_sparsity",
        "layerwise_hybrid_local_fraction": "layerwise_hybrid_local_fraction",
        "coverage_hybrid_stage_id": "coverage_hybrid_stage_id",
        "coverage_hybrid_patch_sparsity": "coverage_hybrid_patch_sparsity",
        "coverage_hybrid_local_fraction": "coverage_hybrid_local_fraction",
        "coverage_hybrid_mid_anchors_per_query": (
            "coverage_hybrid_mid_anchors_per_query"
        ),
        "coverage_hybrid_mid_anchor_fraction": (
            "coverage_hybrid_mid_anchor_fraction"
        ),
        "persistent_view_graph_stage_id": "persistent_view_graph_stage_id",
        "persistent_view_graph_patch_sparsity": (
            "persistent_view_graph_patch_sparsity"
        ),
        "persistent_view_graph_refresh": "persistent_view_graph_refresh",
        "persistent_view_graph_views_per_query": (
            "persistent_view_graph_views_per_query"
        ),
        "persistent_view_graph_remote_views_per_query": (
            "persistent_view_graph_remote_views_per_query"
        ),
        "persistent_view_graph_anchors_per_query": (
            "persistent_view_graph_anchors_per_query"
        ),
        "persistent_view_graph_score_mean": "persistent_view_graph_score_mean",
        "persistent_view_graph_bias_std": "persistent_view_graph_bias_std",
        "persistent_view_graph_head_gate_mean": (
            "persistent_view_graph_head_gate_mean"
        ),
    }
    output_names.update(
        {name: name for name in DEBT_MEMORY_DIAGNOSTIC_METRICS}
    )
    output_names.update(
        {name: name for name in PV_IMPORTANCE_ALIGNMENT_METRICS}
    )
    output_names.update(
        {name: name for name in CARRIER_COMPENSATION_OBSERVER_METRICS}
    )
    output_names.update({name: name for name in PROJECTED_QK_METRICS})
    values = defaultdict(list)
    for layer_store in store.values():
        if not isinstance(layer_store, dict):
            continue
        for source_name, output_name in output_names.items():
            value = layer_store.get(source_name)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            if np.isfinite(value):
                values[output_name].append(value)
        for source_name, value in layer_store.items():
            if not source_name.startswith("service_obs_"):
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            if np.isfinite(value):
                values[source_name].append(value)
    return {
        name: float(np.mean(metric_values))
        for name, metric_values in values.items()
        if metric_values
    }


def collect_scout_oracle_per_layer(model):
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name, value in layer_store.items():
            if not name.startswith("scout_"):
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            row[name] = float(value)
        if len(row) > 1:
            rows.append(row)
    return rows


def collect_projected_qk_per_layer(model):
    """Return layerwise queue, support-exchange, and PV-service diagnostics."""
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name in PROJECTED_QK_METRICS:
            value = layer_store.get(name)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            if np.isfinite(value):
                row[name] = value
        if len(row) > 1:
            rows.append(row)
    return rows


def collect_service_observation_per_layer(model):
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name, value in layer_store.items():
            if not name.startswith(("service_obs_", "transported_debt_")):
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            row[name] = float(value)
        if len(row) > 1:
            rows.append(row)
    return rows


def collect_debt_memory_diagnostic_per_layer(model):
    """Return layerwise service-conditioned memory distribution statistics."""
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name in DEBT_MEMORY_DIAGNOSTIC_METRICS:
            value = layer_store.get(name)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            if np.isfinite(value):
                row[name] = value
        if len(row) > 1:
            rows.append(row)
    return rows


def collect_carrier_observer_per_layer(model):
    """Return layerwise QKV residual-bound diagnostics."""
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name in CARRIER_COMPENSATION_OBSERVER_METRICS:
            if "carrier_observer_" not in name:
                continue
            value = layer_store.get(name)
            if value is None:
                continue
            if torch.is_tensor(value):
                value = value.detach().float().mean().cpu().item()
            value = float(value)
            if np.isfinite(value):
                row[name] = value
        if len(row) > 1:
            rows.append(row)
    return rows


def collect_preview_adaptive_per_layer(model):
    store = getattr(model, "sparse_aux_output_store", None)
    if not isinstance(store, dict):
        return []
    rows = []
    for layer, layer_store in sorted(store.items()):
        if not isinstance(layer_store, dict):
            continue
        row = {"layer": int(layer)}
        for name, value in layer_store.items():
            if not name.startswith("preview_adaptive_"):
                continue
            if torch.is_tensor(value):
                value = value.detach().float().cpu()
                if value.numel() == 1:
                    value = value.item()
                else:
                    if value.ndim > 1:
                        value = value.mean(dim=0)
                    value = value.flatten().tolist()
            row[name] = value if isinstance(value, list) else float(value)
        if len(row) > 1:
            rows.append(row)
    return rows


def main(args):
    add_repo_paths(args.streamvggt_root, args.vggt_root)
    runtime_provenance = collect_runtime_provenance(args.vanilla_vggt)
    set_random_seeds(args.seed)

    from eval.mv_recon.data import NRGBD, SevenScenes
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    if not args.pose_only:
        import open3d as o3d
        from eval.mv_recon.utils import accuracy, completion
        from eval.mv_recon.criterion import Regr3D_t_ScaleShiftInv, L21
        from vggt.utils.geometry import unproject_depth_map_to_point_map

    if args.size == 512:
        resolution = (512, 384)
    elif args.size == 224:
        resolution = 224
    elif args.size == 518:
        resolution = (518, 392)
    else:
        raise NotImplementedError(f"Unsupported size: {args.size}")

    common_dataset_args = {
        "split": "test",
        "ROOT": args.data_root,
        "resolution": resolution,
        "num_seq": 1,
        "full_video": True,
        "test_id": args.scene,
        "kf_every": args.kf_every,
    }
    if args.dataset == "7scenes":
        dataset = SevenScenes(seq_id=args.seq_id, **common_dataset_args)
    else:
        if args.seq_id is not None:
            raise ValueError("--seq_id is only supported for the 7scenes dataset")
        dataset = NRGBD(**common_dataset_args)
        if hasattr(dataset, "scene_list"):
            dataset.scene_list = sorted(dataset.scene_list)

    accelerator = Accelerator()
    device = accelerator.device
    if device.type != "cuda":
        raise RuntimeError("Sparse VGGT evaluation requires a CUDA PyTorch environment")
    if args.pose_only and accelerator.num_processes != 1:
        raise RuntimeError("--pose_only metric aggregation currently requires a single process")
    if args.timing_warmup < 0 or args.timing_repeats < 1:
        raise ValueError("--timing_warmup must be >= 0 and --timing_repeats must be >= 1")
    if args.num_frames is not None and args.num_frames < 1:
        raise ValueError("--num_frames must be >= 1")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("--max_points must be >= 1")
    model = build_model(args, device)

    os.makedirs(args.output_dir, exist_ok=True)
    implementation_name = (
        "vggt_baseline" if args.vanilla_vggt else "sparse_vggt"
    )
    save_path = os.path.join(
        args.output_dir, f"{args.dataset}_{implementation_name}"
    )
    os.makedirs(save_path, exist_ok=True)
    pose_metrics_path = args.metrics_json or os.path.join(
        save_path, "pose_metrics.json"
    )
    os.makedirs(os.path.dirname(os.path.abspath(pose_metrics_path)), exist_ok=True)
    partial_metrics_path = f"{pose_metrics_path}.partial"
    log_file = os.path.join(save_path, f"logs_{accelerator.process_index}.txt")
    with open(log_file, "w"):
        pass

    criterion = None
    if not args.pose_only:
        criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    pose_records = []
    checkpoint_reconstruction_records = []
    with torch.no_grad():
        with accelerator.split_between_processes(list(range(len(dataset)))) as idxs:
            for data_idx in tqdm(idxs):
                sequence_views = dataset[data_idx]
                if args.num_frames is not None:
                    if len(sequence_views) < args.num_frames and not args.num_frames_is_cap:
                        raise ValueError(
                            f"Sequence index {data_idx} contains {len(sequence_views)} "
                            "sampled frames, "
                            f"fewer than requested --num_frames {args.num_frames}"
                        )
                    sequence_views = sequence_views[: args.num_frames]
                batch = default_collate([sequence_views])

                ignore_keys = {
                    "depthmap",
                    "dataset",
                    "label",
                    "instance",
                    "idx",
                    "true_shape",
                    "rng",
                }

                for view in batch:
                    for name in view.keys():
                        if name in ignore_keys:
                            continue
                        if isinstance(view[name], (tuple, list)):
                            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
                        else:
                            view[name] = view[name].to(device, non_blocking=True)

                images = batch_views_to_images(batch).to(device)
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

                for _ in range(args.timing_warmup):
                    with torch.cuda.amp.autocast(dtype=dtype):
                        model(images)
                torch.cuda.synchronize(device)

                if args.torch_profiler_trace:
                    trace_path = os.path.abspath(args.torch_profiler_trace)
                    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=False,
                        profile_memory=True,
                        with_stack=False,
                    ) as profiler:
                        with torch.cuda.amp.autocast(dtype=dtype):
                            model(images)
                        torch.cuda.synchronize(device)
                    profiler.export_chrome_trace(trace_path)
                    print(
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=40,
                        )
                    )

                attention_repeats_ms = []
                attention_timer = CudaAttentionTimer(model) if args.profile_attention else None
                if attention_timer is not None:
                    for _ in range(args.timing_repeats):
                        attention_timer.begin()
                        with torch.cuda.amp.autocast(dtype=dtype):
                            model(images)
                        torch.cuda.synchronize(device)
                        attention_repeats_ms.append(attention_timer.finish())
                    attention_timer.close()
                attention_ms = (
                    float(np.median(attention_repeats_ms)) if attention_repeats_ms else None
                )

                torch.cuda.reset_peak_memory_stats(device)
                timing_ms = []
                predictions = None
                for _ in range(args.timing_repeats):
                    if predictions is not None:
                        del predictions
                    inference_start = time.perf_counter()
                    with torch.cuda.amp.autocast(dtype=dtype):
                        predictions = model(images)
                    torch.cuda.synchronize(device)
                    timing_ms.append((time.perf_counter() - inference_start) * 1000.0)
                inference_ms = float(np.median(timing_ms))
                peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                pred_extrinsic, _ = pose_encoding_to_extri_intri(
                    predictions["pose_enc"], images.shape[-2:]
                )
                gt_c2w = np.stack(
                    [view["camera_pose"][0].detach().cpu().numpy() for view in batch]
                )
                scene_id = batch[-1]["label"][0].rsplit("/", 1)[0]
                pose_result = trajectory_metrics(
                    pred_extrinsic[0].detach().cpu().numpy(), gt_c2w
                )
                if args.skip_block_sparsity_collect:
                    block_sparsity, block_sparsity_per_layer = None, None
                else:
                    block_sparsity, block_sparsity_per_layer = collect_block_sparsity(model)
                pose_result.update(
                    {
                        "scene_id": scene_id,
                        "end_to_end_ms": inference_ms,
                        "inference_ms": inference_ms,
                        "inference_repeats_ms": timing_ms,
                        "attention_ms": attention_ms,
                        "attention_repeats_ms": attention_repeats_ms,
                        "peak_memory_mb": peak_memory_mb,
                        "block_sparsity": block_sparsity,
                        "block_sparsity_per_layer": block_sparsity_per_layer,
                    }
                )
                if block_selection_analysis_enabled(args):
                    selection_mode = get_block_selection_mode(args)
                    analysis_summary, analysis_layers = collect_block_selection_analysis(
                        model, selection_mode=selection_mode
                    )
                    pose_result.update(analysis_summary)
                    pose_result["block_selection_mode"] = selection_mode
                    pose_result["block_selection_analysis_per_layer"] = analysis_layers
                    pose_result["importance_block_analysis_per_layer"] = analysis_layers
                elif args.analyze_scout_oracle:
                    pose_result.update(collect_sparse_aux_metrics(model))
                    pose_result["scout_oracle_per_layer"] = (
                        collect_scout_oracle_per_layer(model)
                    )
                else:
                    pose_result.update(collect_sparse_aux_metrics(model))
                    pose_result["projected_qk_per_layer"] = (
                        collect_projected_qk_per_layer(model)
                    )
                    if args.use_preview_adaptive_routing:
                        pose_result["preview_adaptive_per_layer"] = (
                            collect_preview_adaptive_per_layer(model)
                        )
                if os.environ.get(
                    "SPARSE_VGGT_SERVICE_OBSERVATION", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    pose_result["service_observation_per_layer"] = (
                        collect_service_observation_per_layer(model)
                    )
                if os.environ.get(
                    "SPARSE_VGGT_DEBT_MEMORY_DIAGNOSTIC", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    pose_result["debt_memory_diagnostic_per_layer"] = (
                        collect_debt_memory_diagnostic_per_layer(model)
                    )
                if os.environ.get(
                    "SPARSE_VGGT_CARRIER_COMPENSATION_RESIDUAL_OBSERVER", "0"
                ).strip().lower() in {"1", "true", "yes", "on"}:
                    pose_result["carrier_observer_per_layer"] = (
                        collect_carrier_observer_per_layer(model)
                    )
                pose_records.append(pose_result)
                rpe_trans = pose_result["rpe_trans_rmse_m"]
                rpe_rot = pose_result["rpe_rot_rmse_deg"]
                print(
                    f"Pose: {scene_id}, ATE_RMSE_m: {pose_result['ate_rmse_m']:.6f}, "
                    f"RPE_trans_RMSE_m: {rpe_trans if rpe_trans is not None else 'n/a'}, "
                    f"RPE_rot_RMSE_deg: {rpe_rot if rpe_rot is not None else 'n/a'}, "
                    f"Inference_ms: {inference_ms:.3f}, "
                    f"Attention_ms: {attention_ms if attention_ms is not None else 'n/a'}, "
                    f"Block_sparsity: {pose_result.get('block_sparsity', 'n/a')}, "
                    f"Core_fraction: "
                    f"{pose_result.get('soft_geometry_core_fraction', 'n/a')}, "
                    f"Transition_fraction: "
                    f"{pose_result.get('soft_geometry_transition_fraction', 'n/a')}, "
                    f"Far_fraction: "
                    f"{pose_result.get('soft_geometry_far_fraction', 'n/a')}"
                )

                if args.pose_only:
                    if (
                        accelerator.is_main_process
                        and accelerator.num_processes == 1
                    ):
                        write_metrics_checkpoint(
                            partial_metrics_path,
                            args,
                            runtime_provenance,
                            pose_records,
                            checkpoint_reconstruction_records,
                        )
                    torch.cuda.empty_cache()
                    continue

                preds = predictions_to_per_view_preds(predictions)

                if args.use_proj:
                    with torch.cuda.amp.autocast(dtype=torch.float64):
                        extrinsic, intrinsic = pose_encoding_to_extri_intri(
                            predictions["pose_enc"], images.shape[-2:]
                        )
                        point_map_by_unprojection = unproject_depth_map_to_point_map(
                            predictions["depth"].squeeze(0),
                            extrinsic.squeeze(0),
                            intrinsic.squeeze(0),
                        )
                    for j in range(len(preds)):
                        preds[j]["pts3d"] = point_map_by_unprojection[j][None]
                        preds[j]["conf"] = predictions["depth_conf"][:, j]

                print(f"Evaluation for {args.dataset} {data_idx + 1}/{len(dataset)}")
                gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(batch, preds)

                pts_all = []
                pts_gt_all = []
                images_all = []
                masks_all = []

                for j, view in enumerate(batch):
                    image = view["img"].permute(0, 2, 3, 1).cpu().numpy()[0]
                    image = (image + 1.0) / 2.0
                    mask = view["valid_mask"].cpu().numpy()[0]

                    pts = pred_pts[j].cpu().numpy()[0]
                    conf = preds[j]["conf"].cpu().numpy()[0]
                    pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                    if args.conf_thresh > 0:
                        mask = mask & (conf > args.conf_thresh)

                    H, W = image.shape[:2]
                    cx = W // 2
                    cy = H // 2
                    l, t = cx - 112, cy - 112
                    r, b = cx + 112, cy + 112
                    image = image[t:b, l:r]
                    mask = mask[t:b, l:r]
                    pts = pts[t:b, l:r]
                    pts_gt = pts_gt[t:b, l:r]

                    images_all.append(image[None, ...])
                    pts_all.append(pts[None, ...])
                    pts_gt_all.append(pts_gt[None, ...])
                    masks_all.append(mask[None, ...])

                images_all = np.concatenate(images_all, axis=0)
                pts_all = np.concatenate(pts_all, axis=0)
                pts_gt_all = np.concatenate(pts_gt_all, axis=0)
                masks_all = np.concatenate(masks_all, axis=0)

                np.save(
                    os.path.join(save_path, f"{scene_id.replace('/', '_')}.npy"),
                    {
                        "images_all": images_all,
                        "pts_all": pts_all,
                        "pts_gt_all": pts_gt_all,
                        "masks_all": masks_all,
                    },
                )

                threshold = 0.1
                pts_all_masked = pts_all[masks_all > 0]
                pts_gt_all_masked = pts_gt_all[masks_all > 0]
                images_all_masked = images_all[masks_all > 0]

                pred_finite = np.isfinite(pts_all_masked).all(axis=-1)
                gt_finite = np.isfinite(pts_gt_all_masked).all(axis=-1)
                pred_colors = images_all_masked[pred_finite].reshape(-1, 3)
                gt_colors = images_all_masked[gt_finite].reshape(-1, 3)
                pts_all_masked = pts_all_masked[pred_finite].reshape(-1, 3)
                pts_gt_all_masked = pts_gt_all_masked[gt_finite].reshape(-1, 3)

                if args.max_points is not None:
                    rng = np.random.default_rng(args.seed + int(data_idx))
                    if len(pts_all_masked) > args.max_points:
                        pred_indices = rng.choice(
                            len(pts_all_masked), args.max_points, replace=False
                        )
                        pts_all_masked = pts_all_masked[pred_indices]
                        pred_colors = pred_colors[pred_indices]
                    if len(pts_gt_all_masked) > args.max_points:
                        gt_indices = rng.choice(
                            len(pts_gt_all_masked), args.max_points, replace=False
                        )
                        pts_gt_all_masked = pts_gt_all_masked[gt_indices]
                        gt_colors = gt_colors[gt_indices]

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_all_masked)
                pcd.colors = o3d.utility.Vector3dVector(pred_colors)

                pcd_gt = o3d.geometry.PointCloud()
                pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked)
                pcd_gt.colors = o3d.utility.Vector3dVector(gt_colors)

                o3d.io.write_point_cloud(
                    os.path.join(save_path, f"{scene_id.replace('/', '_')}-mask.ply"),
                    pcd,
                )
                o3d.io.write_point_cloud(
                    os.path.join(save_path, f"{scene_id.replace('/', '_')}-gt.ply"),
                    pcd_gt,
                )

                reg_p2p = o3d.pipelines.registration.registration_icp(
                    pcd,
                    pcd_gt,
                    threshold,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )
                pcd = pcd.transform(reg_p2p.transformation)

                pcd.estimate_normals()
                pcd_gt.estimate_normals()

                gt_normal = np.asarray(pcd_gt.normals)
                pred_normal = np.asarray(pcd.normals)

                acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
                comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

                line = (
                    f"Idx: {scene_id}, Acc: {acc}, Comp: {comp}, NC1: {nc1}, NC2: {nc2} - "
                    f"Acc_med: {acc_med}, Compc_med: {comp_med}, NC1c_med: {nc1_med}, NC2c_med: {nc2_med}"
                )
                print(line)
                print(line, file=open(log_file, "a"))
                checkpoint_reconstruction_records.append(
                    {
                        "scene_id": scene_id,
                        "acc": float(acc),
                        "comp": float(comp),
                        "nc1": float(nc1),
                        "nc2": float(nc2),
                        "acc_med": float(acc_med),
                        "comp_med": float(comp_med),
                        "nc1_med": float(nc1_med),
                        "nc2_med": float(nc2_med),
                        "nc": float((nc1 + nc2) / 2),
                        "nc_med": float((nc1_med + nc2_med) / 2),
                    }
                )
                if (
                    accelerator.is_main_process
                    and accelerator.num_processes == 1
                ):
                    write_metrics_checkpoint(
                        partial_metrics_path,
                        args,
                        runtime_provenance,
                        pose_records,
                        checkpoint_reconstruction_records,
                    )
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            metrics_payload = {
                "config": vars(args),
                "runtime_provenance": runtime_provenance,
                "records": pose_records,
                "summary": summarize_pose_records(pose_records),
            }
            with open(pose_metrics_path, "w") as f:
                json.dump(metrics_payload, f, indent=2)
            if block_selection_analysis_enabled(args):
                analysis_csv_path = (
                    args.block_selection_analysis_csv
                    or args.importance_block_analysis_csv
                    or os.path.join(
                        os.path.dirname(os.path.abspath(pose_metrics_path)),
                        "block_selection_analysis_per_layer.csv",
                    )
                )
                write_block_selection_analysis_csv(pose_records, analysis_csv_path)
            if args.pose_only:
                if os.path.exists(partial_metrics_path):
                    os.remove(partial_metrics_path)
                return

            to_write = ""
            for i in range(8):
                cur = os.path.join(save_path, f"logs_{i}.txt")
                if not os.path.exists(cur):
                    break
                with open(cur, "r") as f_sub:
                    to_write += f_sub.read()

            with open(os.path.join(save_path, "logs_all.txt"), "w") as f:
                metrics = defaultdict(list)
                reconstruction_records = []
                for line in to_write.strip().split("\n"):
                    match = regex.match(line)
                    if match:
                        data = match.groupdict()
                        reconstruction_record = {
                            "scene_id": data["scene_id"].strip()
                        }
                        for key, value in data.items():
                            if key != "scene_id":
                                numeric_value = float(value)
                                metrics[key].append(numeric_value)
                                reconstruction_record[key] = numeric_value
                        reconstruction_record["nc"] = (
                            reconstruction_record["nc1"]
                            + reconstruction_record["nc2"]
                        ) / 2
                        reconstruction_record["nc_med"] = (
                            reconstruction_record["nc1_med"]
                            + reconstruction_record["nc2_med"]
                        ) / 2
                        metrics["nc"].append(reconstruction_record["nc"])
                        metrics["nc_med"].append(
                            reconstruction_record["nc_med"]
                        )
                        reconstruction_records.append(reconstruction_record)

                mean_metrics = {
                    metric: sum(values) / len(values)
                    for metric, values in metrics.items()
                }
                mean_metrics["num_sequences"] = len(reconstruction_records)

                print_str = "mean".ljust(20) + ": "
                for m_name in mean_metrics:
                    print_str += f"{m_name}: {mean_metrics[m_name]:.3f} | "
                print_str += "\n"
                f.write(to_write + print_str)

            metrics_payload["reconstruction_records"] = reconstruction_records
            metrics_payload["reconstruction_summary"] = mean_metrics
            metrics_payload["summary"].update(
                {
                    f"reconstruction_{name}": value
                    for name, value in mean_metrics.items()
                }
            )
            with open(pose_metrics_path, "w") as f:
                json.dump(metrics_payload, f, indent=2)
            if os.path.exists(partial_metrics_path):
                os.remove(partial_metrics_path)


pattern = r"""
    Idx:\s*(?P<scene_id>[^,]+),\s*
    Acc:\s*(?P<acc>[^,]+),\s*
    Comp:\s*(?P<comp>[^,]+),\s*
    NC1:\s*(?P<nc1>[^,]+),\s*
    NC2:\s*(?P<nc2>[^,]+)\s*-\s*
    Acc_med:\s*(?P<acc_med>[^,]+),\s*
    Compc_med:\s*(?P<comp_med>[^,]+),\s*
    NC1c_med:\s*(?P<nc1_med>[^,]+),\s*
    NC2c_med:\s*(?P<nc2_med>[^,]+)
"""

regex = re.compile(pattern, re.VERBOSE)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
