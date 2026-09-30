"""
Layer-wise sparse attention for VGGT.

This module extends the base sparse-vggt implementation to support
different sparsity ratios for different layers, enabling more aggressive
optimization while maintaining accuracy.
"""

from functools import partial
from types import MethodType
from typing import Literal

from vggt.models.aggregator import Aggregator

from sparse_vggt.models.attention import adaptive_sparse_attention_forward
from sparse_vggt.models.vggt import sparse_vggt_aggregator_forward
from sparse_vggt.models.utils import print_sparse_info
from sparse_vggt.utils.sparse_wrapper import check_sparse_mode


def sparse_aggregator_from_vggt_layerwise(
    aggregator: Aggregator,
    use_hilbert: bool = False,
    layer_config: dict[int, dict] | None = None,
    layer_sparsity_ratios: list[float] | None = None,
    sparse_ratio: float | None = None,
    cdf_threshold: float | None = None,
    pool_mode: Literal["max", "avg"] = "avg",
    aux_output: bool = False,
    aux_sparsity_only: bool = True,
    num_special_tokens: int = 5,
    verbose: bool = True,
    # Radial + layer-wise fusion parameters
    use_radial_layerwise: bool = False,
    decay_factor: float = 1.0,
    dense_neighbor: int = 1,
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
    multiresolution_layer_samples: list[int] | None = None,
    multiresolution_layer_pool_sizes: list[int] | None = None,
    multiresolution_layer_local_radii: list[int] | None = None,
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
    residual_budget_fine_refresh_layers: tuple[int, ...] = (),
    residual_budget_fine_refresh_sparsity: float = 0.70,
    residual_budget_budget_neutral_fine_refresh: bool = False,
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
    layer_geometry_weights: list[float] | None = None,
    geometry_decay: str = "linear",
    decay_gamma: float = 0.25,
    geometry_sigma: float | None = None,
    frame_normalize_importance: bool = False,
    distance_calibrate_importance: bool = False,
    entropy_adaptive_geometry: bool = False,
    head_adaptive_geometry: bool = False,
    analyze_importance_blocks: bool = False,
    analyze_block_selection: bool = False,
):
    """Convert VGGT aggregator to sparse aggregator with layer-wise sparsity control.

    Args:
        aggregator: Original VGGT aggregator.
        use_hilbert: If True, use Hilbert permutation on patch tokens.
        layer_config: Dict mapping layer_id to config dict with keys:
            - "sparse_ratio": float | None
            - "cdf_threshold": float | None
            - "decay_factor": float (for radial attention, optional)
            Example:
            {
                0: {"sparse_ratio": 0.15, "cdf_threshold": None},
                1: {"sparse_ratio": 0.12, "cdf_threshold": None},
                ...
            }
        layer_sparsity_ratios: List of sparsity ratios per layer (simpler alternative to layer_config).
            Length must match number of layers. Example: [0.95, 0.90, 0.85, 0.80].
            If provided, takes precedence over layer_config.
        sparse_ratio: Default sparse ratio if neither layer_config nor layer_sparsity_ratios provided.
        cdf_threshold: Default CDF threshold if layer_config not provided.
        pool_mode: Avg or Max pooling for the global attention.
        aux_output: If True, store auxiliary output from the global attention.
        aux_sparsity_only: If True, only store sparsity from the global attention.
        num_special_tokens: Number of special tokens (camera + register).
        verbose: Print configuration info.
        use_radial_layerwise: If True, use fused radial+layer-wise+importance mask.
        decay_factor: Radial decay factor (for radial constraint).
        dense_neighbor: Number of neighboring frames with full attention.
        use_distance_routed: Route near pairs to radial and far pairs to importance.
        route_frame_threshold: Maximum frame distance handled by radial attention.
        use_covariance_aware_importance: Calibrate far importance with K-block variance.
        use_adaptive_slit_routing: If True, use adaptive local slit blocks for near routed pairs.
        layer_geometry_weights: Optional per-layer soft-geometry weights. If provided,
            length must match the number of global attention layers.

    Returns:
        Aggregator: Modified Aggregator
        aux_output_store (dict | None): Auxiliary output storage.

    Warnings:
        UserWarning: If layer_sparsity_ratios length doesn't match number of layers,
                     or if ratios are invalid (out of range [0, 1], NaN, or non-numeric).
                     Falls back to uniform sparsity in these cases.
    """
    # Validate radial parameters
    if decay_factor <= 0:
        raise ValueError(f"decay_factor must be positive, got {decay_factor}")
    if dense_neighbor < 0:
        raise ValueError(f"dense_neighbor must be non-negative, got {dense_neighbor}")
    if route_frame_threshold < 0:
        raise ValueError(
            f"route_frame_threshold must be non-negative, got {route_frame_threshold}"
        )
    if use_adaptive_slit_routing and not use_distance_routed:
        raise ValueError(
            "use_adaptive_slit_routing requires use_distance_routed=True"
        )
    if use_covariance_aware_importance and not use_distance_routed:
        raise ValueError(
            "use_covariance_aware_importance requires use_distance_routed=True"
        )
    if covariance_weight < 0:
        raise ValueError("covariance_weight must be non-negative")
    if covariance_eps <= 0:
        raise ValueError("covariance_eps must be positive")
    if adaptive_slit_temporal_window < 1:
        raise ValueError("adaptive_slit_temporal_window must be >= 1")
    if not 0.0 <= adaptive_slit_stable_quantile <= 1.0:
        raise ValueError("adaptive_slit_stable_quantile must be in [0, 1]")
    if not 0.0 <= adaptive_slit_change_quantile <= 1.0:
        raise ValueError("adaptive_slit_change_quantile must be in [0, 1]")
    if adaptive_slit_narrow_width < 0 or adaptive_slit_base_width < 0 or adaptive_slit_expand_width < 0:
        raise ValueError("adaptive slit widths must be non-negative")
    if adaptive_slit_narrow_width > adaptive_slit_base_width:
        raise ValueError("adaptive_slit_narrow_width must be <= adaptive_slit_base_width")
    if adaptive_slit_base_width > adaptive_slit_expand_width:
        raise ValueError("adaptive_slit_base_width must be <= adaptive_slit_expand_width")
    routed_modes = (
        use_radial_layerwise,
        use_distance_routed,
        use_soft_geometry_routing,
        use_preview_adaptive_routing,
        use_multiresolution_routing,
        use_residual_budget_routing,
        use_dual_path_routing,
        use_layerwise_hybrid_routing,
        use_local_layerwise_hybrid_routing,
        use_coverage_layerwise_routing,
        use_persistent_view_graph_routing,
    )
    if sum(routed_modes) > 1:
        raise ValueError(
            "use_radial_layerwise, use_distance_routed, and "
            "use_soft_geometry_routing/use_preview_adaptive_routing/"
            "use_multiresolution_routing/"
            "use_residual_budget_routing/"
            "use_dual_path_routing/"
            "use_layerwise_hybrid_routing/"
            "use_local_layerwise_hybrid_routing/"
            "use_coverage_layerwise_routing/"
            "use_persistent_view_graph_routing are mutually exclusive"
        )
    if hybrid_early_end < 0 or hybrid_mid_end < hybrid_early_end:
        raise ValueError(
            "hybrid_early_end and hybrid_mid_end must satisfy "
            "0 <= early_end <= mid_end"
        )
    if multiresolution_local_radius < 0:
        raise ValueError("multiresolution_local_radius must be non-negative")
    if not 0.0 <= residual_budget_target_sparsity < 1.0:
        raise ValueError("residual_budget_target_sparsity must be in [0, 1)")
    if residual_budget_local_radius < 0:
        raise ValueError("residual_budget_local_radius must be non-negative")
    if residual_budget_remote_pool_size < 2:
        raise ValueError("residual_budget_remote_pool_size must be at least 2")
    if residual_budget_routing_parent_size < residual_budget_remote_pool_size:
        raise ValueError(
            "residual_budget_routing_parent_size must be at least the remote pool"
        )
    if (
        residual_budget_routing_parent_size
        % residual_budget_remote_pool_size
        != 0
    ):
        raise ValueError(
            "residual budget routing parent size must divide into execution cells"
        )
    if residual_budget_routing_phase_mode not in {
        "rotating",
        "fixed",
        "anchored",
        "anchored_heads",
    }:
        raise ValueError(
            "residual_budget_routing_phase_mode must be rotating, fixed, "
            "anchored, or anchored_heads"
        )
    if not 0.0 <= residual_budget_parent_hard_fraction <= 1.0:
        raise ValueError(
            "residual_budget_parent_hard_fraction must be in [0, 1]"
        )
    if residual_budget_parent_cost_power < 0.0:
        raise ValueError(
            "residual_budget_parent_cost_power must be non-negative"
        )
    if not 0.0 <= residual_budget_frame_balance_fraction <= 1.0:
        raise ValueError(
            "residual_budget_frame_balance_fraction must be in [0, 1]"
        )
    if not 0.0 <= residual_budget_fine_refresh_sparsity < 1.0:
        raise ValueError(
            "residual_budget_fine_refresh_sparsity must be in [0, 1)"
        )
    if (
        residual_budget_fine_refresh_layers
        and not use_residual_budget_routing
    ):
        raise ValueError(
            "fine refresh layers require residual budget routing"
        )
    grouped_parent_routing = (
        residual_budget_routing_parent_size
        > residual_budget_remote_pool_size
    )
    if residual_budget_mixed_parent_execution and not grouped_parent_routing:
        raise ValueError(
            "mixed parent execution requires grouped parent routing"
        )
    if residual_budget_additive_parent_service and not grouped_parent_routing:
        raise ValueError(
            "additive parent service requires grouped parent routing"
        )
    if residual_budget_unified_incremental_service:
        if not use_residual_budget_routing:
            raise ValueError(
                "unified incremental service requires residual budget routing"
            )
        if residual_budget_routing_parent_size != 2 * residual_budget_remote_pool_size:
            raise ValueError(
                "unified incremental service currently requires 4x4 parents "
                "over 2x2 execution cells"
            )
        if (
            residual_budget_fine_refresh_layers
            and not residual_budget_budget_neutral_fine_refresh
        ):
            raise ValueError(
                "unified incremental service only supports fixed fine refresh "
                "layers in budget-neutral mode"
            )
        if (
            residual_budget_adaptive_parent_service
            or residual_budget_mixed_parent_execution
            or residual_budget_additive_parent_service
            or residual_budget_additive_full_upgrade_fraction != 0.0
            or residual_budget_additive_parent_substitution_fraction != 0.0
            or residual_budget_frame_balance_fraction != 0.0
        ):
            raise ValueError(
                "unified incremental service cannot be combined with legacy "
                "parent buckets, fixed upgrades, substitutions, or FBD"
            )
    if (
        residual_budget_bounded_wait_reobservation
        and not residual_budget_unified_incremental_service
    ):
        raise ValueError(
            "bounded-wait reobservation requires unified incremental service"
        )
    if (
        residual_budget_additive_parent_service
        and residual_budget_mixed_parent_execution
    ):
        raise ValueError(
            "additive and mixed parent execution are mutually exclusive"
        )
    if (
        residual_budget_additive_parent_service
        and residual_budget_adaptive_parent_service
    ):
        raise ValueError(
            "additive parent service uses its own service capacity"
        )
    if not 0.0 <= residual_budget_additive_full_upgrade_fraction <= 0.25:
        raise ValueError(
            "residual_budget_additive_full_upgrade_fraction must be "
            "in [0, 0.25]"
        )
    if (
        residual_budget_additive_full_upgrade_fraction > 0.0
        and not residual_budget_additive_parent_service
    ):
        raise ValueError(
            "additive full upgrades require additive parent service"
        )
    if not (
        0.0
        <= residual_budget_additive_parent_substitution_fraction
        <= 1.0
    ):
        raise ValueError(
            "residual_budget_additive_parent_substitution_fraction must "
            "be in [0, 1]"
        )
    if (
        residual_budget_additive_parent_substitution_fraction > 0.0
        and not residual_budget_additive_parent_service
    ):
        raise ValueError(
            "additive parent substitution requires additive parent service"
        )
    mixed_children_per_parent = (
        residual_budget_routing_parent_size
        // residual_budget_remote_pool_size
    ) ** 2
    if residual_budget_mixed_parent_execution and not 0 <= (
        residual_budget_mixed_residuals_per_parent
    ) <= mixed_children_per_parent:
        raise ValueError(
            "residual_budget_mixed_residuals_per_parent must fit one parent"
        )
    if residual_budget_query_frame_chunk < 1:
        raise ValueError(
            "residual_budget_query_frame_chunk must be at least 1"
        )
    if not 0.0 <= residual_budget_momentum <= 1.0:
        raise ValueError("residual_budget_momentum must be in [0, 1]")
    if (
        residual_budget_service_conditioned_momentum
        and not grouped_parent_routing
    ):
        raise ValueError(
            "service-conditioned momentum requires grouped parent routing"
        )
    if (
        residual_budget_service_conditioned_momentum
        and residual_budget_repayment_mode == "reset"
    ):
        raise ValueError(
            "service-conditioned momentum requires deficit repayment"
        )
    if not 0.0 <= residual_budget_repayment <= 1.0:
        raise ValueError("residual_budget_repayment must be in [0, 1]")
    if residual_budget_repayment_mode not in {
        "reset",
        "deficit",
        "bounded_deficit",
        "effective_bounded_deficit",
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }:
        raise ValueError(
            "residual_budget_repayment_mode must be reset, deficit, "
            "bounded_deficit, effective_bounded_deficit, "
            "age_bounded_deficit, or "
            "frontier_age_bounded_deficit"
        )
    if residual_budget_service_credit_scale < 0.0:
        raise ValueError(
            "residual_budget_service_credit_scale must be non-negative"
        )
    if residual_budget_temperature <= 0:
        raise ValueError("residual_budget_temperature must be positive")
    if not 0.0 <= residual_budget_exact_fraction <= 1.0:
        raise ValueError("residual_budget_exact_fraction must be in [0, 1]")
    if not 0.0 <= residual_budget_surface_weight <= 1.0:
        raise ValueError("residual_budget_surface_weight must be in [0, 1]")
    if residual_budget_selection_granularity not in {"frame", "cell"}:
        raise ValueError(
            "residual_budget_selection_granularity must be 'frame' or 'cell'"
        )
    if (
        grouped_parent_routing
        and residual_budget_selection_granularity != "cell"
    ):
        raise ValueError("grouped parent routing requires cell selection")
    if grouped_parent_routing and residual_budget_repayment_mode in {
        "age_bounded_deficit",
        "frontier_age_bounded_deficit",
    }:
        raise ValueError(
            "grouped parent routing does not yet support age repayment"
        )
    if residual_budget_cell_scorer not in {
        "uniform",
        "residual",
        "carrier",
        "carrier_residual",
        "patch_qk",
        "scout_service",
    }:
        raise ValueError(
            "residual_budget_cell_scorer must be uniform, residual, carrier, "
            "carrier_residual, patch_qk, or scout_service"
        )
    if (
        residual_budget_selection_granularity == "frame"
        and residual_budget_cell_scorer != "carrier_residual"
    ):
        raise ValueError("non-default residual scorers require cell selection")
    if (
        residual_budget_selection_granularity == "frame"
        and residual_budget_repayment_mode != "reset"
    ):
        raise ValueError("deficit repayment requires cell selection")
    if (
        residual_budget_selection_granularity == "cell"
        and residual_budget_exact_fraction > 0.0
    ):
        raise ValueError(
            "cell residual selection currently requires exact_fraction=0"
        )
    if (
        residual_budget_exact_mass_conserving_refinement
        and not residual_budget_mass_conserving_refinement
    ):
        raise ValueError(
            "exact mass conservation requires mass conservation"
        )
    if (
        residual_budget_additive_parent_service
        and not residual_budget_mass_conserving_refinement
    ):
        raise ValueError(
            "additive parent service requires mass conservation"
        )
    if (
        residual_budget_additive_parent_service
        and residual_budget_exact_mass_conserving_refinement
    ):
        raise ValueError(
            "additive parent service does not support exact mass conservation"
        )
    if (
        residual_budget_additive_parent_service
        and residual_budget_local_radius != 0
    ):
        raise ValueError(
            "additive parent service currently requires radius 0"
        )
    if not 0.0 <= residual_budget_frame_detail_power <= 1.0:
        raise ValueError(
            "residual_budget_frame_detail_power must be in [0, 1]"
        )
    if not 0.0 <= residual_budget_spatial_detail_power <= 1.0:
        raise ValueError(
            "residual_budget_spatial_detail_power must be in [0, 1]"
        )
    if not 0.0 <= residual_budget_carrier_residual_balance <= 1.0:
        raise ValueError(
            "residual_budget_carrier_residual_balance must be in [0, 1]"
        )
    if residual_budget_cell_refinement_phases not in {1, 2, 3, 15}:
        raise ValueError(
            "residual_budget_cell_refinement_phases must be 1, 2, 3, or 15"
        )
    if (
        residual_budget_additive_parent_service
        and residual_budget_cell_refinement_phases != 1
    ):
        raise ValueError(
            "additive parent service currently requires one child phase"
        )
    if residual_budget_cell_refinement_phases == 15 and (
        residual_budget_remote_pool_size != 4
        or residual_budget_selection_granularity != "cell"
        or residual_budget_cell_precision_fraction != 0.0
    ):
        raise ValueError(
            "15-phase hierarchical refinement requires 4x4 cell selection "
            "without precision cells"
        )
    if (
        grouped_parent_routing
        and residual_budget_cell_refinement_phases == 15
    ):
        raise ValueError(
            "grouped parent routing supports one, two, or three child phases"
        )
    if (
        residual_budget_selection_granularity == "frame"
        and residual_budget_cell_refinement_phases != 1
    ):
        raise ValueError(
            "frame residual selection requires cell_refinement_phases=1"
        )
    if not 0.0 <= residual_budget_cell_precision_fraction <= 0.5:
        raise ValueError(
            "residual_budget_cell_precision_fraction must be in [0, 0.5]"
        )
    if (
        residual_budget_selection_granularity == "frame"
        and residual_budget_cell_precision_fraction > 0.0
    ):
        raise ValueError(
            "frame residual selection requires cell_precision_fraction=0"
        )
    if (
        residual_budget_cell_refinement_phases != 1
        and residual_budget_cell_precision_fraction > 0.0
    ):
        raise ValueError(
            "cell_precision_fraction requires cell_refinement_phases=1"
        )
    if not 0.0 <= residual_budget_cell_importance_weight <= 1.0:
        raise ValueError(
            "residual_budget_cell_importance_weight must be in [0, 1]"
        )
    if grouped_parent_routing and residual_budget_cell_importance_weight > 0.0:
        raise ValueError(
            "grouped parent routing computes parent relevance directly"
        )
    if not (
        0.0
        <= residual_budget_cell_importance_floor
        <= residual_budget_cell_importance_weight
    ):
        raise ValueError(
            "residual_budget_cell_importance_floor must be in "
            "[0, cell_importance_weight]"
        )
    if residual_budget_cell_importance_gate_threshold < 0.0:
        raise ValueError(
            "residual_budget_cell_importance_gate_threshold must be non-negative"
        )
    if residual_budget_cell_importance_gate_temperature <= 0.0:
        raise ValueError(
            "residual_budget_cell_importance_gate_temperature must be positive"
        )
    if multiresolution_remote_pool_size < 1:
        raise ValueError(
            "multiresolution_remote_pool_size must be at least 1"
        )
    if multiresolution_query_frame_chunk < 1:
        raise ValueError(
            "multiresolution_query_frame_chunk must be at least 1"
        )
    if multiresolution_remote_mode not in {"avg", "strided", "detail"}:
        raise ValueError(
            "multiresolution_remote_mode must be 'avg', 'strided', or 'detail'"
        )
    if multiresolution_remote_phase_mode not in {"layer", "query", "head"}:
        raise ValueError(
            "multiresolution_remote_phase_mode must be 'layer', 'query', "
            "or 'head'"
        )
    if (
        multiresolution_remote_phase_mode == "head"
        and multiresolution_remote_mode != "strided"
    ):
        raise ValueError(
            "head multiresolution phase mode requires strided remote tokens"
        )
    if not 1 <= multiresolution_remote_samples_per_cell <= (
        multiresolution_remote_pool_size**2
    ):
        raise ValueError(
            "multiresolution_remote_samples_per_cell must be in "
            "[1, remote_pool_size squared]"
        )
    if (
        multiresolution_remote_mode == "avg"
        and multiresolution_remote_samples_per_cell != 1
    ):
        raise ValueError("avg multiresolution mode supports one sample per cell")
    if (
        multiresolution_remote_mode == "detail"
        and multiresolution_remote_samples_per_cell != 1
    ):
        raise ValueError(
            "detail multiresolution mode supports one sample per cell"
        )
    if (
        multiresolution_area_bias
        and multiresolution_remote_samples_per_cell != 1
    ):
        raise ValueError("multiresolution area bias requires one sample per cell")
    if multiresolution_remote_phase_mode == "query":
        if multiresolution_remote_mode != "strided":
            raise ValueError(
                "query-aligned multiresolution routing requires strided mode"
            )
        if multiresolution_remote_samples_per_cell != 1:
            raise ValueError(
                "query-aligned multiresolution routing requires one sample"
            )
        if multiresolution_area_bias:
            raise ValueError(
                "query-aligned multiresolution routing does not use area bias"
            )
    if hybrid_early_dense_neighbor < 0:
        raise ValueError("hybrid_early_dense_neighbor must be non-negative")
    if hybrid_early_local_radius < 0:
        raise ValueError("hybrid_early_local_radius must be non-negative")
    if not 0.0 <= mid_frame_coverage_ratio <= 1.0:
        raise ValueError("mid_frame_coverage_ratio must be in [0, 1]")
    if view_graph_early_end < 0 or view_graph_mid_end < view_graph_early_end:
        raise ValueError(
            "view graph boundaries must satisfy 0 <= early_end <= mid_end"
        )
    if not 0.0 <= view_graph_mid_sparse_ratio <= 1.0:
        raise ValueError("view_graph_mid_sparse_ratio must be in [0, 1]")
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
    if not 0.0 <= hybrid_mid_sparse_ratio <= 1.0:
        raise ValueError("hybrid_mid_sparse_ratio must be in [0, 1]")
    if not 0.0 <= hybrid_late_sparse_ratio <= 1.0:
        raise ValueError("hybrid_late_sparse_ratio must be in [0, 1]")
    if core_frame_radius < 0:
        raise ValueError("core_frame_radius must be non-negative")
    if transition_frame_radius < core_frame_radius:
        raise ValueError(
            "transition_frame_radius must be >= core_frame_radius"
        )
    if geometry_weight < 0:
        raise ValueError("geometry_weight must be non-negative")
    if not 0.0 <= preview_redistribution_fraction < 1.0:
        raise ValueError("preview_redistribution_fraction must be in [0, 1)")
    if preview_activation_threshold < 0:
        raise ValueError("preview_activation_threshold must be non-negative")
    if preview_local_radius < 0:
        raise ValueError("preview_local_radius must be non-negative")
    if not 0.0 <= preview_local_fraction < 1.0:
        raise ValueError("preview_local_fraction must be in [0, 1)")
    if preview_geometry_weight < 0:
        raise ValueError("preview_geometry_weight must be non-negative")
    if preview_head_protection not in {"none", "value_detail", "locality"}:
        raise ValueError("invalid preview_head_protection")
    if not 0.0 <= preview_protected_head_fraction < 1.0:
        raise ValueError("preview_protected_head_fraction must be in [0, 1)")
    if not 0.0 <= preview_donor_retention_threshold <= 1.0:
        raise ValueError(
            "preview_donor_retention_threshold must be in [0, 1]"
        )
    if preview_donor_exchange_scope not in {"head", "layer"}:
        raise ValueError("invalid preview_donor_exchange_scope")
    if (
        preview_donor_exchange_scope == "layer"
        and preview_donor_retention_threshold <= 0
    ):
        raise ValueError(
            "layer donor exchange requires a positive retention threshold"
        )
    if not 0.0 <= preview_receiver_gain_threshold <= 1.0:
        raise ValueError(
            "preview_receiver_gain_threshold must be in [0, 1]"
        )
    if preview_exchange_gain_cost_ratio < 0:
        raise ValueError(
            "preview_exchange_gain_cost_ratio must be non-negative"
        )
    if not 0.0 <= preview_head_exchange_cap_fraction <= 1.0:
        raise ValueError(
            "preview_head_exchange_cap_fraction must be in [0, 1]"
        )
    if preview_layer_confidence_threshold < 0:
        raise ValueError(
            "preview_layer_confidence_threshold must be non-negative"
        )
    if (
        preview_receiver_gain_threshold > 0
        or preview_exchange_gain_cost_ratio > 0
        or preview_head_exchange_cap_fraction < 1.0
        or preview_layer_confidence_threshold > 0
    ) and preview_donor_exchange_scope != "layer":
        raise ValueError(
            "preview utility certification, head caps, and confidence gating "
            "require layer exchange"
        )
    if preview_exchange_layer_start < 0:
        raise ValueError("preview_exchange_layer_start must be non-negative")
    if (
        preview_exchange_layer_end >= 0
        and preview_exchange_layer_end <= preview_exchange_layer_start
    ):
        raise ValueError(
            "preview_exchange_layer_end must be -1 or greater than start"
        )
    if geometry_decay not in {"linear", "exponential"}:
        raise ValueError("geometry_decay must be 'linear' or 'exponential'")
    if dual_path_local_radius < 0:
        raise ValueError("dual_path_local_radius must be non-negative")
    if dual_path_context_geometry_weight < 0:
        raise ValueError("dual_path_context_geometry_weight must be non-negative")
    if dual_path_context_schedule not in {"flat", "late_ramp", "late_only"}:
        raise ValueError("invalid dual_path_context_schedule")
    if dual_path_context_gate not in {"none", "disagreement"}:
        raise ValueError("invalid dual_path_context_gate")
    if dual_path_context_alignment_temperature <= 0:
        raise ValueError(
            "dual_path_context_alignment_temperature must be positive"
        )
    if min(
        scout_oracle_query_blocks,
        scout_oracle_queries_per_block,
        scout_oracle_coarse_group_blocks,
    ) <= 0:
        raise ValueError("scout oracle sampling parameters must be positive")
    if scout_oracle_local_radius < 0:
        raise ValueError("scout_oracle_local_radius must be non-negative")
    if scout_oracle_candidate_multiplier < 1.0:
        raise ValueError("scout_oracle_candidate_multiplier must be at least 1")
    if not (
        0.0
        <= dual_path_min_local_fraction
        <= dual_path_max_local_fraction
        <= 1.0
    ):
        raise ValueError(
            "dual-path local fractions must satisfy 0 <= min <= max <= 1"
        )
    if dual_path_layer_schedule not in {
        "flat",
        "early_decay",
        "middle_peak",
    }:
        raise ValueError("invalid dual_path_layer_schedule")

    # Replace aggregator forward function
    aggregator_fwd = partial(sparse_vggt_aggregator_forward, use_hilbert=use_hilbert)
    aggregator.forward = MethodType(aggregator_fwd, aggregator)
    routing_state = {}
    aggregator._sparse_routing_state = routing_state

    num_layers = len(aggregator.global_blocks)
    fine_refresh_layers = frozenset(residual_budget_fine_refresh_layers)
    if len(fine_refresh_layers) != len(residual_budget_fine_refresh_layers):
        raise ValueError("fine refresh layers must be unique")
    if any(
        layer < 0 or layer >= num_layers for layer in fine_refresh_layers
    ):
        raise ValueError(
            "fine refresh layers must index existing global layers"
        )
    if residual_budget_budget_neutral_fine_refresh:
        if not residual_budget_unified_incremental_service:
            raise ValueError(
                "budget-neutral fine refresh requires unified incremental "
                "service"
            )
        if not fine_refresh_layers:
            raise ValueError(
                "budget-neutral fine refresh requires at least one refresh "
                "layer"
            )
        ordinary_layer_count = num_layers - len(fine_refresh_layers)
        if ordinary_layer_count == 0:
            raise ValueError(
                "budget-neutral fine refresh requires at least one ordinary "
                "layer"
            )
        ordinary_residual_budget_target_sparsity = (
            num_layers * residual_budget_target_sparsity
            - len(fine_refresh_layers)
            * residual_budget_fine_refresh_sparsity
        ) / ordinary_layer_count
        if not 0.0 <= ordinary_residual_budget_target_sparsity < 1.0:
            raise ValueError(
                "budget-neutral fine refresh implies an invalid ordinary-layer "
                "sparsity: "
                f"{ordinary_residual_budget_target_sparsity:.6f}"
            )
    else:
        ordinary_residual_budget_target_sparsity = (
            residual_budget_target_sparsity
        )
    if multiresolution_layer_pool_sizes is not None:
        if len(multiresolution_layer_pool_sizes) != num_layers:
            raise ValueError(
                "multiresolution_layer_pool_sizes length must match "
                f"{num_layers} layers, got "
                f"{len(multiresolution_layer_pool_sizes)}"
            )
        for layer_id, pool_size in enumerate(multiresolution_layer_pool_sizes):
            if not isinstance(pool_size, int) or pool_size < 1:
                raise ValueError(
                    "multiresolution_layer_pool_sizes values must be positive "
                    f"integers, got layer {layer_id}: {pool_size}"
                )
    resolved_multiresolution_pool_sizes = (
        list(multiresolution_layer_pool_sizes)
        if multiresolution_layer_pool_sizes is not None
        else [multiresolution_remote_pool_size] * num_layers
    )
    if multiresolution_layer_local_radii is not None:
        if len(multiresolution_layer_local_radii) != num_layers:
            raise ValueError(
                "multiresolution_layer_local_radii length must match "
                f"{num_layers} layers, got "
                f"{len(multiresolution_layer_local_radii)}"
            )
        for layer_id, radius in enumerate(multiresolution_layer_local_radii):
            if not isinstance(radius, int) or radius < 0:
                raise ValueError(
                    "multiresolution_layer_local_radii values must be "
                    f"non-negative integers, got layer {layer_id}: {radius}"
                )
    resolved_multiresolution_local_radii = (
        list(multiresolution_layer_local_radii)
        if multiresolution_layer_local_radii is not None
        else [multiresolution_local_radius] * num_layers
    )
    if multiresolution_layer_samples is not None and (
        len(multiresolution_layer_samples) != num_layers
    ):
        raise ValueError(
            "multiresolution_layer_samples length must match "
            f"{num_layers} layers, got {len(multiresolution_layer_samples)}"
        )
    resolved_multiresolution_samples = (
        list(multiresolution_layer_samples)
        if multiresolution_layer_samples is not None
        else [multiresolution_remote_samples_per_cell] * num_layers
    )
    for layer_id, (sample_count, pool_size) in enumerate(
        zip(
            resolved_multiresolution_samples,
            resolved_multiresolution_pool_sizes,
        )
    ):
        max_samples = pool_size**2
        if not isinstance(sample_count, int) or not (
            1 <= sample_count <= max_samples
        ):
            raise ValueError(
                "multiresolution layer samples must be integers in "
                "[1, layer pool size squared], got layer "
                f"{layer_id}: samples={sample_count}, pool_size={pool_size}"
            )
    if multiresolution_remote_mode in {"avg", "detail"} and any(
        sample_count != 1
        for sample_count in resolved_multiresolution_samples
    ):
        raise ValueError(
            "avg/detail multiresolution modes support one sample per layer"
        )
    if multiresolution_area_bias and any(
        sample_count != 1
        for sample_count in resolved_multiresolution_samples
    ):
        raise ValueError(
            "multiresolution area bias requires one sample per layer"
        )
    if multiresolution_remote_phase_mode == "query" and any(
        sample_count != 1
        for sample_count in resolved_multiresolution_samples
    ):
        raise ValueError(
            "query-aligned multiresolution routing requires one sample "
            "in every layer"
        )
    if analyze_scout_oracle_routing and any(
        layer < 0 or layer >= num_layers for layer in scout_oracle_layers
    ):
        raise ValueError("scout_oracle_layers contains an invalid layer index")
    modified_layers = [i for i in range(num_layers)]
    if aux_output:
        aux_output_store = {i: {} for i in modified_layers}
    else:
        aux_output_store = {i: None for i in modified_layers}
    aggregator._sparse_aux_output_store = aux_output_store

    # Helper function to validate and convert numeric values
    import math
    import warnings

    validated_layer_geometry_weights = None
    if layer_geometry_weights is not None:
        if len(layer_geometry_weights) != num_layers:
            raise ValueError(
                f"layer_geometry_weights length must match {num_layers} layers, "
                f"got {len(layer_geometry_weights)}"
            )
        validated_layer_geometry_weights = []
        for idx, value in enumerate(layer_geometry_weights):
            try:
                converted = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"layer_geometry_weights[{idx}] must be numeric, got {value!r}"
                ) from exc
            if not math.isfinite(converted) or converted < 0:
                raise ValueError(
                    f"layer_geometry_weights[{idx}] must be finite and non-negative"
                )
            validated_layer_geometry_weights.append(converted)

    def validate_numeric(value, name, allow_none=True):
        """Validate and convert numeric value to float.

        Returns:
            (is_valid: bool, converted_value: float | None, error_msg: str | None)
        """
        if value is None:
            return (True, None, None) if allow_none else (False, None, f"{name} cannot be None")

        try:
            value_float = float(value)
            if math.isnan(value_float):
                return (False, None, f"{name} ({value}) is NaN")
            if value_float < 0 or value_float > 1.0:
                return (False, None, f"{name} ({value}) out of range [0, 1]")
            return (True, value_float, None)
        except (TypeError, ValueError):
            return (False, None, f"{name} ({value}) is not numeric")

    # Build layer-wise configuration
    # Priority: layer_sparsity_ratios > layer_config > uniform defaults
    # IMPORTANT: Any validation failure triggers complete fallback to uniform

    use_layer_config = False  # Track whether to use layer_config

    if layer_sparsity_ratios is not None:
        # Validate container type first
        if not isinstance(layer_sparsity_ratios, (list, tuple)):
            warnings.warn(
                f"layer_sparsity_ratios must be a list or tuple, got {type(layer_sparsity_ratios).__name__}. "
                f"Falling back to uniform sparsity.",
                UserWarning
            )
            layer_sparsity_ratios = None
            layer_config = None
        elif len(layer_sparsity_ratios) != num_layers:
            warnings.warn(
                f"layer_sparsity_ratios length ({len(layer_sparsity_ratios)}) "
                f"doesn't match num_layers ({num_layers}). "
                f"Falling back to uniform sparsity.",
                UserWarning
            )
            layer_sparsity_ratios = None
            # Force skip layer_config, go directly to uniform
            layer_config = None
        else:
            # Validate and convert each ratio
            invalid_ratios = []
            converted_ratios = []
            for i, ratio in enumerate(layer_sparsity_ratios):
                is_valid, converted, error_msg = validate_numeric(ratio, f"ratio[{i}]", allow_none=False)
                if not is_valid:
                    invalid_ratios.append((i, ratio, error_msg))
                else:
                    converted_ratios.append(converted)

            if invalid_ratios:
                warnings.warn(
                    f"Invalid sparsity ratios found: {[(i, r) for i, r, _ in invalid_ratios]}. "
                    f"Ratios must be numeric and in [0, 1]. "
                    f"Falling back to uniform sparsity.",
                    UserWarning
                )
                layer_sparsity_ratios = None
                # Force skip layer_config, go directly to uniform
                layer_config = None
            else:
                # Replace with converted float values
                layer_sparsity_ratios = converted_ratios

    if layer_sparsity_ratios is not None:
        # Validate and convert cdf_threshold if provided
        if cdf_threshold is not None:
            is_valid, converted_cdf, error_msg = validate_numeric(cdf_threshold, "cdf_threshold")
            if not is_valid:
                warnings.warn(
                    f"{error_msg}. Falling back to uniform sparsity.",
                    UserWarning
                )
                layer_sparsity_ratios = None
                cdf_threshold = None
                # Force skip layer_config, go directly to uniform
                layer_config = None
            else:
                cdf_threshold = converted_cdf

    if layer_sparsity_ratios is not None:
        # Convert list to layer_config format with validated float values
        layer_config = {
            i: {"sparse_ratio": layer_sparsity_ratios[i], "cdf_threshold": cdf_threshold}
            for i in modified_layers
        }
        use_layer_config = True
    elif layer_config is not None:
        # Validate container type first
        if not isinstance(layer_config, dict):
            warnings.warn(
                f"layer_config must be a dict, got {type(layer_config).__name__}. "
                f"Falling back to uniform sparsity.",
                UserWarning
            )
            layer_config = None
        else:
            # Validate layer_config: must be complete and all values valid
            # Check completeness first
            missing_layers = [i for i in modified_layers if i not in layer_config]
            if missing_layers:
                warnings.warn(
                    f"layer_config is incomplete: missing layers {missing_layers}. "
                    f"Falling back to uniform sparsity with sparse_ratio={sparse_ratio}",
                    UserWarning
                )
                layer_config = None
            else:
                # Validate and convert each layer's config
                # ANY invalid value triggers complete fallback
                validated_layer_config = {}
                has_invalid = False

                for layer_id in modified_layers:
                    config = layer_config[layer_id]

                    # Validate that config is a dict-like object
                    if not isinstance(config, dict) or config is None:
                        warnings.warn(
                            f"layer_config[{layer_id}] is not a valid dict: {type(config).__name__}. "
                            f"Falling back to uniform sparsity with sparse_ratio={sparse_ratio}",
                            UserWarning
                        )
                        has_invalid = True
                        break

                    layer_sparse_ratio = config.get("sparse_ratio")
                    layer_cdf = config.get("cdf_threshold")

                    # Validate and convert sparse_ratio
                    if layer_sparse_ratio is not None:
                        is_valid, converted, error_msg = validate_numeric(layer_sparse_ratio, f"layer_config[{layer_id}]['sparse_ratio']")
                        if not is_valid:
                            warnings.warn(
                                f"{error_msg}. Falling back to uniform sparsity with sparse_ratio={sparse_ratio}",
                                UserWarning
                            )
                            has_invalid = True
                            break
                        else:
                            layer_sparse_ratio = converted

                    # Validate and convert cdf_threshold
                    if layer_cdf is not None:
                        is_valid, converted, error_msg = validate_numeric(layer_cdf, f"layer_config[{layer_id}]['cdf_threshold']")
                        if not is_valid:
                            warnings.warn(
                                f"{error_msg}. Falling back to uniform sparsity with sparse_ratio={sparse_ratio}",
                                UserWarning
                            )
                            has_invalid = True
                            break
                        else:
                            layer_cdf = converted

                    # Validate that the combination is valid (not both None)
                    if layer_sparse_ratio is None and layer_cdf is None:
                        warnings.warn(
                            f"layer_config[{layer_id}] has both sparse_ratio and cdf_threshold as None. "
                            f"Falling back to uniform sparsity with sparse_ratio={sparse_ratio}",
                            UserWarning
                        )
                        has_invalid = True
                        break

                    validated_layer_config[layer_id] = {
                        "sparse_ratio": layer_sparse_ratio,
                        "cdf_threshold": layer_cdf
                    }

                if has_invalid:
                    # Complete fallback to uniform
                    layer_config = None
                else:
                    layer_config = validated_layer_config
                    use_layer_config = True

    if not use_layer_config:
        # Fallback to uniform config - validate defaults
        validated_sparse_ratio = sparse_ratio
        validated_cdf_threshold = cdf_threshold

        if sparse_ratio is not None:
            is_valid, converted, error_msg = validate_numeric(sparse_ratio, "sparse_ratio")
            if not is_valid:
                warnings.warn(f"{error_msg}. Setting to None.", UserWarning)
                validated_sparse_ratio = None
            else:
                validated_sparse_ratio = converted

        if cdf_threshold is not None:
            is_valid, converted, error_msg = validate_numeric(cdf_threshold, "cdf_threshold")
            if not is_valid:
                warnings.warn(f"{error_msg}. Setting to None.", UserWarning)
                validated_cdf_threshold = None
            else:
                validated_cdf_threshold = converted

        # Final fallback: if both are None, disable sparsity completely
        if validated_sparse_ratio is None and validated_cdf_threshold is None:
            warnings.warn(
                "All sparsity configurations are invalid or None. "
                "Disabling sparsity (sparse_ratio=0.0) to prevent errors.",
                UserWarning
            )
            validated_sparse_ratio = 0.0
            validated_cdf_threshold = None

        layer_config = {
            i: {"sparse_ratio": validated_sparse_ratio, "cdf_threshold": validated_cdf_threshold}
            for i in modified_layers
        }

    # Apply layer-specific sparse attention
    for i in range(len(aggregator.global_blocks)):
        if i not in modified_layers:
            continue

        # Get layer-specific config
        layer_sparse_ratio = layer_config[i].get("sparse_ratio", sparse_ratio)
        layer_cdf_threshold = layer_config[i].get("cdf_threshold", cdf_threshold)
        layer_geometry_weight = (
            validated_layer_geometry_weights[i]
            if validated_layer_geometry_weights is not None
            else geometry_weight
        )
        layer_multiresolution_pool_size = (
            resolved_multiresolution_pool_sizes[i]
        )
        layer_multiresolution_local_radius = (
            resolved_multiresolution_local_radii[i]
        )
        layer_multiresolution_samples = resolved_multiresolution_samples[i]
        fine_refresh = i in fine_refresh_layers
        layer_residual_budget_target_sparsity = (
            residual_budget_fine_refresh_sparsity
            if fine_refresh
            else ordinary_residual_budget_target_sparsity
        )
        hps_structural_sparsity_limit = (
            1.0 - 1.0 / (residual_budget_remote_pool_size ** 2)
        )
        unified_carrier_execution = (
            residual_budget_unified_incremental_service
            and layer_residual_budget_target_sparsity
            >= hps_structural_sparsity_limit - 1e-8
        )
        layer_residual_budget_parent_size = (
            residual_budget_routing_parent_size
            if residual_budget_budget_neutral_fine_refresh
            else (
                residual_budget_remote_pool_size
                if fine_refresh
                else residual_budget_routing_parent_size
            )
        )

        # Replace attention forward function with layer-specific params
        attn_fwd = partial(
            adaptive_sparse_attention_forward,
            sparse_ratio=layer_sparse_ratio,
            cdf_threshold=layer_cdf_threshold,
            pool_mode=pool_mode,
            aux_output_store=aux_output_store[i],
            aux_sparsity_only=aux_sparsity_only,
            num_special_tokens=num_special_tokens,
            use_radial_layerwise=use_radial_layerwise,
            decay_factor=decay_factor,
            dense_neighbor=dense_neighbor,
            layer_idx=i,
            layer_sparsity_ratios=layer_sparsity_ratios,
            use_distance_routed=use_distance_routed,
            route_frame_threshold=route_frame_threshold,
            use_covariance_aware_importance=use_covariance_aware_importance,
            covariance_weight=covariance_weight,
            covariance_eps=covariance_eps,
            use_adaptive_slit_routing=use_adaptive_slit_routing,
            adaptive_slit_temporal_window=adaptive_slit_temporal_window,
            adaptive_slit_stable_quantile=adaptive_slit_stable_quantile,
            adaptive_slit_change_quantile=adaptive_slit_change_quantile,
            adaptive_slit_narrow_width=adaptive_slit_narrow_width,
            adaptive_slit_base_width=adaptive_slit_base_width,
            adaptive_slit_expand_width=adaptive_slit_expand_width,
            use_soft_geometry_routing=use_soft_geometry_routing,
            use_preview_adaptive_routing=use_preview_adaptive_routing,
            use_multiresolution_routing=use_multiresolution_routing,
            use_residual_budget_routing=use_residual_budget_routing,
            multiresolution_local_radius=layer_multiresolution_local_radius,
            multiresolution_remote_pool_size=layer_multiresolution_pool_size,
            multiresolution_query_frame_chunk=(
                multiresolution_query_frame_chunk
            ),
            multiresolution_remote_mode=multiresolution_remote_mode,
            multiresolution_remote_samples_per_cell=(
                layer_multiresolution_samples
            ),
            multiresolution_remote_phase_mode=(
                multiresolution_remote_phase_mode
            ),
            multiresolution_area_bias=multiresolution_area_bias,
            residual_budget_target_sparsity=(
                layer_residual_budget_target_sparsity
            ),
            residual_budget_local_radius=residual_budget_local_radius,
            residual_budget_remote_pool_size=(
                residual_budget_remote_pool_size
            ),
            residual_budget_routing_parent_size=(
                layer_residual_budget_parent_size
            ),
            residual_budget_routing_phase_mode=(
                residual_budget_routing_phase_mode
            ),
            residual_budget_unified_incremental_service=(
                residual_budget_unified_incremental_service
            ),
            residual_budget_bounded_wait_reobservation=(
                residual_budget_bounded_wait_reobservation
            ),
            residual_budget_adaptive_parent_service=(
                residual_budget_adaptive_parent_service
                and not fine_refresh
            ),
            residual_budget_mixed_parent_execution=(
                residual_budget_mixed_parent_execution
                and not fine_refresh
            ),
            residual_budget_additive_parent_service=(
                unified_carrier_execution
                or (
                    residual_budget_additive_parent_service
                    and not fine_refresh
                )
            ),
            residual_budget_additive_full_upgrade_fraction=(
                0.0
                if fine_refresh
                else residual_budget_additive_full_upgrade_fraction
            ),
            residual_budget_additive_parent_substitution_fraction=(
                0.0
                if fine_refresh
                else residual_budget_additive_parent_substitution_fraction
            ),
            residual_budget_mixed_residuals_per_parent=(
                residual_budget_mixed_residuals_per_parent
            ),
            residual_budget_parent_hard_fraction=(
                residual_budget_parent_hard_fraction
            ),
            residual_budget_parent_cost_power=(
                residual_budget_parent_cost_power
            ),
            residual_budget_frame_balance_fraction=(
                0.0
                if fine_refresh
                else residual_budget_frame_balance_fraction
            ),
            residual_budget_fine_refresh=fine_refresh,
            residual_budget_query_frame_chunk=(
                residual_budget_query_frame_chunk
            ),
            residual_budget_momentum=residual_budget_momentum,
            residual_budget_service_conditioned_momentum=(
                residual_budget_unified_incremental_service
                or (
                    residual_budget_service_conditioned_momentum
                    and not fine_refresh
                )
            ),
            residual_budget_repayment=residual_budget_repayment,
            residual_budget_repayment_mode=(
                "effective_bounded_deficit"
                if residual_budget_unified_incremental_service
                else (
                    "reset" if fine_refresh else residual_budget_repayment_mode
                )
            ),
            residual_budget_service_credit_scale=(
                residual_budget_service_credit_scale
            ),
            residual_budget_temperature=residual_budget_temperature,
            residual_budget_exact_fraction=(
                residual_budget_exact_fraction
            ),
            residual_budget_surface_weight=(
                residual_budget_surface_weight
            ),
            residual_budget_selection_granularity=(
                residual_budget_selection_granularity
            ),
            residual_budget_frame_detail_power=(
                0.0 if fine_refresh else residual_budget_frame_detail_power
            ),
            residual_budget_spatial_detail_power=(
                1.0 if fine_refresh else residual_budget_spatial_detail_power
            ),
            residual_budget_cell_scorer=(
                "carrier_residual"
                if fine_refresh
                else residual_budget_cell_scorer
            ),
            residual_budget_carrier_residual_balance=(
                0.5
                if fine_refresh
                else residual_budget_carrier_residual_balance
            ),
            residual_budget_cell_refinement_phases=(
                residual_budget_cell_refinement_phases
            ),
            residual_budget_cell_precision_fraction=(
                residual_budget_cell_precision_fraction
            ),
            residual_budget_mass_conserving_refinement=(
                residual_budget_unified_incremental_service
                or residual_budget_mass_conserving_refinement
            ),
            residual_budget_exact_mass_conserving_refinement=(
                residual_budget_exact_mass_conserving_refinement
            ),
            residual_budget_cell_importance_weight=(
                residual_budget_cell_importance_weight
            ),
            residual_budget_cell_importance_floor=(
                residual_budget_cell_importance_floor
            ),
            residual_budget_cell_importance_gate_threshold=(
                residual_budget_cell_importance_gate_threshold
            ),
            residual_budget_cell_importance_gate_temperature=(
                residual_budget_cell_importance_gate_temperature
            ),
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
            dual_path_num_layers=num_layers,
            dual_path_context_geometry_weight=dual_path_context_geometry_weight,
            dual_path_context_schedule=dual_path_context_schedule,
            dual_path_context_gate=dual_path_context_gate,
            dual_path_context_alignment_threshold=(
                dual_path_context_alignment_threshold
            ),
            dual_path_context_alignment_temperature=(
                dual_path_context_alignment_temperature
            ),
            analyze_scout_oracle_routing=analyze_scout_oracle_routing,
            scout_oracle_layers=scout_oracle_layers,
            scout_oracle_query_blocks=scout_oracle_query_blocks,
            scout_oracle_queries_per_block=scout_oracle_queries_per_block,
            scout_oracle_local_radius=scout_oracle_local_radius,
            scout_oracle_coarse_group_blocks=(
                scout_oracle_coarse_group_blocks
            ),
            scout_oracle_candidate_multiplier=(
                scout_oracle_candidate_multiplier
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
            geometry_weight=layer_geometry_weight,
            geometry_decay=geometry_decay,
            decay_gamma=decay_gamma,
            geometry_sigma=geometry_sigma,
            frame_normalize_importance=frame_normalize_importance,
            distance_calibrate_importance=distance_calibrate_importance,
            entropy_adaptive_geometry=entropy_adaptive_geometry,
            head_adaptive_geometry=head_adaptive_geometry,
            analyze_importance_blocks=analyze_importance_blocks,
            analyze_block_selection=analyze_block_selection,
            routing_state=routing_state,
        )
        aggregator.global_blocks[i].attn.forward = MethodType(
            attn_fwd, aggregator.global_blocks[i].attn
        )
        aggregator.global_blocks[i].attn._sparse_routing_state = routing_state

    if verbose:
        print_layerwise_sparse_info(
            layer_config,
            pool_mode,
            use_hilbert,
            aux_output,
            used_layer_sparsity_ratios=(layer_sparsity_ratios is not None)
        )
        if validated_layer_geometry_weights is not None:
            print(
                "Layer-wise geometry weights: "
                f"{validated_layer_geometry_weights}"
            )
        if use_layerwise_hybrid_routing:
            print(
                "Layer-wise hybrid routing: "
                f"radial[0:{hybrid_early_end}) "
                f"importance[{hybrid_early_end}:{hybrid_mid_end}) "
                f"soft[{hybrid_mid_end}:{num_layers})"
            )
        if use_local_layerwise_hybrid_routing:
            print(
                "Local layer-wise hybrid routing: "
                f"local[0:{hybrid_early_end}) "
                f"local+far-importance[{hybrid_early_end}:{hybrid_mid_end}) "
                f"soft[{hybrid_mid_end}:{num_layers})"
            )
        if use_coverage_layerwise_routing:
            print(
                "Coverage layer-wise routing: "
                f"local[0:{hybrid_early_end}) "
                f"coverage[{hybrid_early_end}:{hybrid_mid_end}) "
                f"reference-soft[{hybrid_mid_end}:{num_layers}) "
                f"coverage_ratio={mid_frame_coverage_ratio}"
            )
        if use_persistent_view_graph_routing:
            print(
                "Persistent view-graph routing: "
                f"soft[0:{view_graph_early_end}) "
                f"view-graph[{view_graph_early_end}:{view_graph_mid_end}) "
                f"soft[{view_graph_mid_end}:{num_layers}) "
                f"remote_topk={view_graph_remote_topk}, "
                f"remote_ratio={view_graph_remote_ratio}, "
                f"refresh={view_graph_refresh_interval}, "
                f"momentum={view_graph_momentum}, "
                f"anchors={view_graph_force_anchors}, "
                f"hard={view_graph_hard_routing}, "
                f"weight={view_graph_weight}, "
                f"head_adaptive={view_graph_head_adaptive}"
            )
        if use_multiresolution_routing:
            print(
                "Multi-resolution interaction routing: "
                f"local_radius={multiresolution_local_radius}, "
                f"remote_pool={multiresolution_remote_pool_size}x"
                f"{multiresolution_remote_pool_size}, "
                f"remote_mode={multiresolution_remote_mode}, "
                f"remote_samples={multiresolution_remote_samples_per_cell}, "
                f"phase_mode={multiresolution_remote_phase_mode}, "
                f"area_bias={multiresolution_area_bias}, "
                f"query_chunk={multiresolution_query_frame_chunk}"
            )
            if multiresolution_layer_samples is not None:
                print(
                    "Multi-resolution per-layer samples: "
                    f"{multiresolution_layer_samples}"
                )
            if multiresolution_layer_pool_sizes is not None:
                print(
                    "Multi-resolution per-layer pool sizes: "
                    f"{multiresolution_layer_pool_sizes}"
                )
            if multiresolution_layer_local_radii is not None:
                print(
                    "Multi-resolution per-layer local radii: "
                    f"{multiresolution_layer_local_radii}"
                )
        if use_residual_budget_routing:
            print(
                "Residual-debt interaction routing: "
                f"target_sparsity={residual_budget_target_sparsity}, "
                f"local_radius={residual_budget_local_radius}, "
                f"remote_pool={residual_budget_remote_pool_size}x"
                f"{residual_budget_remote_pool_size}, "
                f"routing_parent={residual_budget_routing_parent_size}x"
                f"{residual_budget_routing_parent_size}, "
                f"routing_phase={residual_budget_routing_phase_mode}, "
                f"momentum={residual_budget_momentum}, "
                f"service_conditioned_momentum="
                f"{residual_budget_service_conditioned_momentum}, "
                f"repayment={residual_budget_repayment}, "
                f"repayment_mode={residual_budget_repayment_mode}, "
                f"service_credit_scale="
                f"{residual_budget_service_credit_scale}, "
                f"temperature={residual_budget_temperature}, "
                f"exact_fraction={residual_budget_exact_fraction}, "
                f"surface_weight={residual_budget_surface_weight}, "
                f"granularity={residual_budget_selection_granularity}, "
                f"frame_detail_power={residual_budget_frame_detail_power}, "
                f"spatial_detail_power={residual_budget_spatial_detail_power}, "
                f"cell_scorer={residual_budget_cell_scorer}, "
                f"carrier_residual_balance="
                f"{residual_budget_carrier_residual_balance}, "
                f"cell_phases={residual_budget_cell_refinement_phases}, "
                f"cell_precision={residual_budget_cell_precision_fraction}, "
                f"mass_conserving_refinement="
                f"{residual_budget_mass_conserving_refinement}, "
                f"exact_mass_conserving_refinement="
                f"{residual_budget_exact_mass_conserving_refinement}, "
                f"cell_importance={residual_budget_cell_importance_weight}, "
                f"cell_importance_floor={residual_budget_cell_importance_floor}, "
                f"cell_importance_gate="
                f"{residual_budget_cell_importance_gate_threshold}@"
                f"{residual_budget_cell_importance_gate_temperature}, "
                f"query_chunk={residual_budget_query_frame_chunk}"
            )
            if residual_budget_unified_incremental_service:
                structural_limit = (
                    1.0 - 1.0 / (residual_budget_remote_pool_size ** 2)
                )
                execution = (
                    "parent-carrier"
                    if residual_budget_target_sparsity
                    >= structural_limit - 1e-8
                    else "child-carrier"
                )
                print(
                    "Unified debt service: enabled, "
                    f"execution={execution}, "
                    f"structural_switch={structural_limit:.4f}, "
                    "service_conditioned_momentum=True, "
                    "repayment_mode=effective_bounded_deficit, "
                    "mass_conserving_refinement=True"
                )

    return aggregator, aux_output_store


def print_layerwise_sparse_info(layer_config, pool_mode, use_hilbert, aux_output, used_layer_sparsity_ratios=False):
    """Print layer-wise sparsity configuration."""
    print("=" * 60)
    print("Layer-wise Sparse VGGT Configuration")
    print("=" * 60)
    print(f"Pool mode: {pool_mode}")
    print(f"Use Hilbert: {use_hilbert}")
    print(f"Aux output: {aux_output}")
    if used_layer_sparsity_ratios:
        print(f"Config source: layer_sparsity_ratios (list)")
    else:
        print(f"Config source: layer_config (dict) or uniform defaults")
    print("-" * 60)
    print("Layer-wise sparsity:")

    for layer_id in sorted(layer_config.keys()):
        config = layer_config[layer_id]
        sr = config.get("sparse_ratio")
        cdf = config.get("cdf_threshold")

        if sr is not None and cdf is not None:
            print(f"  Layer {layer_id:2d}: sparse_ratio={sr:.3f}, cdf_threshold={cdf:.3f}")
        elif sr is not None:
            print(f"  Layer {layer_id:2d}: sparse_ratio={sr:.3f}")
        elif cdf is not None:
            print(f"  Layer {layer_id:2d}: cdf_threshold={cdf:.3f}")

    # Compute average sparsity
    sparse_ratios = [
        c.get("sparse_ratio", 0.0) for c in layer_config.values()
        if c.get("sparse_ratio") is not None
    ]
    if sparse_ratios:
        avg_sparsity = sum(sparse_ratios) / len(sparse_ratios)
        print("-" * 60)
        print(f"Average sparse_ratio: {avg_sparsity:.3f}")
        print(f"Expected sparsity: ~{avg_sparsity * 100:.1f}%")
    print("=" * 60)


def load_layer_config_from_json(json_path: str) -> dict[int, dict]:
    """Load layer-wise sparsity config from JSON file.

    Expected JSON format:
    {
        "0": {"sparse_ratio": 0.15, "cdf_threshold": null},
        "1": {"sparse_ratio": 0.12, "cdf_threshold": null},
        ...
    }

    Args:
        json_path: Path to JSON config file.

    Returns:
        Dict mapping layer_id (int) to config dict.
    """
    import json
    from pathlib import Path

    with Path(json_path).open("r") as f:
        config = json.load(f)

    # Convert string keys to int
    return {int(k): v for k, v in config.items()}
