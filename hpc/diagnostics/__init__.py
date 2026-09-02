"""HPC / NTPC Diagnostic Suite (D0): D-R, D-K, D-L, D-M."""

from .phase_shift import evaluate_phase_shift_single_image
from .separability import evaluate_separability_single_image, sample_feature_at_image_coord
from .effective_rank import evaluate_effective_rank_single_image, compute_spectral_rank_metrics
from .gradient_allocation import evaluate_gradient_allocation_single_batch
from .cardinality_sufficiency_v2 import (
    PCAProjector,
    TinyMLPProbe,
    avgpool2x,
    blurpool2x,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_mlp,
    fit_predict_ridge,
    pack_2x2_features,
    pack_child_counts,
    summarize_prediction,
)
from .cardinality_sufficiency_v3 import (
    ImageCellCollector,
    compute_image_weighted_metrics,
    image_weighted_bootstrap_diff,
    image_weighted_mean,
    paired_seed_mlp_eval,
)
from .objective_mechanism_audit import (
    compute_audit_for_mode,
    compute_component_gradients,
    compute_gradient_metrics,
    compute_pairwise_cosine,
    stratify_by_density,
    summarize_audit_group,
    sweep_kappa_on_crop,
)

__all__ = [
    "evaluate_phase_shift_single_image",
    "evaluate_separability_single_image",
    "sample_feature_at_image_coord",
    "evaluate_effective_rank_single_image",
    "compute_spectral_rank_metrics",
    "evaluate_gradient_allocation_single_batch",
    # v2
    "PCAProjector",
    "TinyMLPProbe",
    "avgpool2x",
    "blurpool2x",
    "bootstrap_image_mean_difference",
    "build_representation_grid",
    "fit_predict_mlp",
    "fit_predict_ridge",
    "pack_2x2_features",
    "pack_child_counts",
    "summarize_prediction",
    # v3
    "ImageCellCollector",
    "compute_image_weighted_metrics",
    "image_weighted_bootstrap_diff",
    "image_weighted_mean",
    "paired_seed_mlp_eval",
    # objective audit
    "compute_component_gradients",
    "compute_gradient_metrics",
    "compute_pairwise_cosine",
    "compute_audit_for_mode",
    "sweep_kappa_on_crop",
    "stratify_by_density",
    "summarize_audit_group",
]
