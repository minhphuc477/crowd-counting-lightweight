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

__all__ = [
    "evaluate_phase_shift_single_image",
    "evaluate_separability_single_image",
    "sample_feature_at_image_coord",
    "evaluate_effective_rank_single_image",
    "compute_spectral_rank_metrics",
    "evaluate_gradient_allocation_single_batch",
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
]
