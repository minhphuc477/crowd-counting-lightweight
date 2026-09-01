"""HPC / NTPC Diagnostic Suite (D0): D-R, D-K, D-L, D-M."""

from .phase_shift import evaluate_phase_shift_single_image
from .separability import evaluate_separability_single_image, sample_feature_at_image_coord
from .effective_rank import evaluate_effective_rank_single_image, compute_spectral_rank_metrics
from .gradient_allocation import evaluate_gradient_allocation_single_batch

__all__ = [
    "evaluate_phase_shift_single_image",
    "evaluate_separability_single_image",
    "sample_feature_at_image_coord",
    "evaluate_effective_rank_single_image",
    "compute_spectral_rank_metrics",
    "evaluate_gradient_allocation_single_batch",
]
