"""HPC / NTPC Diagnostic Suite (D0): D-R, D-K, D-L, D-M."""

from .phase_shift import evaluate_phase_shift_single_image
from .separability import evaluate_separability_single_image, compute_knn_spacing
from .effective_rank import evaluate_effective_rank_single_image, compute_spectral_rank_metrics
from .gradient_allocation import evaluate_gradient_allocation_single_batch

__all__ = [
    "evaluate_phase_shift_single_image",
    "evaluate_separability_single_image",
    "compute_knn_spacing",
    "evaluate_effective_rank_single_image",
    "compute_spectral_rank_metrics",
    "evaluate_gradient_allocation_single_batch",
]
