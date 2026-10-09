from __future__ import annotations

from .config import RMRv3LossConfig
from .router import TargetSupervisionRouter
from .dense_scaling import compute_elementwise_dense_scaling
from .allocation import compute_single_allocation, route_allocation_loss
from .cell import (
    compute_cell_loss,
    count_harmonized_cell_loss,
    count_invariant_cell_loss,
    mass_weighted_cell_loss,
)
from .regional import (
    hurdle_focal_bce_loss,
    scale_balanced_regional_nb_nll,
    truncated_nb_nll_loss,
)
from .spatial_priors import (
    curvature_power_loss,
    foreground_gating_bce_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from .point_supervision import bayesian_loss, fidt_loss, sinkhorn_ot_loss
from .dual_supervision import compute_dual_lattice_losses
from .orchestration import compute_rmr_v3_losses

__all__ = [
    "RMRv3LossConfig",
    "compute_rmr_v3_losses",
    "compute_dual_lattice_losses",
    "TargetSupervisionRouter",
    "compute_elementwise_dense_scaling",
    "compute_single_allocation",
    "route_allocation_loss",
    "compute_cell_loss",
    "count_harmonized_cell_loss",
    "count_invariant_cell_loss",
    "mass_weighted_cell_loss",
    "scale_balanced_regional_nb_nll",
    "hurdle_focal_bce_loss",
    "truncated_nb_nll_loss",
    "curvature_power_loss",
    "topk_hard_background_loss",
    "physical_scale_alignment_loss",
    "foreground_gating_bce_loss",
    "bayesian_loss",
    "fidt_loss",
    "sinkhorn_ot_loss",
]
