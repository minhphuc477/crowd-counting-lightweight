"""Auxiliary loss facade re-exporting specialized loss modules for 100% backward compatibility.

Refactored architecture:
- rmr_v3.losses.cell: Cell-level spatial supervision losses (count-harmonized, CI-cell, mass-weighted).
- rmr_v3.losses.regional: Probabilistic regional NB and Hurdle occupancy losses.
- rmr_v3.losses.spatial_priors: Geometric curvature, hard background mining, and scale alignment losses.
"""
from __future__ import annotations

from .cell import (
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
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)

__all__ = [
    "count_harmonized_cell_loss",
    "count_invariant_cell_loss",
    "mass_weighted_cell_loss",
    "hurdle_focal_bce_loss",
    "truncated_nb_nll_loss",
    "scale_balanced_regional_nb_nll",
    "curvature_power_loss",
    "topk_hard_background_loss",
    "physical_scale_alignment_loss",
]
