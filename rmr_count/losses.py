"""Backward-compatibility shim re-exporting from rmr_v2.losses."""
from __future__ import annotations

from rmr_v2.losses import (
    LossConfig,
    balanced_smooth_l1,
    block_sum_2d,
    compute_losses,
    count_magnitude_loss,
    dm_nll_none,
    flat_dm16_loss,
    negative_binomial_nll_mean_dispersion,
    probs_from_positive_mass,
    scale_balanced_region_rate_loss,
)

__all__ = [
    "LossConfig",
    "balanced_smooth_l1",
    "block_sum_2d",
    "compute_losses",
    "count_magnitude_loss",
    "dm_nll_none",
    "flat_dm16_loss",
    "negative_binomial_nll_mean_dispersion",
    "probs_from_positive_mass",
    "scale_balanced_region_rate_loss",
]
