"""Loss functions and multi-task loss utilities for RMR Core.

Re-exports modular loss implementations:
  - rmr_core.cell_loss: balanced_smooth_l1
  - rmr_core.count_loss: negative_binomial_nll_mean_dispersion, count_magnitude_loss, global_count_loss
  - rmr_core.region_loss: scale_balanced_region_rate_loss
  - rmr_core.dm_loss: block_sum_2d, flat_dm_block_loss, flat_dm16_loss, hierarchical_dm_loss, multiscale_dm_loss
Provides:
  - LossConfig & compute_losses: Legacy RMR-v2 composite loss computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from rmr_core.operators import RegionSet, regional_sum
from .cell_loss import balanced_smooth_l1
from .count_loss import (
    count_magnitude_loss,
    global_count_loss,
    negative_binomial_nll_mean_dispersion,
)
from .region_loss import scale_balanced_region_rate_loss
from .dm_loss import (
    block_sum_2d,
    dm_nll_none,
    flat_dm_block_loss,
    flat_dm16_loss,
    hierarchical_dm_loss,
    multiscale_dm_loss,
    probs_from_positive_mass,
)

__all__ = [
    "LossConfig",
    "balanced_smooth_l1",
    "block_sum_2d",
    "compute_losses",
    "count_magnitude_loss",
    "dm_nll_none",
    "flat_dm16_loss",
    "flat_dm_block_loss",
    "global_count_loss",
    "hierarchical_dm_loss",
    "multiscale_dm_loss",
    "negative_binomial_nll_mean_dispersion",
    "probs_from_positive_mass",
    "scale_balanced_region_rate_loss",
]


@dataclass
class LossConfig:
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.25
    lambda_region_head: float = 0.20
    lambda_region_map: float = 0.20
    lambda_deep_supervision: float = 0.0
    count_loss_mode: str = "nb"
    nb_dispersion: float = 50.0
    kappa_flat16: float = 20.0
    cell_beta: float = 1.0
    region_beta: float = 0.1
    normalize_flat_dm16: bool = True

    @property
    def lambda_global(self) -> float:
        return self.lambda_count

    @lambda_global.setter
    def lambda_global(self, val: float) -> None:
        self.lambda_count = val


def compute_losses(
    outputs: dict,
    target_y: torch.Tensor,
    variant: str,
    cfg: LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants (RMR-v2)."""
    if cfg is None:
        cfg = LossConfig()
    y = outputs["y"]
    y0 = outputs["y0"]
    regions: RegionSet | None = outputs.get("regions")
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["count"] = count_magnitude_loss(
        y, target_y, mode=cfg.count_loss_mode, dispersion=cfg.nb_dispersion
    )
    losses["global"] = losses["count"]

    if cfg.lambda_flat_dm16 > 0:
        losses["flat_dm16"] = flat_dm16_loss(
            y0,
            target_y,
            kappa=cfg.kappa_flat16,
            normalize_by_count=cfg.normalize_flat_dm16,
        )
    else:
        losses["flat_dm16"] = y0.new_tensor(0.0)

    if variant in {"region_loss", "region_aux", "learned_project", "rmr"}:
        if regions is None:
            raise ValueError(f"Variant {variant} requires regions in outputs")
        target_region = regional_sum(target_y, regions.boxes)

        if variant == "region_loss":
            pred_region = regional_sum(y, regions.boxes)
            losses["region_map"] = scale_balanced_region_rate_loss(
                pred_region, target_region, regions, beta=cfg.region_beta
            )

        if variant in {"region_aux", "learned_project", "rmr"}:
            b_region = outputs["b_region"]
            losses["region_head"] = scale_balanced_region_rate_loss(
                b_region, target_region, regions, beta=cfg.region_beta
            )

    iterates = outputs.get("iterates", [])
    if variant in {"local_refine", "learned_project", "rmr"} and len(iterates) > 2:
        mids = iterates[1:-1]
        if mids:
            losses["deep"] = torch.stack([
                balanced_smooth_l1(m, target_y, beta=cfg.cell_beta) for m in mids
            ]).mean()

    total = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * losses["flat_dm16"]
        + cfg.lambda_cell * losses["cell"]
    )
    if "region_map" in losses:
        total = total + cfg.lambda_region_map * losses["region_map"]
    if "region_head" in losses:
        total = total + cfg.lambda_region_head * losses["region_head"]
    if "deep" in losses and cfg.lambda_deep_supervision > 0:
        total = total + cfg.lambda_deep_supervision * losses["deep"]

    losses["total"] = total
    return losses
