from __future__ import annotations

from dataclasses import dataclass

import torch

from rmr_count.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    negative_binomial_nll_mean_dispersion,
)
from rmr_count.operators import (
    RegionSet,
    regional_sum,
)


def scale_balanced_regional_nb_nll(
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Average proper NB NLL within scale, then average scales."""

    if target_region.shape != mean_region.shape:
        raise ValueError(
            f"target/mean mismatch: "
            f"{target_region.shape} vs {mean_region.shape}"
        )

    if dispersion_region.shape != mean_region.shape:
        raise ValueError(
            f"dispersion/mean mismatch: "
            f"{dispersion_region.shape} vs {mean_region.shape}"
        )

    per_region = negative_binomial_nll_mean_dispersion(
        target_region,
        mean_region,
        dispersion=dispersion_region,
        reduction="none",
    )

    losses = []

    for sid in torch.unique(regions.scale_id):
        if int(sid.item()) < 0:
            continue

        mask = regions.scale_id == sid

        if bool(mask.any()):
            losses.append(
                per_region[..., mask].mean()
            )

    if not losses:
        raise RuntimeError(
            "No valid regional scales for NB loss"
        )

    return torch.stack(losses).mean()


@dataclass
class RMRv3LossConfig:
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.25
    lambda_region_nb: float = 0.20

    count_loss_mode: str = "nb"
    count_nb_dispersion: float = 50.0

    kappa_flat16: float = 20.0
    normalize_flat_dm16: bool = True

    cell_beta: float = 1.0


def compute_rmr_v3_losses(
    outputs: dict,
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig = RMRv3LossConfig(),
) -> dict[str, torch.Tensor]:
    y = outputs["y"]
    y0 = outputs["y0"]

    regions: RegionSet = outputs["regions"]

    mean_region = outputs["b_region"]
    dispersion_region = outputs["region_dispersion"]

    target_region = regional_sum(
        target_y,
        regions.boxes,
        out_dtype=torch.float32,
    )

    losses: dict[str, torch.Tensor] = {}

    losses["count"] = count_magnitude_loss(
        y,
        target_y,
        mode=cfg.count_loss_mode,
        dispersion=cfg.count_nb_dispersion,
    )

    losses["flat_dm16"] = flat_dm16_loss(
        y0,
        target_y,
        kappa=cfg.kappa_flat16,
        normalize_by_count=cfg.normalize_flat_dm16,
    )

    losses["cell"] = balanced_smooth_l1(
        y,
        target_y,
        beta=cfg.cell_beta,
    )

    losses["region_nb"] = scale_balanced_regional_nb_nll(
        target_region.float(),
        mean_region.float(),
        dispersion_region.float(),
        regions,
    )

    losses["total"] = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * losses["flat_dm16"]
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_nb * losses["region_nb"]
    )

    return losses
