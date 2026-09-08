from __future__ import annotations

from dataclasses import dataclass

import torch

from rmr_core.operators import (
    RegionSet,
    regional_sum,
)
from rmr_v2.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    hierarchical_dm_loss,
    negative_binomial_nll_mean_dispersion,
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

    use_hierarchical_dm: bool = False
    dm_block_sizes_px: tuple[int, ...] = (16, 32, 64)
    dm_weights: tuple[float, ...] = (0.50, 0.30, 0.20)
    dm_kappas: tuple[float, ...] = (20.0, 20.0, 20.0)

    count_loss_mode: str = "nb"
    count_nb_dispersion: float = 50.0

    kappa_flat16: float = 20.0
    normalize_flat_dm16: bool = True

    cell_beta: float = 1.0


def compute_rmr_v3_losses(
    outputs: dict,
    target_y: torch.Tensor,
    cfg: RMRv3LossConfig | None = None,
) -> dict[str, torch.Tensor]:
    if cfg is None:
        cfg = RMRv3LossConfig()

    y = outputs["y"].float()
    y0 = outputs["y0"].float()
    target_float = target_y.float()

    regions: RegionSet = outputs["regions"]

    mean_region = outputs["b_region"].float()
    dispersion_region = outputs["region_dispersion"].float()

    target_region = regional_sum(
        target_float,
        regions.boxes,
        out_dtype=torch.float32,
    )

    losses: dict[str, torch.Tensor] = {}

    losses["count"] = count_magnitude_loss(
        y,
        target_float,
        mode=cfg.count_loss_mode,
        dispersion=cfg.count_nb_dispersion,
    )

    if cfg.use_hierarchical_dm:
        loss_allocation = hierarchical_dm_loss(
            y0,
            target_float,
            block_sizes_px=tuple(int(x) for x in cfg.dm_block_sizes_px),
            weights=tuple(float(x) for x in cfg.dm_weights),
            kappas=tuple(float(x) for x in cfg.dm_kappas),
            stride=4,
            normalize_by_count=cfg.normalize_flat_dm16,
        )
    else:
        loss_allocation = flat_dm16_loss(
            y0,
            target_float,
            kappa=cfg.kappa_flat16,
            normalize_by_count=cfg.normalize_flat_dm16,
        )

    losses["allocation"] = loss_allocation
    losses["flat_dm16"] = loss_allocation  # backward compatibility alias

    losses["cell"] = balanced_smooth_l1(
        y,
        target_float,
        beta=cfg.cell_beta,
    )

    losses["region_nb"] = scale_balanced_regional_nb_nll(
        target_region,
        mean_region,
        dispersion_region,
        regions,
    )

    losses["total"] = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * loss_allocation
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_nb * losses["region_nb"]
    )

    return losses
