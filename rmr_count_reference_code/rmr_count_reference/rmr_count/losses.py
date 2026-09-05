from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F

from .operators import RegionSet, regional_sum


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions.

    This is deliberately a simple shared carrier loss, not a paper contribution.
    """
    per = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    pos = target > 0
    neg = ~pos
    terms = []
    if pos.any():
        terms.append(per[pos].mean())
    if neg.any():
        terms.append(per[neg].mean())
    if not terms:
        return per.mean()
    return torch.stack(terms).mean()


def global_count_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Stable global count loss on log1p counts."""
    pn = pred.sum(dim=(-2, -1))
    tn = target.sum(dim=(-2, -1))
    return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)


def scale_balanced_region_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 1.0,
) -> torch.Tensor:
    """Average region-count SmoothL1 equally across region scales."""
    if pred_region.shape != target_region.shape:
        raise ValueError(f"shape mismatch: {pred_region.shape} vs {target_region.shape}")
    losses = []
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        if mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_region[..., mask],
                    target_region[..., mask],
                    reduction="mean",
                    beta=beta,
                )
            )
    return torch.stack(losses).mean()


@dataclass
class LossConfig:
    lambda_global: float = 0.10
    lambda_region_map: float = 0.20
    lambda_region_head: float = 0.20
    lambda_deep_supervision: float = 0.10
    cell_beta: float = 1.0
    region_beta: float = 2.0


def compute_losses(
    outputs: dict,
    target_y: torch.Tensor,
    variant: str,
    cfg: LossConfig = LossConfig(),
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants.

    Variant semantics:
      direct:          fine + global only
      region_loss:     direct + training-only regional loss on final map
      region_aux:      direct + auxiliary regional evidence head
      learned_project: region_aux + learned inference projector
      rmr:             region_aux + exact-adjoint reconciliation
    """
    y = outputs["y"]
    regions: RegionSet = outputs["regions"]
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["global"] = global_count_loss(y, target_y)

    target_region = regional_sum(target_y, regions.boxes)

    if variant == "region_loss":
        pred_region = regional_sum(y, regions.boxes)
        losses["region_map"] = scale_balanced_region_loss(
            pred_region, target_region, regions, beta=cfg.region_beta
        )

    if variant in {"region_aux", "learned_project", "rmr"}:
        b_region = outputs["b_region"]
        losses["region_head"] = scale_balanced_region_loss(
            b_region, target_region, regions, beta=cfg.region_beta
        )

    # Optional weak deep supervision on intermediate positive measures for iterative variants.
    iterates = outputs.get("iterates", [])
    if variant in {"learned_project", "rmr"} and len(iterates) > 2:
        mids = iterates[1:-1]
        if mids:
            losses["deep"] = torch.stack([
                balanced_smooth_l1(m, target_y, beta=cfg.cell_beta) for m in mids
            ]).mean()

    total = losses["cell"] + cfg.lambda_global * losses["global"]
    if "region_map" in losses:
        total = total + cfg.lambda_region_map * losses["region_map"]
    if "region_head" in losses:
        total = total + cfg.lambda_region_head * losses["region_head"]
    if "deep" in losses:
        total = total + cfg.lambda_deep_supervision * losses["deep"]
    losses["total"] = total
    return losses
