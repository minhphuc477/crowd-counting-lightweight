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


def scale_balanced_region_rate_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 1.0,
) -> torch.Tensor:
    """Scale-balanced loss on per-cell RATE, not raw count.

    P0 #2 fix — normalize by area before computing loss:
        pred_rate   = b_R   / |R|    [predicted count/cell for each region]
        target_rate = N_R   / |R|    [GT count/cell for each region]
        L = (1/S) sum_s mean_{R in scale s} SmoothL1(pred_rate_R, target_rate_R)

    Rationale: the regional head predicts b_R = |R| * softplus(z_R), so:
        dL/dz_R  ∝  |R| * sigma(z_R)     [via chain rule through b_R]
    This means gradient magnitude scales with |R| even when loss is averaged per-scale.
    A 128px region (1024 cells) would produce gradients 16× larger than a 32px region
    (64 cells), breaking scale balance despite the scale-averaging wrapper.

    Training on rate eliminates this: dL_rate/dz_R ∝ sigma(z_R) regardless of |R|.
    This is consistent with the head actually estimating a PER-CELL rate.

    Applied to both `region_head` (regional evidence head b_R vs N_R) and
    `region_map` (B1 control: sum-of-map vs N_R). Fair comparison requires same objective.
    """
    if pred_region.shape != target_region.shape:
        raise ValueError(f"shape mismatch: {pred_region.shape} vs {target_region.shape}")
    area = regions.area.to(dtype=pred_region.dtype).view(1, 1, -1).clamp_min(1.0)
    pred_rate = pred_region / area
    target_rate = target_region / area
    losses = []
    for sid in torch.unique(regions.scale_id):
        mask = regions.scale_id == sid
        if mask.any():
            losses.append(
                F.smooth_l1_loss(
                    pred_rate[..., mask],
                    target_rate[..., mask],
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
    # P1 fix: regional loss operates on rate (count/cell), magnitude ~0.001–0.1.
    # beta=2.0 (old, for raw counts) placed all rate residuals in quadratic regime,
    # giving near-zero gradients. beta=0.1 keeps typical rate errors in linear regime.
    region_beta: float = 0.1


def compute_losses(
    outputs: dict,
    target_y: torch.Tensor,
    variant: str,
    cfg: LossConfig = LossConfig(),
) -> dict[str, torch.Tensor]:
    """Losses for all matched RQ variants.

    Variant semantics:
      direct:          fine + global only
      region_loss:     direct + training-only regional rate loss on final map (B1)
      region_aux:      direct + auxiliary regional evidence rate head (B2)
      local_refine:    direct + purely local learned inference refinement
      learned_project: region_aux + learned regional-membership projector
      rmr:             region_aux + exact-adjoint reconciliation
    """
    y = outputs["y"]
    regions: RegionSet = outputs["regions"]
    losses: dict[str, torch.Tensor] = {}

    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)
    losses["global"] = global_count_loss(y, target_y)

    target_region = regional_sum(target_y, regions.boxes)

    if variant == "region_loss":
        # B1 control: impose regional rate loss on the output density map.
        # Uses rate loss (P0 #2 fix) for fair comparison with B2.
        pred_region = regional_sum(y, regions.boxes)
        losses["region_map"] = scale_balanced_region_rate_loss(
            pred_region, target_region, regions, beta=cfg.region_beta
        )

    if variant in {"region_aux", "learned_project", "rmr"}:
        # Regional evidence head loss: rate-normalized for scale-balanced gradient.
        b_region = outputs["b_region"]
        losses["region_head"] = scale_balanced_region_rate_loss(
            b_region, target_region, regions, beta=cfg.region_beta
        )

    # Optional weak deep supervision on intermediate positive measures for iterative variants.
    iterates = outputs.get("iterates", [])
    if variant in {"local_refine", "learned_project", "rmr"} and len(iterates) > 2:
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
