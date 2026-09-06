"""rmr_v3/losses.py — Loss functions for RMR-v3.

All losses re-use proven building blocks from rmr_count.losses wherever possible.
New contributions:
  - reliability_calibration_loss: diagnostic correlation gate (Section 41)
  - compute_losses_v3: top-level orchestrator for V3-A (control) and V3-B (main)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from rmr_count.operators import RegionSet, regional_sum
from rmr_count.losses import (
    balanced_smooth_l1,
    count_magnitude_loss,
    flat_dm16_loss,
    scale_balanced_region_rate_loss,
    negative_binomial_nll_mean_dispersion,
)


# ---------------------------------------------------------------------------
# Reliability calibration diagnostic (NOT a training loss)
# ---------------------------------------------------------------------------
@torch.no_grad()
def reliability_correlation_stats(
    w_region: torch.Tensor,    # [B, 1, M]
    b_region: torch.Tensor,    # [B, 1, M]
    target_map: torch.Tensor,  # [B, 1, H, W]
    regions: RegionSet,
    rate_std_floor: float = 0.01,
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute Pearson/Spearman correlation between rate_variance and abs_rate_error.

    This is the RELIABILITY MECHANISM GATE (Section 28, 41):
        If rho(rate_variance, abs_rate_error) is near zero, the weighted
        reconciliation has no scientific support even if MAE changes.

    The gate uses rate_variance = rho_R (from learned dispersion or raw rate)
    as a proxy for uncertainty, vs actual abs_rate_error = |b_R - b_R_gt| / |R|.

    Returns:
        pearson_r:  float Pearson correlation (rate_variance, abs_rate_error)
        spearman_r: float Spearman correlation (approximate via rank)
        n_regions:  int number of regions in this batch
    """
    b_region_gt = regional_sum(target_map, regions.boxes).unsqueeze(1)  # [B, 1, M]
    area = regions.area.to(dtype=torch.float32).view(1, 1, -1).clamp_min(1.0)

    rate_gt = b_region_gt.to(torch.float32) / area
    rate_pred = b_region.to(torch.float32) / area
    abs_rate_err = (rate_pred - rate_gt).abs().flatten()   # [B*M]

    # Use w_region as variance proxy (lower w -> higher uncertainty signal)
    # Invert: uncertainty = 1/w so that high uncertainty -> low w
    w_f32 = w_region.to(torch.float32).flatten().clamp_min(eps)
    uncertainty = 1.0 / w_f32                              # [B*M]

    # Pearson correlation
    def _pearson(a: torch.Tensor, b_: torch.Tensor) -> float:
        a = a - a.mean()
        b_ = b_ - b_.mean()
        denom = (a.norm() * b_.norm()).clamp_min(eps)
        return (a * b_).sum().item() / denom.item()

    # Approximate Spearman via rank (argsort twice)
    def _spearman(a: torch.Tensor, b_: torch.Tensor) -> float:
        rank_a = a.argsort().argsort().float()
        rank_b = b_.argsort().argsort().float()
        return _pearson(rank_a, rank_b)

    pearson_r = _pearson(uncertainty, abs_rate_err)
    spearman_r = _spearman(uncertainty, abs_rate_err)

    return {
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "n_regions": int(abs_rate_err.numel()),
    }


# ---------------------------------------------------------------------------
# Reliability regularization (optional soft entropy)
# ---------------------------------------------------------------------------
def reliability_entropy_loss(
    w_region: torch.Tensor,  # [B, 1, M]
    w_min: float = 0.25,
    w_max: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft entropy regularizer to prevent reliability collapse (all same weight).

    Encourages the learned weights to be differentiated across regions.
    Normalize w to [0, 1] range, then compute -sum(p * log(p)).
    lambda_reliability_entropy in config controls its contribution.
    Default: 0.0 (disabled unless user opts in).
    """
    w_norm = (w_region - w_min) / (w_max - w_min + eps)   # [B, 1, M] in [0,1]
    w_norm = w_norm.clamp(eps, 1.0 - eps)
    # Normalize to probability distribution over regions
    p = w_norm / w_norm.sum(dim=-1, keepdim=True).clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1).mean()
    return -entropy  # minimize negative entropy = maximize entropy


# ---------------------------------------------------------------------------
# Loss config
# ---------------------------------------------------------------------------
@dataclass
class LossConfigV3:
    # Standard losses (matching RMR-v2 for fair comparison)
    lambda_count: float = 1.0
    lambda_flat_dm16: float = 1.0
    lambda_cell: float = 0.25
    lambda_region_head: float = 0.20

    count_loss_mode: str = "nb"
    nb_dispersion: float = 50.0        # legacy fallback; V3 uses learned dispersion
    kappa_flat16: float = 20.0
    cell_beta: float = 1.0
    region_beta: float = 0.1
    normalize_flat_dm16: bool = True

    # Reliability regularization (optional)
    lambda_reliability_entropy: float = 0.0   # 0 = disabled


# ---------------------------------------------------------------------------
# Top-level loss orchestrator for RMR-v3
# ---------------------------------------------------------------------------
def compute_losses_v3(
    outputs: dict,
    target_y: torch.Tensor,    # [B, 1, H, W] GT density map
    cfg: LossConfigV3 = LossConfigV3(),
) -> dict[str, torch.Tensor]:
    """Compute all losses for RMR-v3 (both V3-A control and V3-B main).

    Loss components (identical structure to RMR-v2 for causal comparison):
        L_count:   NB global count loss on Y_final
        L_dm16:    Flat-DM16 allocation loss on Y0 (observer)
        L_cell:    Balanced SmoothL1 on Y_final vs GT
        L_region:  Scale-balanced rate loss on b_region vs GT

    No new loss term is added relative to RMR-v2. The only structural
    difference is that b_region and w_region come from ReliabilityRegionalHead
    instead of ScaleMatchedRegionalEvidenceHead.

    NOTE: w_region does NOT enter the loss — it is only used in the solver
    forward pass. The reliability calibration check is a diagnostic (Section 32).
    """
    y = outputs["y"]           # [B, 1, H, W] final refined
    y0 = outputs["y0"]         # [B, 1, H, W] initial
    b_region = outputs["b_region"]  # [B, 1, M]
    regions: RegionSet = outputs["regions"]
    losses: dict[str, torch.Tensor] = {}

    # Cell-level loss
    losses["cell"] = balanced_smooth_l1(y, target_y, beta=cfg.cell_beta)

    # Global count loss (NB on final Y)
    losses["count"] = count_magnitude_loss(
        y, target_y, mode=cfg.count_loss_mode, dispersion=cfg.nb_dispersion
    )
    losses["global"] = losses["count"]  # backward-compatible key

    # FlatDM16 allocation on Y0 (observer, not solver output)
    losses["flat_dm16"] = flat_dm16_loss(
        y0, target_y,
        kappa=cfg.kappa_flat16,
        normalize_by_count=cfg.normalize_flat_dm16,
    )

    # Regional evidence head loss
    target_region = regional_sum(target_y, regions.boxes).unsqueeze(1)  # [B, 1, M]
    losses["region_head"] = scale_balanced_region_rate_loss(
        b_region, target_region, regions, beta=cfg.region_beta
    )

    # Optional reliability entropy regularization
    if cfg.lambda_reliability_entropy > 0.0:
        w_region = outputs.get("w_region")
        if w_region is not None:
            losses["reliability_entropy"] = reliability_entropy_loss(w_region)
        else:
            losses["reliability_entropy"] = y.new_tensor(0.0)
    else:
        losses["reliability_entropy"] = y.new_tensor(0.0)

    # Total loss
    total = (
        cfg.lambda_count * losses["count"]
        + cfg.lambda_flat_dm16 * losses["flat_dm16"]
        + cfg.lambda_cell * losses["cell"]
        + cfg.lambda_region_head * losses["region_head"]
        + cfg.lambda_reliability_entropy * losses["reliability_entropy"]
    )
    losses["total"] = total
    return losses
