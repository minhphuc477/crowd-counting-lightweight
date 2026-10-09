"""Loss functions and multi-task loss utilities for RMR Core.

Provides:
1. balanced_smooth_l1: Cell-level balanced Smooth L1 with per-sample isolation.
2. negative_binomial_nll_mean_dispersion: Exact Negative-Binomial distribution NLL.
3. count_magnitude_loss: Macro crop count loss (Negative-Binomial, log1p, Anscombe, Huber).
4. scale_balanced_region_rate_loss: Regional rate loss normalized by scale box areas.
5. Dirichlet-Multinomial losses re-exported from rmr_core.dm_loss.
6. LossConfig & compute_losses: Legacy RMR-v2 composite loss computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet, regional_sum
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
    "balanced_smooth_l1",
    "negative_binomial_nll_mean_dispersion",
    "count_magnitude_loss",
    "global_count_loss",
    "block_sum_2d",
    "probs_from_positive_mass",
    "dm_nll_none",
    "flat_dm_block_loss",
    "multiscale_dm_loss",
    "hierarchical_dm_loss",
    "flat_dm16_loss",
    "scale_balanced_region_rate_loss",
    "LossConfig",
    "compute_losses",
]


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    stride: int = 4,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions with strict per-sample isolation."""
    if pred.numel() == 0 or target.numel() == 0:
        return (pred.sum() + target.sum()) * 0.0
    per = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=beta)
    b_sz = pred.shape[0] if pred.ndim >= 3 else 1
    if b_sz <= 1:
        pos = target > 0
        neg = ~pos
        pos_loss = per[pos].mean() if pos.any() else per.new_tensor(0.0)
        neg_loss = per[neg].mean() if neg.any() else per.new_tensor(0.0)
        return 0.5 * (pos_loss + neg_loss)

    sample_losses = []
    for i in range(b_sz):
        per_i = per[i]
        tgt_i = target[i]
        pos_i = tgt_i > 0
        neg_i = ~pos_i
        pos_loss_i = per_i[pos_i].mean() if pos_i.any() else per_i.new_tensor(0.0)
        neg_loss_i = per_i[neg_i].mean() if neg_i.any() else per_i.new_tensor(0.0)
        sample_losses.append(0.5 * (pos_loss_i + neg_loss_i))
    return torch.stack(sample_losses).mean()


_MAX_DISPERSION = 1e4


def negative_binomial_nll_mean_dispersion(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: float | torch.Tensor = 50.0,
    eps: float = 1e-8,
    reduction: str = "mean",
    check_bounds: bool = True,
) -> torch.Tensor:
    """Negative-Binomial NLL with Var(Y) = mu + mu^2 / r evaluated in float32."""
    y = target.to(device=mean.device, dtype=torch.float32)
    mu = mean.to(dtype=torch.float32).clamp_min(eps)
    r = torch.as_tensor(dispersion, device=mean.device, dtype=torch.float32)

    if check_bounds:
        if torch.any(r <= 0) or torch.any(r > _MAX_DISPERSION) or not torch.isfinite(r).all():
            raise ValueError(
                f"Negative-Binomial dispersion parameter r must be in (0, {_MAX_DISPERSION}], got {dispersion}"
            )
        if not torch.isfinite(y).all() or torch.any(y < 0):
            raise ValueError("Negative-Binomial targets must be finite non-negative numbers")
    else:
        y = y.clamp_min(0.0)
        r = r.clamp(1e-6, _MAX_DISPERSION)

    log_r_plus_mu = torch.log(r + mu)
    nll = -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * (torch.log(r) - log_r_plus_mu)
        + y * (torch.log(mu) - log_r_plus_mu)
    )
    if reduction == "none":
        return nll
    if reduction == "sum":
        return nll.sum()
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def count_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "nb",
    dispersion: float = 50.0,
) -> torch.Tensor:
    """Total crop count loss using Negative Binomial NLL or log1p smooth L1."""
    if pred.numel() == 0 or target.numel() == 0:
        return (pred.sum() + target.sum()) * 0.0
    pn = pred.float().sum(dim=(-2, -1)).view(-1) if pred.ndim >= 2 else pred.float().view(-1)
    tn = target.float().sum(dim=(-2, -1)).view(-1) if target.ndim >= 2 else target.float().view(-1)
    if mode == "nb":
        return negative_binomial_nll_mean_dispersion(
            tn, pn, dispersion=dispersion, reduction="mean", check_bounds=False
        )
    elif mode == "log1p":
        return F.smooth_l1_loss(torch.log1p(pn), torch.log1p(tn), reduction="mean", beta=0.2)
    elif mode == "l1":
        return F.l1_loss(pn, tn, reduction="mean")
    elif mode in ("smooth_l1", "huber"):
        return F.smooth_l1_loss(pn, tn, reduction="mean", beta=1.0)
    elif mode == "anscombe":
        g_pn = 2.0 * torch.sqrt(torch.clamp_min(pn, 0.0) + 0.375)
        g_tn = 2.0 * torch.sqrt(torch.clamp_min(tn, 0.0) + 0.375)
        return F.smooth_l1_loss(g_pn, g_tn, reduction="mean", beta=0.2)
    raise ValueError(f"Unsupported count loss mode: {mode}")


global_count_loss = count_magnitude_loss


def scale_balanced_region_rate_loss(
    pred_region: torch.Tensor,
    target_region: torch.Tensor,
    regions: RegionSet,
    beta: float = 0.1,
) -> torch.Tensor:
    """Scale-balanced loss on per-cell RATE, not raw count."""
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
