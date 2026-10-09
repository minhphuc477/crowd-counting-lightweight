"""Probabilistic regional Negative-Binomial and Hurdle occupancy losses.

Provides:
1. scale_balanced_regional_nb_nll: Regional Negative-Binomial NLL balanced across box scales.
2. hurdle_focal_bce_loss: Focal BCE loss for Hurdle occupancy logit prediction.
3. truncated_nb_nll_loss: Truncated Negative-Binomial NLL computed only on non-zero regions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rmr_core.losses import negative_binomial_nll_mean_dispersion
from rmr_core.operators import RegionSet


def hurdle_focal_bce_loss(
    pi_logit: torch.Tensor,
    target_region: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Focal BCE loss for Hurdle head occupancy prediction."""
    if pi_logit.ndim == 1:
        pi_logit = pi_logit.unsqueeze(0).unsqueeze(0)
    elif pi_logit.ndim == 2:
        pi_logit = pi_logit.unsqueeze(1)
    if target_region.ndim == 1:
        target_region = target_region.unsqueeze(0).unsqueeze(0)
    elif target_region.ndim == 2:
        target_region = target_region.unsqueeze(1)

    y_bin = (target_region > 0.5).float()
    bce = F.binary_cross_entropy_with_logits(
        pi_logit.float(),
        y_bin,
        reduction="none",
    )

    with torch.no_grad():
        p_t = torch.where(
            y_bin > 0.5,
            torch.sigmoid(pi_logit.float()),
            1.0 - torch.sigmoid(pi_logit.float()),
        )
        focal_weight = (1.0 - p_t.clamp(min=1e-6, max=1.0 - 1e-6)).pow(float(gamma))

    return (focal_weight * bce).mean()


def truncated_nb_nll_loss(
    mu_count: torch.Tensor,
    dispersion: torch.Tensor,
    target_region: torch.Tensor,
) -> torch.Tensor:
    """Truncated NB NLL computed only on occupied regions (target >= 1)."""
    if target_region.ndim == 1:
        target_region = target_region.unsqueeze(0).unsqueeze(0)
    elif target_region.ndim == 2:
        target_region = target_region.unsqueeze(1)
    if mu_count.ndim == 1:
        mu_count = mu_count.unsqueeze(0).unsqueeze(0)
    elif mu_count.ndim == 2:
        mu_count = mu_count.unsqueeze(1)
    if dispersion.ndim == 1:
        dispersion = dispersion.unsqueeze(0).unsqueeze(0)
    elif dispersion.ndim == 2:
        dispersion = dispersion.unsqueeze(1)

    occ_mask = (target_region > 0.5).float()
    occ_count = occ_mask.sum(dim=-1, keepdim=True)

    per_region_nll = negative_binomial_nll_mean_dispersion(
        target_region,
        mu_count,
        dispersion=dispersion,
        reduction="none",
    )
    sample_nll = (per_region_nll * occ_mask).sum(dim=-1, keepdim=True) / occ_count.clamp_min(1.0)
    has_occ = (occ_count > 0).float()
    total_samples = has_occ.sum().clamp_min(1.0)
    return (sample_nll * has_occ).sum() / total_samples


def scale_balanced_regional_nb_nll(
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    regions: RegionSet,
    mass_weight_alpha: float = 0.0,
) -> torch.Tensor:
    """Average proper NB NLL within scale, then average across multiscale dictionaries.

    When mass_weight_alpha > 0, weights region errors by target mass to prevent
    empty background boxes from dominating dense crowd boxes.
    """
    if target_region.shape != mean_region.shape:
        raise ValueError(
            f"target/mean mismatch: {target_region.shape} vs {mean_region.shape}"
        )
    if dispersion_region.shape != mean_region.shape:
        raise ValueError(
            f"dispersion/mean mismatch: {dispersion_region.shape} vs {mean_region.shape}"
        )

    per_region = negative_binomial_nll_mean_dispersion(
        target_region,
        mean_region,
        dispersion=dispersion_region,
        reduction="none",
        check_bounds=False,
    )

    scale_losses = []
    num_scales = int(getattr(regions, "num_scales", 3))
    for sid in range(num_scales):
        mask = (regions.scale_id == sid)
        t_count = mask.float().sum()
        if mass_weight_alpha > 0.0:
            t_s = target_region[..., mask].float()
            t_mean = t_s.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            w_raw = 1.0 + float(mass_weight_alpha) * (t_s / t_mean)
            w_norm = w_raw / w_raw.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            loss_s = (w_norm * per_region[..., mask]).mean()
        else:
            loss_s = (per_region[..., mask].sum(dim=-1) / t_count.clamp_min(1.0)).mean()
        scale_losses.append(loss_s)

    return torch.stack(scale_losses).mean() if scale_losses else (per_region.sum()) * 0.0
