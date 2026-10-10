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


def _decoupled_beta_nb_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    dispersion: torch.Tensor,
    beta: float,
    faithful: bool,
    norm_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Seitzer et al. (ICLR 2022) Beta-NB NLL with Stirn et al. (AISTATS 2023) faithful decoupling."""
    beta_val = float(max(0.0, min(1.0, beta)))
    if not (beta_val > 0.0 or faithful):
        return negative_binomial_nll_mean_dispersion(
            target, mean, dispersion=dispersion, reduction="none", check_bounds=False
        )

    nll_mean = negative_binomial_nll_mean_dispersion(
        target, mean, dispersion=dispersion.detach(), reduction="none", check_bounds=False
    )
    nll_disp = negative_binomial_nll_mean_dispersion(
        target, mean.detach(), dispersion=dispersion, reduction="none", check_bounds=False
    )
    if beta_val > 0.0:
        mu_d = mean.detach().float().clamp_min(1e-6)
        r_d = dispersion.detach().float().clamp_min(1e-6)
        var_d = mu_d + mu_d.square() / r_d
        w_beta = var_d.pow(beta_val)
        if norm_mask is not None:
            m_sum = norm_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            w_mean = ((w_beta * norm_mask).sum(dim=-1, keepdim=True) / m_sum).clamp_min(1e-6)
        else:
            w_mean = w_beta.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        w_beta = (w_beta / w_mean).to(dtype=nll_mean.dtype)
    else:
        w_beta = torch.ones_like(nll_mean)

    return w_beta * nll_mean + nll_disp - nll_disp.detach()


def truncated_nb_nll_loss(
    mu_count: torch.Tensor,
    dispersion: torch.Tensor,
    target_region: torch.Tensor,
    regional_nb_beta: float = 0.0,
    faithful_regional_nb: bool = False,
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

    per_region_nll = _decoupled_beta_nb_nll(
        target_region,
        mu_count,
        dispersion=dispersion,
        beta=regional_nb_beta,
        faithful=faithful_regional_nb,
        norm_mask=occ_mask,
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
    regional_nb_beta: float = 0.0,
    faithful_regional_nb: bool = False,
) -> torch.Tensor:
    """Average proper NB NLL within scale, then average across multiscale dictionaries."""
    if target_region.shape != mean_region.shape:
        raise ValueError(
            f"target/mean mismatch: {target_region.shape} vs {mean_region.shape}"
        )
    if dispersion_region.shape != mean_region.shape:
        raise ValueError(
            f"dispersion/mean mismatch: {dispersion_region.shape} vs {mean_region.shape}"
        )

    scale_losses = []
    num_scales = int(getattr(regions, "num_scales", 3))
    for sid in range(num_scales):
        mask = (regions.scale_id == sid)
        t_count = mask.float().sum()
        if t_count <= 0:
            continue

        nll_s = _decoupled_beta_nb_nll(
            target_region[..., mask],
            mean_region[..., mask],
            dispersion=dispersion_region[..., mask],
            beta=regional_nb_beta,
            faithful=faithful_regional_nb,
        )

        if mass_weight_alpha > 0.0:
            t_s = target_region[..., mask].float()
            t_mean = t_s.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            w_raw = 1.0 + float(mass_weight_alpha) * (t_s / t_mean)
            w_norm = w_raw / w_raw.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            loss_s = (w_norm * nll_s).mean()
        else:
            loss_s = (nll_s.sum(dim=-1) / t_count.clamp_min(1.0)).mean()

        scale_losses.append(loss_s)

    return torch.stack(scale_losses).mean() if scale_losses else (mean_region.sum()) * 0.0
