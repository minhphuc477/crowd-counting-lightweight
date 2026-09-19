from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from rmr_core.operators import RegionSet
from rmr_core.losses import negative_binomial_nll_mean_dispersion


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
        p_t = torch.where(y_bin > 0.5, torch.sigmoid(pi_logit.float()), 1.0 - torch.sigmoid(pi_logit.float()))
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

    occ_mask = (target_region > 0.5)
    if not occ_mask.any():
        return (mu_count * 0.0 + dispersion * 0.0).sum()

    per_region_nll = negative_binomial_nll_mean_dispersion(
        target_region,
        mu_count,
        dispersion=dispersion,
        reduction="none",
    )
    b_sz = target_region.shape[0]
    sample_losses = []
    for b_idx in range(b_sz):
        occ_b = occ_mask[b_idx]
        if occ_b.any():
            sample_losses.append(per_region_nll[b_idx][occ_b].mean())
        else:
            sample_losses.append((mu_count[b_idx] * 0.0 + dispersion[b_idx] * 0.0).sum())
    return torch.stack(sample_losses).mean()


def scale_balanced_regional_nb_nll(
    target_region: torch.Tensor,
    mean_region: torch.Tensor,
    dispersion_region: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Average proper NB NLL within scale, then average scales."""
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
    )

    losses = []
    for sid in torch.unique(regions.scale_id):
        if int(sid.item()) < 0:
            continue
        mask = regions.scale_id == sid
        if bool(mask.any()):
            losses.append(per_region[..., mask].mean())

    if not losses:
        raise RuntimeError("No valid regional scales for NB loss")

    return torch.stack(losses).mean()


def curvature_power_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.01,
    threshold: float = 0.0,
    kernel_size: int = 5,
    mode: str = "none",
    smooth_scale: float = 0.02,
) -> torch.Tensor:
    """Curvature-preserving square-root power loss for high-density crowds (RMR-v11/v12)."""
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype).clamp_min(0.0)
    t_f = target.to(dtype=work_dtype).clamp_min(0.0)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    diff = torch.sqrt(y_f + float(eps)) - torch.sqrt(t_f + float(eps))
    diff_sq = diff.square()

    if mode == "none" or float(threshold) <= 0.0:
        return torch.mean(diff_sq)

    pad = int(kernel_size) // 2
    local_density = F.avg_pool2d(
        t_f, kernel_size=int(kernel_size), stride=1, padding=pad, count_include_pad=False
    )

    if mode == "hard":
        gate = (local_density >= float(threshold)).float()
    elif mode == "soft":
        gate = torch.sigmoid((local_density - float(threshold)) / float(smooth_scale))
    else:
        raise ValueError(f"Unknown curvature gate mode: '{mode}'. Expected 'none', 'hard', or 'soft'.")

    gate_sum = gate.sum()
    if gate_sum > 0:
        return (gate * diff_sq).sum() / gate_sum
    return (y_f.sum() + t_f.sum()) * 0.0


def topk_hard_background_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.05,
    bg_threshold: float = 1e-5,
) -> torch.Tensor:
    """Top-K Hard Negative Background Mining Loss (RMR-v11)."""
    if target.ndim == 3:
        target = target.unsqueeze(1)
    elif target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    if y.ndim == 3:
        y = y.unsqueeze(1)
    elif y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)

    y_f = y.float()
    t_f = target.float()
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    bg_mask = (t_f <= float(bg_threshold))
    if not bg_mask.any():
        return (y_f.sum() + t_f.sum()) * 0.0

    bg_preds = torch.clamp_min(y_f[bg_mask], 0.0)
    num_bg = bg_preds.numel()
    k = min(num_bg, max(1, int(float(ratio) * num_bg)))

    topk_vals, _ = torch.topk(bg_preds, k=k, largest=True, sorted=False)
    return torch.mean(topk_vals.square())


def mass_weighted_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Mass-weighted cell allocation loss (RMR-v8 Stage 2 / RMR-v11)."""
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)
    if y.ndim == 2:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)

    work_dtype = y.dtype if y.dtype in (torch.float32, torch.float64) else torch.float32
    y_f = y.to(dtype=work_dtype)
    t_f = target.to(dtype=work_dtype)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    total_mass = t_f.sum(dim=(-2, -1), keepdim=True)
    p = torch.where(total_mass > float(eps), t_f / total_mass.clamp_min(float(eps)), torch.zeros_like(t_f))

    if abs(float(gamma) - 1.0) > 1e-5:
        p_pow = p.pow(float(gamma))
        sum_p_pow = p_pow.sum(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
        p = p_pow / sum_p_pow

    hw = float(t_f.shape[-2] * t_f.shape[-1])
    raw_weight = 1.0 + float(alpha) * hw * p

    norm_factor = raw_weight.mean(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    weights = raw_weight / norm_factor

    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")
    return (weights * per_pixel).mean()


def physical_scale_alignment_loss(
    scale_weights: torch.Tensor,
    target_y: torch.Tensor,
    tau_dense: float = 0.12,
    tau_sparse: float = 0.03,
    kernel_size: int = 5,
    eps: float = 1e-6,
    mask_background: bool = True,
) -> torch.Tensor:
    """Physical Scale Alignment Loss (RMR-v13 / RMR-v14)."""
    if target_y.ndim == 2:
        target_y = target_y.unsqueeze(0).unsqueeze(0)
    elif target_y.ndim == 3:
        target_y = target_y.unsqueeze(1)

    b, k, h, w = scale_weights.shape
    if k < 2 or scale_weights.numel() == 0 or target_y.numel() == 0:
        return (scale_weights.float().sum() + target_y.float().sum()) * 0.0

    work_dtype = scale_weights.dtype if scale_weights.dtype in (torch.float32, torch.float64) else torch.float32
    t_f = target_y.to(dtype=work_dtype).clamp_min(0.0)
    if scale_weights.shape[-2:] != t_f.shape[-2:]:
        scale_weights = F.interpolate(
            scale_weights, size=t_f.shape[-2:], mode="bilinear", align_corners=False
        )

    pad = int(kernel_size) // 2
    local_density = F.avg_pool2d(
        t_f, kernel_size=int(kernel_size), stride=1, padding=pad, count_include_pad=False
    )

    delta_tau = max(float(tau_dense) - float(tau_sparse), 1e-6)
    s = torch.clamp((local_density - float(tau_sparse)) / delta_tau, 0.0, 1.0)

    u = s * float(k - 1)
    target_pi_list = []
    for scale_idx in range(k):
        center = float(k - 1 - scale_idx)
        weight_k = torch.clamp(1.0 - torch.abs(u - center), min=0.0)
        target_pi_list.append(weight_k)
    target_pi = torch.cat(target_pi_list, dim=1).to(dtype=work_dtype)

    pred_pi = scale_weights.to(dtype=work_dtype).clamp(min=float(eps), max=1.0)
    target_log_target = torch.where(
        target_pi > 1e-6,
        target_pi * torch.log(target_pi.clamp_min(1e-6)),
        torch.zeros_like(target_pi),
    )
    target_log_pred = target_pi * torch.log(pred_pi)
    kl_per_pixel = (target_log_target - target_log_pred).sum(dim=1)

    if mask_background:
        fg_mask = (local_density >= float(tau_sparse)).float().squeeze(1)
        fg_sum = fg_mask.sum()
        if fg_sum > 0:
            return (kl_per_pixel * fg_mask).sum() / fg_sum
        return (scale_weights * 0.0).sum()

    return kl_per_pixel.mean()
