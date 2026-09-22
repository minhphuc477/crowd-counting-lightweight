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
    """Average proper NB NLL within scale, then average scales.

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
    )

    scale_losses = []
    scale_weights = []
    num_scales = int(getattr(regions, "num_scales", 3))
    # Support full-image scale -1 if present, and standard scales [0, num_scales-1]
    for sid in range(-1, num_scales):
        mask = (regions.scale_id == sid)
        count = mask.float().sum()
        has_scale = (count > 0).float()
        if mass_weight_alpha > 0.0 and count > 0:
            t_s = target_region[..., mask].float()
            t_mean = t_s.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            w_raw = 1.0 + float(mass_weight_alpha) * (t_s / t_mean)
            w_norm = w_raw / w_raw.mean(dim=-1, keepdim=True).clamp_min(1e-4)
            loss_s = (w_norm * per_region[..., mask]).mean()
        else:
            loss_s = (per_region[..., mask].sum(dim=-1) / count.clamp_min(1.0)).mean()
        scale_losses.append(loss_s * has_scale)
        scale_weights.append(has_scale)

    total_weight = torch.stack(scale_weights).sum()
    total_loss = torch.stack(scale_losses).sum()
    return torch.where(total_weight > 0, total_loss / total_weight.clamp_min(1.0), (per_region.sum()) * 0.0)



def curvature_power_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    eps: float = 0.01,
    threshold: float = 0.0,
    kernel_size: int = 5,
    mode: str = "none",
    smooth_scale: float = 0.02,
    stride: int = 4,
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

    # NOTE: y_f and t_f are in people/cell (Dirac discrete masses, stride-invariant).
    # Do NOT divide by area_scale here: the Anscombe VST must be evaluated in the
    # native counting unit (people/cell), otherwise the operating point shifts by
    # 1/sqrt(area_scale) per cell, inflating the loss by up to 4.4x at stride 2.
    diff = torch.sqrt(y_f + float(eps)) - torch.sqrt(t_f + float(eps))
    diff_sq = diff.square()

    if mode == "none" or float(threshold) <= 0.0:
        return torch.mean(diff_sq)

    # area_scale is only needed for the gating path:
    # threshold is expressed in 'people per Stride-4 cell' units, so we scale
    # it to 'people per current-stride cell' using area_scale = (stride/4)^2.
    area_scale = (float(stride) / 4.0) ** 2
    eff_threshold = float(threshold) * area_scale
    eff_smooth_scale = float(smooth_scale) * area_scale
    eff_kernel = int(round(kernel_size * (4.0 / float(stride))))
    if eff_kernel % 2 == 0:
        eff_kernel += 1
    pad = eff_kernel // 2
    local_density = F.avg_pool2d(
        t_f, kernel_size=eff_kernel, stride=1, padding=pad, count_include_pad=False
    )

    if mode == "hard":
        gate = (local_density >= eff_threshold).float()
    elif mode == "soft":
        gate = torch.sigmoid((local_density - eff_threshold) / eff_smooth_scale)
    else:
        raise ValueError(f"Unknown curvature gate mode: '{mode}'. Expected 'none', 'hard', or 'soft'.")

    gate_sum = gate.sum()
    return (gate * diff_sq).sum() / gate_sum.clamp_min(1.0)


def topk_hard_background_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.05,
    bg_threshold: float = 1e-5,
    stride: int = 4,
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

    b_size = y_f.shape[0]
    sample_losses = []
    for b in range(b_size):
        y_b = y_f[b : b + 1]
        t_b = t_f[b : b + 1]
        bg_mask = (t_b <= float(bg_threshold))
        if not bg_mask.any():
            sample_losses.append((y_b.sum() + t_b.sum()) * 0.0)
            continue

        bg_preds = torch.clamp_min(y_b[bg_mask], 0.0)
        num_bg = bg_preds.numel()
        k = min(num_bg, max(1, int(float(ratio) * num_bg)))
        topk_vals, _ = torch.topk(bg_preds, k=k, largest=True, sorted=False)
        sample_losses.append(torch.mean(topk_vals.square()))

    return torch.stack(sample_losses).mean()


def mass_weighted_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 1.0,
    gamma: float = 1.0,
    stride: int = 4,
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


def count_invariant_cell_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    alpha: float = 2.0,
    tau_head: float = 0.08,
    eps: float = 1e-4,
    stride: int = 4,
) -> torch.Tensor:
    """Count-Invariant Two-Stream Cell Loss v2 (CI-Cell v2).

    Corrected formulation for integer-domain density maps where each cell
    stores the number of people (0, 1, 2, 3, 4...), NOT a Gaussian amplitude.

    Weighting strategy:
        fg_fraction(x) = t(x) / max(per_image_max, 1)  in [0, 1]
        W(x) = 1 + (alpha - 1) * fg_fraction(x)

    This guarantees O(1) gradient per head because:
    - Weight is normalized by local peak count, not global sum N
    - A cell with 2 heads in a 10-person image gets the same relative weight
      as a cell with 2 heads in a 1000-person image
    - Background cells (t=0) retain W=1.0 for false-alarm suppression

    Args:
        y: Predicted density map, shape (B, 1, H, W) or (B, H, W) or (H, W).
        target: Ground-truth density map (integer counts per cell).
        beta: Smooth-L1 transition point.
        alpha: Foreground boost factor (W_fg = alpha at peak cell).
        tau_head: Deprecated — ignored in v2. Kept for backward compat.
        eps: Minimum denominator guard.
        stride: Spatial stride (informational only in v2).
    """
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

    # Normalize by per-image peak to get fg_fraction in [0, 1]
    # Works correctly for both integer counts and float density maps
    t_peak = t_f.amax(dim=(-2, -1), keepdim=True).clamp_min(float(eps))
    fg_fraction = (t_f / t_peak).clamp(0.0, 1.0)

    # Foreground-boosted weight: alpha at peak cell, 1.0 at background
    weight = 1.0 + (float(alpha) - 1.0) * fg_fraction

    per_pixel = F.smooth_l1_loss(y_f, t_f, beta=float(beta), reduction="none")
    return (weight * per_pixel).mean()



def physical_scale_alignment_loss(
    scale_weights: torch.Tensor,
    target_y: torch.Tensor,
    tau_dense: float = 0.12,
    tau_sparse: float = 0.03,
    kernel_size: int = 5,
    eps: float = 1e-6,
    mask_background: bool = True,
    stride: int = 4,
) -> torch.Tensor:
    """Physical Scale Alignment Loss (RMR-v13 / RMR-v14)."""
    if scale_weights.ndim == 3:
        scale_weights = scale_weights.unsqueeze(0)
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

    area_scale = (float(stride) / 4.0) ** 2
    eff_tau_dense = float(tau_dense) * area_scale
    eff_tau_sparse = float(tau_sparse) * area_scale
    eff_kernel = int(round(kernel_size * (4.0 / float(stride))))
    if eff_kernel % 2 == 0:
        eff_kernel += 1
    pad = eff_kernel // 2
    local_density = F.avg_pool2d(
        t_f, kernel_size=eff_kernel, stride=1, padding=pad, count_include_pad=False
    )

    delta_tau = max(float(eff_tau_dense) - float(eff_tau_sparse), 1e-6)
    s = torch.clamp((local_density - float(eff_tau_sparse)) / delta_tau, 0.0, 1.0)

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
        fg_mask = (local_density >= float(eff_tau_sparse)).float().squeeze(1)
        fg_sum = fg_mask.sum()
        return (kl_per_pixel * fg_mask).sum() / fg_sum.clamp_min(1.0)

    return kl_per_pixel.mean()
