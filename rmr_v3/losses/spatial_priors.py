"""Spatial geometric, background suppression, and scale alignment prior losses.

Provides:
1. curvature_power_loss: Curvature-preserving square-root power loss for high-density crowds.
2. topk_hard_background_loss: Top-K hard negative background mining loss.
3. physical_scale_alignment_loss: Cross-entropy alignment between predicted scale routing and local crowd density.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


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
    """Curvature-preserving square-root power loss for high-density crowds (RMR-v11/v12).

    Applies Anscombe variance-stabilizing transform sqrt(y + eps) - sqrt(t + eps)
    in native counting units (people/cell), preventing variance explosion in dense clusters.
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
    y_f = torch.where(y >= 0.0, y, torch.zeros_like(y)).to(dtype=work_dtype)
    t_f = torch.where(target >= 0.0, target, torch.zeros_like(target)).to(dtype=work_dtype)
    if y_f.numel() == 0 or t_f.numel() == 0:
        return (y_f.sum() + t_f.sum()) * 0.0

    diff = torch.sqrt(y_f + float(eps)) - torch.sqrt(t_f + float(eps))
    diff_sq = diff.square()

    if mode == "none" or float(threshold) <= 0.0:
        return torch.mean(diff_sq)

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

    gate_sum_sample = gate.sum(dim=(-2, -1), keepdim=True)
    has_gate = (gate_sum_sample > 0).float()
    sample_loss = (gate * diff_sq).sum(dim=(-2, -1), keepdim=True) / gate_sum_sample.clamp_min(1.0)
    total_valid = has_gate.sum().clamp_min(1.0)
    return (sample_loss * has_gate).sum() / total_valid


def topk_hard_background_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    ratio: float = 0.05,
    bg_threshold: float = 1e-5,
    stride: int = 4,
) -> torch.Tensor:
    """Top-K Hard Negative Background Mining Loss (RMR-v11).

    Penalizes the top-K highest predicted false alarms on true background cells (target <= bg_threshold).
    """
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
        if k == 0:
            sample_losses.append((y_b.sum() + t_b.sum()) * 0.0)
            continue
        topk_vals, _ = torch.topk(bg_preds, k=k, largest=True, sorted=False)
        sample_losses.append(torch.mean(topk_vals.square()))

    return torch.stack(sample_losses).mean()


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
    """Physical Scale Alignment Loss (RMR-v13 / RMR-v14).

    Guides spatial scale routing weights pi(x, y) to match the physical perspective rule:
    High local crowd density -> smaller box scales; low density -> larger box scales.
    """
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
    target_log_pred = torch.where(
        target_pi > 1e-6,
        target_pi * torch.log(pred_pi.clamp_min(max(float(eps), 1e-7))),
        torch.zeros_like(target_pi),
    )
    kl_per_pixel = (target_log_target - target_log_pred).sum(dim=1)

    if mask_background:
        fg_mask = (local_density >= float(eff_tau_sparse)).float().squeeze(1)
        fg_sum_sample = fg_mask.sum(dim=(-2, -1), keepdim=True)
        has_fg = (fg_sum_sample > 0).float()
        sample_loss = (kl_per_pixel * fg_mask).sum(dim=(-2, -1), keepdim=True) / fg_sum_sample.clamp_min(1.0)
        return (sample_loss * has_fg).sum() / has_fg.sum().clamp_min(1.0)

    return kl_per_pixel.mean()


def foreground_gating_bce_loss(
    fg_logit: torch.Tensor,
    target_float: torch.Tensor,
) -> torch.Tensor:
    """Foreground gating binary cross entropy loss against dilated GT binary mask."""
    t_bin = (target_float > 0.0).float()
    if t_bin.ndim == 3:
        t_bin = t_bin.unsqueeze(1)
    t_dilated = F.max_pool2d(t_bin, kernel_size=3, stride=1, padding=1)
    fg_pred = fg_logit.float()
    if fg_pred.shape[-2:] != t_dilated.shape[-2:]:
        fg_pred = F.interpolate(fg_pred, size=t_dilated.shape[-2:], mode="bilinear", align_corners=False)
    return F.binary_cross_entropy_with_logits(fg_pred, t_dilated)

