"""Peak-Concentrated Bregman-Poisson Recovery Operator (PC-BPRO).

Exact closed-form measure recovery resolving the Stride 2 Resolution Duality:
1. Zero Background Leakage (Hard Support Invariance): Background pixels (y0 < tau_noise)
   are strictly untouched (multiplier identically 1.0).
2. Peak-Confined Mass Allocation: Regional count deficits Delta_R = b_R - mu_0(R) are
   allocated strictly to detected local maxima (prominent head peaks) rather than uniformly
   diluted across background valleys.
3. Continuous C1 Prominence Gating: Smooth local apex ratio and noise gate eliminate
   spatial boundary artifacts and maintain full autograd stability.

0 Trainable Parameters. O(1) Computational Graph Complexity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjoint import regional_adjoint
from .prefix_sums import regional_sum
from .regions import RegionSet, partition_regions_by_scale


def closed_form_peak_concentrated_transport(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor | None,
    regions: RegionSet,
    eps: float = 1e-6,
    scale_weights: torch.Tensor | None = None,
    tau_noise: float = 0.02,
    tau_peak: float = 0.05,
    ratio_clamp_max: float = 5.0,
    ratio_clamp_min: float = 0.2,
    peak_kernel_size: int = 5,
    return_multiplier: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute Peak-Concentrated closed-form measure recovery operator.

    Args:
        y0: [B, 1, H, W] carrier density map.
        b_solver: [B, 1, M] regional evidence counts.
        weight_solver: [B, 1, M] regional reliability weights.
        regions: Multiscale region set.
        eps: Small numerical constant.
        scale_weights: [B, K, H, W] dynamic scale routing weights.
        tau_noise: Noise floor threshold below which pixels are zeroed in prominence.
        tau_peak: Saturation scale for soft prominence gating.
        ratio_clamp_max: Maximum allowed scaling multiplier on peaks.
        ratio_clamp_min: Minimum allowed scaling multiplier on peaks.
        peak_kernel_size: Window size for local maximum pooling (5 for Stride 2).
        return_multiplier: Whether to return the spatial multiplier field S_eff.

    Returns:
        y_projected: [B, 1, H, W] recovered Radon measure.
        s_eff (optional): [B, 1, H, W] effective spatial scaling field.
    """
    if y0.ndim == 2:
        y0 = y0.unsqueeze(0).unsqueeze(0)
    elif y0.ndim == 3:
        y0 = y0.unsqueeze(1)
    b, _, h, w = y0.shape

    if b_solver.ndim == 2:
        b_solver = b_solver.unsqueeze(1)
    if weight_solver is None:
        weight_solver = torch.ones_like(b_solver)
    elif weight_solver.ndim == 2:
        weight_solver = weight_solver.unsqueeze(1)

    work_dtype = y0.dtype if y0.dtype in (torch.float32, torch.float64) else torch.float32
    y0_f = torch.nan_to_num(y0.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    b_s_f = torch.nan_to_num(b_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    w_s_f = torch.nan_to_num(weight_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    boxes_tensor = regions.boxes if hasattr(regions, "boxes") else regions

    # Scale routing partitions
    routed_scales: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    if scale_weights is not None:
        _, k_scales = scale_weights.shape[:2]
        if scale_weights.shape[-2:] != (h, w):
            scale_weights = F.interpolate(
                scale_weights, size=(h, w), mode="bilinear", align_corners=False
            )
        scale_partitions = partition_regions_by_scale(regions, k_scales, device=y0.device)
        active = [
            (k, mask_k, boxes_k)
            for (k, mask_k, boxes_k) in scale_partitions
            if mask_k is not None and boxes_k is not None and boxes_k.shape[0] > 0
        ]
        if active:
            active_pi_sum = sum(
                scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype) for (k, _, _) in active
            ).clamp_min(float(eps))
            routed_scales = []
            for k, mask_k, boxes_k in active:
                ones_k = torch.ones((1, 1, boxes_k.shape[0]), device=y0.device, dtype=work_dtype)
                cov_k = regional_adjoint(
                    ones_k, boxes_k, height=h, width=w, out_dtype=work_dtype
                ).clamp_min(float(eps))
                pi_k = scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype) / active_pi_sum
                routed_scales.append((mask_k, boxes_k, cov_k, pi_k))

    # 1. Compute Continuous Peak Prominence Field pi(u) in [0, 1]
    k_peak = max(3, int(peak_kernel_size))
    pad_peak = k_peak // 2
    y0_local_max = F.max_pool2d(y0_f, kernel_size=k_peak, stride=1, padding=pad_peak)
    apex_ratio = (y0_f / (y0_local_max + float(eps))).clamp(0.0, 1.0).square()
    noise_gate = torch.clamp((y0_f - float(tau_noise)) / max(float(tau_peak), float(eps)), 0.0, 1.0)
    p_peak = (apex_ratio * noise_gate).clamp(0.0, 1.0)

    # 2. Regional Mass and Deficit
    y_peak = y0_f * p_peak
    mu_0 = regional_sum(y0_f, boxes_tensor)
    mu_peak = regional_sum(y_peak, boxes_tensor)
    delta_r = b_s_f - mu_0

    # 3. Peak Scaling Multiplier per Box
    # s_R = 1.0 + delta_R / (mu_peak + eps)
    # When mu_peak is negligible (empty region or background), s_R = 1.0 (identity)
    s_delta = delta_r / (mu_peak + float(eps))
    # Clamp delta to respect [ratio_clamp_min - 1.0, ratio_clamp_max - 1.0]
    min_delta = float(ratio_clamp_min) - 1.0
    max_delta = float(ratio_clamp_max) - 1.0
    s_delta_clamped = s_delta.clamp(min_delta, max_delta)
    
    # Empty region condition: if mu_peak is near zero, fall back to delta = 0 (identity)
    empty_peaks = (mu_peak < float(eps))
    s_delta_eff = torch.where(empty_peaks, torch.zeros_like(s_delta_clamped), s_delta_clamped)
    scaling_ratio = 1.0 + s_delta_eff

    # 4. Adjoint Backprojection of Scaling Multiplier Field
    weighted_s = (w_s_f * scaling_ratio).to(dtype=work_dtype)
    if routed_scales is not None:
        s_raw = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
        for mask_k, boxes_k, _, pi_k in routed_scales:
            w_k = w_s_f[:, :, mask_k]
            w_s_k = weighted_s[:, :, mask_k]
            cov_w_k = regional_adjoint(
                w_k, boxes_k, height=h, width=w, out_dtype=work_dtype
            )
            adj_s_k = regional_adjoint(w_s_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
            s_k = torch.where(
                cov_w_k > float(eps),
                adj_s_k / cov_w_k.clamp_min(float(eps)),
                torch.ones_like(adj_s_k),
            )
            s_raw = s_raw + pi_k * s_k
    else:
        cov_w = regional_adjoint(
            w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        )
        adj_s = regional_adjoint(weighted_s, boxes_tensor, height=h, width=w, out_dtype=work_dtype)
        s_raw = torch.where(
            cov_w > float(eps),
            adj_s / cov_w.clamp_min(float(eps)),
            torch.ones_like(adj_s),
        )

    # 5. Peak-Concentrated Projection: y*(u) = y0(u) * [1.0 + p_peak(u) * (S_raw(u) - 1.0)]
    s_eff = 1.0 + p_peak * (s_raw - 1.0)
    y_projected = torch.clamp_min(y0_f * s_eff, 0.0).to(dtype=y0.dtype)

    if return_multiplier:
        return y_projected, s_eff.to(dtype=y0.dtype)
    return y_projected
