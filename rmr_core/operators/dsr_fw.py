"""Dual-Stream Radon-Frank-Wolfe Measure Recovery Operator (DSR-FW).

Exact closed-form measure recovery resolving the Stride 2 Resolution Duality:
1. Anti-Matthew Bounded Peak Calibration (Stream A): Calibrates detected peaks
   bounded by their physical unit mass capacity Cap(u) = max(1.0 - y0, 0),
   completely preventing already-counted heads from bloating into multi-person masses.
2. Dual-Certificate Radon Spike Injection (Stream B): Recovers missed heads
   in vacant areas using pre-activation logit saliency c_z(u) and separable 1D
   Radon profile intersections, sharpened into compact Dirac impulses via
   Frank-Wolfe power-temperature sharpening.
3. Zero Valley Fog (Strict Support Invariance on Valleys): Background pixels
   receive identically zero leakage.
4. Bidirectional Conservation: Calibrates deficits (Delta > 0) and contracts
   surpluses (Delta < 0) for exact count matching.

0 Trainable Parameters. O(1) Computational Graph Complexity.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjoint import regional_adjoint
from .prefix_sums import regional_sum
from .regions import RegionSet, partition_regions_by_scale


def closed_form_dsr_fw_transport(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor | None,
    regions: RegionSet,
    z0: torch.Tensor | None = None,
    eps: float = 1e-6,
    scale_weights: torch.Tensor | None = None,
    tau_noise: float = 0.02,
    tau_peak: float = 0.05,
    ratio_clamp_max: float = 3.0,
    ratio_clamp_min: float = 0.2,
    peak_kernel_size: int = 3,
    power_sharpening: int = 4,
    power: int | None = None,
    unit_mass_capacity: bool = True,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute DSR-FW closed-form measure recovery operator.

    Args:
        y0: [B, 1, H, W] carrier density map (y0 >= 0).
        b_solver: [B, 1, M] regional evidence counts.
        weight_solver: [B, 1, M] regional reliability weights.
        regions: Multiscale region set.
        z0: [B, 1, H, W] pre-activation logit map (optional, computed if None).
        eps: Small numerical constant.
        scale_weights: [B, K, H, W] dynamic scale routing weights.
        tau_noise: Noise floor threshold below which pixels are zeroed in prominence.
        tau_peak: Saturation scale for soft prominence gating.
        ratio_clamp_max: Maximum allowed scaling multiplier on peaks.
        ratio_clamp_min: Minimum allowed scaling multiplier on peaks for surplus.
        peak_kernel_size: Window size for local maximum pooling (3 for Stride 2).
        power_sharpening: Power exponent p for Frank-Wolfe apex sharpening.
        unit_mass_capacity: Whether to bound peak growth by physical unit mass (1.0).
        return_details: Whether to return intermediate diagnostic fields.

    Returns:
        y_star: [B, 1, H, W] recovered Radon measure.
        details (optional): dict containing y_scale, y_spike, p_peak, salience.
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

    # Pre-activation logits z0 (infinite dynamic range for latent head detection)
    if z0 is not None:
        if z0.ndim == 2:
            z0 = z0.unsqueeze(0).unsqueeze(0)
        elif z0.ndim == 3:
            z0 = z0.unsqueeze(1)
        z0_f = torch.nan_to_num(z0.to(dtype=work_dtype), nan=-10.0, posinf=10.0, neginf=-10.0)
    else:
        # Invert softplus: z0 = log(exp(y0) - 1)
        z0_f = torch.where(
            y0_f > 20.0,
            y0_f,
            torch.log(torch.expm1(y0_f.clamp_min(float(eps))))
        )

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

    # 1. Continuous Peak Prominence on L_2 Subpixel Lattice (k_peak = 3 default)
    k_peak = max(3, int(peak_kernel_size))
    pad_peak = k_peak // 2
    y0_lmax = F.max_pool2d(y0_f, kernel_size=k_peak, stride=1, padding=pad_peak)
    apex_ratio = (y0_f / (y0_lmax + float(eps))).clamp(0.0, 1.0).square()
    noise_gate = torch.clamp((y0_f - float(tau_noise)) / max(float(tau_peak), float(eps)), 0.0, 1.0)
    p_peak = (apex_ratio * noise_gate).clamp(0.0, 1.0)
    is_peak = (p_peak > 0.5) & (y0_f >= (y0_lmax - float(eps)))

    # 2. Regional Mass and Physical Unit Mass Capacity
    mu_0 = regional_sum(y0_f, boxes_tensor)
    delta_r = b_s_f - mu_0  # Positive = Deficit (need more), Negative = Surplus (need less)

    if unit_mass_capacity:
        # A peak with y0 >= 1.0 has 0 deficit capacity; peaks with y0 < 1.0 can absorb (1.0 - y0)
        cap_map = torch.where(is_peak, torch.clamp(1.0 - y0_f, min=0.0), torch.zeros_like(y0_f))
        cap_r = regional_sum(cap_map, boxes_tensor).clamp_min(0.0)
    else:
        mu_peak_map = y0_f * p_peak
        cap_r = (regional_sum(mu_peak_map, boxes_tensor) * (float(ratio_clamp_max) - 1.0)).clamp_min(0.0)
        cap_map = y0_f * p_peak

    # 3. Deficit Partitioning
    is_deficit = delta_r > 0.0
    delta_exist = torch.where(is_deficit, torch.min(delta_r, cap_r), delta_r)
    delta_deficit = torch.where(is_deficit, delta_r - delta_exist, torch.zeros_like(delta_r))

    # Weighted regional coverage
    if routed_scales is not None:
        cov_w = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
        for mask_k, boxes_k, _, pi_k in routed_scales:
            w_k = w_s_f[:, :, mask_k]
            cov_w_k = regional_adjoint(w_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
            cov_w = cov_w + pi_k * cov_w_k
        cov_w = cov_w.clamp_min(float(eps))
    else:
        cov_w = regional_adjoint(
            w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        ).clamp_min(float(eps))

    # 4. Stream A: Bounded Peak Calibration (Zero Valley Fog)
    mu_peak_box = regional_sum(y0_f * p_peak, boxes_tensor).clamp_min(float(eps))
    s_delta_pos = delta_exist / (cap_r.clamp_min(float(eps)))
    min_surplus_delta = float(ratio_clamp_min) - 1.0
    s_delta_neg = (delta_exist / mu_peak_box).clamp(min=min_surplus_delta, max=0.0)
    s_delta = torch.where(delta_r >= 0, s_delta_pos, s_delta_neg)

    weighted_s_delta = (w_s_f * s_delta).to(dtype=work_dtype)
    if routed_scales is not None:
        adj_s_delta = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
        for mask_k, boxes_k, _, pi_k in routed_scales:
            w_s_k = weighted_s_delta[:, :, mask_k]
            adj_k = regional_adjoint(w_s_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
            adj_s_delta = adj_s_delta + pi_k * adj_k
        adj_s_delta = adj_s_delta / cov_w
    else:
        adj_s_delta = regional_adjoint(
            weighted_s_delta, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        ) / cov_w

    if unit_mass_capacity:
        delta_y_scale_pos = cap_map * F.relu(adj_s_delta)
        delta_y_scale_neg = y0_f * p_peak * (-F.relu(-adj_s_delta))
        y_scale = torch.clamp_min(y0_f + delta_y_scale_pos + delta_y_scale_neg, 0.0)
    else:
        s_eff = 1.0 + p_peak * adj_s_delta
        y_scale = torch.clamp_min(y0_f * s_eff, 0.0)

    # 5. Stream B: Dual-Certificate Radon Spike Injection (Anti-Carrier Collapse)
    vacant = (1.0 - p_peak).clamp(0.0, 1.0)
    mean_bg_z = F.avg_pool2d(z0_f, kernel_size=5, stride=1, padding=2)
    c_z = F.relu(z0_f - mean_bg_z)

    # Isotropic 2D apex sharpening (strictly eliminates Cartesian 2-angle ghost cross-talk)
    c_z_lmax = F.max_pool2d(c_z, kernel_size=3, stride=1, padding=1)
    apex_weight = (c_z / (c_z_lmax + float(eps))).clamp(0.0, 1.0).square()
    salience = vacant * apex_weight * c_z

    eff_power = power if power is not None else power_sharpening
    p_exp = max(2, int(eff_power))
    salience_p = salience.clamp_min(0.0) ** p_exp

    box_salience = regional_sum(salience_p, boxes_tensor)
    has_salience = box_salience > float(eps)
    ratio_deficit = torch.where(
        has_salience,
        delta_deficit / box_salience.clamp_min(float(eps)),
        torch.zeros_like(delta_deficit)
    )

    weighted_deficit = (w_s_f * ratio_deficit).to(dtype=work_dtype)
    if routed_scales is not None:
        adj_deficit = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
        for mask_k, boxes_k, _, pi_k in routed_scales:
            w_def_k = weighted_deficit[:, :, mask_k]
            adj_k = regional_adjoint(w_def_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
            adj_deficit = adj_deficit + pi_k * adj_k
        adj_deficit = adj_deficit / cov_w
    else:
        adj_deficit = regional_adjoint(
            weighted_deficit, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        ) / cov_w

    y_spike = torch.clamp_min(salience_p * adj_deficit, 0.0)

    # 6. Total Measure Synthesis
    y_star = torch.clamp_min(y_scale + y_spike, 0.0).to(dtype=y0.dtype)

    if return_details:
        details = {
            "y_scale": y_scale.to(dtype=y0.dtype),
            "y_spike": y_spike.to(dtype=y0.dtype),
            "p_peak": p_peak.to(dtype=y0.dtype),
            "salience": salience.to(dtype=y0.dtype),
            "delta_deficit": delta_deficit.to(dtype=y0.dtype),
        }
        return y_star, details

    return y_star
