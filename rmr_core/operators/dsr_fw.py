"""Dual-Stream Radon-Frank-Wolfe Measure Recovery Operator (DSR-FW).

Exact closed-form measure recovery resolving the Stride 2 Resolution Duality:
1. Anti-Matthew Bounded Peak Calibration (Stream A): Calibrates detected peaks
   bounded by their physical footprint capacity Cap(u) = max(1.0 - M_footprint(u), 0),
   preventing broad-shouldered detected heads from bloating into multi-person masses.
2. Hierarchical Fog-First Surplus Contraction (HFSC): Partitions regional surplus
   into low-prominence carrier fog and confirmed peaks, aggressively eliminating
   background clutter/pedestal while preserving confirmed heads.
3. Dual-Certificate Radon Spike Injection (Stream B): Recovers missed heads
   in vacant areas using pre-activation logit saliency c_z(u) and 2D isotropic apex
   sharpening via Frank-Wolfe power sharpening, bounded by unit mass ceiling.
4. Scale-Partitioned Adjoint Normalization: Each scale partition is normalized by
   its own coverage before convex combination, eliminating grid-density dilution.
5. Morozov Discrepancy Regularization: Soft-thresholds regional residuals against
   measurement noise floor, suppressing high-frequency noise amplification.

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
    regions: RegionSet | torch.Tensor,
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
    tau_z_floor: float = -3.5,
    max_spike_mass: float = 1.5,
    unit_mass_capacity: bool = False,
    morozov_tol: float = 0.0,
    morozov_gamma: float = 0.0,
    b_variance: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Execute DSR-FW closed-form measure recovery operator.

    Args:
        y0: [B, 1, H, W] carrier density map (y0 >= 0).
        b_solver: [B, 1, M] regional evidence counts.
        weight_solver: [B, 1, M] regional reliability weights.
        regions: Multiscale region set or box tensor.
        z0: [B, 1, H, W] pre-activation logit map (optional, computed if None).
        eps: Small numerical constant.
        scale_weights: [B, K, H, W] dynamic scale routing weights.
        tau_noise: Noise floor threshold below which pixels are zeroed in prominence.
        tau_peak: Saturation scale for soft prominence gating.
        ratio_clamp_max: Maximum allowed scaling multiplier on peaks.
        ratio_clamp_min: Minimum allowed scaling multiplier on peaks for surplus.
        peak_kernel_size: Window size for local maximum pooling (3 for Stride 2).
        power_sharpening: Power exponent p for Frank-Wolfe apex sharpening.
        power: Alias for power_sharpening.
        tau_z_floor: Pre-activation logit threshold floor for noise gating.
        max_spike_mass: Maximum mass injectable into a single Dirac spike.
        unit_mass_capacity: Whether to bound peak growth by physical unit mass (1.0).
        morozov_tol: Morozov discrepancy tolerance for noise suppression.
        morozov_gamma: Morozov noise multiplier for variance-based shrinkage.
        b_variance: [B, 1, M] regional count variance.
        return_details: Whether to return intermediate diagnostic fields.

    Returns:
        y_star: [B, 1, H, W] recovered Radon measure.
        details (optional): dict containing diagnostic fields.
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

    # Scale routing partitions and per-scale coverage
    routed_scales: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
    if isinstance(regions, RegionSet) and getattr(regions, "num_scales", 1) > 1:
        k_scales = regions.num_scales
        if scale_weights is not None:
            if scale_weights.shape[-2:] != (h, w):
                scale_weights = F.interpolate(
                    scale_weights, size=(h, w), mode="bilinear", align_corners=False
                )
            pi_maps = scale_weights.to(dtype=work_dtype)
        else:
            pi_maps = torch.full((b, k_scales, h, w), 1.0 / k_scales, device=y0.device, dtype=work_dtype)

        scale_partitions = partition_regions_by_scale(regions, k_scales, device=y0.device)
        active = [
            (k, mask_k, boxes_k)
            for (k, mask_k, boxes_k) in scale_partitions
            if mask_k is not None and boxes_k is not None and boxes_k.shape[0] > 0
        ]
        if active:
            active_pi_sum = sum(pi_maps[:, k : k + 1, :, :] for (k, _, _) in active).clamp_min(float(eps))
            routed_scales = []
            for k, mask_k, boxes_k in active:
                w_k = w_s_f[:, :, mask_k].to(dtype=work_dtype)
                cov_w_k = regional_adjoint(
                    w_k, boxes_k, height=h, width=w, out_dtype=work_dtype
                ).clamp_min(float(eps))
                pi_k = pi_maps[:, k : k + 1, :, :] / active_pi_sum
                routed_scales.append((mask_k, boxes_k, cov_w_k, pi_k))

    def project_adjoint(field_per_box: torch.Tensor) -> torch.Tensor:
        if routed_scales is not None:
            out_proj = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
            for mask_k, boxes_k, cov_w_k, pi_k in routed_scales:
                w_box_k = (w_s_f[:, :, mask_k] * field_per_box[:, :, mask_k]).to(dtype=work_dtype)
                adj_k = regional_adjoint(w_box_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
                out_proj = out_proj + pi_k * (adj_k / cov_w_k)
            return out_proj
        cov_single = regional_adjoint(
            w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        ).clamp_min(float(eps))
        return regional_adjoint(
            (w_s_f * field_per_box).to(dtype=work_dtype),
            boxes_tensor,
            height=h,
            width=w,
            out_dtype=work_dtype
        ) / cov_single

    # 1. Local Peak Prominence & Footprint-Integrated Capacity
    k_peak = max(3, int(peak_kernel_size))
    pad_peak = k_peak // 2
    y0_padded = F.pad(y0_f, (pad_peak, pad_peak, pad_peak, pad_peak), mode="replicate")
    y0_lmax = F.max_pool2d(y0_padded, kernel_size=k_peak, stride=1)
    y0_avg = F.avg_pool2d(y0_padded, kernel_size=k_peak, stride=1)
    # Footprint mass: integrated mass across the local neighborhood of the head
    y0_footprint = (k_peak * k_peak) * y0_avg

    prominence = F.relu(y0_f - y0_avg)
    prominence_gate = torch.clamp(prominence / max(float(tau_noise), float(eps)), 0.0, 1.0)
    apex_ratio = (y0_f / (y0_lmax + float(eps))).clamp(0.0, 1.0).square()
    noise_gate = torch.clamp((y0_f - float(tau_noise)) / max(float(tau_peak), float(eps)), 0.0, 1.0)
    p_peak = (apex_ratio * noise_gate * prominence_gate).clamp(0.0, 1.0)
    is_peak = (p_peak > 0.5) & (y0_f >= (y0_lmax - float(eps)))

    w_peak = p_peak
    w_fog = (1.0 - w_peak).clamp(0.0, 1.0)

    # 2. Regional Mass and Morozov Discrepancy Shrinkage
    mu_0 = regional_sum(y0_f, boxes_tensor)
    delta_r = b_s_f - mu_0

    if morozov_tol > 0.0 or (morozov_gamma > 0.0 and b_variance is not None):
        tau_morozov = float(morozov_tol)
        if b_variance is not None and morozov_gamma > 0.0:
            var_r = b_variance.to(dtype=work_dtype).clamp_min(0.0)
            if var_r.ndim == 2:
                var_r = var_r.unsqueeze(1)
            tau_morozov = tau_morozov + float(morozov_gamma) * torch.sqrt(var_r)
        delta_r = torch.sign(delta_r) * F.relu(delta_r.abs() - tau_morozov)

    # 3. Deficit Partitioning (Stream A vs Stream B)
    if unit_mass_capacity:
        # Footprint-integrated capacity: peak absorbs deficit only if footprint mass < 1.0
        cap_map = torch.where(is_peak, torch.clamp(1.0 - y0_footprint, min=0.0), torch.zeros_like(y0_f))
        cap_r = regional_sum(cap_map, boxes_tensor).clamp_min(0.0)
    else:
        # Proportional apex scaling with exact mathematical adjoint parity: cap_map matches cap_r
        cap_map = y0_f * p_peak * (float(ratio_clamp_max) - 1.0)
        cap_r = regional_sum(cap_map, boxes_tensor).clamp_min(0.0)

    is_deficit = delta_r > 0.0
    delta_deficit_exist = torch.where(is_deficit, torch.min(delta_r, cap_r), torch.zeros_like(delta_r))
    delta_deficit_latent = torch.where(is_deficit, delta_r - delta_deficit_exist, torch.zeros_like(delta_r))

    # 4. Stream A: Bounded Peak Calibration (Deficit) & Hierarchical Fog-First Contraction (Surplus)
    # Deficit scaling ratio on existing peaks
    s_delta_pos = delta_deficit_exist / cap_r.clamp_min(float(eps))

    # Hierarchical Fog-First Surplus Contraction
    surplus_r = F.relu(-delta_r)
    mu_fog_r = regional_sum(y0_f * w_fog, boxes_tensor)
    mu_peak_r = regional_sum(y0_f * w_peak, boxes_tensor)

    # Fog absorbs surplus first up to 100% of its volume
    m_fog_rem = torch.min(surplus_r, mu_fog_r)
    r_fog = m_fog_rem / mu_fog_r.clamp_min(float(eps))

    # Remaining surplus is removed from peaks bounded by ratio_clamp_min
    surplus_rem_peak = surplus_r - m_fog_rem
    max_peak_contraction = 1.0 - float(ratio_clamp_min)
    r_peak = torch.clamp(
        surplus_rem_peak / mu_peak_r.clamp_min(float(eps)),
        0.0,
        max(0.0, max_peak_contraction),
    )

    # Project scaling factors to pixel grid via scale-partitioned adjoint
    adj_s_pos = project_adjoint(s_delta_pos)
    adj_r_fog = project_adjoint(r_fog)
    adj_r_peak = project_adjoint(r_peak)

    # Apply Stream A updates (Zero Valley Fog on Deficit, Fog-First on Surplus)
    delta_y_scale_pos = cap_map * F.relu(adj_s_pos)
    delta_y_scale_neg = -(adj_r_fog * w_fog + adj_r_peak * w_peak) * y0_f
    y_scale = torch.clamp_min(y0_f + delta_y_scale_pos + delta_y_scale_neg, 0.0)

    # 5. Stream B: Dual-Certificate Radon Spike Injection (Latent Peaks)
    # Vacancy gate respects both peak prominence and local footprint mass
    vacant = (1.0 - p_peak).clamp(0.0, 1.0)
    if unit_mass_capacity:
        vacant = vacant * torch.clamp(1.0 - y0_footprint, min=0.0)

    z0_padded = F.pad(z0_f, (2, 2, 2, 2), mode="replicate")
    mean_bg_z = F.avg_pool2d(z0_padded, kernel_size=5, stride=1)
    c_z = F.relu(z0_f - mean_bg_z)

    # Absolute logit confidence floor
    conf_gate = torch.clamp((z0_f - float(tau_z_floor)) / 1.0, 0.0, 1.0)
    c_z_gated = c_z * conf_gate

    # Isotropic 2D apex sharpening
    c_z_padded = F.pad(c_z_gated, (1, 1, 1, 1), mode="replicate")
    c_z_lmax = F.max_pool2d(c_z_padded, kernel_size=3, stride=1)
    apex_weight = (c_z_gated / (c_z_lmax + float(eps))).clamp(0.0, 1.0).square()
    salience = vacant * apex_weight * c_z_gated

    eff_power = power if power is not None else power_sharpening
    p_exp = max(2, int(eff_power))
    salience_p = salience.clamp_min(0.0) ** p_exp

    box_salience = regional_sum(salience_p, boxes_tensor)
    has_salience = box_salience > float(eps)
    ratio_deficit_latent = torch.where(
        has_salience,
        delta_deficit_latent / box_salience.clamp_min(float(eps)),
        torch.zeros_like(delta_deficit_latent)
    )

    adj_deficit_latent = project_adjoint(ratio_deficit_latent)
    y_spike_raw = salience_p * adj_deficit_latent
    y_spike = torch.clamp(y_spike_raw, min=0.0, max=float(max_spike_mass))

    # 6. Total Measure Synthesis
    y_star = torch.clamp_min(y_scale + y_spike, 0.0).to(dtype=y0.dtype)

    if return_details:
        details = {
            "y_scale": y_scale.to(dtype=y0.dtype),
            "y_spike": y_spike.to(dtype=y0.dtype),
            "p_peak": p_peak.to(dtype=y0.dtype),
            "salience": salience.to(dtype=y0.dtype),
            "delta_r": delta_r.to(dtype=y0.dtype),
        }
        return y_star, details

    return y_star
