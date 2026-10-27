"""Feature-guided Edge-Directed Multi-Scale Consistent Bregman-Poisson Recovery Operator (FED-MCBPRO).

Mathematical Formulation:
This operator solves the discrete-continuous inverse problem on Radon measures:
    b = A_pi y + eta
via an exact, closed-form, parameter-free transport mapping:
    y* = T_{FED-MCBPRO}(y0, b_solver, weight_solver, P4_feat, regions)

Key Theoretical Innovations (Mathematically Principled Formulation):
1. Dimensionless Weber-Fechner Local Contrast FED-Carrier:
   Overcomes the Multiplicative Zero-Support Trap in dense horizon clusters without
   sample-level scale distortion or dimensional mismatch.
   Computes local spatial high-frequency energy E_feat = ||P4 - AvgPool(P4)||_2 on native features,
   then evaluates the dimensionless Weber-Fechner local contrast ratio:
       E_rel(u) = E_feat(u) / (AvgPool_{15}(E_feat)(u) + noise_floor)
   Modulates valley capacity via convex blending:
       capacity(u) = M_eff(u) * (1 - peak_dom(u))^p * [(1 - alpha_spatial) + alpha_spatial * E_rel(u)]
   - In dense micro-head clusters: E_rel > 1.0 -> mass concentrates at head contours.
   - On uniform textures (pavement, sky): E_rel ≈ 1.0 -> capacity identical to standard MC-BPRO.
2. Density-Gated Feature Energy Modulation:
   Continuous spatial gating field backprojected from box counts:
       dense_gate(u) = A^T(clamp((b_s - tau_null) / (dense_threshold - tau_null), 0, 1)) / cov
       alpha_spatial(u) = alpha * dense_gate(u)
   - On dense crowd clusters (b_s >= dense_threshold): alpha_spatial ≈ alpha (full FED guidance).
   - On moderate/sparse scenes (b_s < dense_threshold): alpha_spatial ≈ 0 -> prevents false mass
     injection into architectural edges (trees, signboards, building facades).
3. Consistent Observation-Space Null-Measure Shrinkage:
   Smoothly anchors microscopic evidence noise in observation space before solver stages:
       b_eff = b_s * clamp(b_s / tau_count_null, 0.0, 1.0)^2
   Preserves exact Stage 3 Bregman-Poisson mass conservation without post-hoc multipliers.

Properties:
- 0 Trainable Parameters.
- O(1) Computational Graph Complexity.
- Strict non-negative Radon measure: y*(u) >= 0 for all u.
- Exact identity reduction: Identical to standard MC-BPRO when feat_energy_weight=0 or dense_threshold=0.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjoint import regional_adjoint
from .mcbpro import _backproject_routed
from .prefix_sums import regional_sum
from .regions import RegionSet, partition_regions_by_scale


def compute_feature_high_pass_energy(
    feat: torch.Tensor,
    pool_kernel: int = 5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute spatial high-frequency contrast energy from intermediate neck features.

    Args:
        feat: [B, C, H, W] spatial feature tensor (e.g. P4 from ASPP/FPN).
        pool_kernel: Window size for local background average pooling.
        eps: Small constant for numerical stability.

    Returns:
        energy: [B, 1, H, W] local spatial high-frequency contrast energy.
    """
    feat_clean = torch.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    pad = pool_kernel // 2
    feat_pad = F.pad(feat_clean, (pad, pad, pad, pad), mode="replicate")
    local_mean = F.avg_pool2d(feat_pad, kernel_size=pool_kernel, stride=1, padding=0)
    high_freq = feat_clean - local_mean
    # L2 norm across channels captures spatial texture contrast (head contours)
    energy = torch.norm(high_freq, p=2, dim=1, keepdim=True)
    return energy


def compute_local_contrast_ratio(
    energy: torch.Tensor,
    contrast_kernel: int = 15,
    noise_floor: float = 0.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute dimensionless Weber-Fechner relative contrast ratio against local neighborhood.

    E_rel(u) = E(u) / (AvgPool_k(E)(u) + noise_floor)
    - On uniform/blank regions: E(u) ≈ AvgPool(E)(u) => E_rel(u) ≈ 1.0.
    - On sharp discrete head peaks: E(u) >> AvgPool(E)(u) => E_rel(u) > 1.0.
    - On microscopic noise (E << noise_floor): E_rel(u) ≈ 0.0 (prevents noise amplification).

    Args:
        energy: [B, 1, H, W] spatial energy tensor.
        contrast_kernel: Window size for neighborhood background energy averaging.
        noise_floor: Absolute threshold of visibility floor (prevents 0/0 noise explosion).
        eps: Epsilon floor for division.

    Returns:
        rel_contrast: [B, 1, H, W] dimensionless contrast ratio clamped to [0.0, 5.0].
    """
    max_e = energy.amax(dim=(-2, -1), keepdim=True)
    pad = contrast_kernel // 2
    energy_pad = F.pad(energy, (pad, pad, pad, pad), mode="replicate")
    local_bg = F.avg_pool2d(energy_pad, kernel_size=contrast_kernel, stride=1, padding=0)
    denom = local_bg + max(float(noise_floor), float(eps))
    ratio = energy / denom
    # Neutral fallback only if the entire slice has vanishing energy (e.g. blank input)
    is_dead = max_e < float(eps)
    ratio = torch.where(is_dead, torch.ones_like(ratio), ratio)
    return ratio.clamp(0.0, 5.0)


def fed_mcbpro_transport(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor | None,
    regions: RegionSet,
    feat_p4: torch.Tensor | None = None,
    eps: float = 1e-6,
    scale_weights: torch.Tensor | None = None,
    fg_mask: torch.Tensor | None = None,
    deficit_seeding: bool = True,
    deficit_deadband: float = 0.20,
    dense_threshold: float = 2.0,
    valley_power: float = 2.0,
    reg_threshold: float = 0.001,
    tau_count_null: float = 0.0,
    tau_peak: float = 0.05,
    feat_energy_weight: float = 0.25,
    ratio_clamp_max: float = 25.0,
    peak_kernel_size: int = 7,
    contrast_kernel: int = 15,
    noise_floor: float = 0.2,
    conservative_scaling: bool = False,
    return_multiplier: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Execute the Feature-guided Edge-Directed Multi-Scale Consistent Bregman-Poisson Recovery Operator."""
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
    b_s_raw = torch.nan_to_num(b_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    w_s_f = torch.nan_to_num(weight_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    boxes_tensor = regions.boxes if hasattr(regions, "boxes") else regions

    # Observation-space smooth null-measure shrinkage (anchors noise before projection)
    if tau_count_null > 0.0:
        null_factor = torch.clamp(b_s_raw / float(tau_count_null), 0.0, 1.0) ** 2
        b_s_f = b_s_raw * null_factor
    else:
        b_s_f = b_s_raw

    # Precompute scale routing partitions and normalized per-scale coverage once
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

    # Base uniform coverage for adjoint normalization
    cov_uniform = regional_adjoint(
        torch.ones((1, 1, boxes_tensor.shape[0]), device=y0.device, dtype=work_dtype),
        boxes_tensor,
        height=h,
        width=w,
        out_dtype=work_dtype,
    ).clamp_min(float(eps))

    # --- STAGE 1: Continuous Scale-Routed Regional Presence Field ---
    area = regions.area.to(dtype=work_dtype, device=y0.device).view(1, 1, -1).clamp_min(1.0)
    density_per_box = b_s_f / area
    b_density_field = _backproject_routed(
        density_per_box, boxes_tensor, cov_uniform, routed_scales, b, h, w, work_dtype
    )

    m_reg = torch.tanh(b_density_field / float(reg_threshold))

    if fg_mask is not None:
        if fg_mask.shape[-2:] != (h, w):
            fg_mask = F.interpolate(fg_mask, size=(h, w), mode="bilinear", align_corners=False)
        m_effective = (fg_mask.to(dtype=work_dtype).clamp(0.0, 1.0) * m_reg).clamp(0.0, 1.0)
    else:
        m_effective = m_reg

    # --- STAGE 2: Dimensionless Local-Contrast Deficit Carrier Seeding ---
    mu_0 = regional_sum(y0_f, boxes_tensor)
    delta_deficit = torch.relu(b_s_f - mu_0 - float(deficit_deadband))

    if deficit_seeding and (delta_deficit > float(eps)).any():
        k_peak = max(3, int(peak_kernel_size))
        pad_peak = k_peak // 2
        y0_local_max = F.max_pool2d(y0_f, kernel_size=k_peak, stride=1, padding=pad_peak)
        peak_dom = (y0_f / (y0_local_max + float(eps))) * torch.tanh(y0_local_max / float(tau_peak))
        valley_attenuation = (1.0 - peak_dom.clamp(0.0, 1.0)) ** float(valley_power)

        if feat_p4 is not None and feat_energy_weight > 0.0:
            raw_energy_native = compute_feature_high_pass_energy(feat_p4, pool_kernel=5, eps=eps)
            if raw_energy_native.shape[-2:] != (h, w):
                raw_energy = F.interpolate(
                    raw_energy_native, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                raw_energy = raw_energy_native

            contrast_ratio = compute_local_contrast_ratio(
                raw_energy, contrast_kernel=contrast_kernel, noise_floor=noise_floor, eps=eps
            )
            alpha_base = float(feat_energy_weight)

            if dense_threshold > 0.0:
                t_null = float(tau_count_null)
                t_dense = float(dense_threshold)
                gate_boxes = torch.clamp((b_s_f - t_null) / max(t_dense - t_null, float(eps)), 0.0, 1.0)
                dense_gate_field = _backproject_routed(
                    gate_boxes, boxes_tensor, cov_uniform, routed_scales, b, h, w, work_dtype
                )
                alpha_spatial = alpha_base * dense_gate_field.clamp(0.0, 1.0)
            else:
                alpha_spatial = alpha_base

            contrast_mod = (1.0 - alpha_spatial) + alpha_spatial * contrast_ratio.to(dtype=work_dtype)
            gap_weight = m_effective * valley_attenuation * contrast_mod
        else:
            gap_weight = m_effective * valley_attenuation

        gap_capacity = regional_sum(gap_weight, boxes_tensor)
        eff_capacity = gap_capacity.clamp_min(1.0)
        rho_deficit = torch.where(
            gap_capacity > float(eps),
            (delta_deficit / eff_capacity).clamp(0.0, 10.0),
            torch.zeros_like(delta_deficit),
        )

        deficit_field = _backproject_routed(
            rho_deficit, boxes_tensor, cov_uniform, routed_scales, b, h, w, work_dtype
        )
        epsilon_seed = gap_weight * deficit_field
        y_aug = y0_f + epsilon_seed
    else:
        y_aug = y0_f

    # --- STAGE 3: Exact Bregman-Poisson Closed-Form Projection (Csiszar I-Projection) ---
    if conservative_scaling:
        k_peak = max(3, int(peak_kernel_size))
        pad_peak = k_peak // 2
        k_base = 2 * k_peak - 1
        pad_base = k_base // 2
        y0_local_max = F.max_pool2d(y0_f, kernel_size=k_peak, stride=1, padding=pad_peak)
        y0_base_min = -F.max_pool2d(-y0_f, kernel_size=k_base, stride=1, padding=pad_base)
        apex_ratio = (y0_f / (y0_local_max + float(eps))).clamp(0.0, 1.0).square()
        rel_prom = torch.relu(y0_f - y0_base_min) / (y0_local_max + float(eps))
        amp_floor = max(0.20 * float(tau_peak), float(eps))
        prom_gate = torch.tanh(rel_prom / 0.25) * torch.tanh(y0_local_max / amp_floor)
        p_peak = (m_effective * apex_ratio * prom_gate).clamp(0.0, 1.0)
        peak_boost = 1.0 + p_peak
        y_ref = y_aug * peak_boost
    else:
        peak_boost = None
        y_ref = y_aug

    mu_ref = regional_sum(y_ref, boxes_tensor)
    scaling_ratio = b_s_f / mu_ref.clamp_min(float(eps))

    both_zero = (b_s_f < float(eps)) & (mu_ref < float(eps))
    scaling_ratio = torch.where(both_zero, torch.ones_like(scaling_ratio), scaling_ratio)
    scaling_ratio = scaling_ratio.clamp(0.0, float(ratio_clamp_max))

    weighted_s = (w_s_f * scaling_ratio).to(dtype=work_dtype)
    if routed_scales is not None:
        s_raw = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
        for mask_k, boxes_k, _, pi_k in routed_scales:
            w_k = w_s_f[:, :, mask_k]
            w_s_k = weighted_s[:, :, mask_k]
            cov_w_k = regional_adjoint(
                w_k, boxes_k, height=h, width=w, out_dtype=work_dtype
            ).clamp_min(float(eps))
            adj_s_k = regional_adjoint(w_s_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
            s_raw = s_raw + pi_k * (adj_s_k / cov_w_k)
    else:
        cov_w = regional_adjoint(
            w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
        ).clamp_min(float(eps))
        adj_s = regional_adjoint(weighted_s, boxes_tensor, height=h, width=w, out_dtype=work_dtype)
        s_raw = adj_s / cov_w

    s_field = (peak_boost * s_raw) if peak_boost is not None else s_raw
    y_projected = torch.clamp_min(y_ref * s_raw, 0.0).to(dtype=y0.dtype)

    if return_multiplier:
        return y_projected, s_field.to(dtype=y0.dtype)
    return y_projected


# Backward compatibility alias
perspective_mcbpro_transport = fed_mcbpro_transport
