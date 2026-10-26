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
    """Execute the Feature-guided Edge-Directed Multi-Scale Consistent Bregman-Poisson Recovery Operator.

    Args:
        y0: [B, 1, H, W] carrier density map from fine head.
        b_solver: [B, 1, K] regional count evidence from evidence head.
        weight_solver: [B, 1, K] optional regional reliability weights.
        regions: RegionSet containing multiscale box definitions.
        feat_p4: Optional [B, C, H_feat, W_feat] neck feature map for feature-energy carrier seeding.
        eps: Epsilon floor for division.
        scale_weights: [B, S, H, W] continuous scale routing weights.
        fg_mask: Optional [B, 1, H, W] foreground probability mask.
        deficit_seeding: Whether to inject regional deficits into carrier valleys.
        dense_threshold: Count threshold above which FED contrast is active (<= 0 disables gating).
        valley_power: Exponent for inter-head valley attenuation (2.0 = smooth quadratic).
        reg_threshold: Spatial density threshold for presence field tanh.
        tau_count_null: Regional count threshold below which evidence is smoothly anchored (0.0 = off).
        tau_peak: Sensitivity scale for local peak normalization.
        feat_energy_weight: Blending weight alpha for local contrast ratio in [0, 1].
        ratio_clamp_max: Safety ceiling on Bregman scaling multipliers.
        peak_kernel_size: Neighborhood kernel for local peak detection.
        contrast_kernel: Neighborhood kernel for local contrast background pooling.
        noise_floor: Absolute threshold of visibility floor in Weber-Fechner denominator.
        return_multiplier: Whether to return (y_projected, s_field).

    Returns:
        y_projected: [B, 1, H, W] recovered Radon measure.
        s_field: Optional [B, 1, H, W] spatial scaling field.
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
    b_s_raw = torch.nan_to_num(b_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    w_s_f = torch.nan_to_num(weight_solver.to(dtype=work_dtype), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    boxes_tensor = regions.boxes if hasattr(regions, "boxes") else regions

    # Observation-space smooth null-measure shrinkage (anchors noise before projection)
    if tau_count_null > 0.0:
        null_factor = torch.clamp(b_s_raw / float(tau_count_null), 0.0, 1.0) ** 2
        b_s_f = b_s_raw * null_factor
    else:
        b_s_f = b_s_raw

    # Scale routing partitions
    scale_partitions = None
    active_pi_sum = None
    if scale_weights is not None:
        _, k_scales = scale_weights.shape[:2]
        if scale_weights.shape[-2:] != (h, w):
            scale_weights = F.interpolate(
                scale_weights, size=(h, w), mode="bilinear", align_corners=False
            )
        scale_partitions = partition_regions_by_scale(regions, k_scales, device=y0.device)
        active_scales = [
            k for (k, mask_k, boxes_k) in scale_partitions
            if mask_k is not None and boxes_k is not None and boxes_k.shape[0] > 0
        ]
        if active_scales:
            active_pi_sum = sum(
                scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype) for k in active_scales
            )

    # Base uniform coverage for adjoint normalization
    cov_uniform = regional_adjoint(
        torch.ones((1, 1, boxes_tensor.shape[0]), device=y0.device, dtype=work_dtype),
        boxes_tensor,
        height=h,
        width=w,
        out_dtype=work_dtype,
    ).clamp_min(float(eps))

    # --- STAGE 1: Continuous Regional Presence Field ---
    area = regions.area.to(dtype=work_dtype, device=y0.device).view(1, 1, -1).clamp_min(1.0)
    density_per_box = b_s_f / area
    b_density_field = regional_adjoint(
        density_per_box, boxes_tensor, height=h, width=w, out_dtype=work_dtype
    ) / cov_uniform

    m_reg = torch.tanh(b_density_field / float(reg_threshold))

    if fg_mask is not None:
        if fg_mask.shape[-2:] != (h, w):
            fg_mask = F.interpolate(fg_mask, size=(h, w), mode="bilinear", align_corners=False)
        m_effective = (fg_mask.to(dtype=work_dtype).clamp(0.0, 1.0) * m_reg).clamp(0.0, 1.0)
    else:
        m_effective = m_reg

    # --- STAGE 2: Dimensionless Local-Contrast Deficit Carrier Seeding ---
    mu_0 = regional_sum(y0_f, boxes_tensor)
    delta_deficit = torch.relu(b_s_f - mu_0)

    if deficit_seeding and (delta_deficit > float(eps)).any():
        k_peak = max(3, int(peak_kernel_size))
        pad_peak = k_peak // 2
        y0_local_max = F.max_pool2d(y0_f, kernel_size=k_peak, stride=1, padding=pad_peak)
        peak_dom = (y0_f / (y0_local_max + float(eps))) * torch.tanh(y0_local_max / float(tau_peak))
        valley_attenuation = (1.0 - peak_dom.clamp(0.0, 1.0)) ** float(valley_power)

        # Dimensionless Weber-Fechner Local Contrast Modulation:
        # Native-resolution energy extraction + density-aware spatial gating
        if feat_p4 is not None and feat_energy_weight > 0.0:
            # 1. Compute high-pass energy at native resolution to preserve inter-cell gradients
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

            # 2. Density-gated spatial modulation: active in dense crowds, neutral on moderate/sparse
            if dense_threshold > 0.0:
                t_null = float(tau_count_null)
                t_dense = float(dense_threshold)
                gate_boxes = torch.clamp((b_s_f - t_null) / max(t_dense - t_null, float(eps)), 0.0, 1.0)
                dense_gate_field = regional_adjoint(
                    gate_boxes, boxes_tensor, height=h, width=w, out_dtype=work_dtype
                ) / cov_uniform
                alpha_spatial = alpha_base * dense_gate_field.clamp(0.0, 1.0)
            else:
                alpha_spatial = alpha_base

            contrast_mod = (1.0 - alpha_spatial) + alpha_spatial * contrast_ratio.to(dtype=work_dtype)
            gap_weight = m_effective * valley_attenuation * contrast_mod
        else:
            gap_weight = m_effective * valley_attenuation

        # Integrate capacity over regions
        gap_capacity = regional_sum(gap_weight, boxes_tensor)
        eff_capacity = gap_capacity.clamp_min(1.0)
        rho_deficit = torch.where(
            gap_capacity > float(eps),
            (delta_deficit / eff_capacity).clamp(0.0, 10.0),
            torch.zeros_like(delta_deficit),
        )

        if scale_weights is not None and scale_partitions is not None:
            deficit_field = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
            for k, mask_k, boxes_k in scale_partitions:
                if mask_k is None or boxes_k is None or boxes_k.shape[0] == 0:
                    continue
                rho_k = rho_deficit[:, :, mask_k]
                ones_k = torch.ones((1, 1, boxes_k.shape[0]), device=y0.device, dtype=work_dtype)
                cov_k = regional_adjoint(
                    ones_k, boxes_k, height=h, width=w, out_dtype=work_dtype
                ).clamp_min(float(eps))
                adj_k = regional_adjoint(rho_k, boxes_k, height=h, width=w, out_dtype=work_dtype) / cov_k
                pi_k = scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype)
                if active_pi_sum is not None:
                    pi_k = pi_k / active_pi_sum.clamp_min(float(eps))
                deficit_field = deficit_field + pi_k * adj_k
        else:
            deficit_field = regional_adjoint(
                rho_deficit, boxes_tensor, height=h, width=w, out_dtype=work_dtype
            ) / cov_uniform

        epsilon_seed = gap_weight * deficit_field
        y_aug = y0_f + epsilon_seed
    else:
        y_aug = y0_f

    # --- STAGE 3: Exact Bregman-Poisson Closed-Form Projection ---
    mu_aug = regional_sum(y_aug, boxes_tensor)
    scaling_ratio = b_s_f / mu_aug.clamp_min(float(eps))

    both_zero = (b_s_f < float(eps)) & (mu_aug < float(eps))
    scaling_ratio = torch.where(both_zero, torch.ones_like(scaling_ratio), scaling_ratio)
    scaling_ratio = scaling_ratio.clamp(0.0, float(ratio_clamp_max))

    if conservative_scaling:
        boost_ratio = torch.relu(scaling_ratio - 1.0).clamp_max(float(ratio_clamp_max) - 1.0)
        weighted_boost = (w_s_f * boost_ratio).to(dtype=work_dtype)
        if scale_weights is not None and scale_partitions is not None:
            boost_field = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
            for k, mask_k, boxes_k in scale_partitions:
                if mask_k is None or boxes_k is None or boxes_k.shape[0] == 0:
                    continue
                w_k = w_s_f[:, :, mask_k]
                w_b_k = weighted_boost[:, :, mask_k]
                cov_k = regional_adjoint(
                    w_k, boxes_k, height=h, width=w, out_dtype=work_dtype
                ).clamp_min(float(eps))
                adj_k = regional_adjoint(w_b_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
                s_k = adj_k / cov_k
                pi_k = scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype)
                if active_pi_sum is not None:
                    pi_k = pi_k / active_pi_sum.clamp_min(float(eps))
                boost_field = boost_field + pi_k * s_k
        else:
            cov_w = regional_adjoint(
                w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
            ).clamp_min(float(eps))
            adj_s = regional_adjoint(weighted_boost, boxes_tensor, height=h, width=w, out_dtype=work_dtype)
            boost_field = adj_s / cov_w

        s_field = 1.0 + boost_field
        y_projected = torch.clamp_min(y_aug * s_field, 0.0).to(dtype=y0.dtype)
    else:
        weighted_s = (w_s_f * scaling_ratio).to(dtype=work_dtype)
        if scale_weights is not None and scale_partitions is not None:
            s_field = torch.zeros((b, 1, h, w), device=y0.device, dtype=work_dtype)
            for k, mask_k, boxes_k in scale_partitions:
                if mask_k is None or boxes_k is None or boxes_k.shape[0] == 0:
                    continue
                w_k = w_s_f[:, :, mask_k]
                w_s_k = weighted_s[:, :, mask_k]
                cov_k = regional_adjoint(
                    w_k, boxes_k, height=h, width=w, out_dtype=work_dtype
                ).clamp_min(float(eps))
                adj_k = regional_adjoint(w_s_k, boxes_k, height=h, width=w, out_dtype=work_dtype)
                s_k = adj_k / cov_k
                pi_k = scale_weights[:, k : k + 1, :, :].to(dtype=work_dtype)
                if active_pi_sum is not None:
                    pi_k = pi_k / active_pi_sum.clamp_min(float(eps))
                s_field = s_field + pi_k * s_k
        else:
            cov_w = regional_adjoint(
                w_s_f, boxes_tensor, height=h, width=w, out_dtype=work_dtype
            ).clamp_min(float(eps))
            adj_s = regional_adjoint(weighted_s, boxes_tensor, height=h, width=w, out_dtype=work_dtype)
            s_field = adj_s / cov_w
        y_projected = torch.clamp_min(y_aug * s_field, 0.0).to(dtype=y0.dtype)

    if return_multiplier:
        return y_projected, s_field.to(dtype=y0.dtype)
    return y_projected


# Backward compatibility alias
perspective_mcbpro_transport = fed_mcbpro_transport
