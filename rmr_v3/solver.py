from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    charbonnier_tv_step,
    partition_regions_by_scale,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)

# Standard 2D 5-point discrete Laplacian kernel for isotropic TV diffusion
_LAPLACE_KERNEL: torch.Tensor = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def proximal_soft_threshold(y: torch.Tensor, tau: float) -> torch.Tensor:
    """Exact proximal operator for non-negative L1-shrinkage: S_tau^+(z) = max(0, z - tau).

    Provides a noise deadband in [0, tau] that completely suppresses background
    phantom mass accumulation without zero-absorbing barriers.
    """
    if tau <= 0.0:
        return torch.clamp_min(y, 0.0)
    return torch.clamp_min(y - tau, 0.0)


def proximal_firm_threshold(
    y: torch.Tensor,
    tau: float,
    mu: float = 3.0,
) -> torch.Tensor:
    """Exact proximal operator for Minimax Concave Penalty (MCP) / Firm Thresholding:

        S_firm^+(z; tau, mu) =
            0                                  if z <= tau
            (mu / (mu - 1)) * (z - tau)        if tau < z <= mu * tau
            z                                  if z > mu * tau

    Properties for crowd counting:
    - z <= tau: Background noise is strictly zeroed out (anti-smearing / zero deadband).
    - z > mu * tau: Real crowd peaks suffer ZERO shrinkage (identity mapping),
      completely resolving the dense clump mass erosion caused by soft-thresholding.
    - tau < z <= mu * tau: Smooth, continuous monotonic transition.
    """
    if tau <= 0.0:
        return torch.clamp_min(y, 0.0)
    mu_val = float(max(mu, 1.001))
    mu_tau = mu_val * tau
    slope = mu_val / (mu_val - 1.0)
    ramp = slope * (y - tau)
    out = torch.where(y > mu_tau, y, ramp)
    return torch.clamp_min(out, 0.0)


def laplacian_tv_diffusion(
    y: torch.Tensor,
    tv_lambda: float | torch.Tensor,
    kernel: torch.Tensor | None = None,
    density_gated: bool = False,
    diffusion_dense_threshold: float = 0.15,
    diffusion_gate_beta: float = 0.03,
) -> torch.Tensor:
    """Isotropic Laplacian total-variation diffusion step with Neumann zero-flux boundary:
    y <- max(0, y + lambda * Delta y). Uses replication padding so sum(Delta y) == 0.

    When density_gated=True (RMR-v22):
        The effective diffusion rate is modulated by local density:
            gate(u) = 1.0 - sigmoid((y_smooth(u) - tau_dense) / beta)
        In sparse/background regions (y_smooth < tau_dense), gate -> 1.0 (full diffusion).
        In dense crowd clusters (y_smooth > tau_dense), gate -> 0.0 (strictly zero diffusion),
        preserving sharp peak separation and stopping dense crowd clump mass erosion.
    """
    if isinstance(tv_lambda, (int, float)) and tv_lambda <= 0.0:
        return y
    if isinstance(tv_lambda, torch.Tensor) and tv_lambda.numel() == 1 and tv_lambda.item() <= 0.0:
        return y
    if kernel is None:
        kernel = _LAPLACE_KERNEL.to(device=y.device, dtype=y.dtype)
    else:
        kernel = kernel.to(device=y.device, dtype=y.dtype)
    orig_ndim = y.ndim
    if orig_ndim == 2:
        y_4d = y.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y_4d = y.unsqueeze(0)
    elif orig_ndim == 4:
        y_4d = y
    else:
        raise ValueError(f"laplacian_tv_diffusion expects 2D, 3D, or 4D tensor, got ndim={orig_ndim}")

    y_pad = F.pad(y_4d, (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(y_pad, kernel, padding=0)

    if density_gated:
        y_smooth = F.avg_pool2d(y_4d.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False)
        y_effective = torch.maximum(y_4d.float(), y_smooth)
        gate = 1.0 - torch.sigmoid((y_effective - float(diffusion_dense_threshold)) / float(max(diffusion_gate_beta, 1e-4)))
        step_diff = tv_lambda * gate.to(dtype=y.dtype) * lap
    else:
        step_diff = tv_lambda * lap

    out = torch.clamp_min(y_4d + step_diff, 0.0)
    if orig_ndim == 2:
        return out.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return out.squeeze(0)
    return out


def unrolled_sirt_solver(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor,
    regions: RegionSet,
    iterations: int = 2,
    omega: float = 1.0,
    solver_strength: float = 1.0,
    residual_clip: float = 0.0,
    eps: float = 1e-6,
    solver_mode: str = "additive",
    density_gate_rho: float = 0.02,
    density_gate_floor: float = 0.02,
    proximal_tau: float = 0.0,
    proximal_mode: str = "soft",
    proximal_mu: float = 3.0,
    tv_lambda: float = 0.0,
    tv_type: str = "laplacian",
    tv_eps_c: float = 0.1,
    laplace_kernel: torch.Tensor | None = None,
    scale_routing_weights: torch.Tensor | None = None,
    trust_region_kappa: float = 0.0,
    trust_region_floor: float = 0.005,
    adjoint_mode: str = "flat",
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
    use_barzilai_borwein: bool = False,
    use_alternating_bb: bool = False,
    bb_clamp_min: float = 0.5,
    bb_clamp_max: float = 1.2,
    cyclic_bb_length: int = 1,
    use_scale_entropy_trust: bool = False,
    use_nesterov_momentum: bool = False,
    adaptive_relaxation: bool = False,
    adaptive_relax_sparse: float = 0.70,
    adaptive_relax_dense_boost: float = 0.50,
    adaptive_relax_threshold: float = 0.05,
    adaptive_relax_scale: float = 0.02,
    hybrid_recovery_alpha: float = 0.0,
    density_gated_diffusion: bool = False,
    diffusion_dense_threshold: float = 0.15,
    diffusion_gate_beta: float = 0.03,
    output_stride: int = 4,
) -> dict[str, Any]:
    """Execute unrolled Proximal Reliability-Weighted SIRT measure reconciliation.

    Solves the continuous-discrete inverse problem:
        min_{y >= 0} || W^{1/2} (A y - b) ||_2^2 + lambda_TV * TV(y) + tau * ||y||_1

    via T unrolled projected Richardson-Lucy / SIRT steps with dynamic relaxation:
        y_{t+1} = S_{tau}^+ ( y_t - omega * D_w^{-1} A^T W (A y_t - b) ) + TV_diff(y_{t+1})
    """
    b, _, h, w = y0.shape
    area_scale = (float(output_stride) / 4.0) ** 2
    strength = min(max(float(solver_strength), 0.0), 1.0)
    effective_omega = float(omega) * strength
    effective_tv_lambda = float(tv_lambda) * strength
    effective_tau = float(proximal_tau) * area_scale
    effective_rho = float(density_gate_rho) * area_scale
    effective_diff_thresh = float(diffusion_dense_threshold) * area_scale
    effective_diff_beta = float(diffusion_gate_beta) * area_scale
    effective_relax_thresh = float(adaptive_relax_threshold) * area_scale
    effective_relax_scale = float(adaptive_relax_scale) * area_scale
    effective_trust_floor = float(trust_region_floor) * area_scale
    # tau_step and tv_step are per-iteration budgets.
    # Divide by T so that total shrinkage/diffusion over all iterations equals the hyperparameter,
    # making tau and tv_lambda strictly T-invariant hyperparameters.
    tau_step = (effective_omega * effective_tau) / max(int(iterations), 1)
    tv_step = effective_tv_lambda / max(int(iterations), 1)

    # Short-circuit: When solver strength is 0.0 (e.g. during warmup epochs) or iterations <= 0,
    # y remains identically y0. Bypassing the unrolled loop completely eliminates wasted
    # forward/adjoint passes and GPU allocations during the warmup phase.
    if iterations <= 0 or (effective_omega == 0.0 and effective_tv_lambda == 0.0 and tau_step == 0.0):
        return {
            "y": y0,
            "iterates": [y0],
            "residual_fields": [],
            "energy_trace": [],
            "step_omegas": [],
            "effective_omega": effective_omega,
            "effective_tv_lambda": effective_tv_lambda,
        }

    # Pre-partition regions by scale once before the T-iteration solver loop.
    # This completely eliminates dynamic boolean masking and tensor slicing allocations inside the loop.
    scale_partitions = None
    if scale_routing_weights is not None:
        k_scales = scale_routing_weights.shape[1]
        scale_partitions = partition_regions_by_scale(regions, k_scales, device=y0.device)

    # Compute weighted coverage field: D_w = A^T w (optionally scale-routed)
    cov_w = weighted_coverage(
        weight_solver,
        regions,
        h,
        w,
        eps=eps,
        scale_routing_weights=scale_routing_weights,
        scale_partitions=scale_partitions,
    )

    scale_confidence = None
    if use_scale_entropy_trust and scale_routing_weights is not None:
        k_scales = scale_routing_weights.shape[1]
        if k_scales > 1:
            pi_safe = scale_routing_weights.clamp_min(1e-7)
            entropy = -(pi_safe * torch.log(pi_safe)).sum(dim=1, keepdim=True)
            max_entropy = math.log(float(k_scales))
            scale_confidence = (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
            if scale_confidence.shape[-2:] != (h, w):
                scale_confidence = F.interpolate(
                    scale_confidence, size=(h, w), mode="bilinear", align_corners=False
                )

    y_curr = y0
    y_prev = y0
    theta_curr = 1.0
    iterates: list[torch.Tensor] = [y0]
    residual_fields: list[torch.Tensor] = []
    energy_trace: list[dict[str, torch.Tensor]] = []
    step_omegas: list[torch.Tensor | float] = []

    prev_y: torch.Tensor | None = None
    prev_field: torch.Tensor | None = None
    cached_bb_omega: float | torch.Tensor | None = None

    for iter_idx in range(iterations):
        # Nesterov momentum extrapolation
        if use_nesterov_momentum and iter_idx > 0:
            theta_next = (1.0 + math.sqrt(1.0 + 4.0 * (theta_curr ** 2))) / 2.0
            beta = (theta_curr - 1.0) / theta_next
            z_state = y_curr + beta * (y_curr - y_prev)
            z_state = torch.clamp_min(z_state, 0.0)
            theta_curr = theta_next
        else:
            z_state = y_curr

        energy_before = weighted_regional_energy(
            y_curr,
            b_solver,
            weight_solver,
            regions,
        ).detach()

        # ── Step 1: Adjoint discrepancy scatter evaluated at extrapolated state z ──
        field = weighted_normalized_adjoint_field(
            z_state,
            b_solver,
            weight_solver,
            regions,
            weighted_cov=cov_w,
            residual_clip=residual_clip,
            eps=eps,
            solver_mode=solver_mode,
            density_gate_rho=float(effective_rho),
            density_gate_floor=float(density_gate_floor),
            scale_routing_weights=scale_routing_weights,
            scale_partitions=scale_partitions,
            adjoint_mode=adjoint_mode,
            b_variance=b_variance,
            morozov_gamma=float(morozov_gamma),
            hybrid_recovery_alpha=float(hybrid_recovery_alpha),
        )

        # Adaptive Barzilai-Borwein step size (BB-1, Cyclic BB-1, or Alternating BB-1 / BB-2)
        current_omega: float | torch.Tensor = effective_omega
        if (use_barzilai_borwein or use_alternating_bb) and prev_y is not None and prev_field is not None and effective_omega > 0.0:
            if cyclic_bb_length > 1 and cached_bb_omega is not None and ((iter_idx - 1) % cyclic_bb_length != 0):
                current_omega = cached_bb_omega
            else:
                s_diff = (z_state - prev_y).float()
                r_diff = (field - prev_field).float()
                dot_sr = (s_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True)
                norm_r_sq = (r_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6
                norm_s_sq = (s_diff * s_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6

                if use_alternating_bb and (iter_idx % 2 == 1):
                    # BB-2: Inverse Rayleigh quotient alpha_2 = ||s||^2 / <s, r>
                    omega_candidate = torch.where(
                        dot_sr > 1e-7,
                        norm_s_sq / dot_sr.clamp_min(1e-7),
                        torch.as_tensor(effective_omega, device=dot_sr.device, dtype=dot_sr.dtype),
                    ).detach()
                else:
                    # BB-1: Standard Rayleigh quotient alpha_1 = <s, r> / ||r||^2
                    omega_candidate = torch.where(
                        dot_sr > 0.0,
                        dot_sr / norm_r_sq,
                        torch.as_tensor(effective_omega, device=dot_sr.device, dtype=dot_sr.dtype),
                    ).detach()

                current_omega = torch.clamp(
                    omega_candidate,
                    min=float(bb_clamp_min) * effective_omega,
                    max=float(bb_clamp_max) * effective_omega,
                )
                cached_bb_omega = current_omega

        # Density-Adaptive Over-Relaxation (RMR-v20)
        if adaptive_relaxation:
            z_smooth = F.avg_pool2d(z_state.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False)
            gate_dense = torch.sigmoid((z_smooth - effective_relax_thresh) / max(effective_relax_scale, 1e-6))
            omega_mod = float(adaptive_relax_sparse) + (1.0 - float(adaptive_relax_sparse) + float(adaptive_relax_dense_boost)) * gate_dense
            current_omega = current_omega * omega_mod

        step_omegas.append(current_omega)
        step_delta = current_omega * field
        if trust_region_kappa > 0.0:
            bound = float(trust_region_kappa) * torch.clamp_min(z_state.float(), float(effective_trust_floor))
            if scale_confidence is not None:
                # Modulate trust bound: 25% floor on maximally ambiguous regions, 100% on confident regions
                bound = bound * (0.25 + 0.75 * scale_confidence)
            step_delta = torch.clamp(step_delta, min=-bound, max=bound)

        prev_y = z_state.detach()
        prev_field = field.detach()

        y_step = z_state.float() - step_delta

        # ── Step 2: Proximal thresholding L1-shrinkage ────────────────────
        if proximal_mode == "firm":
            y_next = proximal_firm_threshold(y_step, tau=tau_step, mu=float(proximal_mu))
        elif proximal_mode == "soft":
            y_next = proximal_soft_threshold(y_step, tau=tau_step)
        elif proximal_mode in ("none", "clamp"):
            y_next = torch.clamp_min(y_step, 0.0)
        else:
            raise ValueError(
                f"Unknown proximal_mode: '{proximal_mode}'. Must be 'firm', 'soft', or 'none'."
            )

        # ── Step 3: Total Variation diffusion ─────────────────────────────────
        if tv_step > 0.0:
            if tv_type == "charbonnier":
                y_next = charbonnier_tv_step(y_next, tv_step, float(tv_eps_c))
            else:
                y_next = laplacian_tv_diffusion(
                    y_next,
                    tv_step,
                    kernel=laplace_kernel,
                    density_gated=density_gated_diffusion,
                    diffusion_dense_threshold=effective_diff_thresh,
                    diffusion_gate_beta=effective_diff_beta,
                )

        y_next = y_next.to(dtype=y_curr.dtype)

        energy_after = weighted_regional_energy(
            y_next,
            b_solver,
            weight_solver,
            regions,
        ).detach()

        energy_trace.append({"before": energy_before, "after": energy_after})
        residual_fields.append(field)
        iterates.append(y_next)
        y_prev = y_curr
        y_curr = y_next

    y = y_curr

    return {
        "y": y,
        "iterates": iterates,
        "residual_fields": residual_fields,
        "energy_trace": energy_trace,
        "step_omegas": step_omegas,
        "effective_omega": effective_omega,
        "effective_tv_lambda": effective_tv_lambda,
    }
