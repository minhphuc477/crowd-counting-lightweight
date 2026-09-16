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
    tv_lambda: float,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Isotropic Laplacian total-variation diffusion step with Neumann zero-flux boundary:
    y <- max(0, y + lambda * Delta y). Uses replication padding so sum(Delta y) == 0.
    """
    if tv_lambda <= 0.0:
        return y
    if kernel is None:
        kernel = _LAPLACE_KERNEL.to(device=y.device, dtype=y.dtype)
    else:
        kernel = kernel.to(device=y.device, dtype=y.dtype)
    y_pad = F.pad(y, (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(y_pad, kernel, padding=0)
    return torch.clamp_min(y + tv_lambda * lap, 0.0)


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
    use_scale_entropy_trust: bool = False,
    use_nesterov_momentum: bool = False,
    adaptive_relaxation: bool = False,
    adaptive_relax_sparse: float = 0.70,
    adaptive_relax_dense_boost: float = 0.50,
    adaptive_relax_threshold: float = 0.05,
    adaptive_relax_scale: float = 0.02,
) -> dict[str, Any]:
    """Execute unrolled Proximal Reliability-Weighted SIRT measure reconciliation.

    Solves the continuous-discrete inverse problem:
        min_{y >= 0} || W^{1/2} (A y - b) ||_2^2 + lambda_TV * TV(y) + tau * ||y||_1

    via T unrolled projected Richardson-Lucy / SIRT steps with dynamic relaxation:
        y_{t+1} = S_{tau}^+ ( y_t - omega * D_w^{-1} A^T W (A y_t - b) ) + TV_diff(y_{t+1})

    Args:
        y0: Initial fine density carrier [B, 1, H, W].
        b_solver: Target regional counts [B, 1, M].
        weight_solver: Regional reliability weights [B, 1, M].
        regions: Canonical RegionSet geometric dictionary.
        iterations: Number of unrolled solver iterations T.
        omega: SIRT step relaxation parameter.
        solver_strength: Dynamic ramp factor in [0.0, 1.0].
        residual_clip: Bound on regional discrepancy (0.0 = unclipped).
        eps: Small positive constant for numerical division safety.
        solver_mode: "additive" (standard SIRT) or "multiplicative" (density-gated).
        density_gate_rho: Density gating threshold rho_0 for multiplicative mode.
        density_gate_floor: Minimum gate floor for multiplicative mode.
        proximal_tau: Proximal soft-thresholding parameter tau.
        proximal_mode: "firm", "soft", or "none".
        proximal_mu: MCP threshold parameter.
        tv_lambda: Total variation diffusion coefficient.
        tv_type: "laplacian" (isotropic) or "charbonnier" (edge-preserving).
        tv_eps_c: Charbonnier TV smoothness constant.
        laplace_kernel: Optional pre-allocated 3x3 Laplacian convolution kernel.
        scale_routing_weights: Optional spatial scale routing probabilities [B, K, H, W].
        trust_region_kappa: Morozov trust-region step bounding parameter.
        trust_region_floor: Minimum floor for trust-region step bounding.
        adjoint_mode: "flat" (standard Lebesgue adjoint) or "radon_nikodym" (measure-modulated).
        b_variance: Optional predictive variance of b_solver for Morozov shrinkage [B, 1, M].
        morozov_gamma: Threshold multiplier for Morozov discrepancy deadband (0.0 = disabled).

    Returns:
        Dictionary containing:
            "y": Final reconciled measure field [B, 1, H, W].
            "iterates": List of iterates [y0, y1, ..., yT].
            "residual_fields": List of adjoint scatter fields at each iteration.
            "energy_trace": List of dicts with {"before": E_before, "after": E_after}.
            "effective_omega": Actual omega applied.
            "effective_tv_lambda": Actual TV lambda applied.
    """
    b, _, h, w = y0.shape
    strength = min(max(float(solver_strength), 0.0), 1.0)
    effective_omega = float(omega) * strength
    effective_tv_lambda = float(tv_lambda) * strength
    effective_tau = float(proximal_tau)
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

    y_curr = y0
    y_prev = y0
    theta_curr = 1.0
    iterates: list[torch.Tensor] = [y0]
    residual_fields: list[torch.Tensor] = []
    energy_trace: list[dict[str, torch.Tensor]] = []

    prev_y: torch.Tensor | None = None
    prev_field: torch.Tensor | None = None

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
            density_gate_rho=float(density_gate_rho),
            density_gate_floor=float(density_gate_floor),
            scale_routing_weights=scale_routing_weights,
            scale_partitions=scale_partitions,
            adjoint_mode=adjoint_mode,
            b_variance=b_variance,
            morozov_gamma=float(morozov_gamma),
        )

        # Adaptive Barzilai-Borwein step size
        current_omega: float | torch.Tensor = effective_omega
        if use_barzilai_borwein and prev_y is not None and prev_field is not None and effective_omega > 0.0:
            s_diff = (z_state - prev_y).float()
            r_diff = (field - prev_field).float()
            dot_sr = (s_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True)
            norm_r_sq = (r_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True) + 1e-6
            omega_candidate = torch.where(
                dot_sr > 0.0,
                dot_sr / norm_r_sq,
                torch.as_tensor(effective_omega, device=dot_sr.device, dtype=dot_sr.dtype),
            ).detach()
            current_omega = torch.clamp(omega_candidate, min=0.2 * effective_omega, max=2.0 * effective_omega)

        # Density-Adaptive Over-Relaxation (RMR-v20)
        if adaptive_relaxation:
            z_smooth = F.avg_pool2d(z_state.float(), kernel_size=5, stride=1, padding=2)
            gate_dense = torch.sigmoid((z_smooth - float(adaptive_relax_threshold)) / float(adaptive_relax_scale))
            omega_mod = float(adaptive_relax_sparse) + (1.0 - float(adaptive_relax_sparse) + float(adaptive_relax_dense_boost)) * gate_dense
            current_omega = current_omega * omega_mod

        step_delta = current_omega * field
        if trust_region_kappa > 0.0:
            bound = float(trust_region_kappa) * torch.clamp_min(z_state.float(), float(trust_region_floor))
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
                y_next = laplacian_tv_diffusion(y_next, tv_step, kernel=laplace_kernel)

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
        "effective_omega": effective_omega,
        "effective_tv_lambda": effective_tv_lambda,
    }
