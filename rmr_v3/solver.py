from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    charbonnier_tv_step,
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


def laplacian_tv_diffusion(
    y: torch.Tensor,
    tv_lambda: float,
    kernel: torch.Tensor | None = None,
) -> torch.Tensor:
    """Isotropic Laplacian total-variation diffusion step: y <- max(0, y + lambda * Delta y)."""
    if tv_lambda <= 0.0:
        return y
    if kernel is None:
        kernel = _LAPLACE_KERNEL.to(device=y.device, dtype=y.dtype)
    else:
        kernel = kernel.to(device=y.device, dtype=y.dtype)
    lap = F.conv2d(y, kernel, padding=1)
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
    tv_lambda: float = 0.0,
    tv_type: str = "laplacian",
    tv_eps_c: float = 0.1,
    laplace_kernel: torch.Tensor | None = None,
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
        tv_lambda: Total variation diffusion coefficient.
        tv_type: "laplacian" (isotropic) or "charbonnier" (edge-preserving).
        tv_eps_c: Charbonnier TV smoothness constant.
        laplace_kernel: Optional pre-allocated 3x3 Laplacian convolution kernel.

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
    tau_step = (effective_omega * effective_tau) / max(int(iterations), 1)

    # Compute weighted coverage field: D_w = A^T w
    cov_w = weighted_coverage(
        weight_solver,
        regions,
        h,
        w,
        eps=eps,
    )

    y = y0
    iterates: list[torch.Tensor] = [y0]
    residual_fields: list[torch.Tensor] = []
    energy_trace: list[dict[str, torch.Tensor]] = []

    for _ in range(iterations):
        energy_before = weighted_regional_energy(
            y,
            b_solver,
            weight_solver,
            regions,
        )

        # ── Step 1: Adjoint discrepancy scatter ──────────────────────────────
        field = weighted_normalized_adjoint_field(
            y,
            b_solver,
            weight_solver,
            regions,
            weighted_cov=cov_w,
            residual_clip=residual_clip,
            eps=eps,
            solver_mode=solver_mode,
            density_gate_rho=float(density_gate_rho),
            density_gate_floor=float(density_gate_floor),
        )

        y_step = y.float() - effective_omega * field

        # ── Step 2: Proximal soft-thresholding L1-shrinkage ────────────────────
        y_next = proximal_soft_threshold(y_step, tau=tau_step)

        # ── Step 3: Total Variation diffusion ─────────────────────────────────
        if effective_tv_lambda > 0.0:
            if tv_type == "charbonnier":
                y_next = charbonnier_tv_step(y_next, effective_tv_lambda, float(tv_eps_c))
            else:
                y_next = laplacian_tv_diffusion(y_next, effective_tv_lambda, kernel=laplace_kernel)

        y_next = y_next.to(dtype=y.dtype)

        energy_after = weighted_regional_energy(
            y_next,
            b_solver,
            weight_solver,
            regions,
        )

        energy_trace.append({"before": energy_before, "after": energy_after})
        residual_fields.append(field)
        iterates.append(y_next)
        y = y_next

    return {
        "y": y,
        "iterates": iterates,
        "residual_fields": residual_fields,
        "energy_trace": energy_trace,
        "effective_omega": effective_omega,
        "effective_tv_lambda": effective_tv_lambda,
    }
