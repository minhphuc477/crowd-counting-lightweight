from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def anscombe_transform(y: torch.Tensor, c: float = 0.375) -> torch.Tensor:
    """Compute Anscombe Variance-Stabilizing Transformation: T(y) = 2 * sqrt(max(0, y) + c).

    Under Poisson-distributed point processes, transforms heteroscedastic counts into
    homoscedastic space with asymptotically constant variance Var(T(y)) ≈ 1.0.
    """
    return 2.0 * torch.sqrt(torch.clamp_min(y.float(), 0.0) + float(c))


def anscombe_discrepancy(
    q: torch.Tensor,
    b: torch.Tensor,
    c: float = 0.375,
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
) -> torch.Tensor:
    """Compute Anscombe-stabilized rate discrepancy in float32.

    Formula:
        delta_tilde = 2.0 * (sqrt(q + c) - sqrt(b + c))
        rate_res = delta_tilde / sqrt(q + c)
    With optional Morozov deadband shrinkage in the stabilized domain.
    Eliminates the 1/b^2 gradient starvation on dense crowds, bounding updates to O(1).
    """
    q32 = q.float()
    b32 = b.float()
    c_val = float(c)

    q_stab = torch.clamp_min(q32, 0.0) + c_val
    b_stab = torch.clamp_min(b32, 0.0) + c_val

    # Asymptotically constant-variance discrepancy: 2 * (sqrt(q_stab) - sqrt(b_stab))
    delta_tilde = 2.0 * (torch.sqrt(q_stab) - torch.sqrt(b_stab))

    # Morozov shrinkage in variance-stabilized domain (sigma_tilde ≈ 1)
    if morozov_gamma > 0.0:
        if b_variance is not None:
            sigma_tilde = torch.sqrt(torch.clamp_min(b_variance.float(), 0.0) / b_stab).clamp_min(0.1)
        else:
            sigma_tilde = torch.ones_like(delta_tilde)
        deadband = float(morozov_gamma) * sigma_tilde
        delta_tilde = torch.sign(delta_tilde) * torch.clamp_min(delta_tilde.abs() - deadband, 0.0)

    # Chain rule adjoint projection factor: 1 / sqrt(q_stab)
    rate_res = delta_tilde / torch.sqrt(q_stab).clamp_min(1e-6)
    return rate_res


def compute_adaptive_tau(
    base_tau_step: float,
    y_current: torch.Tensor,
    stride: int = 4,
    mode: str = "density_adaptive",
    rho0: float = 0.05,
    pool_kernel: int = 5,
) -> torch.Tensor | float:
    """Compute spatially density-adaptive proximal threshold tau_eff(x, y).

    In sparse/background regions (rho <= rho0): returns base_tau_step (full noise pruning).
    In dense clusters (rho >> rho0): smoothly attenuates tau_eff -> 0 to prevent mass clipping.
    """
    if mode == "uniform" or base_tau_step <= 0.0 or y_current.numel() == 0:
        return base_tau_step

    y32 = y_current.float()
    pad = pool_kernel // 2
    local_density = F.avg_pool2d(
        y32, kernel_size=pool_kernel, stride=1, padding=pad, count_include_pad=False
    )
    # Inverse density attenuation: min(1.0, rho0 / max(rho, eps))
    rho0_val = float(rho0) * ((float(stride) / 4.0) ** 2)
    attenuation = torch.clamp(rho0_val / local_density.clamp_min(1e-6), max=1.0)
    return base_tau_step * attenuation


def proximal_firm_threshold(
    y: torch.Tensor,
    tau: float | torch.Tensor,
    mu: float = 3.0,
) -> torch.Tensor:
    """Exact MCP / Firm thresholding supporting both scalar and spatial tensor tau.

    - z <= tau: Background noise is strictly zeroed out (anti-smearing / zero deadband).
    - z > mu * tau: Real crowd peaks suffer ZERO shrinkage (identity mapping),
      completely resolving the dense clump mass erosion caused by soft-thresholding.
    - tau < z <= mu * tau: Smooth monotonic linear transition.
    """
    if isinstance(tau, (int, float)) and tau <= 0.0:
        return torch.clamp_min(y, 0.0)
    mu_val = float(max(mu, 1.001))
    mu_tau = mu_val * tau
    slope = mu_val / (mu_val - 1.0)
    ramp = slope * (y - tau)
    out = torch.where(y > mu_tau, y, ramp)
    return torch.clamp_min(out, 0.0)
