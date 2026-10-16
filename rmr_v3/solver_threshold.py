"""Proximal thresholding and shrinkage operators for RMR Unrolled SIRT solver."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "proximal_soft_threshold",
    "proximal_firm_threshold",
    "compute_adaptive_tau",
]


def proximal_soft_threshold(y: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Exact proximal operator for non-negative L1-shrinkage: S_tau^+(z) = max(0, z - tau).

    Provides a noise deadband in [0, tau] that completely suppresses background
    phantom mass accumulation without zero-absorbing barriers.
    """
    if isinstance(tau, (int, float)):
        if tau <= 0.0:
            return torch.clamp_min(y, 0.0)
        tau_val: float | torch.Tensor = float(tau)
    elif isinstance(tau, torch.Tensor):
        if tau.numel() == 1 and tau.item() <= 0.0:
            return torch.clamp_min(y, 0.0)
        if not (tau > 0.0).any():
            return torch.clamp_min(y, 0.0)
        tau_val = tau.clamp_min(0.0)
    else:
        tau_val = tau
    return torch.clamp_min(y - tau_val, 0.0)


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
    if isinstance(tau, (int, float)):
        if tau <= 0.0:
            return torch.clamp_min(y, 0.0)
        tau_val: float | torch.Tensor = float(tau)
    elif isinstance(tau, torch.Tensor):
        if tau.numel() == 1 and tau.item() <= 0.0:
            return torch.clamp_min(y, 0.0)
        if not (tau > 0.0).any():
            return torch.clamp_min(y, 0.0)
        tau_val = tau.clamp_min(0.0)
    else:
        tau_val = tau
    if mu <= 1.0:
        return torch.where(y > tau_val, y, torch.zeros_like(y))

    mu_val = float(max(mu, 1.001))
    mu_tau = mu_val * tau_val
    slope = mu_val / (mu_val - 1.0)
    ramp = slope * (y - tau_val)
    out = torch.where(y > mu_tau, y, ramp)
    return torch.clamp_min(out, 0.0)
