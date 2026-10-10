"""Morozov discrepancy principle and adaptive noise deadband shrinkage operators.

Provides:
1. compute_morozov_discrepancy: Computes symmetric, asymmetric, or spatially routed
   Morozov shrinkage on regional observation residuals delta = q - b.
"""
from __future__ import annotations

import torch
from .prefix_sums import regional_sum
from .regions import RegionSet


def compute_morozov_discrepancy(
    delta: torch.Tensor,
    q: torch.Tensor,
    b32: torch.Tensor,
    b_variance: torch.Tensor,
    regions: RegionSet,
    scale_routing_weights: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
    spatial_morozov: bool = False,
    morozov_gamma_scales: tuple[float, ...] = (0.25, 0.50, 0.75),
    anscombe_morozov: bool = False,
    asymmetric_morozov: bool = False,
    morozov_gamma_under: float = 0.20,
    morozov_rho: float = 0.30,
    morozov_rho_cap: float = 0.0,
    multiscale_morozov: bool = False,
    morozov_scale_beta: float = 1.0,
) -> torch.Tensor:
    """Apply Morozov Discrepancy Shrinkage (Symmetric, Asymmetric, Scale-Routed, or Multiscale Nemirovski).

    When multiscale_morozov=True (Frick, Marnitz & Munk, JMIV 2012):
        Scales the discrepancy threshold inversely with window scale s_k = sqrt(|R_k|):
            gamma_k = gamma_0 * (|R_min| / |R_k|)^(beta / 2) = gamma_0 * (s_0 / s_k)^beta
        eliminating the coarse 128px horizon box deadband blind spot.
    """
    if spatial_morozov and scale_routing_weights is not None:
        pi_sum = regional_sum(
            scale_routing_weights.detach().float(), regions.boxes, out_dtype=torch.float32
        )
        reg_area = regions.area.float().view(1, 1, -1).clamp_min(1.0)
        pi_box = pi_sum / reg_area
        if len(morozov_gamma_scales) == pi_box.shape[1]:
            gamma_s = torch.as_tensor(
                morozov_gamma_scales, dtype=torch.float32, device=q.device
            ).view(1, -1, 1)
            base_gamma: float | torch.Tensor = (pi_box * gamma_s).sum(dim=1, keepdim=True)
        else:
            base_gamma = float(morozov_gamma)
    else:
        base_gamma = float(morozov_gamma)

    if multiscale_morozov and regions.area.numel() > 0:
        reg_area = regions.area.to(device=q.device, dtype=torch.float32).view(1, 1, -1).clamp_min(1.0)
        min_area = reg_area.min().clamp_min(1.0)
        scale_factor = (min_area / reg_area).pow(0.5 * float(morozov_scale_beta))
        base_gamma = base_gamma * scale_factor

    if anscombe_morozov:
        c = 0.375
        g_q = 2.0 * torch.sqrt(q.clamp_min(0.0) + c)
        g_b = 2.0 * torch.sqrt(b32.clamp_min(0.0) + c)
        g_delta = g_q - g_b
        if asymmetric_morozov:
            gamma_u_base = base_gamma if (spatial_morozov or multiscale_morozov) else float(morozov_gamma_under)
            gamma_under = gamma_u_base / (1.0 + float(morozov_rho) * torch.sqrt(b32.clamp_min(0.0)))
            gamma_eff = torch.where(g_delta > 0.0, base_gamma, gamma_under)
        else:
            gamma_eff = base_gamma
        g_shrunk = torch.sign(g_delta) * torch.clamp_min(g_delta.abs() - gamma_eff, 0.0)
        scale_symm = 0.5 * (torch.sqrt(q.clamp_min(0.0) + c) + torch.sqrt(b32.clamp_min(0.0) + c))
        return g_shrunk * scale_symm

    sigma_b = torch.sqrt(b_variance.float().clamp_min(1e-12))
    if asymmetric_morozov:
        gamma_u_base = base_gamma if (spatial_morozov or multiscale_morozov) else float(morozov_gamma_under)
        gamma_under = gamma_u_base / (1.0 + float(morozov_rho) * torch.sqrt(b32.clamp_min(0.0)))
        gamma_eff = torch.where(delta > 0.0, base_gamma, gamma_under)
    else:
        gamma_eff = base_gamma
    deadband = gamma_eff * sigma_b
    if morozov_rho_cap > 0.0 and not asymmetric_morozov:
        deadband = deadband / (1.0 + float(morozov_rho_cap) * sigma_b)
    return torch.sign(delta) * torch.clamp_min(delta.abs() - deadband, 0.0)
