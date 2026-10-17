"""Anscombe variance-stabilizing transformation and rate discrepancy operators for RMR solver."""
from __future__ import annotations

import torch

__all__ = [
    "anscombe_transform",
    "anscombe_discrepancy",
    "density_gated_anscombe_discrepancy",
]


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
    asymmetric_morozov: bool = False,
    morozov_gamma_under: float = 0.20,
    morozov_rho: float = 0.30,
    morozov_rho_cap: float = 0.0,
) -> torch.Tensor:
    """Compute Anscombe-stabilized rate discrepancy in float32 with optional A-SAM."""
    if q.ndim == 2:
        q = q.unsqueeze(1)
    if b.ndim == 2:
        b = b.unsqueeze(1)
    if b_variance is not None and b_variance.ndim == 2:
        b_variance = b_variance.unsqueeze(1)

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
        if asymmetric_morozov:
            gamma_under = float(morozov_gamma_under) / (1.0 + float(morozov_rho) * torch.sqrt(b32.clamp_min(0.0)))
            gamma_eff = torch.where(delta_tilde > 0.0, float(morozov_gamma), gamma_under)
        else:
            gamma_eff = float(morozov_gamma)
        deadband = gamma_eff * sigma_tilde
        if morozov_rho_cap > 0.0:
            deadband = deadband / (1.0 + float(morozov_rho_cap) * sigma_tilde)
        delta_tilde = torch.sign(delta_tilde) * torch.clamp_min(delta_tilde.abs() - deadband, 0.0)

    # Chain rule adjoint projection factor: 1 / sqrt(q_stab)
    rate_res = delta_tilde / torch.sqrt(q_stab).clamp_min(1e-6)
    return rate_res


def density_gated_anscombe_discrepancy(
    q: torch.Tensor,
    b: torch.Tensor,
    area: torch.Tensor,
    c: float = 0.375,
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
    tau_dense: float = 0.08,
    eps: float = 1e-6,
    asymmetric_morozov: bool = False,
    morozov_gamma_under: float = 0.20,
    morozov_rho: float = 0.30,
    morozov_rho_cap: float = 0.0,
) -> torch.Tensor:
    """Density-gated Anscombe variance-stabilized rate discrepancy with A-SAM."""
    if b.ndim == 2:
        b = b.unsqueeze(1)
    if b_variance is not None and b_variance.ndim == 2:
        b_variance = b_variance.unsqueeze(1)

    area_clamped = area.clamp_min(1.0)
    rates = torch.maximum(q.float(), b.float()) / area_clamped
    dense_mask = (rates >= float(tau_dense)).float()

    rate_res_anscombe = anscombe_discrepancy(
        q, b, c=c, b_variance=b_variance, morozov_gamma=morozov_gamma,
        asymmetric_morozov=asymmetric_morozov,
        morozov_gamma_under=morozov_gamma_under,
        morozov_rho=morozov_rho,
        morozov_rho_cap=morozov_rho_cap,
    )

    delta = q.float() - b.float()
    if morozov_gamma > 0.0 and b_variance is not None:
        sigma_b = torch.sqrt(b_variance.float().clamp_min(0.0))
        if asymmetric_morozov:
            denom = 1.0 + float(morozov_rho) * torch.sqrt(b.float().clamp_min(0.0))
            gamma_under = float(morozov_gamma_under) / denom
            gamma_eff = torch.where(delta > 0.0, float(morozov_gamma), gamma_under)
        else:
            gamma_eff = float(morozov_gamma)
        deadband = gamma_eff * sigma_b
        if morozov_rho_cap > 0.0:
            deadband = deadband / (1.0 + float(morozov_rho_cap) * sigma_b)
        delta = torch.sign(delta) * torch.clamp_min(delta.abs() - deadband, 0.0)
    eff_q = q.float() + float(eps) * area_clamped
    rate_res_linear = delta / eff_q.clamp_min(float(eps))

    return dense_mask * rate_res_anscombe + (1.0 - dense_mask) * rate_res_linear


def prepare_anscombe_target(
    z_state: torch.Tensor,
    b_solver: torch.Tensor,
    regions: Any,
    area_scale: float,
    area_normalized_adjoint: bool,
    density_gated_anscombe: bool,
    anscombe_c: float,
    b_variance: torch.Tensor | None,
    morozov_gamma: float,
    anscombe_tau_dense: float,
    asymmetric_morozov: bool,
    morozov_gamma_under: float,
    morozov_rho: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute regional discrepancy under Anscombe VST and return adjusted target b_target."""
    from rmr_core.operators import regional_sum

    q = regional_sum(z_state.float(), regions.boxes, out_dtype=torch.float32)
    area = regions.area.float().view(1, 1, -1)
    eff_area = area * area_scale if area_normalized_adjoint else area

    if density_gated_anscombe:
        rate_res = density_gated_anscombe_discrepancy(
            q,
            b_solver,
            eff_area,
            c=anscombe_c,
            b_variance=b_variance,
            morozov_gamma=float(morozov_gamma),
            tau_dense=float(anscombe_tau_dense),
            asymmetric_morozov=asymmetric_morozov,
            morozov_gamma_under=float(morozov_gamma_under),
            morozov_rho=float(morozov_rho),
        )
    else:
        rate_res = anscombe_discrepancy(
            q,
            b_solver,
            c=anscombe_c,
            b_variance=b_variance,
            morozov_gamma=float(morozov_gamma),
            asymmetric_morozov=asymmetric_morozov,
            morozov_gamma_under=float(morozov_gamma_under),
            morozov_rho=float(morozov_rho),
        )

    eff_q = q + float(eps) * eff_area.clamp_min(1.0)
    return q - rate_res * eff_q.clamp_min(float(eps))

