"""Continuous Radon measure density activation and variance stabilization transforms.

Provides:
1. _pool_local_density: Dimension-safe local average density pooling.
2. _smooth_floor: C1-continuous quadratic floor suppression with non-zero gradients.
3. _apply_curvature: Anscombe/Pade quadratic curvature power boost for dense clusters.
4. _density_activate: Unified activation combining temperature softplus, floor, adaptive scale, and curvature.
5. _softplus_inverse: Exact inverse softplus for calibrated logit bias initialization.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _pool_local_density(y: torch.Tensor, k_pool: int) -> torch.Tensor:
    """Compute local average density with dimension-safe pooling."""
    if y.ndim < 2:
        return y
    orig_ndim = y.ndim
    y_in = y.unsqueeze(0).unsqueeze(0) if orig_ndim == 2 else (y.unsqueeze(0) if orig_ndim == 3 else y)
    y_local = F.avg_pool2d(
        y_in, kernel_size=k_pool, stride=1, padding=k_pool // 2, count_include_pad=False
    )
    if y_local.shape[-2:] != y_in.shape[-2:]:
        y_local = y_local[..., : y_in.shape[-2], : y_in.shape[-1]]
    return y_local.squeeze(0).squeeze(0) if orig_ndim == 2 else (y_local.squeeze(0) if orig_ndim == 3 else y_local)


def _smooth_floor(y_base: torch.Tensor, floor_tau: float) -> torch.Tensor:
    """C1-continuous quadratic floor suppression with strictly non-zero gradients."""
    tau = float(floor_tau)
    if tau <= 0.0:
        return y_base
    return torch.where(y_base > tau, y_base - 0.5 * tau, y_base.square() / (2.0 * max(tau, 1e-8)))


def _apply_curvature(
    y_base: torch.Tensor,
    curvature_alpha: torch.nn.Parameter | None,
    curv_scale: float = 1.0,
    gated: bool = False,
    pool_kernel: int = 8,
    dense_threshold: float = 0.15,
    gate_beta: float = 0.03,
    curvature_pade: bool = False,
) -> torch.Tensor:
    """Curvature power boost for high-density crowds with optional spatial gating and Pade approximant."""
    if curvature_alpha is None:
        return y_base
    alpha_eff = F.softplus(curvature_alpha)
    orig_dtype = y_base.dtype
    y_base_f32 = y_base.float()
    curv_poly = (y_base_f32.square() / (1.0 + y_base_f32)) if curvature_pade else y_base_f32.square()
    if gated:
        y_local = _pool_local_density(y_base_f32, int(pool_kernel))
        gate_dense = torch.sigmoid((y_local - float(dense_threshold)) / float(max(gate_beta, 1e-4)))
        curv_term = float(curv_scale) * alpha_eff.float() * gate_dense * curv_poly
    else:
        curv_term = float(curv_scale) * alpha_eff.float() * curv_poly
    return (y_base_f32 + curv_term).to(orig_dtype)


def _density_activate(
    z: torch.Tensor,
    *,
    temp_softplus: bool,
    tau: torch.nn.Parameter | None,
    density_curvature: bool,
    curvature_alpha: torch.nn.Parameter | None,
    gated_density_curvature: bool,
    curvature_dense_threshold: float,
    curvature_gate_beta: float,
    curvature_pool_kernel: int,
    curv_scale: float = 1.0,
    floor_tau: float = 0.0,
    density_adaptive_scale: bool = False,
    density_scale_gamma: torch.Tensor | float = 0.0,
    curvature_pade: bool = False,
) -> torch.Tensor:
    """Shared density activation: temperature softplus + density adaptive scale + curvature."""
    if temp_softplus and tau is not None:
        tau_clamped = tau.clamp_min(0.1)
        y_base = tau_clamped * F.softplus(z / tau_clamped)
    else:
        y_base = F.softplus(z)

    if floor_tau > 0.0:
        y_base = _smooth_floor(y_base, floor_tau)

    if density_adaptive_scale:
        gamma_val = (
            density_scale_gamma.clamp_min(0.0)
            if isinstance(density_scale_gamma, torch.Tensor)
            else max(float(density_scale_gamma), 0.0)
        )
        y_base = y_base * (1.0 + gamma_val * F.relu(z))

    if density_curvature and curvature_alpha is not None:
        return _apply_curvature(
            y_base,
            curvature_alpha,
            curv_scale=curv_scale,
            gated=gated_density_curvature,
            pool_kernel=curvature_pool_kernel,
            dense_threshold=curvature_dense_threshold,
            gate_beta=curvature_gate_beta,
            curvature_pade=curvature_pade,
        )
    return y_base


def _softplus_inverse(y: float) -> float:
    """Exact inverse of softplus: softplus_inv(y) = log(expm1(y))."""
    y_val = max(float(y), 1e-8)
    return math.log(math.expm1(y_val))
