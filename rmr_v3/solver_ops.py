"""Unified re-export hub for RMR solver operators.

Maintains 100% backward compatibility for all solver operations, which are
modularized into:
- solver_diffusion: laplacian_tv_diffusion, perona_malik_anisotropic_diffusion
- solver_threshold: proximal_soft_threshold, proximal_firm_threshold, compute_adaptive_tau
- solver_anscombe: anscombe_transform, anscombe_discrepancy, density_gated_anscombe_discrepancy
- solver_softbox: make_gaussian_softbox_kernel, gaussian_softbox_forward, gaussian_softbox_adjoint_step
"""
from __future__ import annotations

from .solver_diffusion import (
    _LAPLACE_KERNEL,
    laplacian_tv_diffusion,
    perona_malik_anisotropic_diffusion,
)
from .solver_threshold import (
    compute_adaptive_tau,
    proximal_firm_threshold,
    proximal_soft_threshold,
)
from .solver_anscombe import (
    anscombe_discrepancy,
    anscombe_transform,
    density_gated_anscombe_discrepancy,
)
from .solver_softbox import (
    gaussian_softbox_adjoint_step,
    gaussian_softbox_forward,
    make_gaussian_softbox_kernel,
)

__all__ = [
    "_LAPLACE_KERNEL",
    "laplacian_tv_diffusion",
    "perona_malik_anisotropic_diffusion",
    "proximal_soft_threshold",
    "proximal_firm_threshold",
    "compute_adaptive_tau",
    "anscombe_transform",
    "anscombe_discrepancy",
    "density_gated_anscombe_discrepancy",
    "make_gaussian_softbox_kernel",
    "gaussian_softbox_forward",
    "gaussian_softbox_adjoint_step",
]
