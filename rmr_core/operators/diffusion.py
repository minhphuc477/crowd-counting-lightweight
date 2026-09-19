from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier_tv_step(
    y: torch.Tensor,
    lambda_tv: float,
    eps_c: float = 0.1,
    enforce_cfl: bool = False,
) -> torch.Tensor:
    """One step of anisotropic Charbonnier Total Variation diffusion.

    Applies: y_out = clamp(y + lambda_tv * div(g * grad(y)), min=0)
    where:
        grad(y) -- forward finite differences in x and y
        g(i) = 1 / sqrt(|grad_y(i)|^2 + eps_c^2)   (Charbonnier weight)
        div    -- backward finite difference divergence

    This is edge-preserving: near sharp edges (|grad_y| >> eps_c), g->0 so
    diffusion is suppressed. In flat regions (|grad_y| -> 0), g -> 1/eps_c
    so diffusion is near-isotropic.

    Stability (Von Neumann CFL):
        The strict 2D explicit Euler CFL bound is: lambda_tv <= eps_c / 4.
        If enforce_cfl=True and lambda_tv > eps_c / 4, lambda_tv is safely
        clamped to eps_c / 4 to guarantee contractivity and prevent checkerboard
        instability or exploding autograd Jacobians.

    Args:
        y:           [B, C, H, W] current density map (float32 expected).
        lambda_tv:   TV diffusion coefficient (typical: 0.010-0.020).
        eps_c:       Charbonnier regularization epsilon (default 0.1).
        enforce_cfl: If True, clamp lambda_tv to eps_c / 4 for strict stability.

    Returns:
        [B, C, H, W] diffused density map, clamped >= 0.
    """
    y_f = y.float()
    cfl_bound = float(eps_c) / 4.0
    eff_lambda = min(float(lambda_tv), cfl_bound) if enforce_cfl else float(lambda_tv)

    # Forward finite differences for gradient
    # dy_dx: shift in column direction (right neighbour - current), pad right edge with 0
    dy_dx = F.pad(y_f[..., 1:] - y_f[..., :-1], (0, 1))       # [B, C, H, W]
    # dy_dy: shift in row direction (bottom neighbour - current), pad bottom edge with 0
    dy_dy = F.pad(y_f[..., 1:, :] - y_f[..., :-1, :], (0, 0, 0, 1))  # [B, C, H, W]

    # Charbonnier diffusivity: g = 1 / sqrt(|grad|^2 + eps_c^2)
    grad_sq = dy_dx.pow(2) + dy_dy.pow(2)
    g = (grad_sq + float(eps_c) ** 2).rsqrt()  # [B, C, H, W]

    # Flux: F_x = g * dy_dx,  F_y = g * dy_dy
    flux_x = g * dy_dx  # [B, C, H, W]
    flux_y = g * dy_dy  # [B, C, H, W]

    # Backward finite difference divergence: div(F) = dF_x/dx + dF_y/dy
    # dF_x/dx = F_x(i) - F_x(i-1): pad left edge with 0
    div_x = flux_x - F.pad(flux_x[..., :-1], (1, 0))
    # dF_y/dy = F_y(i) - F_y(i-1): pad top edge with 0
    div_y = flux_y - F.pad(flux_y[..., :-1, :], (0, 0, 1, 0))

    divergence = div_x + div_y  # [B, C, H, W]

    y_out = torch.clamp_min(y_f + eff_lambda * divergence, 0.0)
    return y_out.to(y.dtype)
