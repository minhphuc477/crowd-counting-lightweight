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
    if isinstance(lambda_tv, (int, float)):
        if lambda_tv <= 0.0:
            return y
        raw_lambda = max(float(lambda_tv), 0.0)
    elif isinstance(lambda_tv, torch.Tensor):
        if lambda_tv.numel() == 1 and not (lambda_tv > 0.0):
            return y
        if not (lambda_tv > 0.0).any():
            return y
        raw_lambda = float(lambda_tv.clamp_min(0.0).mean().item()) if lambda_tv.numel() > 1 else max(float(lambda_tv.item()), 0.0)
    else:
        raw_lambda = max(float(lambda_tv), 0.0)

    y_f = y.float()
    cfl_bound = float(eps_c) / 4.0
    eff_lambda = min(raw_lambda, cfl_bound) if enforce_cfl else raw_lambda
    if eff_lambda <= 0.0:
        return y

    # Neumann (zero-flux) boundary: replicate-pad before finite differencing,
    # ensuring sum(div(g * grad y)) == 0 (mass conservation at image boundaries).
    # This is consistent with laplacian_tv_diffusion which also uses replicate padding.
    y_pad = F.pad(y_f, (1, 1, 1, 1), mode="replicate")  # [B, C, H+2, W+2]

    # Forward finite differences for gradient (on padded tensor → valid interior)
    dy_dx = y_pad[:, :, 1:-1, 2:] - y_pad[:, :, 1:-1, 1:-1]   # [B, C, H, W]
    dy_dy = y_pad[:, :, 2:, 1:-1] - y_pad[:, :, 1:-1, 1:-1]   # [B, C, H, W]

    # Charbonnier diffusivity: g = 1 / sqrt(|grad|^2 + eps_c^2)
    grad_sq = dy_dx.pow(2) + dy_dy.pow(2)
    g = (grad_sq + float(eps_c) ** 2).rsqrt()  # [B, C, H, W]

    # Flux: F_x = g * dy_dx,  F_y = g * dy_dy
    flux_x = g * dy_dx  # [B, C, H, W]
    flux_y = g * dy_dy  # [B, C, H, W]

    # Backward finite difference divergence (Neumann BC: zero-flux at boundaries)
    # dF_x/dx = F_x(i,j) - F_x(i,j-1): pad left with 0 (Neumann BC at left edge)
    div_x = flux_x - F.pad(flux_x[:, :, :, :-1], (1, 0))
    # dF_y/dy = F_y(i,j) - F_y(i-1,j): pad top with 0 (Neumann BC at top edge)
    div_y = flux_y - F.pad(flux_y[:, :, :-1, :], (0, 0, 1, 0))

    divergence = div_x + div_y  # [B, C, H, W]

    y_out = torch.clamp_min(y_f + eff_lambda * divergence, 0.0)
    return y_out.to(y.dtype)
