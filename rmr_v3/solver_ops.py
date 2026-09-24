from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# Standard 2D 5-point discrete Laplacian kernel for isotropic TV diffusion
_LAPLACE_KERNEL: torch.Tensor = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
).view(1, 1, 3, 3)


def proximal_soft_threshold(y: torch.Tensor, tau: float | torch.Tensor) -> torch.Tensor:
    """Exact proximal operator for non-negative L1-shrinkage: S_tau^+(z) = max(0, z - tau).

    Provides a noise deadband in [0, tau] that completely suppresses background
    phantom mass accumulation without zero-absorbing barriers.
    """
    if isinstance(tau, (int, float)) and tau <= 0.0:
        return torch.clamp_min(y, 0.0)
    return torch.clamp_min(y - tau, 0.0)


def laplacian_tv_diffusion(
    y: torch.Tensor,
    tv_lambda: float | torch.Tensor,
    kernel: torch.Tensor | None = None,
    density_gated: bool = False,
    diffusion_dense_threshold: float = 0.15,
    diffusion_gate_beta: float = 0.03,
) -> torch.Tensor:
    """Isotropic Laplacian total-variation diffusion step with Neumann zero-flux boundary:
    y <- max(0, y + lambda * Delta y). Uses replication padding so sum(Delta y) == 0.

    When density_gated=True (RMR-v22):
        The effective diffusion rate is modulated by local density:
            gate(u) = 1.0 - sigmoid((y_smooth(u) - tau_dense) / beta)
        In sparse/background regions (y_smooth < tau_dense), gate -> 1.0 (full diffusion).
        In dense crowd clusters (y_smooth > tau_dense), gate -> 0.0 (strictly zero diffusion),
        preserving sharp peak separation and stopping dense crowd clump mass erosion.
    """
    if isinstance(tv_lambda, (int, float)):
        if tv_lambda <= 0.0:
            return y
    elif isinstance(tv_lambda, torch.Tensor):
        if tv_lambda.numel() == 1 and not (tv_lambda > 0.0):
            return y
    if kernel is None:
        kernel = _LAPLACE_KERNEL.to(device=y.device, dtype=y.dtype)
    else:
        kernel = kernel.to(device=y.device, dtype=y.dtype)
    orig_ndim = y.ndim
    if orig_ndim == 2:
        y_4d = y.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y_4d = y.unsqueeze(0)
    elif orig_ndim == 4:
        y_4d = y
    else:
        raise ValueError(f"laplacian_tv_diffusion expects 2D, 3D, or 4D tensor, got ndim={orig_ndim}")

    y_pad = F.pad(y_4d, (1, 1, 1, 1), mode="replicate")
    lap = F.conv2d(y_pad, kernel, padding=0)

    if density_gated:
        y_smooth = F.avg_pool2d(y_4d.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False)
        y_effective = torch.maximum(y_4d.float(), y_smooth)
        gate = 1.0 - torch.sigmoid((y_effective - float(diffusion_dense_threshold)) / float(max(diffusion_gate_beta, 1e-4)))
        step_diff = tv_lambda * gate.to(dtype=y.dtype) * lap
    else:
        step_diff = tv_lambda * lap

    out = torch.clamp_min(y_4d + step_diff, 0.0)
    if orig_ndim == 2:
        return out.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return out.squeeze(0)
    return out


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
) -> torch.Tensor:
    """Compute Anscombe-stabilized rate discrepancy in float32 with optional A-SAM."""
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
    if isinstance(tau, (int, float)):
        if tau <= 0.0:
            return torch.clamp_min(y, 0.0)
    elif isinstance(tau, torch.Tensor):
        if tau.numel() == 1 and tau.item() <= 0.0:
            return torch.clamp_min(y, 0.0)
    mu_val = float(max(mu, 1.001))
    mu_tau = mu_val * tau
    slope = mu_val / (mu_val - 1.0)
    ramp = slope * (y - tau)
    out = torch.where(y > mu_tau, y, ramp)
    return torch.clamp_min(out, 0.0)


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
    )

    delta = q.float() - b.float()
    if morozov_gamma > 0.0 and b_variance is not None:
        sigma_b = torch.sqrt(b_variance.float().clamp_min(0.0))
        if asymmetric_morozov:
            gamma_under = float(morozov_gamma_under) / (1.0 + float(morozov_rho) * torch.sqrt(b.float().clamp_min(0.0)))
            gamma_eff = torch.where(delta > 0.0, float(morozov_gamma), gamma_under)
        else:
            gamma_eff = float(morozov_gamma)
        deadband = gamma_eff * sigma_b
        delta = torch.sign(delta) * torch.clamp_min(delta.abs() - deadband, 0.0)
    eff_q = q.float() + float(eps) * area_clamped
    rate_res_linear = delta / eff_q.clamp_min(float(eps))

    return dense_mask * rate_res_anscombe + (1.0 - dense_mask) * rate_res_linear


def perona_malik_anisotropic_diffusion(
    y: torch.Tensor,
    tv_lambda: float | torch.Tensor,
    kappa: float = 0.05,
) -> torch.Tensor:
    """Edge-preserving Perona-Malik anisotropic diffusion step (RMR-v31).

    Uses 4-directional conductance g(|nabla y|) = 1 / (1 + (|nabla y| / kappa)^2)
    with zero-flux Neumann boundary conditions:
        nabla_N y_{i,j} = y_{i-1, j} - y_{i, j}
        nabla_S y_{i,j} = y_{i+1, j} - y_{i, j}
        nabla_W y_{i,j} = y_{i, j-1} - y_{i, j}
        nabla_E y_{i,j} = y_{i, j+1} - y_{i, j}
        div(g nabla y) = sum_{d in {N,S,W,E}} g(|nabla_d y|) * nabla_d y
    Guarantees exact discrete mass conservation: sum(div(g nabla y)) == 0.
    At steep head peaks (|nabla y| >> kappa), g -> 0, preventing diffusion over-smoothing
    during deep SIRT unrolling (T=6, 8).
    """
    if isinstance(tv_lambda, (int, float)):
        if tv_lambda <= 0.0:
            return y
    elif isinstance(tv_lambda, torch.Tensor):
        if tv_lambda.numel() == 1 and not (tv_lambda > 0.0):
            return y

    orig_ndim = y.ndim
    if orig_ndim == 2:
        y4d = y.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y4d = y.unsqueeze(0)
    elif orig_ndim == 4:
        y4d = y
    else:
        raise ValueError(f"perona_malik_anisotropic_diffusion expects 2D, 3D, or 4D tensor, got ndim={orig_ndim}")

    y_curr = y4d.float()
    kap_sq = float(max(kappa, 1e-6)) ** 2
    dt = tv_lambda if isinstance(tv_lambda, torch.Tensor) else float(tv_lambda)

    y_pad = F.pad(y_curr, (1, 1, 1, 1), mode="replicate")
    y_c = y_pad[:, :, 1:-1, 1:-1]
    y_n = y_pad[:, :, 0:-2, 1:-1]
    y_s = y_pad[:, :, 2:, 1:-1]
    y_w = y_pad[:, :, 1:-1, 0:-2]
    y_e = y_pad[:, :, 1:-1, 2:]

    diff_n = y_n - y_c
    diff_s = y_s - y_c
    diff_w = y_w - y_c
    diff_e = y_e - y_c

    g_n = 1.0 / (1.0 + diff_n.square() / kap_sq)
    g_s = 1.0 / (1.0 + diff_s.square() / kap_sq)
    g_w = 1.0 / (1.0 + diff_w.square() / kap_sq)
    g_e = 1.0 / (1.0 + diff_e.square() / kap_sq)

    flux = g_n * diff_n + g_s * diff_s + g_w * diff_w + g_e * diff_e
    out = torch.clamp_min(y_c + dt * flux, 0.0)

    out = out.to(dtype=y.dtype)
    if orig_ndim == 2:
        return out.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return out.squeeze(0)
    return out

