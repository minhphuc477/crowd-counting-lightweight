"""Total variation and anisotropic diffusion operators for RMR Unrolled SIRT solver."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "laplacian_tv_diffusion",
    "perona_malik_anisotropic_diffusion",
    "apply_tv_diffusion",
]

# Standard 2D 5-point discrete Laplacian kernel for isotropic TV diffusion
_LAPLACE_KERNEL: torch.Tensor = torch.tensor(
    [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
).view(1, 1, 3, 3)


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
        lam_val: float | torch.Tensor = float(tv_lambda)
    elif isinstance(tv_lambda, torch.Tensor):
        if tv_lambda.numel() == 1 and not (tv_lambda > 0.0):
            return y
        if not (tv_lambda > 0.0).any():
            return y
        lam_val = tv_lambda.clamp_min(0.0)
    else:
        lam_val = tv_lambda
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

    if density_gated:
        y_smooth = F.avg_pool2d(y_4d.float(), kernel_size=5, stride=1, padding=2, count_include_pad=False)
        y_effective = torch.maximum(y_4d.float(), y_smooth)
        diff_beta = float(max(diffusion_gate_beta, 1e-4))
        gate = 1.0 - torch.sigmoid((y_effective - float(diffusion_dense_threshold)) / diff_beta)
        gate_pad = F.pad(gate.to(dtype=y.dtype), (1, 1, 1, 1), mode="replicate")


        y_c = y_pad[:, :, 1:-1, 1:-1]
        y_n = y_pad[:, :, 0:-2, 1:-1]
        y_s = y_pad[:, :, 2:, 1:-1]
        y_w = y_pad[:, :, 1:-1, 0:-2]
        y_e = y_pad[:, :, 1:-1, 2:]

        g_c = gate_pad[:, :, 1:-1, 1:-1]
        g_n = gate_pad[:, :, 0:-2, 1:-1]
        g_s = gate_pad[:, :, 2:, 1:-1]
        g_w = gate_pad[:, :, 1:-1, 0:-2]
        g_e = gate_pad[:, :, 1:-1, 2:]

        flux_n = (g_c * g_n) * (y_n - y_c)
        flux_s = (g_c * g_s) * (y_s - y_c)
        flux_w = (g_c * g_w) * (y_w - y_c)
        flux_e = (g_c * g_e) * (y_e - y_c)

        step_diff = lam_val * (flux_n + flux_s + flux_w + flux_e)
    else:
        lap = F.conv2d(y_pad, kernel, padding=0)
        step_diff = lam_val * lap

    out = torch.clamp_min(y_4d + step_diff, 0.0)
    if orig_ndim == 2:
        return out.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return out.squeeze(0)
    return out


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


def apply_tv_diffusion(
    y: torch.Tensor,
    tv_step: float,
    tv_type: str = "laplacian",
    tv_eps_c: float = 0.1,
    anisotropic_diffusion: bool = False,
    pm_kappa: float = 0.05,
    kernel: torch.Tensor | None = None,
    density_gated_diffusion: bool = False,
    effective_diff_thresh: float = 0.15,
    effective_diff_beta: float = 0.03,
) -> torch.Tensor:
    """Execute total-variation diffusion step (Laplacian, Perona-Malik, or Charbonnier)."""
    if tv_step <= 0.0:
        return y
    if tv_type == "charbonnier":
        from rmr_core.operators import charbonnier_tv_step

        return charbonnier_tv_step(y, tv_step, float(tv_eps_c))
    elif tv_type == "perona_malik" or anisotropic_diffusion:
        return perona_malik_anisotropic_diffusion(y, tv_step, kappa=float(pm_kappa))
    else:
        return laplacian_tv_diffusion(
            y,
            tv_step,
            kernel=kernel,
            density_gated=density_gated_diffusion,
            diffusion_dense_threshold=effective_diff_thresh,
            diffusion_gate_beta=effective_diff_beta,
        )

