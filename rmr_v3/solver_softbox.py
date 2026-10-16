"""Gaussian Soft-Box forward and adjoint operators for continuous RMR solver."""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

__all__ = [
    "make_gaussian_softbox_kernel",
    "gaussian_softbox_forward",
    "gaussian_softbox_adjoint_step",
]


def make_gaussian_softbox_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    """Create normalized 2D Gaussian kernel with shape (1, 1, K, K)."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * max(sigma, 1e-4) ** 2))
    g = g / g.sum()
    return (g.view(1, -1) * g.view(-1, 1)).view(1, 1, kernel_size, kernel_size)


def gaussian_softbox_forward(
    y: torch.Tensor,
    sigmas: tuple[float, ...] = (2.0, 4.0, 8.0),
) -> list[torch.Tensor]:
    """Forward observation operator using Gaussian Soft-Boxes G_sigma.

    Eliminates periodic sinc zeros in Fourier space: ker(A) = {0}.
    """
    orig_ndim = y.ndim
    y4d = y.unsqueeze(0).unsqueeze(0) if orig_ndim == 2 else (y.unsqueeze(0) if orig_ndim == 3 else y)
    results = []
    for s in sigmas:
        k_size = int(math.ceil(4.0 * s)) | 1
        k = make_gaussian_softbox_kernel(k_size, s).to(device=y.device, dtype=torch.float32)
        pad = k_size // 2
        y_pad = F.pad(y4d.float(), (pad, pad, pad, pad), mode="reflect")
        b_k = F.conv2d(y_pad, k)
        if orig_ndim == 2:
            results.append(b_k.squeeze(0).squeeze(0).to(dtype=y.dtype))
        elif orig_ndim == 3:
            results.append(b_k.squeeze(0).to(dtype=y.dtype))
        else:
            results.append(b_k.to(dtype=y.dtype))
    return results


def gaussian_softbox_adjoint_step(
    residual_list: list[torch.Tensor],
    scale_routing_weights: torch.Tensor | None = None,
    sigmas: tuple[float, ...] = (2.0, 4.0, 8.0),
) -> torch.Tensor:
    """Adjoint backprojection operator A* for Gaussian Soft-Boxes.

    Since 2D Gaussian convolution is self-adjoint (symmetric kernel),
    A*_k(r_k) = G_sigma_k * r_k. The step is: Delta = sum_k pi_k * (G_sigma_k * r_k).
    """
    accum = None
    for idx, (res, s) in enumerate(zip(residual_list, sigmas)):
        k_size = int(math.ceil(4.0 * s)) | 1
        k = make_gaussian_softbox_kernel(k_size, s).to(device=res.device, dtype=torch.float32)
        pad = k_size // 2
        r4d = res if res.ndim == 4 else (res.unsqueeze(0) if res.ndim == 3 else res.unsqueeze(0).unsqueeze(0))
        r_pad = F.pad(r4d.float(), (pad, pad, pad, pad), mode="reflect")
        adj_k = F.conv2d(r_pad, k)
        if scale_routing_weights is not None and scale_routing_weights.shape[1] > idx:
            pi_k = scale_routing_weights[:, idx:idx+1, :, :].float()
            adj_k = adj_k * pi_k
        accum = adj_k if accum is None else accum + adj_k
    return accum
