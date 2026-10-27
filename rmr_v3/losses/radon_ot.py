"""Radon Sliced Wasserstein (Radon-OT) Loss for Continuous Measure Recovery.

Mathematical formulation:
Let mu = sum_{j=1}^N delta_{x_j} be the true atomic Radon measure, and y_hat in L^1_+(Omega)
be the continuous reconstructed measure.

By the Fourier Slice Theorem, projecting a 2D measure along projection angles theta in [0, pi)
yields 1D measures R_theta[mu] and R_theta[y_hat].
In 1D, the Wasserstein-1 (Earth Mover's Distance) has an exact, closed-form analytical solution:
    W_1(P, Q) = int |F_P(s) - F_Q(s)| ds
where F_P and F_Q are the 1D Cumulative Distribution Functions (CDFs).

Key mathematical properties:
1. Exact total mass preservation: |F_P(infty) - F_Q(infty)| = |N_hat - N| (exact count error at boundary).
2. Linear spatial transport force: W_1(delta_{x+s}, delta_x) = s (no vanishing gradient, zero local minima).
3. Exact 0 parameters: 100% standalone, pure PyTorch autograd, no Sinkhorn matrix scaling iterations.
4. Poisson variance-stabilizing normalization: scaled by sqrt(max(N, 1) / N_ref).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch


def radon_sliced_wasserstein_loss(
    pred: torch.Tensor,
    points_list: Optional[List[torch.Tensor]],
    stride: int = 4,
    num_projections: int = 4,
    norm_ref: float = 100.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute exact closed-form Sliced Wasserstein distance between predicted density and points.

    Args:
        pred: Predicted density tensor of shape [B, 1, H, W] or [B, H, W] (non-negative).
        points_list: List of length B with ground-truth head coordinates [N_i, 2] in image pixels.
        stride: Network spatial downsampling stride (e.g. 2 or 4).
        num_projections: Number of canonical 1D projection angles theta in [0, pi). Default: 4.
        norm_ref: Reference count for Poisson studentization. Default: 100.0.
        eps: Small numerical constant.

    Returns:
        Scalar loss tensor enforcing both spatial transport and global count conservation.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    b, _, h, w = pred.shape
    device = pred.device

    # Canonical projection angles in [0, pi)
    angles = torch.linspace(0.0, math.pi * (1.0 - 1.0 / float(num_projections)), num_projections, device=device)
    cos_a = torch.cos(angles)  # [K]
    sin_a = torch.sin(angles)  # [K]

    # Coordinate grid of feature map
    y_coords = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    x_coords = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")  # [H, W]

    num_bins = int(max(h, w) * 1.5)
    losses: List[torch.Tensor] = []

    for i in range(b):
        y_i = pred[i, 0].clamp_min(0.0)  # [H, W]
        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        n = pts.shape[0] if pts is not None and pts.numel() > 0 else 0

        swd_i = torch.zeros((), device=device, dtype=torch.float32)

        for k in range(num_projections):
            ca, sa = cos_a[k], sin_a[k]

            # 1. Project 2D grid: s = x * cos(a) + y * sin(a)
            s_grid = grid_x * ca + grid_y * sa
            s_min, s_max = s_grid.min(), s_grid.max()
            denom = s_max - s_min + eps
            s_norm = (s_grid - s_min) / denom * float(num_bins - 1)

            # Continuous linear splatting of predicted density onto 1D projection axis
            s0 = s_norm.long().clamp(0, num_bins - 1)
            s1 = (s0 + 1).clamp(0, num_bins - 1)
            alpha = s_norm - s0.float()

            p1d = torch.zeros(num_bins, device=device, dtype=torch.float32)
            p1d.scatter_add_(0, s0.flatten(), (y_i * (1.0 - alpha)).flatten())
            p1d.scatter_add_(0, s1.flatten(), (y_i * alpha).flatten())
            f_p = torch.cumsum(p1d, dim=0)

            if n > 0 and pts is not None:
                # 2. Project ground truth points onto the same 1D axis
                pts_feat = pts.to(device=device, dtype=torch.float32) / float(stride)
                s_pts = pts_feat[:, 0] * ca + pts_feat[:, 1] * sa
                s_pts_norm = ((s_pts - s_min) / denom * float(num_bins - 1)).clamp(0.0, float(num_bins - 1))

                ps0 = s_pts_norm.long().clamp(0, num_bins - 1)
                ps1 = (ps0 + 1).clamp(0, num_bins - 1)
                p_alpha = s_pts_norm - ps0.float()

                q1d = torch.zeros(num_bins, device=device, dtype=torch.float32)
                q1d.scatter_add_(0, ps0, 1.0 - p_alpha)
                q1d.scatter_add_(0, ps1, p_alpha)
                f_q = torch.cumsum(q1d, dim=0)

                # 1D Wasserstein distance is the L1 norm between cumulative distribution functions
                w1_k = torch.abs(f_p - f_q).mean()
            else:
                # Target is zero points: distance is full mass CDF
                w1_k = f_p.abs().mean()

            swd_i = swd_i + w1_k

        swd_i = swd_i / float(num_projections)
        # Poisson variance-stabilizing Studentization: variance of error sum scales with sqrt(N)
        scale_denom = math.sqrt(float(max(n, 1)) / max(float(norm_ref), 1.0))
        losses.append(swd_i / max(scale_denom, 1e-4))

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean().float()
