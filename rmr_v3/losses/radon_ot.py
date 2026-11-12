"""Radon Sliced Wasserstein (Radon-OT) Loss for Continuous Measure Recovery.

Mathematical formulation:
Let mu = sum_{j=1}^N delta_{x_j} be the true atomic Radon measure, and y in L^1_+(Omega)
be the continuous reconstructed measure.

By the Fourier Slice Theorem, projecting a 2D measure along canonical projection angles
theta in [0, pi) yields 1D measures R_theta[mu] and R_theta[y].
In 1D, the Wasserstein-1 (Earth Mover's Distance) between probability distributions
has an exact, closed-form analytical solution:
    W_1(P, Q) = int |F_P(s) - F_Q(s)| ds
where F_P and F_Q are the 1D Cumulative Distribution Functions (CDFs).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

_RADON_GRID_CACHE: dict[tuple, tuple[torch.Tensor, ...]] = {}


def _get_radon_projection_grid(
    h: int, w: int, num_projections: int, device: torch.device, eps: float = 1e-6
) -> tuple[torch.Tensor, ...]:
    num_bins = int(max(h, w) * 1.5)
    key = (h, w, num_projections, str(device), float(eps))
    cached = _RADON_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    angles = torch.linspace(0.0, math.pi * (1.0 - 1.0 / float(num_projections)), num_projections, device=device)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    y_coords = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    x_coords = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    gx_flat = grid_x.flatten()
    gy_flat = grid_y.flatten()

    s_grid = cos_a.unsqueeze(1) * gx_flat.unsqueeze(0) + sin_a.unsqueeze(1) * gy_flat.unsqueeze(0)
    c_proj = torch.stack(
        [
            torch.zeros_like(cos_a),
            float(w) * cos_a,
            float(h) * sin_a,
            float(w) * cos_a + float(h) * sin_a,
        ],
        dim=1,
    )
    s_min = c_proj.min(dim=1, keepdim=True).values
    s_max = c_proj.max(dim=1, keepdim=True).values
    denom = s_max - s_min + eps
    s_norm = (s_grid - s_min) / denom * float(num_bins - 1)
    s0 = s_norm.long().clamp(0, num_bins - 1)
    s1 = (s0 + 1).clamp(0, num_bins - 1)
    alpha = s_norm - s0.float()
    one_m_alpha = 1.0 - alpha

    entry = (cos_a, sin_a, s_min, denom, s0, s1, alpha, one_m_alpha)
    if len(_RADON_GRID_CACHE) >= 16:
        _RADON_GRID_CACHE.pop(next(iter(_RADON_GRID_CACHE)))
    _RADON_GRID_CACHE[key] = entry
    return entry


def radon_sliced_wasserstein_loss(
    pred: torch.Tensor,
    points_list: Optional[List[torch.Tensor]],
    stride: int = 4,
    num_projections: int = 4,
    norm_ref: float = 100.0,
    smooth_beta: float = 0.01,
    scale_by_length: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute exact closed-form Sliced Wasserstein distance between normalized density and points."""
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    b, _, h, w = pred.shape
    device = pred.device
    num_bins = int(max(h, w) * 1.5)

    valid_indices = [
        i
        for i in range(b)
        if points_list is not None and i < len(points_list) and points_list[i] is not None and points_list[i].numel() > 0
    ]
    if not valid_indices:
        return pred.sum() * 0.0

    cos_a, sin_a, s_min, denom, s0, s1, alpha, one_m_alpha = _get_radon_projection_grid(
        h, w, int(num_projections), device, eps=float(eps)
    )

    y_valid = torch.relu(pred[valid_indices, 0].float()).flatten(1)
    probs = y_valid / y_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    bv = len(valid_indices)
    m_cells = s0.shape[1]

    p_exp = probs.unsqueeze(1)
    s0_exp = s0.unsqueeze(0).expand(bv, num_projections, m_cells)
    s1_exp = s1.unsqueeze(0).expand(bv, num_projections, m_cells)

    p1d = torch.zeros((bv, num_projections, num_bins), device=device, dtype=torch.float32)
    p1d.scatter_add_(2, s0_exp, p_exp * one_m_alpha.unsqueeze(0))
    p1d.scatter_add_(2, s1_exp, p_exp * alpha.unsqueeze(0))
    f_p = torch.cumsum(p1d, dim=2)

    q1d = torch.zeros((bv, num_projections, num_bins), device=device, dtype=torch.float32)
    for b_idx, orig_i in enumerate(valid_indices):
        pts_feat = points_list[orig_i].to(device=device, dtype=torch.float32, non_blocking=True) / float(stride)
        n = pts_feat.shape[0]
        s_pts = cos_a.unsqueeze(1) * pts_feat[:, 0].unsqueeze(0) + sin_a.unsqueeze(1) * pts_feat[:, 1].unsqueeze(0)
        s_pts_norm = ((s_pts - s_min) / denom * float(num_bins - 1)).clamp(0.0, float(num_bins - 1))
        ps0 = s_pts_norm.long().clamp(0, num_bins - 1)
        ps1 = (ps0 + 1).clamp(0, num_bins - 1)
        p_alpha = s_pts_norm - ps0.float()
        w_pt = 1.0 / float(n)
        q1d[b_idx].scatter_add_(1, ps0, (1.0 - p_alpha) * w_pt)
        q1d[b_idx].scatter_add_(1, ps1, p_alpha * w_pt)
    f_q = torch.cumsum(q1d, dim=2)

    abs_d = (f_p - f_q).abs()
    huber = torch.where(abs_d < smooth_beta, 0.5 * abs_d.square() / smooth_beta, abs_d - 0.5 * smooth_beta)
    step_size = (denom / float(num_bins)).view(1, num_projections) if scale_by_length else (1.0 / float(num_bins))
    return (huber.sum(dim=2) * step_size).mean().float()
