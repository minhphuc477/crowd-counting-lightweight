"""Canonical Focal Inverse Distance Transform (FIDT) Loss.

Official mathematical formulation adapted from dk-liang/FIDTM:
"Focal Inverse Distance Transform Map for Crowd Counting and Localization" (Liang et al., IEEE TPAMI 2022).
Eliminates density saturation in dense crowds by generating sharp local maxima:
  I(p) = 1 / (1 + d(p)^(0.02 * d(p) + 0.75))
where d(p) is Euclidean distance from pixel p to nearest head point in pixel coordinates.

At every head center d(p) = 0 -> I(p) = 1.0 regardless of crowd density.
Crucially: ZERO normalization by N, preserving distinct separation below Rayleigh limit.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_canonical_fidt_target(
    h: int,
    w: int,
    points: torch.Tensor,
    stride: int = 4,
    device: Optional[torch.device] = None,
    block_size: int = 4096,
) -> torch.Tensor:
    r"""Generate canonical FIDT map with exact 1.0 peak at each head location.

    Args:
        h: Feature map height
        w: Feature map width
        points: Ground-truth points [N, 2] in pixel coordinates (x, y)
        stride: Stride of feature map relative to original image (default: 4)
        device: Torch compute device
        block_size: Chunk size for pairwise distance expansion to bound VRAM

    Returns:
        Target FIDT tensor of shape [H, W] with values in [0, 1].
    """
    if device is None:
        device = points.device if points is not None else torch.device("cpu")

    if points is None or points.numel() == 0:
        return torch.zeros((h, w), device=device, dtype=torch.float32)

    pts = points.to(device=device, dtype=torch.float32)  # [N, 2] in pixels
    n_pts = pts.shape[0]

    # Grid coordinates in pixel space
    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]
    m_total = grid_xy.shape[0]

    # Fast cuBLAS distance expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
    x_sq = (grid_xy ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
    y_sq = (pts ** 2).sum(dim=-1, keepdim=True)      # [N, 1]

    min_dist = torch.empty((m_total,), device=device, dtype=torch.float32)

    for m_start in range(0, m_total, block_size):
        m_end = min(m_start + block_size, m_total)
        sub_x = grid_xy[m_start:m_end]
        sub_x_sq = x_sq[m_start:m_end]
        # [m_chunk, N]
        d2_chunk = sub_x_sq + y_sq.t() - 2.0 * torch.mm(sub_x, pts.t())
        min_dist[m_start:m_end] = torch.sqrt(d2_chunk.clamp_min(0.0).min(dim=-1).values)

    # Official FIDTM power transformation: I(p) = 1 / (1 + d^(0.02 * d + 0.75))
    exp_term = 0.02 * min_dist + 0.75
    fidt_flat = 1.0 / (1.0 + torch.pow(min_dist, exp_term))
    # Threshold background noise below 1e-2 as in official repo
    fidt_flat = torch.where(fidt_flat < 1e-2, torch.zeros_like(fidt_flat), fidt_flat)

    return fidt_flat.view(h, w)


def canonical_fidt_loss(
    pred: torch.Tensor,
    points_list: Optional[List[torch.Tensor]],
    stride: int = 4,
    loss_mode: str = "mse",
) -> torch.Tensor:
    r"""Supervise carrier or density prediction using canonical FIDTM representation.

    Args:
        pred: Predicted continuous measure [B, 1, H, W] or [B, H, W]
        points_list: List of ground-truth head points [N_i, 2] in pixel units
        stride: Output stride (default: 4)
        loss_mode: 'mse' (canonical TPAMI paper) or 'smooth_l1'

    Returns:
        Scalar tensor computing unnormalized peak-preserving FIDT loss.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    b, _, h, w = pred.shape
    device = pred.device

    losses: List[torch.Tensor] = []

    for i in range(b):
        pred_i = pred[i, 0].float()
        pts_i = points_list[i] if points_list is not None and i < len(points_list) else None

        with torch.no_grad():
            target_i = generate_canonical_fidt_target(h, w, pts_i, stride=stride, device=device)

        if loss_mode == "mse":
            loss_i = F.mse_loss(pred_i, target_i)
        else:
            loss_i = F.smooth_l1_loss(pred_i, target_i, beta=0.1)

        losses.append(loss_i)

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean().float()
