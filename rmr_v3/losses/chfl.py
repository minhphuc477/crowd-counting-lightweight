"""Canonical Characteristic Function Loss (ChfL) for Crowd Counting.

Official mathematical formulation adapted from wbshu/Crowd_Counting_in_the_Frequency_Domain:
"Crowd Counting in the Frequency Domain" (Shu et al., CVPR 2022).
Supervises continuous crowd measure distributions without Gaussian kernel blurring:
  Phi_gt(t)   = sum_j exp(i * t^T x_j) * exp(-0.5 * ||t||^2 * sigma^2)
  Phi_pred(t) = sum_u y_pred(u) * exp(i * t^T u)

At t = 0: Phi_gt(0) = N and Phi_pred(0) = sum(y_pred) = N_pred (exact global count preservation).
High frequencies decay smoothly via Gaussian bandwidth damping, eliminating phase noise.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_CANONICAL_CHFL_CACHE: Dict[Tuple[int, int, int, float, int, float, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _build_canonical_chfl_templates(
    h: int,
    w: int,
    stride: int,
    chf_tik: float,
    chf_step: int,
    bandwidth: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and cache deterministic 2D spatial-frequency grid templates."""
    key = (h, w, stride, float(chf_tik), int(chf_step), float(bandwidth), str(device))
    if key not in _CANONICAL_CHFL_CACHE:
        # Spatial grid in pixel coordinates matching image plane
        y_axis = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
        x_axis = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
        grid_y, grid_x = torch.meshgrid(y_axis, x_axis, indexing="ij")
        # [2, M]
        sample_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)

        # 2D Frequency plane in [-chf_step * chf_tik, chf_step * chf_tik]^2
        steps = torch.arange(-chf_step, chf_step, device=device, dtype=torch.float32) * float(chf_tik)
        fy, fx = torch.meshgrid(steps, steps, indexing="ij")
        freq_grid = torch.stack([fx.flatten(), fy.flatten()], dim=-1)  # [K, 2]
        k_total = freq_grid.shape[0]

        # Spatial-frequency angles: omega^T * u -> [K, M]
        angle_pred = torch.matmul(freq_grid, sample_coords)
        cos_pred = torch.cos(angle_pred)
        sin_pred = torch.sin(angle_pred)

        # Gaussian frequency damping envelope: exp(-0.5 * ||omega||^2 * bandwidth^2) -> [K]
        omega_sq = (freq_grid ** 2).sum(dim=-1)
        freq_damping = torch.exp(-0.5 * omega_sq * (bandwidth ** 2))

        _CANONICAL_CHFL_CACHE[key] = (cos_pred, sin_pred, freq_grid, freq_damping)

    return _CANONICAL_CHFL_CACHE[key]


def canonical_chfl_loss(
    pred: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    chf_tik: float = 0.01,
    chf_step: int = 16,
    bandwidth: float = 8.0,
    stride: int = 4,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Compute canonical Characteristic Function Loss without uncalibrated normalization.

    Args:
        pred: Predicted density map [B, 1, H, W] or [B, H, W]
        points_list: List of ground-truth head coordinates [N_i, 2] in pixel units
        chf_tik: Sampling interval in frequency plane (default: 0.01)
        chf_step: Number of frequency steps in each direction (default: 16 -> 32x32 = 1024 frequencies)
        bandwidth: Spatial Gaussian smoothing bandwidth (default: 8.0 pixels)
        stride: Network downsampling stride (default: 4)
        eps: Small positive constant for numerical safety

    Returns:
        Scalar loss tensor enforcing both spatial distribution and global count conservation.
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    b, _, h, w = pred.shape
    device = pred.device

    cos_pred, sin_pred, freq_grid, freq_damping = _build_canonical_chfl_templates(
        h, w, stride, chf_tik, chf_step, bandwidth, device
    )

    # Flatten spatial predictions without dividing by total sum: [B, M]
    pred_flat = pred.flatten(2).float().clamp_min(0.0)  # [B, 1, M] -> [B, M]
    if pred_flat.ndim == 3:
        pred_flat = pred_flat.squeeze(1)

    # Predicted characteristic function: Phi_pred(t) = sum_u y(u) * exp(i * t^T u)
    # [B, M] @ [M, K] -> [B, K]
    phi_pred_real = torch.matmul(pred_flat, cos_pred.t())
    phi_pred_imag = torch.matmul(pred_flat, sin_pred.t())

    losses: list[torch.Tensor] = []

    for i in range(b):
        pts = points_list[i] if points_list is not None and i < len(points_list) else None

        if pts is None or pts.numel() == 0:
            # Ground truth is 0: target characteristic function is identically zero
            diff_sq = phi_pred_real[i].square() + phi_pred_imag[i].square()
            loss_i = torch.sqrt(diff_sq + eps).mean()
            losses.append(loss_i)
            continue

        pts_pix = pts.to(device=device, dtype=torch.float32)  # [N, 2]

        with torch.no_grad():
            # Ground truth characteristic function with Gaussian damping envelope
            # freq_grid [K, 2] @ pts_pix.t() [2, N] -> [K, N]
            angle_gt = torch.matmul(freq_grid, pts_pix.t())
            # Sum over all N points without dividing by N: Phi_gt(0) == N!
            cos_gt = torch.cos(angle_gt).sum(dim=-1)  # [K]
            sin_gt = torch.sin(angle_gt).sum(dim=-1)  # [K]
            phi_gt_real = cos_gt * freq_damping       # [K]
            phi_gt_imag = sin_gt * freq_damping       # [K]

        diff_real = phi_pred_real[i] - phi_gt_real
        diff_imag = phi_pred_imag[i] - phi_gt_imag
        diff_mag = torch.sqrt(diff_real.square() + diff_imag.square() + eps)
        losses.append(diff_mag.mean())

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean().float()
