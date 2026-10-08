from __future__ import annotations

import torch
import torch.nn.functional as F
from .auxiliary import (
    count_harmonized_cell_loss,
    count_invariant_cell_loss,
    mass_weighted_cell_loss,
)
from rmr_core.losses import balanced_smooth_l1


def align_target_to_prediction(
    target_float: torch.Tensor,
    y: torch.Tensor,
    points: list[torch.Tensor] | None = None,
    stride: int = 2,
) -> torch.Tensor:
    """Rigorous mass-conserving spatial alignment between target and prediction lattices.

    Guarantees:
    1. If shapes match, identity return (zero cost).
    2. If points are provided and spatial shapes mismatch by >= 2x, exact re-rasterization at prediction stride.
    3. If points are not provided:
       - 2x upsampling (Stride 4 target -> Stride 2 pred): F.interpolate(mode='nearest') / 4.0
         (exact discrete mass conservation).
       - 2x downsampling (Stride 2 target -> Stride 4 pred): push_forward_stride2_to_stride4(target)
         (exact discrete mass conservation).
    4. For small boundary rounding differences (<= 2 px from integer division):
       symmetric/minimal edge padding or slicing, NEVER zero-padding 50-75% of the frame.
    """
    if target_float.shape[-2:] == y.shape[-2:]:
        return target_float

    b = target_float.shape[0]
    h_y, w_y = y.shape[-2:]
    h_t, w_t = target_float.shape[-2:]

    # Case 1: Point coordinates are available and resolution differs significantly (> 2 px)
    if points is not None and len(points) == b and (abs(h_y - h_t) > 2 or abs(w_y - w_t) > 2):
        from rmr_core.data import rasterize_points
        re_targets = []
        for i in range(b):
            pts_i = points[i]
            img_h = h_y * stride
            img_w = w_y * stride
            re_t = rasterize_points(pts_i, img_h, img_w, stride=stride, dtype=target_float.dtype)
            re_targets.append(re_t.to(device=target_float.device))
        return torch.stack(re_targets, dim=0)

    # Case 2: Target is Stride 4 (~half resolution of y) without points
    if (h_y >= 2 * h_t - 2) and (w_y >= 2 * w_t - 2):
        up = F.interpolate(target_float, size=(h_y, w_y), mode="nearest") / 4.0
        m_t = target_float.sum(dim=(-2, -1), keepdim=True)
        m_up = up.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return up * (m_t / m_up)

    # Case 3: Target is Stride 2 (~double resolution of y) without points
    if (h_t >= 2 * h_y - 2) and (w_t >= 2 * w_y - 2):
        from rmr_v3.model.dual_lattice import push_forward_stride2_to_stride4
        down = push_forward_stride2_to_stride4(target_float)
        if down.shape[-2:] != (h_y, w_y):
            down = down[..., :h_y, :w_y]
        m_t = target_float.sum(dim=(-2, -1), keepdim=True)
        m_down = down.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        return down * (m_t / m_down)

    # Case 4: Minor boundary rounding mismatch (<= 2 px from FPN integer division)
    dh, dw = h_y - h_t, w_y - w_t
    if dh > 0 or dw > 0:
        target_float = F.pad(target_float, (0, max(0, dw), 0, max(0, dh)))
    return target_float[..., :h_y, :w_y]


def compute_dual_lattice_losses(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    target_stride2: torch.Tensor,
    target_stride4: torch.Tensor,
    lambda_carrier_cell: float = 0.50,
    lambda_fine_cell: float = 0.25,
    cell_loss_mode: str = "mass_weighted",
    beta: float = 1.0,
    eps: float = 1e-3,
    alpha: float = 2.0,
    gamma: float = 1.25,
    fg_ratio: float = 0.67,
) -> dict[str, torch.Tensor]:
    """Supervise Stride 4 carrier with canonical v19 convex loss, and Stride 2 with sub-pixel loss.

    Prevents quadratic loss gradient starvation in dense regions by anchoring
    the carrier density on Stride 4, while allowing Stride 2 to resolve sparse heads.
    """
    if target_stride4.shape[-2:] != y_carrier.shape[-2:]:
        dh = y_carrier.shape[-2] - target_stride4.shape[-2]
        dw = y_carrier.shape[-1] - target_stride4.shape[-1]
        if dh > 0 or dw > 0:
            target_stride4 = F.pad(target_stride4, (0, max(0, dw), 0, max(0, dh)))
        target_stride4 = target_stride4[..., :y_carrier.shape[-2], :y_carrier.shape[-1]]

    if cell_loss_mode == "count_harmonized":
        loss_carrier = count_harmonized_cell_loss(
            y_carrier, target_stride4, beta=beta, eps=eps, gamma=gamma, fg_ratio=fg_ratio, stride=4
        )
        loss_fine = count_harmonized_cell_loss(
            y_fine, target_stride2, beta=beta, eps=eps, gamma=gamma, fg_ratio=fg_ratio, stride=2
        )
    elif cell_loss_mode == "mass_weighted":
        loss_carrier = mass_weighted_cell_loss(
            y_carrier, target_stride4, beta=beta, eps=eps, alpha=alpha, gamma=gamma, stride=4
        )
        loss_fine = mass_weighted_cell_loss(
            y_fine, target_stride2, beta=beta, eps=eps, alpha=alpha, gamma=gamma, stride=2
        )
    elif cell_loss_mode in ("count_invariant", "ci_cell"):
        loss_carrier = count_invariant_cell_loss(
            y_carrier, target_stride4, beta=beta, alpha=alpha, stride=4
        )
        loss_fine = count_invariant_cell_loss(
            y_fine, target_stride2, beta=beta, alpha=alpha, stride=2
        )
    else:
        loss_carrier = balanced_smooth_l1(y_carrier, target_stride4, beta=beta, stride=4)
        loss_fine = balanced_smooth_l1(y_fine, target_stride2, beta=beta, stride=2)

    return {
        "cell_carrier": loss_carrier,
        "cell_fine": loss_fine,
        "cell_combined": float(lambda_carrier_cell) * loss_carrier + float(lambda_fine_cell) * loss_fine,
    }
