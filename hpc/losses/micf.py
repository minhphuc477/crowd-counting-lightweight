"""Losses and Discrete Field Operations for Monotonic Integral Count Field (MICF-v2).

Key components:
- discrete_mixed_difference: Inverts cumulative field C -> Y via Delta_xy C.
- cell_counts_to_cumulative_field: Computes cumulative prefix sums Y -> C.
- MICFLoss: Field loss on C + measure validity penalty on ReLU(-Delta_xy C) + boundary count loss.
- IntegralLossOnLocalCount: Loss on P(Y_hat) vs P(Y) for Baseline B2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def discrete_mixed_difference(c: torch.Tensor) -> torch.Tensor:
    """Compute discrete mixed difference Delta_{xy} C = C_{i,j} - C_{i-1,j} - C_{i,j-1} + C_{i-1,j-1}.

    Exact inverse operator D = T^{-1} recovering discrete cell counts Y from cumulative field C.
    Preserves exact spatial shape [B, 1, H, W] (or [H, W] if 2D input) using zero boundary conditions.
    """
    orig_2d = (c.ndim == 2)
    if orig_2d:
        c = c.unsqueeze(0).unsqueeze(0)
    elif c.ndim == 3:
        c = c.unsqueeze(1)
    c_pad = F.pad(c.float(), (1, 0, 1, 0), mode="constant", value=0.0)
    y = (
        c_pad[:, :, 1:, 1:]
        - c_pad[:, :, :-1, 1:]
        - c_pad[:, :, 1:, :-1]
        + c_pad[:, :, :-1, :-1]
    )
    if orig_2d:
        return y.squeeze(0).squeeze(0)
    return y


def cell_counts_to_cumulative_field(
    y: torch.Tensor,
    orientation: str = "TL",
) -> torch.Tensor:
    """Compute 2D cumulative count field C from discrete cell count map Y.

    Args:
        y: Cell count map of shape [B, 1, H, W], [B, H, W], or [H, W].
        orientation: Prefix origin corner: 'TL', 'TR', 'BL', 'BR'.
    Returns:
        Cumulative count field C of matching shape.
    """
    orig_2d = (y.ndim == 2)
    if orig_2d:
        y = y.unsqueeze(0).unsqueeze(0)
    elif y.ndim == 3:
        y = y.unsqueeze(1)
    y = y.float()

    if orientation == "TL":
        out = torch.cumsum(torch.cumsum(y, dim=-2), dim=-1)
    elif orientation == "TR":
        # Flip width, cumsum, flip back
        y_flip = torch.flip(y, dims=[-1])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        out = torch.flip(c_flip, dims=[-1])
    elif orientation == "BL":
        # Flip height, cumsum, flip back
        y_flip = torch.flip(y, dims=[-2])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        out = torch.flip(c_flip, dims=[-2])
    elif orientation == "BR":
        # Flip both, cumsum, flip back
        y_flip = torch.flip(y, dims=[-2, -1])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        out = torch.flip(c_flip, dims=[-2, -1])
    else:
        raise ValueError(f"Unknown orientation '{orientation}'; expected TL, TR, BL, or BR.")

    if orig_2d:
        return out.squeeze(0).squeeze(0)
    return out


def points_to_count_map(
    points_xy: Optional[Union[torch.Tensor, np.ndarray, list]],
    out_h: int,
    out_w: int,
    stride: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build exact 2D integer cell count map Y from 2D point annotations (Section 13).

    Zero Gaussian smoothing; each point increments the cell containing its floor-divided coordinates:
        Y[i, j] = # { n : floor((y_n + 0.5) / stride) == i, floor((x_n + 0.5) / stride) == j }
    Guarantees sum(Y) == N_points exactly.

    Args:
        points_xy: [N, 2] point coordinates (x, y).
        out_h: Output grid height.
        out_w: Output grid width.
        stride: Downsampling factor from image to count map.
        device: Torch device.
        dtype: Output tensor dtype.

    Returns:
        Exact cell count tensor of shape [out_h, out_w].
    """
    y = torch.zeros((out_h, out_w), device=device, dtype=dtype)
    if points_xy is None or len(points_xy) == 0:
        return y

    pts = torch.as_tensor(points_xy, device=device, dtype=torch.float32)
    gx = torch.floor((pts[:, 0] + 0.5) / float(stride)).long()
    gy = torch.floor((pts[:, 1] + 0.5) / float(stride)).long()

    valid = (gx >= 0) & (gx < out_w) & (gy >= 0) & (gy < out_h)
    gx = gx[valid]
    gy = gy[valid]

    if gx.numel() == 0:
        return y

    flat_idx = gy * out_w + gx
    y.view(-1).scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=dtype))
    return y


class MICFLoss(nn.Module):
    """Loss for Direct Cumulative Field Prediction (MICF-v2).

    L = L_field(C_hat, C) + lambda_valid * L_valid(Delta_xy C_hat) + lambda_local_recon * L_recon(Delta_xy C_hat, Y)

    - L_field: Loss on all prefix entries C(i, j). Includes the bottom-right corner C[-1,-1] = N_total.
    - L_valid: Measure validity penalty: mean ReLU(-Delta_xy C_hat). Enforces non-negative counting measure.
    - L_local_recon: Optional direct local count reconstruction loss on Y_hat = Delta_xy C_hat (Section 19 & 20).
    """

    def __init__(
        self,
        field_loss: str = "smooth_l1",
        lambda_valid: float = 1.0,
        lambda_local_recon: float = 0.0,
        beta_smooth: float = 1.0,
        normalize_by: str = "none",   # "none" | "total_count"
        norm_eps: float = 1.0,
    ) -> None:
        super().__init__()
        self.field_loss = field_loss.lower()
        self.lambda_valid = float(lambda_valid)
        self.lambda_local_recon = float(lambda_local_recon)
        self.beta_smooth = float(beta_smooth)
        self.normalize_by = normalize_by.lower()
        if self.normalize_by not in {"none", "total_count"}:
            raise ValueError(f"normalize_by must be 'none' or 'total_count', got {normalize_by}")
        self.norm_eps = float(norm_eps)

    def _field_nll(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.field_loss == "smooth_l1":
            return F.smooth_l1_loss(pred, target, beta=self.beta_smooth, reduction="none")
        elif self.field_loss == "l1":
            return F.l1_loss(pred, target, reduction="none")
        elif self.field_loss == "mse":
            return F.mse_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unsupported field loss: {self.field_loss}")

    def forward(
        self,
        pred_c: torch.Tensor,
        target_c: torch.Tensor,
        target_y: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """Forward loss computation.

        Args:
            pred_c: Predicted cumulative field [B, 1, H, W] or [B, H, W].
            target_c: Ground truth cumulative field [B, 1, H, W] or [B, H, W].
            target_y: Optional ground truth local count map [B, 1, H, W] for local reconstruction loss.
            return_components: Whether to return individual loss components.
        """
        if pred_c.ndim == 2:
            pred_c = pred_c.unsqueeze(0).unsqueeze(0)
        elif pred_c.ndim == 3:
            pred_c = pred_c.unsqueeze(1)
        if target_c.ndim == 2:
            target_c = target_c.unsqueeze(0).unsqueeze(0)
        elif target_c.ndim == 3:
            target_c = target_c.unsqueeze(1)

        # 1. Field loss across all prefix entries (includes corner = N_total)
        #    Optionally normalized by a single per-sample scalar
        #    (design doc sec.10: "one scalar shared by the entire crop/image,
        #    not a position-dependent transform").
        if self.normalize_by == "total_count":
            scale = target_c[..., -1, -1].clamp_min(self.norm_eps).view(-1, 1, 1, 1)
            field_loss = self._field_nll(pred_c / scale, target_c / scale).mean()
        else:
            field_loss = self._field_nll(pred_c, target_c).mean()

        # 2. Measure validity penalty: mean ReLU(-Delta_xy C_hat)
        y_recovered = discrete_mixed_difference(pred_c)
        invalid_violations = F.relu(-y_recovered)
        validity_loss = invalid_violations.mean()

        total_loss = field_loss + self.lambda_valid * validity_loss

        # 3. Optional local reconstruction loss (Section 19 & 20)
        recon_loss = torch.tensor(0.0, device=pred_c.device)
        if self.lambda_local_recon > 0:
            if target_y is None:
                target_y = discrete_mixed_difference(target_c)
            elif target_y.ndim == 2:
                target_y = target_y.unsqueeze(0).unsqueeze(0)
            elif target_y.ndim == 3:
                target_y = target_y.unsqueeze(1)
            recon_loss = F.smooth_l1_loss(y_recovered, target_y.float(), beta=self.beta_smooth)
            total_loss = total_loss + self.lambda_local_recon * recon_loss

        if return_components:
            # Count loss as diagnostic only (not added to total)
            pred_n = pred_c[..., -1, -1]
            target_n = target_c[..., -1, -1]
            count_loss = F.smooth_l1_loss(pred_n, target_n, beta=self.beta_smooth)
            components = {
                "field_loss": float(field_loss.item()),
                "validity_loss": float(validity_loss.item()),
                "recon_loss": float(recon_loss.item()),
                "count_loss": float(count_loss.item()),   # diagnostic only
                "total_loss": float(total_loss.item()),
                "violation_rate": float((y_recovered < 0).float().mean().item()),
            }
            return total_loss, components

        return total_loss


class IntegralLossOnLocalCount(nn.Module):
    """Loss for Baseline B2: Local Count Output Y_hat supervised under Cumulative/Integral Metric.

    Computes loss on P(Y_hat) vs P(Y) where P is the 2D prefix-sum operator.
    """

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        beta_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type.lower()
        self.beta_smooth = float(beta_smooth)

    def forward(
        self,
        pred_y: torch.Tensor,
        target_y: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: computes field loss on cumsum(pred_y) vs cumsum(target_y)."""
        if pred_y.ndim == 2:
            pred_y = pred_y.unsqueeze(0).unsqueeze(0)
        elif pred_y.ndim == 3:
            pred_y = pred_y.unsqueeze(1)
        if target_y.ndim == 2:
            target_y = target_y.unsqueeze(0).unsqueeze(0)
        elif target_y.ndim == 3:
            target_y = target_y.unsqueeze(1)

        pred_c = torch.cumsum(torch.cumsum(pred_y.float(), dim=-2), dim=-1)
        target_c = torch.cumsum(torch.cumsum(target_y.float(), dim=-2), dim=-1)

        if self.loss_type == "smooth_l1":
            return F.smooth_l1_loss(pred_c, target_c, beta=self.beta_smooth)
        elif self.loss_type == "l1":
            return F.l1_loss(pred_c, target_c)
        elif self.loss_type == "mse":
            return F.mse_loss(pred_c, target_c)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
