"""Losses and Discrete Field Operations for Monotonic Integral Count Field (MICF-v2).

Key components:
- discrete_mixed_difference: Inverts cumulative field C -> Y via Delta_xy C.
- cell_counts_to_cumulative_field: Computes cumulative prefix sums Y -> C.
- MICFLoss: Field loss on C + measure validity penalty on ReLU(-Delta_xy C) + boundary count loss.
- IntegralLossOnLocalCount: Loss on P(Y_hat) vs P(Y) for Baseline B2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def discrete_mixed_difference(c: torch.Tensor) -> torch.Tensor:
    """Compute discrete mixed difference Delta_{xy} C = C_{i,j} - C_{i-1,j} - C_{i,j-1} + C_{i-1,j-1}.

    Exact inverse operator D = T^{-1} recovering discrete cell counts Y from cumulative field C.
    Preserves exact spatial shape [B, 1, H, W] using zero boundary conditions C_{0,j} = C_{i,0} = 0.
    """
    if c.ndim == 3:
        c = c.unsqueeze(1)
    c_pad = F.pad(c.float(), (1, 0, 1, 0), mode="constant", value=0.0)
    y = (
        c_pad[:, :, 1:, 1:]
        - c_pad[:, :, :-1, 1:]
        - c_pad[:, :, 1:, :-1]
        + c_pad[:, :, :-1, :-1]
    )
    return y


def cell_counts_to_cumulative_field(
    y: torch.Tensor,
    orientation: str = "TL",
) -> torch.Tensor:
    """Compute 2D cumulative count field C from discrete cell count map Y.

    Args:
        y: Cell count map of shape [B, 1, H, W] or [B, H, W].
        orientation: Prefix origin corner: 'TL', 'TR', 'BL', 'BR'.
    Returns:
        Cumulative count field C of same shape.
    """
    if y.ndim == 3:
        y = y.unsqueeze(1)
    y = y.float()

    if orientation == "TL":
        return torch.cumsum(torch.cumsum(y, dim=-2), dim=-1)
    elif orientation == "TR":
        # Flip width, cumsum, flip back
        y_flip = torch.flip(y, dims=[-1])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        return torch.flip(c_flip, dims=[-1])
    elif orientation == "BL":
        # Flip height, cumsum, flip back
        y_flip = torch.flip(y, dims=[-2])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        return torch.flip(c_flip, dims=[-2])
    elif orientation == "BR":
        # Flip both, cumsum, flip back
        y_flip = torch.flip(y, dims=[-2, -1])
        c_flip = torch.cumsum(torch.cumsum(y_flip, dim=-2), dim=-1)
        return torch.flip(c_flip, dims=[-2, -1])
    else:
        raise ValueError(f"Unknown orientation '{orientation}'; expected TL, TR, BL, or BR.")


class MICFLoss(nn.Module):
    """Loss for Direct Cumulative Field Prediction (MICF-v2).

    L = L_field(C_hat, C) + lambda_valid * L_valid(Delta_xy C_hat)

    The field loss already includes the corner element C[-1,-1] = N_total, so no
    separate count boundary term is needed (and adding one would double-weight the
    corner, biasing toward count accuracy over spatial field consistency).

    lambda_count is retained only as a diagnostic in return_components.
    """

    def __init__(
        self,
        field_loss: str = "smooth_l1",
        lambda_valid: float = 1.0,
        beta_smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.field_loss = field_loss.lower()
        self.lambda_valid = float(lambda_valid)
        self.beta_smooth = float(beta_smooth)

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
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, float]]:
        """Forward loss computation.

        Args:
            pred_c: Predicted cumulative field [B, 1, H, W].
            target_c: Ground truth cumulative field [B, 1, H, W].
        """
        if pred_c.ndim == 3:
            pred_c = pred_c.unsqueeze(1)
        if target_c.ndim == 3:
            target_c = target_c.unsqueeze(1)

        # 1. Field loss across all prefix entries (includes corner = N_total)
        field_loss = self._field_nll(pred_c, target_c).mean()

        # 2. Measure validity penalty: mean ReLU(-Delta_xy C_hat)
        y_recovered = discrete_mixed_difference(pred_c)
        invalid_violations = F.relu(-y_recovered)
        validity_loss = invalid_violations.mean()

        # Total: only field + validity (design doc section 20)
        total_loss = field_loss + self.lambda_valid * validity_loss

        if return_components:
            # Count loss as diagnostic only (not added to total)
            pred_n = pred_c[..., -1, -1]
            target_n = target_c[..., -1, -1]
            count_loss = F.smooth_l1_loss(pred_n, target_n, beta=self.beta_smooth)
            components = {
                "field_loss": float(field_loss.item()),
                "validity_loss": float(validity_loss.item()),
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
        if pred_y.ndim == 3:
            pred_y = pred_y.unsqueeze(1)
        if target_y.ndim == 3:
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
