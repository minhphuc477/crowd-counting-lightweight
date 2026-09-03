"""Directional Integral Context Module for MICF-v2.

Implements the full 4-orientation normalized prefix feature context from MICF design doc
sections 12, 15, and 16 (IntegralContextBlock).

For each location (i,j), provides 4 normalized prefix averages:
    F_bar^TL_{ij} = (1/((i+1)(j+1))) * sum_{a<=i, b<=j} F_{ab}   (top-left prefix)
    F_bar^TR_{ij} = (1/((i+1)(W-j))) * sum_{a<=i, b>=j} F_{ab}   (top-right prefix)
    F_bar^BL_{ij} = (1/((H-i)(j+1))) * sum_{a>=i, b<=j} F_{ab}   (bottom-left prefix)
    F_bar^BR_{ij} = (1/((H-i)(W-j))) * sum_{a>=i, b>=j} F_{ab}   (bottom-right prefix)

Fused via: 1x1 reduce(5C->C) + dw 3x3 refine + 1x1 project(C->C) + residual.
Zero parameters for the pooling operators. O(HW) parallel GPU scan execution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _prefix_area_tl(h: int, w: int, device, dtype) -> torch.Tensor:
    """Area (i+1)(j+1) for TL prefix, shape [H, W]."""
    i = torch.arange(1, h + 1, device=device, dtype=dtype)  # [H]
    j = torch.arange(1, w + 1, device=device, dtype=dtype)  # [W]
    return i[:, None] * j[None, :]  # [H, W]


def normalized_integral_tl(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized TL prefix average: F_bar^TL_{ij} = sum_{a<=i,b<=j} F_{ab} / ((i+1)(j+1))."""
    h, w = x.shape[-2:]
    c = x.cumsum(-2).cumsum(-1)
    area = _prefix_area_tl(h, w, x.device, x.dtype)  # [H, W]
    return c / (area + eps)


def normalized_integral_tr(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized TR prefix average by flip-TL-flip."""
    xr = torch.flip(x, [-1])
    yr = normalized_integral_tl(xr, eps)
    return torch.flip(yr, [-1])


def normalized_integral_bl(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized BL prefix average by flip-TL-flip."""
    xb = torch.flip(x, [-2])
    yb = normalized_integral_tl(xb, eps)
    return torch.flip(yb, [-2])


def normalized_integral_br(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized BR prefix average by flip-TL-flip."""
    xbr = torch.flip(x, [-2, -1])
    ybr = normalized_integral_tl(xbr, eps)
    return torch.flip(ybr, [-2, -1])


class DirectionalIntegralContext(nn.Module):
    """Full 4-direction integral context block (design doc sections 12, 15, 16).

    Fuses local feature F with 4 normalized directional prefix averages
    (TL, TR, BL, BR) via a lightweight learnable projection.

    Architecture (from IntegralContextBlock in design doc):
        [F, F_bar^TL, F_bar^TR, F_bar^BL, F_bar^BR]  (5*C concat)
            -> 1x1 Conv(5C -> C) + BN + SiLU   (reduce)
            -> dw 3x3 Conv(C -> C) + BN + SiLU (refine)
            -> 1x1 Conv(C -> C)                 (project)
            + residual(x)
    """

    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual

        in_channels = channels * 5  # F + TL + TR + BL + BR

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.dw = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map of shape [B, C, H, W].
        Returns:
            Fused feature map of shape [B, C, H, W].
        """
        tl = normalized_integral_tl(x)
        tr = normalized_integral_tr(x)
        bl = normalized_integral_bl(x)
        br = normalized_integral_br(x)

        z = torch.cat([x, tl, tr, bl, br], dim=1)  # [B, 5C, H, W]
        z = self.reduce(z)
        z = self.dw(z)
        z = self.project(z)
        z = self.dropout(z)

        if self.use_residual:
            return x + z
        return z


class AxialIntegralContext(nn.Module):
    """Axial Integral Context Block (Section 31 of design doc).

    A cheaper alternative to the full 4-directional 2D context:
    computes horizontal prefix average R and vertical prefix average V:
        R_{ij} = (1 / (j + 1)) * sum_{b <= j} F_{ib}  (horizontal scan)
        V_{ij} = (1 / (i + 1)) * sum_{a <= i} F_{aj}  (vertical scan)

    Fused via: 1x1 Conv(3C -> C) + DW-Conv3x3 + 1x1 Conv(C -> C) + residual.
    Requires only 3C concatenation instead of 5C.
    """

    def __init__(
        self,
        channels: int,
        use_residual: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.use_residual = use_residual

        in_channels = channels * 3  # F + R + V

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1, groups=channels, bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        i_coords = torch.arange(1, h + 1, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        j_coords = torch.arange(1, w + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, w)

        # Horizontal prefix average: R
        r = x.cumsum(-1) / j_coords
        # Vertical prefix average: V
        v = x.cumsum(-2) / i_coords

        z = torch.cat([x, r, v], dim=1)  # [B, 3C, H, W]
        z = self.reduce(z)
        z = self.dw(z)
        z = self.project(z)
        z = self.dropout(z)

        if self.use_residual:
            return x + z
        return z

