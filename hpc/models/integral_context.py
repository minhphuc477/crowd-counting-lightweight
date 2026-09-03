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


def _apply_blockwise_integral(
    x: torch.Tensor,
    k: int,
    fn,
) -> torch.Tensor:
    """Apply a prefix/integral operator independently inside KxK blocks,
    then reconstruct the original [B,C,H,W] layout.

    IMPORTANT:
    Only the integral operator is block-scoped.
    Learnable fusion / BatchNorm remain global.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

    B, C, H, W = x.shape

    if H % k != 0 or W % k != 0:
        raise ValueError(
            f"Feature grid ({H},{W}) must be divisible by finite_horizon={k}"
        )

    nh = H // k
    nw = W // k

    blocks = (
        x.view(B, C, nh, k, nw, k)
        .permute(0, 2, 4, 1, 3, 5)
        .contiguous()
        .view(B * nh * nw, C, k, k)
    )

    blocks = fn(blocks)

    out = (
        blocks.view(B, nh, nw, C, k, k)
        .permute(0, 3, 1, 4, 2, 5)
        .contiguous()
        .view(B, C, H, W)
    )

    return out


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

    def forward_finite_horizon(
        self,
        x: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """FH version of DirectionalIntegralContext.

        Prefix operators reset every KxK block, but all learnable
        fusion layers operate on the reconstructed full feature map.

        Therefore:
          - same Conv scope as global B5b
          - same BatchNorm scope as global B5b
          - same parameters
          - only integral horizon differs
        """
        tl = _apply_blockwise_integral(x, k, normalized_integral_tl)
        tr = _apply_blockwise_integral(x, k, normalized_integral_tr)
        bl = _apply_blockwise_integral(x, k, normalized_integral_bl)
        br = _apply_blockwise_integral(x, k, normalized_integral_br)

        # IMPORTANT:
        # Reassembled full HxW tensor enters learned fusion.
        z = torch.cat([x, tl, tr, bl, br], dim=1)

        z = self.reduce(z)
        z = self.dw(z)
        z = self.project(z)
        z = self.dropout(z)

        if self.use_residual:
            return x + z
        return z


def _axial_row_prefix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized row-prefix average (design doc sec.31, R_ij):
        R_{ij} = (1/(j+1)) * sum_{b<=j} F_{ib}
    Cumulative sum along the width axis only (independent per row).
    """
    w = x.shape[-1]
    r = x.cumsum(-1)
    denom = torch.arange(1, w + 1, device=x.device, dtype=x.dtype).view(1, 1, 1, -1)
    return r / (denom + eps)


def _axial_col_prefix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized column-prefix average (design doc sec.31, V_ij):
        V_{ij} = (1/(i+1)) * sum_{a<=i} F_{aj}
    Cumulative sum along the height axis only (independent per column).
    """
    h = x.shape[-2]
    v = x.cumsum(-2)
    denom = torch.arange(1, h + 1, device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
    return v / (denom + eps)


class AxialIntegralContext(nn.Module):
    """Cheaper alternative to DirectionalIntegralContext (design doc sec.31).

    Uses only 1D row-prefix (R) and column-prefix (V) averages instead of
    the full 4-orientation 2D prefix context, at roughly 3/5 the concat
    width and half the cumsum/flip work (1 axis each vs. 2 axes x 4 flips).

    Architecture:
        [F, R, V] (3*C concat)
            -> 1x1 Conv(3C -> C) + BN + SiLU   (reduce)
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
        r = _axial_row_prefix(x)
        v = _axial_col_prefix(x)

        z = torch.cat([x, r, v], dim=1)  # [B, 3C, H, W]
        z = self.reduce(z)
        z = self.dw(z)
        z = self.project(z)
        z = self.dropout(z)

        if self.use_residual:
            return x + z
        return z

