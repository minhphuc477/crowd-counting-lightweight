"""Micro-Spatial Allocation Simplex Carrier Head (MPC-Head).

Breaks the Nyquist-Shannon aliasing limit at Stride 4 by explicitly factorizing
the measure density field into:
1. Macro Capacity Map M(i, j) in R_+: Total crowd mass in the 4x4 px macro cell.
2. Micro Allocation Simplex pi(i, j) in Delta^3: Spatial probability distribution
   over the 4 sub-cells (2x2 grid at Stride 2) satisfying:
   sum_{u, v} pi_{i, j}(u, v) == 1.0.

Exact Mass Conservation Theorem:
    sum_{u=0}^1 sum_{v=0}^1 m0_fine(2i + u, 2j + v) == M(i, j) identically.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .necks.blocks import ConvGNAct


def _softplus_inverse(x: float | torch.Tensor, threshold: float = 20.0) -> float | torch.Tensor:
    if isinstance(x, torch.Tensor):
        return torch.where(x > threshold, x, torch.log(torch.expm1(x).clamp_min(1e-12)))
    if x > threshold:
        return x
    return math.log(max(math.expm1(x), 1e-12))


class MicroSpatialPolyphaseCarrierHead(nn.Module):
    """Micro-Spatial Allocation Simplex Carrier Head (MPC-Head)."""

    def __init__(
        self,
        width: int = 32,
        init_bias: float = -4.1422,
        subpixel_stride2: bool = False,
        floor_tau: float = 0.0,
        head_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.subpixel_stride2 = bool(subpixel_stride2)
        self.floor_tau = float(floor_tau)

        # Depthwise-separable refinement trunk
        blocks: list[nn.Module] = []
        for _ in range(max(1, int(head_blocks))):
            blocks.append(ConvGNAct(self.width, self.width, 3, groups=self.width))
            blocks.append(ConvGNAct(self.width, self.width, 1))
        self.trunk = nn.Sequential(*blocks)

        # 1. Macro Capacity Branch: Predicts scalar mass M_ij >= 0 per Stride 4 cell
        self.macro_conv = nn.Conv2d(self.width, 1, kernel_size=1)
        nn.init.normal_(self.macro_conv.weight, std=0.01)
        m0 = math.log1p(math.exp(float(init_bias)))
        eff_bias = _softplus_inverse(m0 + 0.5 * self.floor_tau) if self.floor_tau > 0.0 else init_bias
        nn.init.constant_(self.macro_conv.bias, float(eff_bias))

        # 2. Micro Allocation Simplex Branch: Predicts 4 logits for 2x2 sub-cells
        if self.subpixel_stride2:
            self.micro_conv = nn.Conv2d(self.width, 4, kernel_size=1)
            nn.init.zeros_(self.micro_conv.weight)
            nn.init.zeros_(self.micro_conv.bias)  # Uniform prior: pi_c = 1/4 = 0.25 initially
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        else:
            self.micro_conv = None
            self.pixel_shuffle = None

    def forward_logits(
        self,
        p4: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits tensor.

        Returns:
            macro_raw [B, 1, H, W] for Stride 4, or subpixel_raw [B, 4, H, W] if subpixel_stride2.
        """
        h = self.trunk(p4)
        z_macro = self.macro_conv(h)  # [B, 1, H, W]
        if not self.subpixel_stride2 or self.micro_conv is None:
            return z_macro

        z_micro = self.micro_conv(h)  # [B, 4, H, W]
        z_micro = z_micro - z_micro.mean(dim=1, keepdim=True)
        # Return composite tensor: z_macro broadcast with zero-mean z_micro for gradient tracking
        return z_macro + z_micro

    def forward(
        self,
        p4: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fine density and polyphase decomposition.

        Returns:
            y0: [B, 1, 2H, 2W] if subpixel_stride2 else [B, 1, H, W]
            macro_capacity: [B, 1, H, W]
            allocation_simplex: [B, 4, H, W] if subpixel_stride2 else [B, 1, H, W]
        """
        h = self.trunk(p4)
        z_macro = self.macro_conv(h)

        # 1. Macro Capacity M >= 0
        macro_cap = F.softplus(z_macro)
        if self.floor_tau > 0.0:
            tau = self.floor_tau
            macro_cap = torch.where(
                macro_cap > tau,
                macro_cap - 0.5 * tau,
                macro_cap.square() / (2.0 * tau),
            )

        # 2. Micro Allocation Simplex: pi in Delta^3 (sums to 1.0)
        if self.subpixel_stride2 and self.micro_conv is not None and self.pixel_shuffle is not None:
            z_micro = self.micro_conv(h)
            z_micro = z_micro - z_micro.mean(dim=1, keepdim=True)
            alloc_simplex = F.softmax(z_micro, dim=1)
            # 3. Exact Mass Conservation Reconstruction
            sub_cells = macro_cap * alloc_simplex  # [B, 4, H, W]
            y0 = self.pixel_shuffle(sub_cells)     # [B, 1, 2H, 2W]
        else:
            alloc_simplex = torch.ones_like(macro_cap)
            y0 = macro_cap                         # [B, 1, H, W]

        return y0, macro_cap, alloc_simplex

    def activate(
        self,
        z0: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Activate method matching FineMeasureHead interface."""
        if not self.subpixel_stride2:
            macro_cap = F.softplus(z0)
            if self.floor_tau > 0.0:
                tau = self.floor_tau
                macro_cap = torch.where(
                    macro_cap > tau,
                    macro_cap - 0.5 * tau,
                    macro_cap.square() / (2.0 * tau),
                )
            return macro_cap

        # When subpixel_stride2, z0 is [B, 4, H, W] or composite
        if z0.shape[1] == 4:
            z_macro = z0.mean(dim=1, keepdim=True)
            z_micro = z0 - z_macro
            macro_cap = F.softplus(z_macro)
            if self.floor_tau > 0.0:
                tau = self.floor_tau
                macro_cap = torch.where(
                    macro_cap > tau,
                    macro_cap - 0.5 * tau,
                    macro_cap.square() / (2.0 * tau),
                )
            alloc_simplex = F.softmax(z_micro, dim=1)
            sub_cells = macro_cap * alloc_simplex
            return F.pixel_shuffle(sub_cells, upscale_factor=2)
        return F.softplus(z0)
