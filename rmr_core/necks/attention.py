from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordinateAttention(nn.Module):
    """Coordinate Attention for density-map feature refinement (RMR-v8 Stage 3).

    Applies spatial-aware channel attention by separately encoding horizontal
    and vertical spatial context via strip pooling, then projecting to per-axis
    attention maps gating the input features element-wise.

    Architecture (width=32, reduction=4):
        1. H-pool: AdaptiveAvgPool2d((None, 1)) -> [B, C, H, 1]
        2. V-pool: AdaptiveAvgPool2d((1, None)) -> [B, C, 1, W]
        3. Concat on spatial dim (after transpose) -> [B, C, H+W, 1]
        4. Shared reduction conv: C -> C//reduction via 1x1 Conv + GN + SiLU
        5. Split into H-half and V-half
        6. Two parallel 1x1 Convs (C//reduction -> C) + Sigmoid -> attention gates
        7. Gate: out = x * gate_h.expand_as(x) * gate_v.expand_as(x)

    Parameter count for width=32, reduction=4 (8 mid channels):
        Shared conv: 32 * 8 * 1 + 8 (bias) = 264 + 8 = 272 (but bias=False here)
          -> Conv2d(32, 8, 1, bias=False): 256 params
          -> GroupNorm(1, 8): 16 params (weight + bias)
        H-attention conv: Conv2d(8, 32, 1, bias=False): 256 params
        V-attention conv: Conv2d(8, 32, 1, bias=False): 256 params
        Total: 256 + 16 + 256 + 256 = 784 params

    Note: GroupNorm instead of BatchNorm — BatchNorm is incompatible with
    batch_size=1 inference on variable-resolution images (eval mode).
    GroupNorm(1, channels) is equivalent to LayerNorm over spatial dims and
    works correctly at any batch size.
    """

    def __init__(self, channels: int = 32, reduction: int = 4) -> None:
        super().__init__()
        if channels % reduction != 0:
            raise ValueError(
                f"CoordinateAttention: channels ({channels}) must be divisible by reduction ({reduction})"
            )
        mid = channels // reduction
        self.channels = channels
        self.mid = mid

        # Shared projection: (concat of H-pool and V-pool) -> mid channels
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(1, mid),  # LayerNorm-equivalent, works at batch_size=1
            nn.SiLU(inplace=True),
        )

        # Per-axis attention projections: mid -> channels
        self.h_conv = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Strip pooling: pool along each axis
        x_h = F.adaptive_avg_pool2d(x, (H, 1))   # [B, C, H, 1]
        x_v = F.adaptive_avg_pool2d(x, (1, W))   # [B, C, 1, W]

        # Transpose x_v to match H-dimension for shared projection
        x_v_t = x_v.permute(0, 1, 3, 2)          # [B, C, W, 1]

        # Concatenate along spatial dimension and project
        x_cat = torch.cat([x_h, x_v_t], dim=2)   # [B, C, H+W, 1]
        z = self.shared_conv(x_cat)                # [B, mid, H+W, 1]

        # Split back into H and W halves
        z_h, z_v = z[:, :, :H, :], z[:, :, H:, :]  # [B, mid, H, 1], [B, mid, W, 1]

        # Per-axis attention gates
        gate_h = torch.sigmoid(self.h_conv(z_h))   # [B, C, H, 1]
        gate_v = torch.sigmoid(self.v_conv(z_v.permute(0, 1, 3, 2)))  # [B, C, 1, W]

        # Apply attention: broadcast H-gate and V-gate over spatial dims
        return x * gate_h * gate_v
