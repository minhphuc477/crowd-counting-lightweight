from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleRoutingHead(nn.Module):
    """Ultra-lightweight Spatial Scale Routing Head for RMR.

    Predicts a learned spatial partition of unity pi(x, y) in Delta^{K-1} over K observation
    scales (e.g. scales [32, 64, 128]), allowing the measure reconciliation solver to
    dynamically route regional evidence:
      - Coarse scales on empty/sparse background (suppressing phantom mass back-projection).
      - Fine scales on dense crowd clusters (preserving high-frequency head boundaries).

    Parameter budget:
      - Depthwise 3x3: in_channels * 1 * 9 + in_channels = 320 params.
      - GroupNorm(8, in_channels): 2 * in_channels = 64 params (batch-size independent).
      - Pointwise 1x1: in_channels * num_scales + num_scales = 99 params.
      Total trainable parameters: 483 (< 500 budget).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.temperature = float(max(temperature, 0.1))

        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(
            in_channels,
            self.num_scales,
            kernel_size=1,
            bias=True,
        )

        # Initialize to uniform scale prior: Softmax(0, 0, ...) = (1/K, 1/K, ...)
        # ensuring the model begins with exact isotropic multi-scale observation at step 0.
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W] from carrier neck.

        Returns:
            pi: Softmax probability map [B, num_scales, H, W] summing to 1.0 across dim 1.
        """
        feats = self.act(self.norm(self.dw(x)))
        logits = self.pw(feats)  # [B, num_scales, H, W]
        temp = float(self.temperature)
        pi = F.softmax(logits / temp, dim=1)
        return pi
