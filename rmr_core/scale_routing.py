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
        perspective_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.temperature = float(max(temperature, 0.1))
        self.perspective_bias = bool(perspective_bias)

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

        # Zero-initialization: ensures exact uniform scale prior Softmax(0, 0, ...) = (1/K, 1/K, ...)
        # at step 0 (satisfying Theorem 1). At step 0, pw receives active gradients; at step 1+,
        # non-zero pw weights seamlessly unmask active gradient flow to the upstream depthwise conv and GroupNorm.
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


class FactorizedRoutingHead(nn.Module):
    """Decoupled 2D Spatial Scale and Aspect Ratio Routing Head for RMR (RMR-v19).

    Decouples the observation dictionary into:
      1. Marginal Scale Distribution pi_scale(s | u) in Delta^{S-1} over S strictly
         monotonic isotropic scales (e.g. S=3: [32, 64, 128] px).
      2. Aspect Ratio Distribution pi_aspect(rho | u) in Delta^{A-1} over A aspect ratios
         (e.g. A=2: [1:1 (square), 2:1 (vertical rectangle)]).

    Joint probability distribution for the 4-window dictionary [32x32, 64x64, 64x32, 128x128]:
      - pi_0 = pi_scale[0]                       (32x32 square)
      - pi_1 = pi_scale[1] * pi_aspect[0]        (64x64 square)
      - pi_2 = pi_scale[1] * pi_aspect[1]        (64x32 vertical rectangle)
      - pi_3 = pi_scale[2]                       (128x128 square)

    Properties:
      - Exact Partition of Unity: sum_{k=0}^3 pi_k(u) = 1.0 identically.
      - Marginal Scale Conservation: pi_1 + pi_2 = pi_scale[1], mathematically preventing
        scale starvation of moderate crowds.
      - Strict Monotonicity: physical_scale_alignment_loss operates exclusively on pi_scale
        whose areas [1024, 4096, 16384] are strictly monotonic.
      - Perspective Modulation: pi_aspect is modulated by normalized vertical elevation
        v = y/H in [-0.5, 0.5] via learnable persp_weight_aspect.

    Parameter budget:
      - Shared Depthwise 3x3: in_channels * 1 * 9 + in_channels = 320 params.
      - Shared GroupNorm(8, in_channels): 2 * in_channels = 64 params.
      - Pointwise Conv for Scale (S=3): in_channels * 3 + 3 = 99 params.
      - Pointwise Conv for Aspect (A=2): in_channels * 2 + 2 = 66 params.
      - Perspective bias for Aspect: 2 params.
      Total trainable parameters: 551 params (< 600 budget, +68 over standard router).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        num_aspects: int = 2,
        temperature: float = 1.0,
        perspective_bias: bool = True,
        num_aspect_ratios: int | None = None,
    ) -> None:
        super().__init__()
        if num_aspect_ratios is not None:
            num_aspects = num_aspect_ratios
        self.num_scales = int(num_scales)
        self.num_aspects = int(num_aspects)
        if self.num_scales != 3 or self.num_aspects != 2:
            raise ValueError(
                f"FactorizedRoutingHead currently supports exactly 3 marginal scales and 2 aspect ratios, "
                f"got num_scales={self.num_scales}, num_aspects={self.num_aspects}"
            )
        self.temperature = float(max(temperature, 0.1))
        self.perspective_bias = bool(perspective_bias)

        # Shared feature extraction carrier
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

        # Branch 1: Marginal Scale Head (S=3: 32, 64, 128)
        self.pw_scale = nn.Conv2d(
            in_channels,
            self.num_scales,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.pw_scale.weight)
        nn.init.zeros_(self.pw_scale.bias)

        # Branch 2: Aspect Ratio Head (A=2: 1:1 square, 2:1 vertical rectangle)
        self.pw_aspect = nn.Conv2d(
            in_channels,
            self.num_aspects,
            kernel_size=1,
            bias=True,
        )
        nn.init.zeros_(self.pw_aspect.weight)
        nn.init.zeros_(self.pw_aspect.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Feature map [B, C, H, W] from carrier neck.

        Returns:
            joint_pi:  Joint probability map [B, 4, H, W] summing to 1.0 across dim 1.
            pi_scale:  Marginal scale probability map [B, 3, H, W] summing to 1.0 across dim 1.
            pi_aspect: Aspect ratio probability map [B, 2, H, W] summing to 1.0 across dim 1.
        """
        feats = self.act(self.norm(self.dw(x)))
        logits_scale = self.pw_scale(feats)   # [B, 3, H, W]
        logits_aspect = self.pw_aspect(feats) # [B, 2, H, W]

        temp = float(self.temperature)
        pi_scale = F.softmax(logits_scale / temp, dim=1)   # [B, 3, H, W]
        pi_aspect = F.softmax(logits_aspect / temp, dim=1) # [B, 2, H, W]

        # Construct 4-window joint probability tensor:
        # Scale 0: 32x32 square -> pi_scale[:, 0:1]
        # Scale 1: 64x64 square -> pi_scale[:, 1:2] * pi_aspect[:, 0:1]
        # Scale 1: 64x32 vertical -> pi_scale[:, 1:2] * pi_aspect[:, 1:2]
        # Scale 2: 128x128 square -> pi_scale[:, 2:3]
        pi_0 = pi_scale[:, 0:1]
        pi_1 = pi_scale[:, 1:2] * pi_aspect[:, 0:1]
        pi_2 = pi_scale[:, 1:2] * pi_aspect[:, 1:2]
        pi_3 = pi_scale[:, 2:3]

        joint_pi = torch.cat([pi_0, pi_1, pi_2, pi_3], dim=1)  # [B, 4, H, W]
        return joint_pi, pi_scale, pi_aspect

