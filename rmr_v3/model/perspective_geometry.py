"""Dynamic Image-Adaptive Geometry (DiAG) & Scale Routing Modules for RMR-v3.

Provides:
1. DynamicCameraAnglePredictor (DCAP): Pools deepest semantic features P16 (GAP)
   and projects to global scale logit offsets delta_scale in R^K (zero-init).
2. DiAGScaleRoutingHead: 100% feature-driven multi-scale spatial partition of unity
   pi(x, y) in Delta^{K-1} combining local carrier convolution (P4) and global scene context.

Mathematical Principles:
- Strictly NO static coordinate grids (no torch.linspace, no fixed scanline bands).
- Scale preference is 100% data-driven and image-adaptive.
- Preserves the Universal Multi-Scale Observation Dictionary ([32, 64, 128] px, 1,235 boxes).
- Step 0 Identity Parity: Zero-initialized weights guarantee uniform distribution (1/K, 1/K, ...) at Step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicCameraAnglePredictor(nn.Module):
    """Dynamic Camera Angle & Scene Scale Context Predictor (DCAP).

    Predicts image-level global scale preference dynamically from deepest semantic features P16:
      - delta_scale in R^K: image-level global scale preference logit adjustments
        (e.g., distant vs. close-up scenes, overhead vs. oblique views).
      - scene_tilt in [0, 1]: optional scalar camera tilt intensity from image features.

    Parameter budget:
      - Global Average Pooling: 0 params
      - Linear(in_channels, 1 + num_scales): 32 * (1 + 3) + 4 = 132 params.
      - Zero-initialization: Step 0 Identity Parity (delta_scale = 0.0, scene_tilt = 0.5).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        use_vertical_gradient: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.use_vertical_gradient = bool(use_vertical_gradient)
        feat_dim = self.in_channels * 2 if self.use_vertical_gradient else self.in_channels
        self.proj = nn.Linear(feat_dim, 1 + self.num_scales, bias=True)
        # Step 0 Identity Parity: zero-init guarantees uniform, unbiased initialization
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            p16: [B, C, H_16, W_16] deepest semantic feature map.

        Returns:
            scene_tilt: [B, 1, 1, 1] camera tilt factor in [0, 1] derived from semantic content.
            delta_scale: [B, num_scales, 1, 1] global scale logit adjustment.
        """
        gap = p16.mean(dim=(-2, -1))  # [B, C]
        if self.use_vertical_gradient:
            h = p16.shape[-2]
            h_mid = max(1, h // 2)
            top = p16[..., :h_mid, :].mean(dim=(-2, -1))
            bot = p16[..., h_mid:, :].mean(dim=(-2, -1))
            v_diff = bot - top
            feat = torch.cat([gap, v_diff], dim=-1)
        else:
            feat = gap
        out = self.proj(feat)  # [B, 1 + num_scales]

        scene_tilt = torch.sigmoid(out[:, 0:1]).view(-1, 1, 1, 1)
        delta_scale = out[:, 1: 1 + self.num_scales].view(-1, self.num_scales, 1, 1)

        return scene_tilt, delta_scale


class DiAGScaleRoutingHead(nn.Module):
    """Dynamic Image-Adaptive Geometry (DiAG) Scale Routing Head.

    Dynamically computes the multi-scale spatial partition of unity pi(x, y) in Delta^{K-1}
    using purely local carrier features P4 modulated by image-level semantic context:
      z_local(x, y) = PW(SiLU(GN(DW(P_4))))
      z(x, y) = z_local(x, y) + delta_scale
      pi(x, y) = Softmax(z(x, y) / temperature, dim=1)

    Mathematical Properties:
      1. Pure Dynamism: 100% feature-driven. ZERO static coordinate grids (no torch.linspace).
      2. Translation Equivariance: Carrier convolution preserves strict translation equivariance across crops.
      3. Resolution Preservation: Every scale k exists everywhere across the image support,
         preventing foreground spatial resolution starvation.
      4. Step 0 Identity Parity: Zero-initialized weights output exact uniform scale
         distribution (1/K, 1/K, ...) at Step 0.

    Parameter budget:
      - Depthwise 3x3: in_channels * 9 + in_channels = 320 params.
      - GroupNorm(8, in_channels): 2 * in_channels = 64 params.
      - Pointwise 1x1: in_channels * num_scales + num_scales = 99 params.
      Total trainable parameters: 483 params (< 500 budget).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        temperature: float = 1.0,
        use_tilt: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.temperature = float(max(temperature, 0.1))
        self.use_tilt = bool(use_tilt)

        self.dw = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, self.in_channels)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(self.in_channels, self.num_scales, kernel_size=1, bias=True)

        # Zero-initialization: Step 0 Identity Parity
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(
        self,
        x: torch.Tensor,
        delta_scale: torch.Tensor | None = None,
        scene_tilt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute image-adaptive spatial scale routing weights pi(x, y) in Delta^{K-1}.

        Args:
            x: [B, C, H, W] carrier feature tensor.
            delta_scale: [B, K, 1, 1] optional global scale bias from semantic context.
            scene_tilt: [B, 1, 1, 1] optional scene tilt factor.

        Returns:
            pi: [B, num_scales, H, W] spatial partition of unity (sum_k pi_k == 1.0).
        """
        feat = self.act(self.norm(self.dw(x)))
        logits = self.pw(feat)  # [B, K, H, W]

        # Dynamic perspective contrast modulation: steep camera angle sharpens local scale transitions
        if scene_tilt is not None and self.use_tilt:
            logits = logits * (0.5 + scene_tilt.to(dtype=logits.dtype))

        if delta_scale is not None:
            logits = logits + delta_scale.to(dtype=logits.dtype)

        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.softmax(logits, dim=1)


class DiAGFactorizedRoutingHead(nn.Module):
    """Dynamic Image-Adaptive Geometry (DiAG) Factorized Scale & Aspect Routing Head.

    Combines:
    1. Multi-scale receptive carrier sensing (local DW-Conv 3x3 + 5x5 context pooling).
    2. Decoupled marginal scale distribution pi_scale in Delta^2 over [32, 64, 128] px.
    3. Decoupled aspect ratio distribution pi_aspect in Delta^1 over [1:1 square, 2:1 vertical rectangle].
    4. Global DCAP perspective conditioning:
       - delta_scale modulates marginal scale logits.
       - scene_tilt modulates scale & aspect transition contrast (steeper tilt favors vertical elongation).
    5. Step 0 Identity Parity: zero-initialized weights guarantee exact uniform distributions at Step 0.
    6. 100% translation-equivariant, zero static coordinate grids (no torch.linspace).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        num_aspects: int = 2,
        temperature: float = 1.0,
        use_tilt: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.num_aspects = int(num_aspects)
        self.temperature = float(max(temperature, 0.1))
        self.use_tilt = bool(use_tilt)

        # Context-aware depthwise carrier
        self.dw = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, self.in_channels)
        self.act = nn.SiLU(inplace=True)

        # Branch 1: Marginal Scale Head (S=3: 32, 64, 128)
        self.pw_scale = nn.Conv2d(self.in_channels, self.num_scales, kernel_size=1, bias=True)
        nn.init.zeros_(self.pw_scale.weight)
        nn.init.zeros_(self.pw_scale.bias)

        # Branch 2: Aspect Ratio Head (A=2: 1:1 square, 2:1 vertical rectangle)
        self.pw_aspect = nn.Conv2d(self.in_channels, self.num_aspects, kernel_size=1, bias=True)
        nn.init.zeros_(self.pw_aspect.weight)
        nn.init.zeros_(self.pw_aspect.bias)

    def forward(
        self,
        x: torch.Tensor,
        delta_scale: torch.Tensor | None = None,
        scene_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Context-enhanced receptive field (local + 5x5 pooling)
        x_ctx = 0.5 * (x + F.avg_pool2d(x, kernel_size=5, stride=1, padding=2))
        feat = self.act(self.norm(self.dw(x_ctx)))

        logits_scale = self.pw_scale(feat)
        logits_aspect = self.pw_aspect(feat)

        if scene_tilt is not None and self.use_tilt:
            tilt_f = scene_tilt.to(dtype=logits_scale.dtype)
            # Steep camera tilt sharpens scale transitions
            logits_scale = logits_scale * (0.5 + tilt_f)
            # Tilt provides a differential prior favoring vertical elongation (2:1 aspect) when tilt > 0.5
            tilt_delta = tilt_f - 0.5
            aspect_tilt_bias = torch.cat([-tilt_delta, tilt_delta], dim=1)
            logits_aspect = logits_aspect + aspect_tilt_bias

        if delta_scale is not None:
            logits_scale = logits_scale + delta_scale.to(dtype=logits_scale.dtype)

        temp = float(self.temperature)
        pi_scale = F.softmax(logits_scale / temp, dim=1)
        pi_aspect = F.softmax(logits_aspect / temp, dim=1)

        # 4-window joint partition of unity
        pi_0 = pi_scale[:, 0:1]
        pi_1 = pi_scale[:, 1:2] * pi_aspect[:, 0:1]
        pi_2 = pi_scale[:, 1:2] * pi_aspect[:, 1:2]
        pi_3 = pi_scale[:, 2:3]
        joint_pi = torch.cat([pi_0, pi_1, pi_2, pi_3], dim=1)

        return joint_pi, pi_scale, pi_aspect

