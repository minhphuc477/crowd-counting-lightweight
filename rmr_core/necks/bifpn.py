from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual


class BiFPNLiteNeck(nn.Module):
    """Bidirectional Feature Pyramid Network (BiFPN-Lite) neck (Tan et al., CVPR 2020).

    Implements fast normalized multi-scale feature fusion across reductions 4, 8, 16:
      - Lateral projections: C4, C8, C16 -> width channels via ConvGNAct (1x1).
      - Top-Down pathway:
          p16_td = L16 (+ optional GAP scene context)
          p8_td = DSResidual((w1 * L8 + w2 * Resize(p16_td)) / (w1 + w2 + eps))
          p4_out = DSResidual((w3 * L4 + w4 * Resize(p8_td)) / (w3 + w4 + eps))
      - Bottom-Up pathway:
          p8_out = DSResidual((v1 * L8 + v2 * p8_td + v3 * Downsample(p4_out)) / (v1 + v2 + v3 + eps))
          p16_out = DSResidual((v4 * L16 + v5 * p16_td + v6 * Downsample(p8_out)) / (v4 + v5 + v6 + eps))

    Fast Normalized Fusion:
      Uses positive learnable weights (via ReLU) to ensure numerical stability and bounded weights.
      Initial weights are initialized to 1.0, giving equal weighting at Step 0.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        use_gap: bool = True,
        eps: float = 1e-4,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_gap = bool(use_gap)
        self.eps = float(eps)

        # Lateral 1x1 projections
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        # Global Average Pooling branch on L16 (if enabled)
        if self.use_gap:
            self.aspp_gap: nn.Sequential | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(width, width, bias=True),
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None

        # Top-down fusion weights & refinement
        # P8_td = fuse(L8, Resize(P16_td))
        self.w_td8 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.ref_td8 = DSResidual(width)

        # P4_out = fuse(L4, Resize(P8_td))
        self.w_td4 = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.ref_out4 = DSResidual(width)

        # Downsampling for bottom-up pathway (MaxPool 3x3 s2 preserves highest activations, 0 params)
        self.downsample8 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample16 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Bottom-up fusion weights & refinement
        # P8_out = fuse(L8, P8_td, Downsample(P4_out))
        self.v_bu8 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.ref_out8 = DSResidual(width)

        # P16_out = fuse(L16, P16_td, Downsample(P8_out))
        self.v_bu16 = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.ref_out16 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # P16_td with optional GAP context
        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            p16_td = l16 + gap_broadcast
        else:
            p16_td = l16

        # --- Top-Down Pathway ---
        # 1. P8_td
        up16_to_8 = F.interpolate(p16_td, size=l8.shape[-2:], mode="bilinear", align_corners=False)
        w8 = F.relu(self.w_td8)
        w8_norm = w8 / (w8.sum() + self.eps)
        in_p8_td = w8_norm[0] * l8 + w8_norm[1] * up16_to_8
        p8_td = self.ref_td8(in_p8_td)

        # 2. P4_out
        up8_to_4 = F.interpolate(p8_td, size=l4.shape[-2:], mode="bilinear", align_corners=False)
        w4 = F.relu(self.w_td4)
        w4_norm = w4 / (w4.sum() + self.eps)
        in_p4_out = w4_norm[0] * l4 + w4_norm[1] * up8_to_4
        p4_out = self.ref_out4(in_p4_out)

        # --- Bottom-Up Pathway ---
        # 3. P8_out
        down4_to_8 = self.downsample8(p4_out)
        if down4_to_8.shape[-2:] != l8.shape[-2:]:
            down4_to_8 = F.interpolate(down4_to_8, size=l8.shape[-2:], mode="bilinear", align_corners=False)
        v8 = F.relu(self.v_bu8)
        v8_norm = v8 / (v8.sum() + self.eps)
        in_p8_out = v8_norm[0] * l8 + v8_norm[1] * p8_td + v8_norm[2] * down4_to_8
        p8_out = self.ref_out8(in_p8_out)

        # 4. P16_out
        down8_to_16 = self.downsample16(p8_out)
        if down8_to_16.shape[-2:] != l16.shape[-2:]:
            down8_to_16 = F.interpolate(down8_to_16, size=l16.shape[-2:], mode="bilinear", align_corners=False)
        v16 = F.relu(self.v_bu16)
        v16_norm = v16 / (v16.sum() + self.eps)
        in_p16_out = v16_norm[0] * l16 + v16_norm[1] * p16_td + v16_norm[2] * down8_to_16
        p16_out = self.ref_out16(in_p16_out)

        return p4_out, p8_out, p16_out
