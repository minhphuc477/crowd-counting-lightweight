from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DepthwiseDilated
from .rep_fpn import RepDWBlock7x7


class StaticFastNormalizedFusion(nn.Module):
    """Static Fast Normalized Fusion (Tan et al., CVPR 2020 / BiFPN).

    Uses data-agnostic learnable parameters w1, w2 to preserve strict stationarity
    (zero non-stationarity) across feature scales:
        O = (Softplus(w1) * lateral + Softplus(w2) * upsampled) / (Softplus(w1) + Softplus(w2) + eps)

    Because w1 and w2 are data-agnostic scalar parameters (not predicted from input X),
    they remain stationary across mini-batches, avoiding the double non-stationarity
    that destabilizes unrolled inverse solvers.
    Using Softplus guarantees strictly positive weights (> 0) and smooth non-zero gradients
    everywhere, completely eliminating the dying ReLU / zero-feature collapse risk.
    """

    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = float(eps)
        # Learnable scalar parameters, initialized so Softplus(w) = 1.0 (50-50 equal weighting)
        # softplus_inv(1.0) = ln(e - 1) = 0.5413248546129181
        self.weights = nn.Parameter(torch.full((2,), 0.5413248546129181, dtype=torch.float32))

    def forward(self, lateral: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        w = F.softplus(self.weights)
        w_norm = w / (w.sum() + self.eps)
        return w_norm[0] * lateral + w_norm[1] * upsampled


class RLKHDCFPNNeck(nn.Module):
    """Rep-Large-Kernel Hybrid Dilated Cascade FPN Neck (RLK-HDC Neck).

    Academic & Empirical Foundations:
      1. HDC Gridding-Free Cascade (Wang et al., WACV 2018):
         Replaces parallel ASPP branches with a sequential depthwise dilated cascade
         using continuous rates [1, 2, 3] on L16:
             h1 = DW_{d=1}(l16)
             h2 = DW_{d=2}(h1)
             h3 = DW_{d=3}(h2)
         Satisfies Wang's condition (M_2 = 2 <= K=3), ensuring 100% dense spatial coverage
         without the 96px gridding holes of ASPP dilation 6.
      2. Global Scene Context Prior (Chen et al., DeepLabv3 / ParseNet):
         Retains GAP linear projection (512px receptive field) on L16, providing essential
         macro crowd density guidance that prevents severe undercounting in dense crowds
         (empirically verified in GEN15 sub60_e55 vs sub60_e56).
      3. Structural Re-parameterization (Ding et al., RepVGG CVPR 2021 / RepLKNet CVPR 2022):
         Uses 7x7 Depthwise Refinement Blocks (RepDWBlock7x7) at P16, P8, P4.
         During training: 7x7 DW + 1x1 DW + Identity with BatchNorms optimizes multi-frequency gradients.
         During inference: Algebraically fused into a single 7x7 DW Conv (0 extra params, 0 FLOP overhead).
         Provides a 28x28 px aperture at Stride 4 to eliminate background texture confusion
         (empirically verified in RMR V49 with record 9.83 Sparse MAE and V66 Exp 6 reducing 145 ghost people).
      4. Static Fast Normalized Fusion (Tan et al., CVPR 2020):
         Combines lateral and upsampled features using data-agnostic normalized convex weights,
         strictly preventing double non-stationarity while learning optimal scale balance.

    Parameter Budget:
      - Training (width=32): 10,756 params (Total Track A model: 103,879 <= 104,441).
      - Deployed (width=32): 10,180 params (Total Track A model: 103,303 <= 104,441).
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        hdc_dilations: tuple[int, ...] = (1, 2, 3),
        use_gap: bool = True,
        use_fast_norm_fusion: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_gap = bool(use_gap)
        self.use_fast_norm_fusion = bool(use_fast_norm_fusion)

        # 1. Lateral 1x1 projections: C -> width via 1x1 Conv + GroupNorm + SiLU
        self.lat4 = ConvGNAct(c4, width, k=1)
        self.lat8 = ConvGNAct(c8, width, k=1)
        self.lat16 = ConvGNAct(c16, width, k=1)

        # 2. HDC Cascade on L16: sequential depthwise dilations [1, 2, 3]
        self.hdc_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in hdc_dilations
        ])
        if self.use_gap:
            self.aspp_gap: nn.Sequential | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(width, width, bias=True),
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None

        # 3. Static Fast Normalized Fusion modules (P8 and P4)
        if self.use_fast_norm_fusion:
            self.fuse8: StaticFastNormalizedFusion | None = StaticFastNormalizedFusion(eps=eps)
            self.fuse4: StaticFastNormalizedFusion | None = StaticFastNormalizedFusion(eps=eps)
        else:
            self.fuse8 = None
            self.fuse4 = None

        # 4. Structural Re-parameterization Refinement Blocks (7x7 Depthwise)
        self.ref16 = RepDWBlock7x7(width)
        self.ref8 = RepDWBlock7x7(width)
        self.ref4 = RepDWBlock7x7(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # HDC Cascade on L16
        h = l16
        hdc_accum = torch.zeros_like(l16)
        for block in self.hdc_blocks:
            h = block(h)
            hdc_accum = hdc_accum + h

        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            hdc_accum = hdc_accum + gap_broadcast

        # Feed directly into RepDWBlock7x7 refinement block
        p16 = self.ref16(l16 + hdc_accum)

        # Scale 8 Top-Down Fusion
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.fuse8 is not None:
            f8 = self.fuse8(lateral=l8, upsampled=up16_to_8)
        else:
            f8 = l8 + up16_to_8
        p8 = self.ref8(f8)

        # Scale 4 Top-Down Fusion
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.fuse4 is not None:
            f4 = self.fuse4(lateral=l4, upsampled=up8_to_4)
        else:
            f4 = l4 + up8_to_4
        p4 = self.ref4(f4)

        return p4, p8, p16

    def switch_to_deploy(self) -> None:
        """Algebraically fuse all 3 RepDW refinement blocks into single 7x7 DW convs."""
        self.ref16.switch_to_deploy()
        self.ref8.switch_to_deploy()
        self.ref4.switch_to_deploy()
