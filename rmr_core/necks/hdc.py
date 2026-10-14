from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated


class HDCLiteFPNNeck(nn.Module):
    """Hybrid Dilated Cascade (HDC-Lite) FPN neck (Wang et al., WACV 2018).

    Replaces parallel ASPP-Lite with a sequential cascade of depthwise dilated convolutions
    with continuous dilations [1, 2, 3] on L16:
        h1 = DW_{d=1}(l16)
        h2 = DW_{d=2}(h1)
        h3 = DW_{d=3}(h2)
    Satisfies HDC gridding-free condition (M_2 = 2 <= K=3), eliminating the 24px gridding holes
    of ASPP dilation 6. When use_gap=True, retains the GAP Linear projection branch (1,056 params)
    to provide the essential 512px global scene context prior.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        hdc_dilations: tuple[int, ...] = (1, 2, 3),
        use_gap: bool = False,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_gap = bool(use_gap)

        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

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
        self.hdc_proj = ConvGNAct(width, width, k=1)

        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        h = l16
        hdc_accum = torch.zeros_like(l16)
        for block in self.hdc_blocks:
            h = block(h)
            hdc_accum = hdc_accum + h

        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            hdc_accum = hdc_accum + gap_broadcast

        p16 = self.ref16(self.hdc_proj(l16 + hdc_accum))

        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        return p4, p8, p16
