from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated


class AdditiveFusion(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.p4 = ConvGNAct(24, width, 1)
        self.p8 = ConvGNAct(40, width, 1)
        self.p16 = ConvGNAct(64, width, 1)
        self.out = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
        )

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = c4.shape[-2:]
        p4 = self.p4(c4)
        p8 = self.p8(c8)
        p16 = self.p16(c16)
        p = p4 + F.interpolate(p8, size=size, mode="bilinear", align_corners=False)
        p = p + F.interpolate(p16, size=size, mode="bilinear", align_corners=False)
        return self.out(p), p8, p16


class AdditiveFPNNeck(nn.Module):
    """Additive depthwise-separable FPN neck fusing reductions 4, 8, 16 into (P4, P8, P16)."""

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        context_dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks)
        p16 = self.ref16(l16 + ctx_sum)

        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)
        return p4, p8, p16


class ASPPLiteFPNNeck(nn.Module):
    """Static Additive FPN neck with a true ASPP-lite module on the P16 feature map.

    Architecture:
        - Lateral projections: C4/C8/C16 -> width channels via 1x1 Conv+GN+SiLU.
        - ASPP-lite on L16 (four parallel branches, all static):
            * Branch 0: DW 3x3, dilation=1  (local context)
            * Branch 1: DW 3x3, dilation=3  (medium context)
            * Branch 2: DW 3x3, dilation=6  (wide context, RF=13 cells x stride16 = 208 px)
            * Branch 3: Global Average Pooling -> Linear (width->width) -> broadcast
          All four branches are summed (no learnable fusion weights) then projected by
          a 1x1 Conv+GN back to ``width`` channels before the DSResidual refinement.
        - Top-down FPN (static additive):
            P16 -> upsample + L8 lateral -> DSResidual -> P8
            P8  -> upsample + L4 lateral -> DSResidual -> P4
        - Returns (P4, P8, P16) matching the AdditiveFPNNeck interface exactly.

    Constraint:
        NO dynamic weights, NO softmax, NO attention. Fusion is strictly 1:1 addition.
        This avoids the "double non-stationarity" that collapsed RMR-v5 (MAE 104.24).
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        aspp_dilations: tuple[int, ...] = (1, 3, 6),
        use_aspp_gap: bool = True,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_aspp_gap = bool(use_aspp_gap)

        # Lateral 1x1 projections
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        # ASPP-lite on L16
        self.aspp_dw = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in aspp_dilations
        ])
        if self.use_aspp_gap:
            self.aspp_gap: nn.Sequential | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(width, width, bias=True),
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None
        self.aspp_proj = ConvGNAct(width, width, k=1)

        # Top-down FPN refinement blocks (Static DSResidual)
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # ASPP-lite on L16
        aspp_out = sum(branch(l16) for branch in self.aspp_dw)

        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            aspp_out = aspp_out + gap_broadcast

        p16_pre = self.aspp_proj(l16 + aspp_out)
        p16 = self.ref16(p16_pre)

        # Top-down FPN
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        return p4, p8, p16
