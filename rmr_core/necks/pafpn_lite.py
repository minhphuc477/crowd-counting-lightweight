from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual
from .pafpn import _make_dw_downsample, _match_size


class PAFPNLiteNeck(nn.Module):
    """Paper-faithful lightweight PANet/PAFPN baseline (Liu et al., CVPR 2018).

    Follows PANet Sec. 3.1 (top-down + bottom-up plain addition with stride-2 3x3
    downsampling and nearest-neighbor upsampling, no top-level strip context, no convex
    fusion weights, no extra td4 refinement) and Sec. 3.2 Adaptive Feature Pooling
    element-wise sum into P4_out so bottom-up layers receive gradients in single-head counting.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
    ) -> None:
        super().__init__()
        c4, c8, c16 = (int(c) for c in in_channels)
        w = int(width)
        self.width = w

        self.lat4 = ConvGNAct(c4, w, 1)
        self.lat8 = ConvGNAct(c8, w, 1)
        self.lat16 = ConvGNAct(c16, w, 1)

        self.td8 = DSResidual(w)
        self.td4 = DSResidual(w)

        self.down4_to_8 = _make_dw_downsample(w)
        self.bu8 = DSResidual(w)
        self.down8_to_16 = _make_dw_downsample(w)
        self.bu16 = DSResidual(w)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        p16 = l16
        up16_to_8 = F.interpolate(p16, size=l8.shape[-2:], mode="nearest")
        p8 = self.td8(l8 + up16_to_8)
        up8_to_4 = F.interpolate(p8, size=l4.shape[-2:], mode="nearest")
        p4 = self.td4(l4 + up8_to_4)

        n4 = p4
        n8 = self.bu8(p8 + _match_size(self.down4_to_8(n4), p8))
        n16 = self.bu16(p16 + _match_size(self.down8_to_16(n8), p16))

        p4_out = n4 + _match_size(n8, n4) + _match_size(n16, n4)
        return p4_out, n8, n16
