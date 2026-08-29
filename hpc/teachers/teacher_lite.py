from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Residual3x3(nn.Module):
    """Strong teacher-only spatial refinement."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_gn_groups(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.norm2(self.conv2(y))
        return F.silu(x + y, inplace=True)


class DepthwiseMultiDilationContext(nn.Module):
    """Depthwise Multi-Receptive-Field Context block for P8 and P16."""

    def __init__(self, channels: int = 96, branch_channels: int = 24):
        super().__init__()
        assert branch_channels * 4 == channels

        self.b0 = ConvGNAct(channels, branch_channels, 1)
        self.b1 = nn.Sequential(
            ConvGNAct(channels, channels, 3, padding=1, dilation=1, groups=channels),
            ConvGNAct(channels, branch_channels, 1),
        )
        self.b2 = nn.Sequential(
            ConvGNAct(channels, channels, 3, padding=2, dilation=2, groups=channels),
            ConvGNAct(channels, branch_channels, 1),
        )
        self.b3 = nn.Sequential(
            ConvGNAct(channels, channels, 3, padding=3, dilation=3, groups=channels),
            ConvGNAct(channels, branch_channels, 1),
        )
        self.fuse = ConvGNAct(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [self.b0(x), self.b1(x), self.b2(x), self.b3(x)], dim=1
        )
        return x + self.fuse(y)


class EfficientNetB0Trunk(nn.Module):
    """
    EfficientNet-B0 feature extractor.

    For a 448x448 input:
      C4  = features[2] -> 24x112x112
      C8  = features[3] -> 40x56x56
      C16 = features[5] -> 112x28x28
      C32 = features[7] -> 320x14x14
    """

    out_channels = (24, 40, 112, 320)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        # Drop features[8] (1280 expansion), avgpool and classifier.
        self.features = nn.ModuleList(list(model.features[:8]))

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c4 = c8 = c16 = c32 = None
        for i, block in enumerate(self.features):
            x = block(x)
            if i == 2:
                c4 = x
            elif i == 3:
                c8 = x
            elif i == 5:
                c16 = x
            elif i == 7:
                c32 = x

        assert c4 is not None and c8 is not None
        assert c16 is not None and c32 is not None
        return c4, c8, c16, c32


class TeacherLite(nn.Module):
    """Teacher-Lite v2 crowd-counting teacher with P8 + P16 Multi-Dilation Context."""

    def __init__(
        self,
        width: int = 96,
        pretrained: bool = True,
        eps_d: float = 1e-6,
        use_p8_context: bool = True,
    ):
        super().__init__()
        assert width % 4 == 0
        self.width = width
        self.eps_d = eps_d
        self.use_p8_context = use_p8_context

        self.backbone = EfficientNetB0Trunk(pretrained=pretrained)
        c4, c8, c16, c32 = self.backbone.out_channels

        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)
        self.lat32 = ConvGNAct(c32, width, 1)

        self.ref32 = Residual3x3(width)
        self.ref16 = Residual3x3(width)
        self.ref8 = Residual3x3(width)
        self.ref4 = Residual3x3(width)

        # Context at P16 (/16) and P8 (/8)
        self.context16 = DepthwiseMultiDilationContext(width, width // 4)
        if self.use_p8_context:
            self.context8 = DepthwiseMultiDilationContext(width, width // 4)
        else:
            self.context8 = nn.Identity()

        self.density_head = nn.Sequential(
            ConvGNAct(width, 64, 3, padding=1),
            ConvGNAct(64, 32, 3, padding=1),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

        self.count_head = nn.Sequential(
            nn.Linear(width * 2, width),
            nn.SiLU(inplace=True),
            nn.Linear(width, 1),
        )

        # Start near low positive mass.
        nn.init.constant_(self.density_head[-1].bias, -5.0)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        c4, c8, c16, c32 = self.backbone(x)

        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)
        l32 = self.lat32(c32)

        # Lateral FPN with Multi-Dilation Context at P16 and P8
        p32 = self.ref32(l32)
        p16 = self.ref16(l16 + self._up(p32, l16))
        p16 = self.context16(p16)
        p8 = self.ref8(l8 + self._up(p16, l8))
        p8 = self.context8(p8)
        p4 = self.ref4(l4 + self._up(p8, l4))

        density_logits = self.density_head(p4)
        density = F.softplus(density_logits) + self.eps_d
        count_from_map = density.sum(dim=(-1, -2, -3))

        g16 = F.adaptive_avg_pool2d(p16, 1).flatten(1)
        g32 = F.adaptive_avg_pool2d(p32, 1).flatten(1)
        count_reg = F.softplus(self.count_head(torch.cat([g16, g32], dim=1)))
        count_reg = count_reg.squeeze(1)

        return {
            "density": density,
            "count_map": count_from_map,
            "count_reg": count_reg,
            "p4": p4,
            "p8": p8,
            "p16": p16,
            "p32": p32,
        }
