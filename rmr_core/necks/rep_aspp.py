from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DepthwiseDilated
from .rep_fpn import RepDWBlock7x7


class RepASPPLiteNeck(nn.Module):
    """Hybrid Rep-ASPP Static Additive FPN Neck.

    Combines:
      1. ASPP-Lite on P16 (dilations {1, 3, 6} + Global Average Pooling Linear projection)
         to provide wide receptive field (RF = 208 px) and full-scene global context (512 px)
         for crowded scenes (solving M2's blind spot on extreme crowds).
      2. Static 1:1 Additive Top-Down FPN (NO softmax learnable weights)
         to eliminate double non-stationarity and stabilize gradient propagation.
      3. Structural Re-parameterization 7x7 Depthwise Refinement Blocks (RepDWBlock7x7)
         at P16, P8, and P4 to preserve high-frequency sharpness (retaining M2's record
         sparse MAE = 9.83) with zero deployment overhead (algebraically fused into
         single 7x7 DW Conv).

    Returns:
      (p4, p8, p16) matching the standard FPN neck interface.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        aspp_dilations: tuple[int, ...] = (1, 3, 6),
        use_aspp_gap: bool = True,
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_aspp_gap = bool(use_aspp_gap)

        # Lateral projections: C -> width via 1x1 Conv + GroupNorm + SiLU
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        # ASPP-Lite on L16
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
        # Top-down FPN refinement blocks: Structural Re-parameterized 7x7 Depthwise
        self.ref16 = RepDWBlock7x7(width)
        self.ref8 = RepDWBlock7x7(width)
        self.ref4 = RepDWBlock7x7(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # ASPP-Lite on L16: Multi-dilation + Global Average Pooling context
        aspp_out = sum(branch(l16) for branch in self.aspp_dw)
        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            aspp_out = aspp_out + gap_broadcast

        # Feed directly into RepDWBlock7x7 refinement block
        p16 = self.ref16(l16 + aspp_out)

        # Top-down FPN (Static 1:1 Additive)
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        return p4, p8, p16

    def switch_to_deploy(self) -> None:
        """Algebraically fuse all 3 RepDW refinement blocks into single 7x7 DW convs."""
        self.ref16.switch_to_deploy()
        self.ref8.switch_to_deploy()
        self.ref4.switch_to_deploy()
