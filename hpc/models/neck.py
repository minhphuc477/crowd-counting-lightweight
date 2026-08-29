import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated


class AdditiveFPNNeck(nn.Module):
    """Additive Depthwise-Separable FPN Neck with multi-dilation context blocks at reduction 16 and optional 8.

    Channels: width (default C=32)
    Inputs: c4, c8, c16 from backbone (reductions 4, 8, 16)
    Output: p4 feature map at reduction 4 with width channels
    """

    def __init__(
        self,
        in_channels: tuple = (24, 48, 80),
        width: int = 32,
        context_dilations: tuple = (1, 2, 3),
        use_p8_context: bool = False,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_p8_context = use_p8_context

        # Lateral 1x1 projections
        self.lat4 = ConvGNAct(c4, width, kernel_size=1)
        self.lat8 = ConvGNAct(c8, width, kernel_size=1)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)

        # Multi-dilation coarse context at reduction 16
        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])

        # Optional multi-dilation context at reduction 8
        if self.use_p8_context:
            self.context_p8_blocks = nn.ModuleList([
                DepthwiseDilated(width, dilation=d) for d in (1, 2, 3)
            ])
            self.context_p8_fuse = ConvGNAct(width, width, kernel_size=1)
            nn.init.zeros_(self.context_p8_fuse.conv.weight)

        # DS Residual refinement blocks
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
        return_routes: bool = False,
    ):
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # Coarse context aggregation at P16
        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks) if len(self.context_blocks) > 0 else 0
        p16 = self.ref16(l16 + ctx_sum)

        # Top-down additive fusion at P8
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8_in = l8 + up16_to_8
        if self.use_p8_context:
            ctx_sum8 = sum(ctx(p8_in) for ctx in self.context_p8_blocks)
            p8_in = p8_in + self.context_p8_fuse(ctx_sum8)

        p8 = self.ref8(p8_in)

        # Top-down additive fusion at P4
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        if return_routes:
            return p4, {
                "routes8": None,
                "routes4": None,
                "p4": p4,
                "p8": p8,
                "p16": p16,
            }
        return p4
