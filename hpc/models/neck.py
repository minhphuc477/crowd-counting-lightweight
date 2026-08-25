import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .blocks import ConvGNAct, DepthwiseDilated, DSResidual


class AdditiveFPNNeck(nn.Module):
    """Additive Depthwise-Separable FPN Neck with multi-dilation context block at reduction 16.
    
    Channels: width (default C=32)
    Inputs: c4, c8, c16 from backbone (reductions 4, 8, 16)
    Output: p4 feature map at reduction 4 with width channels
    """
    def __init__(
        self,
        in_channels: List[int],
        width: int = 32,
        context_dilations: Tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        
        # Lateral 1x1 projections
        self.lat4 = ConvGNAct(c4, width, kernel_size=1)
        self.lat8 = ConvGNAct(c8, width, kernel_size=1)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)
        
        # Multi-dilation coarse context at reduction 16
        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])
        
        # DS Residual refinement blocks
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
    ) -> torch.Tensor:
        # Lateral projections
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)
        
        # Multi-dilation coarse context aggregation
        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks) if len(self.context_blocks) > 0 else 0
        p16 = self.ref16(l16 + ctx_sum)
        
        # Top-down additive fusion
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)
        
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)
        
        return p4
