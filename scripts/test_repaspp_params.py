import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.necks.blocks import ConvGNAct, DepthwiseDilated
from rmr_core.necks.rep_fpn import RepDWBlock7x7

class RepASPPLiteNeck(nn.Module):
    """Hybrid RepASPP-Lite FPN Neck.
    
    Combines:
    1. ASPP-Lite at P16 with dilations (1, 3, 6) + Global Average Pooling (GAP) for global scene context.
    2. Static 1:1 additive top-down FPN (no non-stationary softmax fusion).
    3. Structural Re-parameterization (RepDWBlock7x7) at P16, P8, P4 for razor-sharp local resolution.
    """
    def __init__(
        self,
        in_channels: tuple[int, int, int] = (32, 40, 64),
        width: int = 32,
        aspp_dilations: tuple[int, ...] = (1, 3, 6),
        use_aspp_gap: bool = True,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_aspp_gap = use_aspp_gap
        
        # Lateral 1x1 projections
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)
        
        # ASPP branches on P16
        self.aspp_dw = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in aspp_dilations
        ])
        if self.use_aspp_gap:
            self.aspp_gap = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(width, width, bias=True),
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None
            
        num_branches = len(aspp_dilations) + (1 if self.use_aspp_gap else 0)
        self.aspp_proj = ConvGNAct(width * num_branches, width, 1)
        
        # Refinement blocks: RepDWBlock7x7 at each scale
        self.ref16 = RepDWBlock7x7(width)
        self.ref8 = RepDWBlock7x7(width)
        self.ref4 = RepDWBlock7x7(width)
        
    def forward(self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)
        
        # ASPP on P16
        feats = [dw(l16) for dw in self.aspp_dw]
        if self.aspp_gap is not None:
            gap = self.aspp_gap(l16).unsqueeze(-1).unsqueeze(-1)
            gap = gap.expand(-1, -1, l16.shape[-2], l16.shape[-1])
            feats.append(gap)
        aspp_out = self.aspp_proj(torch.cat(feats, dim=1))
        p16 = self.ref16(l16 + aspp_out)
        
        # Top-down FPN (Static Additive)
        up16_to_8 = F.interpolate(p16, size=l8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.ref8(l8 + up16_to_8)
        
        up8_to_4 = F.interpolate(p8, size=l4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.ref4(l4 + up8_to_4)
        
        return p4, p8, p16
        
    def switch_to_deploy(self):
        self.ref16.switch_to_deploy()
        self.ref8.switch_to_deploy()
        self.ref4.switch_to_deploy()

# Measure params
neck_w32 = RepASPPLiteNeck(in_channels=(32, 40, 64), width=32)
neck_w64 = RepASPPLiteNeck(in_channels=(32, 40, 64), width=64)

def count_p(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

print(f"RepASPPNeck (W=32): {count_p(neck_w32):,} parameters")
print(f"RepASPPNeck (W=64): {count_p(neck_w64):,} parameters")

# Test full model integration
from rmr_v3.model import RMRv3, RMRv3Config
cfg_m1 = RMRv3Config.from_dict({
    "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
    "feature_width": 32,
    "neck_type": "aspp_lite",
    "output_stride": 4,
}, pretrained=False)
m1 = RMRv3(cfg_m1)
print(f"M1 Full Model (aspp_lite): {count_p(m1):,} parameters")
