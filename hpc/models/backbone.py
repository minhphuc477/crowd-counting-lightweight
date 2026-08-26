import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights


class ShuffleNetV2PyramidBackbone(nn.Module):
    """ShuffleNetV2 ×0.5 feature pyramid backbone.

    Returns spatial features at reductions 4, 8, 16, 32.
    Intentionally excludes conv5, global pooling, and classifier.

    Expected output channels: [24, 48, 96, 192].
    Expected total parameters: 143,136.
    """

    # Verified against torchvision ShuffleNetV2 x0.5:
    # conv1 → /2 (24 ch), maxpool → /4 (24 ch),
    # stage2 → /8 (48 ch), stage3 → /16 (96 ch), stage4 → /32 (192 ch).
    OUT_CHANNELS = [24, 48, 96, 192]
    OUT_REDUCTIONS = [4, 8, 16, 32]

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
        base = shufflenet_v2_x0_5(weights=weights)

        # Keep only the feature-extraction stages; discard conv5, avgpool, fc.
        self.conv1 = base.conv1       # 3 → 24 ch, /2
        self.maxpool = base.maxpool   # /4
        self.stage2 = base.stage2     # 24 → 48 ch, /8
        self.stage3 = base.stage3     # 48 → 96 ch, /16
        self.stage4 = base.stage4     # 96 → 192 ch, /32

        self.out_channels = self.OUT_CHANNELS
        self.out_reductions = self.OUT_REDUCTIONS

    def forward(self, x: torch.Tensor):
        """Return (c4, c8, c16, c32) feature maps."""
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input (B, 3, H, W), got {tuple(x.shape)}")
        x = self.conv1(x)       # /2, 24 ch
        c4 = self.maxpool(x)    # /4, 24 ch
        c8 = self.stage2(c4)    # /8, 48 ch
        c16 = self.stage3(c8)   # /16, 96 ch
        c32 = self.stage4(c16)  # /32, 192 ch
        return c4, c8, c16, c32
