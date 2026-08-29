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


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature backbone returning features at reductions 4, 8, 16.

    The ``truncate`` parameter controls whether stages beyond ``max(target_reductions)``
    are omitted.  In practice, timm's ``features_only=True`` with explicit ``out_indices``
    already achieves this: stages beyond the last selected index are not instantiated or
    executed, so no additional truncation code is needed.  The parameter is accepted for
    API compatibility but has no further effect beyond index selection.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = True,
        target_reductions: tuple = (4, 8, 16),
        truncate: bool = True,
    ):
        super().__init__()
        import timm
        self.model_name = model_name
        self.target_reductions = tuple(int(r) for r in target_reductions)
        # truncate=True is the intent; timm features_only + out_indices already enforces
        # it by not instantiating stages beyond max(selected_indices).
        # Stored only for serialization / state_dict compatibility; not used in forward.
        self.truncate = bool(truncate)

        probe = timm.create_model(model_name, pretrained=False, features_only=True)
        reductions = list(probe.feature_info.reduction())
        channels = list(probe.feature_info.channels())
        del probe

        selected_indices = []
        selected_channels = []
        for r in self.target_reductions:
            matches = [i for i, rr in enumerate(reductions) if rr == r]
            if not matches:
                raise ValueError(f"Reduction {r} not found in {model_name}: {reductions}")
            idx = matches[-1]
            selected_indices.append(idx)
            selected_channels.append(channels[idx])

        self.selected_indices = tuple(selected_indices)
        self.out_channels = selected_channels
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.selected_indices,
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        return features[0], features[1], features[2]
