from typing import Tuple

import torch
import torch.nn as nn
import timm


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature backbone returning the last feature at reductions 4/8/16."""

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = True,
        target_reductions: Tuple[int, int, int] = (4, 8, 16),
        truncate: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.target_reductions = tuple(int(r) for r in target_reductions)

        # Query feature metadata once without downloading weights.
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
            # Prefer the deepest/most semantic feature if a reduction appears more than once.
            idx = matches[-1]
            selected_indices.append(idx)
            selected_channels.append(channels[idx])
        if selected_indices != sorted(selected_indices):
            raise ValueError(f"Non-monotonic selected feature indices: {selected_indices}")

        self.selected_indices = tuple(selected_indices)
        self.out_channels = selected_channels
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.selected_indices,
        )

        # No manual mutation is performed. timm's FeatureList/FeatureDict wrapper already
        # rebuilds modules only through the last requested return layer. Manual surgery on
        # wrapper internals is version-sensitive and can accidentally remove a requested stage.
        self.truncate = bool(truncate)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        if len(features) != 3:
            raise RuntimeError(f"Expected 3 backbone features, got {len(features)}")
        return features[0], features[1], features[2]
