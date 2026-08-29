from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature backbone returning features at reductions 4, 8, 16.

    Uses timm's ``features_only=True`` with explicit ``out_indices`` to instantiate
    and execute only the required feature stages.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = False,
        target_reductions: Tuple[int, ...] = (4, 8, 16),
    ):
        super().__init__()
        import timm

        self.model_name = model_name
        self.target_reductions = tuple(int(r) for r in target_reductions)

        # Isolate RNG state when creating the probe model so feature inspection does not consume RNG
        with torch.random.fork_rng(devices=[]):
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
