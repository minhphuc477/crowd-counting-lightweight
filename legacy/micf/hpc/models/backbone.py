from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature backbone returning a configured feature pyramid.

    R0--R5 consume reductions 4, 8 and 16 and retain the original physical
    truncation. R6/NPAC additionally consumes the final reduction-32 feature,
    so the complete MobileNetV4 representation participates in the graph.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = False,
        target_reductions: Tuple[int, ...] = (4, 8, 16),
    ):
        super().__init__()
        import timm

        target_reductions = tuple(int(r) for r in target_reductions)
        if target_reductions not in {(4, 8, 16), (4, 8, 16, 32)}:
            raise ValueError(
                "MobileNetV4Backbone requires target_reductions=(4, 8, 16) "
                f"or (4, 8, 16, 32), got {target_reductions}"
            )

        self.model_name = model_name
        self.pretrained = bool(pretrained)
        self.target_reductions = target_reductions

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

        # timm's features_only wrapper returns only the requested tensors but
        # still carries every MobileNet stage. Drop only stages downstream of
        # the last requested feature. This preserves the compact R0--R5 C16
        # graph while keeping all stages for R6's requested C32 feature.
        selected_module_names = list(self.backbone.feature_info.module_name())
        last_module = selected_module_names[-1]
        if not last_module.startswith("blocks."):
            raise RuntimeError(
                f"Cannot safely truncate {model_name}: last selected feature is {last_module!r}"
            )
        last_block_index = int(last_module.split(".")[1])
        blocks = list(self.backbone.blocks.children())
        if last_block_index >= len(blocks):
            raise RuntimeError(
                f"Invalid truncation block {last_block_index} for {len(blocks)} MobileNet stages"
            )
        self.backbone.blocks = nn.Sequential(*blocks[: last_block_index + 1])
        self.truncated_after = last_module

    def forward(self, x: torch.Tensor):
        return tuple(self.backbone(x))
