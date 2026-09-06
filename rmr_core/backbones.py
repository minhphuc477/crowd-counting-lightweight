from __future__ import annotations

import torch
import torch.nn as nn


class MobileNetV4Backbone(nn.Module):
    """MobileNetV4 feature backbone returning a configured feature pyramid (C4, C8, C16).

    Probes feature_info.reduction() dynamically to find target reductions {4, 8, 16},
    selects actual channel dimensions, and physically truncates blocks after the last
    requested feature module.
    """

    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained: bool = False,
        target_reductions: tuple[int, ...] = (4, 8, 16),
    ):
        super().__init__()
        import timm

        target_reductions = tuple(int(r) for r in target_reductions)
        if target_reductions not in {(4, 8, 16), (4, 8, 16, 32)}:
            raise ValueError(
                f"MobileNetV4Backbone requires target_reductions=(4, 8, 16), got {target_reductions}"
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
        self.out_channels = tuple(selected_channels)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.selected_indices,
        )

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = tuple(self.backbone(x))
        return feats[0], feats[1], feats[2]
