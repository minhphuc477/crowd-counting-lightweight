from __future__ import annotations

import torch
import torch.nn as nn


class TimmPyramidBackbone(nn.Module):
    """Pyramid feature backbone returning features at target reductions (C4, C8, C16).

    Probes feature_info.reduction() dynamically to find target reductions {4, 8, 16},
    selects actual channel dimensions, and conditionally truncates stages when supported
    (e.g. MobileNetV4 blocks). Supports diverse model families in timm (MobileNetV4,
    ConvNeXt, ResNet, Swin).
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
        if target_reductions != (4, 8, 16):
            raise ValueError(
                f"TimmPyramidBackbone requires target_reductions=(4, 8, 16), got {target_reductions}"
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
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=self.pretrained,
                features_only=True,
                out_indices=self.selected_indices,
            )
        except Exception as e:
            if self.pretrained:
                import warnings
                warnings.warn(
                    f"Failed to load pretrained weights for '{model_name}' ({e}). "
                    "Falling back to randomly initialized backbone for offline operation.",
                    UserWarning,
                )
                self.pretrained = False
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=False,
                    features_only=True,
                    out_indices=self.selected_indices,
                )
            else:
                raise

        selected_module_names = list(self.backbone.feature_info.module_name())
        last_module = selected_module_names[-1]
        self.truncated_after = last_module

        # Optional physical truncation for MobileNetV4 models with sequential blocks
        if last_module.startswith("blocks.") and hasattr(self.backbone, "blocks"):
            try:
                last_block_index = int(last_module.split(".")[1])
                blocks = list(self.backbone.blocks.children())
                if last_block_index < len(blocks):
                    self.backbone.blocks = nn.Sequential(*blocks[: last_block_index + 1])
            except Exception:
                pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = tuple(self.backbone(x))
        return feats[0], feats[1], feats[2]


# Backward compatibility alias
MobileNetV4Backbone = TimmPyramidBackbone
