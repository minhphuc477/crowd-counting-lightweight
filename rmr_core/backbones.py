from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _StridedShortcutUIB(nn.Module):
    """Wrap a timm UniversalInvertedResidual block to downsample both residual & shortcut."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        if not hasattr(block, "dw_mid") or not isinstance(getattr(block.dw_mid, "conv", None), nn.Conv2d):
            raise ValueError(f"Expected UIB block with dw_mid.conv Conv2d, got {type(block).__name__}")
        self._use_pool_skip = bool(getattr(block, "has_skip", False))
        block.has_skip = False
        block.dw_mid.conv.stride = (2, 2)
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self._use_pool_skip:
            skip = F.avg_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
            if skip.shape[-2:] != out.shape[-2:]:
                skip = skip[..., : out.shape[-2], : out.shape[-1]]
            return out + skip
        return out


def _set_conv_dilation(conv: nn.Conv2d, dilation: int) -> None:
    kh, kw = conv.kernel_size
    conv.dilation = (dilation, dilation)
    conv.padding = (dilation * (kh - 1) // 2, dilation * (kw - 1) // 2)


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
        stride_mode: str = "standard",
        split_block_idx: int = 3,
        max_stage2_blocks: int = 0,
    ):
        super().__init__()
        import timm

        target_reductions = tuple(int(r) for r in target_reductions)
        if target_reductions not in ((4, 8, 16), (2, 4, 8, 16)):
            raise ValueError(
                f"TimmPyramidBackbone requires target_reductions=(4, 8, 16) or (2, 4, 8, 16), got {target_reductions}"
            )
        self.stride_mode = str(stride_mode).lower().strip()
        if self.stride_mode not in ("standard", "split_s8_s16", "dilated_os8", "os8_hdc"):
            raise ValueError(
                f"stride_mode must be 'standard', 'split_s8_s16', 'dilated_os8', or 'os8_hdc', got '{stride_mode}'"
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

        self.split_s8_blocks: nn.Sequential | None = None
        self.split_s16_blocks: nn.Sequential | None = None
        if int(max_stage2_blocks) > 0:
            if not hasattr(self.backbone, "blocks") or len(self.backbone.blocks) < 3:
                raise ValueError(f"max_stage2_blocks={max_stage2_blocks} requires blocks >= 3 in {model_name}")
            b2_all = list(self.backbone.blocks[2].children())
            n_keep = int(max_stage2_blocks)
            if not (1 <= n_keep <= len(b2_all)):
                raise ValueError(f"max_stage2_blocks={n_keep} out of range for {len(b2_all)} blocks")
            self.backbone.blocks[2] = nn.Sequential(*b2_all[:n_keep])

        if self.stride_mode in ("split_s8_s16", "dilated_os8", "os8_hdc"):
            if (
                not hasattr(self.backbone, "blocks")
                or getattr(self.backbone, "feature_hooks", None) is not None
                or len(self.backbone.blocks) < 3
            ):
                raise ValueError(f"stride_mode='{self.stride_mode}' requires sequential blocks >= 3 in {model_name}")
            b2_list = list(self.backbone.blocks[2].children())
            b0 = b2_list[0]
            if not hasattr(b0, "dw_mid") or getattr(b0.dw_mid.conv, "stride", None) != (2, 2):
                raise ValueError(f"Expected blocks[2][0].dw_mid.conv stride=(2,2) in {model_name}")
            b0.dw_mid.conv.stride = (1, 1)
            if self.stride_mode == "split_s8_s16":
                split_idx = int(split_block_idx)
                if not (1 <= split_idx < len(b2_list)):
                    raise ValueError(f"split_block_idx={split_idx} out of range for {len(b2_list)} blocks")
                b2_list[split_idx] = _StridedShortcutUIB(b2_list[split_idx])
                self.split_s8_blocks = nn.Sequential(*b2_list[:split_idx])
                self.split_s16_blocks = nn.Sequential(*b2_list[split_idx:])
                self.backbone.blocks = nn.Sequential(*list(self.backbone.blocks.children())[:2])
                ch_list = list(self.out_channels)
                ch_list[-2] = ch_list[-1]
                self.out_channels = tuple(ch_list)
            else:
                hdc_cycle = (1, 2, 3)
                for idx, blk in enumerate(b2_list[1:]):
                    d_rate = hdc_cycle[idx % 3] if self.stride_mode == "os8_hdc" else 2
                    for attr in ("dw_start", "dw_mid", "dw_end"):
                        sub = getattr(blk, attr, None)
                        conv = getattr(sub, "conv", None) if sub is not None else None
                        if isinstance(conv, nn.Conv2d) and conv.kernel_size != (1, 1):
                            _set_conv_dilation(conv, d_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.stride_mode == "split_s8_s16" and self.split_s8_blocks is not None and self.split_s16_blocks is not None:
            base_feats = tuple(self.backbone(x))
            c8_rich = self.split_s8_blocks(base_feats[-1])
            c16 = self.split_s16_blocks(c8_rich)
            return base_feats[:-1] + (c8_rich, c16)
        feats = tuple(self.backbone(x))
        return tuple(feats[: len(self.target_reductions)])


# Backward compatibility alias
MobileNetV4Backbone = TimmPyramidBackbone
