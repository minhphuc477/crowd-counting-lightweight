"""HPC-Lite / NTPC/NPAC lightweight crowd counter.

Architecture:
  Image -> MobileNetV4 features (C4, C8, C16[, C32])
        -> Additive FPN Neck (32 channels, context dilations {1,2,3})
        -> GroupNorm + SiLU + 1x1 Conv mass head
        -> Continuous positive count-mass map D @ stride 4 (Float32).
"""

from __future__ import annotations

import math
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV4Backbone
from .blocks import make_group_norm
from .neck import AdditiveFPNNeck


def inv_softplus(y: float) -> float:
    """Numerically stable inverse of softplus: softplus^{-1}(y) = y + log(1 - exp(-y))."""
    y = max(float(y), 1e-12)
    return y + math.log(-math.expm1(-y))


class HPCLite(nn.Module):
    """HPC-Lite crowd counter with MobileNetV4 backbone and Additive FPN neck."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = False,
        neck_width: int = 32,
        context_dilations: Tuple[int, ...] = (1, 2, 3),
        use_p8_context: bool = False,
        use_repblock: bool = False,
        eps_d: float = 1e-8,
        output_stride: int = 4,
        feature_reductions: Tuple[int, ...] = (4, 8, 16),
    ):
        super().__init__()
        from .blocks import RepDWBlock

        self.eps_d = float(eps_d)
        self.neck_width = int(neck_width)
        self.output_stride = int(output_stride)
        self.use_p8_context = bool(use_p8_context)
        self.use_repblock = bool(use_repblock)
        self.feature_reductions = tuple(int(r) for r in feature_reductions)

        if self.output_stride != 4:
            raise ValueError("Target and loss formulations assume output_stride=4")
        if not math.isfinite(self.eps_d) or self.eps_d < 1e-8:
            raise ValueError(f"eps_d must be finite and at least 1e-8 for stable FP32 mass, got {self.eps_d}")

        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            target_reductions=self.feature_reductions,
        )

        self.neck = AdditiveFPNNeck(
            in_channels=self.backbone.out_channels,
            width=neck_width,
            context_dilations=context_dilations,
            use_p8_context=self.use_p8_context,
        )

        if self.use_repblock:
            self.head_refine = RepDWBlock(neck_width, act=True)
            self.head_dw = None
            self.head_norm = None
            self.head_act = None
        else:
            self.head_refine = None
            self.head_dw = nn.Conv2d(
                neck_width,
                neck_width,
                kernel_size=3,
                padding=1,
                groups=neck_width,
                bias=False,
            )
            self.head_norm = make_group_norm(neck_width)
            self.head_act = nn.SiLU(inplace=True)

        self.head_out = nn.Conv2d(neck_width, 1, kernel_size=1, bias=True)

        # Conservative initialization: near-zero output count per cell
        nn.init.constant_(self.head_out.bias, -6.0)
        nn.init.normal_(self.head_out.weight, std=0.01)

    def switch_to_deploy(self) -> None:
        """Fuse structural reparameterization branches into a single Conv2d for deployment."""
        if self.use_repblock and self.head_refine is not None:
            self.head_refine.switch_to_deploy()

    def init_head_bias_from_data(
        self,
        mean_crop_count: float,
        crop_size: int,
        output_stride: int = 4,
    ) -> None:
        """Data-driven head bias initialization using inverse softplus."""
        grid_h = math.ceil(crop_size / output_stride)
        grid_w = math.ceil(crop_size / output_stride)
        n_cells = max(grid_h * grid_w, 1)
        m0 = max(float(mean_crop_count) / n_cells, 1e-8)
        m0 = min(m0, 1e4)
        with torch.no_grad():
            nn.init.constant_(self.head_out.bias, inv_softplus(m0))

    def forward_mass(self, x: torch.Tensor) -> torch.Tensor:
        """Branchless mass forward pass for clean tracing and ONNX deployment."""
        features = self.backbone(x)
        p4 = self.neck(*features)
        if self.use_repblock:
            h = self.head_refine(p4)
        else:
            h = self.head_act(self.head_norm(self.head_dw(p4)))
        z = self.head_out(h)
        return F.softplus(z.float()).clamp_min(self.eps_d)

    def forward(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict]]:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected 4D image input (B, 3, H, W), got {tuple(x.shape)}")

        if not return_aux:
            return self.forward_mass(x)

        features = self.backbone(x)
        p4, aux = self.neck(*features, return_routes=True)

        if self.use_repblock:
            h = self.head_refine(p4)
        else:
            h = self.head_act(self.head_norm(self.head_dw(p4)))

        z = self.head_out(h)
        d_map = F.softplus(z.float()).clamp_min(self.eps_d)
        return d_map, aux

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        pad_multiple: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Variable-resolution inference.

        If pad_multiple is None (default for official PyTorch evaluation), processes arbitrary
        resolutions directly without zero-padding distortion on GroupNorm statistics.
        If pad_multiple is an integer, pads input to multiples of pad_multiple and crops output.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor input, got {tuple(x.shape)}")

        if pad_multiple is None:
            d = self.forward_mass(x)
            count = d.sum(dim=(-1, -2, -3))
            return count, d

        if isinstance(pad_multiple, bool) or not isinstance(pad_multiple, int) or pad_multiple <= 0:
            raise ValueError(
                f"pad_multiple must be None or a positive integer, got {pad_multiple!r}"
            )

        _, _, h, w = x.shape
        pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
        pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        d_padded = self.forward_mass(x)
        out_h = math.ceil(h / self.output_stride)
        out_w = math.ceil(w / self.output_stride)

        d_valid = d_padded[..., :out_h, :out_w]
        count = d_valid.sum(dim=(-1, -2, -3))
        return count, d_valid

    @torch.no_grad()
    def predict_tiled(
        self,
        x: torch.Tensor,
        tile_size: int,
        halo: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Explicit memory-bounded inference using disjoint cores and context halos.

        ``tile_size`` and ``halo`` are aligned to the backbone's maximum stride,
        so stitched stride-4 cells retain a consistent global phase. Each
        output cell is written exactly once. GroupNorm statistics are tile-local,
        therefore this mode is deterministic but not numerically equivalent to
        full-image inference and must be reported as a separate protocol.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor input, got {tuple(x.shape)}")
        max_stride = max(self.feature_reductions)
        for name, value in (("tile_size", tile_size), ("halo", halo)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}")
        if tile_size <= 0 or tile_size % max_stride != 0:
            raise ValueError(f"tile_size must be a positive multiple of {max_stride}")
        if halo < 0 or halo % max_stride != 0:
            raise ValueError(f"halo must be a non-negative multiple of {max_stride}")

        batch, _, height, width = x.shape
        if height <= tile_size and width <= tile_size:
            return self.predict(x, pad_multiple=None)

        out_h = math.ceil(height / self.output_stride)
        out_w = math.ceil(width / self.output_stride)
        stitched = x.new_empty((batch, 1, out_h, out_w), dtype=torch.float32)
        for core_y0 in range(0, height, tile_size):
            core_y1 = min(core_y0 + tile_size, height)
            tile_y0 = max(0, core_y0 - halo)
            tile_y1 = min(height, core_y1 + halo)
            for core_x0 in range(0, width, tile_size):
                core_x1 = min(core_x0 + tile_size, width)
                tile_x0 = max(0, core_x0 - halo)
                tile_x1 = min(width, core_x1 + halo)
                tile_mass = self.forward_mass(x[..., tile_y0:tile_y1, tile_x0:tile_x1])

                global_y0 = core_y0 // self.output_stride
                global_y1 = math.ceil(core_y1 / self.output_stride)
                global_x0 = core_x0 // self.output_stride
                global_x1 = math.ceil(core_x1 / self.output_stride)
                local_y0 = (core_y0 - tile_y0) // self.output_stride
                local_x0 = (core_x0 - tile_x0) // self.output_stride
                local_y1 = local_y0 + (global_y1 - global_y0)
                local_x1 = local_x0 + (global_x1 - global_x0)
                stitched[..., global_y0:global_y1, global_x0:global_x1] = tile_mass[
                    ..., local_y0:local_y1, local_x0:local_x1
                ]
        count = stitched.sum(dim=(-1, -2, -3))
        return count, stitched
