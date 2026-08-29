"""HPC-Lite / NTPC Lightweight Crowd Counter (~0.35M Parameters).

Architecture:
  Image -> MobileNetV4 features (C4, C8, C16)
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
    ):
        super().__init__()
        from .blocks import RepDWBlock

        self.eps_d = float(eps_d)
        self.neck_width = int(neck_width)
        self.output_stride = int(output_stride)
        self.use_p8_context = bool(use_p8_context)
        self.use_repblock = bool(use_repblock)

        if self.output_stride != 4:
            raise ValueError("Target and loss formulations assume output_stride=4")
        if not math.isfinite(self.eps_d) or self.eps_d < 1e-8:
            raise ValueError(f"eps_d must be finite and at least 1e-8 for stable FP32 mass, got {self.eps_d}")

        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            target_reductions=(4, 8, 16),
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
        c4, c8, c16 = self.backbone(x)
        p4 = self.neck(c4, c8, c16)
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

        c4, c8, c16 = self.backbone(x)
        p4, aux = self.neck(c4, c8, c16, return_routes=True)

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
