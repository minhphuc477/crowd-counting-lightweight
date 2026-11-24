from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated, _gn


class _LinearDSResidual(nn.Module):
    """Depthwise-separable residual block without trailing ReLU/SiLU clipping."""

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.n1 = _gn(c)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.n2 = _gn(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.n2(self.pw(self.act(self.n1(self.dw(x)))))
        return x + y


class LFIS2FPNNeck(nn.Module):
    """Nyquist-Preserving Stride-2 Laplacian Frequency-Inheritance FPN Neck (LFI-S2).

    Consumes backbone features (C2, C4, C8, C16) so that dense high-frequency head impulses
    at 2-4px spacing are extracted at native Stride 2 BEFORE MobileNetV4 Stride-4 downsampling
    causes Nyquist aliasing, and without tanh amplitude saturation.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int, int] = (32, 16, 32, 48),
        width: int = 32,
        s2_width: int = 16,
        hdc_dilations: tuple[int, ...] = (1, 2, 3),
        use_hdc_proj: bool = True,
        hdc_proj_groups: int = 2,
        hf_act_mode: str = "linear",
        pad_aware_laplacian: bool = True,
    ) -> None:
        super().__init__()
        c2, c4, c8, c16 = in_channels
        self.width = int(width)
        self.s2_width = int(s2_width)
        self.hf_act_mode = str(hf_act_mode).lower().strip()
        self.pad_aware_laplacian = bool(pad_aware_laplacian)
        if self.hf_act_mode not in ("linear", "silu", "tanh"):
            raise ValueError(f"Unsupported hf_act_mode: '{self.hf_act_mode}'")

        self.lat2 = ConvGNAct(c2, self.s2_width, 1)
        self.lat4 = ConvGNAct(c4, self.width, 1)
        self.lat8 = ConvGNAct(c8, self.width, 1)
        self.lat16 = ConvGNAct(c16, self.width, 1)

        self.hdc_blocks = nn.ModuleList([DepthwiseDilated(self.width, dilation=d) for d in hdc_dilations])
        if use_hdc_proj and len(hdc_dilations) > 0:
            groups = max(1, int(hdc_proj_groups)) if (self.width % max(1, int(hdc_proj_groups)) == 0) else 1
            self.hdc_proj: nn.Module | None = ConvGNAct(self.width, self.width, k=1, groups=groups)
        else:
            self.hdc_proj = None

        self.ref16 = DSResidual(self.width)
        self.ref8 = DSResidual(self.width)
        self.ref4 = DSResidual(self.width)
        self.ref2 = _LinearDSResidual(self.s2_width)

        self.hf_enhance = nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False)
        self.gamma_hf = nn.Parameter(torch.full((1, self.width, 1, 1), 0.5))

        self.p4_to_s2 = ConvGNAct(self.width, self.s2_width, 1)
        self.hf2_enhance = nn.Conv2d(
            self.s2_width, self.s2_width, kernel_size=3, padding=1, groups=self.s2_width, bias=False
        )
        self.gamma_s2 = nn.Parameter(torch.full((1, self.s2_width, 1, 1), 0.5))

    def _activate_hf(self, x: torch.Tensor) -> torch.Tensor:
        if self.hf_act_mode == "tanh":
            return torch.tanh(x)
        if self.hf_act_mode == "silu":
            return F.silu(x)
        return x

    def forward(
        self, c2: torch.Tensor, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l2 = self.lat2(c2)
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        if len(self.hdc_blocks) > 0:
            h = l16
            hdc_accum = torch.zeros_like(l16)
            for block in self.hdc_blocks:
                h = block(h)
                hdc_accum = hdc_accum + h
            p16_in = l16 + hdc_accum * (1.0 / float(len(self.hdc_blocks)))
        else:
            p16_in = l16
        if self.hdc_proj is not None:
            p16_in = self.hdc_proj(p16_in)
        p16 = self.ref16(p16_in)

        p8 = self.ref8(l8 + F.interpolate(p16, size=l8.shape[-2:], mode="bilinear", align_corners=False))
        up8_to_4 = F.interpolate(p8, size=l4.shape[-2:], mode="bilinear", align_corners=False)

        l4_lf = F.avg_pool2d(l4, kernel_size=3, stride=1, padding=1, count_include_pad=(not self.pad_aware_laplacian))
        l4_hf = l4 - l4_lf
        f4 = (l4 + up8_to_4) + self.gamma_hf * self._activate_hf(self.hf_enhance(l4_hf))
        p4 = self.ref4(f4)

        l2_lf = F.avg_pool2d(l2, kernel_size=3, stride=1, padding=1, count_include_pad=(not self.pad_aware_laplacian))
        l2_hf = l2 - l2_lf
        up4_to_2 = F.interpolate(self.p4_to_s2(p4), size=l2.shape[-2:], mode="bilinear", align_corners=False)
        f2 = (l2_lf + up4_to_2) + self.gamma_s2 * self._activate_hf(self.hf2_enhance(l2_hf))
        p2 = self.ref2(f2)

        return p2, p8, p16
