from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, DepthwiseDilated, _gn


class SFILiteFPNNeck(nn.Module):
    """Selective Frequency-Inheritance FPN Neck (SFI-Lite).

    Addresses scale-frequency entanglement in lightweight crowd counting pyramids:
      1. Gridding-free sequential Hybrid Dilated Convolutions (default [1, 2, 3])
         on L16 without spatial-diluting Global Average Pooling by default.
      2. Orthogonal high-/low-frequency decomposition on Stride-4 features L4
         to isolate 2px-4px crowd head impulses (L4_hf) from regional background
         context (L4_lf).
      3. Spatially selective context inheritance gate G4(x) in (0, 1) driven by
         (L4_lf + up8_to_4) so coarse P8 context is inherited selectively without
         smearing dense high-frequency head peaks at Stride 4.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        hdc_dilations: tuple[int, ...] = (1, 2, 3),
        use_gap: bool = False,
        aspp_gap_mode: str = "gate",
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = int(width)
        self.use_gap = bool(use_gap)
        self.aspp_gap_mode = str(aspp_gap_mode).lower().strip()
        if self.aspp_gap_mode not in ("add", "gate"):
            raise ValueError(f"aspp_gap_mode must be 'add' or 'gate', got '{self.aspp_gap_mode}'")

        self.lat4 = ConvGNAct(c4, self.width, 1)
        self.lat8 = ConvGNAct(c8, self.width, 1)
        self.lat16 = ConvGNAct(c16, self.width, 1)

        self.hdc_blocks = nn.ModuleList([
            DepthwiseDilated(self.width, dilation=d) for d in hdc_dilations
        ])
        if self.use_gap:
            self.aspp_gap: nn.Sequential | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(self.width, self.width, bias=True),
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None

        self.ref16 = DSResidual(self.width)
        self.ref8 = DSResidual(self.width)
        self.ref4 = DSResidual(self.width)

        # High-frequency impulse enhancer on Laplacian residual L4_hf
        self.hf_enhance = DepthwiseDilated(self.width, dilation=1)
        self.gamma_hf = nn.Parameter(torch.full((1, self.width, 1, 1), 0.5))

        # Spatially selective context inheritance gate G4(x) in (0, 1)
        self.selective_gate = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
            _gn(self.width),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.width, self.width, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        gate_proj: nn.Conv2d = self.selective_gate[-2]  # type: ignore[assignment]
        nn.init.zeros_(gate_proj.bias)

    @staticmethod
    def decompose_frequency(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose feature map into smooth low-frequency base and Laplacian HF impulse residual."""
        x_lf = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_hf = x - x_lf
        return x_lf, x_hf

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        h = l16
        hdc_accum = torch.zeros_like(l16)
        for block in self.hdc_blocks:
            h = block(h)
            hdc_accum = hdc_accum + h

        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            if self.aspp_gap_mode == "gate":
                hdc_accum = hdc_accum * (1.0 + torch.tanh(gap_broadcast))
            else:
                hdc_accum = hdc_accum + gap_broadcast

        p16 = self.ref16(l16 + hdc_accum)

        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        l4_lf, l4_hf = self.decompose_frequency(l4)
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        gate4 = self.selective_gate(l4_lf + up8_to_4)
        f4 = l4 + self.gamma_hf * self.hf_enhance(l4_hf) + gate4 * up8_to_4
        p4 = self.ref4(f4)

        return p4, p8, p16
