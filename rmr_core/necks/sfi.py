from __future__ import annotations

import math
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
        norm_free_hf: bool = False,
        use_hdc_proj: bool = False,
        hdc_proj_groups: int = 2,
        use_laplacian: bool = True,
        use_selective_gate: bool = True,
        use_hdc: bool = True,
        decomp_mode: str = "laplacian",
        gate_mode: str = "selective",
        hf_norm_mode: str = "none",
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = int(width)
        self.use_gap = bool(use_gap)
        self.norm_free_hf = bool(norm_free_hf)
        self.use_hdc_proj = bool(use_hdc_proj)
        self.hdc_proj_groups = max(1, int(hdc_proj_groups))
        self.use_laplacian = bool(use_laplacian)
        self.use_hdc = bool(use_hdc) and len(hdc_dilations) > 0
        self.decomp_mode = str(decomp_mode).lower().strip()
        self.gate_mode = str(gate_mode).lower().strip() if use_selective_gate else "additive"
        self.hf_norm_mode = str(hf_norm_mode).lower().strip()
        if self.gate_mode == "none":
            self.gate_mode = "additive"
        self.use_selective_gate = self.gate_mode != "additive"

        self.aspp_gap_mode = str(aspp_gap_mode).lower().strip()
        if self.aspp_gap_mode not in ("add", "gate"):
            raise ValueError(f"aspp_gap_mode must be 'add' or 'gate', got '{self.aspp_gap_mode}'")
        if self.decomp_mode not in ("laplacian", "gaussian", "bilateral", "haar", "learnable_lp"):
            raise ValueError(f"Unsupported decomp_mode: '{self.decomp_mode}'")
        if self.gate_mode not in ("selective", "asff", "se", "cbam", "concat_conv", "additive"):
            raise ValueError(f"Unsupported gate_mode: '{self.gate_mode}'")
        if self.hf_norm_mode not in ("none", "gn", "bn", "bias"):
            raise ValueError(f"Unsupported hf_norm_mode: '{self.hf_norm_mode}'")

        # Optional mechanism-capture hooks for Layer-4 evaluation (disabled during training)
        self._capture_mechanisms: bool = False
        self._last_l4: torch.Tensor | None = None
        self._last_p4: torch.Tensor | None = None
        self._last_gate4: torch.Tensor | None = None

        self.lat4 = ConvGNAct(c4, self.width, 1)
        self.lat8 = ConvGNAct(c8, self.width, 1)
        self.lat16 = ConvGNAct(c16, self.width, 1)

        if self.use_hdc:
            self.hdc_blocks = nn.ModuleList([
                DepthwiseDilated(self.width, dilation=d) for d in hdc_dilations
            ])
        else:
            self.hdc_blocks = nn.ModuleList()

        if self.use_hdc_proj and self.use_hdc:
            groups = self.hdc_proj_groups if (self.width % self.hdc_proj_groups == 0) else 1
            self.hdc_proj: nn.Module | None = ConvGNAct(self.width, self.width, k=1, groups=groups)
        else:
            self.hdc_proj = None

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

        # Frequency decomposition helper buffers/layers
        if self.use_laplacian and self.decomp_mode == "gaussian":
            coords = torch.arange(5, dtype=torch.float32) - 2.0
            g1d = torch.exp(-(coords ** 2) / 2.0)
            g2d = (g1d[:, None] * g1d[None, :])
            g2d = (g2d / g2d.sum()).view(1, 1, 5, 5).repeat(self.width, 1, 1, 1)
            self.register_buffer("_gauss_kernel", g2d, persistent=False)
        if self.use_laplacian and self.decomp_mode == "learnable_lp":
            self.lp_conv: nn.Conv2d | None = nn.Conv2d(
                self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False
            )
            nn.init.constant_(self.lp_conv.weight, 1.0 / 9.0)
        else:
            self.lp_conv = None

        # High-frequency impulse enhancer on L4_hf (omitted if use_laplacian=False to avoid dead params)
        if self.use_laplacian:
            if self.hf_norm_mode == "gn":
                self.hf_enhance: nn.Module | None = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
                    _gn(self.width),
                )
            elif self.hf_norm_mode == "bn":
                self.hf_enhance = nn.Sequential(
                    nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
                    nn.BatchNorm2d(self.width),
                )
            elif self.hf_norm_mode == "bias":
                self.hf_enhance = nn.Conv2d(
                    self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=True
                )
            elif self.norm_free_hf:
                self.hf_enhance = nn.Conv2d(
                    self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False
                )
            else:
                self.hf_enhance = DepthwiseDilated(self.width, dilation=1)
            self.gamma_hf: nn.Parameter | None = nn.Parameter(torch.full((1, self.width, 1, 1), 0.5))
        else:
            self.hf_enhance = None
            self.gamma_hf = None

        # Context inheritance gate variants (Axis A/B)
        self.selective_gate: nn.Module | None = None
        self.asff_gate: nn.Module | None = None
        self.se_gate: nn.Module | None = None
        self.cbam_spatial: nn.Module | None = None
        self.concat_fuse: nn.Module | None = None

        if self.gate_mode == "selective":
            self.selective_gate = nn.Sequential(
                nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
                _gn(self.width),
                nn.SiLU(inplace=True),
                nn.Conv2d(self.width, self.width, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )
            gate_proj: nn.Conv2d = self.selective_gate[-2]  # type: ignore[assignment]
            nn.init.zeros_(gate_proj.bias)
        elif self.gate_mode == "asff":
            self.asff_gate = nn.Sequential(
                ConvGNAct(2 * self.width, self.width, k=1),
                nn.Conv2d(self.width, 2, kernel_size=1, bias=True),
            )
        elif self.gate_mode == "se":
            mid = max(4, self.width // 4)
            self.se_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(self.width, mid, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(mid, self.width, bias=True),
                nn.Sigmoid(),
            )
        elif self.gate_mode == "cbam":
            mid = max(4, self.width // 4)
            self.se_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(1),
                nn.Linear(self.width, mid, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(mid, self.width, bias=True),
                nn.Sigmoid(),
            )
            self.cbam_spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
                nn.Sigmoid(),
            )
        elif self.gate_mode == "concat_conv":
            self.concat_fuse = ConvGNAct(2 * self.width, self.width, k=1)

    @staticmethod
    def decompose_frequency(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose feature map into smooth low-frequency base and Laplacian HF impulse residual."""
        x_lf = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x_hf = x - x_lf
        return x_lf, x_hf

    def _decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decomp_mode == "laplacian":
            return self.decompose_frequency(x)
        if self.decomp_mode == "gaussian":
            x_lf = F.conv2d(x, self._gauss_kernel, padding=2, groups=self.width)  # type: ignore[arg-type]
            return x_lf, x - x_lf
        if self.decomp_mode == "haar":
            pooled = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_lf = F.interpolate(pooled, size=x.shape[-2:], mode="nearest")
            return x_lf, x - x_lf
        if self.decomp_mode == "learnable_lp" and self.lp_conv is not None:
            x_lf = self.lp_conv(x)
            return x_lf, x - x_lf
        # Bilateral filter (3x3 spatial-intensity window, sigma_r=0.5)
        pad_x = F.pad(x, (1, 1, 1, 1), mode="reflect")
        h_in, w_in = x.shape[-2:]
        num = torch.zeros_like(x)
        den = torch.zeros_like(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nbr = pad_x[:, :, 1 + dy : 1 + dy + h_in, 1 + dx : 1 + dx + w_in]
                sp_w = 1.0 if (dy == 0 and dx == 0) else math.exp(-0.5 * (dy * dy + dx * dx))
                rng_w = torch.exp(-0.5 * ((nbr - x) / 0.5).square())
                w_tot = sp_w * rng_w
                num = num + w_tot * nbr
                den = den + w_tot
        x_lf = num / den.clamp_min(1e-6)
        return x_lf, x - x_lf

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        if self.use_hdc and len(self.hdc_blocks) > 0:
            h = l16
            hdc_accum = torch.zeros_like(l16)
            for block in self.hdc_blocks:
                h = block(h)
                hdc_accum = hdc_accum + h
            if self.use_hdc_proj:
                hdc_accum = hdc_accum * (1.0 / float(len(self.hdc_blocks)))
        else:
            hdc_accum = torch.zeros_like(l16)

        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)
            if self.aspp_gap_mode == "gate":
                hdc_accum = hdc_accum * (1.0 + torch.tanh(gap_broadcast))
            else:
                hdc_accum = hdc_accum + gap_broadcast

        p16_in = l16 + hdc_accum
        if self.hdc_proj is not None:
            p16_in = self.hdc_proj(p16_in)
        p16 = self.ref16(p16_in)

        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )

        if self.use_laplacian:
            l4_lf, l4_hf = self._decompose(l4)
        else:
            l4_lf, l4_hf = l4, None

        gate4: torch.Tensor | None = None
        if self.gate_mode == "selective" and self.selective_gate is not None:
            gate4 = self.selective_gate(l4_lf + up8_to_4)
            fused_base = l4 + gate4 * up8_to_4
        elif self.gate_mode == "asff" and self.asff_gate is not None:
            w_asff = torch.softmax(self.asff_gate(torch.cat([l4_lf, up8_to_4], dim=1)), dim=1)
            gate4 = w_asff[:, 1:2]
            fused_base = w_asff[:, 0:1] * l4 + gate4 * up8_to_4
        elif self.gate_mode == "se" and self.se_gate is not None:
            gate4 = self.se_gate(l4_lf + up8_to_4).unsqueeze(-1).unsqueeze(-1)
            fused_base = l4 + gate4 * up8_to_4
        elif self.gate_mode == "cbam" and self.se_gate is not None and self.cbam_spatial is not None:
            ch_g = self.se_gate(l4_lf + up8_to_4).unsqueeze(-1).unsqueeze(-1)
            ctx = up8_to_4 * ch_g
            sp_in = torch.cat([ctx.mean(dim=1, keepdim=True), ctx.amax(dim=1, keepdim=True)], dim=1)
            sp_g = self.cbam_spatial(sp_in)
            gate4 = ch_g * sp_g
            fused_base = l4 + gate4 * up8_to_4
        elif self.gate_mode == "concat_conv" and self.concat_fuse is not None:
            fused_base = self.concat_fuse(torch.cat([l4_lf, up8_to_4], dim=1))
        else:
            fused_base = l4 + up8_to_4

        if self.use_laplacian and self.hf_enhance is not None and self.gamma_hf is not None and l4_hf is not None:
            is_norm_free_act = self.norm_free_hf or (self.hf_norm_mode in ("gn", "bn", "bias"))
            if is_norm_free_act:
                hf_raw = torch.tanh(self.hf_enhance(l4_hf))
                hf_term = (self.gamma_hf * gate4 * hf_raw) if gate4 is not None else (self.gamma_hf * hf_raw)
            else:
                hf_term = self.gamma_hf * self.hf_enhance(l4_hf)
            f4 = fused_base + hf_term
        else:
            f4 = fused_base

        p4 = self.ref4(f4)

        if self._capture_mechanisms:
            self._last_l4 = l4.detach()
            self._last_p4 = p4.detach()
            self._last_gate4 = gate4.detach() if gate4 is not None else None

        return p4, p8, p16
