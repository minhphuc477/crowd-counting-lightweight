from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual


class FrankensteinSOTALiteNeck(nn.Module):
    """Frankenstein SOTA-Lite Baseline Neck (Axis C Controlled Comparison).

    Assembles lightweight, budget-matched adaptations of three strong literature
    mechanisms within the <= 104,441 total parameter ceiling:
      1. FreqFusion-style ALPF/AHPF kernel-softmax spatial filtering (Chen et al., TPAMI 2024):
         Spatially variant 3x3 softmax weights applied to upsampled P8 (low-pass) and
         kernel-inverted high-pass residual on L4.
      2. HS-FPN-style High-Frequency Channel-Spatial Attention (Shi et al., AAAI 2025):
         2D FFT high-pass mask (alpha=0.25) driving GAP+GMP channel attention & spatial mask.
      3. STEERER-style Scale Disentanglement with BatchNorm (Han et al., ICCV 2023):
         CFN/UFN Conv+BN+ReLU branches with soft-mask inheritance A_c * C(P8).

    Purpose: Empirically test whether assembling strong individual literature modules
    with spatial normalization outperforms SFI-Lite's clean norm-free organization
    under the identical parameter budget.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = int(width)

        self._capture_mechanisms: bool = False
        self._last_l4: torch.Tensor | None = None
        self._last_p4: torch.Tensor | None = None
        self._last_gate4: torch.Tensor | None = None

        self.lat4 = ConvGNAct(c4, self.width, 1)
        self.lat8 = ConvGNAct(c8, self.width, 1)
        self.lat16 = ConvGNAct(c16, self.width, 1)

        # 1. STEERER-style CFN + UFN + Soft-Mask Generator (Conv + BN + ReLU) on P16->P8
        self.steerer_cfn = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
            nn.BatchNorm2d(self.width),
            nn.ReLU(inplace=True),
        )
        self.steerer_ufn = nn.Sequential(
            nn.Conv2d(self.width, self.width, kernel_size=3, padding=1, groups=self.width, bias=False),
            nn.BatchNorm2d(self.width),
            nn.ReLU(inplace=True),
        )
        self.steerer_smg = nn.Conv2d(self.width, 2, kernel_size=1, bias=True)

        # 2. FreqFusion-style ALPF/AHPF 3x3 spatial-variant kernel predictors on P8->P4
        self.alpf_pred = nn.Conv2d(self.width, 9, kernel_size=1, bias=True)
        self.ahpf_pred = nn.Conv2d(self.width, 9, kernel_size=1, bias=True)

        # 3. HS-FPN-style High-Frequency Channel-Spatial Perception on L4
        mid = max(4, self.width // 4)
        self.hs_ch_fc = nn.Sequential(
            nn.Linear(self.width, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, self.width, bias=True),
            nn.Sigmoid(),
        )
        self.hs_sp_conv = nn.Sequential(
            nn.Conv2d(self.width, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.ref16 = DSResidual(self.width)
        self.ref8 = DSResidual(self.width)
        self.ref4 = DSResidual(self.width)

    @staticmethod
    def _apply_dynamic_3x3(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Apply spatially variant 3x3 depthwise filter weights [B, 9, H, W] to x [B, C, H, W]."""
        b, c, h, w = x.shape
        x_unf = F.unfold(x, kernel_size=3, padding=1).view(b, c, 9, h, w)
        return (x_unf * weights.unsqueeze(1)).sum(dim=2)

    @staticmethod
    def _fft_highpass(x: torch.Tensor, alpha: float = 0.25) -> torch.Tensor:
        """2D FFT high-pass filter matching HS-FPN frequency mask."""
        h, w = x.shape[-2:]
        freq = torch.fft.rfft2(x.float(), norm="ortho")
        fy = torch.fft.fftfreq(h, device=x.device).abs().view(-1, 1)
        fx = torch.fft.rfftfreq(w, device=x.device).abs().view(1, -1)
        mask = ((fy >= alpha * 0.5) | (fx >= alpha * 0.5)).to(freq.dtype)
        hp = torch.fft.irfft2(freq * mask.unsqueeze(0).unsqueeze(0), s=(h, w), norm="ortho")
        return hp.to(x.dtype)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        p16 = self.ref16(l16)

        # STEERER-style scale disentanglement on up16_to_8
        up16_to_8 = F.interpolate(p16, size=l8.shape[-2:], mode="bilinear", align_corners=False)
        r8 = l8 + up16_to_8
        smg8 = torch.softmax(self.steerer_smg(r8), dim=1)
        p8 = self.ref8(smg8[:, 0:1] * self.steerer_cfn(r8) + smg8[:, 1:2] * self.steerer_ufn(r8))

        # FreqFusion-style ALPF on up8_to_4 & AHPF on L4
        up8_to_4 = F.interpolate(p8, size=l4.shape[-2:], mode="bilinear", align_corners=False)
        w_alpf = torch.softmax(self.alpf_pred(up8_to_4), dim=1)
        up8_lp = self._apply_dynamic_3x3(up8_to_4, w_alpf)

        w_ahpf_lp = torch.softmax(self.ahpf_pred(l4), dim=1)
        l4_hp = l4 - self._apply_dynamic_3x3(l4, w_ahpf_lp)

        # HS-FPN-style HF channel-spatial attention on L4
        hf_fft = self._fft_highpass(l4, alpha=0.25)
        ch_stat = hf_fft.mean(dim=(-2, -1)) + hf_fft.amax(dim=(-2, -1))
        ch_w = self.hs_ch_fc(ch_stat).unsqueeze(-1).unsqueeze(-1)
        sp_w = self.hs_sp_conv(hf_fft)
        l4_hs = l4 * ch_w + l4 * sp_w

        f4 = l4_hs + l4_hp + up8_lp
        p4 = self.ref4(f4)

        if self._capture_mechanisms:
            self._last_l4 = l4.detach()
            self._last_p4 = p4.detach()
            self._last_gate4 = sp_w.detach()

        return p4, p8, p16
