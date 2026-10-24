"""First-Principles Residual Dilated Fine Carrier Head (RD-FineHead) for RMR.

Resolves the 4 core pathologies identified in architectural audits:
1. Receptive Field Blindness: Stacked Dilated Depthwise Residuals (d=1, d=2) expand ERF
   from 12px to 28px, capturing anatomical shoulder/neck context to reject head-like
   inanimate distractors (fence knobs, rivets, stones).
2. Subpixel Phase Discontinuity: Replaces 1x1 conv with 3x3 polyphase spatial convolution
   prior to PixelShuffle(2), restoring weakly differentiable Sobolev measure continuity.
3. Gradient Attenuation: Employs strict unconditional linear residual shortcuts (x + F(x))
   providing an unattenuated gradient highway for Dirac point losses (Bayesian Point Loss).
4. Scale Decoupling: Modulates spatial features via zero-initialized 2D FiLM Conv2d
   with exact Step-0 Parity, preserving calibrated initial prior (m0 = 0.015763 count/cell).
"""
from __future__ import annotations

import math
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .density_activation import _smooth_floor, _softplus_inverse
from .necks.blocks import _gn

_FINE_HEAD_BIAS_INIT: float = math.log(math.exp(0.015763) - 1.0)  # ≈ -4.1422


class DDRBlock(nn.Module):
    """Dilated Depthwise-Separable Linear Residual Block with GroupNorm and SiLU.

    Guarantees an unconditional, uninhibited linear identity gradient highway:
        d(y) / d(x) = I + d(F) / d(x)
    """

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.dilation = int(dilation)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=self.dilation,
            dilation=self.dilation,
            groups=channels,
            bias=False,
        )
        self.gn1 = _gn(channels)
        self.act = nn.SiLU(inplace=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gn2 = _gn(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dw(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.pw(out)
        out = self.gn2(out)
        return x + out  # Unconditional linear residual addition


class ResidualDilatedFineHead(nn.Module):
    """First-Principles Residual Dilated Fine Carrier Head.

    Combines:
    - Stacked Dilated Depthwise Residuals (default dilations: 1, 2) -> ERF = 28px
    - 2D Spatial FiLM Scale Modulator with Step-0 Parity
    - 3x3 Continuous Spatial Projection prior to PixelShuffle(2)
    - Temperature-Calibrated Softplus with exact initial rate m0
    """

    def __init__(
        self,
        width: int = 48,
        num_scales: int = 3,
        subpixel_stride2: bool = True,
        subpixel_kernel_size: int = 3,
        dilations: Sequence[int] = (1, 2),
        use_film: bool = True,
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        temp_softplus: bool = True,
        floor_tau: float = 0.0,
        head_activation: str = "softplus",
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.num_scales = int(num_scales)
        self.subpixel_stride2 = bool(subpixel_stride2)
        self.use_film = bool(use_film)
        self.head_activation = str(head_activation).lower()
        self.floor_tau = float(floor_tau) / (4.0 if self.subpixel_stride2 else 1.0)
        self.temp_softplus = bool(temp_softplus)

        # 1. Stacked Dilated Depthwise Residual Blocks
        blocks: list[nn.Module] = []
        for d in dilations:
            blocks.append(DDRBlock(self.width, dilation=int(d)))
        self.refinement_blocks = nn.Sequential(*blocks)

        # 2. 2D Spatial FiLM Scale Modulator (Perez et al. AAAI 2018)
        if self.use_film and self.num_scales > 0:
            self.film_conv = nn.Conv2d(self.num_scales, 2 * self.width, kernel_size=1, bias=True)
            nn.init.zeros_(self.film_conv.weight)
            nn.init.zeros_(self.film_conv.bias)
        else:
            self.film_conv = None

        # 3. 3x3 Polyphase Spatial Pre-Shuffle Convolution (Shi et al. CVPR 2016)
        out_channels = 4 if self.subpixel_stride2 else 1
        pad = int(subpixel_kernel_size) // 2
        self.final_conv = nn.Conv2d(
            self.width,
            out_channels,
            kernel_size=int(subpixel_kernel_size),
            padding=pad,
            bias=True,
        )
        nn.init.normal_(self.final_conv.weight, std=0.01)

        # Exact prior calibration: m0 = log1p(exp(init_bias))
        m0 = math.log1p(math.exp(float(init_bias)))
        if self.subpixel_stride2:
            self.pixel_shuffle: nn.PixelShuffle | None = nn.PixelShuffle(upscale_factor=2)
            if self.head_activation == "relu":
                nn.init.constant_(self.final_conv.bias, m0 / 4.0)
            else:
                bias_s2 = _softplus_inverse(m0 / 4.0 + 0.5 * self.floor_tau)
                nn.init.constant_(self.final_conv.bias, bias_s2)
        else:
            self.pixel_shuffle = None
            if self.head_activation == "relu":
                nn.init.constant_(self.final_conv.bias, m0)
            else:
                eff_b = _softplus_inverse(m0 + 0.5 * self.floor_tau) if self.floor_tau > 0.0 else init_bias
                nn.init.constant_(self.final_conv.bias, eff_b)

        # 4. Temperature Softplus Parameter
        if self.temp_softplus:
            self.tau = nn.Parameter(torch.ones(1))

        # Backward compatibility interface: exposes body attribute
        self.body = nn.Sequential(*blocks, self.final_conv)

    def forward_logits(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pre-activation logit field z0 with 2D spatial FiLM modulation."""
        if isinstance(f, tuple):
            f = f[0]

        h = self.refinement_blocks(f)

        # 2D Spatial FiLM scale modulation
        if self.film_conv is not None and scale_weights is not None:
            if scale_weights.dim() == 2:
                sw_in = scale_weights.view(scale_weights.shape[0], scale_weights.shape[1], 1, 1)
            elif scale_weights.dim() == 4:
                sw_in = scale_weights
            else:
                sw_in = scale_weights

            k = min(sw_in.shape[1], self.num_scales)
            if k < self.num_scales:
                pad_k = torch.zeros(
                    sw_in.shape[0], self.num_scales - k, sw_in.shape[-2], sw_in.shape[-1],
                    device=sw_in.device, dtype=sw_in.dtype
                )
                sw_in = torch.cat([sw_in[:, :k], pad_k], dim=1)
            else:
                sw_in = sw_in[:, :self.num_scales]

            if sw_in.shape[-2:] != h.shape[-2:]:
                sw_in = F.interpolate(sw_in, size=h.shape[-2:], mode="bilinear", align_corners=False)

            film_params = self.film_conv(sw_in)
            gamma, beta = torch.chunk(film_params, 2, dim=1)
            gamma = 1.0 + torch.tanh(gamma)
            h = h * gamma + beta

        z = self.final_conv(h)
        if self.subpixel_stride2 and self.pixel_shuffle is not None:
            z = self.pixel_shuffle(z)
        return z

    def activate(
        self,
        z: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Activate raw logit field into continuous non-negative Radon measure field."""
        if self.head_activation == "relu":
            return F.relu(z)

        if self.temp_softplus and hasattr(self, "tau"):
            tau_clamped = self.tau.clamp_min(0.1)
            y_base = tau_clamped * F.softplus(z / tau_clamped)
        else:
            y_base = F.softplus(z)

        if self.floor_tau > 0.0:
            y_base = _smooth_floor(y_base, self.floor_tau)
        return y_base

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass computing calibrated density field y0."""
        z0 = self.forward_logits(f, scale_weights=scale_weights)
        return self.activate(z0, scale_weights=scale_weights)
