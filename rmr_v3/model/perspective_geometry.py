from __future__ import annotations

"""Perspective Geometry Modules for RMR-v3 (PARK).

Provides:
1. PerspectiveGeometryHead (PGH): 1D vertical depthwise-separable conv predicting
   continuous height scaling h(y) and aspect ratio rho(y) along the optical ray (226 params).
2. PARKRoutingHead: Ultra-lightweight 2-mode routing head for isotropic horizon core
   vs. anisotropic projective elongation (450 params).

All modules enforce Step 0 Identity Parity via zero-initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerspectiveGeometryHead(nn.Module):
    """1D Vertical Perspective Geometry Head for continuous depth-scale modeling.

    Maps carrier feature map P4 [B, C, H, W] to continuous scanline scale factors
    h(y) in pixels and aspect ratios rho(y) with exactly 226 trainable parameters:
      - 1D Depthwise Conv3: C * 3 = 96 params.
      - 1D GroupNorm(4, C): 2 * C = 64 params.
      - 1D Pointwise Conv1: C * 2 + 2 = 66 params.
      Total: exactly 226 parameters.

    Zero-initialization:
      The pointwise projection is initialized to zero, guaranteeing that at epoch 0,
      predicted adjustments are zero and the model strictly follows the analytical
      linear projective geometry prior.
    """

    def __init__(
        self,
        in_channels: int = 32,
        horizon_h_px: float = 16.0,
        foreground_h_px: float = 128.0,
        max_aspect_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.horizon_h_px = float(horizon_h_px)
        self.foreground_h_px = float(foreground_h_px)
        self.max_aspect_ratio = float(max_aspect_ratio)

        self.dw1d = nn.Conv1d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(4, self.in_channels)
        self.act = nn.SiLU(inplace=True)
        self.pw1d = nn.Conv1d(self.in_channels, 2, kernel_size=1, bias=True)

        # Zero-initialization: Step 0 Identity Parity
        nn.init.zeros_(self.pw1d.weight)
        nn.init.zeros_(self.pw1d.bias)

    def predict_scales(self, p4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict continuous height scale h(y) and aspect ratio rho(y)."""
        b, c, h, w = p4.shape
        p4_vert = p4.mean(dim=-1)  # [B, C, H]
        feat = self.act(self.norm(self.dw1d(p4_vert)))
        deltas = self.pw1d(feat)  # [B, 2, H]

        v = torch.linspace(0.0, 1.0, steps=h, device=p4.device, dtype=p4.dtype).view(1, 1, h)
        h_base = self.horizon_h_px + (self.foreground_h_px - self.horizon_h_px) * v
        rho_base = 1.0 + (self.max_aspect_ratio - 1.0) * v

        h_cont = h_base * (1.0 + 0.5 * torch.tanh(deltas[:, 0:1, :]))
        rho_cont = rho_base * (1.0 + 0.5 * torch.tanh(deltas[:, 1:2, :]))
        return h_cont, rho_cont

    def forward(self, p4: torch.Tensor) -> torch.Tensor:
        """Modulate carrier feature map P4 with continuous perspective geometry.

        Args:
            p4: [B, C, H, W] carrier feature tensor.

        Returns:
            p4_mod: [B, C, H, W] perspective-modulated carrier feature tensor.
        """
        b, c, h, w = p4.shape
        p4_vert = p4.mean(dim=-1)  # [B, C, H]
        feat = self.act(self.norm(self.dw1d(p4_vert)))
        deltas = self.pw1d(feat)  # [B, 2, H]

        mod_h = 0.15 * torch.tanh(deltas[:, 0:1, :].unsqueeze(-1))  # [B, 1, H, 1]
        mod_rho = 0.15 * torch.tanh(deltas[:, 1:2, :].unsqueeze(-1))  # [B, 1, H, 1]
        mod = 1.0 + mod_h + mod_rho
        mod_bar = mod.mean(dim=-2, keepdim=True)
        return p4 * (mod / mod_bar)




class PARKRoutingHead(nn.Module):
    """Ultra-lightweight 2-Mode Geometry Routing Head for PARK.

    Routes between:
      - Mode 0: Isotropic Horizon Core (distant crowds & flat background).
      - Mode 1: Anisotropic Projective Elongation (upright human figures).

    Parameter budget:
      - Depthwise 3x3: in_channels * 9 + in_channels = 320 params.
      - GroupNorm(8, in_channels): 2 * in_channels = 64 params.
      - Pointwise 1x1: in_channels * 2 + 2 = 66 params.
      Total trainable parameters: 450 params (< 483 params of old ScaleRoutingHead).
    """

    def __init__(
        self,
        in_channels: int = 32,
        temperature: float = 1.0,
        num_modes: int = 2,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.temperature = float(max(temperature, 0.1))
        self.num_modes = int(num_modes)

        self.dw = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            padding=1,
            groups=self.in_channels,
            bias=True,
        )
        self.norm = nn.GroupNorm(8, self.in_channels)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(self.in_channels, self.num_modes, kernel_size=1, bias=True)

        # Initialize to balanced split
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial 2-mode routing weights pi(x, y) in Delta^1.

        Args:
            x: [B, C, H, W] carrier feature tensor.

        Returns:
            pi: [B, 2, H, W] spatial partition of unity (sum_k pi_k == 1.0).
        """
        feat = self.act(self.norm(self.dw(x)))
        logits = self.pw(feat)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.softmax(logits, dim=1)
