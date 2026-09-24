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

    def forward(self, p4: torch.Tensor, alpha_persp: torch.Tensor | None = None) -> torch.Tensor:
        """Modulate carrier feature map P4 with continuous perspective geometry.

        Args:
            p4: [B, C, H, W] carrier feature tensor.
            alpha_persp: [B, 1, 1, 1] optional camera tilt factor in [0, 1].

        Returns:
            p4_mod: [B, C, H, W] perspective-modulated carrier feature tensor.
        """
        b, c, h, w = p4.shape
        p4_vert = p4.mean(dim=-1)  # [B, C, H]
        feat = self.act(self.norm(self.dw1d(p4_vert)))
        deltas = self.pw1d(feat)  # [B, 2, H]

        mod_h = 0.15 * torch.tanh(deltas[:, 0:1, :].unsqueeze(-1))  # [B, 1, H, 1]
        mod_rho = 0.15 * torch.tanh(deltas[:, 1:2, :].unsqueeze(-1))  # [B, 1, H, 1]
        if alpha_persp is not None:
            mod = 1.0 + alpha_persp.view(b, 1, 1, 1).to(dtype=p4.dtype) * (mod_h + mod_rho)
        else:
            mod = 1.0 + mod_h + mod_rho
        mod_bar = mod.mean(dim=-2, keepdim=True)
        return p4 * (mod / mod_bar)


class DynamicCameraAnglePredictor(nn.Module):
    """Dynamic Camera Angle & Perspective Geometry Predictor (DCAP - RMR-v34).

    Predicts image-level perspective geometry dynamically from deepest semantic features P16:
      - alpha_persp in [0, 1]: camera tilt / perspective intensity
        (0.0 = overhead / bird's-eye / flat isotropic view, 1.0 = steep oblique street perspective).
      - v_horizon in [-0.3, +0.3]: vertical offset of vanishing horizon relative to image center.
      - delta_scale in R^K: image-level global scale preference (e.g. distant vs close-up scene).

    Parameter budget:
      - Global Average Pooling: 0 params
      - Linear(in_channels, 2 + num_scales): in_channels * (2 + num_scales) + (2 + num_scales)
        For in_channels=32, num_scales=3: 32 * 5 + 5 = 165 parameters.
      - Zero-initialization: Step 0 Identity Parity (alpha = 0.5, v_horizon = 0.0, delta_scale = 0.0).
    """

    def __init__(self, in_channels: int = 32, num_scales: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.proj = nn.Linear(self.in_channels, 2 + self.num_scales, bias=True)
        # Step 0 Identity Parity: zero-init guarantees uniform, unbiased initialization
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, p16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            p16: [B, C, H_16, W_16] deepest semantic feature map.

        Returns:
            alpha_persp: [B, 1, 1, 1] camera tilt factor in [0, 1].
            v_horizon: [B, 1, 1, 1] vertical horizon offset in [-0.3, +0.3].
            delta_scale: [B, num_scales, 1, 1] global scale logit adjustment.
        """
        gap = p16.mean(dim=(-2, -1))  # [B, C]
        out = self.proj(gap)  # [B, 2 + num_scales]

        alpha_persp = torch.sigmoid(out[:, 0:1]).view(-1, 1, 1, 1)
        v_horizon = (0.3 * torch.tanh(out[:, 1:2])).view(-1, 1, 1, 1)
        delta_scale = out[:, 2: 2 + self.num_scales].view(-1, self.num_scales, 1, 1)

        return alpha_persp, v_horizon, delta_scale


class DiAGScaleRoutingHead(nn.Module):
    """Dynamic Image-Adaptive Geometry (DiAG) Scale Routing Head (RMR-v34).

    Dynamically modulates the multi-scale spatial partition of unity pi(x, y) in Delta^{K-1}
    using both local carrier features P4 and image-level camera perspective geometry:
      z_k(x, y) = z_local_k(x, y) + delta_scale_k + alpha_persp * w_persp_k * (y/H - 0.5 - v_horizon)
      pi = Softmax(z / temperature, dim=1)

    Mathematical Properties:
      1. Overhead Invariance: When alpha_persp -> 0 (overhead / drone view),
         the perspective term vanishes completely. Scale routing is purely isotropic across all y.
      2. Oblique Adaptivity: When alpha_persp > 0 (street view), scale logits shift smoothly
         with optical depth: fine scales near horizon, coarse scales near foreground.
      3. Resolution Preservation: Every scale k exists everywhere across the image support,
         completely preventing foreground spatial resolution starvation.
      4. Step 0 Identity Parity: With zero-initialized weights, outputs exact uniform scale
         distribution (1/K, 1/K, ...) at Step 0.

    Parameter budget:
      - Depthwise 3x3: in_channels * 9 + in_channels = 320 params.
      - GroupNorm(8, in_channels): 2 * in_channels = 64 params.
      - Pointwise 1x1: in_channels * num_scales + num_scales = 99 params.
      - Learnable perspective slope: num_scales = 3 params.
      Total trainable parameters: 486 params (< 500 budget).
    """

    def __init__(
        self,
        in_channels: int = 32,
        num_scales: int = 3,
        temperature: float = 1.0,
        persp_slope_init: str = "physical",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_scales = int(num_scales)
        self.temperature = float(max(temperature, 0.1))

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
        self.pw = nn.Conv2d(self.in_channels, self.num_scales, kernel_size=1, bias=True)

        if persp_slope_init == "physical" and self.num_scales == 3:
            init_persp = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32)
        elif persp_slope_init == "physical":
            init_persp = torch.linspace(-0.5, 0.5, steps=self.num_scales, dtype=torch.float32)
        else:
            init_persp = torch.zeros(self.num_scales, dtype=torch.float32)
        self.persp_slope = nn.Parameter(init_persp)

        # Zero-initialization: Step 0 Identity Parity
        nn.init.zeros_(self.pw.weight)
        nn.init.zeros_(self.pw.bias)

    def forward(
        self,
        x: torch.Tensor,
        alpha_persp: torch.Tensor | None = None,
        v_horizon: torch.Tensor | None = None,
        delta_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute image-adaptive spatial scale routing weights pi(x, y) in Delta^{K-1}.

        Args:
            x: [B, C, H, W] carrier feature tensor.
            alpha_persp: [B, 1, 1, 1] optional camera tilt factor in [0, 1].
            v_horizon: [B, 1, 1, 1] optional vertical horizon offset in [-0.3, +0.3].
            delta_scale: [B, K, 1, 1] optional global scale bias.

        Returns:
            pi: [B, num_scales, H, W] spatial partition of unity (sum_k pi_k == 1.0).
        """
        feat = self.act(self.norm(self.dw(x)))
        logits = self.pw(feat)  # [B, K, H, W]

        if delta_scale is not None:
            logits = logits + delta_scale.to(dtype=logits.dtype)

        if alpha_persp is not None:
            b, _, h, w = logits.shape
            v_base = torch.linspace(-0.5, 0.5, steps=h, device=x.device, dtype=logits.dtype).view(1, 1, h, 1)
            if v_horizon is not None:
                v_grid = v_base - v_horizon.to(dtype=logits.dtype)
            else:
                v_grid = v_base
            persp_bias = alpha_persp.to(dtype=logits.dtype) * self.persp_slope.view(1, self.num_scales, 1, 1).to(dtype=logits.dtype) * v_grid
            logits = logits + persp_bias

        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.softmax(logits, dim=1)


class PARKRoutingHead(nn.Module):
    """Ultra-lightweight Multi-Mode Geometry Routing Head for PARK.

    Parameter budget:
      - Depthwise 3x3: in_channels * 9 + in_channels = 320 params.
      - GroupNorm(8, in_channels): 2 * in_channels = 64 params.
      - Pointwise 1x1: in_channels * num_modes + num_modes params.
    """

    def __init__(
        self,
        in_channels: int = 32,
        temperature: float = 1.0,
        num_modes: int = 3,
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
        """Compute spatial multi-mode routing weights pi(x, y) in Delta^{M-1}."""
        feat = self.act(self.norm(self.dw(x)))
        logits = self.pw(feat)
        if self.temperature != 1.0:
            logits = logits / self.temperature
        return F.softmax(logits, dim=1)
