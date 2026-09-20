from __future__ import annotations

import torch
import torch.nn as nn


class MicroPerspectiveElevation(nn.Module):
    """1D Vertical Perspective Carrier Elevation Modulation (MPE-v2).

    Maps normalized vertical coordinates v in [-1, 1] through a 1D linear projection
    (nn.Linear(1, channels), exactly 64 parameters for channels=32) with mass-conserved
    spatial normalization:
        M(v) = 1.0 + tanh(W v + b)
        M_bar = (1 / H) * sum_{i=1}^H M(v_i)
        P_tilde = P * (M(v) / M_bar)
    Preserves vertical mass invariant: (1/H) sum (P_tilde / P) == 1.000000 +- 1e-6 across
    any height H >= 2, batch size, and channel width. Completely prevents false foreground
    density over-inflation while preserving camera depth calibration.
    Zero-initialized identity warm-start (W=0, b=0 -> M(v)=1.0, M_bar=1.0, P_tilde=P).
    """

    def __init__(self, channels: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(1, channels, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-2:] if x.ndim < 4 else x.shape[-2]
        if isinstance(h, tuple):
            h = h[0]
        v = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=self.proj.weight.dtype).view(h, 1)
        elevation_mod = self.proj(v).transpose(0, 1).unsqueeze(-1)
        if x.ndim == 4:
            elevation_mod = elevation_mod.unsqueeze(0)

        m = 1.0 + torch.tanh(elevation_mod).to(dtype=x.dtype)
        m_bar = m.mean(dim=-2, keepdim=True)
        return x * (m / m_bar)


class MicroCoordAttn(nn.Module):
    """Micro Perspective Coordinate Attention for P4 carrier features (C=32).

    Factorizes spatial context into 1D horizontal and 1D vertical pooling
    to natively capture camera perspective elevation gradients with exactly 456 parameters.
    """

    def __init__(self, channels: int = 32, reduction: int = 8) -> None:
        super().__init__()
        mid_channels = max(4, channels // reduction)
        self.conv_shared = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

        nn.init.normal_(self.conv_h.weight, std=1e-3)
        nn.init.constant_(self.conv_h.bias, 3.5)
        nn.init.normal_(self.conv_w.weight, std=1e-3)
        nn.init.constant_(self.conv_w.bias, 3.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(dim=-1, keepdim=True)
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.gn(self.conv_shared(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return x * a_h * a_w


class ContinuousPerspectiveCarrierModulation(nn.Module):
    """2D Continuous Perspective Carrier Modulation (CPCM - RMR-v32).

    Maps continuous normalized 2D spatial coordinates [u, v] in [0, 1]^2 to channel-wise
    modulation weights for P4 carrier features (channels=32).
    Architecture:
        Conv2d(2, hidden, kernel_size=1)   (2 * 8 + 8 = 24 params for hidden=8)
        SiLU()
        Conv2d(hidden, channels, kernel_size=1) (8 * 32 + 32 = 288 params)
        Total: exactly 312 parameters.

    Zero-initialization:
        The output conv is initialized with weights=0, bias=0.
        Modulation multiplier:
            M(u, v) = 1.0 + tanh(CPCM(u, v))
        At epoch 0, M(u, v) == 1.0 identically everywhere.
        Guarantees 100% bitwise parity with unmodulated baseline at initialization.
    """

    def __init__(self, channels: int = 32, hidden: int = 8) -> None:
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        param_dtype = self.mlp[0].weight.dtype
        v = torch.linspace(0.0, 1.0, steps=h, device=x.device, dtype=param_dtype)
        u = torch.linspace(0.0, 1.0, steps=w, device=x.device, dtype=param_dtype)
        grid_v, grid_u = torch.meshgrid(v, u, indexing="ij")
        coords = torch.stack([grid_u, grid_v], dim=0).unsqueeze(0)  # [1, 2, H, W]
        if b > 1:
            coords = coords.expand(b, -1, -1, -1)

        delta = self.mlp(coords)
        mod = (1.0 + torch.tanh(delta)).to(dtype=x.dtype)
        mod_bar = mod.mean(dim=(-2, -1), keepdim=True)
        return x * (mod / mod_bar)

