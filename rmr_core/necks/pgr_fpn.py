from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, _gn


class PGRFPNNeck(nn.Module):
    """Perspective-Guided Resolution-Preserving FPN Neck (PGR-Neck).

    Designed for Stride-8 collapsed backbones (os8_hdc): extracts a 100% feature-driven
    horizontal-row perspective descriptor g(y) in O(H) time via horizontal (mean + amax)
    pooling and a vertical 5x1 depthwise-separable encoder. Uses g(y) both to FiLM-modulate
    the deepest pyramid features and to supply a vertical perspective prior to LER-Head.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
    ) -> None:
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = int(width)
        self._last_row_context: torch.Tensor | None = None

        self.lat4 = ConvGNAct(c4, self.width, 1)
        self.lat8 = ConvGNAct(c8, self.width, 1)
        self.lat16 = ConvGNAct(c16, self.width, 1)

        self.row_enc = nn.Sequential(
            ConvGNAct(2 * self.width, self.width, 1),
            nn.Conv2d(self.width, self.width, kernel_size=(5, 1), padding=(2, 0), groups=self.width, bias=False),
            _gn(self.width),
            nn.SiLU(inplace=True),
            ConvGNAct(self.width, self.width, 1),
        )
        self.row_film = nn.Conv2d(self.width, 2 * self.width, kernel_size=1, bias=True)
        nn.init.normal_(self.row_film.weight, std=0.01)
        nn.init.zeros_(self.row_film.bias)

        self.ref16 = DSResidual(self.width)
        self.ref8 = DSResidual(self.width)
        self.ref4 = DSResidual(self.width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        row_stats = torch.cat([l16.mean(dim=-1, keepdim=True), l16.amax(dim=-1, keepdim=True)], dim=1)
        g_y = self.row_enc(row_stats)
        self._last_row_context = g_y

        ds, db = self.row_film(g_y).chunk(2, dim=1)
        p16 = self.ref16(l16 * (1.0 + 0.1 * torch.tanh(ds)) + 0.1 * db)

        up16_to_8 = F.interpolate(p16, size=l8.shape[-2:], mode="bilinear", align_corners=False)
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(p8, size=l4.shape[-2:], mode="bilinear", align_corners=False)
        p4 = self.ref4(l4 + up8_to_4)
        return p4, p8, p16
