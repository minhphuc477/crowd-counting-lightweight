"""MICF-Lite lightweight crowd counter.

Unified architecture supporting all pilot models (B1 to B6, plus B7 axial context):
- Backbone: MobileNetV4ConvSmall (C4, C8, C16)
- Neck: Additive FPN Neck with multi-scale routes (P4, P8, P16)
- Stride routing: directly reads P16, P8, or P4 natively from neck routes (no wasted downsampling)
- Context:
  - None / Identity (B1, B2, B3, B4)
  - 4-Directional Normalized Integral Context (B5, B6)
  - Axial Integral Context (B7)
- Head:
  - 'local': Predicts discrete cell counts Y >= 0 via softplus (B1, B2, B6)
  - 'cumulative': Predicts raw linear monotonic integral count field C (B3, B4, B5)
  - 'integrated_local': Valid-by-construction baseline M = softplus(z) -> C = Integral(M) (Section 32)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV4Backbone
from .blocks import make_group_norm
from .integral_context import AxialIntegralContext, DirectionalIntegralContext
from .neck import AdditiveFPNNeck
from hpc.losses.micf import discrete_mixed_difference


class MICFLite(nn.Module):
    """MICF-Lite model supporting local and cumulative count prediction."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = False,
        neck_width: int = 32,
        context_dilations: Tuple[int, ...] = (1, 2, 3),
        use_integral_context: Union[bool, str] = False,
        head_type: str = "cumulative",
        output_stride: int = 16,
        feature_reductions: Tuple[int, ...] = (4, 8, 16),
        eps_d: float = 1e-8,
    ) -> None:
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type not in {"local", "cumulative", "integrated_local"}:
            raise ValueError(
                f"head_type must be 'local', 'cumulative', or 'integrated_local', got {head_type}"
            )

        self.output_stride = int(output_stride)
        if self.output_stride not in {4, 8, 16}:
            raise ValueError(f"output_stride must be 4, 8, or 16, got {self.output_stride}")

        self.neck_width = int(neck_width)
        self.eps_d = float(eps_d)
        self.feature_reductions = tuple(int(r) for r in feature_reductions)

        # 1. Backbone
        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            target_reductions=self.feature_reductions,
        )

        # 2. Additive FPN Neck
        self.neck = AdditiveFPNNeck(
            in_channels=self.backbone.out_channels,
            width=self.neck_width,
            context_dilations=context_dilations,
        )

        # 3. Context Decoder
        # Support None / False, 4-dir (True, "4dir"), and axial ("axial")
        if isinstance(use_integral_context, str):
            ctx_mode = use_integral_context.lower()
        else:
            ctx_mode = "4dir" if use_integral_context else "none"

        self.use_integral_context = ctx_mode != "none"
        self.context_mode = ctx_mode

        if ctx_mode in {"4dir", "directional"}:
            self.context_module = DirectionalIntegralContext(channels=self.neck_width)
        elif ctx_mode == "axial":
            self.context_module = AxialIntegralContext(channels=self.neck_width)
        else:
            self.context_module = nn.Identity()

        # 4. Mass Head (depthwise-separable 1-channel projection)
        self.head_dw = nn.Conv2d(
            self.neck_width, self.neck_width, kernel_size=3, padding=1, groups=self.neck_width, bias=False
        )
        self.head_norm = make_group_norm(self.neck_width)
        self.head_act = nn.SiLU()
        self.head_out = nn.Conv2d(self.neck_width, 1, kernel_size=1)

    def forward_field(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting predicted field (Y for local, C for cumulative).

        Stride selection: reads native feature P4, P8, or P16 directly from the neck
        routes, avoiding wasted upsampling + redundant downsampling convolutions.

        Head behaviors:
        - Local head (B1, B2, B6):
            Applies softplus to enforce Y_hat >= 0.
        - Cumulative head (B3, B4, B5):
            Returns raw linear output. Validity Delta_xy C_hat >= 0 is enforced
            via the validity loss penalty, NOT by elementwise softplus.
        - Integrated Local head (Section 32):
            Predicts local M = softplus(z) >= 0, then computes C = Integral(M).
            Guarantees Delta_xy C >= 0 by construction.
        """
        features = self.backbone(x)
        # Extract native multi-scale routes: p4, p8, p16
        _, routes = self.neck(*features, return_routes=True)
        route_key = f"p{self.output_stride}"
        p_feat = routes[route_key]

        p_context = self.context_module(p_feat)

        h = self.head_dw(p_context)
        h = self.head_norm(h)
        h = self.head_act(h)
        z = self.head_out(h).float()

        if self.head_type == "local":
            return F.softplus(z).clamp_min(self.eps_d)
        elif self.head_type == "integrated_local":
            m = F.softplus(z).clamp_min(self.eps_d)
            return torch.cumsum(torch.cumsum(m, dim=-2), dim=-1)
        else:
            return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_field(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        pad_multiple: int | None = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference method returning (pred_count, pred_map).

        For cumulative & integrated_local heads: pred_count = pred_c[..., -1, -1].
        For local head: pred_count = pred_y.sum().
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")

        _, _, h, w = x.shape
        if pad_multiple is not None:
            pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
            pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        field = self.forward_field(x)
        out_h = math.ceil(h / self.output_stride)
        out_w = math.ceil(w / self.output_stride)
        field_valid = field[..., :out_h, :out_w]

        if self.head_type in {"cumulative", "integrated_local"}:
            count = field_valid[..., -1, -1].squeeze()
            return count, field_valid
        else:
            count = field_valid.sum(dim=(-1, -2, -3))
            return count, field_valid

    @torch.no_grad()
    def predict_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hierarchical Tile Composition inference for full images (Section 30 of design doc).

        Solves the full-image inference problem (Section 29) for models with finite receptive field:
        1. Divides image into non-overlapping tiles of size tile_size.
        2. On each tile, runs the model.
        3. For cumulative models:
           - Reads tile total N_t = C_t[-1, -1]
           - Recovers local tile mass Y_t = Delta_xy C_t
           - Assembles global discrete mass map Y_global
           - Global count is sum_t N_t
           - Global cumulative field is cumsum2d(Y_global)
        4. For local models:
           - Assembles local mass map Y_global
           - Global count is sum(Y_global)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")

        B, _, H, W = x.shape
        stride = self.output_stride
        out_h = math.ceil(H / stride)
        out_w = math.ceil(W / stride)

        y_assembled = torch.zeros((B, 1, out_h, out_w), device=x.device, dtype=torch.float32)
        total_counts = torch.zeros(B, device=x.device, dtype=torch.float32)

        for top in range(0, H, tile_size):
            for left in range(0, W, tile_size):
                bot = min(top + tile_size, H)
                right = min(left + tile_size, W)

                tile = x[:, :, top:bot, left:right]
                th, tw = bot - top, right - left

                # Pad tile to multiple of 64 if needed
                pad_h = (64 - (th % 64)) % 64
                pad_w = (64 - (tw % 64)) % 64
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

                tile_field = self.forward_field(tile)
                tile_out_h = math.ceil(th / stride)
                tile_out_w = math.ceil(tw / stride)
                tile_valid = tile_field[..., :tile_out_h, :tile_out_w]

                grid_top = top // stride
                grid_left = left // stride
                grid_bot = grid_top + tile_out_h
                grid_right = grid_left + tile_out_w

                if self.head_type in {"cumulative", "integrated_local"}:
                    y_tile = discrete_mixed_difference(tile_valid)
                    n_tile = tile_valid[..., -1, -1].squeeze()
                    y_assembled[:, :, grid_top:grid_bot, grid_left:grid_right] = y_tile
                    total_counts += n_tile
                else:
                    y_assembled[:, :, grid_top:grid_bot, grid_left:grid_right] = tile_valid
                    total_counts += tile_valid.sum(dim=(-1, -2, -3))

        if self.head_type in {"cumulative", "integrated_local"}:
            c_assembled = torch.cumsum(torch.cumsum(y_assembled, dim=-2), dim=-1)
            return total_counts.squeeze(), c_assembled
        else:
            return total_counts.squeeze(), y_assembled
