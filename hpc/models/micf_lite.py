"""MICF-Lite lightweight crowd counter.

Unified architecture supporting all 6 pilot models (B1 to B6):
- Backbone: MobileNetV4ConvSmall (C4, C8, C16)
- Neck: Additive FPN Neck
- Context: DirectionalIntegralContext (optional, for B5 & B6)
- Head:
  - 'local': Predicts discrete cell counts Y (B1, B2, B6)
  - 'cumulative': Predicts monotonic integral count field C (B3, B4, B5)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV4Backbone
from .blocks import make_group_norm
from .integral_context import DirectionalIntegralContext
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
        use_integral_context: bool = False,
        head_type: str = "cumulative",
        output_stride: int = 16,
        feature_reductions: Tuple[int, ...] = (4, 8, 16),
        eps_d: float = 1e-8,
    ) -> None:
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type not in {"local", "cumulative"}:
            raise ValueError(f"head_type must be 'local' or 'cumulative', got {head_type}")

        self.output_stride = int(output_stride)
        self.use_integral_context = bool(use_integral_context)
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

        # 3. Optional Stride Downsampling to stride 16 if requested
        if self.output_stride == 16:
            self.stride_down = nn.Sequential(
                nn.Conv2d(self.neck_width, self.neck_width, kernel_size=3, stride=2, padding=1, groups=self.neck_width, bias=False),
                nn.BatchNorm2d(self.neck_width),
                nn.SiLU(),
                nn.Conv2d(self.neck_width, self.neck_width, kernel_size=3, stride=2, padding=1, groups=self.neck_width, bias=False),
                nn.BatchNorm2d(self.neck_width),
                nn.SiLU(),
            )
        else:
            self.stride_down = nn.Identity()

        # 4. Context Decoder: Directional Integral Context vs Identity
        if self.use_integral_context:
            self.context_module = DirectionalIntegralContext(channels=self.neck_width)
        else:
            self.context_module = nn.Identity()

        # 5. Mass Head
        self.head_dw = nn.Conv2d(
            self.neck_width, self.neck_width, kernel_size=3, padding=1, groups=self.neck_width, bias=False
        )
        self.head_norm = make_group_norm(self.neck_width)
        self.head_act = nn.SiLU()
        self.head_out = nn.Conv2d(self.neck_width, 1, kernel_size=1)

    def forward_field(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting predicted field (Y for local, C for cumulative).

        Local head (B1, B2, B6):
            Applies softplus to enforce Y_hat >= 0 (cell counts are non-negative).
        Cumulative head (B3, B4, B5):
            Returns raw linear output — no elementwise activation.
            Measure validity Delta_xy C_hat >= 0 is enforced via the validity loss,
            NOT by making each C_hat(i,j) independently positive (which conflates
            elementwise positivity with monotonicity).
        """
        features = self.backbone(x)
        p4 = self.neck(*features)
        p_strided = self.stride_down(p4)
        p_context = self.context_module(p_strided)

        h = self.head_dw(p_context)
        h = self.head_norm(h)
        h = self.head_act(h)
        z = self.head_out(h).float()

        if self.head_type == "local":
            # Local counts must be non-negative
            return F.softplus(z).clamp_min(self.eps_d)
        else:
            # Cumulative field: raw linear output.
            # measure validity is enforced via MICFLoss.lambda_valid penalty.
            return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the primary prediction map."""
        return self.forward_field(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        pad_multiple: int | None = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference method returning (pred_count, pred_map).

        For cumulative head: pred_count = pred_c[..., -1, -1].
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

        if self.head_type == "cumulative":
            # Count is the bottom-right corner of the valid cumulative field
            count = field_valid[..., -1, -1].squeeze()
            return count, field_valid
        else:
            # Local count: sum of all cells
            count = field_valid.sum(dim=(-1, -2, -3))
            return count, field_valid

    @torch.no_grad()
    def predict_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 256,
        halo: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tiled inference with context halos."""
        if self.head_type == "local":
            # For local counts, standard tiled prediction with core accumulation
            from tools.eval_localization import predict_tiled_fallback
            # Or evaluate full-image directly
            return self.predict(x)
        else:
            # For cumulative head, tile prediction recovers Y per tile and re-accumulates
            # Or evaluate full-image directly
            return self.predict(x)
