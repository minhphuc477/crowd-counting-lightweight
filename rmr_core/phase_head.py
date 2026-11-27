from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .density_activation import _compact_c1_inverse, _density_activate, _softplus_inverse
from .necks import ConvGNAct


class S2DPhaseDSIMRHead(nn.Module):
    """Space-to-Depth C2 Phase-Guided Decoupled Support-Intensity Sub-Pixel Head.

    Folds Stride-2 stem features C2 (H/2 x W/2) via parameter-free PixelUnshuffle(2)
    onto the Stride-4 grid, injecting sub-pixel peak contrast into P4 and applying a
    shift-equivariant shared nonlinear filter across the 4 quadrants to guide PixelShuffle
    sub-pixel mass allocation without Stride-2 GroupNorm or BatchNorm overhead.
    """

    def __init__(
        self,
        width: int = 32,
        c2_channels: int = 32,
        phase_hidden: int = 8,
        init_bias: float = -4.1422,
        floor_tau: float = 0.001,
        density_adaptive_scale: bool = True,
        density_scale_gamma: float = 0.15,
        density_scale_learnable: bool = False,
        head_activation: str = "softplus",
        temp_softplus: bool = False,
        subpixel_kernel_size: int = 3,
        dsimr_init_z_s: float = 0.0,
        head_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.c2_channels = int(c2_channels)
        self.head_activation = str(head_activation).lower()
        self.temp_softplus = bool(temp_softplus)
        self.subpixel_stride2 = True
        self.dsimr_mode = True
        self.floor_tau = float(floor_tau) / 4.0
        self._last_support_logit: torch.Tensor | None = None

        if self.c2_channels > 0:
            self.c2_contrast_proj: nn.Module | None = ConvGNAct(self.c2_channels, self.width, 1)
            self.gamma_c2: nn.Parameter | None = nn.Parameter(torch.full((self.width,), 0.1))
            # Final conv uses bias=False so softmax(z_sub, dim=1) has zero dead bias parameters.
            self.phase_mlp: nn.Sequential | None = nn.Sequential(
                nn.Conv2d(self.c2_channels, int(phase_hidden), kernel_size=1, bias=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(int(phase_hidden), 1, kernel_size=1, bias=False),
            )
            nn.init.normal_(self.phase_mlp[-1].weight, std=0.01)  # type: ignore[arg-type]
            self.gamma_phase: nn.Parameter | None = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.c2_contrast_proj = None
            self.gamma_c2 = None
            self.phase_mlp = None
            self.gamma_phase = None

        blocks = []
        for _ in range(max(1, int(head_blocks))):
            blocks.append(ConvGNAct(self.width, self.width, 3, groups=self.width))
            blocks.append(ConvGNAct(self.width, self.width, 1))
        self.body = nn.Sequential(*blocks)

        pre_k = max(1, int(subpixel_kernel_size))
        self.subpixel_proj = nn.Conv2d(self.width, 4, kernel_size=pre_k, padding=pre_k // 2)
        nn.init.normal_(self.subpixel_proj.weight, std=0.01 / float(pre_k))
        nn.init.zeros_(self.subpixel_proj.bias)

        self.dsimr_proj = nn.Conv2d(self.width, 2, kernel_size=1, bias=True)
        nn.init.normal_(self.dsimr_proj.weight, std=0.01)
        m0 = math.log1p(math.exp(float(init_bias)))
        z_s_init = float(dsimr_init_z_s)
        s0 = 1.0 / (1.0 + math.exp(-z_s_init))
        target_sub = (m0 / 4.0) / s0
        self.adaptive_scale_ref: float = 0.0
        if self.head_activation == "relu":
            b_lambda = target_sub
        elif self.head_activation == "compact_c1":
            b_lambda = _compact_c1_inverse(target_sub, self.floor_tau)
        else:
            b_lambda = _softplus_inverse(target_sub + 0.5 * self.floor_tau)
            self.adaptive_scale_ref = float(b_lambda) + math.log(4.0)
        with torch.no_grad():
            self.dsimr_proj.bias.copy_(torch.tensor([z_s_init, float(b_lambda)], dtype=torch.float32))

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        if self.temp_softplus:
            self.tau: nn.Parameter | None = nn.Parameter(torch.ones(1))
        else:
            self.tau = None
        self.density_adaptive_scale = bool(density_adaptive_scale)
        if self.density_adaptive_scale and bool(density_scale_learnable):
            self.density_scale_gamma: nn.Parameter | float = nn.Parameter(torch.tensor([float(density_scale_gamma)]))
        else:
            self.density_scale_gamma = float(density_scale_gamma)

    def _activate_intensity(self, z_lambda: torch.Tensor) -> torch.Tensor:
        return _density_activate(
            z_lambda,
            temp_softplus=self.temp_softplus,
            tau=self.tau,
            density_curvature=False,
            curvature_alpha=None,
            gated_density_curvature=False,
            curvature_dense_threshold=0.15,
            curvature_gate_beta=0.03,
            curvature_pool_kernel=8,
            curv_scale=4.0,
            floor_tau=self.floor_tau,
            density_adaptive_scale=self.density_adaptive_scale,
            density_scale_gamma=self.density_scale_gamma,
            adaptive_scale_ref=self.adaptive_scale_ref,
            curvature_pade=False,
            head_activation=self.head_activation,
        )

    def forward_with_logits(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
        need_full_z0: bool = False,
        force_activate: bool = True,
        c2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (z0, y0) using Space-to-Depth C2 phase guidance and DSIMR modulation."""
        if isinstance(f, tuple):
            f = f[0]
        b, _, h4, w4 = f.shape
        z_sub_c2: torch.Tensor | None = None

        if c2 is not None and self.c2_contrast_proj is not None and self.phase_mlp is not None:
            if c2.shape[-2] != 2 * h4 or c2.shape[-1] != 2 * w4:
                c2 = F.interpolate(c2, size=(2 * h4, 2 * w4), mode="bilinear", align_corners=False)
            c2_q = F.pixel_unshuffle(c2, 2).view(b, self.c2_channels, 4, h4, w4)
            c2_contrast = c2_q.amax(dim=2) - c2_q.mean(dim=2)
            assert self.gamma_c2 is not None
            f = f + self.gamma_c2.view(1, -1, 1, 1) * self.c2_contrast_proj(c2_contrast)
            z_sub_c2 = self.phase_mlp(c2_q.reshape(b, self.c2_channels, 4 * h4, w4)).view(b, 4, h4, w4)

        h4_feat = self.body(f)
        z_sub = self.subpixel_proj(h4_feat)
        if z_sub_c2 is not None and self.gamma_phase is not None:
            z_sub = z_sub + self.gamma_phase.view(1, 1, 1, 1) * z_sub_c2

        alpha = 4.0 * F.softmax(z_sub, dim=1)
        dsimr_out = self.dsimr_proj(h4_feat)
        z_s = dsimr_out[:, 0:1]
        z_lambda = dsimr_out[:, 1:2]
        self._last_support_logit = z_s

        y_cell = self._activate_intensity(z_lambda)
        y0 = self.pixel_shuffle((y_cell * torch.sigmoid(z_s)) * alpha)
        z0 = self.pixel_shuffle(z_lambda.expand(-1, 4, -1, -1)) if need_full_z0 else z_lambda
        return z0, y0

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
        c2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_logits(f, scale_weights=scale_weights, need_full_z0=False, c2=c2)[1]
