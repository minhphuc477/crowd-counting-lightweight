from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .necks import ConvGNAct
from .density_activation import (
    _apply_curvature,
    _compact_c1_inverse,
    _density_activate,
    _softplus_inverse,
)

_M0_INIT: float = 0.015763
_FINE_HEAD_BIAS_INIT: float = math.log(math.exp(_M0_INIT) - 1.0)
_CURVATURE_ALPHA_INIT: float = -8.0


class ScaleConditionedFineHead(nn.Module):
    """Scale-Conditioned Dynamic Fine Density Head (RMR-v22).
    Combines depthwise context, FiLM scale simplex modulation, and calibrated bias.
    """

    def __init__(
        self,
        width: int = 32,
        num_scales: int = 3,
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        temp_softplus: bool = True,
        density_curvature: bool = True,
        gated_density_curvature: bool = True,
        curvature_dense_threshold: float = 0.15,
        curvature_gate_beta: float = 0.03,
        curvature_pool_kernel: int = 8,
        floor_tau: float = 0.0,
        curvature_alpha_init: float = _CURVATURE_ALPHA_INIT,
        density_adaptive_scale: bool = False,
        density_scale_gamma: float = 0.0,
        density_scale_learnable: bool = False,
        curvature_pade: bool = False,
        head_activation: str = "softplus",
    ):
        super().__init__()
        self.head_activation = str(head_activation)
        self.floor_tau = float(floor_tau)
        self.dw = ConvGNAct(width, width, 3, groups=width)
        self.pw = ConvGNAct(width, width, 1)

        self.num_scales = int(num_scales)
        self.scale_film = nn.Conv2d(self.num_scales, width, kernel_size=1, bias=False)
        nn.init.zeros_(self.scale_film.weight)

        self.out_conv = nn.Conv2d(width + self.num_scales, 1, kernel_size=1)
        nn.init.normal_(self.out_conv.weight, std=0.01)
        m0 = math.log1p(math.exp(float(init_bias)))
        if self.head_activation == "relu":
            eff_bias = float(m0)
        elif self.head_activation == "compact_c1":
            eff_bias = _compact_c1_inverse(m0, self.floor_tau)
        else:
            eff_bias = _softplus_inverse(m0 + 0.5 * self.floor_tau) if self.floor_tau > 0.0 else init_bias
        nn.init.constant_(self.out_conv.bias, eff_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus and self.head_activation not in ("relu", "compact_c1"):
            self.tau = nn.Parameter(torch.ones(1))

        self.density_curvature = bool(density_curvature)
        self.gated_density_curvature = bool(gated_density_curvature)
        self.curvature_pade = bool(curvature_pade)
        self.curv_scale = 1.0
        self.curvature_dense_threshold = float(curvature_dense_threshold)
        self.curvature_gate_beta = float(curvature_gate_beta)
        self.curvature_pool_kernel = int(curvature_pool_kernel)
        if self.density_curvature:
            self.curvature_alpha = nn.Parameter(torch.tensor(float(curvature_alpha_init)))

        self.density_adaptive_scale = bool(density_adaptive_scale)
        if self.density_adaptive_scale and bool(density_scale_learnable):
            self.density_scale_gamma = nn.Parameter(torch.tensor([float(density_scale_gamma)]))
        else:
            self.density_scale_gamma = float(density_scale_gamma)

    def activate(
        self,
        z: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibrated non-negative measure Y0 from latent logit field z0."""
        if getattr(self, "head_activation", "softplus") == "relu":
            y_base = F.relu(z)
            if self.density_adaptive_scale:
                b_eff = (
                    float(self.out_conv.bias.detach().mean().item())
                    if hasattr(self, "out_conv")
                    else 0.0
                )
                if isinstance(self.density_scale_gamma, torch.Tensor):
                    g_val = self.density_scale_gamma.clamp_min(0.0)
                else:
                    g_val = max(float(self.density_scale_gamma), 0.0)
                y_base = y_base * (1.0 + g_val * F.relu(z + b_eff))
            if self.density_curvature:
                return _apply_curvature(
                    y_base,
                    getattr(self, "curvature_alpha", None),
                    curv_scale=self.curv_scale,
                    gated=self.gated_density_curvature,
                    pool_kernel=self.curvature_pool_kernel,
                    dense_threshold=self.curvature_dense_threshold,
                    gate_beta=self.curvature_gate_beta,
                    curvature_pade=getattr(self, "curvature_pade", False),
                )
            return y_base

        return _density_activate(
            z,
            temp_softplus=self.temp_softplus,
            tau=getattr(self, "tau", None),
            density_curvature=self.density_curvature,
            curvature_alpha=getattr(self, "curvature_alpha", None),
            gated_density_curvature=self.gated_density_curvature,
            curvature_dense_threshold=self.curvature_dense_threshold,
            curvature_gate_beta=self.curvature_gate_beta,
            curvature_pool_kernel=self.curvature_pool_kernel,
            curv_scale=self.curv_scale,
            floor_tau=self.floor_tau,
            density_adaptive_scale=self.density_adaptive_scale,
            density_scale_gamma=self.density_scale_gamma,
            curvature_pade=getattr(self, "curvature_pade", False),
            head_activation=getattr(self, "head_activation", "softplus"),
        )

    def forward_logits(
        self, f: tuple[torch.Tensor, ...] | torch.Tensor, scale_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute raw pre-activation logit field z0 modulated by scale routing probabilities."""
        f = f[0] if isinstance(f, tuple) else f
        h = self.pw(self.dw(f))
        if scale_weights is None:
            sw = torch.zeros(h.shape[0], self.num_scales, h.shape[2], h.shape[3], device=h.device, dtype=h.dtype)
            return self.out_conv(torch.cat([h, sw], dim=1))
        sw = F.interpolate(scale_weights, size=h.shape[-2:], mode="bilinear", align_corners=False) if scale_weights.shape[-2:] != h.shape[-2:] else scale_weights
        if sw.shape[1] > self.num_scales:
            sw = sw[:, : self.num_scales, :, :]
        elif sw.shape[1] < self.num_scales:
            sw = F.pad(sw, (0, 0, 0, 0, 0, self.num_scales - sw.shape[1]), mode="constant", value=0.0)
        return self.out_conv(torch.cat([h * (1.0 + self.scale_film(sw)), sw], dim=1))

    def forward(
        self, f: tuple[torch.Tensor, ...] | torch.Tensor, scale_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.activate(self.forward_logits(f, scale_weights=scale_weights), scale_weights=scale_weights)
