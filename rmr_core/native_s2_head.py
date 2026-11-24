from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .density_activation import _compact_c1_inverse, _density_activate, _softplus_inverse


def _conv_bn_silu(cin: int, cout: int, k: int = 3, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, padding=k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.SiLU(inplace=True),
    )


class NativeS2DSIMRHead(nn.Module):
    """Native Stride-2 Decoupled Support-and-Intensity Measure Recovery Head.

    Predicts per-cell support logit z_s and intensity logit z_lambda directly on the
    Stride-2 (H/2 x W/2) Nyquist-preserving feature grid without coarse Stride-4 bottleneck.
    """

    def __init__(
        self,
        width: int = 16,
        init_bias: float = -4.1422,
        floor_tau: float = 0.001,
        density_adaptive_scale: bool = True,
        density_scale_gamma: float = 0.15,
        density_scale_learnable: bool = False,
        head_activation: str = "softplus",
        temp_softplus: bool = False,
        support_init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.head_activation = str(head_activation).lower()
        self.temp_softplus = bool(temp_softplus)
        self.floor_tau = float(floor_tau) / 4.0  # Stride-2 cell convention = m0^(4) / 4
        self.density_adaptive_scale = bool(density_adaptive_scale)
        if self.density_adaptive_scale and bool(density_scale_learnable):
            self.density_scale_gamma = nn.Parameter(torch.tensor([float(density_scale_gamma)]))
        else:
            self.density_scale_gamma = float(density_scale_gamma)

        self.body = nn.Sequential(
            _conv_bn_silu(self.width, self.width, 3, groups=self.width),
            _conv_bn_silu(self.width, self.width, 1),
        )
        self.out_proj = nn.Conv2d(self.width, 2, kernel_size=1, bias=True)
        nn.init.normal_(self.out_proj.weight, std=0.01)

        m0_s4 = math.log1p(math.exp(float(init_bias)))
        s0 = 1.0 / (1.0 + math.exp(-float(support_init_bias)))
        target_s2 = (m0_s4 / 4.0) / max(s0, 1e-4)
        self.adaptive_scale_ref: float = 0.0
        if self.head_activation == "relu":
            b_lambda = target_s2
        elif self.head_activation == "compact_c1":
            b_lambda = _compact_c1_inverse(target_s2, self.floor_tau)
        else:
            b_lambda = _softplus_inverse(target_s2 + 0.5 * self.floor_tau)
            self.adaptive_scale_ref = float(b_lambda) + math.log(4.0)

        with torch.no_grad():
            self.out_proj.bias.copy_(
                torch.tensor([float(support_init_bias), float(b_lambda)], dtype=torch.float32)
            )

    def forward_logits(
        self, f: tuple[torch.Tensor, ...] | torch.Tensor, scale_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        if isinstance(f, tuple):
            f = f[0]
        return self.out_proj(self.body(f))[:, 1:2]

    def activate(
        self, z_lambda: torch.Tensor, scale_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return _density_activate(
            z_lambda,
            temp_softplus=self.temp_softplus,
            tau=None,
            density_curvature=False,
            curvature_alpha=None,
            gated_density_curvature=False,
            curvature_dense_threshold=0.15,
            curvature_gate_beta=0.03,
            curvature_pool_kernel=8,
            curv_scale=1.0,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(f, tuple):
            f = f[0]
        logits = self.out_proj(self.body(f))
        z_s = logits[:, 0:1]
        z_lambda = logits[:, 1:2]
        y_intensity = self.activate(z_lambda)
        y0 = y_intensity * torch.sigmoid(z_s)
        return z_lambda, y0

    def forward(
        self, f: tuple[torch.Tensor, ...] | torch.Tensor, scale_weights: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward_with_logits(f, scale_weights=scale_weights)[1]
