from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .necks import ConvGNAct

# Data-driven prior initialization (RMR-v2/v3):
# Computed empirically from ShanghaiTech Part A training manifest:
# m0 = total_points / total_stride4_cells = 0.015763 count/cell.
# Softplus inverse: softplus(x) = log(exp(x)-1), softplus^{-1}(m0) = log(exp(m0)-1) ≈ -4.1422.
_M0_INIT: float = 0.015763
_FINE_HEAD_BIAS_INIT: float = math.log(math.exp(_M0_INIT) - 1.0)  # ≈ -4.1422


from .density_activation import (
    _apply_curvature,
    _density_activate,
    _pool_local_density,
    _smooth_floor,
    _softplus_inverse,
)

_CURVATURE_ALPHA_INIT: float = -8.0



class FineMeasureHead(nn.Module):
    """Fine-grained density head with calibrated initial rate.

    The final Conv2d bias is initialized so that:
        softplus(bias) ≈ mu0 = 0.015763 count/cell
    i.e. bias ≈ log(exp(0.015763) - 1) ≈ -4.1422.
    Weights of the last conv are initialized to small values (std=0.01).

    When temp_softplus=True (RMR-v7+), the output activation uses a learnable
    temperature τ (initialized to 1):
        output = τ * softplus(z / τ)
    This prevents saturation at high density, fixing the systematic negative bias
    observed in RMR-v6 (Bias=-14.72). τ is clamped to ≥ 0.1 during forward pass.
    """

    def __init__(
        self,
        width: int = 32,
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        temp_softplus: bool = False,
        scale_conditioned: bool = False,
        num_scales: int = 4,
        density_curvature: bool = False,
        gated_density_curvature: bool = False,
        curvature_dense_threshold: float = 0.15,
        curvature_gate_beta: float = 0.03,
        curvature_pool_kernel: int = 8,
        subpixel_stride2: bool = False,
        floor_tau: float = 0.0,
        curvature_alpha_init: float = _CURVATURE_ALPHA_INIT,
        density_adaptive_scale: bool = False,
        density_scale_gamma: float = 0.0,
        density_scale_learnable: bool = False,
        scale_prior_boost: float = 0.0,
        curvature_pade: bool = False,
        head_activation: str = "softplus",
        head_blocks: int = 1,
    ):
        super().__init__()
        self.head_activation = str(head_activation).lower()
        self.scale_prior_boost = float(scale_prior_boost)
        self.subpixel_stride2 = bool(subpixel_stride2)
        self.floor_tau = float(floor_tau) / (4.0 if self.subpixel_stride2 else 1.0)
        out_channels = 4 if self.subpixel_stride2 else 1
        blocks = []
        for _ in range(max(1, int(head_blocks))):
            blocks.append(ConvGNAct(width, width, 3, groups=width))
            blocks.append(ConvGNAct(width, width, 1))
        blocks.append(nn.Conv2d(width, out_channels, 1))
        self.body = nn.Sequential(*blocks)
        final_conv: nn.Conv2d = self.body[-1]  # type: ignore[assignment]
        nn.init.normal_(final_conv.weight, std=0.01)
        m0 = math.log1p(math.exp(float(init_bias)))

        if self.subpixel_stride2:
            self.pixel_shuffle: nn.PixelShuffle | None = nn.PixelShuffle(upscale_factor=2)
            if self.head_activation == "relu":
                nn.init.constant_(final_conv.bias, m0 / 4.0)
            else:
                init_bias_stride2 = _softplus_inverse(m0 / 4.0 + 0.5 * self.floor_tau)
                nn.init.constant_(final_conv.bias, init_bias_stride2)
        else:
            self.pixel_shuffle = None
            if self.head_activation == "relu":
                nn.init.constant_(final_conv.bias, m0)
            else:
                eff_bias = _softplus_inverse(m0 + 0.5 * self.floor_tau) if self.floor_tau > 0.0 else init_bias
                nn.init.constant_(final_conv.bias, eff_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus:
            self.tau = nn.Parameter(torch.ones(1))

        self.density_curvature = bool(density_curvature)
        self.gated_density_curvature = bool(gated_density_curvature)
        self.curvature_pade = bool(curvature_pade)
        self.curv_scale = 4.0 if self.subpixel_stride2 else 1.0
        if self.subpixel_stride2:
            self.curvature_dense_threshold = float(curvature_dense_threshold) / 4.0
            self.curvature_gate_beta = float(curvature_gate_beta) / 4.0
            self.curvature_pool_kernel = int(curvature_pool_kernel) * 2
        else:
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

        self.scale_conditioned = bool(scale_conditioned)
        if self.scale_conditioned:
            self.scale_beta = nn.Parameter(torch.zeros(int(num_scales)))
            self.scale_gamma = nn.Parameter(torch.zeros(int(num_scales)))

    def activate(
        self,
        z: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self, "head_activation", "softplus") == "relu":
            return F.relu(z)

        if self.scale_conditioned and scale_weights is not None:
            if scale_weights.shape[-2:] != z.shape[-2:]:
                scale_weights = F.interpolate(
                    scale_weights, size=z.shape[-2:], mode="bilinear", align_corners=False
                )
            k = self.scale_beta.numel()
            sw = scale_weights[:, :k, :, :]
            b_eff = (sw * self.scale_beta.view(1, k, 1, 1)).sum(dim=1, keepdim=True)
            tau_base = self.tau.clamp_min(0.1) if self.temp_softplus else 1.0
            gamma_shift = (sw * self.scale_gamma.view(1, k, 1, 1)).sum(dim=1, keepdim=True).clamp(-5.0, 5.0)
            tau_eff = (tau_base * torch.exp(gamma_shift)).clamp_min(0.05)
            y_base = tau_eff * F.softplus((z + b_eff) / tau_eff)
            if self.floor_tau > 0.0:
                y_base = _smooth_floor(y_base, self.floor_tau)
            if self.density_adaptive_scale:
                if isinstance(self.density_scale_gamma, torch.Tensor):
                    g_val = self.density_scale_gamma.clamp_min(0.0)
                else:
                    g_val = max(float(self.density_scale_gamma), 0.0)
                y_base = y_base * (1.0 + g_val * F.relu(z + b_eff))
            if self.density_curvature:
                return _apply_curvature(
                    y_base,
                    self.curvature_alpha,
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
            floor_tau=getattr(self, "floor_tau", 0.0),
            density_adaptive_scale=self.density_adaptive_scale,
            density_scale_gamma=self.density_scale_gamma,
            curvature_pade=getattr(self, "curvature_pade", False),
            head_activation=getattr(self, "head_activation", "softplus"),
        )

    def forward_logits(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw pre-activation logit field z0 with scale prior boost."""
        if isinstance(f, tuple):
            f = f[0]
        z = self.body(f)
        if self.subpixel_stride2 and self.pixel_shuffle is not None:
            z = self.pixel_shuffle(z)
        if scale_weights is not None and getattr(self, "scale_prior_boost", 0.0) > 0.0:
            sw_b = scale_weights if scale_weights.shape[-2:] == z.shape[-2:] else F.interpolate(
                scale_weights, size=z.shape[-2:], mode="bilinear", align_corners=False
            )
            z = z + float(self.scale_prior_boost) * sw_b[:, 0:1, :, :].float()
        return z

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return activated density field y0 when activation/curvature is configured, or raw logit for legacy head."""
        z = self.forward_logits(f, scale_weights=scale_weights)
        is_activated = (
            self.temp_softplus
            or self.scale_conditioned
            or self.density_curvature
            or getattr(self, "head_activation", "softplus") == "relu"
        )
        if is_activated:
            return self.activate(z, scale_weights=scale_weights)
        return z


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
        else:
            eff_bias = _softplus_inverse(m0 + 0.5 * self.floor_tau) if self.floor_tau > 0.0 else init_bias
        nn.init.constant_(self.out_conv.bias, eff_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus and self.head_activation != "relu":
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
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw pre-activation logit field z0 modulated by scale routing probabilities."""
        if isinstance(f, tuple):
            f = f[0]
        h = self.pw(self.dw(f))
        if scale_weights is not None:
            if scale_weights.shape[-2:] != h.shape[-2:]:
                scale_weights = F.interpolate(
                    scale_weights, size=h.shape[-2:], mode="bilinear", align_corners=False
                )
            if scale_weights.shape[1] != self.num_scales:
                if scale_weights.shape[1] > self.num_scales:
                    sw = scale_weights[:, :self.num_scales, :, :]
                else:
                    pad_k = self.num_scales - scale_weights.shape[1]
                    sw = F.pad(scale_weights, (0, 0, 0, 0, 0, pad_k), mode="constant", value=0.0)
            else:
                sw = scale_weights
            gamma = 1.0 + self.scale_film(sw)
            h_mod = h * gamma
            h_joint = torch.cat([h_mod, sw], dim=1)
        else:
            pi_zeros = torch.zeros(
                h.shape[0], self.num_scales, h.shape[2], h.shape[3], device=h.device, dtype=h.dtype
            )
            h_joint = torch.cat([h, pi_zeros], dim=1)
        return self.out_conv(h_joint)

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z = self.forward_logits(f, scale_weights=scale_weights)
        return self.activate(z, scale_weights=scale_weights)


def build_fine_head(
    width: int = 32,
    scale_conditioned_fine_head: bool = False,
    num_scales: int = 3,
    init_bias: float = _FINE_HEAD_BIAS_INIT,
    temp_softplus: bool = True,
    scale_conditioned_prior: bool = False,
    density_curvature: bool = False,
    gated_density_curvature: bool = False,
    curvature_dense_threshold: float = 0.15,
    curvature_gate_beta: float = 0.03,
    curvature_pool_kernel: int = 8,
    subpixel_stride2: bool = False,
    floor_tau: float = 0.0,
    curvature_alpha_init: float = _CURVATURE_ALPHA_INIT,
    density_adaptive_scale: bool = False,
    density_scale_gamma: float = 0.0,
    density_scale_learnable: bool = False,
    scale_prior_boost: float = 0.0,
    curvature_pade: bool = False,
    head_activation: str = "softplus",
    head_blocks: int = 1,
    head_type: str = "standard",
    fine_head_dilations: tuple[int, ...] = (1, 2),
    subpixel_kernel_size: int = 3,
    use_film: bool = True,
) -> nn.Module:
    """Factory function for instantiating polymorphic RMR fine density heads."""
    if str(head_type).lower() in ("residual_dilated", "first_principles", "rd_finehead"):
        from .fine_head import ResidualDilatedFineHead
        return ResidualDilatedFineHead(
            width=width,
            num_scales=num_scales,
            subpixel_stride2=subpixel_stride2,
            subpixel_kernel_size=subpixel_kernel_size,
            dilations=fine_head_dilations,
            use_film=use_film,
            init_bias=init_bias,
            temp_softplus=temp_softplus,
            floor_tau=floor_tau,
            head_activation=head_activation,
        )
    kw = {
        "width": width,
        "init_bias": init_bias,
        "temp_softplus": temp_softplus,
        "density_curvature": density_curvature,
        "gated_density_curvature": gated_density_curvature,
        "curvature_dense_threshold": curvature_dense_threshold,
        "curvature_gate_beta": curvature_gate_beta,
        "curvature_pool_kernel": curvature_pool_kernel,
        "floor_tau": floor_tau,
        "curvature_alpha_init": curvature_alpha_init,
        "density_adaptive_scale": density_adaptive_scale,
        "density_scale_gamma": density_scale_gamma,
        "density_scale_learnable": density_scale_learnable,
        "curvature_pade": curvature_pade,
        "head_activation": head_activation,
    }
    if scale_conditioned_fine_head:
        return ScaleConditionedFineHead(num_scales=num_scales, **kw)
    return FineMeasureHead(
        scale_conditioned=scale_conditioned_prior,
        num_scales=num_scales,
        subpixel_stride2=subpixel_stride2,
        scale_prior_boost=scale_prior_boost,
        head_blocks=head_blocks,
        **kw,
    )
