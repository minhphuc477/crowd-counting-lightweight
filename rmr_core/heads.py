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


def _pool_local_density(y: torch.Tensor, k_pool: int) -> torch.Tensor:
    """Compute local average density with dimension-safe pooling.

    Handles 0-D (scalar), 1-D, 2-D [H, W], 3-D [C, H, W], and 4-D [B, C, H, W] tensors
    without dimension mismatch or edge slicing truncation.
    """
    if y.ndim < 2:
        return y
    orig_ndim = y.ndim
    if orig_ndim == 2:
        y_in = y.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y_in = y.unsqueeze(0)
    else:
        y_in = y

    pad = k_pool // 2
    y_local = F.avg_pool2d(
        y_in,
        kernel_size=k_pool,
        stride=1,
        padding=pad,
        count_include_pad=False,
    )
    if y_local.shape[-2:] != y_in.shape[-2:]:
        y_local = y_local[..., : y_in.shape[-2], : y_in.shape[-1]]

    if orig_ndim == 2:
        return y_local.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return y_local.squeeze(0)
    return y_local


def _density_activate(
    z: torch.Tensor,
    *,
    temp_softplus: bool,
    tau: "torch.nn.Parameter | None",
    density_curvature: bool,
    curvature_alpha: "torch.nn.Parameter | None",
    gated_density_curvature: bool,
    curvature_dense_threshold: float,
    curvature_gate_beta: float,
    curvature_pool_kernel: int,
) -> torch.Tensor:
    """Shared density activation: temperature-scaled softplus + optional gated quadratic curvature.

    Eliminates copy-paste between FineMeasureHead and ScaleConditionedFineHead.
    All curvature computation is done in float32 and cast back to z.dtype.
    """
    if temp_softplus and tau is not None:
        tau_clamped = tau.clamp_min(0.1)
        y_base = tau_clamped * F.softplus(z / tau_clamped)
    else:
        y_base = F.softplus(z)

    if density_curvature and curvature_alpha is not None:
        alpha_eff = F.softplus(curvature_alpha)
        orig_dtype = y_base.dtype
        y_base_f32 = y_base.float()
        if gated_density_curvature:
            k_pool = int(curvature_pool_kernel)
            y_local = _pool_local_density(y_base_f32, k_pool)
            tau_dense = float(curvature_dense_threshold)
            beta = float(max(curvature_gate_beta, 1e-4))
            gate_dense = torch.sigmoid((y_local - tau_dense) / beta)
            curv_term = alpha_eff.float() * gate_dense * (y_base_f32 ** 2)
        else:
            curv_term = alpha_eff.float() * (y_base_f32 ** 2)
        return (y_base_f32 + curv_term).to(orig_dtype)
    return y_base


# Learnable density curvature parameter α initialized to -8.0: softplus(-8.0) ≈ 0.000335.
# Preserves seamless Step 0 identity with vanilla softplus, maintains the calibrated m0=0.015763 rate,
# and prevents artificial quadratic overcounting in dense regions (which caused +5.91 MAE regression at -4.0).
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
    ):
        super().__init__()
        self.subpixel_stride2 = bool(subpixel_stride2)
        out_channels = 4 if self.subpixel_stride2 else 1
        self.body = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
            nn.Conv2d(width, out_channels, 1),
        )
        if self.subpixel_stride2:
            self.pixel_shuffle: nn.PixelShuffle | None = nn.PixelShuffle(upscale_factor=2)
            # When subpixel_stride2=True, calibrate bias for Stride 2 cell area (1/4 of Stride 4 cell area)
            init_bias_stride2 = math.log(math.exp(_M0_INIT / 4.0) - 1.0)
            final_conv: nn.Conv2d = self.body[-1]  # type: ignore[assignment]
            nn.init.normal_(final_conv.weight, std=0.01)
            nn.init.constant_(final_conv.bias, init_bias_stride2)
        else:
            self.pixel_shuffle = None
            final_conv = self.body[-1]  # type: ignore[assignment]
            nn.init.normal_(final_conv.weight, std=0.01)
            nn.init.constant_(final_conv.bias, init_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus:
            # Learnable temperature τ; initialized to 1.0 (identical to vanilla softplus)
            self.tau = nn.Parameter(torch.ones(1))

        self.density_curvature = bool(density_curvature)
        self.gated_density_curvature = bool(gated_density_curvature)
        self.curvature_dense_threshold = float(curvature_dense_threshold)
        self.curvature_gate_beta = float(curvature_gate_beta)
        self.curvature_pool_kernel = int(curvature_pool_kernel)
        if self.density_curvature:
            # Learnable density curvature parameter α; initialized to -8.0 so softplus(-8) ≈ 0.000335,
            # providing seamless Step 0 identity with vanilla softplus.
            self.curvature_alpha = nn.Parameter(torch.tensor(_CURVATURE_ALPHA_INIT))

        self.scale_conditioned = bool(scale_conditioned)
        if self.scale_conditioned:
            # Learnable scale coupling vectors β and γ for RMR-v15
            # β modulates effective prior bias b_eff(u) = b0 + β^T π(u)
            # γ modulates effective temperature τ_eff(u) = τ0 * exp(γ^T π(u))
            self.scale_beta = nn.Parameter(torch.zeros(int(num_scales)))
            self.scale_gamma = nn.Parameter(torch.zeros(int(num_scales)))

    def activate(
        self,
        z: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibrated non-negative measure Y0 from latent logit field z0."""
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
            # Apply curvature on top of scale-conditioned base if enabled
            if self.density_curvature:
                alpha_eff = F.softplus(self.curvature_alpha)
                orig_dtype = y_base.dtype
                y_base_f32 = y_base.float()
                if self.gated_density_curvature:
                    k_pool = int(self.curvature_pool_kernel)
                    y_local = _pool_local_density(y_base_f32, k_pool)
                    tau_dense = float(self.curvature_dense_threshold)
                    beta = float(max(self.curvature_gate_beta, 1e-4))
                    gate_dense = torch.sigmoid((y_local - tau_dense) / beta)
                    curv_term = alpha_eff.float() * gate_dense * (y_base_f32 ** 2)
                else:
                    curv_term = alpha_eff.float() * (y_base_f32 ** 2)
                return (y_base_f32 + curv_term).to(orig_dtype)
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
        )

    def forward_logits(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw pre-activation logit field z0."""
        if isinstance(f, tuple):
            f = f[0]
        z = self.body(f)
        if self.subpixel_stride2 and self.pixel_shuffle is not None:
            z = self.pixel_shuffle(z)
        return z

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the pre-activation logit field z0 (legacy interface, preserved for rmr_v2).

        NOTE: This method intentionally returns a raw logit when temp_softplus=False,
        and a density when temp_softplus=True — a known inconsistency retained for
        backward-compatibility with rmr_v2/model.py which calls:
            z0 = fine_head(f)
            y0 = F.softplus(z0)
        New code (rmr_v3+) must call forward_logits() + activate() separately.
        """
        z = self.forward_logits(f)
        if self.temp_softplus or self.scale_conditioned or self.density_curvature:
            return self.activate(z, scale_weights=scale_weights)
        return z



class ScaleConditionedFineHead(nn.Module):
    """Scale-Conditioned Dynamic Fine Density Head (RMR-v22).

    Upgrades the static 1x1 projection into a content-adaptive head:
    1. Local Depthwise Context: 3x3 depthwise conv with groups=width.
       Provides a 12x12 px receptive field on the input image to perceive head contours.
    2. Continuous Scale Simplex Modulation (FiLM):
       Scale routing probabilities pi(u) in Delta^{K-1} predict channel-wise scaling:
           gamma(u) = 1.0 + W_s * pi(u)  (K * width params, zero-initialized)
           h_mod = h * gamma(u)
    3. Pointwise Joint Density Projection:
       Projects concatenated [h_mod, pi] (width + K ch) -> 1 ch.
    4. Calibrated Bias Initialization & Gated Curvature Power:
       Zero init on pi weights ensures exact Step-0 identity with calibrated prior b0.
       Includes learnable temperature tau and gated quadratic curvature expansion.
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
    ):
        super().__init__()
        self.dw = ConvGNAct(width, width, 3, groups=width)
        self.pw = ConvGNAct(width, width, 1)

        self.num_scales = int(num_scales)
        self.scale_film = nn.Conv2d(self.num_scales, width, kernel_size=1, bias=False)
        nn.init.zeros_(self.scale_film.weight)

        self.out_conv = nn.Conv2d(width + self.num_scales, 1, kernel_size=1)
        nn.init.normal_(self.out_conv.weight, std=0.01)
        nn.init.constant_(self.out_conv.bias, init_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus:
            self.tau = nn.Parameter(torch.ones(1))

        self.density_curvature = bool(density_curvature)
        self.gated_density_curvature = bool(gated_density_curvature)
        self.curvature_dense_threshold = float(curvature_dense_threshold)
        self.curvature_gate_beta = float(curvature_gate_beta)
        self.curvature_pool_kernel = int(curvature_pool_kernel)
        if self.density_curvature:
            # Learnable density curvature parameter α; initialized to -8.0 so softplus(-8) ≈ 0.000335,
            # providing seamless Step 0 identity with vanilla softplus.
            self.curvature_alpha = nn.Parameter(torch.tensor(_CURVATURE_ALPHA_INIT))

    def activate(
        self,
        z: torch.Tensor,
        scale_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute calibrated non-negative measure Y0 from latent logit field z0."""
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
        return self.activate(z)


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
) -> nn.Module:
    """Factory function for instantiating polymorphic RMR fine density heads."""
    if scale_conditioned_fine_head:
        return ScaleConditionedFineHead(
            width=width,
            num_scales=num_scales,
            init_bias=init_bias,
            temp_softplus=temp_softplus,
            density_curvature=density_curvature,
            gated_density_curvature=gated_density_curvature,
            curvature_dense_threshold=curvature_dense_threshold,
            curvature_gate_beta=curvature_gate_beta,
            curvature_pool_kernel=curvature_pool_kernel,
        )
    return FineMeasureHead(
        width=width,
        init_bias=init_bias,
        temp_softplus=temp_softplus,
        scale_conditioned=scale_conditioned_prior,
        num_scales=num_scales,
        density_curvature=density_curvature,
        gated_density_curvature=gated_density_curvature,
        curvature_dense_threshold=curvature_dense_threshold,
        curvature_gate_beta=curvature_gate_beta,
        curvature_pool_kernel=curvature_pool_kernel,
        subpixel_stride2=subpixel_stride2,
    )


