from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import build_fine_head
from rmr_core.necks import AdditiveFPNNeck, ASPPLiteFPNNeck, CoordinateAttention, RepWeightedFPNNeck
from rmr_core.scale_routing import ScaleRoutingHead, FactorizedRoutingHead
from rmr_core.types import RMRModelOutput
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    region_mean_std_features,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from .regional_head import (
    ProbabilisticRegionalEvidenceHead,
    apply_scale_consistency_gating,
    reliability_from_nb,
)
from .solver import unrolled_sirt_solver


def _softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


def _deep_tuple(val: Any) -> Any:
    if isinstance(val, (list, tuple)):
        return tuple(_deep_tuple(x) for x in val)
    return val


@dataclass
class RMRv3Config:
    # Fine grid / carrier
    output_stride: int = 4
    feature_width: int = 32
    backbone_name: str = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    pretrained: bool = True
    backbone_lr_scale: float = 0.1
    init_m0: float = 0.015763

    # Neck
    neck_type: str = "additive"  # "additive" | "aspp_lite" | "rep_weighted"
    context_dilations: tuple[int, ...] = (1, 2, 3)  # dilations for AdditiveFPNNeck / RepWeightedFPNNeck
    use_aspp_gap: bool = True  # if True AND neck_type=="aspp_lite", enable GAP branch in ASPP-lite
    aspp_dilations: tuple[int, ...] = (1, 3, 6)  # dilations for ASPPLiteFPNNeck branches

    # Region dictionary
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False

    # Solver
    enable_solver: bool = True
    iterations: int = 2
    omega: float = 1.0
    residual_clip: float = 0.0
    eps: float = 1e-6

    # Negative-Binomial regional uncertainty
    dispersion_init: float = 50.0
    dispersion_min: float = 0.5
    dispersion_max: float = 500.0

    # Reliability: "nb_rate_variance" | "snr" | "hybrid_hurdle" (RMR-v14)
    reliability_mode: str = "nb_rate_variance"
    reliability_rate_std_floor: float = 0.01
    reliability_weight_min: float = 0.25
    reliability_weight_max: float = 4.0
    normalize_reliability_within_scale: bool = True

    # Registered clean-causal variant
    detach_region_mean_in_solver: bool = True
    detach_reliability_in_solver: bool = True
    uniform_reliability: bool = False

    # V4 candidate switches
    native_scale_pooling: bool = False
    regional_feature_stats: str = "mean"

    # Region head MLP hidden dim (default 48 preserves all prior checkpoints)
    region_head_hidden: int = 48

    # ── RMR-v7 additions ──────────────────────────────────────────────────────
    # Hurdle NB head: predicts region occupancy probability π_R.
    # When enabled, the solver target is gated by occupancy probability:
    #     b_solver_R = sigmoid(z_π_R).detach() * mu_count_R.detach()
    # π_R → 0 for empty background regions ⟹ b_solver_R → 0 (suppresses false alarms).
    # π_R → 1 for occupied crowd regions ⟹ b_solver_R ≈ mu_count_R (full guidance).
    hurdle_head: bool = False

    # Total-variation Laplacian smoothing coefficient inside the SIRT loop.
    # Applied after each SIRT update: y ← y - tv_lambda * Δy  (Δ = Laplacian)
    # 0.0 disables smoothing (default, backward compatible).
    tv_lambda: float = 0.0

    # EMA weight tracking decay (0.0 = disabled, typical 0.999).
    # When > 0, train.py maintains an EMA shadow copy and saves it as best checkpoint.
    # Decouples best-epoch tracking from weight drift observed after epoch 445 in v6.
    ema_decay: float = 0.0

    # Learnable Temperature Softplus for FineMeasureHead.
    # When True, FineMeasureHead uses τ * softplus(z/τ) instead of softplus(z).
    # Fixes negative bias saturation at high density (Bias=-14.72 in v6 → near 0 target).
    temp_softplus: bool = False

    # ── RMR-v8 Stage 2 additions ──────────────────────────────────────────────
    # Solver update rule.
    # "additive":       plain SIRT adjoint scatter (v7 default — backward compatible)
    # "multiplicative": gated SIRT adjoint — gate = tanh(|y| / density_gate_rho)
    #                   suppresses correction on near-zero pixels, fixing background
    #                   lift degradation observed at T>=2 with additive mode.
    solver_mode: str = "additive"

    # Gate threshold ρ₀ for multiplicative SIRT.
    # Pixels with density << rho0 have gate ≈ 0 (no correction).
    # Pixels with density >> rho0 have gate ≈ 1 (full correction).
    # Default 0.02 is ~1.27 × init_m0 (empirical mean cell density = 0.015763).
    density_gate_rho: float = 0.02
    density_gate_floor: float = 0.02

    # Total-variation diffusion type applied after each SIRT step.
    # "laplacian":   isotropic Laplacian (v7 default — backward compatible).
    # "charbonnier": anisotropic Charbonnier TV — edge-preserving, suppresses
    #                diffusion across crowd/background boundaries.
    tv_type: str = "laplacian"

    # Charbonnier TV regularization epsilon ε_c.
    # Controls edge-sharpness threshold: |grad_y| >> eps_c → diffusivity → 0.
    # Only used when tv_type=="charbonnier". Default 0.1 ensures CFL stability with lambda_tv=0.015.
    tv_eps_c: float = 0.1

    # ── RMR-v8 Stage 3 additions ──────────────────────────────────────────────
    # Coordinate Attention on the P4 output of ASPPLiteFPNNeck.
    # Uses GroupNorm(1, 8) (NOT BatchNorm — batch_size=1 eval compatibility).
    # Adds 848 parameters. Only valid when neck_type=="aspp_lite".
    use_coord_attn: bool = False

    # ── RMR-v9.1 / AQ-RMR additions ──────────────────────────────────────────
    # Proximal L1-Soft-Thresholding threshold τ in the SIRT solver loop.
    # When > 0, applies proximal operator after the additive update, introducing a noise deadband
    # [0, τ] that completely suppresses background mass smearing (phantom count lift).
    proximal_tau: float = 0.0

    # ── RMR-v10 additions ────────────────────────────────────────────────────
    # Proximal mode:
    # "firm": Minimax Concave Penalty (MCP) firm thresholding. Zero shrinkage on peaks (dense anti-erosion).
    # "soft": Standard L1 soft-thresholding S_tau^+(z) = max(0, z - tau) (backward compatible default).
    # "none": Non-negative clamp only.
    proximal_mode: str = "soft"
    proximal_mu: float = 3.0

    # ── Dynamic Scale Routing (RMR-v10) ──────────────────────────────────────
    # When True, instantiates ScaleRoutingHead (483 params) to predict continuous
    # spatial scale probability pi(x, y) in Delta^{K-1} over the K observation scales.
    dynamic_scale_routing: bool = False
    scale_router_temperature: float = 1.0

    # ── RMR-v11 additions ────────────────────────────────────────────────────
    # Trust-region relative update clamping in SIRT solver:
    # Bounds relative update delta_y_t to [-kappa * y_t, +kappa * max(y_t, floor)]
    trust_region_kappa: float = 0.0
    trust_region_floor: float = 0.005

    # Decoupled foreground gating sub-head (1x1 conv, 33 params)
    # When True, predicts spatial foreground probability map to suppress texture false alarms
    foreground_gate: bool = False

    # ── RMR-v13 additions ────────────────────────────────────────────────────
    # Adjoint back-projection mode in SIRT solver:
    # "flat": standard Lebesgue uniform scatter (v10 default).
    # "radon_nikodym": measure-modulated scatter A_nu^T where mass is back-projected
    #                  proportionally to y_t(u) / ((Ay_t)_m + eps), eliminating background lift.
    adjoint_mode: str = "flat"

    # Morozov discrepancy shrinkage threshold gamma.
    # When > 0, shrinks discrepancy by gamma * sqrt(Var[b]), preventing solver over-fitting
    # to noisy regional measurements on ambiguous regions.
    morozov_gamma: float = 0.0

    # ── RMR-v14 additions ────────────────────────────────────────────────────
    # Top-Down Semantic Context Gating (TDSG):
    # When True, instantiates Conv2d(feature_width, 1, 1) (33 params) on P16
    # to semantically modulate P4 high-resolution features before the fine head.
    use_top_down_semantic_gate: bool = False
    tdsg_floor: float = 0.20

    # Decoupled foreground gating safety floor (default 0.70 for backward compatibility, 0.10 for v14)
    fg_gate_floor: float = 0.70

    # ── RMR-v15 Native Dynamic Geometry additions ────────────────────────────
    # Scale-conditioned fine density prior: modulates FineMeasureHead bias and temperature
    # via spatial scale routing weights pi(u) (8 learnable parameters).
    scale_conditioned_prior: bool = False

    # Pre-solver scale-consistency reliability gating: suppresses regional reliability
    # of coarse boxes overlapping dense clusters using regional_sum(pi_k, boxes_k).
    pre_solver_scale_gating: bool = False
    scale_gating_power: float = 1.0

    # Convex Dynamic Trust Gate: learns image-level trust alpha(x) in [0, 1] on GAP(P16)
    # to eliminate solver harm (33 learnable parameters).
    dynamic_trust_gate: bool = False
    trust_gate_init_bias: float = 1.73  # sigmoid(1.73) ≈ 0.85

    # ── RMR-v17 Multi-Scale Adaptive Measure Reconstruction additions ───────
    # Quadratic density curvature warping in FineMeasureHead (1 parameter).
    # Expands dynamic range on extreme crowd clumps (>1200 people) without feature saturation.
    density_curvature: bool = False

    # Barzilai-Borwein dynamic adaptive step size in unrolled SIRT solver (0 parameters).
    # Adapts omega_t per iteration to local energy curvature, eliminating Dirac oscillations.
    use_barzilai_borwein: bool = False

    # Scale-entropy modulated trust region bound in unrolled SIRT solver (0 parameters).
    # Scales trust bound by local scale router confidence map C(u) = 1 - H(pi) / log(K).
    use_scale_entropy_trust: bool = False

    # ── RMR-v18 Perspective Hybrid Window Dictionary additions ───────────────
    # Learnable vertical linear perspective bias in ScaleRoutingHead (num_scales parameters).
    perspective_scale_bias: bool = False

    # Geometric horizon suppression for anisotropic vertical boxes in pre-solver gating (0 parameters).
    perspective_horizon_gate: bool = False
    horizon_cutoff: float = 0.35

    # ── RMR-v19 Decoupled 2D Factorized Routing & Density-Gated Curvature ─────
    # Decouples scale (S=3) and aspect ratio (A=2) via FactorizedRoutingHead (+68 params).
    # Eliminates scale starvation and preserves strict monotonicity in scale alignment loss.
    factorized_scale_routing: bool = False
    num_marginal_scales: int = 3
    num_aspect_ratios: int = 2

    # Spatially-conditioned density gate on quadratic curvature (0 parameters).
    # Shuts down curvature on low-density pavement/facades (IMG_113) while preserving
    # dynamic range expansion in dense crowd clusters.
    gated_density_curvature: bool = False
    curvature_dense_threshold: float = 0.15
    curvature_gate_beta: float = 0.03
    curvature_pool_kernel: int = 8

    # ── RMR-v20 Sub-60 Mathematical Breakthrough additions ───────────────────
    # Micro Perspective Coordinate Attention on carrier P4 (+456 parameters).
    # Factorizes 1D horizontal and 1D vertical spatial pooling to condition features
    # directly on camera perspective elevation gradients.
    use_micro_coord_attn: bool = False
    micro_coord_reduction: int = 8

    # Nesterov momentum extrapolation in unrolled SIRT solver (0 parameters).
    # Accelerates convergence to Dirac fixed-point y* from O(1/T) to O(1/T^2).
    use_nesterov_momentum: bool = False

    # Density-adaptive solver over-relaxation Omega(u) (0 parameters).
    # Dynamically scales step size omega in [0.70, 1.50] based on smoothed density,
    # boosting recovery on dense clumps while preventing background noise lift.
    adaptive_relaxation: bool = False
    adaptive_relax_sparse: float = 0.70
    adaptive_relax_dense_boost: float = 0.50
    adaptive_relax_threshold: float = 0.05
    adaptive_relax_scale: float = 0.02

    # Hybrid Radon-Nikodym / Lebesgue Recovery Flux (RMR-v21)
    # alpha_recov in [0.0, 1.0] breaks the zero-absorbing barrier by injecting
    # an additive discovery seed when y_0 = 0 in occluded dense crowd clumps.
    hybrid_recovery_alpha: float = 0.0

    # ── RMR-v22 additions ────────────────────────────────────────────────────
    # Scale-conditioned dynamic fine density head:
    # 3x3 depthwise context filter + FiLM scale modulation + joint projection (+99 parameters).
    scale_conditioned_fine_head: bool = False

    # Density-gated anisotropic TV diffusion in unrolled SIRT solver (0 parameters).
    # Smooths noise in background/sparse regions while shutting down diffusion to strictly zero
    # in dense crowd clusters to preserve sharp head separation.
    density_gated_diffusion: bool = False
    diffusion_dense_threshold: float = 0.15
    diffusion_gate_beta: float = 0.03

    # ── RMR-v24 Breakthrough additions ───────────────────────────────────────
    # Micro-Perspective 1D Carrier Elevation:
    # 1D vertical elevation projection (Linear(1, 32), +64 parameters)
    # mod = tanh(W * v + b) where v in [-1, 1] encodes camera elevation foreshortening.
    use_perspective_elevation: bool = False

    # Alternating Barzilai-Borwein (ABB) step size in unrolled SIRT solver (0 parameters):
    # Alternates between BB-1 (Rayleigh quotient) on even secant steps and BB-2
    # (inverse Rayleigh quotient) on odd secant steps to break cyclic attractor limit cycles.
    use_alternating_bb: bool = False

    # ── RMR-v26 Pure BB-1 Rayleigh Contraction with Trust Damping ────────────
    # Bounds Barzilai-Borwein step size to [bb_clamp_min * omega, bb_clamp_max * omega]
    # to eliminate oscillatory gradient swings in hyper-dense crowd clusters.
    bb_clamp_min: float = 0.5
    bb_clamp_max: float = 1.2

    def __post_init__(self) -> None:
        if self.region_sizes_px is not None:
            self.region_sizes_px = _deep_tuple(self.region_sizes_px)
        if self.aspp_dilations is not None:
            self.aspp_dilations = _deep_tuple(self.aspp_dilations)
        if self.context_dilations is not None:
            self.context_dilations = _deep_tuple(self.context_dilations)

        if self.use_coord_attn and self.neck_type != "aspp_lite":
            raise ValueError(
                f"use_coord_attn=True requires neck_type='aspp_lite', got '{self.neck_type}'"
            )
        if self.solver_mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"solver_mode must be 'additive' or 'multiplicative', got '{self.solver_mode}'"
            )
        if self.tv_type not in ("laplacian", "charbonnier"):
            raise ValueError(
                f"tv_type must be 'laplacian' or 'charbonnier', got '{self.tv_type}'"
            )
        if self.proximal_tau < 0.0:
            raise ValueError(
                f"proximal_tau must be non-negative, got {self.proximal_tau}"
            )
        if self.proximal_mode not in ("firm", "soft", "none", "clamp"):
            raise ValueError(
                f"proximal_mode must be 'firm', 'soft', or 'none', got '{self.proximal_mode}'"
            )
        if self.proximal_mu <= 1.0:
            raise ValueError(
                f"proximal_mu must be > 1.0, got {self.proximal_mu}"
            )
        if self.tv_lambda < 0.0:
            raise ValueError(
                f"tv_lambda must be non-negative, got {self.tv_lambda}"
            )
        if self.scale_router_temperature <= 0.0:
            raise ValueError(
                f"scale_router_temperature must be strictly positive, got {self.scale_router_temperature}"
            )
        if self.trust_region_kappa < 0.0:
            raise ValueError(
                f"trust_region_kappa must be non-negative, got {self.trust_region_kappa}"
            )
        if self.trust_region_floor <= 0.0:
            raise ValueError(
                f"trust_region_floor must be strictly positive, got {self.trust_region_floor}"
            )
        if self.adjoint_mode not in ("flat", "radon_nikodym"):
            raise ValueError(
                f"adjoint_mode must be 'flat' or 'radon_nikodym', got '{self.adjoint_mode}'"
            )
        if self.morozov_gamma < 0.0:
            raise ValueError(
                f"morozov_gamma must be non-negative, got {self.morozov_gamma}"
            )
        if self.reliability_mode not in ("nb_rate_variance", "rate_variance", "snr", "hybrid_hurdle"):
            raise ValueError(
                f"Unsupported reliability_mode: {self.reliability_mode}. Must be 'nb_rate_variance', 'snr', or 'hybrid_hurdle'."
            )
        if not (0.0 <= self.tdsg_floor <= 1.0):
            raise ValueError(
                f"tdsg_floor must be in [0.0, 1.0], got {self.tdsg_floor}"
            )
        if not (0.0 <= self.fg_gate_floor <= 1.0):
            raise ValueError(
                f"fg_gate_floor must be in [0.0, 1.0], got {self.fg_gate_floor}"
            )
        if not (0.0 <= self.hybrid_recovery_alpha <= 1.0):
            raise ValueError(
                f"hybrid_recovery_alpha must be in [0.0, 1.0], got {self.hybrid_recovery_alpha}"
            )


    @classmethod
    def from_dict(cls, d: dict | None, **overrides) -> "RMRv3Config":
        """Cleanly instantiate RMRv3Config from dictionary with aliases and type coercion."""
        merged = dict(d or {})
        merged.update(overrides)

        # Handle canonical aliases
        if "backbone" in merged and "backbone_name" not in merged:
            merged["backbone_name"] = merged.pop("backbone")
        if "sirt_omega" in merged and "omega" not in merged:
            merged["omega"] = merged.pop("sirt_omega")

        kwargs: dict = {}
        for f in fields(cls):
            k = f.name
            if k in merged:
                v = merged[k]
                if v is None:
                    continue
                if "tuple" in str(f.type) and isinstance(v, (list, tuple)):
                    kwargs[k] = tuple(
                        tuple(x) if isinstance(x, (list, tuple)) else x
                        for x in v
                    )
                elif f.type is bool or f.type == "bool":
                    kwargs[k] = bool(v)
                elif f.type is int or f.type == "int":
                    kwargs[k] = int(v)
                elif f.type is float or f.type == "float":
                    kwargs[k] = float(v)
                elif f.type is str or f.type == "str":
                    kwargs[k] = str(v)
                else:
                    kwargs[k] = v
        return cls(**kwargs)


__all__ = [
    "RMRv3Config",
    "RMRv3",
    "MicroCoordAttn",
    "MicroPerspectiveElevation",
    "ProbabilisticRegionalEvidenceHead",
    "reliability_from_nb",
    "region_mean_std_features",
    "weighted_coverage",
    "weighted_normalized_adjoint_field",
    "weighted_regional_energy",
]


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
        h = x.shape[-2]
        # v: [H, 1] normalized vertical coordinates from -1.0 (top) to +1.0 (bottom)
        v = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=self.proj.weight.dtype).view(h, 1)
        # elevation_mod: [C, H, 1] broadcasts cleanly across any batch and width dims
        elevation_mod = self.proj(v).transpose(0, 1).unsqueeze(-1)
        if x.ndim == 4:
            elevation_mod = elevation_mod.unsqueeze(0)  # [1, C, H, 1]

        # MPE-v2: Mass-Conserved Zero-Inflation Perspective Modulation
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
        mid_channels = max(4, channels // reduction)  # 32 // 8 = 4
        self.conv_shared = nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False)
        self.gn = nn.GroupNorm(1, mid_channels)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(mid_channels, channels, kernel_size=1, bias=True)

        # Near-identity warm-start initialization:
        # Small weights (std=1e-3) and +3.5 bias ensure sigmoid(3.5)*sigmoid(3.5) ≈ 0.942,
        # preventing the catastrophic 75% carrier feature attenuation at epoch 0 while
        # preserving non-zero backpropagation gradients to shared conv and groupnorm.
        nn.init.normal_(self.conv_h.weight, std=1e-3)
        nn.init.constant_(self.conv_h.bias, 3.5)
        nn.init.normal_(self.conv_w.weight, std=1e-3)
        nn.init.constant_(self.conv_w.bias, 3.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_h = x.mean(dim=-1, keepdim=True)  # [B, C, H, 1]
        x_w = x.mean(dim=-2, keepdim=True).permute(0, 1, 3, 2)  # [B, C, W, 1]

        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.gn(self.conv_shared(y)))

        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.permute(0, 1, 3, 2)  # [B, mid_channels, 1, W]

        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))

        return x * a_h * a_w


class RMRv3(nn.Module):
    """Reliability-Weighted Regional Measure Reconciliation."""

    def __init__(
        self,
        cfg: RMRv3Config | None = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg = RMRv3Config()

        if cfg.output_stride != 4:
            raise ValueError(
                "RMR-v3 registered method requires output_stride=4"
            )

        if cfg.include_full_image:
            raise ValueError(
                "RMR-v3 registered method requires include_full_image=False"
            )

        if cfg.enable_solver and cfg.iterations < 1:
            raise ValueError("iterations must be >= 1 when enable_solver=True")

        if cfg.enable_solver and cfg.omega <= 0:
            raise ValueError("omega must be > 0 when enable_solver=True")

        if cfg.reliability_mode not in ("nb_rate_variance", "rate_variance", "snr", "hybrid_hurdle"):
            raise ValueError(
                f"Unsupported reliability_mode: {cfg.reliability_mode}. Must be 'nb_rate_variance', 'snr', or 'hybrid_hurdle'."
            )

        if len(cfg.region_sizes_px) == 0:
            raise ValueError(
                "region_sizes_px must not be empty"
            )

        if cfg.reliability_weight_min <= 0:
            raise ValueError(
                f"reliability_weight_min ({cfg.reliability_weight_min}) must be > 0"
            )

        if not (cfg.reliability_weight_min < cfg.reliability_weight_max):
            raise ValueError(
                f"reliability_weight_min ({cfg.reliability_weight_min}) must be < reliability_weight_max ({cfg.reliability_weight_max})"
            )

        if cfg.reliability_rate_std_floor <= 0:
            raise ValueError(
                f"reliability_rate_std_floor ({cfg.reliability_rate_std_floor}) must be > 0"
            )

        self.cfg = cfg

        self.encoder = MobileNetV4Backbone(
            model_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            target_reductions=(4, 8, 16),
        )

        if cfg.neck_type == "rep_weighted":
            self.fusion = RepWeightedFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
                context_dilations=cfg.context_dilations,
            )
        elif cfg.neck_type == "aspp_lite":
            self.fusion = ASPPLiteFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
                aspp_dilations=cfg.aspp_dilations,
                use_aspp_gap=cfg.use_aspp_gap,
            )
        elif cfg.neck_type == "additive":
            self.fusion = AdditiveFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
            )
        else:
            raise ValueError(
                f"Unsupported neck_type: '{cfg.neck_type}'. Must be 'additive', 'aspp_lite', or 'rep_weighted'."
            )

        init_bias = _softplus_inverse(cfg.init_m0)

        self.fine_head = build_fine_head(
            width=cfg.feature_width,
            scale_conditioned_fine_head=cfg.scale_conditioned_fine_head,
            num_scales=len(cfg.region_sizes_px),
            init_bias=init_bias,
            temp_softplus=cfg.temp_softplus,
            scale_conditioned_prior=cfg.scale_conditioned_prior,
            density_curvature=cfg.density_curvature,
            gated_density_curvature=cfg.gated_density_curvature,
            curvature_dense_threshold=cfg.curvature_dense_threshold,
            curvature_gate_beta=cfg.curvature_gate_beta,
            curvature_pool_kernel=cfg.curvature_pool_kernel,
        )

        self.region_head = ProbabilisticRegionalEvidenceHead(
            feature_dim=cfg.feature_width,
            hidden=cfg.region_head_hidden,
            init_rate=cfg.init_m0,
            region_sizes_px=cfg.region_sizes_px,
            dispersion_init=cfg.dispersion_init,
            dispersion_min=cfg.dispersion_min,
            dispersion_max=cfg.dispersion_max,
            native_scale_pooling=cfg.native_scale_pooling,
            regional_feature_stats=cfg.regional_feature_stats,
            hurdle_head=cfg.hurdle_head,
        )

        # ── Stage 3: Coordinate Attention on P4 ───────────────────────────────
        if cfg.use_coord_attn:
            if cfg.neck_type != "aspp_lite":
                raise ValueError(
                    f"use_coord_attn=True requires neck_type='aspp_lite', got '{cfg.neck_type}'"
                )
            self.coord_attn: CoordinateAttention | None = CoordinateAttention(
                channels=cfg.feature_width,
                reduction=4,
            )
        else:
            self.coord_attn = None

        # ── RMR-v20: Micro Perspective Coordinate Attention on P4 (+456 params) ──
        if cfg.use_micro_coord_attn:
            self.micro_coord_attn: MicroCoordAttn | None = MicroCoordAttn(
                channels=cfg.feature_width,
                reduction=cfg.micro_coord_reduction,
            )
        else:
            self.micro_coord_attn = None

        # ── RMR-v24: Micro-Perspective 1D Carrier Elevation (+64 params) ─────
        if cfg.use_perspective_elevation:
            self.perspective_elevation: MicroPerspectiveElevation | None = MicroPerspectiveElevation(
                channels=cfg.feature_width
            )
        else:
            self.perspective_elevation = None

        # ── Dynamic Scale Routing (RMR-v10/v18/v19) ──────────────────────────
        if cfg.factorized_scale_routing:
            self.scale_router: FactorizedRoutingHead | ScaleRoutingHead | None = FactorizedRoutingHead(
                in_channels=cfg.feature_width,
                num_scales=cfg.num_marginal_scales,
                num_aspect_ratios=cfg.num_aspect_ratios,
                temperature=cfg.scale_router_temperature,
                perspective_bias=cfg.perspective_scale_bias,
            )
        elif cfg.dynamic_scale_routing:
            self.scale_router = ScaleRoutingHead(
                in_channels=cfg.feature_width,
                num_scales=len(cfg.region_sizes_px),
                temperature=cfg.scale_router_temperature,
                perspective_bias=cfg.perspective_scale_bias,
            )
        else:
            self.scale_router = None

        # ── Foreground Gating Sub-Head (RMR-v11) ──────────────────────────
        if cfg.foreground_gate:
            self.fg_gate: nn.Conv2d | None = nn.Conv2d(
                cfg.feature_width,
                1,
                kernel_size=1,
                bias=True,
            )
            # Initialize with small positive bias (2.0 -> sigmoid ~0.88) so early training passes signal through
            nn.init.normal_(self.fg_gate.weight, std=0.01)
            nn.init.constant_(self.fg_gate.bias, 2.0)
        else:
            self.fg_gate = None

        # ── Top-Down Semantic Context Gating (RMR-v14) ────────────────────────
        if cfg.use_top_down_semantic_gate:
            self.tdsg: nn.Conv2d | None = nn.Conv2d(
                cfg.feature_width,
                1,
                kernel_size=1,
                bias=True,
            )
            # Initialize with positive bias so early training passes full signal through
            nn.init.normal_(self.tdsg.weight, std=0.01)
            nn.init.constant_(self.tdsg.bias, 2.0)
        else:
            self.tdsg = None

        # ── Convex Dynamic Trust Gate (RMR-v15) ──────────────────────────────
        if cfg.dynamic_trust_gate:
            self.trust_gate: nn.Linear | None = nn.Linear(cfg.feature_width, 1)
            nn.init.zeros_(self.trust_gate.weight)
            init_tb = cfg.trust_gate_init_bias
            nn.init.constant_(self.trust_gate.bias, float(init_tb))
        else:
            self.trust_gate = None

        self.solver_strength: float = 1.0

        # Registered non-persistent Laplacian kernel buffer for isotropic TV smoothing
        self.register_buffer(
            "_laplace_kernel",
            torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
            persistent=False,
        )

        self._region_cache: OrderedDict[
            tuple,
            RegionSet,
        ] = OrderedDict()

        # Strict parameter budget assertion (max 105,000 parameters)
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if total_trainable > 105000:
            raise ValueError(
                f"Strict parameter ceiling exceeded: {total_trainable} > 105,000 parameters. "
                "Check architecture configuration."
            )

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

    def _regions(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> RegionSet:
        key = (
            h,
            w,
            self.cfg.output_stride,
            _deep_tuple(self.cfg.region_sizes_px),
            self.cfg.region_overlap,
            device.type,
            device.index if device.type == "cuda" else None,
        )

        if key in self._region_cache:
            self._region_cache.move_to_end(key)
            return self._region_cache[key]

        if len(self._region_cache) >= 32:
            self._region_cache.popitem(last=False)

        region_set = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=False,
            device=device,
        )
        self._region_cache[key] = region_set
        return region_set

    def _extract_carrier_features(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract multi-scale carrier features from backbone and neck with optional context attention."""
        c4, c8, c16 = self.encoder(x)
        p4, p8, p16 = self.fusion(c4, c8, c16)

        # Top-Down Semantic Context Gating (RMR-v14)
        if self.tdsg is not None:
            sem_logit = self.tdsg(p16)
            sem_gate = F.interpolate(
                sem_logit, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )
            sem_floor = float(self.cfg.tdsg_floor)
            sem_mask = sem_floor + (1.0 - sem_floor) * torch.sigmoid(sem_gate)
            p4 = p4 * sem_mask

        # Coordinate Attention on P4 (RMR-v8 Stage 3)
        if self.coord_attn is not None:
            p4 = self.coord_attn(p4)

        # Micro Perspective Coordinate Attention on P4 (RMR-v20)
        if self.micro_coord_attn is not None:
            p4 = self.micro_coord_attn(p4)

        # Micro-Perspective 1D Carrier Elevation (RMR-v24)
        if self.perspective_elevation is not None:
            p4 = self.perspective_elevation(p4)

        return p4, p8, p16

    def _route_scales(
        self, p4: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Predict continuous spatial scale and aspect ratio probability fields."""
        if self.scale_router is None:
            return None, None, None

        router_out = self.scale_router(p4)
        if isinstance(router_out, tuple):
            scale_weights, pi_scale, pi_aspect = router_out
            return scale_weights, pi_scale, pi_aspect
        return router_out, None, None

    def _predict_fine_density(
        self, p4: torch.Tensor, scale_weights: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Predict pre-solver fine density logits z0 and non-negative density y0 with optional foreground gating."""
        z0 = self.fine_head.forward_logits(p4, scale_weights=scale_weights)
        y0 = self.fine_head.activate(z0, scale_weights=scale_weights)

        fg_logit = None
        if self.fg_gate is not None:
            fg_logit = self.fg_gate(p4)
            floor = float(self.cfg.fg_gate_floor)
            fg_mask = floor + (1.0 - floor) * torch.sigmoid(fg_logit)
            y0 = y0 * fg_mask

        return z0, y0, fg_logit

    def _extract_regional_evidence(
        self,
        p4: torch.Tensor,
        p8: torch.Tensor,
        p16: torch.Tensor,
        regions: RegionSet,
        scale_weights: torch.Tensor | None,
        uniform_reliability: bool,
        grid_h: int,
    ) -> dict[str, Any]:
        """Extract regional count evidence, predictive uncertainty, and reliability weights."""
        regional = self.region_head((p4, p8, p16), regions)

        mu_count = regional["mu_count"]
        dispersion = regional["dispersion"]

        hurdle_pi_for_rel = None
        if self.cfg.hurdle_head and "hurdle_logit" in regional:
            hurdle_pi_for_rel = torch.sigmoid(regional["hurdle_logit"])

        reliability = reliability_from_nb(
            mu_count,
            dispersion,
            regions,
            mode=self.cfg.reliability_mode,
            hurdle_pi=hurdle_pi_for_rel,
            rate_std_floor=self.cfg.reliability_rate_std_floor,
            weight_min=self.cfg.reliability_weight_min,
            weight_max=self.cfg.reliability_weight_max,
            normalize_within_scale=self.cfg.normalize_reliability_within_scale,
            eps=self.cfg.eps,
        )

        weight = reliability["weight"]
        weight_solver = torch.ones_like(weight) if uniform_reliability else weight

        b_solver_raw = mu_count.detach() if self.cfg.detach_region_mean_in_solver else mu_count
        b_variance = reliability["count_variance"]

        if self.cfg.hurdle_head and "hurdle_logit" in regional:
            pi_r = torch.sigmoid(regional["hurdle_logit"].detach())
            b_solver = pi_r * b_solver_raw
            b_variance = pi_r.square() * b_variance
        else:
            b_solver = b_solver_raw

        if self.cfg.detach_reliability_in_solver:
            weight_solver = weight_solver.detach()

        # Pre-Solver Scale-Consistency Reliability Gating (RMR-v15/v16/v18)
        if self.cfg.pre_solver_scale_gating and scale_weights is not None:
            power = float(self.cfg.scale_gating_power)
            weight_solver = apply_scale_consistency_gating(
                weight_solver,
                regions,
                scale_weights,
                power=power,
                eps=self.cfg.eps,
                perspective_horizon_gate=self.cfg.perspective_horizon_gate,
                horizon_cutoff=float(self.cfg.horizon_cutoff),
                region_sizes_px=self.cfg.region_sizes_px,
                grid_h=grid_h,
            )

        return {
            "mu_count": mu_count,
            "dispersion": dispersion,
            "log_dispersion": regional["log_dispersion"],
            "rate": regional["rate"],
            "b_solver": b_solver,
            "b_variance": b_variance,
            "weight": weight,
            "weight_solver": weight_solver,
            "precision": reliability["precision"],
            "rate_variance": reliability["rate_variance"],
            "count_variance": reliability["count_variance"],
            "hurdle_logit": regional.get("hurdle_logit", None),
        }

    def _solve_inverse_measure(
        self,
        y0: torch.Tensor,
        z0: torch.Tensor,
        regional_evidence: dict[str, Any],
        regions: RegionSet,
        scale_weights: torch.Tensor | None,
        solver_strength: float | None,
        uniform_reliability: bool,
        p16: torch.Tensor,
        fg_logit: torch.Tensor | None,
        pi_scale: torch.Tensor | None,
        pi_aspect: torch.Tensor | None,
    ) -> RMRModelOutput:
        """Execute unrolled SIRT inverse measure reconstruction or return pre-solver baseline."""
        mu_count = regional_evidence["mu_count"]
        dispersion = regional_evidence["dispersion"]
        b_solver = regional_evidence["b_solver"]
        b_variance = regional_evidence["b_variance"]
        weight = regional_evidence["weight"]
        weight_solver = regional_evidence["weight_solver"]
        hurdle_logit = regional_evidence["hurdle_logit"]

        base_out: dict[str, Any] = {
            "y0": y0,
            "z0": z0,
            "regions": regions,
            "b_region": mu_count,
            "b_solver": b_solver,
            "region_rate": regional_evidence["rate"],
            "region_dispersion": dispersion,
            "region_log_dispersion": regional_evidence["log_dispersion"],
            "region_weight": weight,
            "solver_region_weight": weight_solver,
            "region_precision": regional_evidence["precision"],
            "region_rate_variance": regional_evidence["rate_variance"],
            "region_count_variance": regional_evidence["count_variance"],
            "solver_count_variance": b_variance,
            "uniform_reliability": uniform_reliability,
        }

        if scale_weights is not None:
            base_out["scale_weights"] = scale_weights
        if pi_scale is not None:
            base_out["pi_scale"] = pi_scale
        if pi_aspect is not None:
            base_out["pi_aspect"] = pi_aspect
        if hurdle_logit is not None:
            base_out["hurdle_logit"] = hurdle_logit
        if fg_logit is not None:
            base_out["fg_logit"] = fg_logit

        # Bypass solver loop if solver is disabled
        if not self.cfg.enable_solver:
            base_out.update({
                "y": y0,
                "iterates": [y0],
                "residual_fields": [],
                "energy_trace": [],
                "solver_strength": 0.0,
            })
            return RMRModelOutput(**base_out)

        effective_strength = self.solver_strength if solver_strength is None else solver_strength

        solver_res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=self.cfg.iterations,
            omega=self.cfg.omega,
            solver_strength=effective_strength,
            residual_clip=self.cfg.residual_clip,
            eps=self.cfg.eps,
            solver_mode=self.cfg.solver_mode,
            density_gate_rho=self.cfg.density_gate_rho,
            density_gate_floor=self.cfg.density_gate_floor,
            proximal_tau=self.cfg.proximal_tau,
            proximal_mode=self.cfg.proximal_mode,
            proximal_mu=self.cfg.proximal_mu,
            tv_lambda=self.cfg.tv_lambda,
            tv_type=self.cfg.tv_type,
            tv_eps_c=self.cfg.tv_eps_c,
            laplace_kernel=self._laplace_kernel,
            scale_routing_weights=scale_weights,
            trust_region_kappa=self.cfg.trust_region_kappa,
            trust_region_floor=self.cfg.trust_region_floor,
            adjoint_mode=self.cfg.adjoint_mode,
            b_variance=b_variance,
            morozov_gamma=self.cfg.morozov_gamma,
            use_barzilai_borwein=self.cfg.use_barzilai_borwein,
            use_alternating_bb=self.cfg.use_alternating_bb,
            bb_clamp_min=self.cfg.bb_clamp_min,
            bb_clamp_max=self.cfg.bb_clamp_max,
            use_scale_entropy_trust=self.cfg.use_scale_entropy_trust,
            use_nesterov_momentum=self.cfg.use_nesterov_momentum,
            adaptive_relaxation=self.cfg.adaptive_relaxation,
            adaptive_relax_sparse=self.cfg.adaptive_relax_sparse,
            adaptive_relax_dense_boost=self.cfg.adaptive_relax_dense_boost,
            adaptive_relax_threshold=self.cfg.adaptive_relax_threshold,
            adaptive_relax_scale=self.cfg.adaptive_relax_scale,
            hybrid_recovery_alpha=self.cfg.hybrid_recovery_alpha,
            density_gated_diffusion=self.cfg.density_gated_diffusion,
            diffusion_dense_threshold=self.cfg.diffusion_dense_threshold,
            diffusion_gate_beta=self.cfg.diffusion_gate_beta,
        )

        y = solver_res["y"]
        strength = solver_res["effective_omega"] / max(self.cfg.omega, 1e-8)

        # Convex Dynamic Trust Gate (RMR-v15)
        solver_trust_alpha = None
        if self.trust_gate is not None:
            feat_global = p16.mean(dim=(-2, -1))
            solver_trust_alpha = torch.sigmoid(self.trust_gate(feat_global)).view(-1, 1, 1, 1)
            y = (1.0 - solver_trust_alpha) * y0 + solver_trust_alpha * y
            base_out["solver_trust_alpha"] = solver_trust_alpha

        base_out.update({
            "y": y,
            "iterates": solver_res["iterates"],
            "residual_fields": solver_res["residual_fields"],
            "energy_trace": solver_res["energy_trace"],
            "solver_strength": strength,
        })
        return RMRModelOutput(**base_out)

    def forward(
        self,
        x: torch.Tensor,
        *,
        uniform_reliability: bool = False,
        solver_strength: float | None = None,
    ) -> RMRModelOutput:
        """Forward pass orchestrating carrier extraction, scale routing, regional evidence, and inverse solving."""
        p4, p8, p16 = self._extract_carrier_features(x)
        scale_weights, pi_scale, pi_aspect = self._route_scales(p4)
        z0, y0, fg_logit = self._predict_fine_density(p4, scale_weights)

        h, w = y0.shape[-2:]
        regions = self._regions(h, w, x.device)

        regional_evidence = self._extract_regional_evidence(
            p4=p4,
            p8=p8,
            p16=p16,
            regions=regions,
            scale_weights=scale_weights,
            uniform_reliability=uniform_reliability,
            grid_h=h,
        )

        return self._solve_inverse_measure(
            y0=y0,
            z0=z0,
            regional_evidence=regional_evidence,
            regions=regions,
            scale_weights=scale_weights,
            solver_strength=solver_strength,
            uniform_reliability=uniform_reliability,
            p16=p16,
            fg_logit=fg_logit,
            pi_scale=pi_scale,
            pi_aspect=pi_aspect,
        )

    def switch_to_deploy(self) -> None:
        """Switch internal modules (e.g. RepWeightedFPNNeck) to fused deployment mode."""
        if hasattr(self.fusion, "switch_to_deploy"):
            self.fusion.switch_to_deploy()


