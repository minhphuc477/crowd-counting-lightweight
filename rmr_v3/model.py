from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import FineMeasureHead
from rmr_core.necks import AdditiveFPNNeck, ASPPLiteFPNNeck, CoordinateAttention, RepWeightedFPNNeck
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    region_mean_std_features,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from .regional_head import ProbabilisticRegionalEvidenceHead, reliability_from_nb
from .solver import unrolled_sirt_solver


def _softplus_inverse(y: float) -> float:
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))


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

    # Reliability
    reliability_mode: str = "nb_rate_variance"
    reliability_rate_std_floor: float = 0.01
    reliability_weight_min: float = 0.25
    reliability_weight_max: float = 4.0
    normalize_reliability_within_scale: bool = True

    # Registered clean-causal variant
    detach_region_mean_in_solver: bool = True
    detach_reliability_in_solver: bool = True

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

    def __post_init__(self) -> None:
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
    "ProbabilisticRegionalEvidenceHead",
    "reliability_from_nb",
    "region_mean_std_features",
    "weighted_coverage",
    "weighted_normalized_adjoint_field",
    "weighted_regional_energy",
]


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

        if cfg.iterations < 1:
            raise ValueError("iterations must be >= 1")

        if cfg.omega <= 0:
            raise ValueError("omega must be > 0")

        if cfg.reliability_mode != "nb_rate_variance":
            raise ValueError(
                f"Unsupported reliability_mode: {cfg.reliability_mode}. Only 'nb_rate_variance' is supported."
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

        self.fine_head = FineMeasureHead(
            width=cfg.feature_width,
            init_bias=init_bias,
            temp_softplus=cfg.temp_softplus,
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

        # ── Dynamic Scale Routing (RMR-v10) ──────────────────────────────────
        if cfg.dynamic_scale_routing:
            self.scale_router: ScaleRoutingHead | None = ScaleRoutingHead(
                in_channels=cfg.feature_width,
                num_scales=len(cfg.region_sizes_px),
                temperature=cfg.scale_router_temperature,
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
            self.cfg.region_sizes_px,
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

    def forward(
        self,
        x: torch.Tensor,
        *,
        uniform_reliability: bool = False,
        solver_strength: float | None = None,
    ) -> dict:
        c4, c8, c16 = self.encoder(x)

        p4, p8, p16 = self.fusion(
            c4,
            c8,
            c16,
        )

        # ── Stage 3: Coordinate Attention on P4 (optional) ────────────────────
        if self.coord_attn is not None:
            p4 = self.coord_attn(p4)

        z0 = self.fine_head.forward_logits(p4)
        y0 = self.fine_head.activate(z0)

        # ── Decoupled Foreground Gating (RMR-v11) ────────────────────────────
        fg_logit = None
        if self.fg_gate is not None:
            fg_logit = self.fg_gate(p4)
            # Residual safety floor (0.70 + 0.30 * sigmoid): guarantees gate is strictly in [0.70, 1.0],
            # completely preventing zero-absorbing barriers and false-negative head erasure.
            fg_mask = 0.70 + 0.30 * torch.sigmoid(fg_logit)
            y0 = y0 * fg_mask

        h, w = y0.shape[-2:]

        regions = self._regions(
            h,
            w,
            x.device,
        )

        regional = self.region_head(
            (p4, p8, p16),
            regions,
        )

        mu_count = regional["mu_count"]
        dispersion = regional["dispersion"]

        reliability = reliability_from_nb(
            mu_count,
            dispersion,
            regions,
            rate_std_floor=self.cfg.reliability_rate_std_floor,
            weight_min=self.cfg.reliability_weight_min,
            weight_max=self.cfg.reliability_weight_max,
            normalize_within_scale=(
                self.cfg.normalize_reliability_within_scale
            ),
            eps=self.cfg.eps,
        )

        weight = reliability["weight"]

        # V3-A control: probabilistic head, uniform solver.
        if uniform_reliability:
            weight_solver = torch.ones_like(weight)
        else:
            weight_solver = weight

        # ── Hurdle head: modulate solver target by occupancy probability ──────
        # b_solver_raw is the raw regional NB mean (before hurdle masking).
        b_solver_raw = mu_count.detach() if self.cfg.detach_region_mean_in_solver else mu_count

        if self.cfg.hurdle_head and "hurdle_logit" in regional:
            # π_R = sigmoid(z_π_R): probability region is occupied.
            # b_solver = π_R * mu_count  — background regions approach 0 count target.
            pi_r = torch.sigmoid(regional["hurdle_logit"].detach())
            b_solver = pi_r * b_solver_raw
        else:
            b_solver = b_solver_raw

        if self.cfg.detach_reliability_in_solver:
            weight_solver = weight_solver.detach()

        # Collect hurdle logit for loss computation (not detached)
        hurdle_logit = regional.get("hurdle_logit", None)  # [B,1,M] or None

        # ── Dynamic Scale Routing (RMR-v10) ──────────────────────────────────
        scale_weights = None
        if self.scale_router is not None:
            scale_weights = self.scale_router(p4)

        # Bypass solver loop if solver is disabled (direct baseline mode)
        if not self.cfg.enable_solver:
            out = {
                "y": y0,
                "y0": y0,
                "z0": z0,
                "regions": regions,
                "b_region": mu_count,
                "b_solver": b_solver,
                "region_rate": regional["rate"],
                "region_dispersion": dispersion,
                "region_log_dispersion": regional["log_dispersion"],
                "region_weight": weight,
                "solver_region_weight": weight_solver,
                "region_precision": reliability["precision"],
                "region_rate_variance": reliability["rate_variance"],
                "region_count_variance": reliability["count_variance"],
                "iterates": [y0],
                "residual_fields": [],
                "energy_trace": [],
                "uniform_reliability": uniform_reliability,
                "solver_strength": 0.0,
            }
            if scale_weights is not None:
                out["scale_weights"] = scale_weights
            if hurdle_logit is not None:
                out["hurdle_logit"] = hurdle_logit
            if fg_logit is not None:
                out["fg_logit"] = fg_logit
            return out

        solver_res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=self.cfg.iterations,
            omega=self.cfg.omega,
            solver_strength=self.solver_strength if solver_strength is None else solver_strength,
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
        )

        y = solver_res["y"]
        iterates = solver_res["iterates"]
        residual_fields = solver_res["residual_fields"]
        energy_trace = solver_res["energy_trace"]
        strength = solver_res["effective_omega"] / max(self.cfg.omega, 1e-8)

        out = {
            "y": y,
            "y0": y0,
            "z0": z0,

            "regions": regions,

            "b_region": mu_count,
            "b_solver": b_solver,
            "region_rate": regional["rate"],
            "region_dispersion": dispersion,
            "region_log_dispersion": regional["log_dispersion"],

            "region_weight": weight,
            "solver_region_weight": weight_solver,
            "region_precision": reliability["precision"],
            "region_rate_variance": reliability["rate_variance"],
            "region_count_variance": reliability["count_variance"],

            "iterates": iterates,
            "residual_fields": residual_fields,
            "energy_trace": energy_trace,

            "uniform_reliability": uniform_reliability,
            "solver_strength": strength,
        }

        if scale_weights is not None:
            out["scale_weights"] = scale_weights
        if hurdle_logit is not None:
            out["hurdle_logit"] = hurdle_logit
        if fg_logit is not None:
            out["fg_logit"] = fg_logit
        return out

    def switch_to_deploy(self) -> None:
        """Switch internal modules (e.g. RepWeightedFPNNeck) to fused deployment mode."""
        if hasattr(self.fusion, "switch_to_deploy"):
            self.fusion.switch_to_deploy()


