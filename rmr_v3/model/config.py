from __future__ import annotations

from dataclasses import dataclass, fields
import math
import warnings
from typing import Any


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
    max_trainable_params: int = 105000

    # RMR-v29: Sub-pixel Stride-2 Reconstruction Head (+99 params)
    subpixel_stride2: bool = False

    # Neck
    neck_type: str = "additive"  # "additive" | "aspp_lite" | "rep_weighted"
    context_dilations: tuple[int, ...] = (1, 2, 3)
    use_aspp_gap: bool = True
    aspp_dilations: tuple[int, ...] = (1, 3, 6)

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
    region_head_hidden: int = 48

    # Hurdle NB head
    hurdle_head: bool = False
    tv_lambda: float = 0.0
    ema_decay: float = 0.0
    temp_softplus: bool = False

    # Solver update rule
    solver_mode: str = "additive"
    density_gate_rho: float = 0.02
    density_gate_floor: float = 0.02
    tv_type: str = "laplacian"
    tv_eps_c: float = 0.1

    # Banned coordinate attention (retained for validator rejection)
    use_coord_attn: bool = False
    use_micro_coord_attn: bool = False
    micro_coord_reduction: int = 8

    # Proximal L1-Soft-Thresholding
    proximal_tau: float = 0.0
    proximal_mode: str = "soft"
    proximal_mu: float = 3.0

    # RMR-v30: Anscombe Variance-Stabilizing SIRT & Density-Conditioned Spatial Resolution
    use_anscombe_sirt: bool = False
    anscombe_c: float = 0.375
    adaptive_tau: bool = False
    adaptive_tau_rho0: float = 0.05

    # RMR-v31: Sub-50 MAE Dual-Lattice, Density-Gated Anscombe & Anisotropic SIRT
    anisotropic_diffusion: bool = False
    pm_kappa: float = 0.05
    density_gated_anscombe: bool = False
    anscombe_tau_dense: float = 0.08
    area_normalized_adjoint: bool = False

    # RMR-v32: Continuous Perspective Carrier Modulation & Non-Saturating Floor Subtraction
    use_cpcm: bool = False
    cpcm_hidden: int = 8
    floor_tau: float = 0.0

    # RMR-v33 (Hypothesis H7/H8): Resonant Carrier Adjoint & Anscombe Morozov (0 params)
    resonant_adjoint: bool = False
    resonant_adjoint_lambda: float = 0.5
    anscombe_morozov: bool = False
    crest_discovery_flux: bool = False
    crest_kappa_0: float = 2.0
    crest_eps_seed: float = 0.005
    asymmetric_morozov: bool = False
    morozov_gamma_under: float = 0.20
    morozov_rho: float = 0.30


    # Perspective-Adaptive Regional Kernels (PARK - RMR-v33)
    use_park: bool = False
    park_mode: str = "pcat"
    park_horizon_h: int = 16
    park_foreground_h: int = 128
    park_max_aspect: float = 2.0
    use_pgh: bool = False
    park_routing: bool = False
    park_altitude_bands: int = 3

    # Dynamic Scale Routing & DiAG (RMR-v34)
    use_diag: bool = False
    diag_persp_slope_init: str = "physical"
    use_vertical_gradient_dcap: bool = False
    use_dcap_tilt: bool = True
    dynamic_scale_routing: bool = False
    scale_router_temperature: float = 1.0

    # Trust-region and gating
    trust_region_kappa: float = 0.0
    trust_region_floor: float = 0.005
    foreground_gate: bool = False
    fg_gate_floor: float = 0.70
    adjoint_mode: str = "flat"
    morozov_gamma: float = 0.0
    use_top_down_semantic_gate: bool = False
    tdsg_floor: float = 0.20

    # Dynamic geometry & trust gate
    scale_conditioned_prior: bool = False
    pre_solver_scale_gating: bool = False
    scale_gating_power: float = 1.0
    dynamic_trust_gate: bool = False
    trust_gate_init_bias: float = 1.73

    # Curvature warping & BB solver
    density_curvature: bool = False
    use_barzilai_borwein: bool = False
    use_scale_entropy_trust: bool = False
    perspective_scale_bias: bool = False
    perspective_horizon_gate: bool = False
    horizon_cutoff: float = 0.35

    # Decoupled 2D routing & density-gated curvature
    factorized_scale_routing: bool = False
    num_marginal_scales: int = 3
    num_aspect_ratios: int = 2
    gated_density_curvature: bool = False
    curvature_dense_threshold: float = 0.15
    curvature_gate_beta: float = 0.03
    curvature_pool_kernel: int = 8
    curvature_alpha_init: float = -8.0

    # Banned solver anti-patterns (retained for validator rejection)
    use_nesterov_momentum: bool = False
    adaptive_relaxation: bool = False
    adaptive_relax_sparse: float = 0.70
    adaptive_relax_dense_boost: float = 0.50
    adaptive_relax_threshold: float = 0.05
    adaptive_relax_scale: float = 0.02
    hybrid_recovery_alpha: float = 0.0
    scale_conditioned_fine_head: bool = False
    density_gated_diffusion: bool = False
    diffusion_dense_threshold: float = 0.15
    diffusion_gate_beta: float = 0.03
    use_perspective_elevation: bool = False
    use_alternating_bb: bool = False

    # BB clamp & cyclic BB (v19/v27 restored baseline)
    bb_clamp_min: float = 0.2
    bb_clamp_max: float = 2.0
    cyclic_bb_length: int = 1

    def __post_init__(self) -> None:
        if self.subpixel_stride2 and self.output_stride != 2:
            warnings.warn(
                f"subpixel_stride2=True requires output_stride=2; "
                f"overriding output_stride={self.output_stride} → 2. "
                "Set output_stride: 2 in your YAML to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )
            self.output_stride = 2
        if self.cyclic_bb_length < 1:
            raise ValueError(f"cyclic_bb_length must be >= 1, got {self.cyclic_bb_length}")
        if self.max_trainable_params < 0:
            raise ValueError(f"max_trainable_params must be >= 0, got {self.max_trainable_params}")
        if self.region_sizes_px is not None:
            self.region_sizes_px = _deep_tuple(self.region_sizes_px)
        if self.aspp_dilations is not None:
            self.aspp_dilations = _deep_tuple(self.aspp_dilations)
        if self.context_dilations is not None:
            self.context_dilations = _deep_tuple(self.context_dilations)

        if self.use_park:
            raise ValueError(
                "use_park=True is permanently BANNED (PARK static altitude bands cause spatial starvation "
                "and +32 Dense MAE degradation). Use dynamic DiAG instead."
            )
        if self.use_cpcm:
            raise ValueError(
                "use_cpcm=True is permanently BANNED (CPCM static linspace coordinate aliasing under crops). "
                "Use dynamic DiAG instead."
            )
        if self.use_perspective_elevation:
            raise ValueError(
                "use_perspective_elevation=True is permanently BANNED (MPE static linspace coordinate aliasing)."
            )
        if getattr(self, "use_pgh", False):
            raise ValueError(
                "use_pgh=True is permanently BANNED (PGH static linspace coordinate aliasing)."
            )

        # Permanently Banned Anti-Pattern Guards
        if self.use_coord_attn:
            raise ValueError(
                "use_coord_attn=True is permanently BANNED (Anti-Pattern #2: Rank-1 phantom Dirac spikes). "
                "Remove this field from your config."
            )
        if getattr(self, "use_micro_coord_attn", False):
            raise ValueError(
                "use_micro_coord_attn=True is permanently BANNED (Anti-Pattern #2: Coordinate Attention family). "
                "Remove this field from your config."
            )
        if self.use_nesterov_momentum:
            raise ValueError(
                "use_nesterov_momentum=True is permanently BANNED (Anti-Pattern #5: kinetic overshoot in T=6 solver). "
                "Remove this field from your config."
            )
        if self.use_alternating_bb:
            raise ValueError(
                "use_alternating_bb=True is permanently BANNED (Anti-Pattern #7: exploding variance in dense clumps). "
                "Remove this field from your config. Use pure BB-1 with trust damping instead."
            )
        if self.density_gated_diffusion:
            raise ValueError(
                "density_gated_diffusion=True is permanently BANNED (Anti-Pattern #8: blurs Dirac peaks at stride 4). "
                "Remove this field from your config."
            )
        if self.use_top_down_semantic_gate:
            raise ValueError(
                "use_top_down_semantic_gate=True is permanently BANNED (Anti-Pattern #4: 90% gradient suppression). "
                "Remove this field from your config."
            )
        if self.foreground_gate:
            raise ValueError(
                "foreground_gate=True is permanently BANNED (Anti-Pattern #4: Hard FG-Gate gradient suppression). "
                "Remove this field from your config."
            )
        if self.hybrid_recovery_alpha > 0.0:
            raise ValueError(
                f"hybrid_recovery_alpha={self.hybrid_recovery_alpha} > 0 is permanently BANNED "
                "(Anti-Pattern #6: Lebesgue Discovery Flux causes mass leakage). "
                "Keep hybrid_recovery_alpha=0.0."
            )
        if self.solver_mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"solver_mode must be 'additive' or 'multiplicative', got '{self.solver_mode}'"
            )
        if self.tv_type not in ("laplacian", "charbonnier", "perona_malik"):
            raise ValueError(
                f"tv_type must be 'laplacian', 'charbonnier', or 'perona_malik', got '{self.tv_type}'"
            )
        if self.pm_kappa <= 0.0:
            raise ValueError(
                f"pm_kappa must be strictly positive, got {self.pm_kappa}"
            )
        if self.anscombe_tau_dense <= 0.0:
            raise ValueError(
                f"anscombe_tau_dense must be strictly positive, got {self.anscombe_tau_dense}"
            )
        if self.proximal_tau < 0.0:
            raise ValueError(
                f"proximal_tau must be non-negative, got {self.proximal_tau}"
            )
        if self.floor_tau < 0.0:
            raise ValueError(
                f"floor_tau must be non-negative, got {self.floor_tau}"
            )
        if self.use_cpcm and self.cpcm_hidden <= 0:
            raise ValueError(
                f"cpcm_hidden must be positive when use_cpcm=True, got {self.cpcm_hidden}"
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
        if self.adjoint_mode not in ("flat", "radon_nikodym", "anscombe_vst"):
            raise ValueError(
                f"adjoint_mode must be 'flat', 'radon_nikodym', or 'anscombe_vst', got '{self.adjoint_mode}'"
            )
        if self.adjoint_mode == "anscombe_vst":
            self.use_anscombe_sirt = True
        if self.anscombe_c <= 0.0:
            raise ValueError(
                f"anscombe_c must be strictly positive, got {self.anscombe_c}"
            )
        if self.adaptive_tau_rho0 <= 0.0:
            raise ValueError(
                f"adaptive_tau_rho0 must be strictly positive, got {self.adaptive_tau_rho0}"
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
        if self.crest_kappa_0 < 0.0:
            raise ValueError(f"crest_kappa_0 must be non-negative, got {self.crest_kappa_0}")
        if self.crest_eps_seed < 0.0:
            raise ValueError(f"crest_eps_seed must be non-negative, got {self.crest_eps_seed}")
        if self.morozov_gamma_under < 0.0:
            raise ValueError(f"morozov_gamma_under must be non-negative, got {self.morozov_gamma_under}")
        if self.morozov_rho < 0.0:
            raise ValueError(f"morozov_rho must be non-negative, got {self.morozov_rho}")

    @classmethod
    def from_dict(cls, d: dict | None, **overrides) -> "RMRv3Config":
        """Cleanly instantiate RMRv3Config from dictionary with aliases and type coercion."""
        merged = dict(d or {})
        merged.update(overrides)

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
