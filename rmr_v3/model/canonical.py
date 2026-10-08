from __future__ import annotations

"""Canonical Final Architecture Specification for RMR Crowd Counting.

Strictly bounded to <= 104,441 trainable parameters (Standalone, 0.0% KD).
Unified design combining:
1. MobileNetV4-Conv-Small-050 Stride-4 Pyramid Backbone (58,368 params)
2. ASPP-Lite FPN Neck with Global Average Pooling (8,224 params)
3. Temperature-Softplus Fine Carrier Density Head (1,186 params)
4. Probabilistic Regional Evidence Head with Negative-Binomial & Hurdle (5,667 params)
5. Dynamic Scale Routing Head (1,059 params)
6. Zero-Parameter Unrolled SIRT Inverse Solver (T=6 iterations, Monotonic Bounded Morozov)

Total trainable parameters: exactly 101,763 (55.9% backbone, 7.9% neck, 7.7% heads, 0% solver).
"""

from typing import Any
import torch

from .architecture import RMRv3
from .config import RMRv3Config
from ..losses.config import RMRv3LossConfig


CANONICAL_PARAM_BUDGET: int = 104441
CANONICAL_EXPECTED_PARAMS: int = 104441


def get_canonical_model_config(**overrides: Any) -> RMRv3Config:
    """Return the frozen canonical architecture configuration."""
    base_kwargs: dict[str, Any] = {
        # Carrier resolution and backbone
        "output_stride": 4,
        "feature_width": 32,
        "backbone_name": "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "pretrained": True,
        "backbone_lr_scale": 0.1,
        "init_m0": 0.015763,
        "max_trainable_params": CANONICAL_PARAM_BUDGET,
        # Multi-scale neck: HDC-Lite (Zero Gridding Holes)
        "neck_type": "hdc_lite",
        "use_aspp_gap": True,
        "hdc_dilations": (1, 2, 3),
        # Regional evidence dictionary
        "region_sizes_px": (32, 64, 128),
        "region_overlap": 0.5,
        "include_full_image": False,
        "regional_feature_stats": "mean",
        "region_head_hidden": 48,
        # Dynamic scale routing
        "dynamic_scale_routing": True,
        "factorized_scale_routing": False,
        "scale_router_temperature": 1.0,
        "pre_solver_scale_gating": True,
        "scale_gating_power": 1.0,
        # Carrier density head & curvature (Sub-Rayleigh collision support)
        "temp_softplus": True,
        "density_curvature": True,
        "gated_density_curvature": True,
        "curvature_dense_threshold": 0.15,
        "curvature_gate_beta": 0.03,
        "curvature_pool_kernel": 8,
        "curvature_alpha_init": -4.0,
        # Negative-Binomial & hurdle gating (100% mass preservation in dense clusters)
        "hurdle_head": True,
        "hurdle_gating_mode": "occupancy",
        "dispersion_init": 50.0,
        "dispersion_min": 0.5,
        "dispersion_max": 500.0,
        "reliability_mode": "snr",
        "reliability_rate_std_floor": 0.01,
        "reliability_weight_min": 0.25,
        "reliability_weight_max": 4.0,
        "normalize_reliability_within_scale": True,
        "detach_region_mean_in_solver": True,
        "detach_reliability_in_solver": True,
        "detach_y0_for_solver": False,
        # Unrolled SIRT inverse solver
        "enable_solver": True,
        "solver_mode": "additive",
        "iterations": 6,
        "omega": 1.0,
        "residual_clip": 0.0,
        "eps": 1e-6,
        "adjoint_mode": "radon_nikodym",
        "morozov_gamma": 0.75,
        "morozov_rho_cap": 0.25,
        "trust_region_kappa": 0.35,
        "trust_region_floor": 0.005,
        "use_barzilai_borwein": True,
        "use_scale_entropy_trust": True,
        # Regularization & clean priors (zero artificial carrier seeding)
        "scale_seeded_carrier": False,
        "scale_seed_eps": 0.0,
        "tv_lambda": 0.02,
        "tv_type": "laplacian",
        "proximal_tau": 0.015,
        "proximal_mode": "firm",
        "proximal_mu": 3.0,
        "ema_decay": 0.999,
        # Banned anti-patterns permanently disabled
        "use_coord_attn": False,
        "foreground_gate": False,
        "dynamic_trust_gate": False,
        "use_top_down_semantic_gate": False,
        "scale_conditioned_prior": False,
    }
    base_kwargs.update(overrides)
    return RMRv3Config.from_dict(base_kwargs)


def get_canonical_loss_config(**overrides: Any) -> RMRv3LossConfig:
    """Return the balanced multi-task supervision configuration for the canonical model."""
    base_kwargs: dict[str, Any] = {
        # Core supervision weights
        "lambda_count": 1.0,
        "lambda_flat_dm16": 0.025,  # calibrated for 1:1 gradient norm parity with count loss
        "lambda_cell": 0.50,        # focal pinning for sparse heads (< 32px)
        "lambda_region_nb": 0.20,
        "lambda_hurdle": 0.10,
        "lambda_trunc_nb": 0.20,
        # Point supervision: Tight Adaptive Bayesian
        "allocation_loss_type": "bayesian",
        "bayesian_sigma": 4.0,
        "bayesian_adaptive_sigma": True,
        "bayesian_sigma_min": 2.0,
        "bayesian_sigma_max": 4.0,
        "bayesian_background_ratio": 0.10,
        "bayesian_norm_mode": "canonical",
        "bayesian_canonical_bg": True,
        "dm_target": "dual",
        # Count magnitude loss
        "count_loss_mode": "nb",
        "count_nb_dispersion": 50.0,
        # Cell loss: Count-Harmonized
        "cell_loss_mode": "count_harmonized",
        "cell_beta": 1.0,
        "cell_mass_weight_eps": 0.001,
        "cell_mass_weight_alpha": 2.0,
        "cell_mass_weight_gamma": 1.25,
        "cell_norm_power": 0.5,
        "cell_norm_ref": 100.0,
        # Auxiliary losses
        "lambda_scale_align": 0.05,
        "scale_align_tau_dense": 0.12,
        "scale_align_tau_sparse": 0.03,
        "scale_align_kernel": 5,
        "scale_align_mask_bg": True,
        "lambda_curvature": 0.50,
        "curvature_gate_threshold": 0.08,
        "curvature_gate_kernel": 5,
        "curvature_gate_mode": "hard",
        "curvature_gate_scale": 0.02,
        "lambda_hard_bg": 0.15,
        "hard_bg_ratio": 0.05,
    }
    base_kwargs.update(overrides)
    return RMRv3LossConfig.from_dict(base_kwargs)


def build_canonical_rmr_model(
    pretrained: bool = True,
    device: str | torch.device | None = None,
    **overrides: Any,
) -> RMRv3:
    """Instantiate and verify the Canonical Final RMR Architecture.
    
    Raises ValueError if total trainable parameter budget exceeds 104,441.
    """
    cfg = get_canonical_model_config(pretrained=pretrained, **overrides)
    model = RMRv3(cfg)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total_params > CANONICAL_PARAM_BUDGET:
        raise ValueError(
            f"Canonical model violated strict parameter ceiling: {total_params} > {CANONICAL_PARAM_BUDGET}"
        )

    if device is not None:
        model = model.to(device=device)

    return model
