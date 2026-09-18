"""Shared pytest fixtures, mathematical oracles, and test configuration for RMR-v26 E2E test suite."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import RegionSet, build_multiscale_regions
from rmr_v3.config import load_config
from rmr_v3.model import MicroPerspectiveElevation, RMRv3, RMRv3Config


def mpe_v2_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Authoritative reference implementation of MPE-v2 normalized elevation modulation.

    Mathematical Definition:
        M(v) = 1.0 + tanh(W v + b)
        M_bar = (1 / H) * sum_{i=1}^H M(v_i)
        P4_tilde(v) = P4(v) * (M(v) / M_bar)
    Guarantee:
        (1/H) * sum_{i=1}^H (P4_tilde(v_i) / P4(v_i)) == 1.000000 +- 1e-6
    """
    h = x.shape[-2]
    v = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=torch.float32).view(h, 1)
    elevation_mod = F.linear(v, weight, bias).transpose(0, 1).unsqueeze(-1)
    if x.ndim == 4:
        elevation_mod = elevation_mod.unsqueeze(0)
    m = 1.0 + torch.tanh(elevation_mod).to(dtype=x.dtype)
    m_bar = m.mean(dim=-2, keepdim=True)
    return x * (m / m_bar)


def pure_bb1_oracle(
    s_diff: torch.Tensor,
    r_diff: torch.Tensor,
    omega_0: float = 1.0,
    clamp_min: float = 0.5,
    clamp_max: float = 1.2,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Authoritative reference implementation of pure BB-1 Rayleigh contraction step size.

    Formula:
        alpha_1 = <s_k, r_k> / (||r_k||^2 + eps)
        omega_clamped = clamp(alpha_1, min=clamp_min * omega_0, max=clamp_max * omega_0)
    """
    dot_sr = (s_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True)
    norm_r_sq = (r_diff * r_diff).sum(dim=(-3, -2, -1), keepdim=True) + eps
    raw_step = dot_sr / norm_r_sq
    # If dot_sr <= 0 (non-convex descent), fall back to omega_0
    alpha_1 = torch.where(
        dot_sr > 0.0,
        raw_step,
        torch.as_tensor(omega_0, device=dot_sr.device, dtype=dot_sr.dtype),
    )
    return torch.clamp(alpha_1, min=clamp_min * omega_0, max=clamp_max * omega_0)


def get_rmr_v26_config_spec(variant: str = "canonical") -> Dict[str, Any]:
    """Generates an in-memory dictionary for any of the 6 RMR-v26 model configurations.

    Variants:
        - canonical: MPE-v2 + Pure BB-1 + Morozov 0.75 + lambda_curv = 0.0
        - ablation_no_elevation: use_perspective_elevation = False
        - ablation_no_bb: use_barzilai_borwein = False (fixed omega = 1.0)
        - ablation_no_morozov: morozov_gamma = 0.0
        - ablation_with_curv01: lambda_curvature = 0.10
        - control_no_solver: enable_solver = False (feedforward Y_0)
    """
    base_raw = load_config(Path("configs/rmr_v25/rmr_v25_canonical.yaml"))
    model_cfg = copy.deepcopy(base_raw["model"])
    loss_cfg = copy.deepcopy(base_raw["loss"])

    # v26 Canonical baseline specifications
    loss_cfg["lambda_curvature"] = 0.0
    model_cfg["use_barzilai_borwein"] = True
    model_cfg["use_alternating_bb"] = False
    model_cfg["use_perspective_elevation"] = True
    model_cfg["bb_clamp_min"] = 0.5
    model_cfg["bb_clamp_max"] = 1.2
    model_cfg["morozov_gamma"] = 0.75
    model_cfg["enable_solver"] = True

    if variant == "canonical":
        pass
    elif variant == "ablation_no_elevation":
        model_cfg["use_perspective_elevation"] = False
    elif variant == "ablation_no_bb":
        model_cfg["use_barzilai_borwein"] = False
    elif variant == "ablation_no_morozov":
        model_cfg["morozov_gamma"] = 0.0
    elif variant == "ablation_with_curv01":
        loss_cfg["lambda_curvature"] = 0.10
    elif variant == "control_no_solver":
        model_cfg["enable_solver"] = False
    else:
        raise ValueError(f"Unknown RMR-v26 variant: {variant}")

    return {
        "model": model_cfg,
        "loss": loss_cfg,
    }


@pytest.fixture
def canonical_v26_config_dict() -> Dict[str, Any]:
    """Fixture returning the canonical RMR-v26 configuration dictionary."""
    return get_rmr_v26_config_spec("canonical")


@pytest.fixture
def synthetic_p4_carrier() -> torch.Tensor:
    """Fixture returning a synthetic P4 carrier tensor [B=2, C=32, H=64, W=64]."""
    torch.manual_seed(42)
    return torch.randn(2, 32, 64, 64)


@pytest.fixture
def standard_multiscale_regions() -> RegionSet:
    """Fixture returning standard multiscale regions [32, 64, 128] px for H=64, W=64 at stride 4."""
    return build_multiscale_regions(
        height=64,
        width=64,
        output_stride=4,
        region_sizes_px=(32, 64, 128),
        include_full_image=False,
    )


@pytest.fixture
def acceptance_thresholds() -> Dict[str, float]:
    """Fixture providing the authoritative acceptance criteria thresholds from ORIGINAL_REQUEST.md."""
    return {
        "max_trainable_params": 105_000,
        "target_canonical_params": 104_505,
        "target_no_elevation_params": 104_441,
        "max_overall_mae": 70.0,
        "max_sparse_mae": 18.0,
        "max_dense_mae": 125.0,
        "bias_min": -5.0,
        "bias_max": 2.0,
        "mass_conservation_atol": 1e-6,
        "bb_clamp_min": 0.5,
        "bb_clamp_max": 1.2,
    }
