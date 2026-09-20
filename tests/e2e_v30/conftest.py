"""Shared pytest fixtures, mathematical oracles, and test configuration for RMR-v30 E2E test suite.

Authoritative reference implementations directly derived from:
- RMR-v30 Mathematical Report (Theorems 1-4, explorer_m0_math/handoff.md)
- Evaluation & Benchmark Audit (explorer_m0_eval/handoff.md)
- Architectural Blueprint (explorer_m0_arch/handoff.md)
- Scope Specification (orchestrator_v30/SCOPE.md)
"""
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
from rmr_v3.model import RMRv3, RMRv3Config


# ============================================================================
# Mathematical Oracles for RMR-v30 Core Formulations
# ============================================================================

def anscombe_transform_oracle(y: torch.Tensor, c: float = 0.375) -> torch.Tensor:
    """Authoritative reference implementation of Anscombe Variance-Stabilizing Transformation.

    Mathematical Definition:
        T(y) = 2.0 * sqrt(max(0, y) + c)
    With c = 3/8 = 0.375, Poisson count error variance is asymptotically normalized:
        Var(T(Y)) = 1.0 + O(1/mu^2)
    """
    return 2.0 * torch.sqrt(torch.clamp_min(y.float(), 0.0) + float(c))


def inverse_anscombe_oracle(z: torch.Tensor, c: float = 0.375) -> torch.Tensor:
    """Authoritative reference implementation of algebraic inverse Anscombe transform.

    Mathematical Definition:
        T^{-1}(z) = max(0, (z^2 / 4) - c)
    """
    z32 = z.float()
    return torch.clamp_min((z32 * z32) / 4.0 - float(c), 0.0)


def anscombe_discrepancy_oracle(
    q: torch.Tensor,
    b: torch.Tensor,
    c: float = 0.375,
    b_variance: torch.Tensor | None = None,
    morozov_gamma: float = 0.0,
) -> torch.Tensor:
    """Authoritative reference implementation of Anscombe SIRT Rate Residual.

    Mathematical Definition:
        q_stab = max(0, q) + c
        b_stab = max(0, b) + c
        delta_tilde = 2.0 * (sqrt(q_stab) - sqrt(b_stab))
        r_m = delta_tilde / sqrt(q_stab) = 2.0 * (1.0 - sqrt(b_stab / q_stab))
    Guarantees O(1) gradient dynamics across all crowd densities (Theorem 3).
    """
    q32 = q.float()
    b32 = b.float()
    c_val = float(c)

    q_stab = torch.clamp_min(q32, 0.0) + c_val
    b_stab = torch.clamp_min(b32, 0.0) + c_val

    delta_tilde = 2.0 * (torch.sqrt(q_stab) - torch.sqrt(b_stab))

    if morozov_gamma > 0.0:
        if b_variance is not None:
            sigma_tilde = torch.sqrt(b_variance.float() / b_stab).clamp_min(0.1)
        else:
            sigma_tilde = torch.ones_like(delta_tilde)
        deadband = float(morozov_gamma) * sigma_tilde
        delta_tilde = torch.sign(delta_tilde) * torch.clamp_min(delta_tilde.abs() - deadband, 0.0)

    rate_res = delta_tilde / torch.sqrt(q_stab).clamp_min(1e-6)
    return rate_res


def push_forward_stride2_to_stride4_oracle(y_stride2: torch.Tensor) -> torch.Tensor:
    """Authoritative reference implementation of discrete Radon push-forward P_{2->4}.

    Mathematical Definition:
        [P_{2->4}(y_2)](k) = sum_{u in sub(k)} y_2(u) = 4.0 * AvgPool2d(y_2, k=2, s=2)
    Guarantees bitwise total mass conservation (Theorem 1):
        sum(P_{2->4}(y_2)) == sum(y_2)
    """
    return 4.0 * F.avg_pool2d(y_stride2.float(), kernel_size=2, stride=2, count_include_pad=False)


def prolongation_stride4_to_stride2_rn_oracle(
    y_stride4: torch.Tensor,
    y_stride2_prior: torch.Tensor,
    eps_cell: float = 1e-7,
) -> torch.Tensor:
    """Authoritative reference implementation of Radon-Nikodym prolongation Q_{4->2}^RN.

    Mathematical Definition:
        [Q_{4->2}^RN(y_4; y_2^0)](u) = y_4(k) * [y_2^0(u) / (sum_{v in sub(k)} y_2^0(v) + eps_cell)]
    With fallback to uniform (1/4)*y_4(k) when sum_{v in sub(k)} y_2^0(v) <= eps_cell.
    Strictly preserves carrier mass on every coarse cell k and background support (Theorems 1, 2).
    """
    y4 = y_stride4.float()
    y2_0 = y_stride2_prior.float()

    # Compute coarse mass of prior
    p4_prior = push_forward_stride2_to_stride4_oracle(y2_0)

    # Upsample coarse grids to fine stride 2 via 2x2 nearest neighbor replication
    y4_up = F.interpolate(y4, scale_factor=2.0, mode="nearest")
    p4_prior_up = F.interpolate(p4_prior, scale_factor=2.0, mode="nearest")

    # Modulate fine cells proportionally
    denom = p4_prior_up + float(eps_cell)
    ratio = y2_0 / denom
    out_prolonged = y4_up * ratio

    # Fallback to uniform 1/4 on zero-prior cells
    uniform_fallback = 0.25 * y4_up
    is_empty_prior = p4_prior_up <= float(eps_cell)
    return torch.where(is_empty_prior, uniform_fallback, out_prolonged)


def check_mass_conservation_oracle(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    eps: float = 1e-6,
) -> bool:
    """Verifies that total image measure is strictly conserved between fine and carrier lattices."""
    m_fine = y_fine.double().sum(dim=(-2, -1))
    m_carrier = y_carrier.double().sum(dim=(-2, -1))
    max_diff = (m_fine - m_carrier).abs().max().item()
    denom = max(1.0, m_fine.abs().max().item())
    rel_diff = max_diff / denom
    return rel_diff < eps or max_diff < eps


def compute_adaptive_tau_oracle(
    base_tau_step: float,
    y_current: torch.Tensor,
    stride: int = 4,
    mode: str = "density_adaptive",
    rho0: float = 0.05,
    pool_kernel: int = 5,
) -> torch.Tensor | float:
    """Authoritative reference implementation of Density-Conditioned Spatial Resolution (DCSR) tau.

    Mathematical Definition:
        tau_eff(x, y) = tau_step * min(1.0, rho0 / max(rho(x, y), eps))
    where rho(x, y) = AvgPool2d(y, k=5, s=1, p=2).
    Vanishes on dense crowd clumps (tau_eff -> 0) to eliminate negative undercounting bias (-29.95 in v29),
    while maintaining full noise suppression on empty backgrounds (tau_eff = tau_step).
    """
    if mode == "uniform" or base_tau_step <= 0.0:
        return base_tau_step

    y32 = y_current.float()
    pad = pool_kernel // 2
    local_density = F.avg_pool2d(
        y32, kernel_size=pool_kernel, stride=1, padding=pad, count_include_pad=False
    )
    rho0_val = float(rho0) * ((float(stride) / 4.0) ** 2)
    attenuation = torch.clamp(rho0_val / local_density.clamp_min(1e-6), max=1.0)
    return base_tau_step * attenuation


def proximal_firm_threshold_oracle(
    y: torch.Tensor,
    tau: float | torch.Tensor,
    mu: float = 3.0,
) -> torch.Tensor:
    """Authoritative reference implementation of Minimax Concave Penalty (MCP) / Firm Thresholding.

    Mathematical Definition:
        S_firm^+(y; tau, mu) =
            0,                       if y <= tau
            (mu / (mu - 1)) * (y - tau), if tau < y <= mu * tau
            y,                       if y > mu * tau
    """
    if isinstance(tau, (int, float)) and tau <= 0.0:
        return torch.clamp_min(y, 0.0)
    mu_val = float(max(mu, 1.001))
    mu_tau = mu_val * tau
    slope = mu_val / (mu_val - 1.0)
    ramp = slope * (y - tau)
    out = torch.where(y > mu_tau, y, ramp)
    return torch.clamp_min(out, 0.0)


def pure_bb1_oracle(
    s_diff: torch.Tensor,
    r_diff: torch.Tensor,
    omega_0: float = 1.0,
    clamp_min: float = 0.2,
    clamp_max: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Authoritative reference implementation of pure BB-1 Rayleigh contraction step size.

    Formula:
        alpha_BB1 = <s_{k-1}, r_{k-1}> / (||r_{k-1}||^2 + eps)
        omega_k = clamp(alpha_BB1, min=0.2 * omega_0, max=2.0 * omega_0)
    Falls back to omega_0 when <s, r> <= 0 (non-convex curvature).
    """
    if s_diff.ndim >= 3:
        sum_dims = tuple(range(1, s_diff.ndim))
        dot_sr = (s_diff * r_diff).sum(dim=sum_dims, keepdim=True)
        norm_r_sq = (r_diff * r_diff).sum(dim=sum_dims, keepdim=True) + eps
    else:
        dot_sr = (s_diff * r_diff).sum()
        norm_r_sq = (r_diff * r_diff).sum() + eps
    raw_step = dot_sr / norm_r_sq
    alpha_1 = torch.where(
        dot_sr > 0.0,
        raw_step,
        torch.as_tensor(omega_0, device=dot_sr.device, dtype=dot_sr.dtype),
    )
    return torch.clamp(alpha_1, min=clamp_min * omega_0, max=clamp_max * omega_0)


def support_invariance_oracle(y0: torch.Tensor, delta_y: torch.Tensor, tol: float = 1e-7) -> bool:
    """Verifies that where y0 is zero (background support), update delta_y is identically zero."""
    bg_mask = y0.float() <= 1e-6
    bg_updates = delta_y.float()[bg_mask]
    if bg_updates.numel() == 0:
        return True
    max_bg_leak = bg_updates.abs().max().item()
    return max_bg_leak <= tol


# ============================================================================
# Config Spec Generator for RMR-v30 6-Model Two-Loop Suite
# ============================================================================

def get_rmr_v30_config_spec(variant: str = "step0_v19_anchor") -> Dict[str, Any]:
    """Generates an authoritative dictionary specification for any of the 6 RMR-v30 configurations.

    Variants:
        - step0_v19_anchor: Stride 4, Linear RN SIRT, T=6, BB-1 clamped [0.2, 2.0], 104,441 params
        - h1_anscombe_sirt: Stride 4, Anscombe VST SIRT, T=6, BB-1, 104,441 params
        - h2_dual_lattice_dcsr: Dual Stride 2/4, Linear SIRT, T=6, Adaptive Tau, 104,540 params
        - h3_anscombe_dual_lattice: Dual Stride 2/4, Anscombe VST SIRT, T=6, Adaptive Tau, 104,540 params
        - h4_deep_sirt_t8: Dual Stride 2/4, Anscombe VST SIRT, T=8, Adaptive Tau, 104,540 params
        - control_no_solver: Stride 4, Direct Feedforward Y0 (0 solver iters), 104,441 params
    """
    cfg_file = Path(f"configs/rmr_v30/rmr_v30_{variant}.yaml")
    if cfg_file.is_file():
        import yaml
        return yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

    # Fallback to exact specification derived from canonical base
    base_raw = load_config(Path("configs/rmr_v19/rmr_v19_canonical_isotropic.yaml"))
    model_cfg = copy.deepcopy(base_raw["model"])
    loss_cfg = copy.deepcopy(base_raw["loss"])
    train_cfg = copy.deepcopy(base_raw.get("train", {}))
    eval_cfg = copy.deepcopy(base_raw.get("eval", {}))
    data_cfg = copy.deepcopy(base_raw.get("data", {}))

    # Base invariants
    model_cfg["use_barzilai_borwein"] = True
    model_cfg["use_alternating_bb"] = False
    model_cfg["bb_clamp_min"] = 0.2
    model_cfg["bb_clamp_max"] = 2.0
    model_cfg["morozov_gamma"] = 0.75
    model_cfg["enable_solver"] = True
    model_cfg["iterations"] = 6
    model_cfg["output_stride"] = 4
    model_cfg["subpixel_stride2"] = False
    model_cfg["use_anscombe_sirt"] = False
    model_cfg["anscombe_c"] = 0.375
    model_cfg["adaptive_tau"] = False

    if variant == "step0_v19_anchor":
        pass
    elif variant == "h1_anscombe_sirt":
        model_cfg["use_anscombe_sirt"] = True
    elif variant == "h2_dual_lattice_dcsr":
        model_cfg["output_stride"] = 2
        model_cfg["subpixel_stride2"] = True
        model_cfg["adaptive_tau"] = True
    elif variant == "h3_anscombe_dual_lattice":
        model_cfg["output_stride"] = 2
        model_cfg["subpixel_stride2"] = True
        model_cfg["use_anscombe_sirt"] = True
        model_cfg["adaptive_tau"] = True
    elif variant == "h4_deep_sirt_t8":
        model_cfg["output_stride"] = 2
        model_cfg["subpixel_stride2"] = True
        model_cfg["use_anscombe_sirt"] = True
        model_cfg["adaptive_tau"] = True
        model_cfg["iterations"] = 8
    elif variant == "control_no_solver":
        model_cfg["enable_solver"] = False
        model_cfg["iterations"] = 0
    else:
        raise ValueError(f"Unknown RMR-v30 variant: {variant}")

    return {
        "model": model_cfg,
        "loss": loss_cfg,
        "train": train_cfg,
        "eval": eval_cfg,
        "data": data_cfg,
    }


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def synthetic_carrier_stride4() -> torch.Tensor:
    """Fixture providing positive density tensor on Stride 4 carrier grid (1, 1, 64, 64)."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 64, 64) * 0.5 + 0.01


@pytest.fixture
def synthetic_fine_stride2() -> torch.Tensor:
    """Fixture providing positive density tensor on Stride 2 fine grid (1, 1, 128, 128)."""
    torch.manual_seed(42)
    return torch.rand(1, 1, 128, 128) * 0.25 + 0.005


@pytest.fixture
def standard_multiscale_regions_stride4() -> RegionSet:
    """Fixture providing standard isotropic macro regions [32, 64, 128]px on Stride 4 grid."""
    return build_multiscale_regions(
        height=64,
        width=64,
        output_stride=4,
        region_sizes_px=((32, 32), (64, 64), (128, 128)),
        overlap=0.5,
    )


@pytest.fixture
def standard_multiscale_regions_stride2() -> RegionSet:
    """Fixture providing standard isotropic macro regions [32, 64, 128]px on Stride 2 grid."""
    return build_multiscale_regions(
        height=128,
        width=128,
        output_stride=2,
        region_sizes_px=((32, 32), (64, 64), (128, 128)),
        overlap=0.5,
    )


@pytest.fixture
def canonical_step0_config_dict() -> Dict[str, Any]:
    """Fixture providing configuration dictionary for Step 0 Golden Anchor."""
    return get_rmr_v30_config_spec("step0_v19_anchor")


@pytest.fixture
def canonical_h3_config_dict() -> Dict[str, Any]:
    """Fixture providing configuration dictionary for Primary Sub-60 H3 model."""
    return get_rmr_v30_config_spec("h3_anscombe_dual_lattice")


@pytest.fixture
def sha_a_test_manifest_path() -> Path:
    """Path to canonical ShanghaiTech Part A test manifest (182 samples)."""
    p = Path("data/sha_a_test.jsonl")
    assert p.is_file(), f"Missing canonical test manifest: {p}"
    return p


@pytest.fixture
def sha_a_train_manifest_path() -> Path:
    """Path to canonical ShanghaiTech Part A train manifest (300 samples)."""
    p = Path("data/sha_a_train_all.jsonl")
    assert p.is_file(), f"Missing canonical train manifest: {p}"
    return p
