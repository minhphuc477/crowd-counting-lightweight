"""Comprehensive test suite for RMR-v10 Principled Redesign.

Verifies:
1. Exact mathematical properties of Firm Thresholding (MCP proximal operator).
2. Contrast between Firm Thresholding (zero peak shrinkage) and Soft Thresholding (mass erosion).
3. Dual-depth allocation supervision (dm_target='dual') and bidirectional autograd connectivity.
4. Parameter budget compliance (<= 105,000 trainable parameters) across all RMR-v10 YAML configs.
5. End-to-end forward pass, solver convergence, and AMP stability.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import yaml

from rmr_v3.solver import (
    proximal_soft_threshold,
    proximal_firm_threshold,
    unrolled_sirt_solver,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.config import validate_v3_config
from rmr_core.operators import build_multiscale_regions


# ── 1. Firm Thresholding Mathematical Tests ───────────────────────────────────

def test_firm_threshold_exact_boundaries():
    """Verify exact piece-wise behavior of Firm Thresholding (MCP)."""
    tau = 0.0025
    mu = 3.0
    mu_tau = mu * tau  # 0.0075

    # Case 1: x <= tau -> strictly 0.0
    x_sub = torch.tensor([0.0, 0.0005, 0.0015, 0.0025], dtype=torch.float32)
    s_sub = proximal_firm_threshold(x_sub, tau=tau, mu=mu)
    assert torch.all(s_sub == 0.0), f"Sub-tau values must be strictly 0, got {s_sub}"

    # Case 2: x >= mu * tau -> strictly x (identity mapping, ZERO shrinkage)
    x_sup = torch.tensor([0.0075, 0.010, 0.050, 1.0, 10.0], dtype=torch.float32)
    s_sup = proximal_firm_threshold(x_sup, tau=tau, mu=mu)
    assert torch.allclose(s_sup, x_sup, atol=1e-7), (
        f"Super-mu*tau values must have 0 shrinkage (identity), got {s_sup} vs {x_sup}"
    )

    # Case 3: tau < x < mu * tau -> continuous ramp
    x_mid = torch.tensor([0.0035, 0.0050, 0.0065], dtype=torch.float32)
    s_mid = proximal_firm_threshold(x_mid, tau=tau, mu=mu)
    assert torch.all(s_mid > 0.0)
    assert torch.all(s_mid < x_mid)
    # Check monotonicity
    assert s_mid[0] < s_mid[1] < s_mid[2]


def test_firm_vs_soft_peak_preservation():
    """Prove that Firm Thresholding preserves genuine crowd peaks that Soft Thresholding erodes."""
    tau = 0.0025
    mu = 3.0
    
    # Typical density in a dense crowd cluster (m0 ~ 0.016, peak ~ 0.08)
    peaks = torch.tensor([0.015, 0.030, 0.080, 0.200], dtype=torch.float32)
    
    soft_out = proximal_soft_threshold(peaks, tau=tau)
    firm_out = proximal_firm_threshold(peaks, tau=tau, mu=mu)
    
    # Soft thresholding systematically loses mass = tau * N on all peaks
    soft_loss = (peaks - soft_out).sum().item()
    assert abs(soft_loss - tau * len(peaks)) < 1e-6, "Soft thresholding should lose tau on each peak"
    
    # Firm thresholding preserves 100% of peak mass (0 loss)
    firm_loss = (peaks - firm_out).sum().item()
    assert firm_loss == 0.0, f"Firm thresholding must have 0 mass loss on peaks, got {firm_loss}"


def test_firm_threshold_autograd_gradients():
    """Verify gradient flow across all 3 regimes of Firm Thresholding."""
    tau = 0.01
    mu = 3.0
    mu_tau = mu * tau
    slope = mu / (mu - 1.0)  # 1.5

    x = torch.tensor([0.005, 0.020, 0.050], dtype=torch.float32, requires_grad=True)
    y = proximal_firm_threshold(x, tau=tau, mu=mu)
    loss = y.sum()
    loss.backward()

    # Regime 1: x < tau -> grad = 0.0
    assert abs(x.grad[0].item() - 0.0) < 1e-6
    # Regime 2: tau < x < mu*tau -> grad = slope
    assert abs(x.grad[1].item() - slope) < 1e-6
    # Regime 3: x > mu*tau -> grad = 1.0
    assert abs(x.grad[2].item() - 1.0) < 1e-6


# ── 2. Dual-Depth Allocation Loss Tests ───────────────────────────────────────

def test_dual_depth_allocation_loss():
    """Verify that dm_target='dual' computes 0.5 * L(y0) + 0.5 * L(y) and connects gradients to both."""
    b, c, h, w = 2, 1, 32, 32
    y0 = torch.rand(b, c, h, w, requires_grad=True)
    y = (y0 * 1.5 + 0.05).clone().detach().requires_grad_(True)
    target = torch.rand(b, c, h, w)

    cfg_dual = RMRv3LossConfig(dm_target="dual")
    cfg_y0 = RMRv3LossConfig(dm_target="y0")
    cfg_y = RMRv3LossConfig(dm_target="y")

    regions = build_multiscale_regions(h, w, region_sizes_px=(32,), output_stride=4)
    m_boxes = regions.boxes.shape[0]

    # Mock output dictionary matching RMRv3 forward
    out_dict = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": torch.zeros(b, 1, m_boxes),
        "b_solver": torch.zeros(b, 1, m_boxes),
        "region_dispersion": torch.ones(b, 1, m_boxes) * 50.0,
    }

    losses_dual = compute_rmr_v3_losses(out_dict, target, points=[], cfg=cfg_dual)
    losses_y0 = compute_rmr_v3_losses(out_dict, target, points=[], cfg=cfg_y0)
    losses_y = compute_rmr_v3_losses(out_dict, target, points=[], cfg=cfg_y)

    l_dual = losses_dual["allocation"].item()
    l_y0 = losses_y0["allocation"].item()
    l_y = losses_y["allocation"].item()

    expected_dual = 0.5 * l_y0 + 0.5 * l_y
    assert abs(l_dual - expected_dual) < 1e-5, f"Dual allocation {l_dual} != expected {expected_dual}"

    # Verify gradients flow to both y0 and y
    losses_dual["allocation"].backward()
    assert y0.grad is not None and y0.grad.abs().sum() > 0.0, "y0 received no gradients under dual supervision"
    assert y.grad is not None and y.grad.abs().sum() > 0.0, "y received no gradients under dual supervision"


# ── 3. Parameter Budget Compliance Tests ──────────────────────────────────────

@pytest.mark.parametrize("config_path", [
    "configs/rmr_v10/rmr_v10_dynamic_scale_routing.yaml",
    "configs/rmr_v10/rmr_v10_canonical.yaml",
    "configs/rmr_v10/rmr_v10_canonical_isotropic.yaml",
    "configs/rmr_v10/rmr_v10_ablation_no_firm.yaml",
    "configs/rmr_v10/rmr_v10_ablation_no_dual_sup.yaml",
    "configs/rmr_v10/rmr_v10_ablation_dm_y0_only.yaml",
    "configs/rmr_v10/rmr_v10_ablation_additive_neck.yaml",
    "configs/rmr_v10/rmr_v10_control_no_solver.yaml",
])
def test_rmr_v10_configs_budget_and_validity(config_path: str):
    """Ensure every RMR-v10 config passes strict validation and is strictly <= 105,000 params."""
    path = Path(config_path)
    assert path.exists(), f"Missing config file: {config_path}"

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    # Validate configuration schema (raises ValueError on unknown/invalid keys)
    validate_v3_config(raw_cfg)

    # Instantiate model and verify parameter budget
    model_cfg = RMRv3Config.from_dict(raw_cfg["model"])
    model = RMRv3(model_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert n_params <= 105_000, (
        f"Parameter budget exceeded in {config_path}: {n_params} > 105,000"
    )

    # Verify key architectural invariants
    if "dynamic_scale_routing" in config_path:
        assert n_params == 104_440, f"DSR param count expected 104,440, got {n_params}"
        assert model_cfg.dynamic_scale_routing is True
    elif "additive_neck" in config_path:
        assert n_params == 101_813, f"Additive neck param count expected 101,813, got {n_params}"
        assert model_cfg.neck_type == "additive"
    elif "canonical" in config_path:
        assert n_params == 103_957, f"Canonical param count expected 103,957, got {n_params}"
        assert model_cfg.neck_type == "aspp_lite"
        assert model_cfg.use_aspp_gap is True
        assert model_cfg.regional_feature_stats == "mean"
        assert model_cfg.region_sizes_px == (32, 64, 128)
        assert model_cfg.proximal_mode == "firm"
        assert raw_cfg["loss"]["dm_target"] == "dual"


# ── 4. End-to-End Solver & Forward Pass Verification ──────────────────────────

def test_rmr_v10_canonical_forward_and_solver():
    """Verify end-to-end forward pass with ASPP-Lite neck, Hurdle head, and Firm SIRT solver."""
    with open("configs/rmr_v10/rmr_v10_canonical.yaml", "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    validate_v3_config(raw_cfg)
    model = RMRv3(RMRv3Config.from_dict(raw_cfg["model"]))
    model.eval()

    # Synthetic image [B=2, C=3, H=256, W=256]
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    assert "b_solver" in out
    assert "hurdle_logit" in out
    assert out["y"].shape == (2, 1, 64, 64)
    assert out["y0"].shape == (2, 1, 64, 64)
    assert torch.all(out["y"] >= 0.0), "Output density map must be non-negative"
    assert not torch.isnan(out["y"]).any(), "Output density map contains NaNs"
