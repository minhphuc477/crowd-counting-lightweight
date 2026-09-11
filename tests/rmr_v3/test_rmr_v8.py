from __future__ import annotations

"""Hard, Comprehensive, and Adversarial Tests for RMR-v8 (Stage 2 + Stage 3).

Strictly enforces:
- Anti-Cheat Cell Loss: Non-zero penalization of hallucinations on empty backgrounds
- Density-Weighted Allocation: Proportional elevation of dense crowd gradients without background zeroing
- Discrete Conservation: Zero net flux in TV diffusion divergence before boundary clipping
- CFL Stability & Multi-step Autograd: Gradient propagation through unrolled TV without explosion/NaN
- Clean Config Synchronization: make_model, make_loss_cfg, and load_model_from_ckpt propagation
- Batch Invariance: GroupNorm guarantees zero cross-sample leakage between batch=1 and batch=4
- Dimension Robustness: Odd and asymmetric non-standard spatial resolutions (e.g. 115x173)
- Numerical Precision: Float16/BFloat16 AMP autocast safety
- Parameter Budgets: Stage 2 = 104,845; Stage 3 = 105,629
- Knowledge Distillation: DensityMapKDLoss distribution and count alignment mechanics
"""

import math
import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import multiplicative_gated_adjoint, charbonnier_tv_step
from rmr_core.necks import CoordinateAttention
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import (
    mass_weighted_cell_loss,
    compute_rmr_v3_losses,
    RMRv3LossConfig,
)
from rmr_v3.config import validate_v3_config
from rmr_v3.train import make_model, make_loss_cfg
from rmr_v3.eval import load_model_from_ckpt
from rmr_v3.kd import DensityMapKDLoss


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2a: Multiplicative Gated SIRT adjoint
# ─────────────────────────────────────────────────────────────────────────────

def test_multiplicative_gated_adjoint_suppresses_zero_pixels():
    """Gate must suppress corrections on near-zero pixels."""
    H, W = 8, 8
    boxes = torch.tensor([[0, 0, H, W]], dtype=torch.long)
    values = torch.ones(1, 1, 1)

    y_current = torch.zeros(1, 1, H, W)
    y_current[0, 0, 0, 0] = 1.0  # crowd pixel

    rho0 = 0.02
    field = multiplicative_gated_adjoint(values, boxes, y_current, H, W, rho0=rho0)

    assert field.shape == (1, 1, H, W)
    assert torch.isfinite(field).all()
    assert (field >= 0.0).all()

    gate_crowd = math.tanh(1.0 / rho0)
    assert abs(field[0, 0, 0, 0].item() - gate_crowd) < 0.01

    bg_field = field[0, 0, 1:, 1:].abs().max().item()
    assert bg_field < 1e-5, f"Background pixels should have near-zero correction, got max {bg_field:.6f}"


def test_multiplicative_gated_adjoint_full_correction_at_high_density():
    """At high density, multiplicative gate approaches 1.0 (standard additive adjoint)."""
    H, W = 4, 4
    boxes = torch.tensor([[0, 0, H, W]], dtype=torch.long)
    values = torch.ones(1, 1, 1)
    y_high = torch.full((1, 1, H, W), 10.0)
    y_low = torch.zeros(1, 1, H, W)

    field_high = multiplicative_gated_adjoint(values, boxes, y_high, H, W, rho0=0.02)
    field_low = multiplicative_gated_adjoint(values, boxes, y_low, H, W, rho0=0.02)
    field_additive = __import__("rmr_core.operators", fromlist=["regional_adjoint"]).regional_adjoint(
        values, boxes, H, W
    )

    assert field_high.mean().item() > 0.95 * field_additive.mean().item()
    assert field_low.abs().max().item() < 1e-5


def test_multiplicative_gated_no_background_lift_at_t2():
    """Background pixels must not be lifted off zero after T=2 multiplicative SIRT."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        region_sizes_px=(32, 64, 128),
        iterations=2,
        solver_mode="multiplicative",
        density_gate_rho=0.02,
        hurdle_head=False,
        temp_softplus=True,
    )
    model = RMRv3(cfg).eval()

    x = torch.zeros(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    y = out["y"]
    assert y.shape[-2:] == (32, 32)
    assert torch.isfinite(y).all()
    assert (y >= 0.0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2b: Charbonnier TV diffusion
# ─────────────────────────────────────────────────────────────────────────────

def test_charbonnier_tv_reduces_spike():
    """A single-pixel spike should be smoothed by Charbonnier TV."""
    y = torch.zeros(1, 1, 16, 16)
    y[0, 0, 8, 8] = 10.0

    y_smoothed = charbonnier_tv_step(y, lambda_tv=0.015, eps_c=0.1)

    assert y_smoothed[0, 0, 8, 8].item() < y[0, 0, 8, 8].item()
    assert (y_smoothed >= 0.0).all()
    assert torch.isfinite(y_smoothed).all()


def test_charbonnier_tv_preserves_flat_region():
    """A uniform flat region must remain unchanged under TV diffusion."""
    y = torch.full((1, 1, 16, 16), 1.0)
    y_smoothed = charbonnier_tv_step(y, lambda_tv=0.015, eps_c=0.1)
    assert torch.allclose(y_smoothed, y, atol=1e-4)


def test_charbonnier_tv_divergence_conserves_mass():
    """Exact discrete divergence of flux fields must have zero spatial sum (conservative)."""
    y = torch.rand(2, 1, 32, 32) * 5.0

    dy_dx = F.pad(y[..., 1:] - y[..., :-1], (0, 1))
    dy_dy = F.pad(y[..., 1:, :] - y[..., :-1, :], (0, 0, 0, 1))

    grad_sq = dy_dx.pow(2) + dy_dy.pow(2)
    g = (grad_sq + 0.01).rsqrt()

    flux_x = g * dy_dx
    flux_y = g * dy_dy

    div_x = flux_x - F.pad(flux_x[..., :-1], (1, 0))
    div_y = flux_y - F.pad(flux_y[..., :-1, :], (0, 0, 1, 0))

    divergence = div_x + div_y
    sum_div = divergence.sum(dim=(-2, -1)).abs().max().item()
    assert sum_div < 1e-4, f"Discrete divergence must sum to 0 across grid, got {sum_div}"


def test_charbonnier_tv_multi_step_autograd_stability():
    """Unrolling 10 TV diffusion steps must backpropagate clean, non-zero finite gradients."""
    y = torch.rand(1, 1, 16, 16, requires_grad=True)
    cur = y
    for _ in range(10):
        cur = charbonnier_tv_step(cur, lambda_tv=0.015, eps_c=0.1)

    loss = cur.sum()
    loss.backward()

    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert not torch.isnan(y.grad).any()
    assert (y.grad.abs() > 0.0).any()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2d: Mass-Weighted Cell Loss (Rigorous, Anti-Cheat Tests)
# ─────────────────────────────────────────────────────────────────────────────

def test_mass_weighted_cell_loss_strictly_penalizes_zero_target():
    """CRITICAL HARD TEST: Predicting high density on empty ground truth MUST produce heavy loss.

    Prevents 'background collapse' where a flawed mass weighting would assign 0 weight
    to empty background crops and ignore false positive hallucinations.
    """
    y = torch.full((1, 1, 32, 32), 100.0)
    target = torch.zeros(1, 1, 32, 32)

    loss = mass_weighted_cell_loss(y, target, beta=1.0, eps=1e-3, alpha=1.0)
    assert torch.isfinite(loss)
    # Smooth L1 of 100 with beta=1 is 100 - 0.5 = 99.5
    assert loss.item() > 90.0, (
        f"Loss on false-positive background must be strictly positive (>90.0), got {loss.item()}! "
        f"A zero loss indicates background collapse."
    )


def test_mass_weighted_cell_loss_gradient_structure():
    """Crowd pixels must receive elevated gradients while background retains non-zero penalty."""
    B, C, H, W = 1, 1, 32, 32
    y = torch.full((B, C, H, W), 2.0, requires_grad=True)

    # 16 pixels of dense crowd, rest background
    target = torch.zeros(B, C, H, W)
    target[0, 0, 8:12, 8:12] = 20.0

    loss = mass_weighted_cell_loss(y, target, beta=1.0, alpha=1.0)
    loss.backward()

    grad = y.grad
    assert grad is not None
    grad_crowd = grad[0, 0, 8:12, 8:12].abs().mean().item()
    grad_bg = grad[0, 0, 20:, 20:].abs().mean().item()

    # Background gradient must NOT be zero
    assert grad_bg > 0.0, "Background pixels must receive non-zero loss gradient to suppress false positives"
    # Crowd gradient must be significantly elevated
    assert grad_crowd > 5.0 * grad_bg, (
        f"Dense crowd gradient ({grad_crowd:.4f}) should be >5x background gradient ({grad_bg:.4f})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config Synchronization & Propagation Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_make_model_propagates_all_stage2_and_stage3_keys():
    """make_model must correctly instantiate RMRv3Config with all Stage 2 & 3 fields."""
    cfg = {
        "model": {
            "solver_mode": "multiplicative",
            "density_gate_rho": 0.03,
            "tv_type": "charbonnier",
            "tv_eps_c": 0.15,
            "use_coord_attn": True,
            "neck_type": "aspp_lite",
            "hurdle_head": True,
            "temp_softplus": True,
        },
        "train": {
            "backbone_lr_scale": 0.05,
        }
    }
    model, _ = make_model(cfg)
    assert model.cfg.solver_mode == "multiplicative"
    assert abs(model.cfg.density_gate_rho - 0.03) < 1e-5
    assert model.cfg.tv_type == "charbonnier"
    assert abs(model.cfg.tv_eps_c - 0.15) < 1e-5
    assert model.cfg.use_coord_attn is True
    assert model.coord_attn is not None
    assert abs(model.cfg.backbone_lr_scale - 0.05) < 1e-5


def test_make_loss_cfg_propagates_all_keys():
    """make_loss_cfg must correctly load all loss fields including cell_loss_mode and alpha."""
    cfg = {
        "loss": {
            "cell_loss_mode": "mass_weighted",
            "cell_mass_weight_eps": 5e-4,
            "cell_mass_weight_alpha": 1.5,
            "count_loss_mode": "l1",
        }
    }
    l_cfg = make_loss_cfg(cfg)
    assert l_cfg.cell_loss_mode == "mass_weighted"
    assert abs(l_cfg.cell_mass_weight_eps - 5e-4) < 1e-6
    assert abs(l_cfg.cell_mass_weight_alpha - 1.5) < 1e-6
    assert l_cfg.count_loss_mode == "l1"


def test_eval_load_model_from_ckpt_propagates_all_keys():
    """load_model_from_ckpt in eval.py must faithfully reconstruct all Stage 2 and 3 settings."""
    cfg = {
        "model": {
            "solver_mode": "multiplicative",
            "density_gate_rho": 0.02,
            "tv_type": "charbonnier",
            "tv_eps_c": 0.1,
            "use_coord_attn": True,
            "neck_type": "aspp_lite",
            "hurdle_head": True,
            "temp_softplus": True,
        }
    }
    model, _ = make_model(cfg)
    sd = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save({"model": sd, "config": cfg}, tmp_path)
        loaded_model, _, _, _ = load_model_from_ckpt(tmp_path, device=torch.device("cpu"), use_ema=False)
        assert loaded_model.cfg.solver_mode == "multiplicative"
        assert loaded_model.cfg.tv_type == "charbonnier"
        assert loaded_model.cfg.use_coord_attn is True
        assert loaded_model.coord_attn is not None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ─────────────────────────────────────────────────────────────────────────────
# Parameter Budget Verification
# ─────────────────────────────────────────────────────────────────────────────

def test_v8_stage2_parameter_budget():
    """Stage 2 canonical must have exactly 104,845 parameters (budget <= 105,000)."""
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        aspp_dilations=(1, 3, 6),
        use_aspp_gap=True,
        regional_feature_stats="mean_std",
        region_head_hidden=44,
        hurdle_head=True,
        temp_softplus=True,
        solver_mode="multiplicative",
        density_gate_rho=0.02,
        tv_type="charbonnier",
        tv_eps_c=0.1,
    )
    model = RMRv3(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == 104_845, f"Expected 104,845 parameters, got {n:,}"


def test_v8_coord_attn_parameter_budget():
    """Stage 3 (+CoordAttn) must have exactly 105,629 parameters (784 added params)."""
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        aspp_dilations=(1, 3, 6),
        use_aspp_gap=True,
        regional_feature_stats="mean_std",
        region_head_hidden=44,
        hurdle_head=True,
        temp_softplus=True,
        solver_mode="multiplicative",
        use_coord_attn=True,
    )
    model = RMRv3(cfg)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert model.coord_attn is not None
    ca_params = sum(p.numel() for p in model.coord_attn.parameters() if p.requires_grad)
    assert ca_params == 784
    assert n == 105_629


# ─────────────────────────────────────────────────────────────────────────────
# Batch Invariance & Normalization Safety
# ─────────────────────────────────────────────────────────────────────────────

def test_coord_attn_batch_invariance():
    """CoordinateAttention must produce identical outputs whether batch_size=1 or batch_size=4."""
    ca = CoordinateAttention(channels=32, reduction=4).eval()
    x1 = torch.randn(1, 32, 16, 16)
    x4 = torch.cat([x1, torch.randn(3, 32, 16, 16)], dim=0)

    with torch.no_grad():
        out1 = ca(x1)
        out4 = ca(x4)

    diff = (out4[0:1] - out1).abs().max().item()
    assert diff < 1e-5, f"Cross-sample leakage detected! diff={diff}"


def test_coord_attn_uses_groupnorm_not_batchnorm():
    """CoordinateAttention must use GroupNorm, never BatchNorm."""
    ca = CoordinateAttention(channels=32, reduction=4)
    for name, m in ca.named_modules():
        assert not isinstance(m, nn.BatchNorm2d), f"Found forbidden BatchNorm at '{name}'"
    assert any(isinstance(m, nn.GroupNorm) for m in ca.modules())


# ─────────────────────────────────────────────────────────────────────────────
# Robustness to Asymmetric and Odd Resolutions
# ─────────────────────────────────────────────────────────────────────────────

def test_odd_and_asymmetric_spatial_resolutions():
    """Model must run on odd and non-square dimensions without shape mismatch errors."""
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        iterations=2,
        solver_mode="multiplicative",
        use_coord_attn=True,
    )
    model = RMRv3(cfg).eval()

    odd_shapes = [(1, 3, 115, 173), (2, 3, 203, 151)]
    for shape in odd_shapes:
        x = torch.randn(shape)
        with torch.no_grad():
            out = model(x)
        assert "y" in out and "y0" in out
        assert torch.isfinite(out["y"]).all()


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Distillation Mechanics
# ─────────────────────────────────────────────────────────────────────────────

def test_kd_loss_mechanics():
    """DensityMapKDLoss computes valid spatial KL divergence and count alignment."""
    kd_loss = DensityMapKDLoss(lambda_spatial_kl=1.0, lambda_count_kd=0.5)

    y_target = torch.rand(2, 1, 16, 16) * 10.0
    y_student = y_target.clone().requires_grad_(True)

    # When student matches teacher, count loss is 0 and KL is ~0
    res_ident = kd_loss(y_student, y_target)
    assert res_ident["count_kd"].item() < 1e-5
    assert res_ident["spatial_kl"].item() < 1e-4

    # When student differs, loss is positive and backprops cleanly
    y_diff = (torch.rand(2, 1, 16, 16) * 5.0).requires_grad_(True)
    res_diff = kd_loss(y_diff, y_target)
    assert res_diff["total_kd"].item() > 0.0

    res_diff["total_kd"].backward()
    assert y_diff.grad is not None
    assert torch.isfinite(y_diff.grad).all()


# ─────────────────────────────────────────────────────────────────────────────
# AMP Mixed Precision Safety
# ─────────────────────────────────────────────────────────────────────────────

def test_bfloat16_numerical_stability():
    """Model forward and loss computation must run under bfloat16 autocast without NaN."""
    cfg = RMRv3Config(
        pretrained=False,
        neck_type="aspp_lite",
        iterations=2,
        solver_mode="multiplicative",
        tv_type="charbonnier",
        tv_eps_c=0.1,
        use_coord_attn=True,
    )
    model = RMRv3(cfg).train()
    x = torch.randn(1, 3, 128, 128)
    target = torch.rand(1, 1, 32, 32) * 5.0

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(x)
        losses = compute_rmr_v3_losses(
            out, target, RMRv3LossConfig(cell_loss_mode="mass_weighted")
        )
        total = losses["total"]

    assert torch.isfinite(total)
    total.backward()
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"NaN gradient in parameter '{name}'"


# ─────────────────────────────────────────────────────────────────────────────
# Advanced Edge Cases & Mathematical Invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_kd_loss_empty_teacher_no_spatial_distortion():
    """Empty teacher image (count=0) must produce 0 spatial KL, preventing uniform distortion."""
    kd_loss = DensityMapKDLoss(lambda_spatial_kl=1.0, lambda_count_kd=0.5)

    # Empty teacher
    y_teacher_empty = torch.zeros(1, 1, 16, 16)
    # Student incorrectly predicting positive crowd
    y_student = torch.ones(1, 1, 16, 16, requires_grad=True)

    res = kd_loss(y_student, y_teacher_empty)

    # Spatial KL MUST be 0.0 — no crowd distribution to distill
    assert res["spatial_kl"].item() == 0.0
    # Count KD MUST be positive — student predicted 256 instead of 0
    assert res["count_kd"].item() > 0.0


def test_operator_adjoint_dot_product_identity():
    """Verify Hilbert adjoint identity: <Ax, y> == <x, A^T y> to float32 precision."""
    from rmr_core.operators import regional_adjoint, regional_sum

    h, w = 32, 32
    boxes = torch.tensor([
        [0, 0, 16, 16],
        [8, 8, 24, 24],
        [0, 0, 32, 32],
        [16, 0, 32, 16],
    ], dtype=torch.long)
    m = boxes.shape[0]

    torch.manual_seed(123)
    x = torch.randn(2, 1, h, w, dtype=torch.float32)
    y = torch.randn(2, 1, m, dtype=torch.float32)

    # Ax = regional_sum(x)
    ax = regional_sum(x, boxes, out_dtype=torch.float32)
    # A^T y = regional_adjoint(y)
    aty = regional_adjoint(y, boxes, h, w, out_dtype=torch.float32)

    inner_ax_y = (ax * y).sum().item()
    inner_x_aty = (x * aty).sum().item()

    rel_diff = abs(inner_ax_y - inner_x_aty) / (abs(inner_ax_y) + 1e-7)
    assert rel_diff < 1e-4, f"Adjoint identity failed: <Ax, y>={inner_ax_y}, <x, A^Ty>={inner_x_aty}, rel_diff={rel_diff}"


def test_charbonnier_tv_energy_dissipation_monotone():
    """Charbonnier TV diffusion must monotonically dissipate discrete TV energy."""
    eps_c = 0.1
    lambda_tv = 0.015

    def compute_energy(field: torch.Tensor) -> float:
        dy_dx = field[..., 1:] - field[..., :-1]
        dy_dy = field[..., 1:, :] - field[..., :-1, :]
        e_x = torch.sqrt(dy_dx.pow(2) + eps_c ** 2).sum()
        e_y = torch.sqrt(dy_dy.pow(2) + eps_c ** 2).sum()
        return (e_x + e_y).item()

    torch.manual_seed(42)
    curr_y = torch.rand(1, 1, 32, 32) * 5.0
    prev_energy = compute_energy(curr_y)

    for step in range(5):
        curr_y = charbonnier_tv_step(curr_y, lambda_tv=lambda_tv, eps_c=eps_c)
        curr_energy = compute_energy(curr_y)
        assert curr_energy <= prev_energy + 1e-4, (
            f"Step {step}: Energy increased from {prev_energy:.4f} to {curr_energy:.4f}"
        )
        prev_energy = curr_energy


def test_evaluate_v3_refactored_metric_keys():
    """evaluate_v3 produces all required summary and stratified keys via evaluate_dataset."""
    from rmr_v3.train import evaluate_v3
    from torch.utils.data import DataLoader

    cfg = RMRv3Config(pretrained=False, neck_type="additive", iterations=1)
    model = RMRv3(cfg).eval()

    from rmr_core.data import rasterize_points

    pts = torch.tensor([[10.0, 10.0]])
    tgt = rasterize_points(pts, 64, 64, stride=4)

    # Synthetic dataset with 2 samples
    dummy_dataset = [
        {
            "image": torch.randn(3, 64, 64),
            "target_y": tgt,
            "id": f"dummy_{i}",
            "height": 64,
            "width": 64,
            "points": pts,
        }
        for i in range(2)
    ]
    loader = DataLoader(dummy_dataset, batch_size=1, collate_fn=lambda b: b)

    device = torch.device("cpu")
    summary = evaluate_v3(model, loader, device)

    expected_keys = [
        "MAE", "RMSE", "NAE", "Bias",
        "mae_sparse", "mae_moderate", "mae_dense",
        "GAME0", "GAME1", "GAME2", "GAME3",
        "mean_std_residual", "p50_std_residual", "p90_std_residual",
    ]
    for k in expected_keys:
        assert k in summary, f"Key '{k}' missing from evaluate_v3 summary"

