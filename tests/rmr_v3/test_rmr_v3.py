from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rmr_core.operators import (
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
)
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import (
    RMRv3,
    RMRv3Config,
    reliability_from_nb,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Test 1: Uniform weight parity with standard RMR
# ---------------------------------------------------------------------------
def test_uniform_weight_matches_uniform_rmr():
    """If w_R = 1, RW-RMR normalized adjoint matches standard uniform adjoint."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.rand(2, 1, h, w, dtype=torch.float32)
    b = regional_sum(y, regions.boxes, out_dtype=torch.float32) + 0.5

    # Uniform weights w_R = 1
    w_ones = torch.ones_like(b)

    # RW-RMR field
    r_weighted = weighted_normalized_adjoint_field(y, b, w_ones, regions)

    # Standard uniform RMR field: D_c^-1 A^T D_a^-1 (AY - b)
    q = regional_sum(y, regions.boxes, out_dtype=torch.float32)
    delta = q - b
    area = regions.area.float().view(1, 1, -1)
    rate_res = delta / area
    adjoint_back = regional_adjoint(rate_res, regions.boxes, h, w, out_dtype=torch.float32)
    ones_m = torch.ones_like(b)
    std_cov = regional_adjoint(ones_m, regions.boxes, h, w, out_dtype=torch.float32).clamp_min(1e-6)
    r_standard = adjoint_back / std_cov

    assert torch.allclose(r_weighted, r_standard, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Constant residual identity
# ---------------------------------------------------------------------------
def test_constant_residual_identity():
    """If delta / |R| = delta0 * 1 for all R, adjoint field is exactly delta0 * 1."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.rand(2, 1, h, w, dtype=torch.float32)
    q = regional_sum(y, regions.boxes, out_dtype=torch.float32)

    delta0 = 1.5
    area = regions.area.float().view(1, 1, -1)
    # Construct b such that delta_R / |R| = (q_R - b_R) / |R| = delta0
    b = q - delta0 * area

    # Random positive weights
    weights = torch.rand_like(b) * 3.0 + 0.5

    field = weighted_normalized_adjoint_field(y, b, weights, regions)

    expected = torch.full_like(field, delta0)
    assert torch.allclose(field, expected, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: Positive & bounded reliability
# ---------------------------------------------------------------------------
def test_reliability_is_bounded():
    """w_R must be bounded within [w_min, w_max]."""
    torch.manual_seed(42)
    regions = build_multiscale_regions(
        64, 64, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    m = regions.boxes.shape[0]

    mu = torch.rand(2, 1, m) * 100.0
    disp = torch.rand(2, 1, m) * 100.0 + 0.5

    out = reliability_from_nb(
        mu, disp, regions,
        weight_min=0.25, weight_max=4.0,
    )
    w = out["weight"]

    assert float(w.min()) >= 0.25 - 1e-6
    assert float(w.max()) <= 4.0 + 1e-6
    assert (w > 0).all()


# ---------------------------------------------------------------------------
# Test 4: Scale-neutral normalization
# ---------------------------------------------------------------------------
def test_scale_neutral_normalization():
    """Mean reliability weight within each scale family is approximately 1.0."""
    torch.manual_seed(42)
    regions = build_multiscale_regions(
        64, 64, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    m = regions.boxes.shape[0]

    # Use moderate variance so clipping doesn't dominate
    mu = torch.rand(2, 1, m) * 20.0 + 1.0
    disp = torch.full((2, 1, m), 50.0)

    out = reliability_from_nb(
        mu, disp, regions,
        normalize_within_scale=True,
    )
    w = out["weight"]

    for sid in torch.unique(regions.scale_id):
        if int(sid.item()) < 0:
            continue
        mask = regions.scale_id == sid
        w_scale = w[..., mask]
        mean_scale = float(w_scale.mean().item())
        assert abs(mean_scale - 1.0) < 0.15, f"Scale {sid} mean weight {mean_scale} not close to 1.0"


# ---------------------------------------------------------------------------
# Test 5: FP16 operator parity
# ---------------------------------------------------------------------------
def test_fp16_operator_parity():
    """Weighted normalized adjoint maintains FP32 precision under autocast."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y_f32 = torch.rand(2, 1, h, w, dtype=torch.float32)
    b_f32 = regional_sum(y_f32, regions.boxes, out_dtype=torch.float32) + 0.2
    w_f32 = torch.rand_like(b_f32) + 0.5

    field_fp32 = weighted_normalized_adjoint_field(y_f32, b_f32, w_f32, regions)

    # Test with half precision inputs
    y_half = y_f32.half()
    b_half = b_f32.half()
    w_half = w_f32.half()
    field_from_half = weighted_normalized_adjoint_field(y_half, b_half, w_half, regions)

    # The internal calculations are FP32, so output is close to reference FP32
    assert torch.allclose(field_from_half.float(), field_fp32, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# Test 6: Zero residual yields zero update
# ---------------------------------------------------------------------------
def test_zero_residual_weighted_field():
    """If AY = b, r = 0 identically and Y_{t+1} = Y_t."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.rand(2, 1, h, w, dtype=torch.float32)
    b = regional_sum(y, regions.boxes, out_dtype=torch.float32)
    w = torch.rand_like(b) + 0.5

    r = weighted_normalized_adjoint_field(y, b, w, regions)
    assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)


# ---------------------------------------------------------------------------
# Test 7: Non-negativity of outputs (Softplus on Y0, clamp on Y_T)
# ---------------------------------------------------------------------------
def test_nonnegative_output():
    """Both Y0 and Y_final must be non-negative everywhere."""
    torch.manual_seed(42)
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg).eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert (out["y0"] >= 0).all(), "Found negative cells in initial fine measure Y0"
    assert (out["y"] >= 0).all(), "Found negative cells in final refined measure Y"


# ---------------------------------------------------------------------------
# Test 8: Gradient contract (Solver detach + Regional NB training)
# ---------------------------------------------------------------------------
def test_gradient_contract():
    """Solver detach prevents final count loss from updating regional head;
    regional NB loss updates BOTH mean and dispersion heads.
    """
    cfg = RMRv3Config(
        pretrained=False,
        detach_region_mean_in_solver=True,
        detach_reliability_in_solver=True,
    )
    model = RMRv3(cfg).train()

    x = torch.rand(2, 3, 128, 128)
    target_y = torch.rand(2, 1, 32, 32) * 0.1

    # Forward pass
    out = model(x)

    # 1. Backprop ONLY through final count loss
    model.zero_grad()
    loss_count = out["y"].sum()
    loss_count.backward(retain_graph=True)

    # Solver detach guard: mean_head and dispersion_head must have ZERO grad from solver
    mean_head_grad = model.region_head.mean_head.weight.grad
    disp_head_grad = model.region_head.log_dispersion_head.weight.grad

    assert mean_head_grad is None or mean_head_grad.abs().max().item() == 0.0, \
        f"mean_head has non-zero gradient from solver: {mean_head_grad.abs().max().item()}"
    assert disp_head_grad is None or disp_head_grad.abs().max().item() == 0.0, \
        f"disp_head has non-zero gradient from solver: {disp_head_grad.abs().max().item()}"

    # 2. Backprop through regional NB loss
    model.zero_grad()
    losses = compute_rmr_v3_losses(out, target_y)
    losses["region_nb"].backward()

    mean_grad_nb = model.region_head.mean_head.weight.grad
    disp_grad_nb = model.region_head.log_dispersion_head.weight.grad

    assert mean_grad_nb is not None and mean_grad_nb.abs().max().item() > 1e-7, \
        "mean_head received no gradient from regional NB loss"
    assert disp_grad_nb is not None and disp_grad_nb.abs().max().item() > 1e-7, \
        "log_dispersion_head received no gradient from regional NB loss"


# ---------------------------------------------------------------------------
# Test 9: Loss computation and shape compatibility
# ---------------------------------------------------------------------------
def test_loss_function_and_shapes():
    """compute_rmr_v3_losses runs without error and returns finite total loss."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg).train()

    x = torch.rand(2, 3, 128, 128)
    target_y = torch.rand(2, 1, 32, 32)

    out = model(x)
    losses = compute_rmr_v3_losses(out, target_y)

    assert "total" in losses
    assert "count" in losses
    assert "flat_dm16" in losses
    assert "cell" in losses
    assert "region_nb" in losses

    total_loss = losses["total"]
    assert torch.isfinite(total_loss), f"Total loss is not finite: {total_loss.item()}"


# ---------------------------------------------------------------------------
# Test 10: Parameter budget < 105k
# ---------------------------------------------------------------------------
def test_parameter_budget():
    """Total trainable parameters must remain strictly < 105,000."""
    from rmr_v2.model import count_parameters
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    n = count_parameters(model)
    assert n < 105_000, f"OVER BUDGET: {n:,} >= 105,000"
    assert n == 101_763, f"Expected exactly 101,763 parameters, got {n:,}"


# ---------------------------------------------------------------------------
# Test 11: Solver strength warmup protocol
# ---------------------------------------------------------------------------
def test_solver_strength_warmup():
    """When solver_strength=0.0, final measure equals initial measure Y0."""
    torch.manual_seed(42)
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg).eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out_zero = model(x, solver_strength=0.0)
        out_full = model(x, solver_strength=1.0)

    # When strength = 0.0, y == y0
    assert torch.allclose(out_zero["y"], out_zero["y0"], atol=1e-6)
    # When strength = 1.0, solver updates fine measure
    assert not torch.allclose(out_full["y"], out_full["y0"], atol=1e-6)


# ---------------------------------------------------------------------------
# Test 12: Solver region weight separation for V3-A control
# ---------------------------------------------------------------------------
def test_solver_region_weight():
    """V3-A has uniform solver weights (w=1) while preserving predicted reliability."""
    torch.manual_seed(42)
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg).eval()

    x = torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        out_v3a = model(x, uniform_reliability=True)
        out_v3b = model(x, uniform_reliability=False)

    # In V3-A, solver_region_weight is strictly 1.0 everywhere
    assert torch.allclose(out_v3a["solver_region_weight"], torch.ones_like(out_v3a["solver_region_weight"]))
    # But predicted reliability is preserved
    assert not torch.allclose(out_v3a["region_weight"], torch.ones_like(out_v3a["region_weight"]))

    # In V3-B, solver_region_weight matches region_weight
    assert torch.allclose(out_v3b["solver_region_weight"], out_v3b["region_weight"])


# ---------------------------------------------------------------------------
# Test 13: Strict config validation guards
# ---------------------------------------------------------------------------
def test_config_validation_guards():
    """RMRv3 must strictly reject invalid configs."""
    # Invalid reliability mode
    with pytest.raises(ValueError, match="Unsupported reliability_mode"):
        RMRv3(RMRv3Config(reliability_mode="invalid_mode", pretrained=False))

    # Invalid region sizes
    with pytest.raises(ValueError, match="region_sizes_px"):
        RMRv3(RMRv3Config(region_sizes_px=(32, 64), pretrained=False))

    # Invalid weight bounds
    with pytest.raises(ValueError, match="reliability_weight_min"):
        RMRv3(RMRv3Config(reliability_weight_min=5.0, reliability_weight_max=2.0, pretrained=False))

    # Invalid rate std floor
    with pytest.raises(ValueError, match="reliability_rate_std_floor"):
        RMRv3(RMRv3Config(reliability_rate_std_floor=0.0, pretrained=False))


# ---------------------------------------------------------------------------
# Test 14: Strict backbone reduction guard
# ---------------------------------------------------------------------------
def test_backbone_reduction_guard():
    """MobileNetV4Backbone must strictly require target_reductions=(4, 8, 16)."""
    from rmr_core.backbones import MobileNetV4Backbone

    with pytest.raises(ValueError, match="target_reductions"):
        MobileNetV4Backbone(target_reductions=(4, 8, 16, 32), pretrained=False)

    with pytest.raises(ValueError, match="target_reductions"):
        MobileNetV4Backbone(target_reductions=(4, 8), pretrained=False)


# ---------------------------------------------------------------------------
# Test 15: Expanded reliability calibration & solver trajectory diagnostics
# ---------------------------------------------------------------------------
def test_expanded_diagnostics():
    """Verify all diagnostics: per-scale corrs, calibration bins, saturation, and trajectory."""
    from rmr_v3.diagnostics import (
        compute_dispersion_saturation,
        compute_reliability_correlations,
        compute_solver_trajectory_diagnostics,
        compute_uncertainty_calibration_bins,
        regional_reliability_rows,
    )

    torch.manual_seed(42)
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg).eval()

    x = torch.rand(2, 3, 128, 128)
    target = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        out = model(x, solver_strength=1.0)

    # 1. Regional reliability rows
    rows = regional_reliability_rows(out, target)
    assert len(rows) > 0
    assert "std_residual" in rows[0]
    assert "count_variance" in rows[0]
    assert "rate_variance" in rows[0]

    # 2. Per-scale correlations
    corrs = compute_reliability_correlations(rows)
    for s in (32, 64, 128):
        assert f"pearson_rate_var_error_{s}" in corrs
        assert f"spearman_rate_var_error_{s}" in corrs
        assert f"spearman_weight_error_{s}" in corrs
        assert f"spearman_pred_weight_error_{s}" in corrs
    assert "spearman_pred_weight_error" in corrs

    # 3. Calibration bins (pooled and per-scale)
    calib = compute_uncertainty_calibration_bins(rows, num_bins=4)
    assert len(calib["bins"]) == 4
    assert "mean_std_residual" in calib
    assert "p50_std_residual" in calib
    assert "p90_std_residual" in calib
    assert "calibration_pooled" in calib
    assert "calibration_32" in calib
    assert "calibration_64" in calib
    assert "calibration_128" in calib

    # 4. Negative Binomial Predictive Interval Coverage
    from rmr_v3.diagnostics import compute_nb_interval_coverage
    cov = compute_nb_interval_coverage(rows, nominal_levels=(0.50, 0.80, 0.95))
    for pct in (50, 80, 95):
        assert f"coverage_{pct}" in cov
        assert f"calib_gap_{pct}" in cov
        assert 0.0 <= cov[f"coverage_{pct}"] <= 1.0
        for s in (32, 64, 128):
            assert f"coverage_{pct}_{s}" in cov
            assert f"calib_gap_{pct}_{s}" in cov
            assert 0.0 <= cov[f"coverage_{pct}_{s}"] <= 1.0

    # 5. Dispersion saturation with dynamic bounds
    sat = compute_dispersion_saturation(rows, disp_min=cfg.dispersion_min, disp_max=cfg.dispersion_max)
    assert 0.0 <= sat["dispersion_sat_low_fraction"] <= 1.0
    assert 0.0 <= sat["dispersion_sat_high_fraction"] <= 1.0

    # 6. Solver trajectory
    traj = compute_solver_trajectory_diagnostics(out, target)
    assert "mae_reg_y0" in traj
    assert "mae_reg_y1" in traj
    assert "mae_reg_y2" in traj
    assert "mae_reg_32_y0" in traj
    assert "reg_disagreement_y0" in traj
    assert "energy_monotonic_fraction" in traj
    assert "solver_help_fraction" in traj
    assert "solver_harm_fraction" in traj
    assert 0.0 <= traj["energy_monotonic_fraction"] <= 1.0
    assert 0.0 <= traj["solver_help_fraction"] <= 1.0
    assert 0.0 <= traj["solver_harm_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Test 11: Positive reliability weight guard
# ---------------------------------------------------------------------------
def test_positive_weight_guard():
    """RMR-v3 must strictly enforce reliability_weight_min > 0."""
    with pytest.raises(ValueError, match="reliability_weight_min .* must be > 0"):
        RMRv3(RMRv3Config(reliability_weight_min=0.0, pretrained=False))

    with pytest.raises(ValueError, match="reliability_weight_min .* must be > 0"):
        RMRv3(RMRv3Config(reliability_weight_min=-0.5, pretrained=False))


# ---------------------------------------------------------------------------
# Test 12: Exact RNG state save/restore reproducibility
# ---------------------------------------------------------------------------
def test_rng_state_exact_reproducibility():
    """save_rng_state and load_rng_state must identically reproduce random trajectories."""
    import random
    import numpy as np
    from rmr_core.training import load_rng_state, save_rng_state, seed_everything

    seed_everything(12345, deterministic=True)

    # Advance generators
    _ = [random.random() for _ in range(10)]
    _ = np.random.randn(10)
    _ = torch.randn(10)

    # Save state
    saved_state = save_rng_state()

    # Trajectory 1
    py_1 = [random.random() for _ in range(50)]
    np_1 = np.random.randn(50)
    th_1 = torch.randn(50)

    # Restore state
    load_rng_state(saved_state)

    # Trajectory 2
    py_2 = [random.random() for _ in range(50)]
    np_2 = np.random.randn(50)
    th_2 = torch.randn(50)

    assert py_1 == py_2, "Python random generator did not reproduce identical trajectory"
    assert np.array_equal(np_1, np_2), "NumPy random generator did not reproduce identical trajectory"
    assert torch.equal(th_1, th_2), "PyTorch random generator did not reproduce identical trajectory"


# ---------------------------------------------------------------------------
# Test 13: Manifest density fallback warning
# ---------------------------------------------------------------------------
def test_manifest_density_warning():
    """compute_manifest_density must warn on missing or unparseable manifest."""
    import warnings
    from rmr_core.data import compute_manifest_density

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        val = compute_manifest_density("nonexistent_manifest_file_12345.jsonl", default_m0=0.042)
        assert abs(val - 0.042) < 1e-6
        assert len(w) >= 1
        assert "does not exist" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
