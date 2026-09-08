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
# Test 13: Manifest density fail-hard on missing manifest
# ---------------------------------------------------------------------------
def test_manifest_density_fail_hard():
    """compute_manifest_density must fail hard with FileNotFoundError on missing manifest."""
    import pytest
    from rmr_core.data import compute_manifest_density

    with pytest.raises(FileNotFoundError, match="does not exist"):
        compute_manifest_density("nonexistent_manifest_file_12345.jsonl", default_m0=0.042)

    # When manifest is None, it should return default_m0 safely
    assert compute_manifest_density(None, default_m0=0.042) == 0.042


# ---------------------------------------------------------------------------
# Test 14: Energy autograd vs analytical gradient
# ---------------------------------------------------------------------------
def test_energy_autograd_vs_analytical_gradient():
    """Autograd \nabla_Y E_w(Y) matches analytical adjoint A^T W D_a^-1 (AY - mu)."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        h, w, output_stride=4, region_sizes_px=(32, 64, 128),
        overlap=0.5, include_full_image=False, device="cpu",
    )
    y = torch.rand(2, 1, h, w, dtype=torch.float32, requires_grad=True)
    b = regional_sum(y.detach(), regions.boxes, out_dtype=torch.float32) + 1.0
    w_reg = torch.rand_like(b) * 2.0 + 0.5

    # Scalar energy = 0.5 * sum_R w_R * ((AY)_R - b_R)^2 / |R|
    energy = weighted_regional_energy(y, b, w_reg, regions).sum()
    energy.backward()
    autograd_grad = y.grad.clone()

    # Analytical gradient: A^T (w * (AY - b) / area)
    q = regional_sum(y.detach(), regions.boxes, out_dtype=torch.float32)
    area = regions.area.float().view(1, 1, -1)
    res_scaled = w_reg * (q - b) / area
    analytical_grad = regional_adjoint(res_scaled, regions.boxes, h, w, out_dtype=torch.float32)

    assert torch.allclose(autograd_grad, analytical_grad, atol=1e-6, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 15: Bounded region cache (LRU maxsize=32)
# ---------------------------------------------------------------------------
def test_bounded_region_cache():
    """Region cache in RMRv3 must be an LRU cache bounded at maxsize=32."""
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    device = torch.device("cpu")

    # Access cache with 40 distinct grid dimensions
    for i in range(40):
        h, w = 16 + i, 16 + i
        _ = model._regions(h, w, device)

    assert len(model._region_cache) <= 32
    assert len(model._region_cache) == 32


# ---------------------------------------------------------------------------
# Test 16: Strict YAML configuration validation
# ---------------------------------------------------------------------------
def test_strict_yaml_validation():
    """validate_v3_config must reject unknown/misspelled keys at top level and section levels."""
    from rmr_v3.config import validate_v3_config

    valid_cfg = {
        "seed": 42,
        "output_dir": "runs/test",
        "data": {"train_manifest": "a.jsonl", "val_manifest": "b.jsonl"},
        "model": {"output_stride": 4, "feature_width": 32},
        "loss": {"lambda_count": 1.0},
        "train": {"batch_size": 4, "lr": 1e-4},
        "eval": {"density_bins": [100.0, 500.0]},
    }
    validate_v3_config(valid_cfg)

    # Unknown top-level key
    with pytest.raises(ValueError, match="Unknown top-level config key 'bad_top_level'"):
        validate_v3_config({**valid_cfg, "bad_top_level": 123})

    # Unknown key in section
    bad_model_cfg = dict(valid_cfg)
    bad_model_cfg["model"] = {**valid_cfg["model"], "misspelled_feature_width": 32}
    with pytest.raises(ValueError, match="Unknown config key 'misspelled_feature_width' in section 'model'"):
        validate_v3_config(bad_model_cfg)


# ---------------------------------------------------------------------------
# Test 17: AMP loss numerical precision
# ---------------------------------------------------------------------------
def test_amp_loss_numerical_precision():
    """Loss computation with float32 reductions is numerically stable under FP16."""
    loss_cfg = RMRv3LossConfig()
    b, h, w = 2, 32, 32
    target = torch.rand(b, 1, h, w, dtype=torch.float32)

    regions = build_multiscale_regions(h, w, 4, (32, 64, 128), 0.5, False, "cpu")
    num_regions = regions.boxes.shape[0]

    outputs_fp32 = {
        "y": torch.rand(b, 1, h, w, dtype=torch.float32),
        "y0": torch.rand(b, 1, h, w, dtype=torch.float32),
        "b_region": torch.rand(b, 1, num_regions, dtype=torch.float32),
        "region_dispersion": torch.full((b, 1, num_regions), 50.0, dtype=torch.float32),
        "regions": regions,
    }
    losses_fp32 = compute_rmr_v3_losses(outputs_fp32, target, loss_cfg)

    outputs_fp16 = {
        "y": outputs_fp32["y"].half(),
        "y0": outputs_fp32["y0"].half(),
        "b_region": outputs_fp32["b_region"].half(),
        "region_dispersion": outputs_fp32["region_dispersion"].half(),
        "regions": regions,
    }
    losses_fp16 = compute_rmr_v3_losses(outputs_fp16, target.half(), loss_cfg)

    for k in ("total", "count", "flat_dm16", "cell", "region_nb"):
        assert torch.isfinite(losses_fp16[k]), f"Loss {k} was not finite in FP16"
        assert torch.isfinite(losses_fp32[k]), f"Loss {k} was not finite in FP32"
        rel_diff = abs(losses_fp16[k].item() - losses_fp32[k].item()) / (abs(losses_fp32[k].item()) + 1e-6)
        assert rel_diff < 0.05, f"Loss {k} deviated too much between FP32 and FP16: {rel_diff}"


# ---------------------------------------------------------------------------
# Test 18: Resume exact trajectory reproducibility
# ---------------------------------------------------------------------------
def test_resume_exact_reproducibility():
    """Continuous 2-step training equals 1-step + checkpoint save/restore + 1-step."""
    import random
    import numpy as np
    from rmr_core.training import load_rng_state, save_rng_state, seed_everything

    def make_setup():
        seed_everything(999, deterministic=True)
        m = RMRv3(RMRv3Config(feature_width=16, pretrained=False))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, opt

    # Setup continuous run
    m_cont, opt_cont = make_setup()
    # Step 1: generate x1
    x1 = torch.randn(2, 3, 64, 64)
    out1 = m_cont(x1)
    loss1 = out1["y"].sum()
    loss1.backward()
    opt_cont.step()
    opt_cont.zero_grad()

    # Step 2: generate x2 from continuous RNG trajectory
    x2 = torch.randn(2, 3, 64, 64)
    out2 = m_cont(x2)
    loss2 = out2["y"].sum()
    loss2.backward()
    opt_cont.step()
    opt_cont.zero_grad()

    # Setup resumed run
    m_res, opt_res = make_setup()
    # Step 1: generate _x1 from same initial RNG
    _x1 = torch.randn(2, 3, 64, 64)
    assert torch.equal(_x1, x1), "Initial data generation did not match"
    _out1 = m_res(_x1)
    _loss1 = _out1["y"].sum()
    _loss1.backward()
    opt_res.step()
    opt_res.zero_grad()

    # Checkpoint saved AFTER step 1 (captures post-step-1 RNG state)
    ckpt = {
        "model": m_res.state_dict(),
        "optimizer": opt_res.state_dict(),
        "rng_state": save_rng_state(),
    }

    # Advance and corrupt RNG states to prove that load_rng_state is active
    _ = torch.randn(100, 100)
    _ = [random.random() for _ in range(100)]
    _ = np.random.randn(100)

    # Fresh load
    m_loaded = RMRv3(RMRv3Config(feature_width=16, pretrained=False))
    opt_loaded = torch.optim.AdamW(m_loaded.parameters(), lr=1e-3)
    m_loaded.load_state_dict(ckpt["model"])
    opt_loaded.load_state_dict(ckpt["optimizer"])
    load_rng_state(ckpt["rng_state"])

    # Step 2: generate data AFTER RNG restoration
    _x2 = torch.randn(2, 3, 64, 64)
    assert torch.equal(_x2, x2), "Restored RNG did not reproduce identical step-2 data generation!"

    # Step 2 on loaded model
    _out2 = m_loaded(_x2)
    _loss2 = _out2["y"].sum()
    _loss2.backward()
    opt_loaded.step()
    opt_loaded.zero_grad()

    # Compare parameters
    for p_c, p_l in zip(m_cont.parameters(), m_loaded.parameters()):
        assert torch.equal(p_c, p_l), "Parameters after resumed training did not match continuous run!"


# ---------------------------------------------------------------------------
# Test 19: Resume compatibility validation guards
# ---------------------------------------------------------------------------
def test_resume_compatibility_validation():
    """validate_resume_compatibility must reject incompatible method-critical fields."""
    from rmr_v3.config import validate_resume_compatibility

    base_cfg = {
        "model": {"omega": 1.0, "iterations": 2, "feature_width": 32, "output_stride": 4, "eps": 1e-6},
        "loss": {"lambda_count": 1.0, "lambda_cell": 0.25},
        "train": {"weight_decay": 1e-4, "solver_warmup_epochs": 5},
        "data": {"crop_size": 512},
    }

    # Exact match passes
    validate_resume_compatibility(base_cfg, dict(base_cfg))

    # Changing omega rejected
    bad_omega = {"model": {"omega": 0.5}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'model.omega'"):
        validate_resume_compatibility(base_cfg, bad_omega)

    # Changing iterations rejected
    bad_iters = {"model": {"iterations": 3}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'model.iterations'"):
        validate_resume_compatibility(base_cfg, bad_iters)

    # Changing loss weights rejected
    bad_loss = {"loss": {"lambda_count": 2.0}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'loss.lambda_count'"):
        validate_resume_compatibility(base_cfg, bad_loss)

    # Changing crop size rejected
    bad_crop = {"data": {"crop_size": 256}}
    with pytest.raises(ValueError, match="Resume config mismatch for 'data.crop_size'"):
        validate_resume_compatibility(base_cfg, bad_crop)


# ---------------------------------------------------------------------------
# Test 20: Model eps parameter wiring
# ---------------------------------------------------------------------------
def test_model_eps_wiring():
    """RMRv3Config and make_model/load_model must properly propagate eps."""
    from rmr_v3.train import make_model
    from rmr_v3.eval import load_model_from_ckpt

    cfg = {
        "model": {
            "eps": 1e-4,
            "feature_width": 16,
            "pretrained": False,
        }
    }
    model, _ = make_model(cfg)
    assert model.cfg.eps == 1e-4, f"make_model did not propagate eps: got {model.cfg.eps}"


# ---------------------------------------------------------------------------
# Test 21: Config iterations bound consistency
# ---------------------------------------------------------------------------
def test_validate_v3_config_iterations_bounds():
    """validate_v3_config must reject iterations < 1 consistently with model."""
    from rmr_v3.config import validate_v3_config

    with pytest.raises(ValueError, match="iterations must be >= 1"):
        validate_v3_config({"model": {"iterations": 0}})

    with pytest.raises(ValueError, match="iterations must be >= 1"):
        validate_v3_config({"model": {"iterations": -1}})

    validate_v3_config({"model": {"iterations": 1}})
    validate_v3_config({"model": {"iterations": 2}})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
