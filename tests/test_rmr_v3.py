from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rmr_count.operators import (
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
    from rmr_count.model import count_parameters
    cfg = RMRv3Config(pretrained=False)
    model = RMRv3(cfg)
    n = count_parameters(model)
    assert n < 105_000, f"OVER BUDGET: {n:,} >= 105,000"
    assert n == 101_763, f"Expected exactly 101,763 parameters, got {n:,}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
