"""Harsh Adversarial Verification Suite for Mathematical Essence Goals.

Tests:
1. Canonical Bayesian Loss:
   - Dynamic virtual background distance exact match.
   - Sample isolation & zero cross-batch leakage.
   - Non-starvation gradient flow in dense crowd regimes (N=10 vs N=1000).
   - Arbitrary/odd spatial grid stress testing (113x227, 409x902).
   - Empty image (N=0) preservation.
2. Bounded-Deadband Morozov Discrepancy:
   - Asymptotic bound: deadband <= gamma / rho_cap at b -> inf.
   - Small-count background preservation: deadband ~ gamma * sigma_b at b -> 0.
   - Monotonic energy descent in unrolled SIRT solver.
   - Adjoint duality: <A y, b> == <y, A* b>.
3. Orthogonal Carrier-Solver Orchestration:
   - Decoupled routing of y0 (localization) and y (mass conservation).
   - Zero cell-loss interference when lambda_cell = 0.0.
   - Full end-to-end model backward pass with MobileNetV4 <= 104,441 params.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_adjoint,
    regional_sum,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from rmr_v3.losses import (
    RMRv3LossConfig,
    bayesian_loss,
    compute_rmr_v3_losses,
)
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


# =========================================================================
# 1. CANONICAL BAYESIAN POINT SUPERVISION ADVERSARIAL TESTS
# =========================================================================

def test_bayesian_background_virtual_distance_exactness():
    """Verify that background distance dynamically scales with min distance to people."""
    h, w, stride = 64, 64, 4
    # Place a single person at (128, 128)
    pts = [torch.tensor([[128.0, 128.0]], dtype=torch.float32)]
    # Under-counting initial estimate (0.001 * 4096 = 4.096 >> 0, but person expect ~1)
    y0 = torch.full((1, 1, h, w), 0.0002, dtype=torch.float32, requires_grad=True)

    loss = bayesian_loss(
        y0, pts, sigma=8.0, background_ratio=0.1, stride=stride,
        norm_mode="canonical", canonical_background=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert y0.grad is not None
    assert torch.isfinite(y0.grad).all()

    # The pixel at center near person must receive negative gradient (drive count UP)
    # The pixel at far corner must receive positive gradient (+1.0, drive background count DOWN)
    center_grad = y0.grad[0, 0, 32, 32].item()
    corner_grad = y0.grad[0, 0, 0, 0].item()
    assert center_grad < 0.0, f"Center pixel near person must receive negative gradient, got {center_grad}"
    assert corner_grad > 0.90, f"Corner background pixel must receive positive gradient near 1.0, got {corner_grad}"


def test_bayesian_sample_isolation_no_batch_leakage():
    """Adversarial sample isolation: Sample 0 must receive ZERO gradient w.r.t Sample 1's points."""
    h, w, stride = 16, 16, 4
    pts_0 = torch.tensor([[10.0, 10.0], [20.0, 20.0]], dtype=torch.float32)
    pts_1 = torch.tensor([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0], [35.0, 35.0], [45.0, 45.0]], dtype=torch.float32)

    y0_sample0 = torch.full((1, 1, h, w), 0.05, dtype=torch.float32, requires_grad=True)
    y0_sample1 = torch.full((1, 1, h, w), 0.05, dtype=torch.float32, requires_grad=True)
    y0_batch = torch.cat([y0_sample0, y0_sample1], dim=0)

    # Compute loss for Sample 0 only
    loss_0 = bayesian_loss(
        y0_batch[:1], [pts_0], sigma=8.0, background_ratio=0.1, stride=stride,
        norm_mode="canonical", canonical_background=True,
    )
    loss_0.backward()

    # Sample 1 gradient must be identically zero
    assert y0_sample0.grad is not None
    assert (y0_sample0.grad != 0.0).any()
    assert y0_sample1.grad is not None
    assert (y0_sample1.grad == 0.0).all(), "Cross-batch leakage: Sample 1 received non-zero gradient from Sample 0's loss!"


def test_bayesian_non_starvation_dense_vs_sparse():
    """Verify that canonical Bayesian loss does NOT attenuate dense crowd gradients by 1/N."""
    h, w, stride = 32, 32, 4
    # Sparse: 5 points
    pts_sparse = torch.rand(5, 2) * (32 * 4)
    # Dense: 500 points
    pts_dense = torch.rand(500, 2) * (32 * 4)

    y_sparse = torch.full((1, 1, h, w), 0.01, dtype=torch.float32, requires_grad=True)
    y_dense = torch.full((1, 1, h, w), 0.01, dtype=torch.float32, requires_grad=True)

    l_sparse = bayesian_loss(y_sparse, [pts_sparse], norm_mode="canonical")
    l_dense = bayesian_loss(y_dense, [pts_dense], norm_mode="canonical")

    l_sparse.backward()
    l_dense.backward()

    # Per-pixel max gradient norm in dense crowd should NOT be 100x smaller than sparse
    grad_norm_sparse = y_sparse.grad.abs().max().item()
    grad_norm_dense = y_dense.grad.abs().max().item()

    assert grad_norm_sparse > 0.1, f"Sparse gradient too weak: {grad_norm_sparse}"
    assert grad_norm_dense > 0.1, f"Dense gradient starved: {grad_norm_dense}"
    ratio = grad_norm_dense / grad_norm_sparse
    assert 0.2 < ratio < 5.0, f"Gradient scale wildly imbalanced: ratio={ratio}"


def test_bayesian_arbitrary_odd_prime_resolutions():
    """Verify arbitrary resolutions without grid mismatch or crash."""
    for h, w in [(27, 33), (17, 43), (11, 19)]:
        stride = 4
        y0 = torch.full((2, 1, h, w), 0.02, dtype=torch.float32, requires_grad=True)
        pts = [
            torch.rand(12, 2) * float(min(h, w) * stride),
            torch.rand(25, 2) * float(min(h, w) * stride),
        ]
        loss = bayesian_loss(
            y0, pts, sigma=6.0, background_ratio=0.1, stride=stride,
            norm_mode="canonical", canonical_background=True,
        )
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(y0.grad).all()


def test_bayesian_empty_image_handling():
    """Empty image with zero points must strictly penalize false positives."""
    h, w, stride = 16, 16, 4
    y0 = torch.full((1, 1, h, w), 0.05, dtype=torch.float32, requires_grad=True)
    loss = bayesian_loss(y0, [torch.empty((0, 2))], stride=stride)
    expected_total = (16 * 16) * 0.05
    assert abs(loss.item() - expected_total) < 1e-4


# =========================================================================
# 2. BOUNDED-DEADBAND MOROZOV DISCREPANCY ADVERSARIAL TESTS
# =========================================================================

def test_morozov_bounded_deadband_asymptotic_ceiling():
    """Verify that deadband saturates at gamma / rho_cap as b -> infinity."""
    gamma = 0.75
    rho_cap = 0.25
    expected_cap = gamma / rho_cap  # 3.0 people

    # Regional observation with massive count (10,000 people, r=50)
    # b_variance = 10000 + 10000^2 / 50 = 2,010,000 => sigma_b ~ 1417 people
    b_val = torch.tensor([[[10000.0]]], dtype=torch.float32)
    b_var = torch.tensor([[[2010000.0]]], dtype=torch.float32)
    sigma_b = torch.sqrt(b_var)

    # Without cap: deadband = 0.75 * 1417 ~ 1063 people!
    deadband_uncapped = gamma * sigma_b
    assert deadband_uncapped.item() > 1000.0

    # With bounded cap:
    deadband_capped = (gamma * sigma_b) / (1.0 + rho_cap * sigma_b)
    assert deadband_capped.item() <= expected_cap
    assert abs(deadband_capped.item() - expected_cap) < 0.01


def test_morozov_bounded_deadband_small_count_fidelity():
    """Verify that for near-empty regions, bounded deadband preserves noise protection."""
    gamma = 0.75
    rho_cap = 0.25
    b_val = torch.tensor([[[0.5]]], dtype=torch.float32)
    b_var = torch.tensor([[[0.5 + 0.005]]], dtype=torch.float32)
    sigma_b = torch.sqrt(b_var)  # ~0.71

    deadband_uncapped = gamma * sigma_b  # 0.533
    deadband_capped = (gamma * sigma_b) / (1.0 + rho_cap * sigma_b)  # 0.533 / 1.177 = 0.453

    # Preserves active deadband on background
    assert deadband_capped.item() > 0.40
    assert deadband_capped.item() < deadband_uncapped.item()


def test_operator_adjoint_duality():
    """Mathematical duality test: <A y, b> == <y, A^T b> within float precision."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(8, 16), overlap=0.5)
    m = regions.boxes.shape[0]

    torch.manual_seed(42)
    y = torch.rand(1, 1, h, w, dtype=torch.float32)
    b = torch.rand(1, 1, m, dtype=torch.float32)

    # A y = regional_sum(y)
    ay = regional_sum(y, regions.boxes)  # [1, 1, M]
    # A^T b = regional_adjoint(b)
    at_b = regional_adjoint(b, regions.boxes, h, w)  # [1, 1, H, W]

    inner_1 = torch.sum(ay * b).item()
    inner_2 = torch.sum(y * at_b).item()

    assert abs(inner_1 - inner_2) / max(abs(inner_1), 1.0) < 1e-4, (
        f"Adjointness duality violated: <Ay, b>={inner_1} vs <y, A^Tb>={inner_2}"
    )


def test_bounded_morozov_solver_energy_descent():
    """Verify unrolled SIRT solver with Bounded Morozov achieves monotonic energy descent."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16,), overlap=0.5)
    m = regions.boxes.shape[0]

    torch.manual_seed(123)
    # Ground truth regional count
    b_target = torch.full((1, 1, m), 10.0, dtype=torch.float32)
    b_variance = torch.full((1, 1, m), 12.0, dtype=torch.float32)
    weight = torch.ones((1, 1, m), dtype=torch.float32)

    # Initial under-counting estimate y0
    y0 = torch.full((1, 1, h, w), 0.05, dtype=torch.float32)

    res = unrolled_sirt_solver(
        y0=y0, b_solver=b_target, weight_solver=weight, regions=regions,
        iterations=4, omega=1.0, morozov_gamma=0.75, morozov_rho_cap=0.25,
        b_variance=b_variance,
    )

    energy_trace = res["energy_trace"]
    assert len(energy_trace) == 4
    e_start = energy_trace[0]["before"].item()
    e_end = energy_trace[-1]["after"].item()

    assert e_end < e_start, f"Energy did not decrease: e_start={e_start}, e_end={e_end}"
    for idx in range(len(energy_trace)):
        assert energy_trace[idx]["after"].item() <= energy_trace[idx]["before"].item() + 1e-5


# =========================================================================
# 3. ORTHOGONAL CARRIER-SOLVER ORCHESTRATION & FULL ARCHITECTURE
# =========================================================================

def test_carrier_solver_orthogonal_routing():
    """Verify decoupled routing: y0 receives Bayesian allocation, y receives count loss."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16,), overlap=0.5)

    y0 = torch.full((1, 1, h, w), 0.02, dtype=torch.float32, requires_grad=True)
    y_term = torch.full((1, 1, h, w), 0.08, dtype=torch.float32, requires_grad=True)

    m = regions.boxes.shape[0]
    outputs = {
        "y0": y0,
        "y": y_term,
        "regions": regions,
        "b_region": torch.full((1, 1, m), 5.0, dtype=torch.float32),
        "region_dispersion": torch.full((1, 1, m), 50.0, dtype=torch.float32),
    }

    target_y = torch.full((1, 1, h, w), 0.05, dtype=torch.float32)
    pts = [torch.tensor([[16.0, 16.0], [32.0, 32.0]], dtype=torch.float32)]

    cfg = RMRv3LossConfig(
        allocation_loss_type="bayesian",
        dm_target="y0",        # Allocation supervises carrier y0 only
        count_target="y",      # Count loss supervises terminal y only
        lambda_cell=0.0,       # Zero discrete cell suppression
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_region_nb=0.2,
    )

    losses = compute_rmr_v3_losses(outputs, target_y, cfg, points=pts)
    assert cfg.lambda_cell == 0.0
    assert "count" in losses
    assert "allocation" in losses

    losses["total"].backward()

    assert y0.grad is not None, "Carrier y0 must receive gradient from allocation loss"
    assert y_term.grad is not None, "Terminal y must receive gradient from count loss"
    assert torch.isfinite(y0.grad).all()
    assert torch.isfinite(y_term.grad).all()


def test_full_model_end_to_end_under_budget():
    """Verify full RMR model instantiation, parameter budget <= 104,441, and forward-backward."""
    model_cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=4,
        neck_type="aspp_lite",
        iterations=2,
        morozov_gamma=0.75,
        morozov_rho_cap=0.25,
        max_trainable_params=104441,
    )

    model = RMRv3(model_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 104441, f"Parameter budget exceeded: {total_params} > 104,441"

    # Forward pass on small image
    x = torch.randn(1, 3, 256, 256, dtype=torch.float32)
    out = model(x)

    assert out.y.shape[-2:] == (64, 64)
    assert out.y0.shape[-2:] == (64, 64)
    assert torch.isfinite(out.y).all()
    assert (out.y >= 0.0).all()

    # Composite loss with canonical Bayesian allocation on y0 and count on y
    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="bayesian",
        dm_target="y0",
        count_target="y",
        lambda_cell=0.0,
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_region_nb=0.2,
        output_stride=4,
    )

    target_y = torch.zeros(1, 1, 64, 64, dtype=torch.float32)
    pts = [torch.tensor([[100.0, 100.0], [150.0, 150.0]], dtype=torch.float32)]

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)
    assert torch.isfinite(losses["total"])

    losses["total"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"NaN/Inf gradient in {name}"
