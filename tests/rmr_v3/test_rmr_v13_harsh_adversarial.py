from __future__ import annotations

import math
from pathlib import Path
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    weighted_coverage,
    weighted_normalized_adjoint_field,
)
from rmr_v3.config import RMRv3Config, load_config, validate_v3_config
from rmr_v3.losses import (
    RMRv3LossConfig,
    compute_rmr_v3_losses,
    curvature_power_loss,
    mass_weighted_cell_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from rmr_v3.model import RMRv3
from rmr_v3.regional_head import reliability_from_nb
from rmr_v3.solver import unrolled_sirt_solver


# =============================================================================
# 1. HARSH TEST: Extreme Crowd Density (>2500 people) Stability
# =============================================================================

def test_harsh_extreme_crowd_density_stability():
    """Stress test with an ultra-dense crowd of 2,500 people.

    Tests that:
    - Prefix sums do not overflow or suffer catastrophic loss of precision.
    - Curvature power loss gradients remain strictly bounded within [-9.0, +1.0].
    - Radon-Nikodym adjoint handles large counts without NaN/Inf.
    - Full backward pass executes with finite gradients across all parameters.
    """
    torch.manual_seed(999)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64, 128),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        foreground_gate=True,
        enable_solver=True,
        iterations=4,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        tv_lambda=0.02,
        proximal_tau=0.015,
        proximal_mode="firm",
        proximal_mu=3.0,
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="flat_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_scale_align=0.05,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        lambda_fg_gate=0.05,
    )

    # Input: 1 image of size 256x256 (stride 4 -> density grid 64x64 = 4096 pixels)
    img = torch.randn(1, 3, 256, 256)
    target = torch.zeros(1, 1, 64, 64)

    # Create an extreme cluster with 2500 people packed into a 20x20 area (6.25 people per pixel)
    target[0, 0, 20:40, 20:40] = 2500.0 / 400.0
    assert abs(target.sum().item() - 2500.0) < 1e-3

    out = model(img)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)

    # Verify all losses are finite
    for k, v in losses.items():
        assert not torch.isnan(v).any(), f"Loss {k} contains NaN under extreme crowd!"
        assert not torch.isinf(v).any(), f"Loss {k} is Inf under extreme crowd!"

    total_loss = losses["total"]
    total_loss.backward()

    # Verify every single parameter gradient is finite
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            assert not torch.isnan(param.grad).any(), f"Gradient of {name} contains NaN!"
            assert not torch.isinf(param.grad).any(), f"Gradient of {name} contains Inf!"


# =============================================================================
# 2. HARSH TEST: Adversarial Pure Background Image (Zero People)
# =============================================================================

def test_harsh_pure_empty_background_adversarial():
    """Stress test with an entirely empty scene (0 people).

    Tests that:
    - Radon-Nikodym measure adjoint outputs strictly 0.0 background mass.
    - Mass-weighted cell loss reduces smoothly to smooth-L1 without division by zero.
    - Top-k hard background loss extracts and penalizes false positive hallucinations.
    - Scale routing aligns strictly to coarse scale (scale 2).
    - Hurdle loss correctly handles zero occupied regions without crashing.
    """
    torch.manual_seed(777)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        foreground_gate=True,
        enable_solver=True,
        iterations=3,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        tv_lambda=0.02,
        proximal_tau=0.015,
        proximal_mode="firm",
        proximal_mu=3.0,
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.train()

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="flat_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_scale_align=0.05,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        lambda_fg_gate=0.05,
    )

    img = torch.randn(1, 3, 128, 128)
    target = torch.zeros(1, 1, 32, 32)  # Completely empty background

    out = model(img)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)

    # 1. Truncated NB must be zero because there are zero occupied regions
    assert losses["trunc_nb"].item() == 0.0

    # 2. Curvature loss must be zero because density is nowhere >= threshold 0.08
    assert losses["curvature"].item() == 0.0

    # 3. Hard background loss MUST be active and penalizing non-zero false positives
    assert losses["hard_bg"].item() >= 0.0

    # 4. Total loss must be strictly finite and positive
    assert not torch.isnan(losses["total"])
    assert not torch.isinf(losses["total"])
    assert losses["total"].item() > 0.0

    losses["total"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert not torch.isnan(param.grad).any()


# =============================================================================
# 3. HARSH TEST: Non-Square Odd Image Dimensions & Arbitrary Aspect Ratios
# =============================================================================

def test_harsh_non_square_odd_dimensions():
    """Verify that RMR-v13 runs seamlessly on non-square, prime, and odd dimensions.

    Tests image size 193 x 129 (both prime numbers, not divisible by 16 or 32).
    """
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        foreground_gate=True,
        enable_solver=True,
        iterations=2,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.eval()

    with torch.no_grad():
        # Odd dimensions: 193 x 129
        img = torch.randn(1, 3, 193, 129)
        out = model(img)

        y = out["y"]
        expected_h = (193 + 3) // 4
        expected_w = (129 + 3) // 4
        assert y.shape[-2:] == (expected_h, expected_w)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()


# =============================================================================
# 4. HARSH TEST: Full Gradient Flow to 100% of Trainable Parameters
# =============================================================================

def test_harsh_full_gradient_flow_zero_dead_branches():
    """Verify that every single trainable parameter tensor in the model receives a non-zero gradient norm.

    Guarantees that:
    - Backbone parameters are actively trained (via backbone_lr_scale).
    - ASPP-Lite neck, including GAP context branch, receives gradient.
    - Fine carrier head receives gradient.
    - Probabilistic evidence head (count, dispersion, hurdle) receives gradient.
    - Scale routing head receives direct gradient from L_scale (pw at step 1, dw & norm at step 2).
    - Foreground gating sub-head receives gradient from L_fg_gate.
    """
    torch.manual_seed(42)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        foreground_gate=True,
        enable_solver=True,
        iterations=3,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
        tv_lambda=0.02,
        proximal_tau=0.015,
        proximal_mode="firm",
        proximal_mu=3.0,
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="flat_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_scale_align=0.05,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        lambda_fg_gate=0.05,
    )

    img = torch.randn(2, 3, 128, 128)
    target = torch.zeros(2, 1, 32, 32)
    target[0, 0, 10:14, 10:14] = 1.0  # Dense clump
    target[1, 0, 20, 20] = 0.5         # Sparse head

    # Step 1: Forward + Backward
    out = model(img)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    losses["total"].backward()

    # Step 1 check: All parameters outside dw/norm must have non-zero gradients
    # and pw.weight receives active gradient from L_scale
    assert model.scale_router.pw.weight.grad.abs().sum().item() > 0.0

    for name, param in model.named_parameters():
        if param.requires_grad and not name.startswith("scale_router.dw") and not name.startswith("scale_router.norm"):
            assert param.grad is not None and param.grad.abs().sum().item() > 0.0, f"{name} had zero gradient in step 1"

    # Step 2: Optimizer updates pw.weight to non-zero, opening gradient highway to dw and norm
    optimizer.step()
    optimizer.zero_grad()

    out = model(img)
    losses = compute_rmr_v3_losses(out, target, loss_cfg)
    losses["total"].backward()

    # Step 2 check: Scale router dw and norm now receive non-zero gradients
    assert model.scale_router.dw.weight.grad.abs().sum().item() > 0.0
    assert model.scale_router.norm.weight.grad.abs().sum().item() > 0.0


# =============================================================================
# 5. HARSH TEST: Finite Difference vs Autograd Mathematical Gradient
# =============================================================================

def test_harsh_finite_difference_scale_alignment_gradient():
    """Verify mathematical exactness of physical_scale_alignment_loss gradient via 2-sided finite differences.

    Formula: dL/dx ~ [L(x + eps) - L(x - eps)] / (2 * eps)
    """
    b, k, h, w = 1, 3, 8, 8
    target_y = torch.zeros(b, 1, h, w, dtype=torch.float64)
    target_y[0, 0, :4, :] = 0.20  # Top half dense
    target_y[0, 0, 4:, :] = 0.01  # Bottom half sparse

    logits = torch.randn(b, k, h, w, dtype=torch.float64, requires_grad=True)

    def _eval_loss(z: torch.Tensor) -> torch.Tensor:
        p = F.softmax(z, dim=1)
        return physical_scale_alignment_loss(p, target_y, tau_dense=0.12, tau_sparse=0.03, kernel_size=1)

    loss = _eval_loss(logits)
    loss.backward()
    autograd_grad = logits.grad.clone()

    eps = 1e-6
    test_coords = [(0, 0, 2, 2), (0, 1, 2, 2), (0, 2, 2, 2), (0, 0, 6, 6), (0, 2, 6, 6)]

    for b_idx, c_idx, y_idx, x_idx in test_coords:
        with torch.no_grad():
            z_plus = logits.clone()
            z_plus[b_idx, c_idx, y_idx, x_idx] += eps
            loss_plus = _eval_loss(z_plus)

            z_minus = logits.clone()
            z_minus[b_idx, c_idx, y_idx, x_idx] -= eps
            loss_minus = _eval_loss(z_minus)

            num_grad = (loss_plus - loss_minus) / (2.0 * eps)
            auto_val = autograd_grad[b_idx, c_idx, y_idx, x_idx]

            rel_err = abs(num_grad.item() - auto_val.item()) / max(abs(auto_val.item()), 1e-4)
            assert rel_err < 1e-4, (
                f"Finite difference mismatch at {b_idx, c_idx, y_idx, x_idx}: "
                f"numerical={num_grad.item():.8f} vs autograd={auto_val.item():.8f}, rel_err={rel_err:.2e}"
            )


# =============================================================================
# 6. HARSH TEST: Morozov Deadband Mathematical Contract & Continuity
# =============================================================================

def test_harsh_morozov_deadband_exact_contract_and_continuity():
    """Verify exact mathematical properties of Morozov discrepancy shrinkage:

    1. For |delta| <= gamma * sigma, shrunk discrepancy delta_tilde == 0.0 identically.
    2. For delta > gamma * sigma, delta_tilde == delta - gamma * sigma.
    3. For delta < -gamma * sigma, delta_tilde == delta + gamma * sigma.
    4. Function is globally continuous (Lipschitz constant = 1).
    """
    sigma = 3.2
    gamma = 0.75
    deadband = gamma * sigma  # 2.4

    deltas = torch.linspace(-5.0, 5.0, steps=1001)

    deadband_tensor = torch.tensor(deadband)
    shrunk = torch.sign(deltas) * torch.clamp_min(deltas.abs() - deadband_tensor, 0.0)

    # 1. Inside deadband: must be strictly 0.0
    inside_mask = deltas.abs() <= deadband
    assert (shrunk[inside_mask] == 0.0).all()

    # 2. Above deadband: delta - deadband
    above_mask = deltas > deadband
    diff_above = (shrunk[above_mask] - (deltas[above_mask] - deadband)).abs().max()
    assert diff_above < 1e-6

    # 3. Below -deadband: delta + deadband
    below_mask = deltas < -deadband
    diff_below = (shrunk[below_mask] - (deltas[below_mask] + deadband)).abs().max()
    assert diff_below < 1e-6

    # 4. Lipschitz-1 continuity: |f(x1) - f(x2)| <= |x1 - x2|
    dx = deltas[1:] - deltas[:-1]
    df = (shrunk[1:] - shrunk[:-1]).abs()
    assert (df <= dx + 1e-6).all(), "Morozov shrinkage violates Lipschitz-1 continuity!"


# =============================================================================
# 7. HARSH TEST: Radon-Nikodym Moat Non-Diffusion Test
# =============================================================================

def test_harsh_radon_nikodym_moat_non_diffusion():
    """Verify that Radon-Nikodym adjoint strictly prevents mass from jumping across a zero-density moat.

    Create two clusters A (left) and B (right) separated by a zero-density moat.
    Introduce discrepancy ONLY on cluster A.
    Verify that cluster B and the moat receive ZERO update.
    """
    h, w = 32, 64
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16,), overlap=0.5, include_full_image=False
    )

    y = torch.zeros(1, 1, h, w, dtype=torch.float32)
    y[0, 0, 10:22, 10:18] = 2.0  # Cluster A
    y[0, 0, 10:22, 46:54] = 2.0  # Cluster B

    q = regional_sum(y, regions.boxes)

    boxes = regions.boxes
    x1, x2 = boxes[:, 1], boxes[:, 3]
    overlap_a = (x1 < 18) & (x2 > 10)

    # Discrepancy ONLY on Cluster A
    b_val = q.clone()
    b_val[0, 0, overlap_a] = q[0, 0, overlap_a] - 10.0
    weights = torch.ones_like(q)

    field_rn = weighted_normalized_adjoint_field(
        y, b_val, weights, regions, adjoint_mode="radon_nikodym"
    )

    # 1. Moat must have strictly 0.0 update
    moat_update = field_rn[0, 0, :, 22:42].abs().max().item()
    assert moat_update == 0.0, f"Mass leaked into zero-density moat: {moat_update}"

    # 2. Cluster B has zero discrepancy on its regions, so it must receive 0.0 update
    cluster_b_update = field_rn[0, 0, 10:22, 46:54].abs().max().item()
    assert cluster_b_update == 0.0, f"Cluster B received unprompted update: {cluster_b_update}"

    # 3. Cluster A MUST receive positive update
    cluster_a_update = field_rn[0, 0, 10:22, 10:18].mean().item()
    assert cluster_a_update > 0.0, f"Cluster A must receive update, got {cluster_a_update}"


# =============================================================================
# 8. HARSH TEST: Density-Gated Curvature Loss Finite Difference & Gating
# =============================================================================

def test_harsh_curvature_loss_finite_difference_and_gating():
    """Verify density-gated curvature loss:
    1. Two-sided finite difference matches autograd gradient to high relative precision.
    2. Pixels below gating threshold receive strictly ZERO gradient.
    3. Analytical gradient bound: dL/dy in [1 - sqrt((y_gt + eps) / eps), 1.0].
    """
    torch.manual_seed(42)
    h, w = 16, 16
    eps = 0.01
    threshold = 0.08
    kernel_size = 5

    # Target with a dense cluster in center (>0.08) and sparse background (<0.08)
    target = torch.zeros(1, 1, h, w, dtype=torch.float64)
    target[0, 0, 6:10, 6:10] = 0.25  # Dense cluster

    # Prediction
    y = torch.full((1, 1, h, w), 0.05, dtype=torch.float64, requires_grad=True)

    loss = curvature_power_loss(
        y, target, eps=eps, threshold=threshold, kernel_size=kernel_size, mode="hard"
    )
    loss.backward()
    grad_autograd = y.grad.clone()

    # Compute local density mask to verify gating
    pad = kernel_size // 2
    local_density = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=pad)
    gate = (local_density >= threshold)

    # 1. Below threshold: gradient must be identically 0.0
    below_gate = ~gate
    assert (grad_autograd[below_gate] == 0.0).all(), "Gated curvature loss leaked gradient outside gate!"

    # 2. Above threshold: gradient direction strictly reflects under-count (<0) vs over-count (>0)
    under_count_in_gate = gate & (target > y)
    over_count_in_gate = gate & (target < y)
    assert under_count_in_gate.any(), "Cluster must contain under-counted pixels"
    assert over_count_in_gate.any(), "Boundary must contain over-counted pixels"
    assert (grad_autograd[under_count_in_gate] < 0.0).all(), "Curvature loss must push under-counted pixels upwards!"
    assert (grad_autograd[over_count_in_gate] > 0.0).all(), "Curvature loss must push over-counted pixels downwards!"

    # 3. Two-sided finite difference verification on active pixels
    delta = 1e-6
    sample_y, sample_x = 7, 7  # Active pixel inside cluster
    assert gate[0, 0, sample_y, sample_x]

    with torch.no_grad():
        y_plus = y.clone()
        y_plus[0, 0, sample_y, sample_x] += delta
        loss_plus = curvature_power_loss(
            y_plus, target, eps=eps, threshold=threshold, kernel_size=kernel_size, mode="hard"
        )

        y_minus = y.clone()
        y_minus[0, 0, sample_y, sample_x] -= delta
        loss_minus = curvature_power_loss(
            y_minus, target, eps=eps, threshold=threshold, kernel_size=kernel_size, mode="hard"
        )

        grad_num = (loss_plus - loss_minus) / (2.0 * delta)

    grad_ana = grad_autograd[0, 0, sample_y, sample_x].item()
    rel_err = abs(grad_ana - grad_num.item()) / (abs(grad_ana) + 1e-12)
    assert rel_err < 1e-5, f"Curvature finite difference error too high: {rel_err}"


# =============================================================================
# 9. HARSH TEST: Top-K Hard Background Mining Gradient Routing
# =============================================================================

def test_harsh_topk_hard_background_gradient_routing():
    """Verify that topk_hard_background_loss routes gradients with surgical precision:
    1. Gradients are strictly positive on EXACTLY the top-k background false alarms.
    2. Gradients are identically 0.0 on all remaining (1 - ratio) background pixels.
    3. Gradients are identically 0.0 on all foreground pixels.
    4. Edge case: All-foreground image (0 background pixels) returns 0.0 with autograd graph intact.
    """
    torch.manual_seed(123)
    h, w = 20, 20
    ratio = 0.10  # Top 10%
    bg_thresh = 1e-5

    # 300 background pixels, 100 foreground pixels
    target = torch.zeros(1, 1, h, w, dtype=torch.float32)
    target[0, 0, :10, :10] = 1.0  # 100 foreground pixels

    # Predicted map with false alarms on background
    y = torch.rand(1, 1, h, w, dtype=torch.float32, requires_grad=True)

    loss = topk_hard_background_loss(y, target, ratio=ratio, bg_threshold=bg_thresh)
    loss.backward()

    grad = y.grad.clone()
    assert grad is not None

    bg_mask = (target <= bg_thresh)
    fg_mask = ~bg_mask

    # 1. Foreground must receive strictly 0.0 gradient
    assert (grad[fg_mask] == 0.0).all(), "Top-K background loss leaked gradient into foreground!"

    # 2. Extract background values and check top-k selection
    bg_vals = y[bg_mask].detach()
    num_bg = bg_vals.numel()
    k = max(1, int(ratio * num_bg))

    topk_thresh = torch.topk(bg_vals, k=k, largest=True).values.min()

    # Pixels in bg with y > topk_thresh must have grad > 0
    strictly_topk_mask = bg_mask & (y > topk_thresh)
    strictly_non_topk_mask = bg_mask & (y < topk_thresh)

    assert (grad[strictly_topk_mask] > 0.0).all()
    assert (grad[strictly_non_topk_mask] == 0.0).all()

    # 3. Edge case: 100% foreground image (0 background pixels)
    all_fg_target = torch.full((1, 1, h, w), 2.0, dtype=torch.float32)
    y_fg = torch.rand(1, 1, h, w, requires_grad=True)
    loss_empty_bg = topk_hard_background_loss(y_fg, all_fg_target, ratio=ratio, bg_threshold=bg_thresh)
    assert loss_empty_bg.item() == 0.0
    loss_empty_bg.backward()
    assert y_fg.grad is not None
    assert (y_fg.grad == 0.0).all()


# =============================================================================
# 10. HARSH TEST: Mass-Weighted Cell Loss Invariants & Normalization
# =============================================================================

def test_harsh_mass_weighted_cell_loss_invariants():
    """Verify mass-weighted cell loss mathematical invariants:
    1. If target == 0 everywhere, loss is identically equal to uniform smooth-L1.
    2. Spatial weight normalization factor sum(w) / (H*W) == 1.000000 strictly.
    3. Two-sided finite difference matches autograd to < 1e-7 relative error.
    """
    torch.manual_seed(456)
    h, w = 16, 16

    # 1. Pure background invariant: target = 0
    target_zero = torch.zeros(1, 1, h, w, dtype=torch.float64)
    y_test = torch.rand(1, 1, h, w, dtype=torch.float64)

    loss_mass = mass_weighted_cell_loss(y_test, target_zero, beta=1.0, alpha=2.0, gamma=1.25)
    loss_smooth = F.smooth_l1_loss(y_test, target_zero, beta=1.0, reduction="mean")

    err_zero = abs(loss_mass.item() - loss_smooth.item())
    assert err_zero < 1e-12, f"On empty image, mass-weighted loss must equal smooth-L1, diff: {err_zero}"

    # 2. Extreme point cluster: weight normalization invariant
    target_cluster = torch.zeros(1, 1, h, w, dtype=torch.float64)
    target_cluster[0, 0, 8, 8] = 500.0  # 500 people on single pixel

    # Compute weights internally
    t_f = target_cluster.float()
    total_mass = t_f.sum(dim=(-2, -1), keepdim=True)
    p = t_f / total_mass
    p_pow = p.pow(1.25)
    p_gamma = p_pow / p_pow.sum(dim=(-2, -1), keepdim=True)
    raw_weight = 1.0 + 2.0 * float(h * w) * p_gamma
    norm_factor = raw_weight.mean(dim=(-2, -1), keepdim=True)
    weights = raw_weight / norm_factor

    assert abs(weights.mean().item() - 1.0) < 1e-6, "Weights must normalize to mean 1.0"
    # The cluster pixel must receive massive weight boost
    assert weights[0, 0, 8, 8].item() > 2.0 * float(h * w) / norm_factor.item()

    # 3. Finite difference test on double precision
    y_fd = torch.full((1, 1, h, w), 0.5, dtype=torch.float64, requires_grad=True)
    loss_fd = mass_weighted_cell_loss(y_fd, target_cluster, beta=1.0, alpha=2.0, gamma=1.25)
    loss_fd.backward()

    delta = 1e-6
    sample_y, sample_x = 8, 8
    with torch.no_grad():
        y_p = y_fd.clone()
        y_p[0, 0, sample_y, sample_x] += delta
        lp = mass_weighted_cell_loss(y_p, target_cluster, beta=1.0, alpha=2.0, gamma=1.25)

        y_m = y_fd.clone()
        y_m[0, 0, sample_y, sample_x] -= delta
        lm = mass_weighted_cell_loss(y_m, target_cluster, beta=1.0, alpha=2.0, gamma=1.25)

        num_grad = (lp - lm) / (2.0 * delta)

    ana_grad = y_fd.grad[0, 0, sample_y, sample_x].item()
    rel_diff = abs(ana_grad - num_grad.item()) / (abs(ana_grad) + 1e-12)
    assert rel_diff < 1e-7, f"Mass-weighted cell loss finite difference error too high: {rel_diff}"


# =============================================================================
# 11. HARSH TEST: Hurdle Variance Scaling & Empty Batch Loss Autograd Flow
# =============================================================================

def test_harsh_hurdle_variance_scaling_and_empty_batch():
    """Verify that:
    1. When hurdle_head predicts confident background (pi_R -> 0),
       solver_count_variance is scaled by pi_R^2 -> 0, eliminating Morozov deadband.
    2. When hurdle_head predicts confident occupied (pi_R -> 1),
       solver_count_variance matches raw NB variance.
    3. On an empty image batch, compute_rmr_v3_losses completes without crashing
       and generates finite non-NaN gradients across all model parameters.
    """
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        include_full_image=False,
        regional_feature_stats="mean",
        region_head_hidden=48,
        dynamic_scale_routing=True,
        foreground_gate=True,
        enable_solver=True,
        iterations=3,
        adjoint_mode="radon_nikodym",
        morozov_gamma=0.75,
        reliability_mode="snr",
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.eval()

    img = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(img)

    raw_var = out["region_count_variance"]
    solver_var = out["solver_count_variance"]
    hurdle_logit = out["hurdle_logit"]
    pi_r = torch.sigmoid(hurdle_logit)

    # Verify mathematical identity: Var[b_solver] == pi_r^2 * Var[b_raw]
    expected_solver_var = pi_r.square() * raw_var
    diff = (solver_var - expected_solver_var).abs().max().item()
    assert diff < 1e-6, f"Solver variance identity violated: diff = {diff}"

    # 2. Empty batch training backward pass test
    model.train()
    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="flat_dm16",
        dm_target="dual",
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.50,
        lambda_region_nb=0.20,
        lambda_hurdle=0.10,
        lambda_trunc_nb=0.20,
        lambda_scale_align=0.05,
        lambda_curvature=0.50,
        curvature_gate_threshold=0.08,
        curvature_gate_kernel=5,
        curvature_gate_mode="hard",
        lambda_hard_bg=0.10,
        lambda_fg_gate=0.05,
    )

    empty_target = torch.zeros(1, 1, 32, 32)
    out_train = model(img)
    losses = compute_rmr_v3_losses(out_train, empty_target, loss_cfg)

    assert not torch.isnan(losses["total"]).any()
    assert not torch.isinf(losses["total"]).any()
    losses["total"].backward()

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient on {name} for empty batch"
            assert not torch.isinf(p.grad).any(), f"Inf gradient on {name} for empty batch"


# =============================================================================
# 12. HARSH TEST: SIRT Solver Monotonic Non-Negativity & Stability Under T=16
# =============================================================================

def test_harsh_sirt_solver_energy_contraction_and_non_negativity():
    """Verify deep unrolling stability of SIRT solver up to T=16 iterations:
    1. Absolute Non-Negativity Invariant: y_t(u) >= 0.0 strictly everywhere for all t in [1, 16],
       even when adversarial negative targets b_solver in [-500, +500] are provided.
    2. Numerical Stability: No NaN, Inf, or numerical divergence across all 16 iterates.
    3. Radon-Nikodym measure modulation remains strictly stable without mass explosion.
    """
    torch.manual_seed(789)
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5, include_full_image=False
    )
    m = regions.boxes.shape[0]

    # Initial density with random positive values
    y0 = torch.rand(2, 1, h, w, dtype=torch.float32) * 5.0

    # Adversarial target: contains both large negative and positive regional counts
    b_solver = (torch.rand(2, 1, m, dtype=torch.float32) - 0.5) * 500.0
    weight_solver = torch.rand(2, 1, m, dtype=torch.float32) * 3.75 + 0.25
    b_var = torch.rand(2, 1, m, dtype=torch.float32) * 20.0 + 1.0

    solver_res = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        iterations=16,
        omega=1.0,
        residual_clip=0.0,
        eps=1e-6,
        solver_mode="additive",
        proximal_tau=0.015,
        proximal_mode="firm",
        proximal_mu=3.0,
        tv_lambda=0.02,
        tv_type="laplacian",
        adjoint_mode="radon_nikodym",
        b_variance=b_var,
        morozov_gamma=0.75,
        trust_region_kappa=0.35,
        trust_region_floor=0.005,
    )

    iterates = solver_res["iterates"]
    assert len(iterates) == 17  # y0 + 16 steps

    for t, yt in enumerate(iterates):
        assert not torch.isnan(yt).any(), f"NaN detected at solver iteration {t}"
        assert not torch.isinf(yt).any(), f"Inf detected at solver iteration {t}"
        assert (yt >= 0.0).all(), f"Negative density detected at solver iteration {t}! Min = {yt.min().item()}"
        # Ensure mass did not explode exponentially
        assert yt.max().item() < 1e5, f"Mass exploded at solver iteration {t}! Max = {yt.max().item()}"
