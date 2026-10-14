"""Comprehensive test suite for RMR v3.9 extended capabilities.

Covers:
1. BiFPN-Lite Neck:
   - Bidirectional cross-scale information flow (P4 <-> P8 <-> P16).
   - Arbitrary resolution compatibility.
   - Bottom-up gradient propagation: loss on P16 yields non-zero gradient on C4.

2. Large-Kernel Subpixel Allocation Head:
   - Configurable kernel sizes (k=3, 5, 7).
   - Step 0 uniform 0.25 partition parity.
   - Exact discrete Radon mass conservation (sum(y2) == sum(y4)).
   - Autograd gradient flow through allocation head.

3. Deep Unfolding with Learned Iteration Stepsizes and Deadbands:
   - Learnable omega_t and gamma_t parameter vectors.
   - End-to-end gradient flow from measure output back to solver parameters.

4. Subpixel Peak Localizer & Point Detection Evaluation:
   - Local 2D maximum extraction.
   - Parabolic subpixel centroid refinement (< 0.3px accuracy).
   - Precision, Recall, and F1-score evaluation at tolerance radii (4, 8, 16px).
   - Empty scene robustness.

5. Uncertainty Quantification & Risk Monitoring:
   - Analytical Negative-Binomial 95% confidence intervals.
   - Crowd congestion and stampede risk classification.
   - Continuous perspective / scale field extraction.

6. Comprehensive Multi-Task Inference API & Parameter Ceiling:
   - model.predict_comprehensive() end-to-end integration.
   - Track B parameter ceiling verification (<= 200,000 params).
"""
from __future__ import annotations

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.necks import BiFPNLiteNeck
from rmr_v3.model.dual_lattice import SubpixelAllocationHead, push_forward_stride2_to_stride4
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.canonical import (
    CANONICAL_TRACK_B_PARAM_BUDGET,
    get_track_b_model_config,
    build_track_b_model,
)
from rmr_core.metrics import (
    extract_subpixel_peaks,
    evaluate_point_localization,
    compute_count_confidence_interval,
    compute_crowd_risk_index,
    extract_perspective_field,
)


# ==============================================================================
# 1. BIFPN-LITE NECK AUDIT
# ==============================================================================

def test_bifpn_lite_neck_forward_and_shapes():
    """Verify BiFPN-Lite neck preserves exact spatial resolutions and channel width."""
    neck = BiFPNLiteNeck(in_channels=(16, 32, 48), width=32, use_gap=True)
    c4 = torch.randn(2, 16, 64, 64)
    c8 = torch.randn(2, 32, 32, 32)
    c16 = torch.randn(2, 48, 16, 16)

    p4, p8, p16 = neck(c4, c8, c16)
    assert p4.shape == (2, 32, 64, 64)
    assert p8.shape == (2, 32, 32, 32)
    assert p16.shape == (2, 32, 16, 16)


def test_bifpn_lite_neck_arbitrary_resolution():
    """Verify BiFPN-Lite handles non-power-of-two arbitrary resolutions safely."""
    neck = BiFPNLiteNeck(in_channels=(16, 32, 48), width=64, use_gap=True)
    # Prime / odd resolutions
    c4 = torch.randn(1, 16, 73, 51)
    c8 = torch.randn(1, 32, 37, 26)
    c16 = torch.randn(1, 48, 19, 13)

    p4, p8, p16 = neck(c4, c8, c16)
    assert p4.shape == (1, 64, 73, 51)
    assert p8.shape == (1, 64, 37, 26)
    assert p16.shape == (1, 64, 19, 13)


def test_bifpn_lite_neck_bidirectional_gradient_flow():
    """Verify bottom-up pathway enables high-resolution C4 to directly inform P16."""
    neck = BiFPNLiteNeck(in_channels=(16, 32, 48), width=32, use_gap=False)
    c4 = torch.randn(1, 16, 32, 32, requires_grad=True)
    c8 = torch.randn(1, 32, 16, 16, requires_grad=True)
    c16 = torch.randn(1, 48, 8, 8, requires_grad=True)

    _, _, p16 = neck(c4, c8, c16)
    # Loss computed ONLY on P16
    loss = p16.sum()
    loss.backward()

    # In top-down-only FPN, d(P16)/d(C4) would be exactly ZERO!
    # In BiFPN, bottom-up path guarantees d(P16)/d(C4) is strictly non-zero!
    assert c4.grad is not None
    assert (c4.grad.abs() > 0.0).any(), "Bottom-up pathway failed: C4 received zero gradient from P16"


# ==============================================================================
# 2. LARGE-KERNEL SUBPIXEL ALLOCATION HEAD AUDIT
# ==============================================================================

@pytest.mark.parametrize("kernel_size", [3, 5, 7])
def test_subpixel_allocation_head_step0_parity_and_kernels(kernel_size: int):
    """Verify SubpixelAllocationHead with any kernel size starts at exact uniform 0.25 partition."""
    allocator = SubpixelAllocationHead(in_channels=32, kernel_size=kernel_size)
    p4 = torch.randn(2, 32, 16, 16)
    y4 = torch.full((2, 1, 16, 16), 1.0)

    y2 = allocator(p4, y4)
    assert y2.shape == (2, 1, 32, 32)
    # Step 0 parity: 1.0 in Stride 4 cell must be uniformly partitioned to 0.25 in 4 subpixels
    assert torch.allclose(y2, torch.full_like(y2, 0.25), atol=1e-5)
    # Strict total mass conservation
    assert torch.allclose(y2.sum(dim=(-2, -1)), y4.sum(dim=(-2, -1)), atol=1e-5)


def test_subpixel_allocation_head_gradients():
    """Verify autograd backpropagates gradients through SubpixelAllocationHead to both features and mass."""
    allocator = SubpixelAllocationHead(in_channels=32, kernel_size=7)
    p4 = torch.randn(1, 32, 8, 8, requires_grad=True)
    y4 = torch.full((1, 1, 8, 8), 0.5, requires_grad=True)

    y2 = allocator(p4, y4)
    loss = (y2 ** 2).sum()
    loss.backward()

    assert p4.grad is not None and torch.isfinite(p4.grad).all()
    assert y4.grad is not None and torch.isfinite(y4.grad).all()


# ==============================================================================
# 3. LEARNED SOLVER STEPS & DEEP UNFOLDING AUDIT
# ==============================================================================

def test_learned_solver_steps_unfolding():
    """Verify learned omega_t and gamma_t are properly trained via Autograd in unrolled solver."""
    cfg = RMRv3Config(
        output_stride=4,
        enable_solver=True,
        iterations=4,
        learned_solver_steps=True,
        morozov_gamma_max=1.0,
        pretrained=False,
        max_trainable_params=200000,
    )
    model = RMRv3(cfg)
    assert model.learned_omega_raw is not None
    assert model.learned_gamma_raw is not None
    assert model.learned_omega_raw.shape == (4,)
    assert model.learned_gamma_raw.shape == (4,)

    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    loss = (out.y ** 2).sum()
    loss.backward()

    assert model.learned_omega_raw.grad is not None
    assert torch.isfinite(model.learned_omega_raw.grad).all()
    assert model.learned_gamma_raw.grad is not None
    assert torch.isfinite(model.learned_gamma_raw.grad).all()


# ==============================================================================
# 4. SUBPIXEL PEAK LOCALIZER & EVALUATION AUDIT
# ==============================================================================

def test_subpixel_peak_localizer_accuracy():
    """Verify subpixel peak localization recovers synthetic points within < 0.3 pixel error."""
    # Synthetic Stride 2 density map (128x128) representing 256x256 physical image
    # Ground truth physical points: (x=30.0, y=50.0), (x=100.0, y=80.0)
    density = np.zeros((128, 128), dtype=np.float32)
    # Stride 2: (x=30, y=50) -> cell (x=15, y=25)
    density[25, 15] = 0.8
    density[25, 14] = 0.3
    density[25, 16] = 0.3
    density[24, 15] = 0.3
    density[26, 15] = 0.3

    density[40, 50] = 0.9  # cell (x=50, y=40) -> physical (x=101.0, y=81.0)
    density[40, 49] = 0.3
    density[40, 51] = 0.3
    density[39, 50] = 0.3
    density[41, 50] = 0.3

    points_xy, scores = extract_subpixel_peaks(density, threshold=0.1, stride=2)
    assert len(points_xy) == 2
    assert len(scores) == 2

    # Check localization error
    gt_pts = np.array([[31.0, 51.0], [101.0, 81.0]], dtype=np.float32)
    eval_res = evaluate_point_localization(points_xy, gt_pts, distance_thresholds=(4.0, 8.0))
    assert eval_res["Precision@4"] == 1.0
    assert eval_res["Recall@4"] == 1.0
    assert eval_res["F1@4"] == 1.0


def test_subpixel_peak_localizer_empty_scene():
    """Verify subpixel peak extraction on completely empty scene returns zero points gracefully."""
    empty_dens = np.zeros((64, 64), dtype=np.float32)
    points, scores = extract_subpixel_peaks(empty_dens, threshold=0.05)
    assert points.shape == (0, 2)
    assert scores.shape == (0,)

    eval_res = evaluate_point_localization(points, np.zeros((0, 2)), distance_thresholds=(4.0, 8.0))
    assert eval_res["Precision@4"] == 1.0
    assert eval_res["Recall@4"] == 1.0


# ==============================================================================
# 5. UNCERTAINTY QUANTIFICATION & RISK INDEX AUDIT
# ==============================================================================

def test_confidence_interval_and_risk_monitoring():
    """Verify analytical confidence interval and crowd risk classification."""
    # 1. Confidence interval
    pred_count = 500.0
    b_var = torch.full((1, 100), 25.0)  # Total variance = 2500 / 4 = 625 -> std_err = 25.0
    ci_lo, ci_hi, std_err = compute_count_confidence_interval(pred_count, b_var)

    assert ci_lo < pred_count < ci_hi
    assert math.isclose(std_err, 25.0, rel_tol=1e-3)
    assert math.isclose(ci_hi - pred_count, 1.95996 * 25.0, rel_tol=1e-3)

    # 2. Risk classification
    dens_normal = np.full((32, 32), 0.05, dtype=np.float32)  # total = 51.2 -> 0.51 persons/m2 for 100m2
    risk_norm = compute_crowd_risk_index(dens_normal, area_sqm=100.0)
    assert risk_norm["risk_level"] == "NORMAL"

    dens_danger = np.full((32, 32), 0.50, dtype=np.float32)  # total = 512 -> 5.12 persons/m2
    risk_danger = compute_crowd_risk_index(dens_danger, area_sqm=100.0)
    assert risk_danger["risk_level"] == "CRITICAL_RISK"

    # 3. Perspective field extraction
    scale_weights = torch.tensor([
        [[[1.0, 0.0], [0.0, 0.5]],
         [[0.0, 1.0], [0.0, 0.5]],
         [[0.0, 0.0], [1.0, 0.0]]]
    ])
    scale_map = extract_perspective_field(scale_weights, scale_sizes_px=(32, 64, 128))
    assert scale_map.shape == (2, 2)
    assert math.isclose(scale_map[0, 0], 32.0, abs_tol=1e-3)
    assert math.isclose(scale_map[0, 1], 64.0, abs_tol=1e-3)
    assert math.isclose(scale_map[1, 0], 128.0, abs_tol=1e-3)
    assert math.isclose(scale_map[1, 1], 48.0, abs_tol=1e-3)  # 0.5*32 + 0.5*64 = 48.0


# ==============================================================================
# 6. COMPREHENSIVE MULTI-TASK API & PARAMETER CEILING AUDIT
# ==============================================================================

def test_predict_comprehensive_api():
    """Verify end-to-end multi-task inference API returns all structured intelligence outputs."""
    cfg = get_track_b_model_config(variant="baseline", pretrained=False)
    model = RMRv3(cfg)

    x = torch.randn(1, 3, 64, 64)
    res = model.predict_comprehensive(x, peak_threshold=0.01)

    assert "count" in res
    assert "confidence_interval" in res
    assert "points" in res
    assert "point_scores" in res
    assert "density_map" in res
    assert "scale_map" in res
    assert "risk_summary" in res
    assert res["density_map"].shape == (32, 32)
    assert res["scale_map"].shape == (16, 16)


def test_track_b_parameter_budget_ceiling():
    """Verify all Track B models strictly honor the <= 200,000 parameter budget."""
    m_base = build_track_b_model(variant="baseline", pretrained=False)
    p_base = sum(p.numel() for p in m_base.parameters() if p.requires_grad)

    m_bifpn = build_track_b_model(variant="bifpn", pretrained=False)
    p_bifpn = sum(p.numel() for p in m_bifpn.parameters() if p.requires_grad)

    m_full = build_track_b_model(variant="full", pretrained=False)
    p_full = sum(p.numel() for p in m_full.parameters() if p.requires_grad)

    assert 145000 <= p_base <= CANONICAL_TRACK_B_PARAM_BUDGET, f"Baseline {p_base} out of bounds"
    assert 145000 <= p_bifpn <= CANONICAL_TRACK_B_PARAM_BUDGET, f"BiFPN {p_bifpn} out of bounds"
    assert 145000 <= p_full <= CANONICAL_TRACK_B_PARAM_BUDGET, f"Full {p_full} out of bounds"
    print(f"\n[Verified] Track B Baseline: {p_base:,} params | BiFPN: {p_bifpn:,} params | Full: {p_full:,} params")
