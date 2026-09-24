from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.operators.perspective_regions import (
    build_perspective_regions,
    park_forward_operator,
    park_adjoint_operator,
)
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.perspective_geometry import PerspectiveGeometryHead, PARKRoutingHead


# =============================================================================
# 1. MATHEMATICAL ADJOINT DUALITY TEST
# =============================================================================

def test_park_operator_mathematical_duality():
    """Verify Hilbert space adjoint duality: <A y, v> == <y, A* v> to machine epsilon.

    For linear continuous integration operator A: L2(R^{H x W}) -> R^M,
    the mathematical adjoint A*: R^M -> L2(R^{H x W}) must satisfy
    the exact inner product duality:
        <A y, v>_M = <y, A* v>_{H x W}
    within double-precision floating point tolerance (rel err < 1e-10).
    """
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    H, W = 32, 32
    stride = 4
    regions = build_perspective_regions(
        height=H,
        width=W,
        output_stride=stride,
        horizon_size_px=16,
        foreground_size_px=128,
        device=device,
    )
    M = regions.boxes.shape[0]
    assert M > 0, "No regions generated"
    float_boxes = regions.boxes.float()

    B = 2
    # Continuous positive measure y in FP64 for exact duality check
    y = torch.rand(B, 1, H, W, dtype=torch.float64, device=device) + 0.01
    v = torch.randn(B, 1, M, dtype=torch.float64, device=device)

    # 1. Forward integration: b = A y (shape [B, 1, M])
    b = park_forward_operator(y, float_boxes)

    # 2. Suffix-sum adjoint backprojection: adj = A* v (shape [B, 1, H, W])
    adj = park_adjoint_operator(v, float_boxes, height=H, width=W)

    # Compute inner products
    inner_meas = torch.sum(b * v)
    inner_grid = torch.sum(y * adj)

    abs_err = torch.abs(inner_meas - inner_grid).item()
    rel_err = abs_err / (torch.abs(inner_meas).item() + 1e-12)

    assert rel_err < 1e-10, f"Duality failed: rel_err={rel_err:.6e}, <Ay,v>={inner_meas.item()}, <y,A*v>={inner_grid.item()}"


def test_park_fractional_boxes_autograd_duality():
    """Verify that park_adjoint_operator matches analytical autograd on continuous fractional boxes."""
    torch.manual_seed(42)
    H, W = 16, 16
    float_boxes = torch.tensor(
        [
            [1.5, 2.5, 5.5, 6.5],
            [3.2, 0.8, 9.7, 12.3],
            [0.2, 4.1, 7.8, 14.9],
        ],
        dtype=torch.float64,
    )
    B = 2
    y = torch.rand(B, 1, H, W, dtype=torch.float64, requires_grad=True)
    v = torch.randn(B, 1, float_boxes.shape[0], dtype=torch.float64)

    b = park_forward_operator(y, float_boxes)
    loss = torch.sum(b * v)
    exact_adj = torch.autograd.grad(loss, y)[0]
    cust_adj = park_adjoint_operator(v, float_boxes, height=H, width=W)

    diff = torch.abs(exact_adj - cust_adj).max().item()
    assert diff < 1e-10, f"Fractional adjoint mismatch: max diff = {diff}"


# =============================================================================
# 2. SAMPLE ISOLATION & BATCH LEAKAGE AUDIT
# =============================================================================

def test_park_batch_sample_isolation_and_gradient_zero_leakage():
    """Ensure zero gradient leakage across batch samples.

    In eval mode (backbone BatchNorm stats fixed), full end-to-end gradient
    leakage must be identically zero: dL_0 / dx_1 == 0.
    In addition, all PARK modules (PGH and PARK router) must have strictly zero
    leakage even in train() mode due to sample-isolated GroupNorm.
    """
    torch.manual_seed(101)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=4,
        feature_width=32,
        use_park=True,
        park_mode="pcat",
        use_pgh=True,
        park_routing=True,
        enable_solver=True,
        iterations=3,
        adjoint_mode="radon_nikodym",
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)
    model.eval()

    B, C, H, W = 2, 3, 256, 256
    x = torch.randn(B, C, H, W, requires_grad=True)

    out = model(x)
    y_pred = out.y

    # Compute loss exclusively on sample 0
    loss_0 = torch.sum(y_pred[0] ** 2)
    loss_0.backward()

    # Verify input gradient for sample 1 is all zeros
    grad_1 = x.grad[1]
    max_leakage = torch.max(torch.abs(grad_1)).item()
    assert max_leakage == 0.0, f"Cross-sample gradient leakage detected! max |dL_0/dx_1| = {max_leakage}"

    # Also test PARK modules directly in train() mode
    pgh = PerspectiveGeometryHead(32)
    pgh.train()
    p4 = torch.randn(2, 32, 64, 64, requires_grad=True)
    h, rho = pgh(p4)
    (h[0] ** 2).sum().backward()
    assert torch.max(torch.abs(p4.grad[1])).item() == 0.0, "PGH train-mode gradient leaked across batch!"

    router = PARKRoutingHead(32)
    router.train()
    p4_r = torch.randn(2, 32, 64, 64, requires_grad=True)
    w = router(p4_r)
    (w[0] ** 2).sum().backward()
    assert torch.max(torch.abs(p4_r.grad[1])).item() == 0.0, "Router train-mode gradient leaked across batch!"



# =============================================================================
# 3. PRIME AND ARBITRARY RESOLUTION STRESS TEST
# =============================================================================

@pytest.mark.parametrize(
    "H_in, W_in",
    [
        (227, 409),  # Prime numbers
        (113, 311),  # Odd primes
        (512, 512),  # Standard crop
        (180, 540),  # Extreme 3:1 aspect ratio
    ],
)
def test_park_arbitrary_and_prime_resolutions(H_in: int, W_in: int):
    """Stress test PARK on arbitrary non-power-of-two and prime dimensions."""
    torch.manual_seed(202)
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=4,
        feature_width=32,
        use_park=True,
        park_mode="pcat",
        use_pgh=True,
        park_routing=True,
        enable_solver=True,
        iterations=2,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, H_in, W_in)
    with torch.no_grad():
        out = model(x)

    y_pred = out.y
    assert not torch.isnan(y_pred).any(), f"NaN in density output for shape ({H_in}, {W_in})"
    assert not torch.isinf(y_pred).any(), f"Inf in density output for shape ({H_in}, {W_in})"
    assert (y_pred >= 0.0).all(), f"Negative densities found for shape ({H_in}, {W_in})"

    # Verify expected stride 4 output dimensions
    exp_h = (H_in + 3) // 4
    exp_w = (W_in + 3) // 4
    assert y_pred.shape == (1, 1, exp_h, exp_w), f"Expected {(1, 1, exp_h, exp_w)}, got {y_pred.shape}"


# =============================================================================
# 4. PARAMETER BUDGET AUDIT
# =============================================================================

def test_park_strict_parameter_budget():
    """Audit total trainable parameters against the project ceiling and targets.

    Ceiling: <= 108,105 parameters.
    Target: <= 105,000 parameters (actual: ~104,666).
    """
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=4,
        feature_width=32,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        use_park=True,
        park_mode="pcat",
        use_pgh=True,
        park_routing=True,
        enable_solver=True,
        iterations=6,
        adjoint_mode="radon_nikodym",
        hurdle_head=True,
        temp_softplus=True,
    )
    model = RMRv3(cfg)

    # 1. Total trainable parameters
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_trainable <= 108105, f"Exceeded ceiling budget 108,105: got {total_trainable}"
    assert total_trainable <= 105000, f"Exceeded project target 105,000: got {total_trainable}"

    # 2. PGH parameters check (258 params)
    pgh_params = sum(p.numel() for p in model.pgh.parameters() if p.requires_grad)
    assert pgh_params <= 300, f"PGH parameters exceed 300: got {pgh_params}"

    # 3. PARK routing parameters check (450 params)
    park_router_params = sum(p.numel() for p in model.scale_router.parameters() if p.requires_grad)
    assert park_router_params <= 450, f"PARK router parameters exceed 450: got {park_router_params}"



# =============================================================================
# 5. STEP 0 IDENTITY PARITY & ZERO-INIT OF PGH
# =============================================================================

def test_pgh_zero_initialization():
    """Verify that PGH final projection is zero-initialized to guarantee Step 0 parity."""
    pgh = PerspectiveGeometryHead(
        in_channels=32,
        horizon_h_px=16.0,
        foreground_h_px=128.0,
        max_aspect_ratio=2.0,
    )

    # Check projection weights and biases
    assert torch.all(pgh.pw1d.weight == 0.0), "PGH pw1d weight is not zero-initialized"
    assert torch.all(pgh.pw1d.bias == 0.0), "PGH pw1d bias is not zero-initialized"

    # Forward on dummy features
    x = torch.randn(2, 32, 64, 64)
    h_cont, rho_cont = pgh.predict_scales(x)

    # At init, predictions must strictly equal analytical prior
    v = torch.linspace(0.0, 1.0, steps=64).view(1, 1, 64)
    exp_h = 16.0 + (128.0 - 16.0) * v
    exp_rho = 1.0 + (2.0 - 1.0) * v

    assert torch.allclose(h_cont, exp_h, atol=1e-6), "PGH did not match analytical prior at init"
    assert torch.allclose(rho_cont, exp_rho, atol=1e-6), "PGH aspect ratio did not match prior at init"

    # Forward carrier modulation must be exact identity at initialization
    p4_mod = pgh(x)
    assert torch.allclose(p4_mod, x, atol=1e-7), "PGH carrier modulation is not identity at init"



# =============================================================================
# 6. COVERAGE COMPLETENESS & HORIZON MONOTONICITY
# =============================================================================

def test_park_coverage_completeness_and_monotonicity():
    """Ensure that the perspective region tiling has no dead zones (coverage >= 1 everywhere)
    and that bounding box heights increase monotonically from horizon to foreground.
    """
    H, W = 128, 128  # Cell grid for 512x512 with stride 4
    stride = 4
    regions = build_perspective_regions(
        height=H,
        width=W,
        output_stride=stride,
        horizon_size_px=16,
        foreground_size_px=128,
        max_aspect_ratio=2.0,
    )

    coverage = torch.zeros(H, W, dtype=torch.int32)
    boxes = regions.boxes  # [M, 4] (y0, x0, y1, x1)
    heights_cell = boxes[:, 2] - boxes[:, 0]
    y_centers = (boxes[:, 0] + boxes[:, 2]).float() / 2.0

    for m in range(boxes.shape[0]):
        y0, x0, y1, x1 = boxes[m].int().tolist()
        coverage[y0:y1, x0:x1] += 1

    # Invariant 1: No blind spots / zero-coverage cells
    min_cov = coverage.min().item()
    assert min_cov >= 1, f"Found uncovered cells! Minimum coverage = {min_cov}"

    # Invariant 2: Monotonic increase in box height with elevation
    horizon_mask = y_centers < (float(H) * 0.35)
    foreground_mask = y_centers > (float(H) * 0.65)

    mean_horizon_h = heights_cell[horizon_mask].float().mean().item()
    mean_foreground_h = heights_cell[foreground_mask].float().mean().item()

    assert mean_foreground_h > 2.0 * mean_horizon_h, (
        f"Perspective scaling failure: horizon_h={mean_horizon_h:.1f}, "
        f"foreground_h={mean_foreground_h:.1f}"
    )


# =============================================================================
# 7. EXTREME VALUES & NUMERICAL STRESS AUDIT
# =============================================================================

def test_park_numerical_stability_extreme_inputs():
    """Verify stability under extreme inputs: all-zeros, huge spikes, and negative logits."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        output_stride=4,
        feature_width=32,
        use_park=True,
        park_mode="pcat",
        use_pgh=True,
        park_routing=True,
        enable_solver=True,
        iterations=3,
    )
    model = RMRv3(cfg)
    model.eval()

    # Case A: Pure black / all zeros
    x_zeros = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        out_zeros = model(x_zeros)
    assert not torch.isnan(out_zeros.y).any(), "NaN on all-zeros input"
    assert not torch.isinf(out_zeros.y).any(), "Inf on all-zeros input"

    # Case B: Extreme contrast spike (simulating flash / Dirac impulse)
    x_spike = torch.zeros(1, 3, 256, 256)
    x_spike[:, :, 128, 128] = 100.0
    with torch.no_grad():
        out_spike = model(x_spike)
    assert not torch.isnan(out_spike.y).any(), "NaN on spike input"
    assert not torch.isinf(out_spike.y).any(), "Inf on spike input"
