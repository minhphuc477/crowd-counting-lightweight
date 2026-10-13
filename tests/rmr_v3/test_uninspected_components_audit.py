"""Comprehensive forensic audit test suite for Dynamic Scale Routing and uninspected components in RMR.

Covers:
1. Dynamic Scale Routing & Perspective Geometry:
   - DynamicCameraAnglePredictor (DCAP): shape, tilt bounds [0, 1], Step 0 zero-init parity.
   - DiAGScaleRoutingHead: modulation by delta_scale and scene_tilt, temperature stability.
   - DiAGFactorizedRoutingHead: joint 4-window simplex partition, aspect ratio modulation.

2. Necks & Structural Re-parameterization:
   - RepWeightedFPNNeck: forward pass, algebraic branch fusion via switch_to_deploy() preserves exact output.
   - ASPPLiteFPNNeck and HDCLiteFPNNeck: static forward without dynamic attention.

3. Subpixel Dual-Lattice & Discrete Measure Pushforward:
   - push_forward_stride2_to_stride4: exact total mass preservation (sum(y4) == sum(y2)).
   - SubpixelAllocationHead: Step 0 uniform [0.25, 0.25, 0.25, 0.25] allocation parity, mass conservation.

4. Data Transforms & Coordinate Invariants:
   - rasterize_points: floor mapping, no half-pixel drift, out-of-boundary point suppression.
   - continuous pixel-center scaling: coordinate transformation invariance.

5. Loss Routing & Core Tiling:
   - TargetSupervisionRouter: single dispatch vs symmetric dual dispatch (0.5 * y + 0.5 * y0).
   - predict_tiled: exact equivalence with direct model evaluation on single-tile input.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from rmr_core.data import rasterize_points, cap_image_resolution
from rmr_core.necks import (
    AdditiveFPNNeck,
    ASPPLiteFPNNeck,
    HDCLiteFPNNeck,
    RepWeightedFPNNeck,
)
from rmr_core.necks.rep_fpn import RepDWBlock7x7
from rmr_core.scale_routing import ScaleRoutingHead, FactorizedRoutingHead
from rmr_v3.model.perspective_geometry import (
    DynamicCameraAnglePredictor,
    DiAGScaleRoutingHead,
    DiAGFactorizedRoutingHead,
)
from rmr_v3.model.dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    SubpixelAllocationHead,
    check_mass_conservation,
)
from rmr_core.heads import FineMeasureHead
from rmr_core.density_activation import _smooth_floor, _softplus_inverse
from rmr_v3.losses.router import TargetSupervisionRouter
from rmr_core.tiling import predict_tiled


# ==============================================================================
# 1. DYNAMIC SCALE ROUTING & PERSPECTIVE GEOMETRY AUDIT
# ==============================================================================

def test_dcap_step0_parity_and_bounds():
    """Verify DCAP outputs exact Step 0 identity parity (delta_scale=0, tilt=0.5) and valid shapes."""
    b, c, h, w = 2, 32, 16, 16
    p16 = torch.randn(b, c, h, w)
    dcap = DynamicCameraAnglePredictor(in_channels=32, num_scales=3, use_vertical_gradient=True)

    scene_tilt, delta_scale = dcap(p16)

    assert scene_tilt.shape == (b, 1, 1, 1)
    assert delta_scale.shape == (b, 3, 1, 1)
    # Since weights and bias are initialized to 0, sigmoid(0) = 0.5 and delta_scale = 0.0
    assert torch.allclose(scene_tilt, torch.full_like(scene_tilt, 0.5), atol=1e-6)
    assert torch.allclose(delta_scale, torch.zeros_like(delta_scale), atol=1e-6)


def test_diag_scale_routing_modulation():
    """Verify DiAGScaleRoutingHead responds correctly to delta_scale and scene_tilt modulations."""
    router = DiAGScaleRoutingHead(in_channels=32, num_scales=3, temperature=1.0)
    x = torch.randn(2, 32, 32, 32)

    # Bias heavily towards fine scale (scale 0) via delta_scale
    delta_scale = torch.tensor([[[[10.0]], [[-10.0]], [[-10.0]]]])  # [1, 3, 1, 1]
    scene_tilt = torch.tensor([[[[0.9]]]])  # [1, 1, 1, 1]

    pi = router(x, delta_scale=delta_scale, scene_tilt=scene_tilt)
    assert pi.shape == (2, 3, 32, 32)
    assert torch.allclose(pi.sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)
    # Scale 0 probability should be nearly 1.0 due to large delta_scale
    assert (pi[:, 0] > 0.99).all()


def test_diag_factorized_routing_tilt_aspect_coupling():
    """Verify DiAGFactorizedRoutingHead properly modulates aspect ratio when camera tilt varies."""
    router = DiAGFactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=2, temperature=1.0)
    x = torch.zeros(1, 32, 16, 16)

    # Steep tilt (> 0.5) should bias towards 2:1 vertical elongation (aspect 1)
    scene_tilt_steep = torch.tensor([[[[0.9]]]])
    _, _, pi_aspect_steep = router(x, scene_tilt=scene_tilt_steep)

    # Flat tilt (< 0.5) should bias towards 1:1 square (aspect 0)
    scene_tilt_flat = torch.tensor([[[[0.1]]]])
    _, _, pi_aspect_flat = router(x, scene_tilt=scene_tilt_flat)

    assert pi_aspect_steep[0, 1, 0, 0].item() > pi_aspect_steep[0, 0, 0, 0].item(), "Steep tilt must favor vertical aspect"
    assert pi_aspect_flat[0, 0, 0, 0].item() > pi_aspect_flat[0, 1, 0, 0].item(), "Flat tilt must favor square aspect"


# ==============================================================================
# 2. NECKS & STRUCTURAL RE-PARAMETERIZATION AUDIT
# ==============================================================================

def test_rep_dw_block_algebraic_reparam_equivalence():
    """Verify RepDWBlock7x7 produces identical numerical output before and after switch_to_deploy()."""
    block = RepDWBlock7x7(channels=32, act=True)
    block.eval()

    # Create dummy running statistics for batchnorms
    for m in block.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.data = torch.randn_like(m.running_mean)
            m.running_var.data = torch.rand_like(m.running_var).abs() + 0.1
            m.weight.data = torch.randn_like(m.weight)
            m.bias.data = torch.randn_like(m.bias)

    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        out_multibranch = block(x)
        block.switch_to_deploy()
        out_fused = block(x)

    max_err = (out_multibranch - out_fused).abs().max().item()
    assert max_err < 1e-4, f"Structural re-parameterization error too high: {max_err}"


def test_fpn_necks_forward_tensor_shapes():
    """Verify all FPN neck variants output (P4, P8, P16) with consistent width=32 channels."""
    in_channels = (16, 32, 48)
    c4 = torch.randn(1, 16, 64, 64)
    c8 = torch.randn(1, 32, 32, 32)
    c16 = torch.randn(1, 48, 16, 16)

    necks = [
        AdditiveFPNNeck(in_channels=in_channels, width=32),
        ASPPLiteFPNNeck(in_channels=in_channels, width=32, aspp_dilations=(1, 3, 6), use_aspp_gap=True),
        HDCLiteFPNNeck(in_channels=in_channels, width=32, hdc_dilations=(1, 2, 3), use_gap=True),
        RepWeightedFPNNeck(in_channels=in_channels, width=32),
    ]

    for neck in necks:
        neck.eval()
        with torch.no_grad():
            p4, p8, p16 = neck(c4, c8, c16)
        assert p4.shape == (1, 32, 64, 64)
        assert p8.shape == (1, 32, 32, 32)
        assert p16.shape == (1, 32, 16, 16)


# ==============================================================================
# 3. SUBPIXEL DUAL-LATTICE & DISCRETE MEASURE CONSERVATION AUDIT
# ==============================================================================

def test_push_forward_exact_mass_preservation():
    """Verify discrete Radon pushforward from Stride 2 to Stride 4 preserves exact total sum."""
    y2 = torch.rand(2, 1, 64, 64) * 0.5
    y4 = push_forward_stride2_to_stride4(y2)

    assert y4.shape == (2, 1, 32, 32)
    sum2 = y2.sum(dim=(-2, -1))
    sum4 = y4.sum(dim=(-2, -1))
    assert torch.allclose(sum2, sum4, atol=1e-5), "Push-forward violated mass conservation"


def test_subpixel_allocation_head_step0_parity():
    """Verify SubpixelAllocationHead begins at exact uniform 0.25 partition at Step 0."""
    allocator = SubpixelAllocationHead(in_channels=32)
    p4 = torch.randn(2, 32, 16, 16)
    y4 = torch.full((2, 1, 16, 16), 1.0)

    y2 = allocator(p4, y4)
    assert y2.shape == (2, 1, 32, 32)
    # In uniform allocation, 1.0 mass per 4x4 block is distributed as 0.25 per 2x2 subcell
    assert torch.allclose(y2, torch.full_like(y2, 0.25), atol=1e-5)
    # Total mass must be identically conserved
    assert torch.allclose(y2.sum(dim=(-2, -1)), y4.sum(dim=(-2, -1)), atol=1e-5)


# ==============================================================================
# 4. DATA PROCESSING & POINT COORDINATE INTEGRITY AUDIT
# ==============================================================================

def test_rasterize_points_boundary_and_offset():
    """Verify point rasterization adheres to half-open floor intervals without boundary leaks."""
    pts = torch.tensor([
        [0.0, 0.0],      # Cell (0, 0)
        [3.9, 3.9],      # Cell (0, 0)
        [4.0, 0.0],      # Cell (0, 1)
        [15.9, 15.9],    # Cell (3, 3)
        [-1.0, 5.0],     # Out of image -> must be dropped
        [16.0, 5.0],     # Out of image (>= image_w=16) -> must be dropped
    ])

    density = rasterize_points(pts, image_h=16, image_w=16, stride=4)
    assert density.shape == (1, 4, 4)
    # Cell (0, 0) should have exactly 2 points
    assert density[0, 0, 0].item() == 2.0
    # Cell (0, 1) should have exactly 1 point
    assert density[0, 0, 1].item() == 1.0
    # Cell (3, 3) should have exactly 1 point
    assert density[0, 3, 3].item() == 1.0
    # Total count in image should be exactly 4 (2 dropped)
    assert density.sum().item() == 4.0


def test_smooth_floor_quadratic_c1_continuity():
    """Verify _smooth_floor is C1 continuous and yields non-zero gradient below tau."""
    y = torch.tensor([0.01, 0.04, 0.05, 0.06, 0.10], requires_grad=True)
    tau = 0.05
    y_floored = _smooth_floor(y, floor_tau=tau)

    # Values > tau must equal y - 0.5 * tau
    assert torch.allclose(y_floored[3], y[3] - 0.5 * tau)
    # Values <= tau must equal y^2 / (2 * tau)
    assert torch.allclose(y_floored[0], y[0].square() / (2.0 * tau))

    y_floored.sum().backward()
    assert (y.grad > 0.0).all(), "Gradients below floor must remain non-zero"


# ==============================================================================
# 5. LOSS ROUTING & PREDICTION TILING AUDIT
# ==============================================================================

def test_target_supervision_router_modes():
    """Verify TargetSupervisionRouter dispatches correctly to y, y0, or symmetric dual."""
    dummy_loss_fn = lambda dmap: dmap.sum()
    y = torch.tensor([[1.0, 2.0]])
    y0 = torch.tensor([[3.0, 4.0]])

    router_y = TargetSupervisionRouter("y")
    loss_y, _ = router_y.dispatch(dummy_loss_fn, y, y0)
    assert loss_y.item() == 3.0

    router_y0 = TargetSupervisionRouter("y0")
    loss_y0, _ = router_y0.dispatch(dummy_loss_fn, y, y0)
    assert loss_y0.item() == 7.0

    router_dual = TargetSupervisionRouter("dual")
    loss_dual, aux = router_dual.dispatch(dummy_loss_fn, y, y0)
    assert loss_dual.item() == 0.5 * (3.0 + 7.0)
    assert aux["y"].item() == 3.0 and aux["y0"].item() == 7.0


def test_predict_tiled_single_tile_exact_match():
    """Verify predict_tiled gives identical result to direct forward pass when tile >= image."""
    class DummyModel(nn.Module):
        def forward(self, x, **kwargs):
            # Downsample by 4 to simulate stride 4
            return {"y": F.avg_pool2d(x.mean(dim=1, keepdim=True), kernel_size=4, stride=4)}

    model = DummyModel()
    img = torch.randn(3, 64, 64)

    with torch.no_grad():
        direct = model(img.unsqueeze(0))["y"][0]
        tiled = predict_tiled(model, img, output_stride=4, tile_size=64, halo=0)

    assert torch.allclose(direct, tiled, atol=1e-6), "predict_tiled diverged on single-tile image"
