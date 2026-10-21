"""Unit and mathematical regression test suite for MC-BPRO operator.

Verifies:
- Theorem 4.1: L1 Mass Conservation on consistent evidence.
- Theorem 4.2: Zero Background Leakage (0.000% false alarms on empty background).
- Theorem 4.3: Deficit Recovery & Zero-Support Bypass (restores missed heads in dense crowds).
- Theorem 4.4: Full Autograd Differentiability (no NaN/Inf in backward pass).
- Model End-to-End integration in RMRv3 forward pipeline.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import build_multiscale_regions, closed_form_mcbpro_transport
from rmr_core.operators.prefix_sums import regional_sum
from rmr_v3.model import RMRv3, RMRv3Config


def test_mcbpro_theorem_4_2_zero_background_leakage():
    """Verify Theorem 4.2: Pure background images produce exactly 0.000% leakage."""
    torch.manual_seed(42)
    b, h, w = 2, 64, 64
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32, 64), overlap=0.5
    )
    num_boxes = regions.boxes.shape[0]

    # Scenario: Entire image is background (carrier y0 == 0, evidence b_solver == 0)
    y0_empty = torch.zeros((b, 1, h, w), dtype=torch.float32)
    b_solver_zero = torch.zeros((b, 1, num_boxes), dtype=torch.float32)
    weight_solver = torch.ones_like(b_solver_zero)

    y_star = closed_form_mcbpro_transport(
        y0=y0_empty,
        b_solver=b_solver_zero,
        weight_solver=weight_solver,
        regions=regions,
    )

    assert torch.all(y_star == 0.0), "MC-BPRO must not leak any mass on pure background"
    assert y_star.sum().item() == 0.0, "Sum on background must be exactly 0.0"


def test_mcbpro_theorem_4_3_deficit_recovery_zero_support_bypass():
    """Verify Theorem 4.3: In dense clusters with missed heads, deficit is restored into valleys."""
    torch.manual_seed(42)
    h, w = 64, 64
    # region size 128px at stride 4 -> 32 cells (boxes are [0, 0, 32, 32], etc.)
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(128,),
        overlap=0.0,
        include_full_image=False,
    )
    num_boxes = regions.boxes.shape[0]

    # Create a dense cluster in box 0 (y: 0..32, x: 0..32)
    # Carrier has 2 sharp heads at (16, 8) and (16, 24), each of mass 1.0 (total = 2.0)
    y0 = torch.zeros((1, 1, h, w), dtype=torch.float32)
    y0[0, 0, 16, 8] = 1.0
    y0[0, 0, 16, 24] = 1.0

    # Regional evidence knows there are 5 people in box 0 (deficit = 3 people!)
    b_solver = torch.zeros((1, 1, num_boxes), dtype=torch.float32)
    b_solver[0, 0, 0] = 5.0
    weight_solver = torch.ones_like(b_solver)

    # Foreground mask covers box 0
    fg_mask = torch.zeros((1, 1, h, w), dtype=torch.float32)
    fg_mask[0, 0, 0:32, 0:32] = 1.0

    y_star, mult = closed_form_mcbpro_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        fg_mask=fg_mask,
        deficit_seeding=True,
        return_multiplier=True,
    )

    # 1. Background outside box 0 (y: 32..64, x: 32..64) must remain strictly 0
    assert torch.all(y_star[0, 0, 32:64, 32:64] == 0.0), "No leakage into background region"

    # 2. Deficit recovery must seed mass into the valley between heads at (16, 16)
    assert y_star[0, 0, 16, 16] > 0.0, "Mass must be seeded into the valley where head was missed"

    # 3. Total mass inside box 0 must be elevated towards 5.0
    box0_mass = y_star[0, 0, 0:32, 0:32].sum().item()
    assert box0_mass >= 4.8, f"Box 0 mass {box0_mass} must recover the 5.0 target count"


def test_mcbpro_theorem_4_1_l1_mass_conservation():
    """Verify Theorem 4.1: Consistent regional evidence strictly matches target total count."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.0
    )
    num_boxes = regions.boxes.shape[0]

    # Create random positive carrier
    y0 = torch.rand((1, 1, h, w), dtype=torch.float32) * 0.1
    # Integrate to get ground truth box counts
    b_true = regional_sum(y0, regions.boxes)

    # Now scale b_true by 1.5x (consistent 1.5x uniform boost)
    b_target = b_true * 1.5
    weight_solver = torch.ones_like(b_target)

    y_star = closed_form_mcbpro_transport(
        y0=y0,
        b_solver=b_target,
        weight_solver=weight_solver,
        regions=regions,
        deficit_seeding=False,  # pure scaling test
    )

    expected_total = (y0.sum() * 1.5).item()
    actual_total = y_star.sum().item()
    rel_diff = abs(actual_total - expected_total) / expected_total
    assert rel_diff < 0.02, f"Total count rel_diff {rel_diff} must be < 2%"


def test_mcbpro_theorem_4_4_autograd_differentiability():
    """Verify Theorem 4.4: Full end-to-end backprop without NaN or Inf."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5
    )
    num_boxes = regions.boxes.shape[0]

    y0 = torch.rand((2, 1, h, w), dtype=torch.float32, requires_grad=True)
    b_solver = (torch.rand((2, 1, num_boxes), dtype=torch.float32) * 10.0).requires_grad_(True)
    weight_solver = torch.ones_like(b_solver)
    scale_weights = F.softmax(torch.randn((2, 2, h, w)), dim=1).requires_grad_(True)

    y_star = closed_form_mcbpro_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        scale_weights=scale_weights,
        deficit_seeding=True,
    )

    loss = (y_star.sum() - 50.0) ** 2
    loss.backward()

    assert y0.grad is not None and not torch.isnan(y0.grad).any() and not torch.isinf(y0.grad).any()
    assert b_solver.grad is not None and not torch.isnan(b_solver.grad).any() and not torch.isinf(b_solver.grad).any()
    assert scale_weights.grad is not None and not torch.isnan(scale_weights.grad).any()


def test_mcbpro_in_rmrv3_model_pipeline():
    """Verify that RMRv3 model runs smoothly in solver_mode='mcbpro'."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        feature_width=32,
        region_sizes_px=(32, 64, 128),
        region_overlap=0.5,
        enable_solver=True,
        solver_mode="mcbpro",
        mcbpro_deficit_seeding=True,
        subpixel_stride2=False,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    assert out["y"].shape == out["y0"].shape
    assert not torch.isnan(out["y"]).any()
    assert (out["y"] >= 0.0).all()


def test_mcbpro_subpixel_stride2_model_pipeline():
    """Verify that RMRv3 model runs smoothly in solver_mode='mcbpro' with Subpixel Stride 2."""
    cfg = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        feature_width=32,
        region_sizes_px=(32, 64, 128),
        region_overlap=0.5,
        enable_solver=True,
        solver_mode="mcbpro",
        mcbpro_deficit_seeding=True,
        subpixel_stride2=True,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert "y" in out
    assert "y0" in out
    # Stride 2 output on 128x128 image should be 64x64
    assert out["y"].shape[-2:] == (64, 64)
    assert out["y"].shape == out["y0"].shape
    assert not torch.isnan(out["y"]).any()
    assert (out["y"] >= 0.0).all()

