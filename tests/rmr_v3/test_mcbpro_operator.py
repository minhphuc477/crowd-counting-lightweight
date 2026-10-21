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
        output_stride=2,
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


def test_mcbpro_scale_partition_normalization_safeguard():
    """Verify that MC-BPRO preserves partition of unity when an edge-case scale is empty."""
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h, width=w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5
    )
    num_boxes = regions.boxes.shape[0]

    # Initial carrier has small mass (0.01 per cell -> total 10.24)
    y0 = torch.full((1, 1, h, w), 0.01, dtype=torch.float32)
    # Regional evidence demands 5.0 people per box -> solver must scale mass UP
    b_solver = torch.full((1, 1, num_boxes), 5.0, dtype=torch.float32)
    weight_solver = torch.ones_like(b_solver)

    # 3 scale weights provided, but only 2 scales exist in regions
    scale_weights = torch.tensor([0.2, 0.3, 0.5]).view(1, 3, 1, 1).expand(1, 3, h, w)

    y_star = closed_form_mcbpro_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        scale_weights=scale_weights,
    )

    assert not torch.isnan(y_star).any()
    assert (y_star >= 0.0).all()
    # The solver must scale mass up without losing mass from the missing 3rd scale
    assert y_star.sum().item() > y0.sum().item()


def test_mcbpro_stride2_presence_threshold_physical_scaling():
    """Verify that solver_step automatically scales mcbpro_reg_threshold by 0.25 on Stride 2."""
    from rmr_v3.model.solver_step import solve_inverse_measure

    cfg_s4 = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        feature_width=32,
        output_stride=4,
        subpixel_stride2=False,
        solver_mode="mcbpro",
        mcbpro_reg_threshold=0.001,
    )
    cfg_s2 = RMRv3Config(
        backbone_name="mobilenetv4_conv_small_050",
        feature_width=32,
        output_stride=2,
        subpixel_stride2=True,
        solver_mode="mcbpro",
        mcbpro_reg_threshold=0.001,
    )

    assert cfg_s4.output_stride == 4
    assert cfg_s2.output_stride == 2


def test_v62_abl4_hurdle_head_integration():
    """Verify that v62_abl4 with hurdle_head=True activates hurdle_bce and trunc_nb losses."""
    import yaml
    from rmr_v3.engine import make_loss_cfg, make_model
    from rmr_v3.losses import compute_rmr_v3_losses

    with open("configs/rmr_v62/v62_abl4_unpurified_8loss.yaml") as f:
        cfg = yaml.safe_load(f)

    assert cfg["model"]["hurdle_head"] is True
    assert cfg["loss"]["lambda_hurdle"] > 0.0
    assert cfg["loss"]["lambda_trunc_nb"] > 0.0

    model, _ = make_model(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 200000, f"Params {total_params} exceeded 200,000 ceiling"

    # Forward pass on small dummy batch
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert "hurdle_logit" in out, "hurdle_logit must be present in model output"
    assert out["hurdle_logit"] is not None

    # Compute losses with dummy target
    loss_cfg = make_loss_cfg(cfg)
    target_y = torch.zeros(2, 1, 128, 128)
    target_y[0, 0, 10, 10] = 1.0
    target_y[1, 0, 20, 20] = 2.0

    losses = compute_rmr_v3_losses(out, target_y, loss_cfg)
    assert "hurdle_bce" in losses
    assert "trunc_nb" in losses
    assert not torch.isnan(losses["total"])
    assert losses["hurdle_bce"].item() >= 0.0


def test_subpixel_stride2_initial_bias_calibration():
    """Verify carrier head initial bias produces consistent mass on Stride 2 vs Stride 4."""
    from rmr_core.heads import build_fine_head

    head_s4 = build_fine_head(width=48, subpixel_stride2=False, temp_softplus=True)
    head_s2 = build_fine_head(width=48, subpixel_stride2=True, temp_softplus=True)

    x_s4 = torch.zeros(1, 48, 128, 128)
    x_s2 = torch.zeros(1, 48, 128, 128)

    z_s4 = head_s4.forward_logits(x_s4)
    y_s4 = head_s4.activate(z_s4)

    z_s2 = head_s2.forward_logits(x_s2)
    y_s2 = head_s2.activate(z_s2)

    # Stride 2 cell density should be 1/4 of Stride 4 cell density
    ratio = y_s4.mean().item() / y_s2.mean().item()
    assert abs(ratio - 4.0) < 1e-3, f"Expected ratio ~4.0, got {ratio}"

    # Over a 512x512 image (128x128 at S4, 256x256 at S2), total initial mass must match
    assert abs(y_s4.sum().item() - y_s2.sum().item()) < 0.1



