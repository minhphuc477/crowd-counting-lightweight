from __future__ import annotations

import torch
import pytest

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    closed_form_radon_nikodym_transport,
    regional_sum,
)
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses.config import RMRv3LossConfig
from rmr_v3.losses.orchestration import compute_rmr_v3_losses


def test_closed_form_rn_operator_properties():
    """Verify core mathematical invariants of the closed-form Radon-Nikodym operator."""
    b, h, w = 2, 64, 64
    device = torch.device("cpu")
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32, 64),
        overlap=0.5,
        device=device,
    )

    torch.manual_seed(42)
    # 1. Dirac-like sparse carrier with isolated peaks and zeros
    y0 = torch.zeros((b, 1, h, w), dtype=torch.float32)
    y0[:, :, 10, 10] = 1.0
    y0[:, :, 25, 30] = 2.0
    y0[:, :, 50, 45] = 1.5

    # Simulated regional target
    b_hat = regional_sum(y0, regions.boxes)
    b_solver = b_hat * 1.5  # 50% undercount scale correction

    y_star = closed_form_radon_nikodym_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=None,
        regions=regions,
        eps=1e-6,
    )

    # Invariant A: Support preservation (y_star == 0 wherever y0 == 0)
    zero_mask = y0 == 0.0
    assert torch.all(y_star[zero_mask] == 0.0), "Violated support preservation: mass leaked to zero background!"

    # Invariant B: Non-negativity
    assert torch.all(y_star >= 0.0), "Violated non-negativity: negative values produced!"

    # Invariant C: Scale amplification on peaks
    peak_mask = y0 > 0.0
    assert torch.all(y_star[peak_mask] > y0[peak_mask]), "Scale correction did not boost underestimated peaks!"


def test_closed_form_rn_model_forward_backward():
    """Verify RMRv3 end-to-end forward and backward passes with closed_form_rn."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        solver_mode="closed_form_rn",
        enable_solver=True,
        head_blocks=1,
    )
    model = RMRv3(cfg)
    model.train()

    x = torch.randn(2, 3, 128, 128, requires_grad=False)
    out = model(x)

    assert "y" in out
    assert "y0" in out
    assert out.y.shape == out.y0.shape
    assert len(out.iterates) == 2

    loss_cfg = RMRv3LossConfig(
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_strauss=0.1,
    )
    target_y = torch.zeros(2, 1, 32, 32, dtype=torch.float32)
    target_y[0, 0, 5, 5] = 1.0
    target_y[0, 0, 10, 15] = 1.0
    target_y[1, 0, 2, 3] = 1.0
    pts = [torch.tensor([[20.0, 30.0], [50.0, 70.0]]), torch.tensor([[10.0, 15.0]])]
    losses = compute_rmr_v3_losses(out, target_y, loss_cfg, points=pts)

    losses["total"].backward()

    # Verify gradients reach backbone and head
    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)), "Non-finite gradient detected!"
            has_grad = True
    assert has_grad, "Zero gradients backpropagated!"


def test_direct_mode_model_forward():
    """Verify RMRv3 forward pass in pure direct mode without solver overhead."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        solver_mode="direct",
        enable_solver=False,
    )
    model = RMRv3(cfg)
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)

    assert torch.allclose(out.y, out.y0), "Direct mode output y must be identically equal to y0!"
    assert out.solver_strength == 0.0


def test_subpixel_stride2_deep_head():
    """Verify Subpixel Stride 2 mode with deeper head blocks."""
    cfg = RMRv3Config(
        output_stride=2,
        subpixel_stride2=True,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        solver_mode="closed_form_rn",
        head_blocks=2,
        max_trainable_params=200000,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 200000, f"Track B budget exceeded: {total_params} > 200000"

    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.y.shape[-2:] == (64, 64), f"Expected stride 2 output (64, 64), got {out.y.shape[-2:]}"


def test_batch_independence_and_odd_resolution():
    """Verify strict sample isolation and prime resolution robustness."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=False,
        solver_mode="closed_form_rn",
    )
    model = RMRv3(cfg)
    model.eval()

    torch.manual_seed(123)
    x1 = torch.randn(1, 3, 113, 227)
    x2 = torch.randn(1, 3, 113, 227)
    x_batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out_single = model(x1)
        out_batch = model(x_batch)

    # Batch independence
    diff = (out_batch.y[0:1] - out_single.y).abs().max().item()
    assert diff < 1e-4, f"Batch independence violation: max diff = {diff}"

    # Finite tensors on prime dimensions
    assert torch.all(torch.isfinite(out_single.y)), "Non-finite output on prime resolution!"


def test_closed_form_rn_scale_routing_consistency():
    """Verify that scale_weights does not distort mass when ratios are 1.0."""
    h, w = 64, 64
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32, 64),
    )
    y0 = torch.ones((1, 1, h, w), dtype=torch.float32) * 0.05
    b_exact = regional_sum(y0, regions.boxes)

    # Dynamic non-uniform scale routing weights
    raw = torch.rand(1, 2, h, w)
    scale_weights = torch.softmax(raw, dim=1)

    y_star = closed_form_radon_nikodym_transport(
        y0=y0,
        b_solver=b_exact,
        weight_solver=None,
        regions=regions,
        scale_weights=scale_weights,
    )

    # When regional targets match exactly, y_star must equal y0
    diff = (y_star - y0).abs().max().item()
    assert diff < 1e-5, f"Scale routing caused mass distortion: max diff = {diff}"


def test_asymmetric_trust_region_damping():
    """Verify mathematical properties of asymmetric trust region damping."""
    h, w = 32, 32
    regions = build_multiscale_regions(
        height=h,
        width=w,
        output_stride=4,
        region_sizes_px=(32,),
    )
    y0 = torch.full((1, 1, h, w), 0.1, dtype=torch.float32)
    b_exact = regional_sum(y0, regions.boxes)

    # Test 1: Symmetric vs Asymmetric on 3x Undercount
    b_undercount = b_exact * 3.0
    _, s_sym = closed_form_radon_nikodym_transport(
        y0=y0, b_solver=b_undercount, weight_solver=None, regions=regions,
        trust_alpha=0.12, trust_alpha_up=0.12, return_multiplier=True,
    )
    _, s_asym = closed_form_radon_nikodym_transport(
        y0=y0, b_solver=b_undercount, weight_solver=None, regions=regions,
        trust_alpha=0.12, trust_alpha_up=0.35, return_multiplier=True,
    )
    # Symmetric is capped at 1.12
    assert s_sym.max().item() == pytest.approx(1.12, abs=1e-3)
    # Asymmetric expands up to 1.35
    assert s_asym.max().item() == pytest.approx(1.35, abs=1e-3)

    # Test 2: Downward suppression remains identically 0.88 for both
    b_zero = torch.zeros_like(b_exact)
    _, s_down_sym = closed_form_radon_nikodym_transport(
        y0=y0, b_solver=b_zero, weight_solver=None, regions=regions,
        trust_alpha=0.12, trust_alpha_up=0.12, return_multiplier=True,
    )
    _, s_down_asym = closed_form_radon_nikodym_transport(
        y0=y0, b_solver=b_zero, weight_solver=None, regions=regions,
        trust_alpha=0.12, trust_alpha_up=0.35, return_multiplier=True,
    )
    assert s_down_sym.min().item() == pytest.approx(0.88, abs=1e-3)
    assert s_down_asym.min().item() == pytest.approx(0.88, abs=1e-3)



