"""Unit and invariant test suite for First-Principles Residual Dilated Fine Head.

Verifies:
1. Step-0 Parity: Exact calibrated initial rate preservation (m0 = 0.015763 count/cell).
2. Linear Gradient Highway: Unconditional gradient flow through linear identity shortcuts (x + F(x)).
3. Subpixel Spatial Continuity: 3x3 polyphase filter cross-cell receptive field overlap.
4. Parameter Ceiling Invariants: Track A <= 104,441 params, Track B <= 200,000 params.
5. 2D Spatial FiLM Modulation: Bounded spatial scale modulation with zero-initialization neutrality.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import pytest

from rmr_core.fine_head import DDRBlock, ResidualDilatedFineHead
from rmr_v3.model import RMRv3
from rmr_v3.model.config import RMRv3Config
from rmr_v3.losses import compute_rmr_v3_losses
from rmr_v3.losses.config import RMRv3LossConfig


def test_ddr_block_linear_shortcut():
    """Verify that DDRBlock implements a pure linear identity shortcut x + F(x)."""
    block = DDRBlock(channels=48, dilation=2)
    # Zero out pointwise conv to isolate identity shortcut
    nn.init.zeros_(block.pw.weight)
    x = torch.randn(2, 48, 32, 32)
    out = block(x)
    # With pw = 0, F(x) = 0, out must strictly equal x (pure linear shortcut without activation bending!)
    assert torch.allclose(out, x, atol=1e-5), "Linear identity shortcut failed"


def test_step_zero_parity():
    """Verify that at Step 0, output density strictly matches calibrated prior m0."""
    init_m0 = 0.015763
    init_bias = math.log(math.exp(init_m0) - 1.0)

    # Test Stride 2 Subpixel Head
    head_s2 = ResidualDilatedFineHead(
        width=48,
        num_scales=3,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
        dilations=(1, 2),
        use_film=True,
        init_bias=init_bias,
        temp_softplus=True,
    )
    # Zero input feature map
    f_zero = torch.zeros(1, 48, 64, 64)
    y_init_s2 = head_s2(f_zero)
    expected_s2 = init_m0 / 4.0
    actual_s2 = y_init_s2.mean().item()
    assert abs(actual_s2 - expected_s2) / expected_s2 < 0.01, (
        f"Step 0 Stride 2 rate deviation: expected {expected_s2:.6f}, got {actual_s2:.6f}"
    )

    # Test Stride 4 Standard Head
    head_s4 = ResidualDilatedFineHead(
        width=32,
        num_scales=3,
        subpixel_stride2=False,
        subpixel_kernel_size=3,
        dilations=(1, 2),
        use_film=True,
        init_bias=init_bias,
        temp_softplus=True,
    )
    f_zero_s4 = torch.zeros(1, 32, 64, 64)
    y_init_s4 = head_s4(f_zero_s4)
    expected_s4 = init_m0
    actual_s4 = y_init_s4.mean().item()
    assert abs(actual_s4 - expected_s4) / expected_s4 < 0.01, (
        f"Step 0 Stride 4 rate deviation: expected {expected_s4:.6f}, got {actual_s4:.6f}"
    )


def test_film_step_zero_spatial_neutrality():
    """Verify that 2D Spatial FiLM modulation is strictly neutral (gamma=1, beta=0) at initialization."""
    head = ResidualDilatedFineHead(
        width=48,
        num_scales=3,
        use_film=True,
        subpixel_stride2=True,
    )
    f = torch.randn(2, 48, 32, 32)
    # Case 1: scale_weights is None
    z_none = head.forward_logits(f, scale_weights=None)
    # Case 2: scale_weights is a 4D spatial simplex map [B, 3, H, W]
    sw_4d = torch.softmax(torch.randn(2, 3, 32, 32), dim=1)
    z_sw_4d = head.forward_logits(f, scale_weights=sw_4d)
    # Case 3: scale_weights is a 2D global vector [B, 3]
    sw_2d = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    z_sw_2d = head.forward_logits(f, scale_weights=sw_2d)

    # At Step 0, since film_conv is zero-initialized, all outputs must be identical!
    assert torch.allclose(z_none, z_sw_4d, atol=1e-5), "FiLM 4D Step-0 neutrality violated"
    assert torch.allclose(z_none, z_sw_2d, atol=1e-5), "FiLM 2D Step-0 neutrality violated"


def test_subpixel_spatial_receptive_field():
    """Verify that 3x3 pre-shuffle conv provides non-zero spatial cross-cell gradients."""
    head = ResidualDilatedFineHead(
        width=48,
        num_scales=3,
        subpixel_stride2=True,
        subpixel_kernel_size=3,
    )
    f = torch.zeros(1, 48, 16, 16, requires_grad=True)
    z = head.forward_logits(f)
    assert z.shape == (1, 1, 32, 32)

    # Perturb a single subpixel at (16, 16)
    loss = z[0, 0, 16, 16]
    loss.backward()

    # In a 3x3 conv with stride 1, the gradient on input f must be non-zero in a 3x3 spatial window
    # around the coarse cell (8, 8), NOT just at a single pixel!
    grad_spatial = f.grad[0].abs().sum(dim=0)  # [16, 16]
    active_cells = (grad_spatial > 1e-6).sum().item()
    assert active_cells >= 9, f"Expected 3x3 spatial support (>=9 cells), got {active_cells}"


def test_parameter_ceilings_track_a_and_track_b():
    """Verify that full models strictly satisfy Track A <= 104,441 and Track B <= 200,000."""
    # Track B
    cfg_b = RMRv3Config(
        feature_width=48,
        output_stride=2,
        subpixel_stride2=True,
        neck_type="aspp_lite",
        aspp_dilations=(2, 4, 8),
        region_head_hidden=96,
        head_type="residual_dilated",
        fine_head_dilations=(1, 2),
        subpixel_kernel_size=3,
        max_trainable_params=200000,
    )
    m_b = RMRv3(cfg_b)
    params_b = sum(p.numel() for p in m_b.parameters() if p.requires_grad)
    assert params_b <= 200000, f"Track B exceeded limit: {params_b} > 200,000"

    # Track A
    cfg_a = RMRv3Config(
        feature_width=32,
        output_stride=4,
        subpixel_stride2=False,
        head_type="residual_dilated",
        fine_head_dilations=(1, 2),
        subpixel_kernel_size=3,
        max_trainable_params=104441,
    )
    m_a = RMRv3(cfg_a)
    params_a = sum(p.numel() for p in m_a.parameters() if p.requires_grad)
    assert params_a <= 104441, f"Track A exceeded limit: {params_a} > 104,441"


def test_full_forward_backward_gradient_flow():
    """Verify end-to-end forward-backward pass and non-vanishing gradients across all layers."""
    cfg = RMRv3Config(
        feature_width=48,
        output_stride=2,
        subpixel_stride2=True,
        neck_type="aspp_lite",
        aspp_dilations=(2, 4, 8),
        region_head_hidden=96,
        head_type="residual_dilated",
        fine_head_dilations=(1, 2),
        subpixel_kernel_size=3,
        max_trainable_params=200000,
    )
    model = RMRv3(cfg)

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="dual_bayesian_dm16",
        lambda_bayesian=1.0,
        lambda_flat_dm16=0.5,
        lambda_count=1.0,
        count_loss_mode="l1",
        lambda_region_nb=0.5,
    )

    x = torch.randn(2, 3, 256, 256)
    pts = [
        torch.tensor([[50.0, 50.0], [120.0, 140.0]]),
        torch.tensor([[80.0, 90.0]]),
    ]
    target_y = torch.zeros(2, 1, 128, 128)
    for b_idx, p in enumerate(pts):
        for pt in p:
            target_y[b_idx, 0, int(pt[1] / 2), int(pt[0] / 2)] = 1.0

    out = model(x)
    assert out.y.shape == (2, 1, 128, 128)
    assert out.y0.shape == (2, 1, 128, 128)
    assert (out.y >= 0.0).all()

    losses = compute_rmr_v3_losses(out, target_y=target_y, cfg=loss_cfg, points=pts)
    loss = losses["total"]
    loss.backward()

    # Inspect gradients on fine head
    fine_grads = [p.grad for p in model.fine_head.parameters() if p.grad is not None]
    assert len(fine_grads) > 0, "No gradients on fine head"
    assert all(torch.isfinite(g).all() for g in fine_grads), "Non-finite gradients"
    assert all(g.abs().max() > 0 for g in fine_grads), "Zero gradients on fine head layer"
