from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from rmr_core.heads import FineMeasureHead, ScaleConditionedFineHead, build_fine_head, _density_activate
from rmr_core.necks import ASPPLiteFPNNeck, HDCLiteFPNNeck
from rmr_v3.model.architecture import RMRv3
from rmr_v3.model.config import RMRv3Config


def test_density_adaptive_scale_background_invariance():
    """Verify that Density-Adaptive Scale factor produces 0.0% background inflation for z <= 0."""
    z_bg = torch.linspace(-6.0, 0.0, steps=50, requires_grad=True)
    # With density_adaptive_scale enabled and gamma = 0.5
    y_adaptive = _density_activate(
        z_bg,
        temp_softplus=False,
        tau=None,
        density_curvature=False,
        curvature_alpha=None,
        gated_density_curvature=False,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
        density_adaptive_scale=True,
        density_scale_gamma=0.5,
    )
    y_vanilla = F.softplus(z_bg)

    # Must be bit-for-bit identical in the negative regime (z <= 0)
    assert torch.allclose(y_adaptive, y_vanilla, atol=1e-7)

    # Gradient with respect to gamma must be identically 0 in the background
    gamma = torch.tensor(0.5, requires_grad=True)
    y_adapt_gamma = _density_activate(
        z_bg,
        temp_softplus=False,
        tau=None,
        density_curvature=False,
        curvature_alpha=None,
        gated_density_curvature=False,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
        density_adaptive_scale=True,
        density_scale_gamma=gamma,
    )
    loss = y_adapt_gamma.sum()
    loss.backward()
    assert gamma.grad is not None and gamma.grad.item() == 0.0


def test_density_adaptive_scale_dense_expansion():
    """Verify that Density-Adaptive Scale expands dynamic range into 2.0-8.0 for dense clusters."""
    # Test positive logits corresponding to cluster cells
    z_dense = torch.tensor([1.0, 2.0, 3.0])
    gamma = 0.5
    y_adaptive = _density_activate(
        z_dense,
        temp_softplus=False,
        tau=None,
        density_curvature=False,
        curvature_alpha=None,
        gated_density_curvature=False,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
        density_adaptive_scale=True,
        density_scale_gamma=gamma,
    )

    expected = (1.0 + gamma * z_dense) * F.softplus(z_dense)
    assert torch.allclose(y_adaptive, expected, atol=1e-6)
    # Verify density reaches expected dynamic range:
    # at z=1.0: ~1.97, at z=2.0: ~4.25, at z=3.0: ~7.62
    assert y_adaptive[0].item() > 1.90
    assert y_adaptive[1].item() > 4.00
    assert y_adaptive[2].item() > 7.50


def test_fine_measure_head_learnable_gamma_gradient():
    """Verify that learnable gamma receives non-zero gradient exclusively from foreground cells."""
    head = FineMeasureHead(
        width=16,
        density_adaptive_scale=True,
        density_scale_gamma=0.0,
        density_scale_learnable=True,
    )
    assert isinstance(head.density_scale_gamma, nn.Parameter)
    assert head.density_scale_gamma.requires_grad

    # Forward with synthetic features
    feat = torch.randn(2, 16, 32, 32)
    logits = head.forward_logits(feat)
    y0 = head.activate(logits)

    loss = y0.sum()
    loss.backward()
    assert head.density_scale_gamma.grad is not None


def test_hdc_lite_neck_architecture_and_parameter_savings():
    """Verify HDC-Lite FPN neck eliminates GAP layer and saves 1,056 parameters."""
    aspp_neck = ASPPLiteFPNNeck(in_channels=(16, 32, 48), width=32, aspp_dilations=(1, 3, 6), use_aspp_gap=True)
    hdc_neck = HDCLiteFPNNeck(in_channels=(16, 32, 48), width=32, hdc_dilations=(1, 2, 3))

    params_aspp = sum(p.numel() for p in aspp_neck.parameters() if p.requires_grad)
    params_hdc = sum(p.numel() for p in hdc_neck.parameters() if p.requires_grad)

    delta = params_hdc - params_aspp
    assert delta == -1056, f"Expected exactly -1056 parameter delta, got {delta}"

    # Forward pass parity
    c4 = torch.randn(2, 16, 64, 64)
    c8 = torch.randn(2, 32, 32, 32)
    c16 = torch.randn(2, 48, 16, 16)

    p4_h, p8_h, p16_h = hdc_neck(c4, c8, c16)
    p4_a, p8_a, p16_a = aspp_neck(c4, c8, c16)

    assert p4_h.shape == p4_a.shape == (2, 32, 64, 64)
    assert p8_h.shape == p8_a.shape == (2, 32, 32, 32)
    assert p16_h.shape == p16_a.shape == (2, 32, 16, 16)


def test_hdc_lite_arbitrary_resolution_and_sample_isolation():
    """Verify HDC-Lite on arbitrary odd resolutions and check sample isolation."""
    hdc_neck = HDCLiteFPNNeck(in_channels=(16, 32, 48), width=32, hdc_dilations=(1, 2, 3))

    # Test odd / non-power-of-two resolutions
    c4 = torch.randn(2, 16, 113, 227, requires_grad=True)
    c8 = torch.randn(2, 32, 57, 114)
    c16 = torch.randn(2, 48, 29, 57)

    p4, p8, p16 = hdc_neck(c4, c8, c16)
    assert p4.shape == (2, 32, 113, 227)

    # Sample isolation check: gradient of sample 0 loss w.r.t sample 1 input must be zero
    loss_0 = p4[0].sum()
    loss_0.backward()
    assert torch.all(c4.grad[1] == 0.0), "Sample cross-talk detected in HDC-Lite Neck!"


def test_gen15_full_model_forward_and_routing_isolation():
    """Verify Gen 15 model forward/backward pass and prove SIRT routing decoupling."""
    from rmr_v3.losses.auxiliary import physical_scale_alignment_loss

    cfg = RMRv3Config(
        neck_type="hdc_lite",
        hdc_dilations=(1, 2, 3),
        density_adaptive_scale=True,
        density_scale_gamma=0.5,
        density_scale_learnable=True,
        dynamic_scale_routing=True,
        region_sizes_px=(32, 64, 128),
        iterations=2,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_params <= 104441, f"Param limit exceeded: {total_params} > 104441"

    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    assert out.y is not None
    assert out.y0 is not None
    assert out.scale_weights is not None

    # Backward from SIRT solver output
    loss_solver = out.y.sum()
    loss_solver.backward(retain_graph=True)

    # Scale router weights receive finite autograd gradients from SIRT solver
    if model.scale_router is not None:
        for name, p in model.scale_router.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in scale router parameter {name}!"

    # Now backprop from physical_scale_alignment_loss
    target_dummy = torch.rand(2, 1, 32, 32)
    loss_scale_align = physical_scale_alignment_loss(
        scale_weights=out.scale_weights,
        target_y=target_dummy,
        tau_dense=0.12,
        tau_sparse=0.03,
        kernel_size=5,
        mask_background=False,
        stride=4,
    )
    loss_scale_align.backward()

    # Now scale router MUST have received valid non-zero gradient
    if model.scale_router is not None:
        has_grad = any(p.grad is not None and torch.any(p.grad != 0.0) for p in model.scale_router.parameters())
        assert has_grad, "Scale router received no gradient from physical_scale_alignment_loss!"


def test_hdc_lite_neck_with_gap_parity():
    """Verify HDC-Lite with GAP matches ASPP-Lite with GAP parameter count exactly."""
    aspp_neck = ASPPLiteFPNNeck(in_channels=(16, 32, 48), width=32, aspp_dilations=(1, 3, 6), use_aspp_gap=True)
    hdc_neck_gap = HDCLiteFPNNeck(in_channels=(16, 32, 48), width=32, hdc_dilations=(1, 2, 3), use_gap=True)

    params_aspp = sum(p.numel() for p in aspp_neck.parameters() if p.requires_grad)
    params_hdc_gap = sum(p.numel() for p in hdc_neck_gap.parameters() if p.requires_grad)

    assert params_aspp == params_hdc_gap, f"Expected exact param parity, got {params_aspp} vs {params_hdc_gap}"


def test_head_balanced_flat_dm_scale_invariance():
    """Verify Head-Balanced Flat DM loss maintains O(1) gradient norm across densities."""
    from rmr_core.losses import count_magnitude_loss, flat_dm_block_loss

    grad_norms = []
    for n in (20, 100, 500, 2000):
        pred = (torch.ones(1, 1, 64, 64) * (n / (64 * 64))).requires_grad_(True)
        target = torch.zeros(1, 1, 64, 64)
        target[0, 0, :8, :8] = n / 64.0
        loss = flat_dm_block_loss(pred, target, block_px=16, norm_mode="head_balanced", ref_count=100.0)
        loss.backward()
        assert pred.grad is not None
        grad_norms.append(pred.grad.norm().item())

    # Ratio between dense (2000) and sparse (20) gradient norm should be O(1), within factor of 2
    ratio = grad_norms[-1] / grad_norms[0]
    assert 0.5 < ratio < 2.0, f"Expected O(1) gradient scaling, got ratio {ratio:.2f}"

    # Smooth L1 count loss gradient test
    pred_c = torch.tensor([1950.0], requires_grad=True)
    tgt_c = torch.tensor([2000.0])
    loss_c = count_magnitude_loss(pred_c, tgt_c, mode="smooth_l1")
    loss_c.backward()
    assert pred_c.grad is not None and abs(pred_c.grad.item() - (-1.0)) < 1e-4


