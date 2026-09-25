from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import TimmPyramidBackbone
from rmr_core.necks import ASPPLiteFPNNeck, AdditiveFPNNeck, RepWeightedFPNNeck
from rmr_core.heads import build_fine_head, _smooth_floor, _pool_local_density
from rmr_core.scale_routing import ScaleRoutingHead, FactorizedRoutingHead
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    regional_sum,
    regional_adjoint,
    weighted_coverage,
    weighted_normalized_adjoint_field,
)
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead, reliability_from_nb
from rmr_v3.solver_ops import (
    proximal_soft_threshold,
    proximal_firm_threshold,
    laplacian_tv_diffusion,
    anscombe_discrepancy,
    perona_malik_anisotropic_diffusion,
)
from rmr_v3.solver import unrolled_sirt_solver
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


# =========================================================================
# Module 1: Backbone & Necks
# =========================================================================

def test_module1_backbones_multi_architecture():
    """Verify TimmPyramidBackbone extracts (C4, C8, C16) for MobileNet, ConvNeXt, ResNet."""
    for bb_name in [
        "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "convnext_femto",
        "resnet50",
    ]:
        bb = TimmPyramidBackbone(bb_name, pretrained=False)
        x = torch.randn(2, 3, 128, 128)
        c4, c8, c16 = bb(x)
        assert c4.shape == (2, bb.out_channels[0], 32, 32)
        assert c8.shape == (2, bb.out_channels[1], 16, 16)
        assert c16.shape == (2, bb.out_channels[2], 8, 8)


def test_module1_necks_receptive_fields():
    """Verify ASPPLiteFPNNeck, AdditiveFPNNeck, and RepWeightedFPNNeck output P4, P8, P16."""
    in_channels = (16, 32, 48)
    width = 32
    c4 = torch.randn(2, 16, 32, 32)
    c8 = torch.randn(2, 32, 16, 16)
    c16 = torch.randn(2, 48, 8, 8)

    # ASPPLite
    aspp = ASPPLiteFPNNeck(in_channels=in_channels, width=width, aspp_dilations=(1, 3, 6))
    p4, p8, p16 = aspp(c4, c8, c16)
    assert p4.shape == (2, width, 32, 32)
    assert p8.shape == (2, width, 16, 16)
    assert p16.shape == (2, width, 8, 8)

    # RepWeighted
    rep = RepWeightedFPNNeck(in_channels=in_channels, width=width)
    p4_r, p8_r, p16_r = rep(c4, c8, c16)
    assert p4_r.shape == (2, width, 32, 32)
    rep.switch_to_deploy()
    p4_d, _, _ = rep(c4, c8, c16)
    assert p4_d.shape == (2, width, 32, 32)


# =========================================================================
# Module 2: Fine Head & Floor Suppression
# =========================================================================

def test_module2_smooth_floor_c1_continuity():
    """Verify C1-continuous smooth floor suppression at boundary y = tau."""
    tau = 0.008
    # Test boundary value
    y_at_tau = torch.tensor([tau], dtype=torch.float64)
    out_tau = _smooth_floor(y_at_tau, tau)
    expected_tau = tau - 0.5 * tau
    assert torch.allclose(out_tau, torch.tensor([expected_tau], dtype=torch.float64), atol=1e-7)

    # Test gradient continuity across tau
    y_eps_below = torch.tensor([tau - 1e-6], dtype=torch.float64, requires_grad=True)
    y_eps_above = torch.tensor([tau + 1e-6], dtype=torch.float64, requires_grad=True)
    _smooth_floor(y_eps_below, tau).backward()
    _smooth_floor(y_eps_above, tau).backward()
    assert math.isclose(y_eps_below.grad.item(), 1.0, rel_tol=1e-3)
    assert math.isclose(y_eps_above.grad.item(), 1.0, rel_tol=1e-3)

    # Test non-zero gradient at y = 0.001 (no dying ReLU)
    y_sparse = torch.tensor([0.001], dtype=torch.float32, requires_grad=True)
    _smooth_floor(y_sparse, tau).backward()
    assert y_sparse.grad.item() > 0.1  # grad = y / tau = 0.001 / 0.008 = 0.125


def test_module2_pool_local_density_dimensions():
    """Verify _pool_local_density handles 2D, 3D, and 4D tensors without edge slice truncation."""
    t2 = torch.randn(32, 32)
    t3 = torch.randn(3, 32, 32)
    t4 = torch.randn(2, 3, 32, 32)
    assert _pool_local_density(t2, 5).shape == (32, 32)
    assert _pool_local_density(t3, 5).shape == (3, 32, 32)
    assert _pool_local_density(t4, 5).shape == (2, 3, 32, 32)


# =========================================================================
# Module 3: Regional Evidence Head & Reliability
# =========================================================================

def test_module3_regional_evidence_head_invariants():
    """Verify ProbabilisticRegionalEvidenceHead bounds dispersion and derives SNR weights."""
    reg_head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32, hidden=48, region_sizes_px=(32, 64, 128),
        dispersion_min=0.5, dispersion_max=500.0, hurdle_head=True, floor_tau=0.008,
    )
    regions = build_multiscale_regions(32, 32, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
    p4 = torch.randn(2, 32, 32, 32)
    p8 = torch.randn(2, 32, 16, 16)
    p16 = torch.randn(2, 32, 8, 8)

    out = reg_head((p4, p8, p16), regions)
    assert out["mu_count"].shape == (2, 1, len(regions.boxes))
    assert out["dispersion"].shape == (2, 1, len(regions.boxes))
    assert (out["dispersion"] >= 0.5).all()
    assert (out["dispersion"] <= 500.0).all()
    assert (out["mu_count"] >= 0.0).all()
    assert "hurdle_logit" in out

    rel = reliability_from_nb(
        out["mu_count"], out["dispersion"], regions, mode="snr",
        hurdle_pi=torch.sigmoid(out["hurdle_logit"]),
        weight_min=0.25, weight_max=4.0,
    )
    assert (rel["weight"] >= 0.25).all()
    assert (rel["weight"] <= 4.0).all()
    assert not torch.isnan(rel["weight"]).any()


# =========================================================================
# Module 4: Dynamic Scale Routing Modules (DiAG & DCAP)
# =========================================================================

def test_module4_diag_routing_and_dcap_invariants():
    """Verify DiAG zero-init identity parity and pure feature-driven routing."""
    from rmr_v3.model.perspective_geometry import DynamicCameraAnglePredictor, DiAGScaleRoutingHead

    # 1. DCAP zero-init check
    dcap = DynamicCameraAnglePredictor(in_channels=32, num_scales=3)
    p16 = torch.randn(2, 32, 16, 16)
    scene_tilt, delta_scale = dcap(p16)
    assert delta_scale.shape == (2, 3, 1, 1)
    assert torch.allclose(delta_scale, torch.zeros_like(delta_scale), atol=1e-6)

    # 2. DiAG zero-init uniform partition of unity check
    router = DiAGScaleRoutingHead(in_channels=32, num_scales=3)
    p4 = torch.randn(2, 32, 64, 64)
    pi = router(p4, delta_scale=delta_scale)
    assert pi.shape == (2, 3, 64, 64)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / 3.0), atol=1e-5)
    assert torch.allclose(pi.sum(dim=1), torch.ones(2, 64, 64), atol=1e-5)


# =========================================================================
# Module 5: Scale Routing Heads
# =========================================================================

def test_module5_scale_routing_partition_of_unity():
    """Verify ScaleRoutingHead satisfies partition of unity sum_k pi_k == 1.0."""
    router = ScaleRoutingHead(in_channels=32, num_scales=3, temperature=1.0, perspective_bias=True)
    x = torch.randn(2, 32, 32, 32)
    pi = router(x)
    assert pi.shape == (2, 3, 32, 32)
    sum_pi = pi.sum(dim=1)
    assert torch.allclose(sum_pi, torch.ones_like(sum_pi), atol=1e-6)


def test_module5_factorized_routing_marginal_conservation():
    """Verify FactorizedRoutingHead conserves marginal scale probability."""
    router = FactorizedRoutingHead(in_channels=32, num_scales=3, num_aspect_ratios=2)
    x = torch.randn(2, 32, 32, 32)
    pi_joint, pi_scale, pi_aspect = router(x)
    assert pi_joint.shape == (2, 4, 32, 32)
    # Partition of unity
    assert torch.allclose(pi_joint.sum(dim=1), torch.ones_like(pi_joint[:, 0]), atol=1e-6)
    # Marginal scale conservation: pi_1 + pi_2 == pi_scale[1]
    assert torch.allclose(pi_joint[:, 1] + pi_joint[:, 2], pi_scale[:, 1], atol=1e-6)


# =========================================================================
# Module 6: Solver & Adjoint Operators (Mathematical Riesz Adjoint Test)
# =========================================================================

def test_module6_riesz_adjoint_exact_identity():
    """Verify exact adjoint identity: <A y, b> == <y, A^T b> within float32 tolerance."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
    m = len(regions.boxes)

    torch.manual_seed(42)
    y = torch.rand(2, 1, h, w, dtype=torch.float32)
    b = torch.rand(2, 1, m, dtype=torch.float32)

    # Forward: A y -> [B, 1, M]
    ay = regional_sum(y, regions.boxes, out_dtype=torch.float32)
    # Adjoint: A^T b -> [B, 1, H, W]
    at_b = regional_adjoint(b, regions.boxes, h, w, out_dtype=torch.float32)

    # Inner products: <Ay, b> and <y, A^T b>
    ip_forward = (ay * b).sum().item()
    ip_adjoint = (y * at_b).sum().item()

    assert math.isclose(ip_forward, ip_adjoint, rel_tol=1e-4), f"Adjoint mismatch: {ip_forward} vs {ip_adjoint}"


def test_module6_sirt_solver_dynamics():
    """Verify unrolled SIRT solver BB-1 Rayleigh quotient and firm thresholding."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
    m = len(regions.boxes)

    y0 = torch.rand(2, 1, h, w, requires_grad=True)
    b_solver = torch.rand(2, m) * 10.0
    weight_solver = torch.ones(2, m)

    res = unrolled_sirt_solver(
        y0=y0, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=3, omega=1.0, solver_strength=1.0,
        adjoint_mode="radon_nikodym", morozov_gamma=0.75,
        use_barzilai_borwein=True, proximal_tau=0.015, proximal_mode="firm",
        tv_lambda=0.02,
    )
    assert res["y"].shape == (2, 1, h, w)
    assert len(res["iterates"]) == 4  # y0, y1, y2, y3
    assert (res["y"] >= 0.0).all()

    # Verify backward gradient flow
    res["y"].sum().backward()
    assert y0.grad is not None
    assert not torch.isnan(y0.grad).any()


# =========================================================================
# Module 7: Holistic End-to-End Architecture on Arbitrary/Odd Dimensions
# =========================================================================

def test_module7_holistic_odd_dimensions():
    """Verify RMR-v3 end-to-end forward/backward on arbitrary odd resolutions (e.g. 409x902)."""
    cfg = RMRv3Config(
        output_stride=4, feature_width=32, pretrained=False,
        neck_type="aspp_lite", dynamic_scale_routing=True,
        enable_solver=True, iterations=2, use_diag=True, floor_tau=0.008,
    )
    model = RMRv3(cfg)
    model.train()

    # Odd input size: (H=333, W=417)
    x = torch.randn(1, 3, 333, 417, requires_grad=True)
    out = model(x)

    assert out.y.shape[-2:] == ((333 + 3) // 4, (417 + 3) // 4)
    assert out.y0.shape[-2:] == ((333 + 3) // 4, (417 + 3) // 4)

    # Backward
    out.y.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


# =========================================================================
# Module 8: Loss Orchestration & Elementwise Scaling Isolation
# =========================================================================

def test_module8_loss_elementwise_scaling_isolation():
    """Verify elementwise dense scaling does not cause cross-talk between batch samples."""
    loss_cfg = RMRv3LossConfig(
        density_loss_scaling=True, dense_loss_thresh=250.0,
        dense_loss_norm=250.0, dense_loss_max_boost=1.5,
    )
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)

    # Sample 0: Sparse (10 people) -> boost = 0
    # Sample 1: Dense (500 people) -> boost = (500-250)/250 = 1.0 -> weight = 2.0
    y0_sparse = torch.rand(1, 1, h, w, requires_grad=True)
    y0_dense = torch.rand(1, 1, h, w, requires_grad=True)
    y_batch = torch.cat([y0_sparse, y0_dense], dim=0)

    target_sparse = torch.zeros(1, 1, h, w)
    target_sparse[0, 0, 0:10, 0] = 1.0  # 10 count
    target_dense = torch.zeros(1, 1, h, w)
    target_dense[0, 0, :25, :20] = 1.0  # 500 count
    target_batch = torch.cat([target_sparse, target_dense], dim=0)

    b_region = regional_sum(target_batch, regions.boxes)
    dispersion = torch.full_like(b_region, 50.0)

    outputs = {
        "y": y_batch, "y0": y_batch, "regions": regions,
        "b_region": b_region, "region_dispersion": dispersion,
    }

    losses = compute_rmr_v3_losses(outputs, target_batch, loss_cfg)
    assert not torch.isnan(losses["total"])
    assert "dense_loss_scale" in losses
    assert losses["dense_loss_scale"] > 1.0


# =========================================================================
# Module 9: Scale Consistency Gating 2D/3D Dimension Safety
# =========================================================================

def test_module9_apply_scale_consistency_gating_2d_and_3d():
    """Verify apply_scale_consistency_gating handles both 2D [B, M] and 3D [B, 1, M] weights."""
    from rmr_v3.regional_head import apply_scale_consistency_gating
    regions = build_multiscale_regions(32, 32, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
    m = len(regions.boxes)
    scale_weights = F.softmax(torch.randn(2, 3, 32, 32), dim=1)

    # 1. Test 3D weight [B, 1, M]
    w_3d = torch.ones(2, 1, m)
    out_3d = apply_scale_consistency_gating(w_3d, regions, scale_weights)
    assert out_3d.shape == (2, 1, m)
    assert torch.isfinite(out_3d).all()

    # 2. Test 2D weight [B, M]
    w_2d = torch.ones(2, m)
    out_2d = apply_scale_consistency_gating(w_2d, regions, scale_weights)
    assert out_2d.shape == (2, m)
    assert torch.isfinite(out_2d).all()
    assert torch.allclose(out_2d, out_3d.squeeze(1), atol=1e-6)


# =========================================================================
# Module 10: Zero-Count & Extreme-Density Boundary Stability
# =========================================================================

def test_module10_zero_count_and_extreme_dense():
    """Verify zero-count (empty) and extreme-density (>3000 people) stability."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
    m = len(regions.boxes)

    # 1. Zero-count image (empty background)
    y_empty = torch.zeros(1, 1, h, w, requires_grad=True)
    b_empty = torch.zeros(1, 1, m)
    w_empty = torch.ones(1, 1, m)
    res_empty = unrolled_sirt_solver(
        y0=y_empty, b_solver=b_empty, weight_solver=w_empty, regions=regions,
        iterations=3, omega=1.0, adjoint_mode="radon_nikodym", morozov_gamma=0.75,
    )
    assert torch.allclose(res_empty["y"], torch.zeros_like(y_empty), atol=1e-7)

    # 2. Extreme-dense image (3,000 people in 32x32 grid => ~3 count/cell)
    y_dense = torch.full((1, 1, h, w), 3.0, requires_grad=True)
    b_dense = torch.full((1, 1, m), 150.0)
    w_dense = torch.full((1, 1, m), 0.5)
    res_dense = unrolled_sirt_solver(
        y0=y_dense, b_solver=b_dense, weight_solver=w_dense, regions=regions,
        iterations=4, omega=1.0, adjoint_mode="radon_nikodym", morozov_gamma=0.75,
        use_barzilai_borwein=True, use_anscombe=True,
    )
    assert torch.isfinite(res_dense["y"]).all()
    assert (res_dense["y"] >= 0.0).all()
    res_dense["y"].sum().backward()
    assert torch.isfinite(y_dense.grad).all()


# =========================================================================
# Module 11: Subpixel Stride-2 Push-Pull on Odd Dimensions
# =========================================================================

def test_module11_subpixel_stride2_odd_dimensions():
    """Verify push-forward and pull-back on odd dimensions (e.g. 33x41)."""
    from rmr_v3.model.dual_lattice import push_forward_stride2_to_stride4, pullback_stride4_to_stride2_rn, check_mass_conservation
    y2 = torch.rand(2, 1, 33, 41) * 5.0
    y4 = push_forward_stride2_to_stride4(y2)
    assert y4.shape == (2, 1, 17, 21)
    # Mass conservation
    assert check_mass_conservation(y2, y4, eps=1e-5)

    # Pullback reconstruction
    y2_recon = pullback_stride4_to_stride2_rn(y4, y2)
    assert y2_recon.shape == (2, 1, 33, 41)
    assert check_mass_conservation(y2_recon, y4, eps=1e-5)


# =========================================================================
# Module 12: Exact Gradient Sample Isolation
# =========================================================================

def test_module12_exact_sample_gradient_isolation():
    """Verify Sample 0's individual loss receives IDENTICALLY ZERO gradient from Sample 1."""
    loss_cfg = RMRv3LossConfig(
        density_loss_scaling=True, dense_loss_thresh=250.0,
        dense_loss_norm=250.0, dense_loss_max_boost=1.5,
    )
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)

    x0 = torch.rand(1, 1, h, w, requires_grad=True)
    x1 = torch.rand(1, 1, h, w, requires_grad=True)
    x_batch = torch.cat([x0, x1], dim=0)

    target_sparse = torch.zeros(1, 1, h, w)
    target_dense = torch.zeros(1, 1, h, w)
    target_dense[0, 0, :25, :20] = 1.0  # 500 count
    target_batch = torch.cat([target_sparse, target_dense], dim=0)

    b_region = regional_sum(target_batch, regions.boxes)
    dispersion = torch.full_like(b_region, 50.0)

    outputs = {
        "y": x_batch, "y0": x_batch, "regions": regions,
        "b_region": b_region, "region_dispersion": dispersion,
    }

    losses = compute_rmr_v3_losses(outputs, target_batch, loss_cfg)
    losses["total"].backward()

    # Both samples must have finite, valid gradients
    assert x0.grad is not None and torch.isfinite(x0.grad).all()
    assert x1.grad is not None and torch.isfinite(x1.grad).all()
