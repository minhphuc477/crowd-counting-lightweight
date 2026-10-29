import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    closed_form_dsr_fw_transport,
)


def test_support_invariance_on_background():
    """Verify that flat background receives strictly zero mass injection."""
    h, w = 32, 32
    y0 = torch.zeros(1, 1, h, w)
    z0 = torch.full((1, 1, h, w), -10.0)
    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 5.0)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
    )
    assert (y_star == 0.0).all(), "Pure background support must remain strictly zero"


def test_bounded_peak_calibration_no_bloat():
    """Verify that a peak already >= 1.0 does not bloat into multi-person masses."""
    h, w = 32, 32
    z0 = torch.full((1, 1, h, w), -6.0)
    # Head 1 is already 1.70 people (logit 1.5)
    z0[0, 0, 10, 10] = 1.5
    # Head 2 is under-predicted 0.47 people (logit -0.5)
    z0[0, 0, 10, 20] = -0.5
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    # Demand 4.0 people (deficit exists)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 4.0)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
        unit_mass_capacity=True,
    )

    # Head 1 should not bloat beyond its original value
    assert y_star[0, 0, 10, 10] <= y0[0, 0, 10, 10] + 0.05, "Fully formed peak must not bloat"
    # Head 2 should be boosted towards 1.0
    assert y_star[0, 0, 10, 20] > y0[0, 0, 10, 20], "Under-predicted peak must be calibrated upwards"


def test_missed_head_recovery():
    """Verify that a suppressed head (weak logit) is recovered by Stream B."""
    h, w = 32, 32
    z0 = torch.full((1, 1, h, w), -6.0)
    # 1 detected head
    z0[0, 0, 8, 8] = 1.5
    # 1 suppressed head (logit -2.0 -> y0 = 0.12)
    z0[0, 0, 24, 24] = -2.0
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 3.0)

    y_star, details = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
        return_details=True,
    )

    # Suppressed head at (24, 24) must receive significant mass
    assert y_star[0, 0, 24, 24] > y0[0, 0, 24, 24] * 1.5, "Suppressed head must receive spike injection"
    # Valley at (16, 16) must not be polluted
    assert y_star[0, 0, 16, 16] < 0.01, "Valley must not receive sương mù"


def test_surplus_contraction():
    """Verify that when b_solver < y0 (over-prediction), peaks are contracted."""
    h, w = 32, 32
    z0 = torch.full((1, 1, h, w), -6.0)
    z0[0, 0, 16, 16] = 2.0  # y0 ~ 2.12
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    # Target only 0.5 people (surplus of ~1.6 people)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 0.5)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
    )

    # Peak must be contracted downwards
    assert y_star[0, 0, 16, 16] < y0[0, 0, 16, 16], "Peak must be scaled down under surplus"


def test_autograd_differentiability():
    """Verify clean backward pass without NaNs or broken computation graph."""
    h, w = 32, 32
    z0 = torch.full((1, 1, h, w), -4.0, requires_grad=True)
    y0 = F.softplus(z0)
    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 3.0, requires_grad=True)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
    )

    loss = y_star.sum()
    loss.backward()

    assert z0.grad is not None, "Gradient must flow to z0"
    assert not torch.isnan(z0.grad).any(), "Gradient must be finite"
    assert not torch.isnan(b_s.grad).any(), "Gradient must be finite"


def test_ghost_coordinate_rejection():
    """Verify that 2-angle Cartesian ghost coordinates receive identically zero mass."""
    h, w = 48, 48
    z0 = torch.full((1, 1, h, w), -8.0)
    # Head 1 at (15, 15)
    z0[0, 0, 15, 15] = 2.0
    # Head 2 at (30, 35)
    z0[0, 0, 30, 35] = 1.8
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 4.0)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
    )

    # Real heads receive mass
    assert y_star[0, 0, 15, 15] > 0.5
    assert y_star[0, 0, 30, 35] > 0.5
    # Ghost coordinates MUST receive identically zero injected mass (injected_mass == 0.0)
    injected_ghost_1 = (y_star[0, 0, 15, 35] - y0[0, 0, 15, 35]).clamp_min(0.0).item()
    injected_ghost_2 = (y_star[0, 0, 30, 15] - y0[0, 0, 30, 15]).clamp_min(0.0).item()
    assert injected_ghost_1 == 0.0, f"Ghost coordinate (15, 35) received injected mass: {injected_ghost_1}"
    assert injected_ghost_2 == 0.0, f"Ghost coordinate (30, 15) received injected mass: {injected_ghost_2}"


def test_prime_dimensions_and_fp16():
    """Verify numerical stability on prime dimensions and half precision."""
    h, w = 31, 47
    z0 = torch.full((1, 1, h, w), -5.0, dtype=torch.float16)
    z0[0, 0, 11, 23] = 1.0
    y0 = F.softplus(z0.float()).half()

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 2.0, dtype=torch.float16)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
    )

    assert y_star.shape == (1, 1, h, w)
    assert y_star.dtype == torch.float16
    assert not torch.isnan(y_star).any()


def test_multiscale_scale_weights_routing():
    """Verify dynamic scale routing with K=3 scale partitions."""
    h, w = 32, 32
    z0 = torch.full((2, 1, h, w), -5.0)
    z0[:, :, 10, 10] = 1.5
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((2, 1, regions.boxes.shape[0]), 3.0)
    scale_weights = torch.softmax(torch.randn(2, 3, h, w), dim=1)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
        scale_weights=scale_weights,
    )

    assert y_star.shape == (2, 1, h, w)
    assert not torch.isnan(y_star).any()
    assert (y_star >= 0.0).all()


def test_rmrv3_end_to_end_dsr_fw_pipeline():
    """Verify that RMRv3 model forward and backward passes execute cleanly with dsr_fw."""
    from rmr_v3.model import RMRv3
    from rmr_v3.model.config import RMRv3Config

    cfg = RMRv3Config()
    cfg.solver_mode = "dsr_fw"
    cfg.enable_solver = True
    cfg.output_stride = 2
    cfg.subpixel_stride2 = True

    model = RMRv3(cfg)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)

    assert out.y is not None
    assert out.y.shape == (2, 1, 32, 32)
    loss = out.y.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "Backbone and heads must receive gradients"


def test_flat_plateau_zero_peak_leakage():
    """Verify that a flat plateau of fog/pedestal has 0 detected peaks and receives 0 bloat."""
    h, w = 32, 32
    # Flat plateau with positive pedestal y0 = 0.20 (fog / clutter)
    y0 = torch.full((1, 1, h, w), 0.20)
    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 0.0)  # Demand 0 count (pure background)

    y_star, details = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        return_details=True,
    )

    # Flat plateau must NOT produce isolated peak flags
    p_peak = details["p_peak"]
    assert (p_peak == 0.0).all(), "Flat plateau must have identically 0 peak prominence"
    # Surplus contraction should scale down the flat plateau uniformly without creating spikes
    assert (y_star <= y0).all(), "Surplus contraction must not increase any pixel on flat plateau"


def test_empty_scene_zero_hallucination_under_false_alarm():
    """Verify that an empty scene with false regional alarm injects 0 phantom mass into noise."""
    h, w = 32, 32
    torch.manual_seed(42)
    # Background noise z0 ~ -6.0 (y0 < 0.006)
    z0 = -6.0 + 0.3 * torch.randn(1, 1, h, w)
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    # Regional head has false alarm predicting 2.0 people in every box
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 2.0)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
        tau_z_floor=-3.5,
    )

    # Injected mass into background noise must be strictly zero
    injected_mass = (y_star - y0).clamp_min(0.0).sum().item()
    assert injected_mass == 0.0, f"Expected 0 injected phantom mass, got {injected_mass}"


def test_super_spike_ceiling():
    """Verify that an extreme deficit box with only 1 faint head is bounded by max_spike_mass."""
    h, w = 32, 32
    torch.manual_seed(42)
    z0 = torch.full((1, 1, h, w), -6.0)
    # Faint head at (16, 16) with z0 = -2.0 (y0 = 0.126)
    z0[0, 0, 16, 16] = -2.0
    y0 = F.softplus(z0)

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    # Massive regional deficit: 10 people missing in the region
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 10.0)

    y_star = closed_form_dsr_fw_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        z0=z0,
        tau_z_floor=-3.5,
        max_spike_mass=1.5,
    )

    peak_val = y_star[0, 0, 16, 16].item()
    assert peak_val <= 1.5 + y0[0, 0, 16, 16].item() + 1e-4, f"Spike exceeded max_spike_mass: {peak_val}"


def test_footprint_capacity_broad_head_no_bloat():
    """Verify that a broad-shouldered head with footprint mass = 1.0 receives 0 bloat."""
    h, w = 16, 16
    y0 = torch.zeros(1, 1, h, w)
    # Head with total mass = 1.00 (apex 0.32 + 4 neighbors 0.17)
    y0[0, 0, 8, 8] = 0.32
    y0[0, 0, 7, 8] = 0.17
    y0[0, 0, 9, 8] = 0.17
    y0[0, 0, 8, 7] = 0.17
    y0[0, 0, 8, 9] = 0.17
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.float32)
    b_s = torch.tensor([[[2.0]]])  # Regional deficit exists

    y_star = closed_form_dsr_fw_transport(y0, b_s, None, boxes, unit_mass_capacity=True)
    apex_after = y_star[0, 0, 8, 8].item()
    assert abs(apex_after - 0.32) < 1e-4, f"Complete head bloated: {apex_after}"


def test_hierarchical_fog_first_surplus_contraction():
    """Verify that background fog is aggressively purged while real heads are preserved."""
    y0 = torch.full((1, 1, 32, 32), 0.02)
    # 2 real heads with apex 0.80
    y0[0, 0, 10, 10] = 0.80
    y0[0, 0, 20, 20] = 0.80
    boxes = torch.tensor([[0, 0, 32, 32]], dtype=torch.float32)
    b_s = torch.tensor([[[2.0]]])  # Ground truth count = 2.0 (the 2 heads)

    y_star = closed_form_dsr_fw_transport(y0, b_s, None, boxes)
    assert abs(y_star.sum().item() - 2.0) < 0.1, "Total count failed to match target"
    assert y_star[0, 0, 10, 10].item() > 0.75, "Real head was wrongly crushed"
    assert y_star[0, 0, 0, 0].item() < 0.002, "Fog was not aggressively eliminated"


def test_morozov_discrepancy_soft_thresholding():
    """Verify that sub-noise measurement discrepancies are regularized to zero."""
    y0 = torch.full((1, 1, 16, 16), 0.1)  # sum = 25.6
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.float32)
    b_s = torch.tensor([[[25.65]]])  # Discrepancy = 0.05

    # With morozov_tol = 0.1, residual is below noise floor
    y_star = closed_form_dsr_fw_transport(y0, b_s, None, boxes, morozov_tol=0.1)
    diff = (y_star - y0).abs().sum().item()
    assert diff < 1e-6, "Morozov failed to suppress sub-noise jitter"




