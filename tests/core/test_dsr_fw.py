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
