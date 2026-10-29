import pytest
import torch
from rmr_core.operators import RegionSet, build_multiscale_regions
from rmr_core.operators.peak_concentrated import closed_form_peak_concentrated_transport


def test_support_invariance():
    """Verify that empty space receives zero mass injection."""
    h, w = 32, 32
    y0 = torch.zeros(1, 1, h, w)
    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 5.0)

    y_star = closed_form_peak_concentrated_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
    )
    assert (y_star == 0.0).all(), "Background support must remain strictly zero"


def test_zero_valley_fog_and_peak_boost():
    """Verify that valleys are untouched while peaks receive deficit mass."""
    h, w = 64, 64
    y0 = torch.zeros(1, 1, h, w)

    # 1 peak at (32, 32)
    y0[0, 0, 32, 32] = 0.5
    # Valley noise at (10, 10)
    y0[0, 0, 10, 10] = 0.005

    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16, 32], overlap=0.5)
    # Deficit evidence: demanding 5.0 count
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 5.0)

    y_star, s_eff = closed_form_peak_concentrated_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
        tau_noise=0.02,
        return_multiplier=True,
    )

    # Valley noise must be untouched because it is below tau_noise (0.02)
    assert torch.isclose(y_star[0, 0, 10, 10], y0[0, 0, 10, 10], atol=1e-6), "Valley noise must not be amplified"
    # Peak at (32, 32) must be boosted
    assert y_star[0, 0, 32, 32] > y0[0, 0, 32, 32], "Peak must be amplified"


def test_autograd_differentiability():
    """Verify backward gradient propagation."""
    h, w = 32, 32
    y0 = torch.rand(1, 1, h, w, requires_grad=True)
    regions = build_multiscale_regions(h, w, output_stride=2, region_sizes_px=[16], overlap=0.5)
    b_s = torch.full((1, 1, regions.boxes.shape[0]), 2.0, requires_grad=True)

    y_star = closed_form_peak_concentrated_transport(
        y0=y0,
        b_solver=b_s,
        weight_solver=None,
        regions=regions,
    )
    loss = y_star.sum()
    loss.backward()

    assert y0.grad is not None, "Gradient on y0 must exist"
    assert not torch.isnan(y0.grad).any(), "Gradient must be finite"
