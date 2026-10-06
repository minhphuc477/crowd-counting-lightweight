from __future__ import annotations

import torch
import torch.nn.functional as F

from rmr_core.heads import _density_activate
from rmr_core.operators import build_multiscale_regions, weighted_normalized_adjoint_field
from rmr_v3.model.config import RMRv3Config
from rmr_v3.model.evidence import extract_regional_evidence


def test_asymmetric_morozov_executes_when_gamma_zero():
    """Verify that asymmetric_morozov runs shrinkage even when morozov_gamma=0.0."""
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,), overlap=0.0)
    m = regions.boxes.shape[0]

    # Ground truth is 50, prediction is 40 -> undercounting deficit delta = -10
    y = torch.full((1, 1, h, w), 40.0 / (h * w), dtype=torch.float32)
    b = torch.full((1, 1, m), 50.0, dtype=torch.float32)
    weight = torch.ones((1, 1, m), dtype=torch.float32)
    b_var = torch.full((1, 1, m), 50.0, dtype=torch.float32)  # sigma_b = sqrt(50) ≈ 7.07

    # With morozov_gamma=0.0 but asymmetric_morozov=True:
    # Under-deficit delta = -10 should experience deadband shrinkage
    field_asym = weighted_normalized_adjoint_field(
        y, b, weight, regions,
        b_variance=b_var,
        morozov_gamma=0.0,
        asymmetric_morozov=True,
        morozov_gamma_under=0.20,
        morozov_rho=0.0,  # deadband = 0.20 * 7.07 = 1.414 -> delta shrunk from -10 to -8.586
    )

    # Without any Morozov (asymmetric=False, gamma=0): delta remains un-shrunk (-10)
    field_none = weighted_normalized_adjoint_field(
        y, b, weight, regions,
        b_variance=b_var,
        morozov_gamma=0.0,
        asymmetric_morozov=False,
    )

    # field_asym magnitude must be strictly smaller than field_none due to deadband shrinkage
    assert field_asym.abs().max() < field_none.abs().max(), (
        "Asymmetric Morozov failed to execute when morozov_gamma=0.0!"
    )


def test_curvature_pade_saturation_strictly_bounded():
    """Verify that Padé curvature strictly bounds extreme density expansion, preventing O(y^2) blowup."""
    z_extreme = torch.tensor([10.0])
    alpha = torch.tensor([0.0])  # softplus(0.0) ≈ 0.693

    y_pade = _density_activate(
        z_extreme,
        temp_softplus=False,
        tau=None,
        density_curvature=True,
        curvature_alpha=alpha,
        gated_density_curvature=False,
        curvature_dense_threshold=0.15,
        curvature_gate_beta=0.03,
        curvature_pool_kernel=8,
        curvature_pade=True,
    )

    y_base = F.softplus(z_extreme)
    curv_increase = y_pade - y_base
    # Under Padé y^2/(1+y), for y ≈ 10, y^2/(1+y) ≈ 100/11 ≈ 9.09. Increase ≈ 0.693 * 9.09 ≈ 6.30.
    # Under quadratic y^2, increase would be 0.693 * 100 = 69.30 (over 10x larger!).
    assert curv_increase.item() < 10.0, (
        f"Curvature failed to saturate: increase was {curv_increase.item()} >= 10.0!"
    )
    assert curv_increase.item() > 0.0, "Curvature failed to expand density!"


def test_hurdle_occupancy_gating_prevents_dense_mass_erosion():
    """Verify that hurdle occupancy gating does not erode 15-25% mass from dense boxes."""
    cfg_occ = RMRv3Config(hurdle_head=True, hurdle_gating_mode="occupancy")
    cfg_prod = RMRv3Config(hurdle_head=True, hurdle_gating_mode="product")

    # Mock head output with b_raw = 100.0 and hurdle sigmoid = 0.85
    b_raw = torch.tensor([[[100.0]]])
    disp = torch.tensor([[[50.0]]])
    logit = torch.tensor([[[1.7346]]])  # sigmoid(1.7346) ≈ 0.85
    h_dict = {
        "mu_count": b_raw,
        "dispersion": disp,
        "log_dispersion": torch.log(disp),
        "rate": b_raw / 64.0,
        "hurdle_logit": logit,
    }

    class MockHead:
        def __call__(self, pyr, regs):
            return h_dict

    regions = build_multiscale_regions(32, 32, output_stride=4, region_sizes_px=(32,), overlap=0.0)

    res_occ = extract_regional_evidence(
        cfg_occ, MockHead(), torch.empty(1), torch.empty(1), torch.empty(1),
        regions, None, False, 8,
    )
    res_prod = extract_regional_evidence(
        cfg_prod, MockHead(), torch.empty(1), torch.empty(1), torch.empty(1),
        regions, None, False, 8,
    )

    # In occupancy mode: b_raw >= 1.0 means occ_gate = 1.0 -> 0% erosion
    assert torch.allclose(res_occ["b_solver"], b_raw), (
        f"Occupancy gate eroded mass: {res_occ['b_solver']} != {b_raw}"
    )
    # In legacy product mode: b_solver = 0.85 * 100.0 = 85.0 (15% erosion)
    assert res_prod["b_solver"].item() < 90.0, "Product mode failed to demonstrate legacy erosion"


def test_radon_nikodym_scale_seeded_carrier_denominator_normalization():
    """Verify that when carrier y=0 in an undercounted dense region, scale_seeded_carrier
    normalizes the denominator eff_q with the seed measure instead of dividing by eps (1e-6),
    preventing 20,000x gradient explosion.
    """
    h, w = 16, 16  # 16x16 feature grid = 64x64 image pixels
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32,), overlap=0.0)
    m = regions.boxes.shape[0]

    # Empty carrier: y = 0 everywhere
    y = torch.zeros((1, 1, h, w), dtype=torch.float32)
    # Heavy crowd: target b = 100 people per region
    b = torch.full((1, 1, m), 100.0, dtype=torch.float32)
    weight = torch.ones((1, 1, m), dtype=torch.float32)
    # Scale router pointing to fine scale 32px: pi_fine = 1.0
    scale_routing = torch.ones((1, 3, h, w), dtype=torch.float32)
    scale_routing[:, 1:] = 0.0

    field = weighted_normalized_adjoint_field(
        y, b, weight, regions,
        adjoint_mode="radon_nikodym",
        scale_routing_weights=scale_routing,
        scale_seeded_carrier=True,
        scale_seed_eps=0.02,
    )

    # In the buggy implementation, eff_q was q + eps*area ≈ 1e-6*64 = 6.4e-5,
    # causing rate_residual = -100 / 6.4e-5 ≈ -1.5e6, and field magnitude ≈ -20,000.
    # In the mathematically correct implementation, m_base = 0.02, q_m = 0.02 * 64 = 1.28.
    # rate_residual = -100 / 1.28 = -78.125.
    # back = 0.02 * (-78.125) = -1.5625.
    # field = -1.5625 / 1.0 = -1.5625 (O(1), bounded, stable).
    assert field.abs().max() < 50.0, (
        f"Scale seeded carrier denominator blew up: field max={field.abs().max().item()} >= 50.0!"
    )
    assert field.abs().max() > 0.1, "Field failed to inject mass!"

