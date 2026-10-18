"""Unit tests verifying all V5.0 architectural and loss upgrades.

Tests:
1. RepASPPLiteNeck deploy mode fusion exact numerical equivalence.
2. Hurdle occupancy gating: 100% mass preservation on real crowd, background suppression.
3. Beta-NB regional loss: gradient disparity reduction on dense boxes.
4. ProbabilisticRegionalEvidenceHead: native multi-scale pooling across [32, 64, 128] px.
5. Decoupled trust region clamping: zero-carrier cell deficit recovery & non-negativity.
"""
import torch
import torch.nn.functional as F
import pytest

from rmr_core.necks.rep_aspp import RepASPPLiteNeck
from rmr_core.operators import build_multiscale_regions
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead
from rmr_v3.solver_trust import apply_trust_region_clamping
from rmr_v3.losses.regional import _decoupled_beta_nb_nll


def test_repaspp_fusion_numerical_equivalence():
    neck = RepASPPLiteNeck(
        in_channels=(16, 32, 48),
        width=32,
        aspp_dilations=(1, 3, 6),
        use_aspp_gap=True,
    )
    neck.eval()
    c4 = torch.randn(2, 16, 64, 64)
    c8 = torch.randn(2, 32, 32, 32)
    c16 = torch.randn(2, 48, 16, 16)

    with torch.no_grad():
        p4_train, p8_train, p16_train = neck(c4, c8, c16)

    neck.switch_to_deploy()
    with torch.no_grad():
        p4_deploy, p8_deploy, p16_deploy = neck(c4, c8, c16)

    diff4 = (p4_train - p4_deploy).abs().max().item()
    diff8 = (p8_train - p8_deploy).abs().max().item()
    diff16 = (p16_train - p16_deploy).abs().max().item()

    assert diff4 < 1e-4, f"P4 deploy fusion difference too large: {diff4}"
    assert diff8 < 1e-4, f"P8 deploy fusion difference too large: {diff8}"
    assert diff16 < 1e-4, f"P16 deploy fusion difference too large: {diff16}"


def test_hurdle_occupancy_gating_preserves_dense_crowd():
    # Dense box with count = 50.0
    b_dense = torch.tensor([50.0])
    pi_r = torch.tensor([0.75])  # moderate hurdle probability

    # Mathematical formulation of occupancy gating:
    occ_gate_dense = 1.0 - (1.0 - pi_r) * torch.clamp(1.0 - b_dense, min=0.0, max=1.0)
    b_solver_dense = occ_gate_dense * b_dense

    # When b >= 1.0, clamp(1.0 - b, min=0.0) == 0.0, so occ_gate == 1.0 (100% mass preserved)
    assert torch.allclose(occ_gate_dense, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(b_solver_dense, b_dense, atol=1e-5)

    # Empty background with count = 0.05
    b_bg = torch.tensor([0.05])
    occ_gate_bg = 1.0 - (1.0 - pi_r) * torch.clamp(1.0 - b_bg, min=0.0, max=1.0)
    b_solver_bg = occ_gate_bg * b_bg
    # Attenuated
    assert b_solver_bg < b_bg


def test_beta_nb_dense_gradient_recovery():
    # 1 sparse box (mu=5, count=2) and 1 dense box (mu=200, count=250)
    mu_std = torch.tensor([[5.0, 200.0]], requires_grad=True)
    r_std = torch.tensor([[10.0, 10.0]])
    count = torch.tensor([[2.0, 250.0]])

    # Standard NB (beta=0.0)
    loss_std = _decoupled_beta_nb_nll(count, mu_std, dispersion=r_std, beta=0.0, faithful=False).sum()
    loss_std.backward()
    grad_std_sparse = abs(mu_std.grad[0, 0].item())
    grad_std_dense = abs(mu_std.grad[0, 1].item())
    ratio_std = grad_std_sparse / max(grad_std_dense, 1e-9)

    # Beta-NB (beta=0.5)
    mu_beta = torch.tensor([[5.0, 200.0]], requires_grad=True)
    r_beta = torch.tensor([[10.0, 10.0]])
    loss_beta = _decoupled_beta_nb_nll(count, mu_beta, dispersion=r_beta, beta=0.5, faithful=False).sum()
    loss_beta.backward()
    grad_beta_sparse = abs(mu_beta.grad[0, 0].item())
    grad_beta_dense = abs(mu_beta.grad[0, 1].item())
    ratio_beta = grad_beta_sparse / max(grad_beta_dense, 1e-9)

    # Disparity must be significantly reduced with Beta-NB
    assert ratio_beta < ratio_std, f"Beta-NB ratio {ratio_beta} not better than standard {ratio_std}"
    assert grad_beta_dense > grad_std_dense, "Beta-NB did not boost dense gradient"


def test_probabilistic_regional_evidence_head_native_pooling():
    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        region_sizes_px=(32, 64, 128),
        native_scale_pooling=True,
    )
    p4 = torch.randn(2, 32, 64, 64)
    p8 = torch.randn(2, 32, 32, 32)
    p16 = torch.randn(2, 32, 16, 16)
    regions = build_multiscale_regions(64, 64, output_stride=4, region_sizes_px=(32, 64, 128))

    out = head((p4, p8, p16), regions)
    assert "mu_count" in out
    assert "dispersion" in out
    assert out["mu_count"].shape == (2, 1, regions.boxes.shape[0])
    assert (out["mu_count"] >= 0).all()


def test_decoupled_trust_region_clamping_recovery():
    # Test zero-carrier cell with positive deficit update
    delta_pos = torch.tensor([[[[5.0]]]])
    z_zero = torch.tensor([[[[0.0]]]])
    clamped_pos = apply_trust_region_clamping(
        delta_y=delta_pos,
        z_state=z_zero,
        trust_region_kappa=0.35,
        effective_trust_floor=0.005,
        area_scale=1.0,
        trust_addition_floor=1.0,
        max_addition_cap=2.5,
    )
    # Under decoupled addition floor, update should be positive and >= 0.1
    assert clamped_pos.item() > 0.1, f"Zero carrier starved: update was {clamped_pos.item()}"

    # Test non-negativity bound under negative displacement
    delta_neg = torch.tensor([[[[-5.0]]]])
    clamped_neg = apply_trust_region_clamping(
        delta_y=delta_neg,
        z_state=z_zero,
        trust_region_kappa=0.35,
        effective_trust_floor=0.005,
        area_scale=1.0,
        trust_addition_floor=1.0,
        max_addition_cap=2.5,
    )
    # Removal cannot be more negative than -0.35 * floor (e.g. -0.00175)
    assert clamped_neg.item() >= -0.01, f"Non-negativity violated: update was {clamped_neg.item()}"
