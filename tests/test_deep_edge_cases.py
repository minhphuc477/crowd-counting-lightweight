from __future__ import annotations

import math
import torch
import torch.nn.functional as F
import pytest

from rmr_core.operators import (
    RegionSet,
    closed_form_radon_nikodym_transport,
    build_multiscale_regions,
)
from rmr_v3.model.solver_step import solve_inverse_measure
from rmr_v3.model.config import RMRv3Config
from rmr_v3.regional_head import ProbabilisticRegionalEvidenceHead
from rmr_v3.losses.regional import scale_balanced_regional_nb_nll
from rmr_v3.losses.spatial_priors import (
    strauss_repulsion_loss,
    radon_nikodym_transport_energy_loss,
)


def test_radon_nikodym_zero_coverage_preserves_carrier():
    """Edge Case A: When coverage is 0 (m=0 or uncovered pixels), carrier mass must NOT be erased."""
    torch.manual_seed(42)
    y0 = torch.rand(2, 1, 32, 32) + 0.1
    initial_sum = y0.sum().item()

    # Empty regions: m=0
    boxes_empty = torch.zeros((0, 4), dtype=torch.long)
    regions_empty = RegionSet(
        boxes=boxes_empty,
        scale_id=torch.zeros((0,), dtype=torch.long),
        area=torch.zeros((0,)),
    )
    b_solver_empty = torch.zeros((2, 1, 0))

    y_projected, s_field = closed_form_radon_nikodym_transport(
        y0=y0,
        b_solver=b_solver_empty,
        weight_solver=None,
        regions=regions_empty,
        return_multiplier=True,
    )

    # Multiplier must be neutral 1.0 everywhere, preserving carrier
    assert torch.allclose(s_field, torch.ones_like(s_field), atol=1e-5)
    assert torch.allclose(y_projected, y0, atol=1e-5)
    assert math.isclose(y_projected.sum().item(), initial_sum, rel_tol=1e-5)


def test_radon_nikodym_both_zero_neutrality():
    """Edge Case B: When both b_solver=0 and b_hat=0, scaling ratio must be 1.0 (neutral), not attenuated."""
    y0 = torch.zeros(1, 1, 32, 32)
    # A single region covering the whole image
    boxes = torch.tensor([[0, 0, 32, 32]], dtype=torch.long)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.zeros((1,), dtype=torch.long),
        area=torch.tensor([1024.0]),
    )
    b_solver = torch.zeros(1, 1, 1)

    y_proj, s_mult = closed_form_radon_nikodym_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=None,
        regions=regions,
        return_multiplier=True,
    )
    assert torch.allclose(s_mult, torch.ones_like(s_mult), atol=1e-5)
    assert torch.allclose(y_proj, torch.zeros_like(y_proj), atol=1e-5)


def test_radon_nikodym_non_negativity_extreme_trust():
    """Edge Case C: Scaling ratio must remain non-negative even with large trust_alpha."""
    y0 = torch.rand(1, 1, 32, 32)
    boxes = torch.tensor([[0, 0, 16, 16]], dtype=torch.long)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.zeros((1,), dtype=torch.long),
        area=torch.tensor([256.0]),
    )
    # Target solver evidence is 0 while carrier has mass
    b_solver = torch.zeros(1, 1, 1)

    # Test extreme trust_alpha
    y_proj, s_mult = closed_form_radon_nikodym_transport(
        y0=y0,
        b_solver=b_solver,
        weight_solver=None,
        regions=regions,
        trust_alpha=2.5,
        return_multiplier=True,
    )
    assert (s_mult >= 0.0).all()
    assert (y_proj >= 0.0).all()


def test_solver_step_multiplier_identity_with_solver_strength():
    """Edge Case D: When effective_strength < 1.0, y_final == y0 * rn_multiplier identically."""
    cfg = RMRv3Config(
        enable_solver=True,
        solver_mode="closed_form_rn",
        rn_trust_alpha=0.20,
    )
    y0 = torch.rand(1, 1, 32, 32) + 0.1
    z0 = torch.rand(1, 1, 32, 32)
    boxes = torch.tensor([[0, 0, 16, 16], [0, 16, 16, 32]], dtype=torch.long)
    regions = RegionSet(
        boxes=boxes,
        scale_id=torch.tensor([0, 0], dtype=torch.long),
        area=torch.tensor([256.0, 256.0]),
    )
    b_solver = torch.tensor([[[5.0, 12.0]]])
    regional_evidence = {
        "mu_count": b_solver,
        "dispersion": torch.tensor([[[50.0, 50.0]]]),
        "b_solver": b_solver,
        "b_variance": torch.tensor([[[5.5, 14.88]]]),
        "weight": torch.ones_like(b_solver),
        "weight_solver": torch.ones_like(b_solver),
        "hurdle_logit": None,
        "rate": b_solver / 256.0,
        "log_dispersion": torch.log(torch.tensor([[[50.0, 50.0]]])),
        "precision": torch.ones_like(b_solver),
        "rate_variance": torch.ones_like(b_solver) * 0.01,
        "count_variance": torch.tensor([[[5.5, 14.88]]]),
    }

    # Test with partial strength
    out = solve_inverse_measure(
        cfg=cfg,
        y0=y0,
        z0=z0,
        regional_evidence=regional_evidence,
        regions=regions,
        scale_weights=None,
        solver_strength=0.6,
        default_solver_strength=0.6,
        uniform_reliability=True,
        p16=torch.zeros(1, 32, 8, 8),
        fg_logit=None,
        pi_scale=None,
        pi_aspect=None,
        trust_gate=None,
        laplace_kernel=torch.zeros(1, 1, 3, 3),
    )

    y_final = out.y
    rn_mult = out.rn_multiplier
    # Verify exact Radon-Nikodym identity: y_final == y0 * rn_multiplier
    expected_y = y0 * rn_mult
    assert torch.allclose(y_final, expected_y, atol=1e-5)


def test_regional_head_shuffled_scale_id_invariance():
    """Edge Case E: Features and predictions must match region boxes row-for-row, regardless of scale_id ordering."""
    torch.manual_seed(42)
    head = ProbabilisticRegionalEvidenceHead(
        feature_dim=32,
        hidden=48,
        region_sizes_px=(32, 64, 128),
    )
    head.eval()

    p4 = torch.rand(1, 32, 32, 32)
    p8 = torch.rand(1, 32, 16, 16)
    p16 = torch.rand(1, 32, 8, 8)
    pyramid = (p4, p8, p16)

    # Ordered regions: sid 0, 1, 2
    boxes_0 = torch.tensor([[0, 0, 8, 8]], dtype=torch.long)
    boxes_1 = torch.tensor([[0, 0, 16, 16]], dtype=torch.long)
    boxes_2 = torch.tensor([[0, 0, 32, 32]], dtype=torch.long)

    boxes_ordered = torch.cat([boxes_0, boxes_1, boxes_2], dim=0)
    scale_id_ordered = torch.tensor([0, 1, 2], dtype=torch.long)
    area_ordered = torch.tensor([64.0, 256.0, 1024.0])
    regions_ordered = RegionSet(boxes=boxes_ordered, scale_id=scale_id_ordered, area=area_ordered)

    # Shuffled regions: sid 2, 0, 1
    boxes_shuffled = torch.cat([boxes_2, boxes_0, boxes_1], dim=0)
    scale_id_shuffled = torch.tensor([2, 0, 1], dtype=torch.long)
    area_shuffled = torch.tensor([1024.0, 64.0, 256.0])
    regions_shuffled = RegionSet(boxes=boxes_shuffled, scale_id=scale_id_shuffled, area=area_shuffled)

    with torch.no_grad():
        out_ordered = head(pyramid, regions_ordered)
        out_shuffled = head(pyramid, regions_shuffled)

    # Row 0 of shuffled (sid=2) must equal Row 2 of ordered (sid=2)
    assert torch.allclose(out_shuffled["mu_count"][:, :, 0], out_ordered["mu_count"][:, :, 2], atol=1e-5)
    # Row 1 of shuffled (sid=0) must equal Row 0 of ordered (sid=0)
    assert torch.allclose(out_shuffled["mu_count"][:, :, 1], out_ordered["mu_count"][:, :, 0], atol=1e-5)
    # Row 2 of shuffled (sid=1) must equal Row 1 of ordered (sid=1)
    assert torch.allclose(out_shuffled["mu_count"][:, :, 2], out_ordered["mu_count"][:, :, 1], atol=1e-5)


def test_scale_balanced_regional_nb_nll_dimension_invariance():
    """Edge Case F: scale_balanced_regional_nb_nll must accept 1D, 2D, and 3D tensors cleanly."""
    target_3d = torch.tensor([[[2.0, 5.0, 10.0]]])
    mean_3d = torch.tensor([[[2.2, 4.8, 9.5]]])
    disp_3d = torch.tensor([[[50.0, 50.0, 50.0]]])
    regions = RegionSet(
        boxes=torch.tensor([[0, 0, 8, 8], [0, 0, 16, 16], [0, 0, 32, 32]], dtype=torch.long),
        scale_id=torch.tensor([0, 1, 2], dtype=torch.long),
        area=torch.tensor([64.0, 256.0, 1024.0]),
    )

    loss_3d = scale_balanced_regional_nb_nll(target_3d, mean_3d, disp_3d, regions)

    # Pass as 2D tensors [B, M]
    target_2d = target_3d.squeeze(1)
    mean_2d = mean_3d.squeeze(1)
    disp_2d = disp_3d.squeeze(1)
    loss_2d = scale_balanced_regional_nb_nll(target_2d, mean_2d, disp_2d, regions)

    # Pass as mixed dimensions (target 3D, mean 2D)
    loss_mixed = scale_balanced_regional_nb_nll(target_3d, mean_2d, disp_3d, regions)

    assert torch.allclose(loss_3d, loss_2d, atol=1e-6)
    assert torch.allclose(loss_3d, loss_mixed, atol=1e-6)


def test_empty_tensor_safeguards():
    """Edge Case G: Empty tensors must return zero without throwing exceptions or generating NaNs."""
    empty_y = torch.zeros((0, 1, 32, 32))
    empty_s = torch.zeros((0, 1, 32, 32))

    l_strauss = strauss_repulsion_loss(empty_y)
    assert torch.isfinite(l_strauss)
    assert l_strauss.item() == 0.0

    l_transport = radon_nikodym_transport_energy_loss(empty_y, empty_s)
    assert torch.isfinite(l_transport)
    assert l_transport.item() == 0.0


def test_balanced_smooth_l1_dimension_and_empty_scene_invariance():
    """Edge Case H: balanced_smooth_l1 must accept 3D/4D targets without indexing errors and penalize empty scenes at 1.0 weight."""
    from rmr_core.cell_loss import balanced_smooth_l1

    p = torch.ones(1, 1, 10, 10) * 0.5
    # 3D target on empty scene
    t_3d = torch.zeros(1, 10, 10)
    loss_empty = balanced_smooth_l1(p, t_3d)
    # Smooth L1 of 0.5 vs 0 with beta=1 is 0.5 * 0.5^2 = 0.125
    assert abs(loss_empty.item() - 0.125) < 1e-5

    # 4D target with mixed foreground/background
    t_4d = torch.zeros(1, 1, 10, 10)
    t_4d[0, 0, 0, 0] = 1.0
    loss_both = balanced_smooth_l1(p, t_4d)
    assert torch.isfinite(loss_both)
    assert loss_both.item() > 0.0
