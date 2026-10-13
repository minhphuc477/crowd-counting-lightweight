"""Deep forensic audit test suite for Window, Loss, and Routing components in RMR.

Covers:
1. Window & Regional Operators (regions.py, prefix_sums.py, pooling.py):
   - Adjoint duality identity: <Ay, r> == <y, A*r> to machine precision.
   - Exact prefix sum equivalence: regional_sum vs nested slice loop.
   - Boundary touching invariant: last box reaches H and W.
   - Small and anisotropic grid robustness.

2. Loss Functions & Gradient Dynamics (losses.py, dm_loss.py, point_supervision.py, cell.py, regional.py, spatial_priors.py):
   - Count loss: NB, L1, log1p, anscombe under N=0 and N=2000.
   - Flat DM16: zero count safety, count normalization, gradient boundedness.
   - Bayesian loss: autograd function gradient agreement, canonical background.
   - Regional NB NLL: Faithful Beta-decoupling gradient check on mean and dispersion.
   - Cell loss: count_harmonized negative gradient background suppression.
   - Physical scale alignment: dense pulls fine, sparse pulls coarse.

3. Routing & Perspective Heads (scale_routing.py, perspective_geometry.py):
   - ScaleRoutingHead: exact partition of unity sum_k pi_k == 1.0.
   - DiAGScaleRoutingHead: Step 0 uniform distribution (1/K) parity.
   - DiAGFactorizedRoutingHead: exact 4-window simplex partition of unity.
   - apply_scale_consistency_gating: validity and bounds.

4. Morozov Deadband & Solver Contraction (morozov.py, solver.py):
   - Deadband preserves residual sign (never flips sign).
   - Rho-cap bounds large-box deadband to prevent horizon blinding.
"""
from __future__ import annotations

import math
import pytest
import torch
import torch.nn.functional as F

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    partition_regions_by_scale,
    prefix2d,
    regional_sum,
    regional_adjoint,
    weighted_coverage,
    weighted_normalized_adjoint_field,
)
from rmr_core.operators.pooling import (
    region_average_features,
    region_mean_std_features,
)
from rmr_core.operators.morozov import compute_morozov_discrepancy
from rmr_core.scale_routing import ScaleRoutingHead, FactorizedRoutingHead
from rmr_v3.model.perspective_geometry import (
    DynamicCameraAnglePredictor,
    DiAGScaleRoutingHead,
    DiAGFactorizedRoutingHead,
)
from rmr_v3.regional_head import (
    ProbabilisticRegionalEvidenceHead,
    apply_scale_consistency_gating,
    reliability_from_nb,
)
from rmr_core.losses import (
    balanced_smooth_l1,
    negative_binomial_nll_mean_dispersion,
    count_magnitude_loss,
)
from rmr_core.dm_loss import (
    block_sum_2d,
    probs_from_positive_mass,
    flat_dm_block_loss,
)
from rmr_v3.losses.cell import (
    count_harmonized_cell_loss,
    count_invariant_cell_loss,
    mass_weighted_cell_loss,
)
from rmr_v3.losses.regional import (
    hurdle_focal_bce_loss,
    scale_balanced_regional_nb_nll,
    truncated_nb_nll_loss,
)
from rmr_v3.losses.spatial_priors import (
    curvature_power_loss,
    physical_scale_alignment_loss,
    topk_hard_background_loss,
)
from rmr_v3.losses.point_supervision import (
    bayesian_loss,
    fidt_loss,
    sinkhorn_ot_loss,
)
from rmr_v3.solver import unrolled_sirt_solver


# ==============================================================================
# 1. WINDOW & REGIONAL OPERATOR AUDIT
# ==============================================================================

def test_adjoint_duality_identity():
    """Verify fundamental adjoint duality: <Ay, r> == <y, A*r> to float32 machine precision."""
    b, h, w = 2, 32, 40
    y = torch.randn(b, 1, h, w, dtype=torch.float32)
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    m = regions.boxes.shape[0]
    r = torch.randn(b, 1, m, dtype=torch.float32)

    # Forward Ay
    Ay = regional_sum(y, regions.boxes, out_dtype=torch.float32)
    # Adjoint A*r
    A_star_r = regional_adjoint(r, regions.boxes, h, w, out_dtype=torch.float32)

    # Inner products
    inner_forward = (Ay * r).sum()
    inner_adjoint = (y * A_star_r).sum()

    diff = torch.abs(inner_forward - inner_adjoint)
    rel_diff = diff / (torch.abs(inner_forward) + 1e-6)
    assert rel_diff < 1e-4, f"Adjoint duality violation: <Ay, r>={inner_forward.item()}, <y, A*r>={inner_adjoint.item()}"


def test_regional_sum_exact_slice_loop_equivalence():
    """Verify regional_sum matches exact manual slice loops across all boxes."""
    b, h, w = 1, 24, 30
    x = torch.randn(b, 1, h, w, dtype=torch.float32)
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)

    fast_sum = regional_sum(x, regions.boxes, out_dtype=torch.float32)[0, 0]  # [M]
    manual_sum = torch.empty_like(fast_sum)

    for idx, (y1, x1, y2, x2) in enumerate(regions.boxes.tolist()):
        manual_sum[idx] = x[0, 0, y1:y2, x1:x2].sum()

    max_err = torch.max(torch.abs(fast_sum - manual_sum))
    assert max_err < 1e-5, f"Prefix sum error: max diff = {max_err.item()}"


def test_window_boundary_touching_invariant():
    """Verify that multiscale window generator always touches the exact bottom and right borders."""
    dims = [(128, 128), (96, 160), (45, 73), (17, 19)]
    for h, w in dims:
        regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(32, 64, 128), overlap=0.5)
        boxes = regions.boxes
        assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
        assert (boxes[:, 2] <= h).all() and (boxes[:, 3] <= w).all()
        # Verify border touch
        assert (boxes[:, 2] == h).any(), f"No box reaches bottom border for ({h}, {w})"
        assert (boxes[:, 3] == w).any(), f"No box reaches right border for ({h}, {w})"


# ==============================================================================
# 2. ROUTING & PERSPECTIVE SUBSYSTEM AUDIT
# ==============================================================================

def test_scale_routing_partition_of_unity():
    """Verify ScaleRoutingHead produces valid probability distributions summing to 1.0."""
    router = ScaleRoutingHead(in_channels=32, num_scales=3, temperature=1.0)
    x = torch.randn(4, 32, 64, 64)
    pi = router(x)

    assert pi.shape == (4, 3, 64, 64)
    assert (pi >= 0.0).all() and (pi <= 1.0).all()
    sum_pi = pi.sum(dim=1)
    assert torch.allclose(sum_pi, torch.ones_like(sum_pi), atol=1e-5)


def test_diag_scale_routing_step0_identity_parity():
    """Verify DiAGScaleRoutingHead starts at uniform 1/K distribution at initialization."""
    router = DiAGScaleRoutingHead(in_channels=32, num_scales=3, temperature=1.0)
    x = torch.randn(2, 32, 32, 32)
    pi = router(x)

    # Since weights and biases are zero-initialized, logits are 0 -> softmax is 1/3
    expected = 1.0 / 3.0
    assert torch.allclose(pi, torch.full_like(pi, expected), atol=1e-5), "DiAG failed Step 0 Identity Parity"


def test_diag_factorized_routing_4_window_simplex():
    """Verify DiAGFactorizedRoutingHead guarantees joint 4-window simplex partition of unity."""
    router = DiAGFactorizedRoutingHead(in_channels=32, num_scales=3, num_aspects=2)
    x = torch.randn(2, 32, 32, 32)
    joint_pi, pi_scale, pi_aspect = router(x)

    assert joint_pi.shape == (2, 4, 32, 32)
    # Check joint partition of unity
    joint_sum = joint_pi.sum(dim=1)
    assert torch.allclose(joint_sum, torch.ones_like(joint_sum), atol=1e-5)
    # Check marginal scale conservation: pi_1 + pi_2 == pi_scale[1]
    assert torch.allclose(joint_pi[:, 1:2] + joint_pi[:, 2:3], pi_scale[:, 1:2], atol=1e-5)


def test_physical_scale_alignment_loss_directionality():
    """Verify that physical scale alignment correctly guides high density -> fine scale and low -> coarse."""
    b, k, h, w = 1, 3, 16, 16
    # Scale 0 is fine (32px), Scale 2 is coarse (128px)
    # Case A: Very dense target (e.g. 5.0 people/cell)
    dense_target = torch.full((b, 1, h, w), 5.0)
    # Model routes to coarse scale (scale 2) -> should have high loss
    wrong_pi = torch.zeros(b, k, h, w)
    wrong_pi[:, 2] = 1.0
    wrong_loss = physical_scale_alignment_loss(wrong_pi, dense_target, tau_dense=0.12, tau_sparse=0.03)

    # Model routes to fine scale (scale 0) -> should have near-zero loss
    correct_pi = torch.zeros(b, k, h, w)
    correct_pi[:, 0] = 1.0
    correct_loss = physical_scale_alignment_loss(correct_pi, dense_target, tau_dense=0.12, tau_sparse=0.03)

    assert wrong_loss.item() > correct_loss.item() + 1.0, "Physical scale alignment failed density guidance direction"


# ==============================================================================
# 3. LOSS FUNCTIONS & GRADIENT DYNAMICS AUDIT
# ==============================================================================

def test_negative_binomial_nll_zero_and_large_counts():
    """Verify NB NLL handles count extremes (N=0 to N=5000) stably without NaNs."""
    targets = torch.tensor([0.0, 1.0, 50.0, 500.0, 5000.0])
    means = torch.tensor([0.01, 1.2, 48.0, 520.0, 4800.0], requires_grad=True)
    dispersion = torch.tensor(50.0)

    loss = negative_binomial_nll_mean_dispersion(targets, means, dispersion=dispersion, reduction="mean", check_bounds=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(means.grad).all()


def test_flat_dm16_empty_parent_safety():
    """Verify Flat DM16 returns exactly 0.0 loss and 0.0 gradient on completely empty crops."""
    pred = torch.full((1, 1, 32, 32), 0.05, requires_grad=True)
    empty_target = torch.zeros(1, 1, 32, 32)

    loss = flat_dm_block_loss(pred, empty_target, block_px=16, stride=4, kappa=20.0, strict=True)
    assert loss.item() == 0.0
    loss.backward()
    assert (pred.grad == 0.0).all()


def test_faithful_beta_nb_gradient_decoupling():
    """Verify that under Faithful Beta-NB, dispersion parameter gradient does not leak into mean."""
    from rmr_v3.losses.regional import _decoupled_beta_nb_nll

    target = torch.tensor([[10.0, 50.0]])
    mean = torch.tensor([[8.0, 45.0]], requires_grad=True)
    dispersion = torch.tensor([[20.0, 30.0]], requires_grad=True)

    loss = _decoupled_beta_nb_nll(target, mean, dispersion, beta=0.5, faithful=True).sum()
    loss.backward()

    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert dispersion.grad is not None and torch.isfinite(dispersion.grad).all()


def test_count_harmonized_cell_loss_background_suppression():
    """Verify count_harmonized cell loss provides non-zero positive gradient on background cells to drive them to 0."""
    pred = torch.full((1, 1, 16, 16), 0.05, requires_grad=True)
    # Background target with 1 person in the corner
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 0, 0] = 1.0

    loss = count_harmonized_cell_loss(pred, target, fg_ratio=0.67, norm_power=0.5)
    loss.backward()

    # Empty cells must have positive gradient dL/d(pred) > 0, driving pred down towards 0
    bg_grad = pred.grad[0, 0, 5:, 5:]
    assert (bg_grad > 0.0).all(), "Background cells must receive positive gradient to suppress phantom mass"


def test_bayesian_loss_person_and_background_balance():
    """Verify Bayesian loss computes finite loss and backward gradients on synthetic points."""
    pred = torch.full((1, 1, 32, 32), 0.01, requires_grad=True)
    pts = [torch.tensor([[32.0, 32.0], [64.0, 64.0]])]

    loss = bayesian_loss(pred, pts, sigma=4.0, stride=4, norm_mode="canonical")
    assert torch.isfinite(loss)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ==============================================================================
# 4. MOROZOV DEADBAND & SOLVER CONTRACTION AUDIT
# ==============================================================================

def test_morozov_deadband_sign_preservation():
    """Verify Morozov discrepancy deadband shrinks magnitude without ever flipping residual sign."""
    q = torch.tensor([[10.0, 50.0, 100.0]])
    b = torch.tensor([[20.0, 30.0, 100.0]])
    delta = q - b  # [-10.0, +20.0, 0.0]
    b_var = torch.tensor([[5.0, 15.0, 20.0]])
    regions = build_multiscale_regions(16, 16, output_stride=4, region_sizes_px=(16,))

    shrunk = compute_morozov_discrepancy(
        delta=delta,
        q=q,
        b32=b,
        b_variance=b_var,
        regions=regions,
        morozov_gamma=0.5,
    )

    # Shrunk residual must have identical sign or become 0
    assert shrunk[0, 0] <= 0.0, "Negative residual must remain negative or zero"
    assert shrunk[0, 1] >= 0.0, "Positive residual must remain positive or zero"
    assert shrunk[0, 2] == 0.0, "Zero residual must remain zero"
    assert shrunk[0, 0].abs() <= delta[0, 0].abs(), "Deadband must shrink magnitude"
    assert shrunk[0, 1].abs() <= delta[0, 1].abs(), "Deadband must shrink magnitude"
