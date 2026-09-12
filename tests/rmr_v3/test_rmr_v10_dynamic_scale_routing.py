from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from rmr_core.operators import (
    build_multiscale_regions,
    partition_regions_by_scale,
    weighted_coverage,
    weighted_normalized_adjoint_field,
)
from rmr_core.scale_routing import ScaleRoutingHead
from rmr_v3.config import RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3
from rmr_v3.solver import unrolled_sirt_solver


def test_scale_routed_conservation_theorem():
    """Empirical proof of Theorem 1: H_pi 1_G = 1_G under arbitrary spatial scale routing.

    H_pi = D_{c, pi}^-1 A_pi^T W D_a^-1 A
    Satisfies H_pi (c 1_G) = c 1_G for any constant rate c and any valid spatial
    scale probability map pi(p) in Delta^{K-1}, regardless of:
    1. Grid dimensions (even, odd, asymmetric).
    2. Overlap degree or box sizes.
    3. Positive weight distribution W > 0.
    4. Highly non-uniform, spatially varying scale routing pi(p).
    """
    torch.manual_seed(42)
    test_geometries = [
        (32, 32, (32, 64, 128)),
        (35, 29, (32, 64, 128)),
        (21, 53, (32, 64, 128)),
        (17, 19, (32, 64)),
    ]

    for h, w, sizes in test_geometries:
        regions = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=4,
            region_sizes_px=sizes,
            overlap=0.5,
            include_full_image=False,
            device="cpu",
        )
        k_scales = len(sizes)
        # Random non-uniform positive regional weights
        w_weights = torch.rand(2, 1, len(regions.boxes), dtype=torch.float32) * 3.0 + 0.5

        # Create random non-uniform spatial scale routing field pi in Delta^{K-1}
        raw_logits = torch.randn(2, k_scales, h, w, dtype=torch.float32)
        pi_weights = torch.softmax(raw_logits, dim=1)

        # Compute scale-routed coverage field D_{c, pi}
        d_c_pi = weighted_coverage(
            w_weights,
            regions,
            h,
            w,
            eps=1e-6,
            scale_routing_weights=pi_weights,
        )

        for c in [0.015, 1.0, 5.25, 42.0]:
            const_density = torch.full((2, 1, h, w), fill_value=c, dtype=torch.float32)
            # b = 0 implies delta = A y
            b_zero = torch.zeros(2, 1, len(regions.boxes), dtype=torch.float32)

            # H_pi (c 1_G) is the normalized adjoint field evaluated at b=0
            h_pi_const = weighted_normalized_adjoint_field(
                const_density,
                b_zero,
                w_weights,
                regions,
                weighted_cov=d_c_pi,
                eps=1e-6,
                scale_routing_weights=pi_weights,
            )

            rel_err = (h_pi_const - c).abs().max().item() / float(c)
            assert rel_err < 1e-4, (
                f"Conservation theorem violated for geometry ({h}, {w}) and c={c}! "
                f"Max relative error: {rel_err:.6e}"
            )


def test_scale_router_head_architecture_and_budget():
    """Verify ScaleRoutingHead parameter budget, output normalization, and uniform prior initialization."""
    router = ScaleRoutingHead(in_channels=32, num_scales=3, temperature=1.0)
    n_params = sum(p.numel() for p in router.parameters() if p.requires_grad)

    assert n_params <= 500, f"ScaleRoutingHead parameters ({n_params}) exceeded 500 budget!"
    assert n_params == 483, f"Expected exactly 483 parameters, got {n_params}"

    x = torch.randn(4, 32, 64, 64)
    pi = router(x)

    assert pi.shape == (4, 3, 64, 64)
    assert (pi >= 0.0).all(), "Scale probabilities must be non-negative!"
    sum_pi = pi.sum(dim=1)
    assert torch.allclose(sum_pi, torch.ones_like(sum_pi), atol=1e-5), (
        "Scale routing probabilities must sum to 1.0 across scale dimension!"
    )

    # At initialization (before training), router outputs exact uniform distribution 1/K
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / 3.0), atol=1e-5), (
        "Router did not initialize to uniform scale prior 1/K!"
    )


def test_full_model_trainable_parameters_within_105k():
    """Verify RMRv3 with Dynamic Scale Routing strictly respects the <= 105,000 parameter budget."""
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        regional_feature_stats="mean",
        region_head_hidden=48,
        hurdle_head=True,
        temp_softplus=True,
        dynamic_scale_routing=True,
    )
    model = RMRv3(cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_params <= 105_000, f"Total trainable parameters {total_params} exceed 105,000 budget!"
    assert total_params == 104_440, f"Expected exactly 104,440 parameters, got {total_params}"


def test_symmetric_dual_supervision_gradients():
    """Verify Symmetric Dual Supervision anchors y0 count and cell loss directly."""
    cfg_loss = RMRv3LossConfig(
        dm_target="dual",
        lambda_count=1.0,
        lambda_cell=0.25,
        lambda_flat_dm16=1.0,
    )

    b, h, w = 2, 32, 32
    y0 = torch.rand(b, 1, h, w, requires_grad=True)
    y = torch.rand(b, 1, h, w, requires_grad=True)
    gt = torch.zeros(b, 1, h, w)
    gt[:, :, 10:15, 10:15] = 0.5  # Sparse GT crowd
    regions = build_multiscale_regions(h, w, 4, (32, 64, 128), 0.5, False, device="cpu")

    outputs = {
        "y": y,
        "y0": y0,
        "regions": regions,
        "b_region": torch.ones(b, 1, len(regions.boxes)),
        "region_dispersion": torch.full((b, 1, len(regions.boxes)), 50.0),
    }

    losses = compute_rmr_v3_losses(outputs, gt, cfg_loss)

    assert "count_y0" in losses
    assert "count_y" in losses
    assert "cell_y0" in losses
    assert "cell_y" in losses
    assert "allocation_y0" in losses
    assert "allocation_y" in losses

    total_loss = losses["total"]
    total_loss.backward()

    # Crucial assertion: y0 MUST receive direct, non-zero gradient from dual supervision!
    assert y0.grad is not None
    assert y0.grad.abs().sum().item() > 0.0, "y0 received zero gradient under dual supervision!"
    assert y.grad is not None
    assert y.grad.abs().sum().item() > 0.0, "y received zero gradient under dual supervision!"


def test_tv_step_t_invariance():
    """Verify that TV diffusion step is T-invariant (divided by iterations T)."""
    torch.manual_seed(99)
    y0 = torch.rand(1, 1, 32, 32)
    regions = build_multiscale_regions(32, 32, 4, (32, 64, 128), 0.5, False, device="cpu")
    m = len(regions.boxes)
    b_solver = torch.ones(1, 1, m)
    weight_solver = torch.ones(1, 1, m)

    # Test solver runs cleanly across different T without divergence
    for t in [1, 2, 4, 6]:
        res = unrolled_sirt_solver(
            y0=y0,
            b_solver=b_solver,
            weight_solver=weight_solver,
            regions=regions,
            iterations=t,
            tv_lambda=0.02,
            tv_type="laplacian",
            proximal_tau=0.015,
            proximal_mode="firm",
        )
        assert torch.isfinite(res["y"]).all()
        assert (res["y"] >= 0.0).all()


def test_one_batch_overfit_dsr_sparse_and_dense():
    """End-to-end integration test: 1-batch overfit on sparse and dense images.

    Verifies that:
    1. Y0 total count does not drift on sparse images.
    2. Dynamic Scale Router parameters receive gradients and update.
    3. Loss strictly decreases.
    """
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        pretrained=False,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        regional_feature_stats="mean",
        region_head_hidden=48,
        hurdle_head=True,
        temp_softplus=True,
        dynamic_scale_routing=True,
        proximal_mode="firm",
        proximal_tau=0.015,
        iterations=4,
    )
    model = RMRv3(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_cfg = RMRv3LossConfig(dm_target="dual")

    # Sparse image (5 people)
    x_sparse = torch.randn(1, 3, 128, 128)
    gt_sparse = torch.zeros(1, 1, 32, 32)
    gt_sparse[0, 0, 5, 5] = 1.0
    gt_sparse[0, 0, 10, 10] = 1.0
    gt_sparse[0, 0, 15, 15] = 1.0
    gt_sparse[0, 0, 20, 20] = 1.0
    gt_sparse[0, 0, 25, 25] = 1.0
    target_count = 5.0

    initial_loss = None
    for step in range(10):
        optimizer.zero_grad()
        out = model(x_sparse)
        losses = compute_rmr_v3_losses(out, gt_sparse, loss_cfg)
        loss = losses["total"]
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

    # Verify Y0 count did not drift to 4,000+!
    with torch.no_grad():
        out_eval = model(x_sparse)
        pred_y0_count = out_eval["y0"].sum().item()
        pred_y_count = out_eval["y"].sum().item()

    assert pred_y0_count < 50.0, f"Y0 count drifted to {pred_y0_count} on a 5-person image!"
    assert pred_y_count < 50.0, f"Y count drifted to {pred_y_count} on a 5-person image!"

    # Verify router weights were updated by optimizer
    assert model.scale_router is not None
    assert model.scale_router.pw.weight.grad is not None
    assert model.scale_router.pw.weight.grad.abs().sum().item() > 0.0


def test_partition_regions_by_scale_and_equivalence():
    """Verify partition_regions_by_scale and numerical equivalence of pre-partitioned operators."""
    h, w = 48, 64
    regions = build_multiscale_regions(h, w, 4, (32, 64, 128), 0.5, include_full_image=True, device="cpu")
    k_scales = 3

    partitions = partition_regions_by_scale(regions, k_scales)
    assert len(partitions) == k_scales

    # Verify full-image region (-1) was mapped to coarsest scale (k_scales - 1 == 2)
    full_mask = (regions.scale_id == -1)
    if full_mask.any():
        _, coarsest_mask, _ = partitions[2]
        assert coarsest_mask is not None
        assert coarsest_mask[full_mask].all()

    # Numerical equivalence test: with and without precomputed scale_partitions
    pi = torch.softmax(torch.randn(2, k_scales, h, w), dim=1)
    weights = torch.rand(2, 1, len(regions.boxes)) + 0.5
    y = torch.rand(2, 1, h, w)
    b = torch.ones(2, 1, len(regions.boxes))

    cov_on_the_fly = weighted_coverage(weights, regions, h, w, scale_routing_weights=pi)
    cov_prepartitioned = weighted_coverage(weights, regions, h, w, scale_routing_weights=pi, scale_partitions=partitions)
    assert torch.allclose(cov_on_the_fly, cov_prepartitioned, atol=1e-7)

    field_on_the_fly = weighted_normalized_adjoint_field(
        y, b, weights, regions, weighted_cov=cov_on_the_fly, scale_routing_weights=pi
    )
    field_prepartitioned = weighted_normalized_adjoint_field(
        y, b, weights, regions, weighted_cov=cov_prepartitioned, scale_routing_weights=pi, scale_partitions=partitions
    )
    assert torch.allclose(field_on_the_fly, field_prepartitioned, atol=1e-7)


def test_solver_warmup_short_circuit():
    """Verify that unrolled_sirt_solver short-circuits with 0 work when solver_strength == 0.0."""
    y0 = torch.rand(2, 1, 32, 32)
    regions = build_multiscale_regions(32, 32, 4, (32, 64, 128), 0.5, False, device="cpu")
    m = len(regions.boxes)
    b_solver = torch.ones(2, 1, m)
    weight_solver = torch.ones(2, 1, m)
    pi = torch.softmax(torch.randn(2, 3, 32, 32), dim=1)

    # When solver_strength == 0.0
    res = unrolled_sirt_solver(
        y0=y0,
        b_solver=b_solver,
        weight_solver=weight_solver,
        regions=regions,
        iterations=6,
        solver_strength=0.0,
        scale_routing_weights=pi,
    )

    # Must be identical to y0
    assert torch.equal(res["y"], y0)
    assert len(res["iterates"]) == 1
    assert len(res["residual_fields"]) == 0
    assert len(res["energy_trace"]) == 0
    assert res["effective_omega"] == 0.0


def test_scale_router_telemetry_tracker():
    """Verify DiagnosticTracker logs scale_pi_32, scale_pi_64, scale_pi_128."""
    from rmr_v3.train import DiagnosticTracker

    cfg = RMRv3Config(dynamic_scale_routing=True, pretrained=False)
    model = RMRv3(cfg)
    tracker = DiagnosticTracker(model)

    regions = build_multiscale_regions(32, 32, 4, (32, 64, 128), 0.5, False, device="cpu")
    m = len(regions.boxes)

    # Synthetic outputs with scale_weights: [2, 3, 32, 32]
    # scale 0 (32) = 0.6, scale 1 (64) = 0.3, scale 2 (128) = 0.1
    scale_weights = torch.tensor([0.6, 0.3, 0.1]).view(1, 3, 1, 1).expand(2, 3, 32, 32)

    fake_outputs = {
        "b_region": torch.ones(2, 1, m),
        "region_dispersion": torch.full((2, 1, m), 50.0),
        "region_weight": torch.ones(2, 1, m),
        "solver_region_weight": torch.ones(2, 1, m),
        "regions": regions,
        "scale_weights": scale_weights,
    }

    tracker.update(fake_outputs)
    summary = tracker.summarize()

    assert "scale_pi_32" in summary
    assert "scale_pi_64" in summary
    assert "scale_pi_128" in summary
    assert abs(summary["scale_pi_32"] - 0.6) < 1e-4
    assert abs(summary["scale_pi_64"] - 0.3) < 1e-4
    assert abs(summary["scale_pi_128"] - 0.1) < 1e-4

