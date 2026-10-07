"""Deep Audit Suite: Batch Independence, Effective Rank, and Solver Contraction.

Verifies:
1. Batch Forward Independence: Forward outputs of sample i are 100% invariant to other samples in batch.
2. Batch Backward Independence: Gradient of sample i w.r.t total batch loss equals (1/B) * grad(sample_i alone).
3. Effective Rank (erank): Feature representation at stride 4 exhibits healthy SVD spectrum without dimensional collapse.
4. Solver Contraction & Monotonicity: Unrolled SIRT solver achieves monotonic energy descent and bounded Lipschitz dynamics.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from rmr_core.operators import build_multiscale_regions
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.solver import unrolled_sirt_solver


def test_batch_forward_independence():
    """Verify sample i output is mathematically invariant to presence of other samples in batch."""
    torch.manual_seed(123)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    x0 = torch.randn(1, 3, 128, 128)
    x1 = torch.randn(1, 3, 128, 128)
    x_batch = torch.cat([x0, x1], dim=0)

    with torch.no_grad():
        out_batch = model(x_batch)
        out0 = model(x0)
        out1 = model(x1)

    diff0 = (out_batch.y[0:1] - out0.y).abs().max().item()
    diff1 = (out_batch.y[1:2] - out1.y).abs().max().item()
    assert diff0 < 1e-5, f"Batch forward leakage detected on sample 0: diff={diff0}"
    assert diff1 < 1e-5, f"Batch forward leakage detected on sample 1: diff={diff1}"


def test_batch_backward_gradient_independence():
    """Verify autograd backward pass has zero cross-sample gradient contamination in batches."""
    torch.manual_seed(42)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    x0 = torch.randn(1, 3, 128, 128, requires_grad=True)
    x1 = torch.randn(1, 3, 128, 128, requires_grad=True)
    x_batch = torch.cat([x0, x1], dim=0)

    pts0 = torch.tensor([[32.0, 32.0], [64.0, 64.0]])
    pts1 = torch.tensor([[16.0, 16.0], [48.0, 48.0], [96.0, 96.0]])
    points_list = [pts0, pts1]

    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="bayesian",
        dm_target="y0",
        count_target="y",
        lambda_cell=0.0,
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_region_nb=0.2,
        output_stride=4,
    )

    out_batch = model(x_batch)
    target_batch = torch.zeros(2, 1, 32, 32)
    losses_batch = compute_rmr_v3_losses(out_batch, target_batch, loss_cfg, points=points_list)
    grad_batch = torch.autograd.grad(losses_batch["total"], x_batch, retain_graph=False)[0]

    out0 = model(x0)
    target0 = torch.zeros(1, 1, 32, 32)
    losses0 = compute_rmr_v3_losses(out0, target0, loss_cfg, points=[pts0])
    grad0 = torch.autograd.grad(losses0["total"], x0, retain_graph=False)[0]

    expected_grad0 = 0.5 * grad0
    diff = (grad_batch[0:1] - expected_grad0).abs().max().item()
    rel_diff = diff / max(expected_grad0.abs().max().item(), 1e-8)
    assert rel_diff < 1e-4, f"Batch gradient contamination detected! rel_diff={rel_diff}"


def test_effective_rank_non_collapsed_spectrum():
    """Verify effective rank erank(P4) > 10.0 out of 32 channels (no dimensional collapse)."""
    torch.manual_seed(99)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False)
    model = RMRv3(cfg).eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        c4, c8, c16 = model.encoder(x)
        p4, p8, p16 = model.fusion(c4, c8, c16)

    feat = p4[0].flatten(1)  # [32, 4096]
    _, S, _ = torch.linalg.svd(feat, full_matrices=False)
    s_norm = S / S.sum()
    entropy = -(s_norm * torch.log(s_norm + 1e-12)).sum().item()
    erank = math.exp(entropy)
    assert erank > 10.0, f"Feature representation collapsed: erank={erank:.2f} / 32"


def test_solver_energy_monotonicity_and_bounded_growth():
    """Verify unrolled SIRT solver energy decreases monotonically and growth is bounded by trust region."""
    torch.manual_seed(42)
    h, w = 32, 32
    regions = build_multiscale_regions(h, w, output_stride=4, region_sizes_px=(16, 32), overlap=0.5)
    m = regions.boxes.shape[0]

    b_solver = torch.full((1, 1, m), 10.0, dtype=torch.float32)
    weight_solver = torch.ones((1, 1, m), dtype=torch.float32)

    y_init = torch.rand(1, 1, h, w) * 0.1
    kappa = 0.35
    iters = 3
    res = unrolled_sirt_solver(
        y0=y_init, b_solver=b_solver, weight_solver=weight_solver, regions=regions,
        iterations=iters, omega=1.0, trust_region_kappa=kappa,
    )

    energy_trace = res["energy_trace"]
    assert len(energy_trace) == iters
    for step in energy_trace:
        assert step["after"].item() <= step["before"].item() + 1e-5

    # Maximum theoretical growth per step is (1 + kappa)
    max_growth_ratio = (1.0 + kappa) ** iters
    actual_growth_ratio = res["y"].max().item() / max(y_init.max().item(), 1e-8)
    assert actual_growth_ratio <= max_growth_ratio + 1e-3, (
        f"Trust region violated: actual growth {actual_growth_ratio:.2f} > max allowed {max_growth_ratio:.2f}"
    )


def test_effective_receptive_field_coverage():
    """Verify ERF (Luo et al., NeurIPS 2016) has 95% radius R95 >= 64 px to cover 128px boxes."""
    torch.manual_seed(42)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    h, w = 256, 256
    x = torch.randn(1, 3, h, w, requires_grad=True)
    out = model(x)
    y0 = out.y0
    ch, cw = y0.shape[2] // 2, y0.shape[3] // 2

    target = y0[0, 0, ch, cw]
    grad = torch.autograd.grad(target, x, retain_graph=False)[0]
    grad_mag = grad.abs().sum(dim=0)  # [H, W]

    in_cy, in_cx = ch * 4 + 2, cw * 4 + 2
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    dist = torch.sqrt((yy - in_cy).float() ** 2 + (xx - in_cx).float() ** 2)

    dist_flat = dist.flatten()
    grad_flat = grad_mag.flatten()
    idx = torch.argsort(dist_flat)

    cum_grad = torch.cumsum(grad_flat[idx], dim=0)
    total_grad = cum_grad[-1]

    r95_idx = torch.searchsorted(cum_grad, 0.95 * total_grad)
    r95 = dist_flat[idx][r95_idx].item()
    assert r95 >= 64.0, f"Effective Receptive Field too small: R95={r95:.2f} < 64 px"


def test_softplus_logit_health_and_dead_neurons():
    """Verify FineCarrierHead Softplus logit health: no dead neurons (z0 < -6) and positive gradient gain."""
    torch.manual_seed(42)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        c4, c8, c16 = model.encoder(x)
        p4, _, _ = model.fusion(c4, c8, c16)
        z0 = model.fine_head(p4)

    dead_ratio = (z0 < -6.0).float().mean().item()
    grad_gain = torch.sigmoid(z0).mean().item()

    assert dead_ratio < 0.05, f"High dead neuron ratio in FineCarrierHead: {dead_ratio * 100:.2f}%"
    assert grad_gain > 0.005, f"Softplus gradient gain vanishing: {grad_gain:.6f}"


def test_single_batch_overfit_and_loss_decrease():
    """Karpathy & Chase Roberts sanity check: verify optimizer strictly reduces loss on single batch."""
    torch.manual_seed(42)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(1, 3, 128, 128)
    pts = torch.tensor([[32.0, 32.0], [64.0, 64.0]])
    target = torch.zeros(1, 1, 32, 32)
    loss_cfg = RMRv3LossConfig(
        allocation_loss_type="bayesian",
        dm_target="y0",
        count_target="y",
        lambda_cell=0.0,
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_region_nb=0.2,
        output_stride=4,
    )

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        out = model(x)
        loss_dict = compute_rmr_v3_losses(out, target, loss_cfg, points=[pts])
        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"Loss failed to decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    for i in range(len(losses) - 1):
        assert losses[i + 1] <= losses[i] + 1e-4, f"Non-monotonic loss oscillation: {losses}"


def test_dynamic_resolution_and_non_negativity():
    """Verify arbitrary non-square input dimensions yield exact stride-4 non-negative outputs without NaN/Inf."""
    torch.manual_seed(42)
    cfg = RMRv3Config(feature_width=32, output_stride=4, pretrained=False, iterations=2)
    model = RMRv3(cfg).eval()

    test_shapes = [(384, 512), (288, 400), (352, 608)]
    for h, w in test_shapes:
        x = torch.randn(1, 3, h, w)
        with torch.no_grad():
            out = model(x)
        assert out.y.shape == (1, 1, h // 4, w // 4)
        assert out.y0.shape == (1, 1, h // 4, w // 4)
        assert not torch.isnan(out.y).any(), f"NaN detected for shape ({h}, {w})"
        assert not torch.isinf(out.y).any(), f"Inf detected for shape ({h}, {w})"
        assert (out.y >= 0.0).all(), f"Negative density detected for shape ({h}, {w})"

