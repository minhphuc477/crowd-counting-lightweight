"""Unit & Integration Test Suite for Stride-4 DTM Tree and OT-M Localization."""

from __future__ import annotations

import numpy as np
import torch

from hpc.data.point_counts import build_exact_count_pyramid, points_to_y4
from hpc.losses.ntpc import (
    NTPCConfig,
    NTPCLoss,
    dm_from_mass,
    sum_pool_mass_pyramid,
)
from hpc.metrics.localization import evaluate_dataset_localization, localization_metrics, match_points
from hpc.metrics.otm import otm_localize, sinkhorn_log
from hpc.models.hpc_lite import HPCLite


def test_mass_conservation():
    mass = torch.rand(2, 1, 64, 64) * 0.1
    pred = sum_pool_mass_pyramid(mass)
    total = mass.flatten(1).sum(1)

    for b in (4, 8, 16, 32, 64):
        level_total = pred[b].flatten(1).sum(1)
        assert torch.allclose(level_total, total, atol=1e-4, rtol=1e-5), f"Level {b} violates mass conservation"


def test_dm_prefers_correct_allocation():
    y = torch.tensor([[12.0, 5.0, 2.0, 1.0]])
    good = torch.tensor([[12.0, 5.0, 2.0, 1.0]])
    bad = torch.tensor([[5.0, 5.0, 5.0, 5.0]])

    good_loss = dm_from_mass(y, good, kappa=20.0).item()
    bad_loss = dm_from_mass(y, bad, kappa=20.0).item()

    assert good_loss < bad_loss, f"Expected good_loss ({good_loss:.4f}) < bad_loss ({bad_loss:.4f})"


def test_point_tree_exact_integer_conservation():
    torch.manual_seed(42)
    pts = [torch.rand(57, 2) * 255.0]
    tree = build_exact_count_pyramid(pts, 256, 256)

    assert tree["N"].item() == 57.0
    for b in (4, 8, 16, 32, 64):
        assert tree[b].sum().item() == 57.0
        assert torch.equal(tree[b], tree[b].round())


def test_otm_cardinality_and_localization():
    mass = torch.zeros(32, 32)
    mass[5, 5] = 1.0
    mass[10, 10] = 1.0
    mass[20, 20] = 1.0

    points = otm_localize(mass, seed=42, max_source_points=None)
    assert len(points) == 3, f"Expected 3 points, got {len(points)}"

    gt_points = np.array([[22.0, 22.0], [42.0, 42.0], [82.0, 82.0]], dtype=np.float32)
    tp, fp, fn = match_points(points.numpy(), gt_points, threshold=4.0)
    assert tp == 3 and fp == 0 and fn == 0, f"Expected perfect match, got tp={tp}, fp={fp}, fn={fn}"


def test_full_ntpc_loss_stride4_backward():
    mass = (torch.rand(2, 1, 64, 64) * 0.1).detach().requires_grad_(True)
    pts = [
        torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]),
        torch.tensor([[5.0, 5.0]]),
    ]
    tree = build_exact_count_pyramid(pts, height=256, width=256, block_sizes=(4, 8, 16, 32, 64))
    targets = {b: tree[b] for b in (4, 8, 16, 32, 64)}
    targets["N"] = tree["N"]

    criterion = NTPCLoss(NTPCConfig(mode="r4_dtm_tree4"))
    loss, components = criterion(mass, targets)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    loss.backward()

    assert mass.grad is not None
    assert torch.isfinite(mass.grad).all()
    assert (mass.grad.abs().sum() > 0)
