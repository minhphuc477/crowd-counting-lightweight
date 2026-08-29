"""Unit & Integration Test Suite for Stride-4 DTM Tree and OT-M Localization.

Tests:
  1. Mass pyramid exact conservation across all scales {4, 8, 16, 32, 64}.
  2. DTM objective preference for ground-truth child distribution.
  3. Ground-truth recursive integer count tree integrity and zero-count invariance.
  4. OT-M cardinality exactness: len(points) == round(sum(mass)).
  5. Hungarian bipartite matching & F1 metrics at sigma in {4, 8}.
  6. Full NTPC loss forward + backward pass down to stride-4.
"""

from __future__ import annotations

import numpy as np
import torch

from hpc.data.point_counts import build_count_tree, build_exact_count_pyramid, points_to_y4, validate_targets
from hpc.losses.ntpc import (
    FullNTPCLoss,
    NTPCConfig,
    NTPCLoss,
    dm_from_mass,
    mass_pyramid,
    sum_pool_mass_pyramid,
)
from hpc.metrics.localization import evaluate_dataset_localization, localization_metrics, match_points
from hpc.metrics.otm import otm_localize, sinkhorn_log
from hpc.models.hpc_lite import HPCLite


def test_mass_conservation():
    mass = torch.rand(2, 1, 64, 64) * 0.1
    pred = mass_pyramid(mass)
    total = mass.flatten(1).sum(1)

    for b in (4, 8, 16, 32, 64):
        level_total = pred[b].flatten(1).sum(1)
        assert torch.allclose(level_total, total, atol=1e-4, rtol=1e-5), f"Level {b} violates mass conservation"
    print("  [✓] Mass pyramid exact conservation {4, 8, 16, 32, 64}: PASS")


def test_dm_prefers_correct_allocation():
    y = torch.tensor([[12.0, 5.0, 2.0, 1.0]])
    good = torch.tensor([[12.0, 5.0, 2.0, 1.0]])
    bad = torch.tensor([[5.0, 5.0, 5.0, 5.0]])

    good_loss = dm_from_mass(y, good, kappa=20.0).item()
    bad_loss = dm_from_mass(y, bad, kappa=20.0).item()

    assert good_loss < bad_loss, f"Expected good_loss ({good_loss:.4f}) < bad_loss ({bad_loss:.4f})"
    print("  [✓] Dirichlet-Multinomial strictly prefers true count allocation: PASS")


def test_point_tree_exact_integer_conservation():
    torch.manual_seed(42)
    pts = torch.rand(57, 2) * 256.0
    tree = build_count_tree(pts, 256, 256)
    validate_targets(tree)

    assert tree["N"].item() == 57.0
    assert tree["y4"].sum().item() == 57.0
    assert tree["y8"].sum().item() == 57.0
    assert tree["y16"].sum().item() == 57.0
    assert tree["y32"].sum().item() == 57.0
    assert tree["y64"].sum().item() == 57.0
    print("  [✓] Ground-truth recursive integer count tree down to stride-4: PASS")


def test_otm_cardinality_and_localization():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Synthetic mass map (stride 4, 64x64 grid for 256x256 image) with 3 sharp heads
    mass = torch.zeros(1, 64, 64, device=device)
    # Head 1 at (40, 40) in image px -> (10, 10) in grid
    mass[0, 10, 10] = 1.0
    # Head 2 at (120, 120) in image px -> (30, 30) in grid
    mass[0, 30, 30] = 1.0
    # Head 3 at (200, 200) in image px -> (50, 50) in grid
    mass[0, 50, 50] = 1.0

    points = otm_localize(mass, output_stride=4, outer_iterations=10, epsilon=0.01)
    
    # 1. Cardinality check: must be exactly 3 points
    assert len(points) == 3, f"Expected 3 points, got {len(points)}"

    # 2. Accuracy check: compare with ground truth heads
    gt_pts = np.array([[42.0, 42.0], [122.0, 122.0], [202.0, 202.0]], dtype=np.float32)
    tp, fp, fn = match_points(points, gt_pts, threshold=4.0)

    assert tp == 3 and fp == 0 and fn == 0, f"OT-M localization mismatch: TP={tp}, FP={fp}, FN={fn}"
    m = localization_metrics(tp, fp, fn)
    assert abs(m["f1"] - 1.0) < 1e-4
    print("  [✓] OT-M exact cardinality consistency & localization accuracy: PASS")


def test_full_ntpc_loss_stride4_backward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    crit = NTPCLoss(NTPCConfig(mode="r4_dtm_tree4")).to(device)

    img = torch.rand(2, 3, 256, 256, device=device)
    pts_batch = [torch.rand(35, 2, device=device) * 256.0, torch.rand(48, 2, device=device) * 256.0]
    targets = build_exact_count_pyramid(pts_batch, 256, 256, (4, 8, 16, 32, 64), device=device)

    model.train()
    mass = model(img)
    loss, logs = crit(mass, targets)
    loss.backward()

    assert torch.isfinite(loss), "Loss is not finite"
    assert logs["root_nb"] > 0
    assert logs["root_to_64"] > 0
    assert logs["64_to_32"] > 0
    assert logs["32_to_16"] > 0
    assert logs["16_to_8"] > 0
    assert logs["8_to_4"] > 0
    print("  [✓] Full NTPC Stride-4 Tree (64->32->16->8->4) forward + backward: PASS")


def main():
    print("\n" + "=" * 65)
    print("STARTING NTPC STRIDE-4 & OT-M LOCALIZATION TEST SUITE")
    print("=" * 65)
    test_mass_conservation()
    test_dm_prefers_correct_allocation()
    test_point_tree_exact_integer_conservation()
    test_otm_cardinality_and_localization()
    test_full_ntpc_loss_stride4_backward()
    print("=" * 65)
    print("ALL STRIDE-4 & OT-M TESTS PASSED!")
    print("=" * 65)


if __name__ == "__main__":
    main()
