"""Unit test for crowd localization metrics & Hungarian distance matching."""

import numpy as np
import torch

from hpc.metrics.localization import (
    extract_points_from_mass_map,
    evaluate_localization_single_image,
    evaluate_dataset_localization,
    match_points,
    localization_metrics,
)


def test_peak_extraction():
    mass = np.zeros((40, 40), dtype=np.float32)
    mass[10, 10] = 1.0
    mass[20, 20] = 0.8
    mass[30, 30] = 0.9

    pts = extract_points_from_mass_map(mass, stride=4, threshold_abs=0.1, min_distance_px=8)
    assert len(pts) == 3, f"Expected 3 extracted points, got {len(pts)}"
    
    expected = np.array([[42.0, 42.0], [82.0, 82.0], [122.0, 122.0]], dtype=np.float32)
    diff = np.abs(np.sort(pts, axis=0) - np.sort(expected, axis=0))
    assert np.all(diff < 1e-3), "Extracted point coordinates mismatch"
    print("  [✓] Peak extraction from continuous mass map: PASS")


def test_localization_matching():
    gt_pts = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]], dtype=np.float32)
    pred_pts = np.array([[102.0, 100.0], [200.0, 203.0], [50.0, 50.0]], dtype=np.float32)

    tp, fp, fn = match_points(pred_pts, gt_pts, threshold=4.0)
    assert tp == 2 and fp == 1 and fn == 1
    m = localization_metrics(tp, fp, fn)
    assert abs(m["f1"] - (2.0 / 3.0)) < 1e-4
    print("  [✓] Single-image Hungarian matching & F1 metric: PASS")


def test_hungarian_maximizes_valid_matches_not_greedy_order():
    # Greedy nearest-pair matching consumes the shared GT with pred[0] and
    # returns one match. Global one-to-one assignment returns both matches.
    pred_pts = np.array([[1.0, 0.0], [-1.1, 0.0]], dtype=np.float32)
    gt_pts = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    tp, fp, fn = match_points(pred_pts, gt_pts, threshold=2.1)
    assert (tp, fp, fn) == (2, 0, 0)


def test_dataset_localization():
    preds_list = [np.array([[10.0, 10.0]]), np.array([[50.0, 50.0], [60.0, 60.0]])]
    gts_list = [np.array([[10.0, 10.0]]), np.array([[50.0, 50.0], [60.0, 60.0]])]

    summary = evaluate_dataset_localization(preds_list, gts_list, distance_thresholds=(4.0, 8.0))
    assert summary["sigma_4_precision"] == 1.0
    assert summary["sigma_4_recall"] == 1.0
    assert summary["sigma_4_f1"] == 1.0
    print("  [✓] Dataset-level localization aggregation: PASS")


if __name__ == "__main__":
    print("Running localization test suite:")
    test_peak_extraction()
    test_localization_matching()
    test_dataset_localization()
    print("All localization tests PASSED!")
