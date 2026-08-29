"""Unit test for crowd localization metrics & Hungarian distance matching."""

import numpy as np
import torch

from hpc.metrics.localization import (
    extract_points_from_mass_map,
    evaluate_localization_single_image,
    evaluate_dataset_localization,
)


def test_peak_extraction():
    # Synthetic mass map (stride 4) with 3 clear peaks at (10, 10), (20, 20), (30, 30) in mass grid
    mass = np.zeros((40, 40), dtype=np.float32)
    mass[10, 10] = 1.0
    mass[20, 20] = 0.8
    mass[30, 30] = 0.9

    pts = extract_points_from_mass_map(mass, stride=4, threshold_abs=0.1, min_distance_px=8)
    assert len(pts) == 3, f"Expected 3 extracted points, got {len(pts)}"
    
    # Image coordinates should be (col+0.5)*4, (row+0.5)*4 = (42.0, 42.0), (82.0, 82.0), (122.0, 122.0)
    expected = np.array([[42.0, 42.0], [82.0, 82.0], [122.0, 122.0]], dtype=np.float32)
    diff = np.abs(np.sort(pts, axis=0) - np.sort(expected, axis=0))
    assert np.all(diff < 1e-3), "Extracted point coordinates mismatch"
    print("  [✓] Peak extraction from continuous mass map: PASS")


def test_localization_matching():
    # Ground truth: 3 points at (100, 100), (200, 200), (300, 300)
    gt_pts = np.array([[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]], dtype=np.float32)
    
    # Predictions: 2 accurate points (dist=2px, dist=3px), 1 false positive at (50, 50), missing the 3rd point
    pred_pts = np.array([[102.0, 100.0], [200.0, 203.0], [50.0, 50.0]], dtype=np.float32)

    res = evaluate_localization_single_image(pred_pts, gt_pts, distance_thresholds=(4.0, 8.0, 16.0))
    
    # At sigma=4.0: 2 matches (dist <= 4.0), 1 FP, 1 FN -> Precision = 2/3 = 0.667, Recall = 2/3 = 0.667, F1 = 0.667
    assert res[4.0]["tp"] == 2
    assert res[4.0]["fp"] == 1
    assert res[4.0]["fn"] == 1
    assert abs(res[4.0]["f1"] - (2.0 / 3.0)) < 1e-4
    print("  [✓] Single-image Hungarian matching & F1 metric: PASS")


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
