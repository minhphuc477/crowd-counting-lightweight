"""Unit tests for the comprehensive MICF checkpoint evaluator.

Tests cover:
1. count_metric_summary (analytical hand calculations for MAE, RMSE, NAE, SRE, signed bias, percentiles)
2. window_metric_summary (micro vs macro PMAE/PRMSE, full vs edge window partition, empty vs non-empty windows)
3. measure_validity_metrics & aggregate_validity_metrics (macro vs micro violation rate & negative mass ratio)
4. game_pixel_space_errors (exact continuous integral conservation, boundary handling, exact GT matching)
5. game_stride16_errors (diagnostic discrete grid partitioning)
6. compute_cancellation_ratio (mathematical bounds [0, 1], same-sign vs cancelling errors)
7. representation_metrics (conservation error between C[-1, -1] and sum(Y))
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import torch

from tools.eval_micf_comprehensive import (
    aggregate_validity_metrics,
    compute_cancellation_ratio,
    count_metric_summary,
    game_pixel_space_errors,
    game_stride16_errors,
    measure_validity_metrics,
    representation_metrics,
    window_metric_summary,
)
from hpc.losses.micf import (
    cell_counts_to_cumulative_field,
    discrete_mixed_difference,
)


class TestCountMetrics:
    def test_count_metric_summary_analytical(self):
        pred = [10.0, 20.0, 30.0]
        gt = [10.0, 25.0, 20.0]
        # err = [0.0, -5.0, 10.0]
        # ae  = [0.0, 5.0, 10.0]

        summary = count_metric_summary(pred, gt)

        assert pytest.approx(summary["mae"], abs=1e-6) == 5.0
        # rmse = sqrt((0 + 25 + 100) / 3) = sqrt(125/3) = 6.45497224
        assert pytest.approx(summary["rmse"], abs=1e-6) == math.sqrt(125.0 / 3.0)
        # nae = (0/10 + 5/25 + 10/20) / 3 = (0 + 0.2 + 0.5) / 3 = 0.7 / 3
        assert pytest.approx(summary["nae"], abs=1e-6) == 0.7 / 3.0
        # sre = sqrt((0 + 25/25 + 100/20) / 3) = sqrt((0 + 1 + 5) / 3) = sqrt(2.0)
        assert pytest.approx(summary["sre"], abs=1e-6) == math.sqrt(2.0)
        # signed_bias = (0 - 5 + 10) / 3 = 5/3
        assert pytest.approx(summary["signed_bias"], abs=1e-6) == 5.0 / 3.0
        assert pytest.approx(summary["median_ae"], abs=1e-6) == 5.0
        assert pytest.approx(summary["max_ae"], abs=1e-6) == 10.0
        assert pytest.approx(summary["p90_ae"], abs=1e-6) == 9.0
        assert pytest.approx(summary["p95_ae"], abs=1e-6) == 9.5

    def test_count_metric_summary_empty(self):
        summary = count_metric_summary([], [])
        assert math.isnan(summary["mae"])
        assert math.isnan(summary["rmse"])

    def test_count_metric_summary_shape_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            count_metric_summary([1.0, 2.0], [1.0])


class TestWindowMetrics:
    def test_window_metric_summary_micro_vs_macro(self):
        """Micro pools all windows equally; Macro averages per-image window errors."""
        # Image 0 has 1 window with error = 0
        # Image 1 has 3 windows with error = 10 each
        rows = [
            {"image_index": 0, "pred_count": 10.0, "gt_count": 10.0, "is_full_window": True},
            {"image_index": 1, "pred_count": 20.0, "gt_count": 10.0, "is_full_window": True},
            {"image_index": 1, "pred_count": 20.0, "gt_count": 10.0, "is_full_window": True},
            {"image_index": 1, "pred_count": 20.0, "gt_count": 10.0, "is_full_window": True},
        ]
        summary = window_metric_summary(rows)

        # Micro: 4 windows, errors = [0, 10, 10, 10] -> MAE = 30 / 4 = 7.5
        assert pytest.approx(summary["window_mae_micro"], abs=1e-6) == 7.5
        assert pytest.approx(summary["window_mae"], abs=1e-6) == 7.5

        # Macro: Image 0 MAE = 0.0, Image 1 MAE = 10.0 -> Macro MAE = (0 + 10) / 2 = 5.0
        assert pytest.approx(summary["window_mae_macro"], abs=1e-6) == 5.0
        assert summary["window_mae_micro"] != summary["window_mae_macro"]

    def test_window_full_vs_edge_partition(self):
        """Full 256x256 windows vs partial boundary edge windows."""
        rows = [
            # 2 full windows: errors = [0, 4] -> MAE = 2.0
            {"image_index": 0, "pred_count": 10.0, "gt_count": 10.0, "is_full_window": True},
            {"image_index": 0, "pred_count": 14.0, "gt_count": 10.0, "is_full_window": True},
            # 2 edge windows: errors = [-6, 8] -> MAE = 7.0
            {"image_index": 0, "pred_count": 4.0, "gt_count": 10.0, "is_full_window": False},
            {"image_index": 0, "pred_count": 18.0, "gt_count": 10.0, "is_full_window": False},
        ]
        summary = window_metric_summary(rows)

        assert pytest.approx(summary["full_window_mae"], abs=1e-6) == 2.0
        assert summary["full_window_count"] == 2
        assert pytest.approx(summary["edge_window_mae"], abs=1e-6) == 7.0
        assert summary["edge_window_count"] == 2

    def test_empty_vs_nonempty_windows(self):
        rows = [
            {"image_index": 0, "pred_count": 2.0, "gt_count": 0.0, "is_full_window": True},
            {"image_index": 0, "pred_count": 3.0, "gt_count": 0.0, "is_full_window": True},
            {"image_index": 0, "pred_count": 12.0, "gt_count": 10.0, "is_full_window": True},
        ]
        summary = window_metric_summary(rows)

        assert pytest.approx(summary["empty_window_mae"], abs=1e-6) == 2.5
        assert pytest.approx(summary["empty_window_mean_prediction"], abs=1e-6) == 2.5
        assert pytest.approx(summary["empty_window_fraction"], abs=1e-6) == 2.0 / 3.0
        assert pytest.approx(summary["nonempty_window_mae"], abs=1e-6) == 2.0
        assert pytest.approx(summary["nae_nonzero"], abs=1e-6) == 2.0 / 10.0


class TestMeasureValidity:
    def test_measure_validity_metrics_basic(self):
        # 4 cells: 3 positive [1.0, 2.0, 3.0] sum=6.0, 1 negative [-2.0]
        y = torch.tensor([[[[1.0, 2.0], [3.0, -2.0]]]])
        metrics = measure_validity_metrics(y)

        # 1 negative cell out of 4 cells -> 0.25
        assert pytest.approx(metrics["violation_rate"], abs=1e-6) == 0.25
        # violation_magnitude = mean(-min(y, 0)) = (0 + 0 + 0 + 2) / 4 = 0.5
        assert pytest.approx(metrics["violation_magnitude"], abs=1e-6) == 0.5
        # negative_mass_total = 2.0
        assert pytest.approx(metrics["negative_mass_total"], abs=1e-6) == 2.0
        # positive_mass_total = 1 + 2 + 3 = 6.0
        assert pytest.approx(metrics["positive_mass_total"], abs=1e-6) == 6.0
        # canonical negative_mass_ratio = 2.0 / 6.0 = 0.333333...
        assert pytest.approx(metrics["negative_mass_ratio"], abs=1e-4) == 2.0 / 6.0

    def test_aggregate_validity_macro_vs_micro(self):
        """Unequal cell counts across images demonstrate macro vs micro weighting."""
        # Image 0: small (100 cells), 10 negative cells, neg_mass=5.0, pos_mass=15.0
        # Image 1: large (900 cells), 10 negative cells, neg_mass=5.0, pos_mass=175.0
        rows = [
            {
                "violation_rate": 10 / 100,  # 0.10
                "violation_magnitude": 5.0 / 100,  # 0.05
                "negative_mass_ratio": 5.0 / 15.0,  # 0.3333...
                "neg_cell_count": 10,
                "total_cells": 100,
                "negative_mass_total": 5.0,
                "positive_mass_total": 15.0,
            },
            {
                "violation_rate": 10 / 900,  # 0.011111...
                "violation_magnitude": 5.0 / 900,  # 0.005555...
                "negative_mass_ratio": 5.0 / 175.0,  # 0.028571...
                "neg_cell_count": 10,
                "total_cells": 900,
                "negative_mass_total": 5.0,
                "positive_mass_total": 175.0,
            },
        ]
        agg = aggregate_validity_metrics(rows)

        # Macro VR: (0.10 + 0.011111...) / 2 = 0.055555...
        expected_macro_vr = 0.5 * (0.10 + 10.0 / 900.0)
        assert pytest.approx(agg["macro_violation_rate"], abs=1e-6) == expected_macro_vr

        # Micro VR: (10 + 10) / (100 + 900) = 20 / 1000 = 0.02
        assert pytest.approx(agg["micro_violation_rate"], abs=1e-6) == 0.02

        # Macro NMR: (5/15 + 5/175) / 2
        expected_macro_nmr = 0.5 * (5.0 / 15.0 + 5.0 / 175.0)
        assert pytest.approx(agg["macro_negative_mass_ratio"], abs=1e-6) == expected_macro_nmr

        # Micro NMR: (5 + 5) / (15 + 175) = 10 / 190
        expected_micro_nmr = 10.0 / 190.0
        assert pytest.approx(agg["micro_negative_mass_ratio"], abs=1e-4) == expected_micro_nmr

        assert agg["macro_violation_rate"] != agg["micro_violation_rate"]


class TestGAME:
    def test_game_pixel_space_exact_conservation(self):
        """Total predicted count in pixel-space GAME partitions must sum to total measure."""
        # 16x16 grid with stride 16 -> 256x256 image
        # Each cell has 0.5 count -> total = 128.0
        y = torch.full((1, 1, 16, 16), 0.5, dtype=torch.float32)
        stride = 16
        img_h, img_w = 256, 256

        # Zero points -> GT=0 everywhere
        pts_empty = np.empty((0, 2), dtype=np.float32)
        errors = game_pixel_space_errors(y, pts_empty, img_h, img_w, stride, levels=[0, 1, 2, 3])

        # Level 0 error should be |128.0 - 0.0| = 128.0
        assert pytest.approx(errors[0], abs=1e-4) == 128.0

        # Now test with points matching exactly
        # 128 points inside bounds
        pts = np.zeros((128, 2), dtype=np.float32)
        pts[:, 0] = 50.0
        pts[:, 1] = 50.0
        errors_matched = game_pixel_space_errors(y, pts, img_h, img_w, stride, levels=[0])
        # Level 0: Total pred=128.0, Total GT=128.0 -> error = 0.0
        assert pytest.approx(errors_matched[0], abs=1e-4) == 0.0

    def test_game_stride16_errors(self):
        y_pred = torch.full((1, 1, 16, 16), 1.0, dtype=torch.float32)
        y_gt = torch.full((1, 1, 16, 16), 1.0, dtype=torch.float32)

        # Perfect match -> error is 0.0 across all levels
        errors = game_stride16_errors(y_pred, y_gt, levels=[0, 1, 2, 3])
        for lvl in [0, 1, 2, 3]:
            assert pytest.approx(errors[lvl], abs=1e-6) == 0.0

        # Mismatch of 1.0 per cell -> 256 cells -> 256 total error
        y_pred_diff = torch.full((1, 1, 16, 16), 2.0, dtype=torch.float32)
        errors_diff = game_stride16_errors(y_pred_diff, y_gt, levels=[0, 1, 2])
        for lvl in [0, 1, 2]:
            assert pytest.approx(errors_diff[lvl], abs=1e-6) == 256.0


class TestCancellation:
    def test_compute_cancellation_ratio(self):
        # Same sign -> no cancellation
        assert pytest.approx(compute_cancellation_ratio([1.0, 2.0, 3.0]), abs=1e-6) == 0.0
        assert pytest.approx(compute_cancellation_ratio([-1.0, -2.0, -3.0]), abs=1e-6) == 0.0

        # Perfect cancellation
        assert pytest.approx(compute_cancellation_ratio([5.0, -5.0]), abs=1e-6) == 1.0

        # Partial cancellation: net = 5.0, abs_sum = 15.0 -> 1 - 5/15 = 2/3
        assert pytest.approx(compute_cancellation_ratio([10.0, -5.0]), abs=1e-6) == 2.0 / 3.0

        # Empty or zero
        assert compute_cancellation_ratio([]) == 0.0
        assert compute_cancellation_ratio([0.0, 0.0]) == 0.0


class TestRepresentation:
    def test_representation_metrics_conservation(self):
        """When C is the 2D cumsum of Y, conservation error is 0.0."""
        y = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        c = cell_counts_to_cumulative_field(y, orientation="TL")

        metrics = representation_metrics(c, y, c, y, gt_count=10.0)

        assert pytest.approx(metrics["cumulative_field_nmae"], abs=1e-6) == 0.0
        assert pytest.approx(metrics["measure_nl1"], abs=1e-6) == 0.0
        assert pytest.approx(metrics["conservation_error"], abs=1e-6) == 0.0
