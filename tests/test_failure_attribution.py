"""Unit tests for Failure Attribution Audit modules (v2 upgraded)."""
import math
import numpy as np
import pytest
import torch

from hpc.diagnostics.tail_support import (
    compute_crop_percentile,
    compute_dataset_support_profile,
    compute_image_spatial_statistics,
    compute_relative_percentiles,
    profile_crop_support_distribution,
)
from hpc.diagnostics.fg_bg_decomposition import (
    decompose_cell_occupancy_errors,
    decompose_fg_bg_errors,
)
from hpc.diagnostics.multiplicity_calibration import MultiplicityAccumulator


class TestFailureAttributionDiagnostics:
    def test_tail_support_statistics(self):
        # 4 points in a 100x100 box
        pts = np.array([
            [10.0, 10.0],
            [12.0, 10.0],  # dist = 2.0 (< 4 px)
            [50.0, 50.0],
            [55.0, 50.0],  # dist = 5.0 (< 8 px)
        ], dtype=np.float32)

        stats = compute_image_spatial_statistics((100, 100), pts)
        assert stats["gt_count"] == 4.0
        assert stats["density_10k"] == 4.0
        assert stats["nn_min"] == 2.0
        assert stats["nn_frac_lt_4"] == 0.5  # 2 out of 4 points have NN dist < 4
        assert stats["nn_frac_lt_8"] == 1.0  # all points have NN dist < 8
        assert stats["max_y_16"] >= 2.0

    def test_relative_percentiles(self):
        ref = [
            {"gt_count": 10.0, "density_10k": 1.0},
            {"gt_count": 50.0, "density_10k": 5.0},
            {"gt_count": 100.0, "density_10k": 10.0},
            {"gt_count": 200.0, "density_10k": 20.0},
        ]
        query = [
            {"gt_count": 5.0, "density_10k": 0.5},
            {"gt_count": 75.0, "density_10k": 8.0},
            {"gt_count": 500.0, "density_10k": 50.0},
        ]
        pctls = compute_relative_percentiles(query, ref, keys=("gt_count", "density_10k"))
        assert pctls[0]["gt_count_pctl"] == 0.0
        assert pctls[1]["gt_count_pctl"] == 50.0
        assert pctls[2]["gt_count_pctl"] == 100.0

    def test_crop_support_distribution(self):
        # Create 2 mock samples
        sample1 = {
            "image": torch.zeros(3, 300, 300),
            "gt_points": np.array([[50, 50], [60, 60], [200, 200]], dtype=np.float32),
        }
        sample2 = {
            "image": torch.zeros(3, 300, 300),
            "gt_points": np.array([[10, 10]], dtype=np.float32),
        }
        crops = profile_crop_support_distribution([sample1, sample2], crop_size=256, step=128)
        assert len(crops) > 0
        pctl_low = compute_crop_percentile(0, crops)
        pctl_high = compute_crop_percentile(10, crops)
        assert pctl_low >= 0.0
        assert pctl_high == 100.0

    def test_cell_occupancy_decomposition_identity(self):
        # Stride 4 mass map of shape [1, 1, 8, 8] -> stride 16 pooled shape [2, 2]
        pred_mass = torch.ones(1, 1, 8, 8) * 0.5  # total mass = 32.0, each stride-16 cell has 8.0
        targets = {
            16: torch.tensor([[[[10.0, 0.0], [5.0, 0.0]]]]),  # total GT = 15.0
        }
        res = decompose_cell_occupancy_errors(pred_mass, targets, stride=16)
        assert res["gt_total"] == 15.0
        assert res["pred_total"] == 32.0
        assert res["signed_error"] == 17.0
        assert res["n_occupied_cells"] == 2.0
        assert res["n_empty_cells"] == 2.0

        # Empty cell mass: 2 empty cells * 8.0 = 16.0
        assert abs(res["empty_cell_mass"] - 16.0) < 1e-5

        # Check exact arithmetic identity: signed_error == (occupied_surplus - occupied_deficit) + empty_cell_mass
        reconstructed_error = (res["occupied_surplus"] - res["occupied_deficit"]) + res["empty_cell_mass"]
        assert abs(res["signed_error"] - reconstructed_error) < 1e-5

        # Empty compensation = min(occupied_deficit, empty_cell_mass)
        assert res["empty_cell_compensation"] == min(res["occupied_deficit"], res["empty_cell_mass"])

    def test_multiplicity_accumulator_and_bootstrap(self):
        acc1 = MultiplicityAccumulator(strides=(4,), max_k=4)
        acc2 = MultiplicityAccumulator(strides=(4,), max_k=4)

        # Image 1
        m1 = torch.tensor([[[[1.0, 2.0], [2.8, 0.1]]]])
        t1 = {4: torch.tensor([[[[1.0, 2.0], [3.0, 0.0]]]])}
        acc1.add_image(m1, t1)
        acc2.add_image(m1 * 1.1, t1)

        # Image 2
        m2 = torch.tensor([[[[0.9, 1.9], [3.1, 0.2]]]])
        t2 = {4: torch.tensor([[[[1.0, 2.0], [3.0, 0.0]]]])}
        acc1.add_image(m2, t2)
        acc2.add_image(m2 * 1.1, t2)

        summary = acc1.summarize()
        s4 = summary[4]
        assert s4["k_0"]["mean_pred"] == pytest.approx(0.15, abs=1e-5)
        assert s4["k_1"]["mean_pred"] == pytest.approx(0.95, abs=1e-5)
        assert s4["k_2"]["mean_pred"] == pytest.approx(1.95, abs=1e-5)
        assert s4["k_3"]["mean_pred"] == pytest.approx(2.95, abs=1e-5)
        assert s4["k_1"]["n_contributing_images"] == 2.0

        # Cluster bootstrap
        boot = acc1.cluster_bootstrap(stride=4, n_boot=50, seed=42)
        assert "k_1" in boot
        assert not math.isnan(boot["k_1"]["mean"])
        assert boot["k_1"]["ci_lower"] <= boot["k_1"]["ci_upper"]

        # Paired bootstrap diff
        diff = MultiplicityAccumulator.cluster_bootstrap_paired_diff(acc2, acc1, stride=4, n_boot=50, seed=42)
        assert "k_1" in diff
        assert not math.isnan(diff["k_1"]["diff_mean"])
