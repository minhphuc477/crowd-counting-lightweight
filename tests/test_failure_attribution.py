"""Unit tests for Failure Attribution Audit modules."""
import math
import numpy as np
import pytest
import torch

from hpc.diagnostics.tail_support import (
    compute_dataset_support_profile,
    compute_image_spatial_statistics,
    compute_relative_percentiles,
)
from hpc.diagnostics.fg_bg_decomposition import decompose_fg_bg_errors
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

    def test_fg_bg_decomposition_identity(self):
        # Stride 4 mass map of shape [1, 1, 8, 8] -> stride 16 pooled shape [2, 2]
        pred_mass = torch.ones(1, 1, 8, 8) * 0.5  # total mass = 32.0, each stride-16 cell has 8.0
        targets = {
            16: torch.tensor([[[[10.0, 0.0], [5.0, 0.0]]]]),  # total GT = 15.0
        }
        res = decompose_fg_bg_errors(pred_mass, targets, stride=16)
        assert res["gt_total"] == 15.0
        assert res["pred_total"] == 32.0
        assert res["signed_error"] == 17.0
        assert res["n_fg_cells"] == 2.0
        assert res["n_bg_cells"] == 2.0

        # Background false positive mass: 2 empty cells * 8.0 = 16.0
        assert abs(res["bg_pred"] - 16.0) < 1e-5

        # Check exact arithmetic identity: signed_error == (fg_surplus - fg_deficit) + bg_pred
        reconstructed_error = (res["fg_surplus"] - res["fg_deficit"]) + res["bg_pred"]
        assert abs(res["signed_error"] - reconstructed_error) < 1e-5

        # BG compensation = min(fg_deficit, bg_pred)
        assert res["bg_compensation"] == min(res["fg_deficit"], res["bg_pred"])

    def test_multiplicity_accumulator(self):
        acc = MultiplicityAccumulator(strides=(4,), max_k=4)
        m_s4 = torch.tensor([[[[1.0, 2.0], [2.8, 0.1]]]])
        targets = {
            4: torch.tensor([[[[1.0, 2.0], [3.0, 0.0]]]]),
        }
        acc.add_image(m_s4, targets)
        summary = acc.summarize()
        s4 = summary[4]
        assert s4["k_0"]["mean_pred"] == pytest.approx(0.1, abs=1e-5)
        assert s4["k_1"]["mean_pred"] == pytest.approx(1.0, abs=1e-5)
        assert s4["k_2"]["mean_pred"] == pytest.approx(2.0, abs=1e-5)
        assert s4["k_3"]["mean_pred"] == pytest.approx(2.8, abs=1e-5)
        assert s4["k_4"]["n_cells"] == 0.0
