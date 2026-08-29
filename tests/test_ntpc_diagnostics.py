import math

import torch

from hpc.metrics.subgroup import evaluate_subgroup_diagnostics
from hpc.metrics.tree import finalize_tree_diagnostics, tree_allocation_raw_diagnostics
from hpc.losses.ntpc import NTPCConfig, block_sum, sum_pool_mass_pyramid
from train_ntpc import gradient_norm, nonfinite_gradient_report


def test_gradient_helpers_report_finite_and_nonfinite_parameters():
    model = torch.nn.Linear(2, 1)
    model(torch.ones(1, 2)).sum().backward()
    assert math.isfinite(float(gradient_norm(model.parameters())))
    assert nonfinite_gradient_report(model) == []
    model.weight.grad[0, 0] = float("inf")
    assert not math.isfinite(float(gradient_norm(model.parameters())))
    report = nonfinite_gradient_report(model)
    assert len(report) == 1 and report[0].startswith("weight:") and "inf=1" in report[0]


def test_subgroup_diagnostics_include_counts_calibration_and_ratios():
    result = evaluate_subgroup_diagnostics([0.5, 8.0, 250.0, 800.0, 1200.0], [0, 10, 200, 1000, 1500])
    assert result["bin_sparse_count"] == 3
    assert result["bin_medium_count"] == 0 if "bin_medium_count" in result else True
    assert result["bin_dense_count"] == 2
    assert result["top10_dense_count"] == 1
    assert result["empty_count"] == 1
    assert "signed_error_median" in result
    assert math.isclose(result["bin_dense_pred_gt_ratio"], 2000.0 / 2500.0)


def test_tree_diagnostics_normalize_by_active_parents():
    mass = torch.ones(1, 1, 16, 16)
    targets = {"N": torch.tensor([2.0])}
    targets.update(sum_pool_mass_pyramid(torch.zeros_like(mass), (16, 32, 64)))
    # Put two points in separate stride-16 cells; coarser maps remain conserved.
    targets[16][0, 0, 0] = 1
    targets[16][0, 1, 1] = 1
    targets[32] = block_sum(targets[16], 2)
    targets[64] = block_sum(targets[32], 2)
    raw = tree_allocation_raw_diagnostics(mass, targets, NTPCConfig(mode="r4_dtm_tree16"))
    metrics = finalize_tree_diagnostics(raw)
    assert metrics["tree_root_64_active_parents_per_image"] == 1.0
    assert metrics["tree_32_16_active_parents_per_image"] == 1.0
    assert metrics["tree_32_16_zero_parent_fraction"] == 0.75
    assert math.isfinite(metrics["tree_32_16_nll_per_active_parent"])
