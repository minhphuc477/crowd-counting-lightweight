from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from rmr_count.aggregate import aggregate_summaries, compare_predictions
from rmr_count.eval import evaluate
from rmr_count.model import RMRConfig, RMRCount


def create_synthetic_eval_loader(stride: int = 4, size: int = 128):
    """Create a mock dataloader with 2 samples (one with crowd points, one zero-GT)."""
    gh, gw = size // stride, size // stride

    # Sample 1: 5 points
    pts = torch.tensor([[10.0, 15.0], [20.0, 25.0], [30.0, 35.0], [50.0, 60.0], [70.0, 80.0]], dtype=torch.float32)
    tgt1 = torch.zeros((1, gh, gw), dtype=torch.float32)
    for p in pts:
        cy = min(gh - 1, int(p[0] / stride))
        cx = min(gw - 1, int(p[1] / stride))
        tgt1[0, cy, cx] += 1.0

    sample1 = {
        "id": "img_001",
        "image": torch.randn(3, size, size),
        "target_y": tgt1,
        "points": pts,
        "height": size,
        "width": size,
    }

    # Sample 2: zero-GT
    sample2 = {
        "id": "img_002",
        "image": torch.randn(3, size, size),
        "target_y": torch.zeros((1, gh, gw), dtype=torch.float32),
        "points": torch.zeros((0, 2), dtype=torch.float32),
        "height": size,
        "width": size,
    }

    dataset = [sample1, sample2]

    def collate_fn(batch):
        return batch

    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


@pytest.mark.parametrize("variant", [
    "direct",
    "region_loss",
    "region_aux",
    "local_refine",
    "learned_project",
    "rmr",
])
def test_evaluate_all_variants(variant: str):
    """Verify that evaluate() runs successfully for all 6 model variants."""
    cfg = RMRConfig(
        feature_width=8,
        region_sizes_px=(32, 64),
        region_overlap=0.5,
        iterations=2,
    )
    model = RMRCount(cfg, variant=variant).eval()
    device = torch.device("cpu")
    loader = create_synthetic_eval_loader(stride=4, size=128)

    rows, summary, regional_trace, solver_trace = evaluate(
        model=model,
        loader=loader,
        device=device,
        tile_size=128,
        practical_halo=0,
        region_mode="predicted",
    )

    # 2 samples in loader
    assert len(rows) == 2
    for r in rows:
        assert "id" in r
        assert "gt" in r
        assert "pred" in r
        assert "abs_err" in r
        assert "GAME0" in r
        assert "GAME1" in r

    # Summary metrics (uppercase in rmr_count.metrics)
    assert "MAE" in summary
    assert "RMSE" in summary
    assert "NAE" in summary
    assert summary["MAE"] >= 0.0

    # Regional scale metrics are computed for all variants (post-hoc for direct & local_refine)
    assert any("RegionMAE_px_" in k for k in summary.keys())

    # If model has regional head, check b_R accuracy metrics
    if model.region_head is not None:
        assert "BHead_Pearson_Correlation" in summary
    else:
        assert "BHead_Pearson_Correlation" not in summary

    # Check solver traces for iterative variants (rmr, learned_project, and local_refine)
    if variant in ("rmr", "learned_project", "local_refine"):
        assert len(solver_trace) > 0
        assert "iteration" in solver_trace[0]
    else:
        assert len(solver_trace) == 0


def test_evaluator_file_output(tmp_path: Path):
    """Verify that evaluate() outputs write valid CSV and JSON artifacts."""
    cfg = RMRConfig(feature_width=8, region_sizes_px=(32, 64), iterations=2)
    model = RMRCount(cfg, variant="rmr").eval()
    device = torch.device("cpu")
    loader = create_synthetic_eval_loader(stride=4, size=128)

    rows, summary, regional_trace, solver_trace = evaluate(
        model=model,
        loader=loader,
        device=device,
        tile_size=128,
        practical_halo=0,
    )

    out_dir = tmp_path / "eval_run"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    solver_path = out_dir / "solver_trace.csv"
    with solver_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(solver_trace[0].keys()))
        writer.writeheader()
        writer.writerows(solver_trace)

    regional_path = out_dir / "regional_trace.csv"
    with regional_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(regional_trace[0].keys()))
        writer.writeheader()
        writer.writerows(regional_trace)

    assert pred_path.exists()
    assert sum_path.exists()
    assert solver_path.exists()
    assert regional_path.exists()

    # Verify summary JSON is valid
    loaded_sum = json.loads(sum_path.read_text(encoding="utf-8"))
    assert "MAE" in loaded_sum


def test_paired_prediction_comparison(tmp_path: Path):
    """Test paired comparison with bootstrap CI in aggregate.py."""
    p_a = tmp_path / "preds_rmr.csv"
    p_b = tmp_path / "preds_direct.csv"

    # Introduce some realistic variance so std > 0
    rows_a = [
        {"id": f"img_{i:03d}", "gt": 10.0 + i, "pred": 10.0 + i + 1.0 + (i % 3) * 0.1}
        for i in range(20)
    ]
    rows_b = [
        {"id": f"img_{i:03d}", "gt": 10.0 + i, "pred": 10.0 + i + 3.0 + (i % 2) * 0.2}
        for i in range(20)
    ]

    for path, r_list in [(p_a, rows_a), (p_b, rows_b)]:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "gt", "pred"])
            writer.writeheader()
            writer.writerows(r_list)

    res = compare_predictions(p_a, p_b, name_a="RMR", name_b="Direct")

    assert res["n_samples"] == 20
    assert res["RMR"]["mae"] < res["Direct"]["mae"]
    assert res["delta_mae"] < 0.0
    assert res["RMR"]["wins"] == 20
    assert res["Direct"]["wins"] == 0

    p_diff = res["paired_difference"]
    assert p_diff["mean"] < 0.0
    ci_lo, ci_hi = p_diff["bootstrap_95_ci"]
    assert ci_lo <= p_diff["mean"] <= ci_hi
    assert p_diff["p_value_paired_ttest"] is not None


def test_aggregate_multiple_summaries(tmp_path: Path):
    """Test summary aggregation across multiple seeds."""
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"

    s1.write_text(json.dumps({"MAE": 10.0, "RMSE": 15.0, "variant": "rmr"}), encoding="utf-8")
    s2.write_text(json.dumps({"MAE": 12.0, "RMSE": 17.0, "variant": "rmr"}), encoding="utf-8")

    agg = aggregate_summaries([s1, s2])
    assert agg["MAE"]["mean"] == pytest.approx(11.0)
    assert agg["MAE"]["n"] == 2
    assert agg["RMSE"]["mean"] == pytest.approx(16.0)
