import json
from pathlib import Path
import numpy as np
import pytest
import torch

from rmr_count.localization.metrics import LocalizationMeter, match_points
from rmr_count.localization.occupancy import compute_manifest_occupancy
from rmr_count.localization.otm import otm_density_to_points


def test_match_points_empty():
    # Both empty
    tp, fp, fn = match_points([], [], sigma=4.0)
    assert (tp, fp, fn) == (0, 0, 0)

    # Empty preds, non-empty gts
    tp, fp, fn = match_points([], [[10.0, 10.0]], sigma=4.0)
    assert (tp, fp, fn) == (0, 0, 1)

    # Non-empty preds, empty gts
    tp, fp, fn = match_points([[10.0, 10.0]], [], sigma=4.0)
    assert (tp, fp, fn) == (0, 1, 0)


def test_match_points_exact():
    pts = np.array([[10.0, 10.0], [20.0, 30.0], [50.0, 60.0]])
    tp, fp, fn = match_points(pts, pts, sigma=4.0)
    assert (tp, fp, fn) == (3, 0, 0)


def test_match_points_threshold_cutoff():
    preds = np.array([[10.0, 10.0], [20.0, 20.0]])
    # First point is distance 3.0 (< 4.0), second point is distance 5.0 (> 4.0)
    gts = np.array([[10.0, 13.0], [20.0, 25.0]])

    tp, fp, fn = match_points(preds, gts, sigma=4.0)
    assert tp == 1
    assert fp == 1
    assert fn == 1

    # If sigma is increased to 6.0, both should match
    tp_wide, fp_wide, fn_wide = match_points(preds, gts, sigma=6.0)
    assert tp_wide == 2
    assert fp_wide == 0
    assert fn_wide == 0


def test_localization_meter_aggregation():
    meter = LocalizationMeter(sigmas=[4.0, 8.0])

    # Image 1: 2 GTs, 2 Preds, 1 match under sigma=4, 2 under sigma=8
    p1 = np.array([[10.0, 10.0], [20.0, 20.0]])
    g1 = np.array([[10.0, 12.0], [20.0, 26.0]])  # dists: 2.0 and 6.0
    meter.update(p1, g1)

    # Image 2: 1 GT, 1 Pred, exact match
    p2 = np.array([[50.0, 50.0]])
    g2 = np.array([[50.0, 50.0]])
    meter.update(p2, g2)

    summary = meter.compute_summary()
    assert summary["total_predictions"] == 3
    assert summary["total_ground_truth"] == 3

    # Sigma 4:
    # Img 1: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F1=0.5
    # Img 2: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F1=1.0
    # Total TP=2, FP=1, FN=1 -> Micro P=2/3, R=2/3, F1=2/3
    s4 = summary["thresholds"]["sigma_4"]
    assert s4["tp"] == 2
    assert s4["fp"] == 1
    assert s4["fn"] == 1
    assert pytest.approx(s4["micro_f1"], rel=1e-4) == 2.0 / 3.0
    assert pytest.approx(s4["macro_f1"], rel=1e-4) == 0.75

    # Sigma 8: all match
    s8 = summary["thresholds"]["sigma_8"]
    assert s8["tp"] == 3
    assert s8["fp"] == 0
    assert s8["fn"] == 0
    assert pytest.approx(s8["micro_f1"], rel=1e-4) == 1.0


def test_otm_density_to_points_empty():
    empty_density = np.zeros((10, 10), dtype=np.float32)
    pts = otm_density_to_points(empty_density, stride=4)
    assert pts.shape == (0, 2)


def test_otm_density_to_points_single_impulse():
    density = np.zeros((16, 16), dtype=np.float32)
    # Single point with mass 1.0 at cell (3, 5)
    density[3, 5] = 1.0
    pts = otm_density_to_points(density, stride=4, outer_iters=5, device="cpu")

    assert pts.shape == (1, 2)
    # Expected center: x = 4*5 + 2 = 22.0, y = 4*3 + 2 = 14.0
    expected_x = 4 * 5 + 2.0
    expected_y = 4 * 3 + 2.0
    assert pytest.approx(pts[0, 0], abs=0.5) == expected_x
    assert pytest.approx(pts[0, 1], abs=0.5) == expected_y


def test_otm_cardinality_conservation():
    density = np.zeros((20, 20), dtype=np.float32)
    density[2, 2] = 1.8
    density[5, 5] = 3.3
    density[10, 10] = 0.4
    # Total mass: 1.8 + 3.3 + 0.4 = 5.5 -> round = 6
    pts = otm_density_to_points(density, stride=4, device="cpu")
    assert pts.shape == (6, 2)


def test_occupancy_analysis(tmp_path: Path):
    from PIL import Image

    img_file = tmp_path / "dummy.jpg"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_file)

    manifest_file = tmp_path / "test_manifest.jsonl"
    # Create mock manifest with stride 4:
    # Cell 0,0: 3 points
    # Cell 1,1: 1 point
    data = [
        {
            "image": str(img_file),
            "points": [
                [0.5, 0.5],
                [1.0, 1.0],
                [1.5, 1.5],
                [5.0, 5.0],
            ],
            "id": "1",
        }
    ]
    manifest_file.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")

    res = compute_manifest_occupancy(manifest_file, stride=4)
    assert res["total_heads"] == 4
    assert res["max_occupancy"] == 3
    # 3 heads are in the cell with occupancy 3, 1 head in cell with occupancy 1
    assert res["heads_in_multi_cells"] == 3
    assert pytest.approx(res["multi_head_ratio"]) == 3.0 / 4.0


def test_empty_gt_manifest(tmp_path: Path):
    from PIL import Image
    from rmr_count.localization.oracle_cell import evaluate_oracle_cell_centers
    from rmr_count.localization.otm import evaluate_oracle_otm

    # 1. Non-existent image must fail hard with FileNotFoundError across all functions
    missing_manifest = tmp_path / "missing_img_manifest.jsonl"
    missing_manifest.write_text(
        json.dumps({"image": "nonexistent_empty.jpg", "points": [], "id": "0"}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Image not found"):
        evaluate_oracle_cell_centers(missing_manifest, stride=4)
    with pytest.raises(FileNotFoundError, match="Image not found"):
        compute_manifest_occupancy(missing_manifest, stride=4)
    with pytest.raises(FileNotFoundError, match="Image not found"):
        evaluate_oracle_otm(missing_manifest, stride=4)

    # 2. When valid empty image exists
    img_file = tmp_path / "empty_real.jpg"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_file)

    manifest_file = tmp_path / "empty_manifest.jsonl"
    data = [
        {"image": str(img_file), "points": [], "id": "0"},
    ]
    manifest_file.write_text("\n".join(json.dumps(d) for d in data), encoding="utf-8")

    occ = compute_manifest_occupancy(manifest_file, stride=4)
    assert occ["total_heads"] == 0

    oracle = evaluate_oracle_cell_centers(manifest_file, stride=4)
    assert oracle["total_ground_truth"] == 0

    otm = evaluate_oracle_otm(manifest_file, stride=4)
    assert otm["total_ground_truth"] == 0


def test_otm_deterministic_seed():
    density = np.random.RandomState(123).uniform(0, 1, size=(16, 16)).astype(np.float32)
    pts1 = otm_density_to_points(density, stride=4, seed=42, device="cpu")
    pts2 = otm_density_to_points(density, stride=4, seed=42, device="cpu")
    assert np.allclose(pts1, pts2)


def test_evaluate_model_otm_v3_checkpoint(tmp_path: Path):
    from PIL import Image
    from rmr_count.localization.otm import evaluate_model_otm
    from rmr_v3.model import RMRv3, RMRv3Config

    # 1. Create a synthetic image and manifest
    img_file = tmp_path / "test_img.jpg"
    Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_file)
    manifest_file = tmp_path / "test_manifest.jsonl"
    manifest_file.write_text(
        json.dumps({"image": str(img_file), "points": [[20.0, 20.0], [40.0, 40.0]], "id": "sample_0"}) + "\n",
        encoding="utf-8",
    )

    # 2. Instantiate and save a miniature RMR-v3 model checkpoint
    cfg_model = RMRv3Config(feature_width=16, pretrained=False, iterations=1, region_sizes_px=(32, 64, 128))
    model = RMRv3(cfg_model)
    ckpt_path = tmp_path / "model_v3.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "model": {
                "output_stride": 4,
                "feature_width": 16,
                "iterations": 1,
                "region_sizes_px": [32, 64, 128],
                "pretrained": False,
            }
        },
        "git_commit": "testcommit",
        "git_dirty": False,
    }, ckpt_path)

    # 3. Evaluate OT-M with V3 checkpoint
    summary = evaluate_model_otm(
        checkpoint_path=ckpt_path,
        manifest_path=manifest_file,
        device="cpu",
        stride=4,
    )

    assert "thresholds" in summary
    assert "total_predictions" in summary
    assert "total_ground_truth" in summary
    assert summary["total_ground_truth"] == 2

