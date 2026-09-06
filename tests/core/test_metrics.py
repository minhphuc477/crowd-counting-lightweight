import numpy as np
import pytest
import torch

from rmr_core.metrics import (
    bootstrap_ci,
    compute_nae,
    density_stratified_mae,
    game_physical_image,
    game_single,
    summarize_predictions,
)


def test_compute_nae():
    preds = np.array([10.0, 20.0, 30.0])
    targets = np.array([10.0, 20.0, 30.0])
    assert compute_nae(preds, targets) == 0.0

    preds = np.array([12.0, 18.0])
    targets = np.array([10.0, 20.0])
    # |12-10|/10 = 0.2, |18-20|/20 = 0.1 -> mean = 0.15
    assert abs(compute_nae(preds, targets) - 0.15) < 1e-6


def test_game_physical_image_level_zero_matches_ae():
    y = np.ones((16, 16), dtype=np.float64) * 0.5  # sum = 128.0
    pts = np.zeros((100, 2), dtype=np.float64)     # 100 points
    pts[:, 0] = 30.0  # within 64x64
    pts[:, 1] = 30.0

    games = game_physical_image(y, pts, image_h=64, image_w=64, stride=4, levels=(0, 1, 2, 3))
    assert abs(games[0] - abs(128.0 - 100.0)) < 1e-6
    assert games[1] >= games[0]  # Monotonic with subdivision
    assert games[2] >= games[1]
    assert games[3] >= games[2]


def test_summarize_and_bootstrap_ci():
    rows = [
        {"gt": 50.0, "pred": 55.0, "GAME0": 5.0, "GAME1": 8.0},
        {"gt": 200.0, "pred": 190.0, "GAME0": 10.0, "GAME1": 15.0},
        {"gt": 600.0, "pred": 620.0, "GAME0": 20.0, "GAME1": 25.0},
    ]
    summary = summarize_predictions(rows)
    assert abs(summary["MAE"] - (5 + 10 + 20) / 3) < 1e-6

    aes = [abs(r["pred"] - r["gt"]) for r in rows]
    lo, hi = bootstrap_ci(aes, n_boot=500)
    assert lo <= summary["MAE"] <= hi

    strat = density_stratified_mae(rows)
    assert "sparse_le100" in strat
    assert "mid_101_500" in strat
    assert "dense_gt500" in strat
