import pytest
import torch
import numpy as np
from hpc.targets.block_counts import build_integer_block_counts, build_hierarchical_block_counts


def test_t2_t3_exact_block_count_conservation():
    """T2 & T3: Test global count and exact integer block count conservation."""
    crop_h, crop_w = 448, 448
    n_points = 250
    np.random.seed(42)
    pts = np.random.uniform(0, 448, size=(n_points, 2)).astype(np.float32)
    
    for b in [16, 32, 64]:
        y_b = build_integer_block_counts(pts, crop_h, crop_w, block_size=b)
        assert y_b.shape == (crop_h // b, crop_w // b)
        sum_y = float(y_b.sum().item())
        assert abs(sum_y - n_points) < 1e-5, f"Scale B={b}: Block count sum {sum_y} != GT count {n_points}"


def test_t5_border_and_boundary_points():
    """T5: Points exactly on boundaries or borders."""
    crop_h, crop_w = 448, 448
    pts = np.array([
        [0.0, 0.0],
        [15.999, 15.999],
        [16.0, 16.0],
        [447.99, 447.99],
        [32.0, 64.0],
    ], dtype=np.float32)
    
    y16 = build_integer_block_counts(pts, crop_h, crop_w, block_size=16)
    assert y16.sum().item() == 5.0
    assert y16[0, 0] == 2.0  # (0,0) and (15.999, 15.999)
    assert y16[1, 1] == 1.0  # (16.0, 16.0)
    assert y16[27, 27] == 1.0  # (447.99, 447.99)
    assert y16[4, 2] == 1.0  # (32.0, 64.0) -> bx=2, by=4
