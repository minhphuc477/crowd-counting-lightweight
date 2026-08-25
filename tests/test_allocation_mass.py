import pytest
import torch
import numpy as np
from hpc.targets.allocation_target import build_block_constrained_allocation_target
from hpc.targets.block_counts import build_integer_block_counts


def test_t4_allocation_mass_conservation():
    """T4: Test that allocation target preserves mass locally per 16x16 block and globally."""
    crop_h, crop_w = 448, 448
    n_points = 300
    np.random.seed(42)
    pts = np.random.uniform(0, 448, size=(n_points, 2)).astype(np.float32)
    
    z_map = build_block_constrained_allocation_target(pts, crop_h, crop_w, block_size=16, output_stride=4)
    y16 = build_integer_block_counts(pts, crop_h, crop_w, block_size=16)
    
    # 1. Global conservation
    total_z = float(z_map.sum().item())
    assert abs(total_z - n_points) < 1e-4, f"Global allocation sum {total_z} != N {n_points}"
    
    # 2. Local per-block conservation: sum_{k=1..16} Z_{bk} == Y_b^(16)
    # Reshape z_map into 16x16 block cells (4x4 cells)
    out_h, out_w = crop_h // 4, crop_w // 4
    h_16, w_16 = crop_h // 16, crop_w // 16
    z_blocks = z_map.view(h_16, 4, w_16, 4).permute(0, 2, 1, 3).contiguous().view(h_16, w_16, 16)
    block_sums = z_blocks.sum(dim=-1)
    
    diff = torch.abs(block_sums - y16)
    max_diff = float(diff.max().item())
    assert max_diff < 1e-4, f"Max local block mass discrepancy: {max_diff}"


def test_t5_allocation_border_leakage_prevention():
    """T5: Points near the edge of 16x16 blocks must NEVER leak to neighboring blocks."""
    crop_h, crop_w = 448, 448
    # Point at (15.9, 15.9) - very close to block (0,0) right/bottom boundary
    pts = np.array([[15.9, 15.9]], dtype=np.float32)
    
    z_map = build_block_constrained_allocation_target(pts, crop_h, crop_w, block_size=16, output_stride=4)
    y16 = build_integer_block_counts(pts, crop_h, crop_w, block_size=16)
    
    # Block (0, 0) should contain 1.0 mass
    h_16, w_16 = crop_h // 16, crop_w // 16
    z_blocks = z_map.view(h_16, 4, w_16, 4).permute(0, 2, 1, 3).contiguous().view(h_16, w_16, 16)
    block_sums = z_blocks.sum(dim=-1)
    
    assert abs(block_sums[0, 0].item() - 1.0) < 1e-5
    # Neighbors (0,1), (1,0), (1,1) MUST have strictly 0 mass
    assert block_sums[0, 1].item() == 0.0
    assert block_sums[1, 0].item() == 0.0
    assert block_sums[1, 1].item() == 0.0
