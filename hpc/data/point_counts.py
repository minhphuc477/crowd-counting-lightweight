"""Recursive exact integer target generation for Neural Tree-Pólya Crowd Counting (NTPC).

Follows §11 of the NTPC Audit Errata:
  1. Rasterize continuous point coordinates directly into finest block grid y8 (H//8, W//8).
  2. Recursively sum 2x2 blocks:
       y16 = block_sum(y8, 2)
       y32 = block_sum(y16, 2)
       y64 = block_sum(y32, 2)
  3. Image total count: N = y8.sum()
  4. Exact conservation asserted across all levels: sum(y16) == sum(y32) == sum(y64) == N.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def pad_hw_to_multiple(h: int, w: int, multiple: int) -> Tuple[int, int]:
    hp = ((h + multiple - 1) // multiple) * multiple
    wp = ((w + multiple - 1) // multiple) * multiple
    return hp, wp


@torch.no_grad()
def points_to_impulse_map(
    points_batch: Sequence[torch.Tensor],
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    batch_size = len(points_batch)
    impulse = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)
    for b, pts in enumerate(points_batch):
        if pts is None or pts.numel() == 0:
            continue
        pts = pts.to(device=device)
        x = torch.floor(pts[:, 0]).long()
        y = torch.floor(pts[:, 1]).long()
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y = x[valid], y[valid]
        if x.numel() == 0:
            continue
        flat_idx = y * width + x
        flat = impulse[b, 0].view(-1)
        flat.scatter_add_(0, flat_idx, torch.ones(flat_idx.numel(), device=device, dtype=dtype))
    return impulse


def block_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping exact block sum via reshape (avoids pooling roundoff).
    
    Args:
        x: (B, C, H, W) or (B, H, W) tensor where H, W are divisible by k.
        k: integer downscale factor.
        
    Returns:
        (B, C, H//k, W//k) or (B, H//k, W//k) pooled sum tensor.
    """
    has_channel = (x.ndim == 4)
    if not has_channel:
        x = x.unsqueeze(1)
        
    B, C, H, W = x.shape
    if H % k != 0 or W % k != 0:
        raise ValueError(f"Shape ({H}, {W}) not divisible by factor {k}")
        
    out = x.reshape(B, C, H // k, k, W // k, k).sum(dim=(3, 5))
    return out if has_channel else out.squeeze(1)


def points_to_y8_grid(
    points_xy: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize points directly onto stride-8 integer count grid (1, H//8, W//8).
    
    Coordinates use continuous image pixel space: [0, width) x [0, height).
    (y indexes rows, x indexes columns).
    """
    if device is None:
        device = points_xy.device if isinstance(points_xy, torch.Tensor) else torch.device("cpu")
        
    if height % 64 != 0 or width % 64 != 0:
        raise ValueError(f"Canvas size ({height}, {width}) must be divisible by 64")
        
    gh = height // 8
    gw = width // 8
    grid = torch.zeros((1, gh, gw), device=device, dtype=dtype)
    
    if points_xy is None or points_xy.numel() == 0:
        return grid
        
    pts = points_xy.to(device=device, dtype=torch.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    
    if x.numel() == 0:
        return grid
        
    bx = torch.floor(x / 8.0).long()
    by = torch.floor(y / 8.0).long()
    
    # Tensor convention: flat_idx = row * num_cols + col
    flat_idx = by * gw + bx
    ones = torch.ones(flat_idx.numel(), device=device, dtype=dtype)
    grid.view(-1).scatter_add_(0, flat_idx, ones)
    
    return grid


def assert_integer_counts(y: torch.Tensor, tol: float = 1e-5) -> None:
    """Verify that counts are strictly non-negative integers."""
    if not torch.allclose(y, y.round(), atol=tol, rtol=0):
        raise RuntimeError("Target contains fractional counts; DTM requires exact integers.")


def assert_parent_child_conservation(
    parent: torch.Tensor,
    children_4: torch.Tensor,
    tol: float = 1e-5,
) -> None:
    """Verify that sum of 4 child counts strictly equals parent count."""
    p = parent.float().reshape(parent.shape[0], -1)
    c = children_4.float().sum(dim=-1).reshape(parent.shape[0], -1)
    if not torch.allclose(p, c, atol=tol, rtol=0):
        raise RuntimeError("Broken target hierarchy: children sum does not equal parent count.")


@torch.no_grad()
def build_exact_count_pyramid(
    points_batch: Sequence[torch.Tensor],
    height: int,
    width: int,
    block_sizes: Iterable[int] = (8, 16, 32, 64),
    pad_multiple: int = 64,
    device: torch.device | None = None,
) -> Dict[int | str, torch.Tensor]:
    """Build exact recursive integer count pyramid for a batch of images.
    
    Returns:
        dict with keys {8: (B, H/8, W/8), 16: (B, H/16, W/16), 32: (B, H/32, W/32), 64: (B, H/64, W/64), 'N': (B,)}
    """
    if device is None:
        device = torch.device("cpu")
        
    # Ensure height and width are multiples of 64
    hp = ((height + pad_multiple - 1) // pad_multiple) * pad_multiple
    wp = ((width + pad_multiple - 1) // pad_multiple) * pad_multiple
    
    y8_list = []
    for pts in points_batch:
        y8 = points_to_y8_grid(pts, height=hp, width=wp, device=device)
        y8_list.append(y8)
        
    y8_batch = torch.stack(y8_list, dim=0).squeeze(1)  # (B, H/8, W/8)
    
    # Recursive 2x2 sum-pooling
    y16_batch = block_sum(y8_batch, 2)   # (B, H/16, W/16)
    y32_batch = block_sum(y16_batch, 2)  # (B, H/32, W/32)
    y64_batch = block_sum(y32_batch, 2)  # (B, H/64, W/64)
    
    n_batch = y8_batch.flatten(1).sum(dim=1)  # (B,)
    
    # Assert exact hierarchy consistency
    assert torch.allclose(y16_batch.flatten(1).sum(dim=1), n_batch, atol=1e-5)
    assert torch.allclose(y32_batch.flatten(1).sum(dim=1), n_batch, atol=1e-5)
    assert torch.allclose(y64_batch.flatten(1).sum(dim=1), n_batch, atol=1e-5)
    assert_integer_counts(y8_batch)
    
    out: Dict[int | str, torch.Tensor] = {
        8: y8_batch,
        16: y16_batch,
        32: y32_batch,
        64: y64_batch,
        "N": n_batch,
    }
    return out
