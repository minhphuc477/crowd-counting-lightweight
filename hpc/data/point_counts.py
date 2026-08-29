"""Recursive exact integer target tree generation for NTPC down to stride-4.

Generates ground truth count pyramid:
  Y4 (H//4, W//4) -> Y8 -> Y16 -> Y32 -> Y64 -> N
with exact integer assertions and conservation checks:
  sum(Y4) == sum(Y8) == sum(Y16) == sum(Y32) == sum(Y64) == N.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch


def pad_hw_to_multiple(h: int, w: int, multiple: int = 64) -> Tuple[int, int]:
    hp = ((h + multiple - 1) // multiple) * multiple
    wp = ((w + multiple - 1) // multiple) * multiple
    return hp, wp


def block_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Non-overlapping exact sum pooling via reshape (preserves float32/int exactness).
    
    Args:
        x: [B, C, H, W] or [B, H, W] or [1, H, W]
        k: integer downscaling factor
    """
    has_batch_and_chan = (x.ndim == 4)
    if not has_batch_and_chan:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"Expected tensor of ndim 2, 3, or 4, got {x.ndim}")

    B, C, H, W = x.shape
    if H % k != 0 or W % k != 0:
        raise ValueError(f"Shape ({H}, {W}) not divisible by factor k={k}")

    out = x.reshape(B, C, H // k, k, W // k, k).sum(dim=(3, 5))
    if has_batch_and_chan:
        return out
    return out.squeeze(1)


def sum_2x2(x: torch.Tensor) -> torch.Tensor:
    """Sum adjacent 2x2 cells into a parent cell."""
    return block_sum(x, 2)


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


@torch.no_grad()
def points_to_y8_grid(
    points_xy: torch.Tensor,
    height: int,
    width: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize points directly onto stride-8 integer count grid (1, H//8, W//8)."""
    if device is None:
        device = points_xy.device if isinstance(points_xy, torch.Tensor) else torch.device("cpu")
    gh = height // 8
    gw = width // 8
    grid = torch.zeros((1, gh, gw), device=device, dtype=dtype)
    if points_xy is None or points_xy.numel() == 0:
        return grid
    pts = points_xy.to(device=device, dtype=torch.float32)
    x, y = pts[:, 0], pts[:, 1]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y = x[valid], y[valid]
    if x.numel() == 0:
        return grid
    bx = torch.floor(x / 8.0).long()
    by = torch.floor(y / 8.0).long()
    flat_idx = by * gw + bx
    grid.view(-1).scatter_add_(0, flat_idx, torch.ones(flat_idx.numel(), device=device, dtype=dtype))
    return grid


@torch.no_grad()
def points_to_y4(
    points_xy: torch.Tensor,
    H: int,
    W: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Rasterize points onto stride-4 grid (1, H/4, W/4).
    
    points_xy: [N, 2], zero-based continuous coordinates (x, y) with x in [0, W), y in [0, H).
    H, W must be divisible by 64 for tree hierarchy.
    """
    if H % 64 != 0 or W % 64 != 0:
        raise ValueError(f"H, W must be divisible by 64, got ({H}, {W})")

    if device is None:
        device = points_xy.device if isinstance(points_xy, torch.Tensor) else torch.device("cpu")

    gh = H // 4
    gw = W // 4

    y4 = torch.zeros((1, gh, gw), dtype=torch.float32, device=device)

    if points_xy is None or points_xy.numel() == 0:
        return y4

    pts = points_xy.to(device=device, dtype=torch.float32)
    x = pts[:, 0]
    y = pts[:, 1]

    valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0:
        return y4

    cell_x = torch.floor(x / 4.0).long()
    cell_y = torch.floor(y / 4.0).long()

    # Tensor flat indexing: row (y) * width + col (x)
    flat_idx = cell_y * gw + cell_x

    ones = torch.ones(flat_idx.numel(), dtype=torch.float32, device=device)
    y4.view(-1).scatter_add_(0, flat_idx, ones)

    return y4


@torch.no_grad()
def build_count_tree(
    points_xy: torch.Tensor,
    H: int,
    W: int,
    device: torch.device | None = None,
) -> Dict[str | int, torch.Tensor]:
    """Build single-image recursive count tree down to stride-4."""
    y4 = points_to_y4(points_xy, H, W, device=device)
    y8 = sum_2x2(y4)
    y16 = sum_2x2(y8)
    y32 = sum_2x2(y16)
    y64 = sum_2x2(y32)

    N = y4.sum()

    # Exact hierarchy assertions
    assert torch.equal(y8.sum(), N), "y8 count mismatch"
    assert torch.equal(y16.sum(), N), "y16 count mismatch"
    assert torch.equal(y32.sum(), N), "y32 count mismatch"
    assert torch.equal(y64.sum(), N), "y64 count mismatch"

    return {
        4: y4,
        8: y8,
        16: y16,
        32: y32,
        64: y64,
        "y4": y4,
        "y8": y8,
        "y16": y16,
        "y32": y32,
        "y64": y64,
        "N": N,
    }


def assert_integer_tensor(x: torch.Tensor, atol: float = 1e-5) -> None:
    """Verify that counts are non-negative integers."""
    if not torch.allclose(x, x.round(), atol=atol, rtol=0):
        raise RuntimeError("Count target contains fractional values; DTM requires exact integers.")


assert_integer_counts = assert_integer_tensor


def assert_parent_child_conservation(
    parent: torch.Tensor,
    children_4: torch.Tensor,
    tol: float = 1e-5,
) -> None:
    p = parent.float().reshape(parent.shape[0], -1)
    c = children_4.float().sum(dim=-1).reshape(parent.shape[0], -1)
    if not torch.allclose(p, c, atol=tol, rtol=0):
        raise RuntimeError("Broken target hierarchy: children sum does not equal parent count.")


def validate_targets(t: Dict[str | int, torch.Tensor]) -> None:
    """Validate full count tree targets for integer validity and conservation."""
    for name in (4, 8, 16, 32, 64, "y4", "y8", "y16", "y32", "y64"):
        if name in t:
            assert_integer_tensor(t[name])

    N = t["N"].reshape(-1)
    for name in (4, 8, 16, 32, 64):
        if name in t:
            s = t[name].flatten(1).sum(1)
            if not torch.allclose(s, N, atol=1e-4, rtol=0):
                raise RuntimeError(f"Target level {name} violates count conservation (sum {s} != {N})")


@torch.no_grad()
def build_exact_count_pyramid(
    points_batch: Sequence[torch.Tensor],
    height: int,
    width: int,
    block_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    pad_multiple: int = 64,
    device: torch.device | None = None,
) -> Dict[int | str, torch.Tensor]:
    """Batch target builder for training DataLoader collate."""
    if device is None:
        device = torch.device("cpu")

    hp = ((height + pad_multiple - 1) // pad_multiple) * pad_multiple
    wp = ((width + pad_multiple - 1) // pad_multiple) * pad_multiple

    y4_list = []
    for pts in points_batch:
        y4 = points_to_y4(pts, H=hp, W=wp, device=device)
        y4_list.append(y4)

    y4_batch = torch.stack(y4_list, dim=0).squeeze(1)  # (B, H/4, W/4)

    y8_batch = block_sum(y4_batch, 2)
    y16_batch = block_sum(y8_batch, 2)
    y32_batch = block_sum(y16_batch, 2)
    y64_batch = block_sum(y32_batch, 2)

    n_batch = y4_batch.flatten(1).sum(dim=1)

    assert torch.allclose(y8_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4)
    assert torch.allclose(y16_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4)
    assert torch.allclose(y32_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4)
    assert torch.allclose(y64_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4)

    return {
        4: y4_batch,
        8: y8_batch,
        16: y16_batch,
        32: y32_batch,
        64: y64_batch,
        "y4": y4_batch,
        "y8": y8_batch,
        "y16": y16_batch,
        "y32": y32_batch,
        "y64": y64_batch,
        "N": n_batch,
    }
