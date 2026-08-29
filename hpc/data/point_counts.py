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
def points_to_y4(
    points_xy: torch.Tensor,
    H: int,
    W: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Rasterize points onto stride-4 grid (1, H/4, W/4).

    points_xy: [N, 2], zero-based continuous coordinates (x, y) with
               support x in [-0.5, W - 0.5), y in [-0.5, H - 0.5).
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

    valid = (x >= -0.5) & (x <= float(W) - 0.5) & (y >= -0.5) & (y <= float(H) - 0.5)
    x = x[valid]
    y = y[valid]

    if x.numel() == 0:
        return y4

    cell_x = torch.floor((x + 0.5) / 4.0).long().clamp(0, gw - 1)
    cell_y = torch.floor((y + 0.5) / 4.0).long().clamp(0, gh - 1)

    # Tensor flat indexing: row (y) * width + col (x)
    flat_idx = cell_y * gw + cell_x

    ones = torch.ones(flat_idx.numel(), dtype=torch.float32, device=device)
    y4.view(-1).scatter_add_(0, flat_idx, ones)

    return y4


@torch.no_grad()
def build_exact_count_pyramid(
    points_batch: Sequence[torch.Tensor],
    height: int,
    width: int,
    block_sizes: Sequence[int] = (4, 8, 16, 32, 64),
    pad_multiple: int = 64,
    device: torch.device | None = None,
) -> Dict[int | str, torch.Tensor]:
    """Build exact block-count levels {4, 8, 16, 32, 64} plus total count ``N``."""
    if device is None:
        device = torch.device("cpu")

    hp = ((height + pad_multiple - 1) // pad_multiple) * pad_multiple
    wp = ((width + pad_multiple - 1) // pad_multiple) * pad_multiple

    requested = tuple(dict.fromkeys(int(b) for b in block_sizes))
    valid_levels = {4, 8, 16, 32, 64}
    invalid = [b for b in requested if b not in valid_levels]
    if invalid:
        raise ValueError(f"Unsupported block sizes: {invalid}; expected subset of {sorted(valid_levels)}")
    if not requested:
        raise ValueError("block_sizes cannot be empty")

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

    if not torch.allclose(y8_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4):
        raise RuntimeError("Count conservation failed between Y4 and Y8")
    if not torch.allclose(y16_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4):
        raise RuntimeError("Count conservation failed between Y4 and Y16")
    if not torch.allclose(y32_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4):
        raise RuntimeError("Count conservation failed between Y4 and Y32")
    if not torch.allclose(y64_batch.flatten(1).sum(dim=1), n_batch, atol=1e-4):
        raise RuntimeError("Count conservation failed between Y4 and Y64")

    all_levels = {4: y4_batch, 8: y8_batch, 16: y16_batch, 32: y32_batch, 64: y64_batch}
    result: Dict[int | str, torch.Tensor] = {"N": n_batch}
    for block_size in requested:
        result[block_size] = all_levels[block_size]
        result[f"y{block_size}"] = all_levels[block_size]
    return result
