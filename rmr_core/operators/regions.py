from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class RegionSet:
    """Rectangular regions on a feature/count grid.

    boxes: [M, 4] int64 with half-open coordinates (y1, x1, y2, x2).
    scale_id: [M] int64 index of the image-pixel region scale; -1 for full image.
    area: [M] float count-grid area.
    boxes_list: cached Python list of (y1, x1, y2, x2) tuples to eliminate GPU->CPU sync.
    """

    boxes: torch.Tensor
    scale_id: torch.Tensor
    area: torch.Tensor
    boxes_list: list[tuple[int, int, int, int]] | None = None

    @property
    def areas(self) -> torch.Tensor:
        """Alias for `area` for spec-consistency (RegionSet.areas == RegionSet.area)."""
        return self.area

    def to(self, device: torch.device | str) -> "RegionSet":
        return RegionSet(
            boxes=self.boxes.to(device),
            scale_id=self.scale_id.to(device),
            area=self.area.to(device),
            boxes_list=self.boxes_list,
        )


def _axis_starts(length: int, window: int, step: int) -> list[int]:
    if window >= length:
        return [0]
    starts = list(range(0, max(1, length - window + 1), max(1, step)))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def _canonicalize_region_size(s: int | Sequence[int]) -> tuple[int, int]:
    """Convert any region size specification to an explicit (height_px, width_px) tuple."""
    if isinstance(s, (tuple, list)):
        if len(s) != 2:
            raise ValueError(f"Region size specification must be an integer or (height, width) pair, got {s}")
        hy, wx = int(s[0]), int(s[1])
        if hy <= 0 or wx <= 0:
            raise ValueError(f"Region dimensions must be strictly positive, got ({hy}, {wx})")
        return (hy, wx)
    val = int(s)
    if val <= 0:
        raise ValueError(f"Region dimension must be strictly positive, got {val}")
    return (val, val)


@functools.lru_cache(maxsize=32)
def _build_multiscale_regions_cached(
    height: int,
    width: int,
    output_stride: int,
    region_sizes_px: tuple[tuple[int, int], ...],
    overlap: float,
    include_full_image: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    boxes: list[tuple[int, int, int, int]] = []
    scale_ids: list[int] = []

    for sid, (hy_px, wx_px) in enumerate(region_sizes_px):
        win_y = max(1, int(round(hy_px / output_stride)))
        win_x = max(1, int(round(wx_px / output_stride)))
        wy = min(win_y, height)
        wx = min(win_x, width)
        sy = max(1, int(round(wy * (1.0 - overlap))))
        sx = max(1, int(round(wx * (1.0 - overlap))))
        ys = _axis_starts(height, wy, sy)
        xs = _axis_starts(width, wx, sx)
        for y1 in ys:
            for x1 in xs:
                boxes.append((y1, x1, y1 + wy, x1 + wx))
                scale_ids.append(sid)

    if include_full_image:
        full = (0, 0, height, width)
        if full not in boxes:
            boxes.append(full)
            scale_ids.append(-1)

    box_t = torch.tensor(boxes, dtype=torch.long).reshape(-1, 4)
    scale_t = torch.tensor(scale_ids, dtype=torch.long).reshape(-1)
    area_t = ((box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])).float()
    return box_t.clone(), scale_t.clone(), area_t.clone(), list(boxes)


def build_multiscale_regions(
    height: int,
    width: int,
    output_stride: int,
    region_sizes_px: Sequence[int | tuple[int, int] | list[int]] = (16, 32, 64, 128),
    overlap: float = 0.5,
    include_full_image: bool = True,
    device: torch.device | str | None = None,
) -> RegionSet:
    """Build deterministic overlapping rectangular regions with LRU caching.

    Region sizes can be specified as scalar image pixels (square windows)
    or (height_px, width_px) tuples for anisotropic perspective windows.
    Windows are quantized to the output grid.
    The last window on each axis is forced to touch the image/grid boundary.
    Cached across calls with maxsize=32.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Image grid dimensions must be strictly positive, got height={height}, width={width}")
    if output_stride <= 0:
        raise ValueError(f"output_stride must be strictly positive, got {output_stride}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    canonical_sizes = tuple(_canonicalize_region_size(s) for s in region_sizes_px)
    box_t, scale_t, area_t, boxes_list = _build_multiscale_regions_cached(
        height, width, output_stride, canonical_sizes, float(overlap), bool(include_full_image)
    )
    if device is not None:
        box_t = box_t.to(device)
        scale_t = scale_t.to(device)
        area_t = area_t.to(device)
    return RegionSet(boxes=box_t, scale_id=scale_t, area=area_t, boxes_list=list(boxes_list))


def partition_regions_by_scale(
    regions: RegionSet,
    k_scales: int,
    device: torch.device | str | None = None,
) -> list[tuple[int, torch.Tensor | None, torch.Tensor | None]]:
    """Partition RegionSet boxes by scale index k in [0, k_scales - 1].

    Assigns full-image regions (scale_id == -1) to the coarsest scale (k_scales - 1).
    Returns list of (k, mask_k, boxes_k) tuples to eliminate dynamic boolean masking and
    allocation churn during iterative solver loops.
    """
    dev = device if device is not None else regions.boxes.device
    scale_ids = regions.scale_id.to(device=dev)
    boxes = regions.boxes.to(device=dev)
    # Full-image regions (scale_id == -1) map to the coarsest scale
    scale_ids = torch.where(
        scale_ids < 0,
        torch.tensor(k_scales - 1, device=scale_ids.device, dtype=scale_ids.dtype),
        scale_ids.clamp(0, k_scales - 1),
    )
    partitions: list[tuple[int, torch.Tensor | None, torch.Tensor | None]] = []
    for k in range(k_scales):
        mask_k = (scale_ids == k)
        if mask_k.any():
            partitions.append((k, mask_k, boxes[mask_k]))
        else:
            partitions.append((k, None, None))
    return partitions


def region_geometry(
    boxes: torch.Tensor,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Position-free geometry features [M, 4]: log_h, log_w, log_area, log_aspect.

    Retained features (all position-free, scale-invariant):
        log(h)      absolute grid height (same for same physical scale, any crop)
        log(w)      absolute grid width
        log(h*w)    absolute area = log_h + log_w
        log(w/h)    aspect ratio
    """
    boxes = boxes.float()
    y1, x1, y2, x2 = boxes.unbind(-1)
    h = (y2 - y1).abs().clamp_min(1.0)
    w = (x2 - x1).abs().clamp_min(1.0)
    log_h = torch.log(h + eps)
    log_w = torch.log(w + eps)
    log_area = torch.log(h * w + eps)
    log_aspect = torch.log(w / (h + eps) + eps)
    return torch.stack([log_h, log_w, log_area, log_aspect], dim=-1)
