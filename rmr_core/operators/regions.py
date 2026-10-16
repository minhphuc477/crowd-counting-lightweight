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
    num_scales: number of multiscale scales K (default: 3) to eliminate GPU-CPU syncs.
    """

    boxes: torch.Tensor
    scale_id: torch.Tensor
    area: torch.Tensor
    boxes_list: list[tuple[int, int, int, int]] | None = None
    num_scales: int = 3
    scale_sizes_px: tuple[tuple[int, int], ...] | None = None

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
            num_scales=self.num_scales,
            scale_sizes_px=self.scale_sizes_px,
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
    return RegionSet(
        boxes=box_t,
        scale_id=scale_t,
        area=area_t,
        boxes_list=list(boxes_list),
        num_scales=len(canonical_sizes),
        scale_sizes_px=canonical_sizes,
    )


@functools.lru_cache(maxsize=32)
def _build_perspective_adaptive_regions_cached(
    height: int,
    width: int,
    output_stride: int,
    canonical_sizes: tuple[tuple[int, int], ...],
    overlap: float,
    camera_tilt: float,
    horizon_pos: float,
    min_scale_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int, int, int], ...]]:
    boxes: list[tuple[int, int, int, int]] = []
    scale_ids: list[int] = []

    for sid, (hy_px, wx_px) in enumerate(canonical_sizes):
        base_h = max(1, int(round(hy_px / output_stride)))
        base_w = max(1, int(round(wx_px / output_stride)))

        cur_y = 0
        while cur_y < height:
            y_norm = cur_y / float(max(height - 1, 1))
            if y_norm <= horizon_pos:
                scale_factor = float(min_scale_ratio)
            else:
                rel = (y_norm - horizon_pos) / max(1.0 - horizon_pos, 1e-4)
                scale_factor = float(min_scale_ratio) + (1.0 - float(min_scale_ratio)) * (rel ** (1.0 + float(camera_tilt)))

            win_h = max(2, min(height, int(round(base_h * scale_factor))))
            win_w = max(2, min(width, int(round(base_w * scale_factor))))
            y1 = cur_y
            y2 = min(height, y1 + win_h)

            step_x = max(1, int(round(win_w * (1.0 - overlap))))
            cur_x = 0
            while cur_x < width:
                x1 = cur_x
                x2 = min(width, x1 + win_w)
                boxes.append((y1, x1, y2, x2))
                scale_ids.append(sid)
                if x2 >= width:
                    break
                cur_x += step_x
                if cur_x + win_w > width and cur_x < width:
                    boxes.append((y1, max(0, width - win_w), y2, width))
                    scale_ids.append(sid)
                    break

            if y2 >= height:
                break
            step_y = max(1, int(round(win_h * (1.0 - overlap))))
            cur_y += step_y
            if cur_y + win_h > height and cur_y < height:
                y1_final = max(0, height - win_h)
                cur_x = 0
                while cur_x < width:
                    x1 = cur_x
                    x2 = min(width, x1 + win_w)
                    boxes.append((y1_final, x1, height, x2))
                    scale_ids.append(sid)
                    if x2 >= width:
                        break
                    cur_x += step_x
                break

    unique_boxes: list[tuple[int, int, int, int]] = []
    unique_scales: list[int] = []
    seen = set()
    for b_coord, sid in zip(boxes, scale_ids):
        if b_coord not in seen:
            seen.add(b_coord)
            unique_boxes.append(b_coord)
            unique_scales.append(sid)

    box_t = torch.tensor(unique_boxes, dtype=torch.long)
    scale_t = torch.tensor(unique_scales, dtype=torch.long)
    area_t = ((box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])).float()
    return box_t.clone(), scale_t.clone(), area_t.clone(), tuple(unique_boxes)


def build_perspective_adaptive_regions(
    height: int,
    width: int,
    output_stride: int,
    base_sizes_px: Sequence[int | tuple[int, int] | list[int]] = (32, 64, 128),
    overlap: float = 0.4,
    camera_tilt: float = 0.5,
    horizon_pos: float = 0.1,
    min_scale_ratio: float = 0.5,
    device: torch.device | str | None = None,
) -> RegionSet:
    """Construct truly perspective-adaptive multiscale bounding boxes with LRU caching.

    At horizon (y -> horizon_pos), window size shrinks by min_scale_ratio.
    At foreground (y -> 1.0), window size expands to full base size.
    Vertical stride dynamically contracts at the horizon to maintain exact constant overlap.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Image grid dimensions must be strictly positive, got height={height}, width={width}")
    if output_stride <= 0:
        raise ValueError(f"output_stride must be strictly positive, got {output_stride}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    if not (0.0 < min_scale_ratio <= 1.0):
        raise ValueError(f"min_scale_ratio must be in (0, 1], got {min_scale_ratio}")
    if not (0.0 <= horizon_pos < 1.0):
        raise ValueError(f"horizon_pos must be in [0, 1), got {horizon_pos}")

    canonical_sizes = tuple(_canonicalize_region_size(s) for s in base_sizes_px)
    box_t, scale_t, area_t, boxes_tuple = _build_perspective_adaptive_regions_cached(
        height, width, output_stride, canonical_sizes,
        float(overlap), float(camera_tilt), float(horizon_pos), float(min_scale_ratio),
    )
    if device is not None:
        box_t = box_t.to(device)
        scale_t = scale_t.to(device)
        area_t = area_t.to(device)

    return RegionSet(
        boxes=box_t,
        scale_id=scale_t,
        area=area_t,
        boxes_list=list(boxes_tuple),
        num_scales=len(canonical_sizes),
        scale_sizes_px=canonical_sizes,
    )


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
