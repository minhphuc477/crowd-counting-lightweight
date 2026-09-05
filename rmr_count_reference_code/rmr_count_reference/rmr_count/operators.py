from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegionSet:
    """Rectangular regions on a feature/count grid.

    boxes: [M, 4] int64 with half-open coordinates (y1, x1, y2, x2).
    scale_id: [M] int64 index of the image-pixel region scale; -1 for full image.
    area: [M] float count-grid area.
    """

    boxes: torch.Tensor
    scale_id: torch.Tensor
    area: torch.Tensor

    def to(self, device: torch.device | str) -> "RegionSet":
        return RegionSet(
            boxes=self.boxes.to(device),
            scale_id=self.scale_id.to(device),
            area=self.area.to(device),
        )


def prefix2d(x: torch.Tensor) -> torch.Tensor:
    """Inclusive 2-D prefix sum with a zero top row/left column.

    Input:  [B, C, H, W]
    Output: [B, C, H+1, W+1]
    """
    if x.ndim != 4:
        raise ValueError(f"prefix2d expects [B,C,H,W], got {tuple(x.shape)}")
    p = x.cumsum(dim=-2).cumsum(dim=-1)
    return F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)


def _gather_prefix(prefix: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gather prefix values at M coordinates for every batch/channel."""
    b, c, hp, wp = prefix.shape
    idx = (y * wp + x).view(1, 1, -1).expand(b, c, -1)
    return torch.gather(prefix.flatten(-2), dim=-1, index=idx)


def rectangle_sum_from_prefix(prefix: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Rectangle sums using a padded prefix table.

    prefix: [B,C,H+1,W+1]
    boxes:  [M,4] in half-open grid coordinates
    returns [B,C,M]
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [M,4]")
    boxes = boxes.long()
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    br = _gather_prefix(prefix, y2, x2)
    tr = _gather_prefix(prefix, y1, x2)
    bl = _gather_prefix(prefix, y2, x1)
    tl = _gather_prefix(prefix, y1, x1)
    return br - tr - bl + tl


def regional_sum(x: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Linear regional-count operator A: [B,C,H,W] -> [B,C,M]."""
    return rectangle_sum_from_prefix(prefix2d(x), boxes)


def regional_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Exact adjoint A^T of rectangular summation.

    values: [B,C,M]
    boxes:  [M,4]
    returns [B,C,H,W]

    Uses a 2-D difference buffer followed by cumulative sums.
    """
    if values.ndim != 3:
        raise ValueError(f"values must be [B,C,M], got {tuple(values.shape)}")
    b, c, m = values.shape
    if boxes.shape != (m, 4):
        raise ValueError(f"boxes must be [{m},4], got {tuple(boxes.shape)}")

    boxes = boxes.long()
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    hp, wp = height + 1, width + 1

    diff = values.new_zeros((b, c, hp * wp))

    def scatter(y: torch.Tensor, x: torch.Tensor, src: torch.Tensor) -> None:
        idx = (y * wp + x).view(1, 1, -1).expand(b, c, -1)
        diff.scatter_add_(dim=-1, index=idx, src=src)

    scatter(y1, x1, values)
    scatter(y1, x2, -values)
    scatter(y2, x1, -values)
    scatter(y2, x2, values)

    diff = diff.view(b, c, hp, wp)
    field = diff.cumsum(dim=-2).cumsum(dim=-1)
    return field[..., :height, :width]


def _axis_starts(length: int, window: int, step: int) -> list[int]:
    if window >= length:
        return [0]
    starts = list(range(0, max(1, length - window + 1), max(1, step)))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return sorted(set(starts))


def build_multiscale_regions(
    height: int,
    width: int,
    output_stride: int,
    region_sizes_px: Sequence[int] = (16, 32, 64, 128),
    overlap: float = 0.5,
    include_full_image: bool = True,
    device: torch.device | str | None = None,
) -> RegionSet:
    """Build deterministic overlapping rectangular regions.

    Region sizes are specified in image pixels and quantized to the output grid.
    The last window on each axis is forced to touch the image/grid boundary.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    boxes: list[tuple[int, int, int, int]] = []
    scale_ids: list[int] = []

    for sid, size_px in enumerate(region_sizes_px):
        win = max(1, int(round(size_px / output_stride)))
        wy = min(win, height)
        wx = min(win, width)
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

    box_t = torch.tensor(boxes, dtype=torch.long, device=device)
    scale_t = torch.tensor(scale_ids, dtype=torch.long, device=device)
    area_t = ((box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])).float()
    return RegionSet(boxes=box_t, scale_id=scale_t, area=area_t)


def region_geometry(
    boxes: torch.Tensor,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Geometry features [M,6]: cy/H, cx/W, log_h, log_w, log_area, log_aspect.

    P0 fix — absolute region extent encoding:
        cy/H, cx/W : normalized center (scale-invariant position)
        log(h), log(w) : absolute grid extents in log scale
        log(|R|)       : absolute area = log(h*w)
        log(w/h)       : aspect ratio

    Previously used h/H, w/W (relative fractions) which caused a 32px region on a
    512px image and a 32px region on a 1024px image to have DIFFERENT geometry
    descriptors, even though they represent the same physical scale window.
    With absolute encoding, the 32px region always contributes the same geometric
    identity regardless of full image resolution.
    """
    boxes = boxes.float()
    y1, x1, y2, x2 = boxes.unbind(-1)
    h = (y2 - y1).clamp_min(1.0)
    w = (x2 - x1).clamp_min(1.0)
    cy = 0.5 * (y1 + y2) / max(float(height), 1.0)   # normalized center y
    cx = 0.5 * (x1 + x2) / max(float(width), 1.0)    # normalized center x
    log_h = torch.log(h + eps)                         # absolute height (log)
    log_w = torch.log(w + eps)                         # absolute width  (log)
    log_area = torch.log(h * w + eps)                  # absolute area   (log)
    log_aspect = torch.log(w / (h + eps) + eps)        # aspect ratio
    return torch.stack([cy, cx, log_h, log_w, log_area, log_aspect], dim=-1)



def region_average_features(features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Average pooled region features: [B,C,H,W] -> [B,M,C]."""
    sums = regional_sum(features, boxes)  # [B,C,M]
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).to(features.dtype)
    avg = sums / area.view(1, 1, -1).clamp_min(1.0)
    return avg.transpose(1, 2).contiguous()


def center_scatter(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Sparse learned-projection control: place each region residual at its center.

    values: [B,1,M]
    returns [B,1,H,W] with collision averaging.
    """
    if values.ndim != 3 or values.shape[1] != 1:
        raise ValueError("center_scatter expects values [B,1,M]")
    b, _, m = values.shape
    y = ((boxes[:, 0] + boxes[:, 2] - 1) // 2).long().clamp(0, height - 1)
    x = ((boxes[:, 1] + boxes[:, 3] - 1) // 2).long().clamp(0, width - 1)
    idx = (y * width + x).view(1, 1, m).expand(b, 1, -1)
    out = values.new_zeros((b, 1, height * width))
    cnt = values.new_zeros((b, 1, height * width))
    out.scatter_add_(-1, idx, values)
    cnt.scatter_add_(-1, idx, torch.ones_like(values))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, 1, height, width)
