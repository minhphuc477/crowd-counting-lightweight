from __future__ import annotations

"""Perspective-Adaptive Regional Kernels (PARK) & Perspective Tiling Operators.

Provides perspective-conditioned altitude tiling (PCAT) and continuous differentiable
window operators for Radon Measure Recovery (RMR).

Mathematical Principles:
1. Pinhole Projective Geometry:
   Object image scale on an oblique ground plane satisfies:
       s(y) propto 1 / Z(y) propto (y - y_horizon)
   Horizon (v -> 0): Small isotropic boxes [16x16, 24x24] px.
   Foreground (v -> 1): Large vertically elongated boxes [64x36, 128x64] px with aspect ratio ~ 2:1.
2. Perspective Coverage Invariance:
   Step sizes proportional to local window size maintain constant coverage:
       D_c(u) approx 4.0 = const (at 50% overlap).
3. Exact O(1) Bilinear Prefix Integration & O(M + HW) Continuous Adjoint Backprojection.
"""

import functools
import math
from typing import Sequence

import torch
import torch.nn.functional as F

from .prefix_sums import continuous_prefix_eval, prefix2d
from .regions import RegionSet, _axis_starts


@functools.lru_cache(maxsize=32)
def _build_perspective_regions_cached(
    height: int,
    width: int,
    output_stride: int,
    horizon_size_px: int = 16,
    foreground_size_px: int = 128,
    max_aspect_ratio: float = 2.0,
    overlap: float = 0.5,
    num_altitude_bands: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[tuple[int, int, int, int]]]:
    """Deterministic perspective-conditioned altitude tiling with LRU caching.

    Generates three altitude bands (Horizon, Midground, Foreground) where:
    - Window dimensions (hy, wx) adapt continuously to optical depth.
    - Aspect ratio rho = hy / wx transitions from 1.0 (circular at horizon) to 2.0 (upright ellipse in foreground).
    - Grid step sizes scale with local box dimensions, guaranteeing O(1) stationary density coverage.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Grid dimensions must be positive, got height={height}, width={width}")
    if output_stride <= 0:
        raise ValueError(f"output_stride must be positive, got {output_stride}")
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    boxes: list[tuple[int, int, int, int]] = []
    scale_ids: list[int] = []

    if num_altitude_bands == 3:
        band_defs = [
            (0, 0.00, 0.40, 0.20),  # Band 0: Horizon / Far field (small, isotropic)
            (1, 0.25, 0.75, 0.50),  # Band 1: Midground (medium, moderate elongation)
            (2, 0.60, 1.00, 1.00),  # Band 2: Foreground (large, anisotropic upright ellipse)
        ]
    else:
        k_bands = max(1, int(num_altitude_bands))
        step_v = 1.0 / float(k_bands)
        overlap_v = 0.25 * step_v
        band_defs = []
        for k in range(k_bands):
            v_start = max(0.0, k * step_v - (overlap_v if k > 0 else 0.0))
            v_end = min(1.0, (k + 1) * step_v + (overlap_v if k < k_bands - 1 else 0.0))
            s_factor = float(k + 1) / float(k_bands)
            band_defs.append((k, v_start, v_end, s_factor))

    band_sizes: list[tuple[int, int]] = []
    for band_id, v_start, v_end, s_factor in band_defs:
        v_center = 0.5 * (v_start + v_end)
        hy_px = int(round(horizon_size_px + (foreground_size_px - horizon_size_px) * s_factor))
        rho = 1.0 + (max_aspect_ratio - 1.0) * v_center
        wx_px = max(8, int(round(hy_px / rho)))
        band_sizes.append((hy_px, wx_px))

        win_y = max(1, min(height, int(round(hy_px / output_stride))))
        win_x = max(1, min(width, int(round(wx_px / output_stride))))

        sy = max(1, int(round(win_y * (1.0 - overlap))))
        sx = max(1, int(round(win_x * (1.0 - overlap))))

        y_min_grid = int(round(v_start * height))
        y_max_grid = int(round(v_end * height))
        band_height = max(win_y, y_max_grid - y_min_grid)

        # Generate vertical starts within band
        local_ys = _axis_starts(band_height, win_y, sy)
        ys = [min(height - win_y, y_min_grid + y) for y in local_ys if y_min_grid + y + win_y <= height]
        if not ys:
            ys = [min(height - win_y, y_min_grid)]
        xs = _axis_starts(width, win_x, sx)

        for y1 in ys:
            for x1 in xs:
                boxes.append((y1, x1, y1 + win_y, x1 + win_x))
                scale_ids.append(band_id)

    # Sort boxes deterministically by (scale_id, y1, x1, y2, x2) to maintain scale-grouping parity
    combined = sorted(zip(boxes, scale_ids, strict=False), key=lambda item: (item[1], item[0]))
    sorted_boxes = [c[0] for c in combined]
    sorted_scales = [c[1] for c in combined]

    box_t = torch.tensor(sorted_boxes, dtype=torch.long).reshape(-1, 4)
    scale_t = torch.tensor(sorted_scales, dtype=torch.long).reshape(-1)
    area_t = ((box_t[:, 2] - box_t[:, 0]) * (box_t[:, 3] - box_t[:, 1])).float()

    return box_t.clone(), scale_t.clone(), area_t.clone(), list(sorted_boxes), tuple(band_sizes)


def build_perspective_regions(
    height: int,
    width: int,
    output_stride: int = 4,
    horizon_size_px: int = 16,
    foreground_size_px: int = 128,
    max_aspect_ratio: float = 2.0,
    overlap: float = 0.5,
    num_altitude_bands: int = 3,
    device: torch.device | str | None = None,
) -> RegionSet:
    """Build deterministic Perspective-Adaptive Regional Kernels (PARK) with LRU caching.

    Returns RegionSet compatible with all RMR forward and adjoint solvers.
    """
    box_t, scale_t, area_t, boxes_list, band_sizes = _build_perspective_regions_cached(
        height=int(height),
        width=int(width),
        output_stride=int(output_stride),
        horizon_size_px=int(horizon_size_px),
        foreground_size_px=int(foreground_size_px),
        max_aspect_ratio=float(max_aspect_ratio),
        overlap=float(overlap),
        num_altitude_bands=int(num_altitude_bands),
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
        num_scales=int(num_altitude_bands),
        scale_sizes_px=band_sizes,
    )


def park_forward_operator(
    y_density: torch.Tensor,
    float_boxes: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact O(1) continuous Radon integration over arbitrary continuous boxes.

    y_density: [B, C, H, W] or [B, H, W] or [H, W]
    float_boxes: [M, 4] with (y1, x1, y2, x2) in continuous coordinates.
    Returns: [B, C, M] continuous area integrals.
    """
    if y_density.ndim == 2:
        y_density = y_density.unsqueeze(0).unsqueeze(0)
    elif y_density.ndim == 3:
        y_density = y_density.unsqueeze(1)
    elif y_density.ndim != 4:
        raise ValueError(f"y_density must have 2, 3, or 4 dims, got shape {tuple(y_density.shape)}")

    orig_dtype = y_density.dtype if out_dtype is None else out_dtype
    pref = prefix2d(y_density, preserve_fp32=True)
    float_boxes = float_boxes.to(device=y_density.device, dtype=torch.float32)

    y1, x1, y2, x2 = float_boxes.unbind(dim=-1)
    br = continuous_prefix_eval(pref, y2, x2)
    tr = continuous_prefix_eval(pref, y1, x2)
    bl = continuous_prefix_eval(pref, y2, x1)
    tl = continuous_prefix_eval(pref, y1, x1)

    res = br - tr - bl + tl
    return res.to(orig_dtype) if res.dtype != orig_dtype else res


def park_adjoint_operator(
    residuals: torch.Tensor,
    float_boxes: torch.Tensor,
    height: int,
    width: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact O(M + HW) Continuous Bilinear Adjoint Backprojection A_persp^*.

    Backprojects discrete regional residuals to a continuous 2D adjoint correction field
    via Bilinear Corner Scattering and 2D cumulative summation.

    residuals: [B, C, M] or [B, M]
    float_boxes: [M, 4] with (y1, x1, y2, x2) continuous coordinates.
    height, width: output grid dimensions.
    Returns: [B, C, H, W] adjoint field.
    """
    if residuals.ndim == 2:
        residuals = residuals.unsqueeze(1)
    elif residuals.ndim != 3:
        raise ValueError(f"residuals must be [B, C, M] or [B, M], got {tuple(residuals.shape)}")

    b, c, m = residuals.shape
    if float_boxes.shape != (m, 4):
        raise ValueError(f"float_boxes must be [{m}, 4], got {tuple(float_boxes.shape)}")

    float_boxes = float_boxes.to(device=residuals.device, dtype=torch.float32)
    orig_dtype = residuals.dtype if out_dtype is None else out_dtype
    work_res = residuals.float() if residuals.dtype in (torch.float16, torch.bfloat16) else residuals

    hp, wp = height + 1, width + 1
    diff = work_res.new_zeros((b, c, hp * wp))

    y1, x1, y2, x2 = float_boxes.unbind(dim=-1)

    def _scatter_corner(y_f: torch.Tensor, x_f: torch.Tensor, sign: float) -> None:
        yc = y_f.clamp(0.0, float(height))
        xc = x_f.clamp(0.0, float(width))
        r = torch.floor(yc).long().clamp(0, height - 1)
        col = torch.floor(xc).long().clamp(0, width - 1)
        u = (yc - r.float()).view(1, 1, -1)
        v = (xc - col.float()).view(1, 1, -1)

        val = work_res * float(sign)

        # 4 continuous bilinear vertices
        w00 = (1.0 - u) * (1.0 - v)
        w01 = (1.0 - u) * v
        w10 = u * (1.0 - v)
        w11 = u * v

        idx00 = (r * wp + col).view(1, 1, -1).expand(b, c, -1)
        idx01 = (r * wp + (col + 1)).view(1, 1, -1).expand(b, c, -1)
        idx10 = ((r + 1) * wp + col).view(1, 1, -1).expand(b, c, -1)
        idx11 = ((r + 1) * wp + (col + 1)).view(1, 1, -1).expand(b, c, -1)

        diff.scatter_add_(-1, idx00, val * w00)
        diff.scatter_add_(-1, idx01, val * w01)
        diff.scatter_add_(-1, idx10, val * w10)
        diff.scatter_add_(-1, idx11, val * w11)

    # 4 corners with alternating signs: br(+) - tr(-) - bl(-) + tl(+)
    _scatter_corner(y1, x1, +1.0)
    _scatter_corner(y1, x2, -1.0)
    _scatter_corner(y2, x1, -1.0)
    _scatter_corner(y2, x2, +1.0)

    diff_2d = diff.view(b, c, hp, wp)
    # Exact adjoint of padded prefix2d is reverse cumsum (suffix sum)
    suff = torch.flip(
        torch.cumsum(torch.cumsum(torch.flip(diff_2d, dims=[-2, -1]), dim=-2), dim=-1),
        dims=[-2, -1],
    )
    res = suff[..., 1:, 1:]
    return res.to(orig_dtype) if res.dtype != orig_dtype else res

