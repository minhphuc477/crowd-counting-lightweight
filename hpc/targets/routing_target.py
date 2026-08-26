"""Soft routing target builder for the Shared Scale-Evidence Router (SSER).

For each /8-resolution grid cell, compute a soft 4-way scale preference
distribution q_scale ∈ Δ³ from the nearest-neighbour distances of annotated
points that fall in that cell.

The scale-preference mapping (Table):
  d_NN ≤ 16 px   → dense/small crowd   → prefer /4    [0.70, 0.20, 0.08, 0.02]
  16 < d_NN ≤ 32 → medium density       → prefer /8    [0.20, 0.60, 0.16, 0.04]
  32 < d_NN ≤ 64 → sparse/large crowd   → prefer /16   [0.05, 0.20, 0.60, 0.15]
  d_NN > 64 px   → isolated/near-camera → prefer /32   [0.02, 0.08, 0.20, 0.70]

Cells with no annotated points are left at uniform [0.25]*4 but flagged as
``no supervision`` via the returned boolean mask (KL loss is only applied where
the mask is True).

When multiple annotated points land in the same /8 cell, their soft
distributions are averaged in the probability simplex before normalising.

Usage:
    result = build_routing_target(crop_pts, d_nn, 448, 448)
    q = result["gt_route_q"]          # torch.FloatTensor (4, 56, 56)
    mask = result["gt_route_mask"]    # torch.BoolTensor  (56, 56)
"""

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Scale preference distributions (sum to 1.0, all positive for numeric stability)
# Ordering: [α_/4, α_/8, α_/16, α_/32]
# ──────────────────────────────────────────────────────────────────────────────
_DENSE_DIST  = np.array([0.70, 0.20, 0.08, 0.02], dtype=np.float32)  # d_nn ≤ 16
_MEDIUM_DIST = np.array([0.20, 0.60, 0.16, 0.04], dtype=np.float32)  # 16 < d_nn ≤ 32
_SPARSE_DIST = np.array([0.05, 0.20, 0.60, 0.15], dtype=np.float32)  # 32 < d_nn ≤ 64
_LARGE_DIST  = np.array([0.02, 0.08, 0.20, 0.70], dtype=np.float32)  # d_nn > 64
_UNIFORM     = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)  # no points

assert all(np.isclose(d.sum(), 1.0) for d in [_DENSE_DIST, _MEDIUM_DIST, _SPARSE_DIST, _LARGE_DIST])


def _dnn_to_soft_dist(dnn_value: float) -> np.ndarray:
    """Map a scalar nearest-neighbour distance to a 4-way soft distribution."""
    if dnn_value <= 16.0:
        return _DENSE_DIST
    elif dnn_value <= 32.0:
        return _MEDIUM_DIST
    elif dnn_value <= 64.0:
        return _SPARSE_DIST
    else:
        return _LARGE_DIST


def build_routing_target(
    crop_points: np.ndarray,
    d_nn: np.ndarray,
    crop_h: int,
    crop_w: int,
    route_stride: int = 8,
) -> dict:
    """Build per-cell soft routing target at /8 resolution.

    Args:
        crop_points: (M, 2) float32 array of (x, y) point centres in crop space.
        d_nn: (M,) float32 array of nearest-neighbour distances (scale proxy).
        crop_h, crop_w: spatial dimensions of the crop in input pixels.
            Both must be divisible by route_stride.
        route_stride: backbone stride for routing (default 8 → /8).

    Returns:
        dict with:
            ``gt_route_q``    FloatTensor (4, H/stride, W/stride) — soft target
            ``gt_route_mask`` BoolTensor  (H/stride, W/stride)    — True where supervised
    """
    if crop_h % route_stride != 0 or crop_w % route_stride != 0:
        raise ValueError(
            f"crop ({crop_h}, {crop_w}) must be divisible by route_stride={route_stride}"
        )

    h_r = crop_h // route_stride
    w_r = crop_w // route_stride

    pts = np.asarray(crop_points, dtype=np.float32).reshape(-1, 2)
    dnn = np.asarray(d_nn, dtype=np.float32).reshape(-1)

    if len(pts) != len(dnn):
        raise ValueError(
            f"crop_points ({len(pts)}) and d_nn ({len(dnn)}) must have equal length"
        )

    # Accumulate soft distributions per /8 cell
    q_accum = np.zeros((4, h_r, w_r), dtype=np.float64)
    count   = np.zeros((h_r, w_r), dtype=np.int32)

    if len(pts) > 0:
        # Convert point coordinates to /8 grid indices
        gx = np.clip((pts[:, 0] / route_stride).astype(np.int32), 0, w_r - 1)
        gy = np.clip((pts[:, 1] / route_stride).astype(np.int32), 0, h_r - 1)

        for i in range(len(pts)):
            soft = _dnn_to_soft_dist(float(dnn[i]))
            q_accum[:, gy[i], gx[i]] += soft
            count[gy[i], gx[i]] += 1

    # Build output tensors
    q_out = np.broadcast_to(_UNIFORM[:, None, None], (4, h_r, w_r)).copy()
    mask   = count > 0
    if mask.any():
        q_out[:, mask] = (q_accum[:, mask] / count[None, mask]).astype(np.float32)
        # Re-normalise in float32 to avoid any accumulated rounding errors
        q_sum = q_out[:, mask].sum(axis=0, keepdims=True)
        q_out[:, mask] /= np.where(q_sum > 0, q_sum, 1.0)

    return {
        "gt_route_q":    torch.from_numpy(q_out.astype(np.float32)),
        "gt_route_mask": torch.from_numpy(mask),
    }
