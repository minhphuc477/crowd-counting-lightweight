from __future__ import annotations

import torch
import torch.nn.functional as F


def prefix2d(x: torch.Tensor, preserve_fp32: bool = True) -> torch.Tensor:
    """Inclusive 2-D prefix sum with a zero top row/left column.

    Input:  [B, C, H, W]
    Output: [B, C, H+1, W+1]

    Forces FP32 accumulation during autocast to maintain exact precision.
    When preserve_fp32=True (default), output is kept in at least float32
    so subsequent rectangle subtractions (br - tr - bl + tl) do not suffer
    from catastrophic cancellation in float16/bfloat16.
    """
    if x.ndim != 4:
        raise ValueError(f"prefix2d expects [B,C,H,W], got {tuple(x.shape)}")
    orig_dtype = x.dtype
    work = x.float() if orig_dtype in (torch.float16, torch.bfloat16) else x
    p = work.cumsum(dim=-2).cumsum(dim=-1)
    p = F.pad(p, (1, 0, 1, 0), mode="constant", value=0.0)
    if not preserve_fp32 and orig_dtype in (torch.float16, torch.bfloat16):
        return p.to(orig_dtype)
    return p


def _gather_prefix(prefix: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gather prefix values at M coordinates for every batch/channel with strict bounds clamping."""
    b, c, hp, wp = prefix.shape
    yc = y.clamp(0, hp - 1)
    xc = x.clamp(0, wp - 1)
    idx = (yc * wp + xc).view(1, 1, -1).expand(b, c, -1)
    return torch.gather(prefix.flatten(-2), dim=-1, index=idx)


def rectangle_sum_from_prefix(prefix: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Rectangle sums using a padded prefix table.

    prefix: [B,C,H+1,W+1]
    boxes:  [M,4] in half-open grid coordinates
    returns [B,C,M]
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("boxes must have shape [M,4]")
    boxes = boxes.to(device=prefix.device, dtype=torch.long)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    _, _, hp, wp = prefix.shape
    y1 = y1.clamp(0, hp - 1)
    x1 = x1.clamp(0, wp - 1)
    y2 = y2.clamp(0, hp - 1)
    x2 = x2.clamp(0, wp - 1)
    br = _gather_prefix(prefix, y2, x2)
    tr = _gather_prefix(prefix, y1, x2)
    bl = _gather_prefix(prefix, y2, x1)
    tl = _gather_prefix(prefix, y1, x1)
    return br - tr - bl + tl


def continuous_prefix_eval(prefix: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate padded prefix sum table at continuous coordinates via exact 2-D bilinear interpolation.

    prefix: [B, C, H+1, W+1]
    y, x: [M] float coordinates on the feature lattice [0, H] x [0, W]
    returns: [B, C, M] interpolated prefix values
    """
    b, c, hp, wp = prefix.shape
    h_max = float(hp - 1)
    w_max = float(wp - 1)

    y_clamped = y.to(device=prefix.device, dtype=torch.float32).clamp(0.0, h_max)
    x_clamped = x.to(device=prefix.device, dtype=torch.float32).clamp(0.0, w_max)

    r = torch.floor(y_clamped).long().clamp(0, hp - 2)
    c_idx = torch.floor(x_clamped).long().clamp(0, wp - 2)

    u = (y_clamped - r.float()).view(1, 1, -1)
    v = (x_clamped - c_idx.float()).view(1, 1, -1)

    p00 = _gather_prefix(prefix, r, c_idx)
    p10 = _gather_prefix(prefix, r + 1, c_idx)
    p01 = _gather_prefix(prefix, r, c_idx + 1)
    p11 = _gather_prefix(prefix, r + 1, c_idx + 1)

    return (1.0 - u) * (1.0 - v) * p00 + u * (1.0 - v) * p10 + (1.0 - u) * v * p01 + u * v * p11


def fractional_box_sum(prefix: torch.Tensor, float_boxes: torch.Tensor) -> torch.Tensor:
    """Exact continuous 2-D integral over continuous bounding boxes via continuous prefix evaluation.

    prefix: [B, C, H+1, W+1]
    float_boxes: [M, 4] with (y1, x1, y2, x2) in continuous feature coordinates
    returns: [B, C, M] continuous area integral
    """
    if float_boxes.ndim != 2 or float_boxes.shape[-1] != 4:
        raise ValueError("float_boxes must have shape [M, 4]")
    float_boxes = float_boxes.to(device=prefix.device, dtype=torch.float32)
    y1, x1, y2, x2 = float_boxes.unbind(dim=-1)
    br = continuous_prefix_eval(prefix, y2, x2)
    tr = continuous_prefix_eval(prefix, y1, x2)
    bl = continuous_prefix_eval(prefix, y2, x1)
    tl = continuous_prefix_eval(prefix, y1, x1)
    return br - tr - bl + tl


def regional_sum(
    x: torch.Tensor,
    boxes: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Linear regional-count operator A: [B,C,H,W] -> [B,C,M].

    Accepts [B,C,H,W], [B,H,W], or [H,W] inputs.
    Always evaluates prefix accumulation and 4-point rectangle difference in FP32
    before converting to out_dtype (defaults to x.dtype).
    """
    orig_dtype = x.dtype if out_dtype is None else out_dtype
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim != 4:
        raise ValueError(f"regional_sum expects 2D, 3D, or 4D tensor, got shape {tuple(x.shape)}")

    pref = prefix2d(x, preserve_fp32=True)
    res = rectangle_sum_from_prefix(pref, boxes)
    return res.to(orig_dtype) if res.dtype != orig_dtype else res
