from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


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
    boxes = boxes.to(device=prefix.device, dtype=torch.long)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
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

    Always evaluates prefix accumulation and 4-point rectangle difference in FP32
    before converting to out_dtype (defaults to x.dtype).
    """
    orig_dtype = x.dtype if out_dtype is None else out_dtype
    pref = prefix2d(x, preserve_fp32=True)
    res = rectangle_sum_from_prefix(pref, boxes)
    return res.to(orig_dtype) if res.dtype != orig_dtype else res


def regional_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    height: int,
    width: int,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Exact adjoint A^T of rectangular summation.

    values: [B,C,M]
    boxes:  [M,4]
    returns [B,C,H,W]

    Uses a 2-D difference buffer followed by cumulative sums.
    Forces FP32 accumulation during autocast to maintain exact precision.
    """
    if values.ndim != 3:
        raise ValueError(f"values must be [B,C,M], got {tuple(values.shape)}")
    b, c, m = values.shape
    if boxes.shape != (m, 4):
        raise ValueError(f"boxes must be [{m},4], got {tuple(boxes.shape)}")

    boxes = boxes.to(device=values.device, dtype=torch.long)
    y1, x1, y2, x2 = boxes.unbind(dim=-1)
    hp, wp = height + 1, width + 1

    orig_dtype = values.dtype if out_dtype is None else out_dtype
    work = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    diff = work.new_zeros((b, c, hp * wp))

    def scatter(y: torch.Tensor, x: torch.Tensor, src: torch.Tensor) -> None:
        idx = (y * wp + x).view(1, 1, -1).expand(b, c, -1)
        diff.scatter_add_(dim=-1, index=idx, src=src)

    scatter(y1, x1, work)
    scatter(y1, x2, -work)
    scatter(y2, x1, -work)
    scatter(y2, x2, work)

    diff = diff.view(b, c, hp, wp)
    field = diff.cumsum(dim=-2).cumsum(dim=-1)
    res = field[..., :height, :width]
    return res.to(orig_dtype) if res.dtype != orig_dtype else res


def multiplicative_gated_adjoint(
    values: torch.Tensor,
    boxes: torch.Tensor,
    y_current: torch.Tensor,
    height: int,
    width: int,
    rho0: float = 0.02,
    gate_floor: float = 0.0,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Multiplicative Gated SIRT adjoint step with recovery floor.

    Suppresses the correction signal on near-zero pixels via a tanh gate,
    preventing background pixels from being lifted off zero during repeated
    SIRT iterations (the "background lift" degradation observed at T>=2 with
    the plain additive adjoint).

    A small gate_floor > 0 (default: 0.02) prevents the "zero-absorbing state"
    where a false-negative zero prediction in y_0 can never receive a positive
    correction from regional evidence.

    The gate is:
        gate(i) = (1.0 - floor) * tanh(|y_current(i)| / rho0) + floor

    Where:
        - At y ~ 0: gate = floor (default 0.02, suppressing background lift by 98%
          while allowing false-negative regions to recover).
        - At y >> rho0: gate = 1.0 (full correction for crowd clusters).

    Args:
        values:     [B, C, M]  residual values to scatter (same as regional_adjoint).
        boxes:      [M, 4]     half-open box coordinates (y1, x1, y2, x2).
        y_current:  [B, 1, H, W]  current density iterate for gate computation.
        height:     output height H.
        width:      output width W.
        rho0:       gate threshold; default 0.02 matches the empirical mean cell density prior.
        gate_floor: lower floor for gate to prevent permanent zero traps (default: 0.02).
        out_dtype:  output dtype (default: values.dtype).

    Returns:
        [B, C, H, W] multiplicatively gated correction field.
    """
    # Compute the standard additive adjoint field in fp32
    field_additive = regional_adjoint(values, boxes, height, width, out_dtype=torch.float32)

    # Gate: (1 - floor) * tanh(|y| / rho0) + floor
    tanh_gate = torch.tanh(y_current.float().abs() / float(rho0))  # [B, 1, H, W]
    floor_val = float(max(0.0, min(gate_floor, 1.0)))
    gate = (1.0 - floor_val) * tanh_gate + floor_val

    # Broadcast gate over C channels if needed
    gated = gate * field_additive  # [B, C, H, W]

    orig_dtype = values.dtype if out_dtype is None else out_dtype
    return gated.to(orig_dtype) if gated.dtype != orig_dtype else gated


def charbonnier_tv_step(
    y: torch.Tensor,
    lambda_tv: float,
    eps_c: float = 0.1,
    enforce_cfl: bool = False,
) -> torch.Tensor:
    """One step of anisotropic Charbonnier Total Variation diffusion.

    Applies: y_out = clamp(y + lambda_tv * div(g * grad(y)), min=0)
    where:
        grad(y) -- forward finite differences in x and y
        g(i) = 1 / sqrt(|grad_y(i)|^2 + eps_c^2)   (Charbonnier weight)
        div    -- backward finite difference divergence

    This is edge-preserving: near sharp edges (|grad_y| >> eps_c), g->0 so
    diffusion is suppressed. In flat regions (|grad_y| -> 0), g -> 1/eps_c
    so diffusion is near-isotropic.

    Stability (Von Neumann CFL):
        The strict 2D explicit Euler CFL bound is: lambda_tv <= eps_c / 4.
        If enforce_cfl=True and lambda_tv > eps_c / 4, lambda_tv is safely
        clamped to eps_c / 4 to guarantee contractivity and prevent checkerboard
        instability or exploding autograd Jacobians.

    Args:
        y:           [B, C, H, W] current density map (float32 expected).
        lambda_tv:   TV diffusion coefficient (typical: 0.010-0.020).
        eps_c:       Charbonnier regularization epsilon (default 0.1).
        enforce_cfl: If True, clamp lambda_tv to eps_c / 4 for strict stability.

    Returns:
        [B, C, H, W] diffused density map, clamped >= 0.
    """
    y_f = y.float()
    cfl_bound = float(eps_c) / 4.0
    eff_lambda = min(float(lambda_tv), cfl_bound) if enforce_cfl else float(lambda_tv)

    # Forward finite differences for gradient
    # dy_dx: shift in column direction (right neighbour - current), pad right edge with 0
    dy_dx = F.pad(y_f[..., 1:] - y_f[..., :-1], (0, 1))       # [B, C, H, W]
    # dy_dy: shift in row direction (bottom neighbour - current), pad bottom edge with 0
    dy_dy = F.pad(y_f[..., 1:, :] - y_f[..., :-1, :], (0, 0, 0, 1))  # [B, C, H, W]

    # Charbonnier diffusivity: g = 1 / sqrt(|grad|^2 + eps_c^2)
    grad_sq = dy_dx.pow(2) + dy_dy.pow(2)
    g = (grad_sq + float(eps_c) ** 2).rsqrt()  # [B, C, H, W]

    # Flux: F_x = g * dy_dx,  F_y = g * dy_dy
    flux_x = g * dy_dx  # [B, C, H, W]
    flux_y = g * dy_dy  # [B, C, H, W]

    # Backward finite difference divergence: div(F) = dF_x/dx + dF_y/dy
    # dF_x/dx = F_x(i) - F_x(i-1): pad left edge with 0
    div_x = flux_x - F.pad(flux_x[..., :-1], (1, 0))
    # dF_y/dy = F_y(i) - F_y(i-1): pad top edge with 0
    div_y = flux_y - F.pad(flux_y[..., :-1, :], (0, 0, 1, 0))

    divergence = div_x + div_y  # [B, C, H, W]

    y_out = torch.clamp_min(y_f + eff_lambda * divergence, 0.0)
    return y_out.to(y.dtype)


def weighted_coverage(
    weight: torch.Tensor,
    regions: RegionSet,
    height: int,
    width: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute D_{c,w} diagonal field = A^T w."""
    cov = regional_adjoint(
        weight.float(),
        regions.boxes,
        height,
        width,
        out_dtype=torch.float32,
    )
    return cov.clamp_min(float(eps))


def weighted_normalized_adjoint_field(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
    *,
    weighted_cov: torch.Tensor | None = None,
    residual_clip: float = 0.0,
    eps: float = 1e-6,
    solver_mode: str = "additive",
    density_gate_rho: float = 0.02,
    density_gate_floor: float = 0.02,
) -> torch.Tensor:
    """Compute:

        r = D_cw^-1 A^T W D_a^-1 (A y - b)

    entirely in float32. In multiplicative mode, A^T is replaced with
    multiplicative_gated_adjoint to suppress corrections on near-zero pixels.
    """
    _, _, h, w = y.shape

    y32 = y.float()
    b32 = b_region.float()
    weight32 = weight.float()

    q = regional_sum(
        y32,
        regions.boxes,
        out_dtype=torch.float32,
    )

    delta = q - b32

    area = regions.area.float().view(1, 1, -1)
    rate_residual = delta / area.clamp_min(1.0)

    weighted_residual = weight32 * rate_residual

    if solver_mode == "multiplicative":
        back = multiplicative_gated_adjoint(
            weighted_residual,
            regions.boxes,
            y32,
            h,
            w,
            rho0=density_gate_rho,
            gate_floor=density_gate_floor,
            out_dtype=torch.float32,
        )
    else:
        back = regional_adjoint(
            weighted_residual,
            regions.boxes,
            h,
            w,
            out_dtype=torch.float32,
        )

    if weighted_cov is None:
        weighted_cov = weighted_coverage(
            weight32,
            regions,
            h,
            w,
            eps=eps,
        )

    field = back / weighted_cov.float().clamp_min(eps)

    if residual_clip > 0:
        field = field.clamp(
            -float(residual_clip),
            float(residual_clip),
        )

    return field


def weighted_regional_energy(
    y: torch.Tensor,
    b_region: torch.Tensor,
    weight: torch.Tensor,
    regions: RegionSet,
) -> torch.Tensor:
    """Per-sample weighted regional energy.

        E = 1/2 sum_R w_R * (Ay-b)^2 / area_R
    """
    q = regional_sum(
        y.float(),
        regions.boxes,
        out_dtype=torch.float32,
    )

    delta = q - b_region.float()

    area = regions.area.float().view(1, 1, -1)

    energy = 0.5 * (
        weight.float()
        * delta.square()
        / area.clamp_min(1.0)
    ).sum(dim=(-2, -1))

    return energy


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

    box_t = torch.tensor(boxes, dtype=torch.long)
    scale_t = torch.tensor(scale_ids, dtype=torch.long)
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


def region_average_features(features: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Average pooled region features: [B,C,H,W] -> [B,M,C]."""
    sums = regional_sum(features, boxes, out_dtype=torch.float32)  # [B,C,M]
    area = ((boxes[:, 2] - boxes[:, 0]).abs() * (boxes[:, 3] - boxes[:, 1]).abs()).float()
    avg = sums / area.view(1, 1, -1).clamp_min(1.0)
    return avg.transpose(1, 2).contiguous().to(features.dtype)


def region_mean_std_features(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract both spatial mean and standard deviation over bounding boxes in FP32.

    Uses a shifted two-pass variance calculation (subtracting channel spatial mean)
    to eliminate floating-point catastrophic cancellation under unnormalized or shifted features.
    """
    f32 = feature.float()
    shift = f32.mean(dim=(-2, -1), keepdim=True)
    f32_centered = f32 - shift

    mean_centered = region_average_features(f32_centered, boxes)
    mean_sq_centered = region_average_features(f32_centered.square(), boxes)
    var = (mean_sq_centered - mean_centered.square()).clamp_min(0.0)
    std = torch.sqrt(var + eps)
    mean = mean_centered + shift.view(f32.shape[0], 1, f32.shape[1])
    return torch.cat([mean, std], dim=-1).to(feature.dtype)


def fractional_region_average_features(
    features: torch.Tensor,
    float_boxes: torch.Tensor,
) -> torch.Tensor:
    """Average pooled region features using exact continuous fractional overlap.

    features: [B, C, H, W]
    float_boxes: [M, 4] with (y1, x1, y2, x2) in continuous coordinates on the features grid.
    returns: [B, M, C] in features.dtype
    """
    if float_boxes.ndim != 2 or float_boxes.shape[-1] != 4:
        raise ValueError("float_boxes must have shape [M, 4]")

    h, w = features.shape[-2:]
    y1, x1, y2, x2 = float_boxes.float().unbind(dim=-1)
    y_min = torch.minimum(y1, y2).clamp(0.0, float(h))
    y_max = torch.maximum(y1, y2).clamp(0.0, float(h))
    x_min = torch.minimum(x1, x2).clamp(0.0, float(w))
    x_max = torch.maximum(x1, x2).clamp(0.0, float(w))

    clamped_boxes = torch.stack([y_min, x_min, y_max, x_max], dim=-1)
    pref = prefix2d(features, preserve_fp32=True)
    sums = fractional_box_sum(pref, clamped_boxes)  # [B, C, M] in fp32

    area = ((y_max - y_min) * (x_max - x_min)).clamp_min(1e-6)  # [M]
    avg = sums / area.view(1, 1, -1)
    return avg.transpose(1, 2).contiguous().to(features.dtype)


def fractional_region_mean_std_features(
    feature: torch.Tensor,
    float_boxes: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Extract both spatial mean and standard deviation over continuous boxes in FP32.

    Uses a shifted two-pass variance calculation (subtracting channel spatial mean)
    to eliminate floating-point catastrophic cancellation under unnormalized or shifted features.
    """
    f32 = feature.float()
    shift = f32.mean(dim=(-2, -1), keepdim=True)
    f32_centered = f32 - shift

    mean_centered = fractional_region_average_features(f32_centered, float_boxes)
    mean_sq_centered = fractional_region_average_features(f32_centered.square(), float_boxes)
    var = (mean_sq_centered - mean_centered.square()).clamp_min(0.0)
    std = torch.sqrt(var + eps)
    mean = mean_centered + shift.view(f32.shape[0], 1, f32.shape[1])
    return torch.cat([mean, std], dim=-1).to(feature.dtype)


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
    boxes = boxes.to(device=values.device, dtype=torch.long)
    y = ((boxes[:, 0] + boxes[:, 2] - 1) // 2).long().clamp(0, height - 1)
    x = ((boxes[:, 1] + boxes[:, 3] - 1) // 2).long().clamp(0, width - 1)
    idx = (y * width + x).view(1, 1, m).expand(b, 1, -1)
    out = values.new_zeros((b, 1, height * width))
    cnt = values.new_zeros((b, 1, height * width))
    out.scatter_add_(-1, idx, values)
    cnt.scatter_add_(-1, idx, torch.ones_like(values))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, 1, height, width)
