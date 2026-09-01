"""Parameter-free OT-M localization for a conserved count-mass map.

This implementation adapts the alternating OT-step/M-step of Lin & Chan,
CVPR 2023 to NTPC's zero-based pixel-center convention. Coordinates returned
by this module are always ``(x, y)`` in input-image pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


_TINY = 1e-12
DEFAULT_OTM_MAX_SOURCE_POINTS: int = 4096
DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS: int = 5_000_000
DEFAULT_OTM_MAX_INITIALIZATION_PIXELS: int = 16_000_000


@dataclass(frozen=True)
class OTMConfig:
    output_stride: int = 4
    max_iterations: int = 16
    ot_scaling: float = 0.75
    blur: float = 0.01
    cost_factor: float = 1.0 / 32.0
    mean_stop_px: float = 1.0
    source_relative_threshold: float = 1e-8
    max_source_points: int | None = DEFAULT_OTM_MAX_SOURCE_POINTS
    max_transport_elements: int = DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS
    initialization_mode: str = "fullres_bilinear"
    max_initialization_pixels: int = DEFAULT_OTM_MAX_INITIALIZATION_PIXELS
    seed: int = 42

    def __post_init__(self) -> None:
        if self.output_stride <= 0 or self.max_iterations <= 0:
            raise ValueError("output_stride and max_iterations must be positive")
        if not 0.0 < self.ot_scaling < 1.0:
            raise ValueError("ot_scaling must lie in (0, 1)")
        if self.blur <= 0 or self.cost_factor <= 0 or self.mean_stop_px <= 0:
            raise ValueError("blur, cost_factor, and mean_stop_px must be positive")
        if self.source_relative_threshold < 0:
            raise ValueError("source_relative_threshold cannot be negative")
        if self.max_source_points is not None and self.max_source_points <= 0:
            raise ValueError("max_source_points must be positive or None")
        if self.max_transport_elements <= 0:
            raise ValueError("max_transport_elements must be positive")
        if self.initialization_mode not in {"fullres_bilinear", "stride_grid"}:
            raise ValueError(
                "initialization_mode must be 'fullres_bilinear' or 'stride_grid'"
            )
        if self.max_initialization_pixels <= 0:
            raise ValueError("max_initialization_pixels must be positive")


def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.02,
    iterations: int = 50,
) -> torch.Tensor:
    """Balanced fixed-epsilon Sinkhorn helper retained for unit-level checks."""
    a = a.float().clamp_min(_TINY)
    b = b.float().clamp_min(_TINY)
    cost = cost.float()
    if not torch.allclose(a.sum(), b.sum(), atol=1e-3, rtol=1e-3):
        raise ValueError(f"Unbalanced OT: sum(a)={a.sum():.4f} != sum(b)={b.sum():.4f}")
    log_a, log_b = a.log(), b.log()
    log_kernel = -cost / float(epsilon)
    log_u, log_v = torch.zeros_like(a), torch.zeros_like(b)
    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
    return torch.exp(log_u[:, None] + log_kernel + log_v[None, :])


def _epsilon_schedule(diameter: float, blur: float, scaling: float) -> list[float]:
    diameter = max(float(diameter), 8.0)
    values = np.arange(np.log(diameter), np.log(blur), np.log(scaling))
    return [diameter, *[float(np.exp(value)) for value in values], float(blur)]


def _softmin(log_weight: torch.Tensor, potential: torch.Tensor, cost: torch.Tensor, epsilon: float) -> torch.Tensor:
    batch = cost.shape[0]
    values = log_weight.view(batch, 1, -1) + (
        potential.view(batch, 1, -1) - cost
    ) / epsilon
    return -epsilon * values.logsumexp(dim=2).view(batch, -1, 1)


@torch.no_grad()
def _epsilon_scaling_transport_plan(
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
    cost: torch.Tensor,
    blur: float,
    scaling: float,
) -> torch.Tensor:
    """Official balanced SampleOT dual updates followed by plan recovery."""
    log_source = source_weight.clamp_min(_TINY).log()
    log_target = target_weight.clamp_min(_TINY).log()
    reverse_cost = cost.permute(0, 2, 1)
    epsilons = _epsilon_schedule(float(cost.max()), blur, scaling)
    source_potential = _softmin(
        log_target, torch.zeros_like(target_weight), cost, epsilons[0]
    )
    target_potential = _softmin(
        log_source, torch.zeros_like(source_weight), reverse_cost, epsilons[0]
    )
    for epsilon in epsilons:
        next_source = _softmin(log_target, target_potential, cost, epsilon)
        next_target = _softmin(log_source, source_potential, reverse_cost, epsilon)
        source_potential = 0.5 * (source_potential + next_source)
        target_potential = 0.5 * (target_potential + next_target)

    # Official upstream simultaneous update for final plan
    new_source = _softmin(log_target, target_potential, cost, blur)
    new_target = _softmin(log_source, source_potential, reverse_cost, blur)
    source_potential = new_source
    target_potential = new_target

    kernel = torch.exp(
        (source_potential + target_potential.permute(0, 2, 1) - cost) / blur
    )
    return kernel * source_weight * target_weight.permute(0, 2, 1)


def _source_distribution(
    mass: torch.Tensor,
    config: OTMConfig,
    image_hw: Tuple[int, int] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    """Convert the stride mass grid into weighted source coordinates ``(y,x)``."""
    height, width = mass.shape
    flat = mass.reshape(-1)
    threshold = max(_TINY, float(flat.max()) * config.source_relative_threshold)
    indices = torch.nonzero(flat > threshold, as_tuple=False).flatten()
    if indices.numel() == 0:
        indices = torch.argmax(flat).reshape(1)
    candidate_weight = flat[indices]
    candidate_mass = candidate_weight.sum()
    rows = torch.div(indices, width, rounding_mode="floor").float()
    columns = (indices % width).float()
    stride = float(config.output_stride)

    if image_hw is not None:
        image_h, image_w = float(image_hw[0]), float(image_hw[1])
        y0 = rows * stride
        x0 = columns * stride
        y1 = torch.minimum(y0 + stride - 1.0, y0.new_tensor(image_h - 1.0))
        x1 = torch.minimum(x0 + stride - 1.0, x0.new_tensor(image_w - 1.0))
        coordinates_yx = torch.stack((0.5 * (y0 + y1), 0.5 * (x0 + x1)), dim=-1)
    else:
        cell_center_offset = (stride - 1.0) / 2.0
        coordinates_yx = torch.stack(
            (rows * stride + cell_center_offset, columns * stride + cell_center_offset), dim=-1
        )
    compaction = "none"
    coarse_height, coarse_width = height, width
    if config.max_source_points is not None and indices.numel() > config.max_source_points:
        # Aggregate the complete thresholded measure into a deterministic
        # coarse grid. Preserves total source mass and represents each bin at its mass-weighted barycenter.
        max_points = int(config.max_source_points)
        aspect = float(height) / max(float(width), 1.0)
        target_height = max(1, int(round(math.sqrt(max_points * aspect))))
        coarse_height = min(height, max_points, target_height)
        coarse_width = min(width, max(1, max_points // coarse_height))
        assert coarse_height >= 1
        assert coarse_width >= 1
        assert coarse_height * coarse_width <= max_points

        row_bucket = torch.clamp((rows * coarse_height / height).long(), max=coarse_height - 1)
        col_bucket = torch.clamp((columns * coarse_width / width).long(), max=coarse_width - 1)
        bucket = row_bucket * coarse_width + col_bucket
        bucket_count = coarse_height * coarse_width
        aggregate_weight = candidate_weight.new_zeros(bucket_count)
        aggregate_coord = coordinates_yx.new_zeros(bucket_count, 2)
        aggregate_weight.scatter_add_(0, bucket, candidate_weight)
        aggregate_coord.scatter_add_(
            0, bucket[:, None].expand(-1, 2), coordinates_yx * candidate_weight[:, None]
        )
        nonempty = aggregate_weight > 0
        coordinates_yx = aggregate_coord[nonempty] / aggregate_weight[nonempty, None]
        candidate_weight = aggregate_weight[nonempty]
        compaction = "mass_weighted_grid_barycenter"
    retained_mass = candidate_weight.sum()
    diagnostics = {
        "source_candidates": int(torch.count_nonzero(flat > threshold)),
        "source_points": int(candidate_weight.numel()),
        "source_retained_mass_ratio": float(retained_mass / mass.sum().clamp_min(_TINY)),
        "threshold_retained_mass_ratio": float(candidate_mass / mass.sum().clamp_min(_TINY)),
        "source_compaction": compaction,
        "source_coarse_height": coarse_height,
        "source_coarse_width": coarse_width,
    }
    return candidate_weight, coordinates_yx, diagnostics


def _initialize_target_points(
    mass: torch.Tensor,
    point_count: int,
    image_height: int,
    image_width: int,
    seed: int,
    output_stride: int,
    mode: str,
    max_initialization_pixels: int,
) -> torch.Tensor:
    """Density-weighted initialization with an explicit memory policy."""
    if mode == "fullres_bilinear":
        pixel_count = image_height * image_width
        if pixel_count > max_initialization_pixels:
            raise MemoryError(
                f"OT-M full-resolution initialization requires {pixel_count:,} pixels; "
                f"limit is {max_initialization_pixels:,}. Select initialization_mode='stride_grid' "
                "explicitly for large images."
            )
        resized = F.interpolate(
            mass[None, None],
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0].clamp_min(0.0)
        flat = resized.reshape(-1)
        grid_width = image_width
    else:
        flat = mass.clamp_min(0.0).reshape(-1)
        grid_width = int(mass.shape[1])
    weights = torch.where(flat > _TINY, flat, torch.zeros_like(flat))
    pos_count = int(torch.count_nonzero(weights))
    if pos_count == 0:
        weights = torch.ones_like(flat)
        pos_count = weights.numel()
    replacement = point_count > pos_count
    generator = torch.Generator(device=mass.device)
    generator.manual_seed(int(seed))
    chosen = torch.multinomial(
        weights,
        num_samples=point_count,
        replacement=replacement,
        generator=generator,
    )
    rows = torch.div(chosen, grid_width, rounding_mode="floor")
    cols = chosen % grid_width
    if mode == "stride_grid":
        stride = float(output_stride)
        y0 = rows.float() * stride
        x0 = cols.float() * stride
        y1 = torch.minimum(y0 + stride - 1.0, y0.new_tensor(float(image_height - 1)))
        x1 = torch.minimum(x0 + stride - 1.0, x0.new_tensor(float(image_width - 1)))
        rows = 0.5 * (y0 + y1)
        cols = 0.5 * (x0 + x1)
    return torch.stack([rows, cols], dim=-1).float().reshape(1, point_count, 2)


@torch.no_grad()
def otm_localize(
    mass: torch.Tensor,
    output_stride: int = 4,
    outer_iterations: int = 16,
    ot_scaling: float = 0.75,
    blur: float = 0.01,
    cost_factor: float = 1.0 / 32.0,
    mean_stop_px: float = 1.0,
    source_relative_threshold: float = 1e-8,
    max_source_points: int | None = DEFAULT_OTM_MAX_SOURCE_POINTS,
    max_transport_elements: int = DEFAULT_OTM_MAX_TRANSPORT_ELEMENTS,
    initialization_mode: str = "fullres_bilinear",
    max_initialization_pixels: int = DEFAULT_OTM_MAX_INITIALIZATION_PIXELS,
    seed: int = 42,
    image_hw: Tuple[int, int] | None = None,
    return_diagnostics: bool = False,
    sinkhorn_iterations: int | None = None,
    epsilon: float | None = None,
    target_point_count: int | None = None,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, object]]:
    """Decode a single positive mass map with OT-M.
    
    Args:
        target_point_count: Optional explicit point cardinality (e.g. Oracle GT cardinality
            for decoupling count error from spatial allocation error in mechanism analysis).
    """
    if sinkhorn_iterations is not None or epsilon is not None:
        raise ValueError(
            "Fixed-epsilon/sinkhorn_iterations belong to the old approximation; "
            "use ot_scaling and blur for official OT-M."
        )
    if mass.ndim == 3:
        if mass.shape[0] != 1:
            raise ValueError(f"Expected a single-channel map, got {tuple(mass.shape)}")
        mass = mass[0]
    if mass.ndim != 2:
        raise ValueError(f"Expected a 2D mass map, got {tuple(mass.shape)}")
    if not torch.isfinite(mass).all():
        raise ValueError("OT-M input mass contains NaN or Inf values")

    mass = mass.detach().float().clamp_min(0.0)
    config = OTMConfig(
        output_stride=output_stride,
        max_iterations=outer_iterations,
        ot_scaling=ot_scaling,
        blur=blur,
        cost_factor=cost_factor,
        mean_stop_px=mean_stop_px,
        source_relative_threshold=source_relative_threshold,
        max_source_points=max_source_points,
        max_transport_elements=max_transport_elements,
        initialization_mode=initialization_mode,
        max_initialization_pixels=max_initialization_pixels,
        seed=seed,
    )
    predicted_count = float(mass.sum())
    if not math.isfinite(predicted_count) or predicted_count < 0:
        raise ValueError(f"Invalid predicted count from mass map: {predicted_count}")

    if target_point_count is not None:
        point_count = max(0, int(target_point_count))
    else:
        point_count = max(0, int(predicted_count + 0.5))

    diagnostics: Dict[str, object] = {
        "predicted_count": predicted_count,
        "localized_count": point_count,
        "target_point_count_override": target_point_count is not None,
        "cardinality_gap": abs(point_count - predicted_count),
        "iterations": 0,
        **{f"config_{key}": value for key, value in asdict(config).items() if value is not None},
    }
    if point_count == 0:
        diagnostics.update({
            "source_candidates": 0,
            "source_points": 0,
            "source_retained_mass_ratio": 1.0,
            "threshold_retained_mass_ratio": 1.0,
            "source_compaction": "none",
            "source_coarse_height": int(mass.shape[0]),
            "source_coarse_width": int(mass.shape[1]),
            "transport_elements": 0,
        })
        points = mass.new_empty((0, 2))
        return (points, diagnostics) if return_diagnostics else points

    grid_height, grid_width = mass.shape
    if image_hw is None:
        image_height = grid_height * output_stride
        image_width = grid_width * output_stride
    else:
        image_height, image_width = int(image_hw[0]), int(image_hw[1])
        if image_height <= 0 or image_width <= 0:
            raise ValueError("image_hw must contain positive dimensions")

    effective_max_source = config.max_source_points
    if config.max_transport_elements is not None and point_count > 0:
        max_src_by_transport = max(1, config.max_transport_elements // point_count)
        if effective_max_source is None or max_src_by_transport < effective_max_source:
            effective_max_source = max_src_by_transport

    source_config = replace(config, max_source_points=effective_max_source)
    source_mass, source_yx, source_diagnostics = _source_distribution(
        mass, source_config, image_hw=(image_height, image_width)
    )
    diagnostics.update(source_diagnostics)
    transport_elements = int(source_mass.numel()) * point_count
    rough_working_set_mb = float(transport_elements * 32) / (1024.0 * 1024.0)
    diagnostics["transport_elements"] = transport_elements
    diagnostics["rough_transport_working_set_mb_est"] = rough_working_set_mb
    if transport_elements > config.max_transport_elements:
        raise MemoryError(
            f"OT-M transport matrix requires {transport_elements:,} elements "
            f"(~{rough_working_set_mb:.1f} MB estimated working set); "
            f"limit is {config.max_transport_elements:,} elements. Reduce max_source_points explicitly."
        )

    # OT-M uses equal total mass m on the source and m unit target atoms.
    source_weight = (source_mass / source_mass.sum().clamp_min(_TINY) * point_count).reshape(1, -1, 1)
    target_weight = mass.new_ones((1, point_count, 1))
    source_yx = source_yx.reshape(1, -1, 2)
    target_yx = _initialize_target_points(
        mass,
        point_count,
        image_height,
        image_width,
        config.seed,
        config.output_stride,
        config.initialization_mode,
        config.max_initialization_pixels,
    )
    final_yx = target_yx
    for iteration in range(config.max_iterations):
        delta = source_yx.unsqueeze(-2) - target_yx.unsqueeze(-3)
        cost = 0.5 * delta.square().sum(dim=-1) * config.cost_factor
        plan = _epsilon_scaling_transport_plan(
            source_weight, target_weight, cost, config.blur, config.ot_scaling
        )
        normalized_plan = plan / plan.sum(dim=1, keepdim=True).clamp_min(_TINY)
        final_yx = normalized_plan.permute(0, 2, 1) @ source_yx
        movement = torch.linalg.vector_norm(final_yx - target_yx, dim=-1)
        diagnostics["iterations"] = iteration + 1
        diagnostics["mean_movement_px"] = float(movement.mean())
        diagnostics["max_movement_px"] = float(movement.max())
        target_yx = final_yx
        if (
            float(movement.mean()) < config.mean_stop_px
            and float(movement.max()) < config.output_stride
        ):
            break

    # Internal convention is (row=y, column=x); public convention is (x,y).
    points_xy = final_yx.reshape(-1, 2)[:, [1, 0]]
    points_xy[:, 0].clamp_(0.0, max(float(image_width - 1), 0.0))
    points_xy[:, 1].clamp_(0.0, max(float(image_height - 1), 0.0))
    return (points_xy, diagnostics) if return_diagnostics else points_xy


@torch.no_grad()
def infer_count_and_localization(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    pad_multiple: int | None = None,
    seed: int = 42,
) -> Dict[str, Union[torch.Tensor, float, dict]]:
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.shape[0] != 1:
        raise ValueError("OT-M inference accepts one image at a time")
    image_hw = tuple(int(value) for value in image.shape[-2:])
    count, valid_mass = model.predict(image, pad_multiple=pad_multiple)
    points, diagnostics = otm_localize(
        valid_mass[0, 0],
        output_stride=output_stride,
        image_hw=image_hw,
        seed=seed,
        return_diagnostics=True,
    )
    return {
        "mass": valid_mass,
        "count": float(count[0]),
        "points": points,
        "otm_diagnostics": diagnostics,
    }
