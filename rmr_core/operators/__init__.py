from __future__ import annotations

from .adjoint import (
    center_scatter,
    multiplicative_gated_adjoint,
    regional_adjoint,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from .diffusion import charbonnier_tv_step
from .pooling import (
    fractional_region_average_features,
    fractional_region_mean_std_features,
    region_average_features,
    region_mean_std_features,
)
from .prefix_sums import (
    _gather_prefix,
    continuous_prefix_eval,
    fractional_box_sum,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_sum,
)
from .regions import (
    RegionSet,
    _axis_starts,
    _build_multiscale_regions_cached,
    _canonicalize_region_size,
    build_multiscale_regions,
    partition_regions_by_scale,
    region_geometry,
)

__all__ = [
    "RegionSet",
    "_axis_starts",
    "_build_multiscale_regions_cached",
    "_canonicalize_region_size",
    "_gather_prefix",
    "build_multiscale_regions",
    "center_scatter",
    "charbonnier_tv_step",
    "continuous_prefix_eval",
    "fractional_box_sum",
    "fractional_region_average_features",
    "fractional_region_mean_std_features",
    "multiplicative_gated_adjoint",
    "partition_regions_by_scale",
    "prefix2d",
    "rectangle_sum_from_prefix",
    "region_average_features",
    "region_geometry",
    "region_mean_std_features",
    "regional_adjoint",
    "regional_sum",
    "weighted_coverage",
    "weighted_normalized_adjoint_field",
    "weighted_regional_energy",
]
