"""Backward-compatibility shim re-exporting from rmr_core.operators."""
from __future__ import annotations

from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    prefix2d,
    rectangle_sum_from_prefix,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)

__all__ = [
    "RegionSet",
    "build_multiscale_regions",
    "center_scatter",
    "prefix2d",
    "rectangle_sum_from_prefix",
    "region_average_features",
    "region_geometry",
    "regional_adjoint",
    "regional_sum",
]
