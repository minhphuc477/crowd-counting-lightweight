from __future__ import annotations

from .adjoint import (
    center_scatter,
    multiplicative_gated_adjoint,
    regional_adjoint,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from .carrier import (
    compute_deficit_carrier_seed,
    modulate_carrier_field,
)
from .diffusion import charbonnier_tv_step
from .fed_mcbpro import (
    compute_feature_high_pass_energy,
    compute_local_contrast_ratio,
    fed_mcbpro_transport,
    perspective_mcbpro_transport,
)
from .mcbpro import closed_form_mcbpro_transport
from .peak_concentrated import closed_form_peak_concentrated_transport
from .morozov import compute_morozov_discrepancy
from .pooling import (
    fractional_region_average_features,
    fractional_region_mean_std_features,
    fractional_region_sum_features,
    fractional_region_sum_std_features,
    region_average_features,
    region_mean_std_features,
    region_sum_features,
    region_sum_std_features,
)
from .prefix_sums import (
    _gather_prefix,
    continuous_prefix_eval,
    fractional_box_sum,
    prefix2d,
    rectangle_sum_from_prefix,
    regional_sum,
)
from .radon_nikodym import closed_form_radon_nikodym_transport
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
    "closed_form_mcbpro_transport",
    "closed_form_peak_concentrated_transport",
    "closed_form_radon_nikodym_transport",
    "compute_deficit_carrier_seed",
    "compute_feature_high_pass_energy",
    "compute_local_contrast_ratio",
    "compute_morozov_discrepancy",
    "continuous_prefix_eval",
    "fed_mcbpro_transport",
    "fractional_box_sum",
    "fractional_region_average_features",
    "fractional_region_mean_std_features",
    "modulate_carrier_field",
    "multiplicative_gated_adjoint",
    "partition_regions_by_scale",
    "perspective_mcbpro_transport",
    "prefix2d",
    "rectangle_sum_from_prefix",
    "region_average_features",
    "region_geometry",
    "region_mean_std_features",
    "region_sum_features",
    "region_sum_std_features",
    "fractional_region_sum_features",
    "fractional_region_sum_std_features",
    "regional_adjoint",
    "regional_sum",
    "weighted_coverage",
    "weighted_normalized_adjoint_field",
    "weighted_regional_energy",
]
