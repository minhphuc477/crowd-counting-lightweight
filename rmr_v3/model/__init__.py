from __future__ import annotations

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import build_fine_head
from rmr_core.necks import AdditiveFPNNeck, ASPPLiteFPNNeck, CoordinateAttention, RepWeightedFPNNeck
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    charbonnier_tv_step,
    multiplicative_gated_adjoint,
    region_mean_std_features,
    regional_adjoint,
    regional_sum,
    weighted_coverage,
    weighted_normalized_adjoint_field,
    weighted_regional_energy,
)
from rmr_core.scale_routing import FactorizedRoutingHead, ScaleRoutingHead
from rmr_core.types import RMRModelOutput
from ..regional_head import (
    ProbabilisticRegionalEvidenceHead,
    apply_scale_consistency_gating,
    reliability_from_nb,
)
from ..solver import unrolled_sirt_solver
from .architecture import RMRv3
from .config import RMRv3Config
from .evidence import extract_regional_evidence
from .perspective import MicroCoordAttn, MicroPerspectiveElevation
from .solver_step import solve_inverse_measure

__all__ = [
    "AdditiveFPNNeck",
    "ASPPLiteFPNNeck",
    "CoordinateAttention",
    "FactorizedRoutingHead",
    "MicroCoordAttn",
    "MicroPerspectiveElevation",
    "MobileNetV4Backbone",
    "ProbabilisticRegionalEvidenceHead",
    "RMRModelOutput",
    "RMRv3",
    "RMRv3Config",
    "RegionSet",
    "RepWeightedFPNNeck",
    "ScaleRoutingHead",
    "apply_scale_consistency_gating",
    "build_fine_head",
    "build_multiscale_regions",
    "center_scatter",
    "charbonnier_tv_step",
    "extract_regional_evidence",
    "multiplicative_gated_adjoint",
    "region_mean_std_features",
    "regional_adjoint",
    "regional_sum",
    "reliability_from_nb",
    "solve_inverse_measure",
    "unrolled_sirt_solver",
    "weighted_coverage",
    "weighted_normalized_adjoint_field",
    "weighted_regional_energy",
]
