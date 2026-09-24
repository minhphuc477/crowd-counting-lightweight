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
from .perspective_geometry import (
    DiAGScaleRoutingHead,
    DynamicCameraAnglePredictor,
    PARKRoutingHead,
    PerspectiveGeometryHead,
)
from .solver_step import solve_inverse_measure
from .dual_lattice import (
    push_forward_stride2_to_stride4,
    pullback_stride4_to_stride2_rn,
    check_mass_conservation,
)

__all__ = [
    "AdditiveFPNNeck",
    "ASPPLiteFPNNeck",
    "CoordinateAttention",
    "DiAGScaleRoutingHead",
    "DynamicCameraAnglePredictor",
    "FactorizedRoutingHead",
    "MicroCoordAttn",
    "MicroPerspectiveElevation",
    "MobileNetV4Backbone",
    "PARKRoutingHead",
    "PerspectiveGeometryHead",
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
    "check_mass_conservation",
    "extract_regional_evidence",
    "multiplicative_gated_adjoint",
    "pullback_stride4_to_stride2_rn",
    "push_forward_stride2_to_stride4",
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
