from __future__ import annotations

from .backbones import MobileNetV4Backbone
from .data import (
    CrowdManifestDataset,
    collate_eval,
    collate_train,
    compute_manifest_density,
    normalize_image,
    rasterize_points,
    train_transform,
)
from .evaluation import evaluate_dataset, predict_tiled, save_evaluation_artifacts
from .heads import FineMeasureHead
from .metrics import (
    bootstrap_ci,
    compute_nae,
    density_stratified_mae,
    game_physical_image,
    game_single,
    summarize_predictions,
)
from .necks import (
    AdditiveFPNNeck,
    AdditiveFusion,
    ConvGNAct,
    DepthwiseDilated,
    DSResidual,
    TinyIR,
)
from .operators import (
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
from .training import make_scheduler, seed_everything

__all__ = [
    "MobileNetV4Backbone",
    "CrowdManifestDataset",
    "collate_eval",
    "collate_train",
    "compute_manifest_density",
    "normalize_image",
    "rasterize_points",
    "train_transform",
    "evaluate_dataset",
    "predict_tiled",
    "save_evaluation_artifacts",
    "FineMeasureHead",
    "bootstrap_ci",
    "compute_nae",
    "density_stratified_mae",
    "game_physical_image",
    "game_single",
    "summarize_predictions",
    "AdditiveFPNNeck",
    "AdditiveFusion",
    "ConvGNAct",
    "DepthwiseDilated",
    "DSResidual",
    "TinyIR",
    "RegionSet",
    "build_multiscale_regions",
    "center_scatter",
    "prefix2d",
    "rectangle_sum_from_prefix",
    "region_average_features",
    "region_geometry",
    "regional_adjoint",
    "regional_sum",
    "make_scheduler",
    "seed_everything",
]
