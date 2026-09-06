"""Backward-compatibility shim re-exporting from rmr_core.metrics."""
from __future__ import annotations

from rmr_core.metrics import (
    bootstrap_ci,
    compute_nae,
    count_from_map,
    density_stratified_mae,
    game_physical_image,
    game_single,
    summarize_predictions,
)

__all__ = [
    "bootstrap_ci",
    "compute_nae",
    "count_from_map",
    "density_stratified_mae",
    "game_physical_image",
    "game_single",
    "summarize_predictions",
]
