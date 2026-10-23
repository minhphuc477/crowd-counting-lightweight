"""Perspective-Aware MC-BPRO (PA-MCBPRO) compatibility facade.

NOTE: Renamed to Feature-guided Edge-Directed Bregman-Poisson Recovery Operator (FED-MCBPRO)
in `rmr_core.operators.fed_mcbpro` to reflect true mathematical formulation and eliminate
the perspective misnomer. This module provides 100% backward compatibility for imports.
"""
from __future__ import annotations

from .fed_mcbpro import (
    compute_feature_high_pass_energy,
    compute_local_contrast_ratio,
    fed_mcbpro_transport,
    perspective_mcbpro_transport,
)

__all__ = [
    "compute_feature_high_pass_energy",
    "compute_local_contrast_ratio",
    "fed_mcbpro_transport",
    "perspective_mcbpro_transport",
]
