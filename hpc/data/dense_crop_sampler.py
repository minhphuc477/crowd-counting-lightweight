"""Dense crop sampling utilities for HPC-Lite."""
from __future__ import annotations

from .sampler import (
    compute_image_density_and_luminance,
    build_density_luminance_sampler,
)

__all__ = [
    "compute_image_density_and_luminance",
    "build_density_luminance_sampler",
]
