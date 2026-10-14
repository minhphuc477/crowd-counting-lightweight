"""Autonomous and Distance-Adaptive Optimization components for RMR."""
from __future__ import annotations

from rmr_core.schedulers import IdentityScheduler, make_cosine_scheduler, make_scheduler
from .builder import build_v3_optimizer, build_v3_scheduler, maybe_run_safe_lr_finder
from .lr_finder import SafeLRFinder
from .safe_prodigy import SafeProdigy
from .wsd_scheduler import make_wsd_scheduler

__all__ = [
    "IdentityScheduler",
    "SafeLRFinder",
    "SafeProdigy",
    "build_v3_optimizer",
    "build_v3_scheduler",
    "make_cosine_scheduler",
    "make_scheduler",
    "make_wsd_scheduler",
    "maybe_run_safe_lr_finder",
]
