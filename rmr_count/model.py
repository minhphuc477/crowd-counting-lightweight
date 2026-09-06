"""Backward-compatibility shim re-exporting from rmr_v2.model and rmr_core."""
from __future__ import annotations

from rmr_core.heads import _FINE_HEAD_BIAS_INIT, _M0_INIT
from rmr_v2.model import (
    RMRConfig,
    RMRCount,
    RegionalEvidenceHead,
    ScaleMatchedRegionalEvidenceHead,
    count_parameters,
)

__all__ = [
    "RMRConfig",
    "RMRCount",
    "RegionalEvidenceHead",
    "ScaleMatchedRegionalEvidenceHead",
    "count_parameters",
    "_M0_INIT",
    "_FINE_HEAD_BIAS_INIT",
]
