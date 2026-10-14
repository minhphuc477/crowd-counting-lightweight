from __future__ import annotations

"""FPN Module Facade.

Re-exports all Feature Pyramid Network neck architectures to guarantee 100% backward
compatibility with historical imports:
    from rmr_core.necks.fpn import AdditiveFPNNeck, ASPPLiteFPNNeck, HDCLiteFPNNeck, BiFPNLiteNeck, AdditiveFusion

The implementations reside in their respective modular files:
    - additive.py: AdditiveFusion, AdditiveFPNNeck
    - aspp.py: ASPPLiteFPNNeck
    - hdc.py: HDCLiteFPNNeck
    - bifpn.py: BiFPNLiteNeck
    - rep_fpn.py: RepWeightedFPNNeck, RepDWBlock7x7
"""

from .additive import AdditiveFPNNeck, AdditiveFusion
from .aspp import ASPPLiteFPNNeck
from .bifpn import BiFPNLiteNeck
from .hdc import HDCLiteFPNNeck
from .rep_fpn import RepDWBlock7x7, RepWeightedFPNNeck

__all__ = [
    "ASPPLiteFPNNeck",
    "AdditiveFPNNeck",
    "AdditiveFusion",
    "BiFPNLiteNeck",
    "HDCLiteFPNNeck",
    "RepDWBlock7x7",
    "RepWeightedFPNNeck",
]
