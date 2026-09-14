from __future__ import annotations

"""Typed Data Transfer Objects and Backward-Compatible Mapping Interfaces."""

from typing import Any
import torch

from rmr_core.operators import RegionSet


class RMRModelOutput(dict):
    """Encapsulated output of RMR model forward pass.

    Inherits from built-in `dict` to guarantee 100% interoperability with:
    - PyTorch ONNX export & tracing
    - PyTorch DataLoader collate functions
    - Legacy dictionary subscripting (`out["y"]`, `"y" in out`)
    - Typed dot-notation attribute access (`out.y`, `out.y0`)
    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'RMRModelOutput' object has no attribute '{key}'") from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'RMRModelOutput' object has no attribute '{key}'") from None

    # Attribute type annotations for IDE autocompletion & static analysis
    y: torch.Tensor
    y0: torch.Tensor
    z0: torch.Tensor
    regions: RegionSet
    b_region: torch.Tensor
    b_solver: torch.Tensor
    region_rate: torch.Tensor
    region_dispersion: torch.Tensor
    region_log_dispersion: torch.Tensor
    region_weight: torch.Tensor
    solver_region_weight: torch.Tensor
    region_precision: torch.Tensor
    region_rate_variance: torch.Tensor
    region_count_variance: torch.Tensor
    solver_count_variance: torch.Tensor
    iterates: list[torch.Tensor]
    residual_fields: list[torch.Tensor]
    energy_trace: list[dict[str, torch.Tensor]]
    uniform_reliability: bool
    solver_strength: float
    scale_weights: torch.Tensor | None
    hurdle_logit: torch.Tensor | None
    fg_logit: torch.Tensor | None


class MappingMixin:
    """Enables dictionary access on custom objects."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        return hasattr(self, str(key))

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
