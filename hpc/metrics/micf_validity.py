from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class ValidityRow:
    vr_tau: float
    positive_variation: float
    negative_variation: float
    nvr: float
    violating_cells: int
    total_cells: int


def recovered_measure_validity(
    y_pred: torch.Tensor,
    *,
    tau: float = 1e-6,
    eps: float = 1e-12,
) -> ValidityRow:
    """
    Validity diagnostics for a recovered signed counting measure.

    Parameters
    ----------
    y_pred:
        Recovered measure Y = Delta_xy C.
        Any shape is allowed; all cells are pooled.
    tau:
        Numerical tolerance in count units.
    eps:
        Stabilizer for NVR denominator.

    Returns
    -------
    ValidityRow
        Per-sample quantities.
    """
    if not isinstance(y_pred, torch.Tensor):
        raise TypeError(
            "y_pred must be a torch.Tensor"
        )

    y = (
        y_pred.detach()
        .to(dtype=torch.float64)
    )

    if not torch.isfinite(y).all():
        raise ValueError(
            "Recovered measure contains NaN/Inf"
        )

    positive = torch.clamp(
        y,
        min=0.0,
    )

    negative = torch.clamp(
        -y,
        min=0.0,
    )

    positive_variation = float(
        positive.sum().item()
    )

    negative_variation = float(
        negative.sum().item()
    )

    violating_cells = int(
        (y < -float(tau))
        .sum()
        .item()
    )

    total_cells = int(
        y.numel()
    )

    vr_tau = float(
        violating_cells
        / max(total_cells, 1)
    )

    nvr = float(
        negative_variation
        / max(
            positive_variation,
            float(eps),
        )
    )

    return ValidityRow(
        vr_tau=vr_tau,
        positive_variation=positive_variation,
        negative_variation=negative_variation,
        nvr=nvr,
        violating_cells=violating_cells,
        total_cells=total_cells,
    )


def aggregate_validity(
    rows: Iterable[ValidityRow],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    rows = list(rows)

    if not rows:
        return {
            "vr_tau_micro": float("nan"),
            "vr_tau_macro": float("nan"),
            "nvr_micro": float("nan"),
            "nvr_macro": float("nan"),
            "positive_variation_total": 0.0,
            "negative_variation_total": 0.0,
            "violating_cells_total": 0,
            "cells_total": 0,
        }

    violating_cells_total = int(
        sum(
            r.violating_cells
            for r in rows
        )
    )

    cells_total = int(
        sum(
            r.total_cells
            for r in rows
        )
    )

    positive_total = float(
        sum(
            r.positive_variation
            for r in rows
        )
    )

    negative_total = float(
        sum(
            r.negative_variation
            for r in rows
        )
    )

    vr_tau_micro = float(
        violating_cells_total
        / max(cells_total, 1)
    )

    vr_tau_macro = float(
        np.mean(
            [r.vr_tau for r in rows]
        )
    )

    nvr_micro = float(
        negative_total
        / max(
            positive_total,
            float(eps),
        )
    )

    nvr_macro = float(
        np.mean(
            [r.nvr for r in rows]
        )
    )

    return {
        "vr_tau_micro": vr_tau_micro,
        "vr_tau_macro": vr_tau_macro,
        "nvr_micro": nvr_micro,
        "nvr_macro": nvr_macro,
        "positive_variation_total": positive_total,
        "negative_variation_total": negative_total,
        "violating_cells_total": violating_cells_total,
        "cells_total": cells_total,
    }
