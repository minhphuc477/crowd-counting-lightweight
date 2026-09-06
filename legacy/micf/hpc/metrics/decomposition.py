from __future__ import annotations

from typing import Union

import numpy as np
import torch


ArrayLike = Union[
    np.ndarray,
    torch.Tensor,
]


def _flat(
    x: ArrayLike,
) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = (
            x.detach()
            .cpu()
            .numpy()
        )

    arr = np.asarray(
        x,
        dtype=np.float64,
    ).reshape(-1)

    if not np.isfinite(arr).all():
        raise ValueError(
            "Non-finite values in decomposition metric input"
        )

    return arr


def direct_tiled_discrepancy(
    direct_predictions: ArrayLike,
    tiled_predictions: ArrayLike,
    ground_truth: ArrayLike,
) -> dict[str, float]:
    d = _flat(direct_predictions)
    t = _flat(tiled_predictions)
    g = _flat(ground_truth)

    if not (
        d.shape == t.shape == g.shape
    ):
        raise ValueError(
            f"Shape mismatch: "
            f"direct={d.shape}, "
            f"tiled={t.shape}, "
            f"gt={g.shape}"
        )

    if d.size == 0:
        return {
            "mean_abs_prediction_discrepancy": float("nan"),
            "mean_normalized_prediction_discrepancy": float("nan"),
        }

    delta = np.abs(d - t)

    return {
        "mean_abs_prediction_discrepancy":
            float(
                np.mean(delta)
            ),
        "mean_normalized_prediction_discrepancy":
            float(
                np.mean(
                    delta
                    / np.maximum(g, 1.0)
                )
            ),
    }
