from __future__ import annotations

from typing import Dict, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _as_flat(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _validate_pair(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    preds = _as_flat(predictions)
    gts = _as_flat(targets)

    if preds.shape != gts.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: "
            f"{preds.shape} vs {gts.shape}"
        )

    if not np.isfinite(preds).all():
        raise ValueError(
            "Predictions contain non-finite values (NaN or Inf)"
        )

    if not np.isfinite(gts).all():
        raise ValueError(
            "Ground-truth targets contain non-finite values (NaN or Inf)"
        )

    return preds, gts


def compute_mae(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    return float(
        np.mean(
            np.abs(preds - gts)
        )
    )


def compute_rmse(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    err = preds - gts

    return float(
        np.sqrt(
            np.mean(err * err)
        )
    )


def compute_nae(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    """
    Canonical crowd-counting NAE.

    Matches the NWPU-Crowd treatment:
    samples with gt == 0 are excluded.

    NAE =
        mean_{i: gt_i > 0}
        |pred_i - gt_i| / gt_i

    Returns NaN if there is no positive-ground-truth sample.
    """
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    mask = gts > 0.0

    if not np.any(mask):
        return float("nan")

    return float(
        np.mean(
            np.abs(
                preds[mask] - gts[mask]
            )
            / gts[mask]
        )
    )


def benchmark_count_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    """
    Paper-facing crowd-counting benchmark summary.

    Keep this function intentionally small.
    """
    preds, gts = _validate_pair(predictions, targets)

    return {
        "mae": compute_mae(preds, gts),
        "rmse": compute_rmse(preds, gts),
        "nae": compute_nae(preds, gts),
        "num_images": int(preds.size),
    }


# ------------------------------------------------------------------
# Diagnostics below are NOT part of the canonical benchmark summary.
# Keep them only if other code depends on them.
# ------------------------------------------------------------------

def compute_bias(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    return float(
        np.mean(preds - gts)
    )


def compute_sre(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> float:
    """
    Diagnostic only.

    SRE = sqrt(
        mean_{i: gt_i > 0}
        (pred_i - gt_i)^2 / gt_i
    )

    Zero-GT samples are excluded for the same reason as NAE.
    """
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return float("nan")

    mask = gts > 0.0

    if not np.any(mask):
        return float("nan")

    err = preds[mask] - gts[mask]

    return float(
        np.sqrt(
            np.mean(
                (err * err)
                / gts[mask]
            )
        )
    )


def diagnostic_count_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    preds, gts = _validate_pair(predictions, targets)

    if preds.size == 0:
        return {
            "signed_bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "p95_ae": float("nan"),
            "max_ae": float("nan"),
            "sre": float("nan"),
        }

    err = preds - gts
    ae = np.abs(err)

    return {
        "signed_bias": float(np.mean(err)),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(np.max(ae)),
        "sre": compute_sre(preds, gts),
    }


def count_metric_summary(
    predictions: ArrayLike,
    targets: ArrayLike,
    eps: float | None = None,
) -> Dict[str, float]:
    """
    Backward-compatible extended summary.

    New evaluator code should call benchmark_count_summary().
    `eps` is accepted only so old callers do not break.
    """
    del eps

    out = benchmark_count_summary(
        predictions,
        targets,
    )
    out.update(
        diagnostic_count_summary(
            predictions,
            targets,
        )
    )
    return out


def evaluate_counting_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
) -> Dict[str, float]:
    """
    Backward-compatible entry point.
    """
    out = benchmark_count_summary(
        predictions,
        targets,
    )
    out["bias"] = compute_bias(
        predictions,
        targets,
    )
    return out
