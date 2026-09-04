from typing import Dict, Union

import numpy as np
import torch


def _as_flat(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def _validate_pair(predictions, targets):
    preds, gts = _as_flat(predictions), _as_flat(targets)
    if preds.shape != gts.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {gts.shape}")
    if not np.isfinite(preds).all():
        raise ValueError("Predictions contain non-finite values (NaN or Inf)")
    if not np.isfinite(gts).all():
        raise ValueError("Ground-truth targets contain non-finite values (NaN or Inf)")
    return preds, gts


def compute_mae(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    preds, gts = _validate_pair(predictions, targets)
    return float(np.mean(np.abs(preds - gts))) if len(preds) else 0.0


def compute_rmse(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    preds, gts = _validate_pair(predictions, targets)
    return float(np.sqrt(np.mean((preds - gts) ** 2))) if len(preds) else 0.0


def compute_bias(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    """Signed bias: mean(pred - gt)."""
    preds, gts = _validate_pair(predictions, targets)
    return float(np.mean(preds - gts)) if len(preds) else 0.0


def compute_nae(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    ignore_zero: bool = True,
) -> float:
    """Normalized absolute error.

    NWPU official evaluation uses |pred-gt|/gt only for gt>0; empty samples are
    excluded from NAE. Set ignore_zero=False only for a custom diagnostic.
    """
    preds, gts = _validate_pair(predictions, targets)
    if not len(preds):
        return 0.0
    if ignore_zero:
        mask = gts > 0
        if not np.any(mask):
            return 0.0
        return float(np.mean(np.abs(preds[mask] - gts[mask]) / gts[mask]))
    return float(np.mean(np.abs(preds - gts) / np.maximum(gts, 1.0)))


def compute_sre(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12,
) -> float:
    """Squared Relative Error: sqrt(mean((pred - gt)^2 / max(gt, eps)))."""
    preds, gts = _validate_pair(predictions, targets)
    if not len(preds):
        return 0.0
    denom = np.maximum(gts, eps)
    return float(np.sqrt(np.mean(((preds - gts) ** 2) / denom)))


def count_metric_summary(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Authoritative summary of standard crowd counting metrics."""
    preds = np.asarray(list(predictions), dtype=np.float64)
    gts = np.asarray(list(targets), dtype=np.float64)

    if preds.shape != gts.shape:
        raise ValueError(f"Prediction/target shape mismatch: {preds.shape} vs {gts.shape}")

    if preds.size == 0:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "nae": float("nan"),
            "sre": float("nan"),
            "signed_bias": float("nan"),
            "median_ae": float("nan"),
            "p90_ae": float("nan"),
            "p95_ae": float("nan"),
            "max_ae": float("nan"),
        }

    err = preds - gts
    ae = np.abs(err)
    denom = np.maximum(gts, eps)

    return {
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "nae": float(np.mean(ae / denom)),
        "sre": float(np.sqrt(np.mean((err * err) / denom))),
        "signed_bias": float(np.mean(err)),
        "median_ae": float(np.median(ae)),
        "p90_ae": float(np.percentile(ae, 90)),
        "p95_ae": float(np.percentile(ae, 95)),
        "max_ae": float(np.max(ae)),
    }


def evaluate_counting_metrics(predictions, targets) -> Dict[str, float]:
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "bias": compute_bias(predictions, targets),
        "nae": compute_nae(predictions, targets, ignore_zero=True),
        "sre": compute_sre(predictions, targets),
    }
