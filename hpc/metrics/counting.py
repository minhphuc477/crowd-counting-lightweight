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
        raise ValueError(f"Prediction/target shapes differ: {preds.shape} vs {gts.shape}")
    return preds, gts


def compute_mae(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    preds, gts = _validate_pair(predictions, targets)
    return float(np.mean(np.abs(preds - gts))) if len(preds) else 0.0


def compute_rmse(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    preds, gts = _validate_pair(predictions, targets)
    return float(np.sqrt(np.mean((preds - gts) ** 2))) if len(preds) else 0.0


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


def compute_bias(predictions: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> float:
    preds, gts = _validate_pair(predictions, targets)
    return float(np.mean(preds - gts)) if len(preds) else 0.0


def evaluate_counting_metrics(predictions, targets) -> Dict[str, float]:
    return {
        "mae": compute_mae(predictions, targets),
        "rmse": compute_rmse(predictions, targets),
        "bias": compute_bias(predictions, targets),
        "nae": compute_nae(predictions, targets, ignore_zero=True),
    }
