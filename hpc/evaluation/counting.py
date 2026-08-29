"""Evaluation utilities for crowd counting models."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..metrics.counting import evaluate_counting_metrics
from ..metrics.subgroup import evaluate_subgroup_diagnostics


@torch.no_grad()
def evaluate_counting(model: nn.Module, dataset, device: torch.device) -> dict:
    """Single-scale full-image evaluation on a labeled dataset.

    Returns overall metrics (MAE, RMSE, NAE, Bias) and subgroup diagnostics.
    """
    model.eval()
    predictions, ground_truths = [], []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not bool(sample.get("has_gt", True)):
            raise ValueError("Evaluation requires a labeled validation/test split")
        image = sample["image"].unsqueeze(0).to(device)
        prediction, _ = model.predict(image, pad_multiple=None)
        predictions.append(float(prediction.item()))
        ground_truths.append(float(sample["gt_count"]))
    result = evaluate_counting_metrics(predictions, ground_truths)
    result.update(evaluate_subgroup_diagnostics(predictions, ground_truths))
    return result
