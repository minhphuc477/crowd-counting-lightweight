"""Evaluation pipelines and metric generation for RMR Core.

Provides:
- evaluate_dataset: Unified validation loop over PyTorch DataLoaders.
- save_evaluation_artifacts: Persists summary.json and predictions.csv.
- predict_tiled, predict_multiscale_tta: Re-exported from rmr_core.tiling.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import (
    bootstrap_ci,
    game_physical_image,
    game_single,
    summarize_predictions,
)
from .tiling import predict_multiscale_tta, predict_tiled

__all__ = [
    "predict_tiled",
    "predict_multiscale_tta",
    "evaluate_dataset",
    "save_evaluation_artifacts",
]



@torch.no_grad()
def evaluate_dataset(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_stride: int = 4,
    run_tiling: bool = True,
    tile_size: int = 512,
    practical_halo: int = 64,
    forward_kwargs: dict[str, Any] | None = None,
    extra_sample_callback: Callable[[dict, dict, torch.Tensor, dict], dict] | None = None,
    enforce_gt_consistency: bool = False,
    density_bins: tuple[float, float] = (100.0, 500.0),
    use_tta: bool = False,
) -> tuple[list[dict], dict[str, Any]]:
    """Canonical unified evaluation loop over a dataset loader.

    Args:
        model: Evaluated PyTorch model.
        loader: DataLoader returning batches of sample dicts (collate_eval).
        device: Target execution device.
        output_stride: Stride of the predicted density representation (default 4).
        run_tiling: Whether to compute halo=0 and halo=practical_halo tiled inference.
        tile_size: Pixel size of square tiles.
        practical_halo: Context halo size in pixels.
        forward_kwargs: Extra keyword arguments passed to model forward.
        extra_sample_callback: Optional hook (sample, out, y, row) -> dict to record extra sample metrics.
        enforce_gt_consistency: Whether to enforce sum(target_y) == valid_raw_points.
        density_bins: Tuple of (sparse_threshold, dense_threshold) for density stratification.
        use_tta: Whether to apply Horizontal Flip Test-Time Augmentation (TTA).

    Returns:
        rows: List of per-sample prediction records.
        summary: Dictionary of aggregated dataset metrics.
    """
    forward_kwargs = forward_kwargs or {}
    model_stride = int(getattr(getattr(model, "cfg", None), "output_stride", output_stride))
    eff_output_stride = model_stride if model_stride in (1, 2, 4, 8) else output_stride
    model.eval()

    rows: list[dict] = []
    sample_index = 0

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
            target = sample["target_y"].to(device, non_blocking=True)
            sample["target_device"] = target

            out = model(image, **forward_kwargs)
            if use_tta:
                image_flip = torch.flip(image, dims=[-1])
                out_flip = model(image_flip, **forward_kwargs)
                y_flip = torch.flip(out_flip["y"], dims=[-1])
                y_tta = 0.5 * (out["y"] + y_flip)
                out["y"] = y_tta
                y = y_tta[0]
            else:
                y = out["y"][0]

            gt = float(target.sum().item())
            pred = float(y.sum().item())

            sample_id = sample.get("id", str(sample_index))

            # GT consistency invariant: rasterized count sum must exactly match valid raw points
            if enforce_gt_consistency and "points" in sample and "height" in sample and "width" in sample:
                pts = sample["points"]
                h_img = sample["height"]
                w_img = sample["width"]
                if pts.numel() > 0:
                    px = pts[:, 0]
                    py = pts[:, 1]
                    valid_mask = (px >= 0) & (px < w_img) & (py >= 0) & (py < h_img)
                    n_valid = int(valid_mask.sum().item())
                else:
                    n_valid = 0
                if abs(gt - n_valid) > 1e-4:
                    raise ValueError(
                        f"GT consistency invariant violated for sample '{sample_id}': "
                        f"target_y.sum()={gt} != valid_points={n_valid} (image_size={w_img}x{h_img})"
                    )

            row: dict[str, Any] = {
                "id": sample_id,
                "index": sample_index,
                "gt": gt,
                "pred": pred,
                "abs_err": abs(pred - gt),
                "sq_err": (pred - gt) ** 2,
            }

            # Tiled prediction comparison
            if run_tiling:
                if use_tta:
                    y_t0 = predict_multiscale_tta(
                        model, sample["image"].to(device),
                        output_stride=eff_output_stride, tile_size=tile_size, halo=0,
                        use_hflip=True,
                        forward_kwargs=forward_kwargs,
                    )
                    y_th = predict_multiscale_tta(
                        model, sample["image"].to(device),
                        output_stride=eff_output_stride, tile_size=tile_size, halo=practical_halo,
                        use_hflip=True,
                        forward_kwargs=forward_kwargs,
                    )
                else:
                    y_t0 = predict_tiled(
                        model, sample["image"].to(device),
                        output_stride=eff_output_stride, tile_size=tile_size, halo=0,
                        forward_kwargs=forward_kwargs,
                    )
                    y_th = predict_tiled(
                        model, sample["image"].to(device),
                        output_stride=eff_output_stride, tile_size=tile_size, halo=practical_halo,
                        forward_kwargs=forward_kwargs,
                    )
                pred_t0 = float(y_t0.sum().item())
                pred_th = float(y_th.sum().item())
                row["pred_tiled_h0"] = pred_t0
                row["pred_tiled_practical"] = pred_th
                row["direct_tiled_h0_abs"] = abs(pred - pred_t0)
                row["direct_tiled_practical_abs"] = abs(pred - pred_th)
                row["direct_tiled_h0_norm"] = abs(pred - pred_t0) / max(gt, 1.0)
                row["direct_tiled_practical_norm"] = abs(pred - pred_th) / max(gt, 1.0)

            # Physical-support GAME when raw points and original dimensions are available
            if "points" in sample and "height" in sample and "width" in sample:
                game_stride = round(float(sample["height"]) / float(y.shape[-2])) if (y.ndim >= 2 and y.shape[-2] > 0) else eff_output_stride
                game_dict = game_physical_image(
                    y,
                    sample["points"],
                    image_h=sample["height"],
                    image_w=sample["width"],
                    stride=game_stride if game_stride in (1, 2, 4, 8) else eff_output_stride,
                    levels=(0, 1, 2, 3),
                )
                for level in range(4):
                    row[f"GAME{level}"] = game_dict[level]
            else:
                for level in range(4):
                    row[f"GAME{level}"] = game_single(y, target, level)

            if extra_sample_callback is not None:
                extra = extra_sample_callback(sample, out, y, row)
                if extra:
                    row.update(extra)

            rows.append(row)
            sample_index += 1

    summary = summarize_predictions(rows)

    # Standard lowercase aliases for compatibility
    summary["mae"] = summary["MAE"]
    summary["rmse"] = summary["RMSE"]
    summary["nae"] = summary["NAE"]
    summary["bias"] = summary["Bias"]
    summary["num_samples"] = len(rows)

    # Bootstrap 95% confidence intervals
    aes = [r["abs_err"] for r in rows]
    sqs = [r["sq_err"] for r in rows]
    mae_lo, mae_hi = bootstrap_ci(aes, statistic=np.mean)
    rmse_lo, rmse_hi = bootstrap_ci(sqs, statistic=lambda v: float(np.sqrt(np.mean(v))))
    summary["mae_ci95"] = [mae_lo, mae_hi]
    summary["rmse_ci95"] = [rmse_lo, rmse_hi]

    # Density stratification with parameterized bins
    gts = np.array([r["gt"] for r in rows])
    aes_arr = np.array(aes)

    lo, hi = density_bins
    sparse_mask = gts <= lo
    mod_mask = (gts > lo) & (gts <= hi)
    dense_mask = gts > hi

    summary["mae_sparse"] = float(np.mean(aes_arr[sparse_mask])) if np.any(sparse_mask) else 0.0
    summary["mae_moderate"] = float(np.mean(aes_arr[mod_mask])) if np.any(mod_mask) else 0.0
    summary["mae_dense"] = float(np.mean(aes_arr[dense_mask])) if np.any(dense_mask) else 0.0

    summary["n_sparse"] = int(np.sum(sparse_mask))
    summary["n_moderate"] = int(np.sum(mod_mask))
    summary["n_dense"] = int(np.sum(dense_mask))

    if run_tiling and rows and "direct_tiled_practical_abs" in rows[0]:
        tiled_diffs = [r["direct_tiled_practical_abs"] for r in rows]
        tiled_norms = [r["direct_tiled_practical_norm"] for r in rows]
        tiled_h0_diffs = [r["direct_tiled_h0_abs"] for r in rows]
        tiled_h0_norms = [r["direct_tiled_h0_norm"] for r in rows]

        summary["direct_tiled_discrepancy_mean"] = float(np.mean(tiled_diffs))
        summary["direct_tiled_discrepancy_max"] = float(np.max(tiled_diffs))
        summary["direct_tiled_normalized_discrepancy_mean"] = float(np.mean(tiled_norms))
        summary["direct_tiled_h0_discrepancy_mean"] = float(np.mean(tiled_h0_diffs))
        summary["direct_tiled_h0_normalized_discrepancy_mean"] = float(np.mean(tiled_h0_norms))

        disc_lo, disc_hi = bootstrap_ci(tiled_diffs, statistic=np.mean)
        norm_lo, norm_hi = bootstrap_ci(tiled_norms, statistic=np.mean)
        summary["direct_tiled_discrepancy_ci95"] = [disc_lo, disc_hi]
        summary["direct_tiled_normalized_discrepancy_ci95"] = [norm_lo, norm_hi]

        # Tiled prediction absolute accuracy
        if "pred_tiled_h0" in rows[0]:
            ae_h0 = [abs(r["pred_tiled_h0"] - r["gt"]) for r in rows]
            summary["mae_tiled_h0"] = float(np.mean(ae_h0))
            summary["rmse_tiled_h0"] = float(np.sqrt(np.mean([err ** 2 for err in ae_h0])))
        if "pred_tiled_practical" in rows[0]:
            ae_th = [abs(r["pred_tiled_practical"] - r["gt"]) for r in rows]
            summary["mae_tiled_practical"] = float(np.mean(ae_th))
            summary["rmse_tiled_practical"] = float(np.sqrt(np.mean([err ** 2 for err in ae_th])))

        # Standard canonical aliases
        summary["mean_abs_prediction_discrepancy"] = summary["direct_tiled_discrepancy_mean"]
        summary["mean_normalized_prediction_discrepancy"] = summary["direct_tiled_normalized_discrepancy_mean"]

    return rows, summary


def save_evaluation_artifacts(
    out_dir: Path | str,
    rows: list[dict],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    """Save canonical summary.json and predictions.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pred_path = out_dir / "predictions.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return summary_path, pred_path
