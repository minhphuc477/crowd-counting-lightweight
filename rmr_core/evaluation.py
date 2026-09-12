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


def _aligned_floor(val: int, s: int) -> int:
    return (val // s) * s


def _aligned_ceil(val: int, s: int) -> int:
    return math.ceil(val / s) * s


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    tile_size: int = 512,
    halo: int = 0,
    forward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Core/halo tiled prediction assembled without double-counting.

    Core boundaries are aligned to output stride except the final image boundary.
    Halo affects context only; only the core prediction is written to the output.
    """
    forward_kwargs = forward_kwargs or {}
    was_training = model.training
    model.eval()
    try:
        _, h, w = image.shape
        s = output_stride
        tile_size = max(s, _aligned_floor(tile_size, s))
        halo = max(0, _aligned_floor(halo, s))
        gh, gw = math.ceil(h / s), math.ceil(w / s)
        canvas = image.new_zeros((1, gh, gw))

        ys = list(range(0, h, tile_size))
        xs = list(range(0, w, tile_size))
        for y0 in ys:
            y1 = min(h, y0 + tile_size)
            for x0 in xs:
                x1 = min(w, x0 + tile_size)

                sy0 = max(0, _aligned_floor(y0 - halo, s))
                sx0 = max(0, _aligned_floor(x0 - halo, s))
                sy1 = min(h, _aligned_ceil(y1 + halo, s))
                sx1 = min(w, _aligned_ceil(x1 + halo, s))
                patch = image[:, sy0:sy1, sx0:sx1].unsqueeze(0)
                y_patch = model(patch, **forward_kwargs)["y"][0]

                gy0 = y0 // s
                gx0 = x0 // s
                gy1 = math.ceil(y1 / s)
                gx1 = math.ceil(x1 / s)
                ly0 = (y0 - sy0) // s
                lx0 = (x0 - sx0) // s
                hh = gy1 - gy0
                ww = gx1 - gx0
                canvas[:, gy0:gy1, gx0:gx1] = y_patch[:, ly0:ly0 + hh, lx0:lx0 + ww]
        return canvas
    finally:
        if was_training:
            model.train()


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

    Returns:
        rows: List of per-sample prediction records.
        summary: Dictionary of aggregated dataset metrics.
    """
    forward_kwargs = forward_kwargs or {}
    model.eval()

    rows: list[dict] = []
    sample_index = 0

    for batch_list in loader:
        for sample in batch_list:
            image = sample["image"].unsqueeze(0).to(device)
            target = sample["target_y"].to(device)

            out = model(image, **forward_kwargs)
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
                y_t0 = predict_tiled(
                    model, sample["image"].to(device),
                    output_stride=output_stride, tile_size=tile_size, halo=0,
                    forward_kwargs=forward_kwargs,
                )
                y_th = predict_tiled(
                    model, sample["image"].to(device),
                    output_stride=output_stride, tile_size=tile_size, halo=practical_halo,
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
                game_dict = game_physical_image(
                    y,
                    sample["points"],
                    image_h=sample["height"],
                    image_w=sample["width"],
                    stride=output_stride,
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
