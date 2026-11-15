#!/usr/bin/env python3
"""Evaluate Density Map Quality (PSNR & SSIM) on ShanghaiTech Part A & Part B.

Follows the canonical crowd counting evaluation protocols (CSRNet CVPR 2018,
MCNN CVPR 2016, SANet ECCV 2018, TEDNet CVPR 2019):
- Ground Truth generation:
    * Part A: Geometry-adaptive Gaussian filtering (k=3 nearest neighbors, beta=0.3).
    * Part B: Fixed Gaussian filtering (sigma=15).
- Model Prediction upsampling:
    * Bilinear interpolation to the exact original image resolution (H, W).
- Metrics:
    * PSNR = 10 * log10(MAX_GT^2 / MSE)
    * SSIM = Structural Similarity Index between GT and prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

# Add repo root to sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree
from torchvision.transforms import functional as TF

from rmr_core.metrics import compute_psnr, compute_ssim
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def generate_gaussian_gt(
    points: np.ndarray,
    h: int,
    w: int,
    dataset_type: str = "sha",
) -> np.ndarray:
    """Generate canonical ground-truth density map."""
    gt_map = np.zeros((h, w), dtype=np.float32)
    n_points = len(points)
    if n_points == 0:
        return gt_map

    pts = points.astype(np.float32)

    if dataset_type == "sha":
        # Geometry-adaptive Gaussian kernel for congested scenes (Part A)
        if n_points > 3:
            tree = KDTree(pts)
            distances, _ = tree.query(pts, k=4)
            sigmas = np.mean(distances[:, 1:4], axis=-1) * 0.3
            sigmas = np.clip(sigmas, 1.0, 15.0)
        else:
            sigmas = np.full(n_points, 4.0, dtype=np.float32)

        for (x, y), s in zip(pts, sigmas):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                rad = int(round(3 * s))
                x0, x1 = max(0, ix - rad), min(w, ix + rad + 1)
                y0, y1 = max(0, iy - rad), min(h, iy + rad + 1)
                grid_y, grid_x = np.ogrid[y0 - iy : y1 - iy, x0 - ix : x1 - ix]
                g = np.exp(-(grid_x**2 + grid_y**2) / (2 * s**2))
                g_sum = g.sum()
                if g_sum > 0:
                    gt_map[y0:y1, x0:x1] += g / g_sum
    else:
        # Fixed Gaussian kernel for sparser scenes (Part B, sigma=15)
        for x, y in pts:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                gt_map[iy, ix] += 1.0
        gt_map = gaussian_filter(gt_map, sigma=15.0, mode="constant")

    return gt_map


def evaluate_density_quality(
    manifest_path: Path,
    ckpt_path: Path,
    device: torch.device,
    dataset_type: str = "sha",
    stride: int = 2,
    norm_mode: str = "inception",
) -> dict:
    """Evaluate PSNR and SSIM across all images in a manifest."""
    print(f"\n[EVAL] Loading model from: {ckpt_path}")
    model, _, _, _ = load_model_from_ckpt(ckpt_path, device, use_ema=True)
    model.eval()

    with open(manifest_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"[EVAL] Evaluating {len(items)} images from {manifest_path.name} ({dataset_type.upper()})...")

    records = []
    psnr_list = []
    ssim_list = []
    abs_errors = []
    sq_errors = []

    t0 = time.time()
    for idx, it in enumerate(items):
        img_rel = it["image"]
        img_path = _REPO_ROOT / img_rel
        if not img_path.is_file():
            img_path = _REPO_ROOT / "data" / img_rel
        if not img_path.is_file():
            raise FileNotFoundError(f"Image not found: {img_rel}")

        with Image.open(img_path) as pil_img:
            w0, h0 = pil_img.size
            img_t = TF.to_tensor(pil_img.convert("RGB"))

        img_t_norm = normalize_image(img_t, mode=norm_mode).unsqueeze(0).to(device)

        pts = np.array(it.get("points", []), dtype=np.float32)
        gt_count = float(len(pts))

        # 1. Forward model
        with torch.no_grad():
            out = model(img_t_norm)
            y_pred = out["y"]  # (1, 1, gh, gw) or (1, gh, gw)
            if y_pred.ndim == 3:
                y_pred = y_pred.unsqueeze(1)

            pred_count = float(y_pred.sum().item())

            # 2. Rescale prediction to original image dimensions preserving mass
            pred_up = F.interpolate(
                y_pred,
                size=(h0, w0),
                mode="bilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()

            # Ensure mass conservation on the upsampled grid
            if pred_up.sum() > 0:
                pred_up = pred_up * (pred_count / pred_up.sum())

        # 3. Generate canonical Ground Truth density map
        gt_map = generate_gaussian_gt(pts, h0, w0, dataset_type=dataset_type)

        # 4. Compute PSNR & SSIM
        psnr_val = compute_psnr(pred_up, gt_map)
        ssim_val = compute_ssim(pred_up, gt_map)

        ae = abs(pred_count - gt_count)
        se = (pred_count - gt_count) ** 2

        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        abs_errors.append(ae)
        sq_errors.append(se)

        records.append({
            "id": it.get("id", f"IMG_{idx+1}"),
            "image": img_rel,
            "gt_count": gt_count,
            "pred_count": round(pred_count, 2),
            "abs_error": round(ae, 2),
            "psnr": round(psnr_val, 4),
            "ssim": round(ssim_val, 4),
            "height": h0,
            "width": w0,
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(items):
            print(f"  Processed [{idx+1}/{len(items)}] | Running PSNR: {np.mean(psnr_list):.2f} dB, SSIM: {np.mean(ssim_list):.4f}, MAE: {np.mean(abs_errors):.2f}")

    elapsed = time.time() - t0
    summary = {
        "dataset": dataset_type.upper(),
        "manifest": manifest_path.name,
        "num_samples": len(items),
        "mean_psnr_db": float(np.mean(psnr_list)),
        "std_psnr_db": float(np.std(psnr_list)),
        "mean_ssim": float(np.mean(ssim_list)),
        "std_ssim": float(np.std(ssim_list)),
        "MAE": float(np.mean(abs_errors)),
        "RMSE": float(np.sqrt(np.mean(sq_errors))),
        "elapsed_sec": round(elapsed, 2),
    }

    return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="Evaluate PSNR and SSIM on ShanghaiTech.")
    parser.add_argument("--dataset", choices=["sha", "shb", "both"], default="both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/density_map_quality")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    if args.dataset in ("sha", "both"):
        targets.append((
            "sha",
            _REPO_ROOT / "data" / "sha_a_test.jsonl",
            _REPO_ROOT / "runs" / "benchmarks" / "exp_sha_direct_pretrained_104k" / "best_model.pt",
        ))
    if args.dataset in ("shb", "both"):
        targets.append((
            "shb",
            _REPO_ROOT / "data" / "shb_test.jsonl",
            _REPO_ROOT / "runs" / "benchmarks" / "exp_shb_direct_pretrained_104k" / "best_model.pt",
        ))

    all_summaries = {}
    for d_type, man_path, ckpt_path in targets:
        res = evaluate_density_quality(man_path, ckpt_path, device, dataset_type=d_type)
        all_summaries[d_type] = res["summary"]

        csv_path = out_dir / f"predictions_{d_type}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(res["records"][0].keys()))
            writer.writeheader()
            writer.writerows(res["records"])
        print(f"[SAVED] Per-sample results saved to: {csv_path}")

    summary_path = out_dir / "density_quality_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[SAVED] Summary metrics saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("DENSITY MAP QUALITY AUDIT SUMMARY (PSNR & SSIM)")
    print("=" * 70)
    for d_type, s in all_summaries.items():
        print(f"Dataset: ShanghaiTech {s['dataset']}")
        print(f"  - PSNR: {s['mean_psnr_db']:.2f} ± {s['std_psnr_db']:.2f} dB")
        print(f"  - SSIM: {s['mean_ssim']:.4f} ± {s['std_ssim']:.4f}")
        print(f"  - MAE : {s['MAE']:.2f}")
        print(f"  - RMSE: {s['RMSE']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
