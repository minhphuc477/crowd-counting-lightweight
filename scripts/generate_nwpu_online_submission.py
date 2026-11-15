#!/usr/bin/env python3
"""Generate Online Benchmark Submission Predictions for NWPU-Crowd Test Set (1,500 Images).

Runs direct and tiled inference on the unannotated test split (`data/nwpu_test.jsonl`),
preventing GPU Out-of-Memory (OOM) on ultra-high-resolution images via boundary-aware tiling.
Outputs the prediction files formatted for submission to the official NWPU Benchmark Server:
http://www.crowdbenchmark.com/
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from rmr_core.data import CrowdManifestDataset
from rmr_core.tiling import predict_tiled
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def main():
    parser = argparse.ArgumentParser(description="Generate NWPU Test Online Submission")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--manifest", default="data/nwpu_test.jsonl", help="NWPU test manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default="results/nwpu_submission", help="Output directory")
    parser.add_argument("--use-tiling", action="store_true", default=True, help="Use tiled inference for ultra-high-res images")
    parser.add_argument("--tile-size", type=int, default=768, help="Tile size for inference")
    parser.add_argument("--halo", type=int, default=128, help="Context halo size in pixels")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[NWPU SUBMISSION] Loading checkpoint from: {args.checkpoint}")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    m_cfg = cfg.get("model", {})
    d_cfg = cfg.get("data", {})
    eff_stride = int(
        getattr(
            getattr(model, "cfg", None),
            "output_stride",
            2 if (bool(m_cfg.get("subpixel_stride2", False)) or bool(m_cfg.get("subpixel_dm", False))) else int(m_cfg.get("output_stride", 4)),
        )
    )
    backbone_name = m_cfg.get("backbone_name", m_cfg.get("backbone"))
    norm_mode = CrowdManifestDataset.resolve_norm(backbone_name, d_cfg.get("norm_mode", "auto"))

    manifest_p = Path(args.manifest)
    with open(manifest_p, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"[NWPU SUBMISSION] Generating predictions for {len(items)} test images...")
    predictions = []

    t0 = time.time()
    for idx, it in enumerate(items):
        img_id = str(it.get("id", f"{3610 + idx}"))
        img_rel = it.get("image", f"data/nwpu/images/{img_id}.jpg")
        img_path = Path(_REPO_ROOT) / img_rel
        if not img_path.is_file():
            img_path = Path(_REPO_ROOT) / "data" / img_rel

        if not img_path.is_file():
            raise FileNotFoundError(f"Test image not found on disk: {img_path}")

        with Image.open(img_path) as pil_img:
            w0, h0 = pil_img.size
            img_t = TF.to_tensor(pil_img.convert("RGB"))

        max_dim = max(w0, h0)
        # Use tiled inference if image is large (> 1200px) and tiling is enabled
        if args.use_tiling and max_dim > 1200:
            pred_map = predict_tiled(
                model=model,
                image=img_t.to(device),
                output_stride=eff_stride,
                tile_size=args.tile_size,
                halo=args.halo,
                blend_mode="hann",
            )
            count = float(pred_map.sum().item())
        else:
            img_norm = normalize_image(img_t, mode=norm_mode).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_norm)
                y_pred = out["y"]
                count = float(y_pred.sum().item())

        predictions.append({"id": img_id, "pred_count": round(count, 2)})

        if (idx + 1) % 100 == 0 or (idx + 1) == len(items):
            print(f"  Processed [{idx+1}/{len(items)}] | Latest count: {count:.2f}")

    elapsed = time.time() - t0
    print(f"[NWPU SUBMISSION] Completed {len(predictions)} images in {elapsed:.2f}s ({elapsed/len(predictions):.3f}s/img)")

    # 1. Output space-separated text format: <id> <count>
    txt_path = out_dir / "nwpu_test_pred.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(f"{p['id']} {p['pred_count']:.2f}\n")
    print(f"[SAVED] Official submission text file: {txt_path}")

    # 2. Output CSV format
    csv_path = out_dir / "nwpu_test_pred.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "pred_count"])
        writer.writeheader()
        writer.writerows(predictions)
    print(f"[SAVED] Submission CSV file: {csv_path}")


if __name__ == "__main__":
    main()
