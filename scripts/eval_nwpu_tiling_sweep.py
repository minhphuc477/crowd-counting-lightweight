#!/usr/bin/env python3
"""Sliding-Window Tiling & Boundary Blending Optimization Sweep for NWPU.

Sweeps across:
1. Tile Sizes: 512, 768, 1024
2. Context Halos: 0 (hard cut), 64, 128
3. Blend Modes: 'core' (discontinuity prone) vs 'hann' (smooth 2D cosine window)
4. Full Image Downscaling Baseline (cap at 2048)

Measures:
- Overall Validation MAE / MSE / NAE
- Peak GPU VRAM Memory (MB)
- Inference Latency (ms per image)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rmr_core.data import CrowdManifestDataset
from rmr_core.tiling import predict_tiled
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def evaluate_configuration(
    model: torch.nn.Module,
    items: list[dict],
    device: torch.device,
    eff_stride: int,
    norm_mode: str,
    tile_size: int | None,
    halo: int,
    blend_mode: str,
    max_eval_samples: int = 50,
) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    eval_items = items[:max_eval_samples] if max_eval_samples > 0 else items
    aes, sqs, naes, times = [], [], [], []

    for it in eval_items:
        img_path = Path(_REPO_ROOT) / it["image"]
        if not img_path.is_file():
            img_path = Path(_REPO_ROOT) / "data" / it["image"]
        if not img_path.is_file():
            continue

        with Image.open(img_path) as pil_img:
            w0, h0 = pil_img.size
            img_t = TF.to_tensor(pil_img.convert("RGB"))

        pts = it.get("points", [])
        gt_count = float(len(pts))

        t0 = time.time()
        with torch.no_grad():
            if tile_size is None:
                # Direct downscaled forward (cap at 2048)
                max_dim = max(w0, h0)
                if max_dim > 2048:
                    sc = 2048.0 / float(max_dim)
                    pil_capped = pil_img.resize((int(round(w0 * sc)), int(round(h0 * sc))), Image.Resampling.BILINEAR)
                    img_t = TF.to_tensor(pil_capped.convert("RGB"))
                img_norm = normalize_image(img_t, mode=norm_mode).unsqueeze(0).to(device)
                out = model(img_norm)
                pred_count = float(out["y"].sum().item())
            else:
                pred_map = predict_tiled(
                    model=model,
                    image=img_t.to(device),
                    output_stride=eff_stride,
                    tile_size=tile_size,
                    halo=halo,
                    blend_mode=blend_mode,
                )
                pred_count = float(pred_map.sum().item())

        elapsed_ms = (time.time() - t0) * 1000.0
        times.append(elapsed_ms)

        err = abs(pred_count - gt_count)
        aes.append(err)
        sqs.append(err ** 2)
        if gt_count > 0:
            naes.append(err / gt_count)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    return {
        "tile_size": tile_size if tile_size is not None else "Downscale_2048",
        "halo": halo,
        "blend_mode": blend_mode,
        "mae": round(float(np.mean(aes)), 2) if aes else 0.0,
        "rmse": round(float(np.sqrt(np.mean(sqs))), 2) if sqs else 0.0,
        "nae": round(float(np.mean(naes)), 3) if naes else 0.0,
        "latency_ms_per_img": round(float(np.mean(times)), 1) if times else 0.0,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "evaluated_samples": len(aes),
    }


def main():
    parser = argparse.ArgumentParser(description="NWPU-Crowd Tiling Heuristics Benchmark Sweep")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--manifest", default="data/nwpu_val.jsonl", help="Validation manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, default=100, help="Number of validation samples to evaluate")
    parser.add_argument("--output-json", default="results/nwpu_tiling_sweep_results.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[TILING SWEEP] Loading checkpoint: {args.checkpoint}")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    m_cfg = cfg.get("model", {})
    d_cfg = cfg.get("data", {})
    eff_stride = 2 if bool(m_cfg.get("subpixel_stride2", False)) else int(m_cfg.get("output_stride", 4))
    backbone_name = m_cfg.get("backbone_name", m_cfg.get("backbone"))
    norm_mode = CrowdManifestDataset.resolve_norm(backbone_name, d_cfg.get("norm_mode", "auto"))

    with open(args.manifest, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    print(f"[TILING SWEEP] Testing on {min(args.max_samples, len(items))} samples across 5 configurations...")

    configs_to_test = [
        {"tile_size": None, "halo": 0, "blend_mode": "none"},               # Downscaled baseline
        {"tile_size": 512,  "halo": 64, "blend_mode": "hann"},               # Small tile + hann
        {"tile_size": 768,  "halo": 0,  "blend_mode": "core"},               # Medium tile + hard seam
        {"tile_size": 768,  "halo": 128, "blend_mode": "hann"},              # Proposed Canonical
        {"tile_size": 1024, "halo": 128, "blend_mode": "hann"},              # Large tile + hann
    ]

    all_results = []
    print("\n" + "=" * 80)
    print(f"{'Config':<18} | {'Halo':<5} | {'Blend':<6} | {'MAE':<7} | {'RMSE':<7} | {'Latency':<9} | {'Peak VRAM':<9}")
    print("=" * 80)

    for c in configs_to_test:
        res = evaluate_configuration(
            model=model,
            items=items,
            device=device,
            eff_stride=eff_stride,
            norm_mode=norm_mode,
            tile_size=c["tile_size"],
            halo=c["halo"],
            blend_mode=c["blend_mode"],
            max_eval_samples=args.max_samples,
        )
        all_results.append(res)
        tag = f"Tile_{c['tile_size']}" if c['tile_size'] else "Downscale_2048"
        print(f"{tag:<18} | {c['halo']:<5} | {c['blend_mode']:<6} | {res['mae']:<7.2f} | {res['rmse']:<7.2f} | {res['latency_ms_per_img']:<7.1f}ms | {res['peak_vram_mb']:<6.1f}MB")

    print("=" * 80)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[SAVED] Detailed sweep metrics exported to: {out_path}")


if __name__ == "__main__":
    main()
