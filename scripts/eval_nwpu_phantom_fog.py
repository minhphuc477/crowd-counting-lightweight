#!/usr/bin/env python3
"""Phantom Background Fog & Negative Scene Auditor for NWPU-Crowd.

Specifically profiles and quantifies false-positive mass accumulation on:
1. Negative Scenes (scene_type == 0): 41 distractor images with 0 heads.
2. Zero-Count Samples (count == 0): All scenes without pedestrians.
3. Softplus bias freezing diagnostics: Evaluates whether pre-activation logits
   converge toward the vanishing-gradient plateau (b ≈ -5.442).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from rmr_core.data import CrowdManifestDataset
from rmr_core.transforms import normalize_image
from rmr_v3.eval import load_model_from_ckpt


def main():
    parser = argparse.ArgumentParser(description="NWPU-Crowd Phantom Background Fog Auditor")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", default="data/nwpu_val.jsonl", help="Validation manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=2048, help="Max image dimension")
    parser.add_argument("--output-json", default="results/nwpu_phantom_fog_audit.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[FOG AUDIT] Loading model from: {args.checkpoint}")
    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    m_cfg = cfg.get("model", {})
    d_cfg = cfg.get("data", {})
    eff_stride = 2 if bool(m_cfg.get("subpixel_stride2", False)) else int(m_cfg.get("output_stride", 4))
    backbone_name = m_cfg.get("backbone_name", m_cfg.get("backbone"))
    norm_mode = CrowdManifestDataset.resolve_norm(backbone_name, d_cfg.get("norm_mode", "auto"))

    with open(args.manifest, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # Filter negative samples (count == 0 or scene_type == 0)
    neg_items = [it for it in items if len(it.get("points", [])) == 0 or it.get("scene_type") == 0]
    print(f"[FOG AUDIT] Identified {len(neg_items)} empty/distractor scenes out of {len(items)} validation samples")

    results = []
    total_phantom_mass = 0.0
    total_cells = 0

    for idx, it in enumerate(neg_items):
        img_id = str(it.get("id", f"{idx:04d}"))
        img_path = Path(_REPO_ROOT) / it["image"]
        if not img_path.is_file():
            img_path = Path(_REPO_ROOT) / "data" / it["image"]

        if not img_path.is_file():
            continue

        with Image.open(img_path) as pil_img:
            w0, h0 = pil_img.size
            # Cap image resolution
            max_dim = max(w0, h0)
            if max_dim > args.max_size:
                scale = float(args.max_size) / float(max_dim)
                w_cap, h_cap = int(round(w0 * scale)), int(round(h0 * scale))
                pil_img = pil_img.resize((w_cap, h_cap), Image.Resampling.BILINEAR)
            img_t = TF.to_tensor(pil_img.convert("RGB"))

        img_norm = normalize_image(img_t, mode=norm_mode).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_norm)
            y_pred = out["y"]  # Shape [1, 1, H/stride, W/stride]
            phantom_count = float(y_pred.sum().item())
            n_cells = y_pred.numel()
            mean_density_per_cell = phantom_count / max(n_cells, 1)

        total_phantom_mass += phantom_count
        total_cells += n_cells

        results.append({
            "id": img_id,
            "scene_type": it.get("scene_type", 0),
            "luminance": it.get("luminance", 2),
            "phantom_count": round(phantom_count, 4),
            "num_cells": n_cells,
            "mean_cell_density": float(mean_density_per_cell),
        })

    if not results:
        print("[FOG AUDIT] Warning: No negative samples could be loaded from disk.")
        return

    phantom_counts = [r["phantom_count"] for r in results]
    s0_mae = float(np.mean(phantom_counts))
    s0_rmse = float(np.sqrt(np.mean([c ** 2 for c in phantom_counts])))
    max_phantom = float(np.max(phantom_counts))
    global_mean_cell = total_phantom_mass / max(total_cells, 1)

    # Invert softplus to compute approximate pre-activation bias: y = log(1 + exp(z)) -> z = log(exp(y) - 1)
    if global_mean_cell > 0.0:
        pre_bias_est = float(np.log(np.expm1(global_mean_cell)))
    else:
        pre_bias_est = -float("inf")

    summary = {
        "num_negative_samples": len(results),
        "s0_mae": round(s0_mae, 4),
        "s0_rmse": round(s0_rmse, 4),
        "max_phantom_count": round(max_phantom, 4),
        "global_mean_density_per_cell": float(global_mean_cell),
        "estimated_effective_pre_bias": round(pre_bias_est, 4),
        "per_sample": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(" NWPU PHANTOM BACKGROUND FOG AUDIT RESULTS")
    print("=" * 60)
    print(f" Analyzed Negative Samples : {len(results)}")
    print(f" S0 MAE (False Positive)   : {s0_mae:.4f} heads")
    print(f" S0 RMSE                   : {s0_rmse:.4f}")
    print(f" Max Phantom on Single Pic : {max_phantom:.4f} heads")
    print(f" Mean Cell Residual Density: {global_mean_cell:.6e}")
    print(f" Inferred Pre-Softplus Bias: {pre_bias_est:.4f}")
    print("=" * 60)
    print(f"[SAVED] Results exported to: {out_path}")


if __name__ == "__main__":
    main()
