#!/usr/bin/env python3
"""Fine-Grained Attribute & Environmental Robustness Evaluator for JHU-CROWD++.

Evaluates any checkpoint across:
1. Weather Degradation: Clear (0), Fog/Haze (1), Rain (2), Snow (3)
2. Distractor Negative Scenes: Normal (0) vs Distractor (1)
3. Density Tiers: Zero (0), Low (<=50), Medium (51-500), High (>500)
4. Scene Categories: Top categories (stadium, street, protest, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.eval import load_model_from_ckpt


def load_jhu_attributes(labels_file: Path) -> dict[str, dict]:
    """Parse image_labels.txt: filename,total-count,scene-type,weather-condition,distractor"""
    attrs = {}
    if not labels_file.is_file():
        return attrs
    with open(labels_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                fname = parts[0]
                attrs[fname] = {
                    "total_count": int(parts[1]),
                    "scene_type": parts[2],
                    "weather": int(parts[3]),
                    "distractor": int(parts[4]),
                }
    return attrs


def compute_subset_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"mae": 0.0, "rmse": 0.0, "n": 0}
    aes = [r["abs_err"] for r in rows]
    sqs = [r["sq_err"] for r in rows]
    return {
        "mae": round(float(np.mean(aes)), 2),
        "rmse": round(float(np.sqrt(np.mean(sqs))), 2),
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="JHU-CROWD++ Attribute Evaluator")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", default="data/jhu_test.jsonl", help="Evaluation manifest")
    parser.add_argument("--labels", default="data/jhu_crowd_v2.0/test/image_labels.txt", help="Path to image_labels.txt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--output-json", default=None, help="Save summary to JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading checkpoint from: {args.checkpoint}")
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

    print(f"Loading evaluation dataset from: {args.manifest} (stride={eff_stride}, norm={norm_mode})")
    dataset = CrowdManifestDataset(
        manifest=args.manifest,
        train=False,
        data_root="data",
        max_size=args.max_size,
        output_stride=eff_stride,
        norm_mode=norm_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval, num_workers=2)

    print(f"Running full-image inference across {len(dataset)} images on {device}...")
    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=eff_stride,
        density_bins=(50.0, 500.0),
    )

    print("\n" + "=" * 80)
    print(" OVERALL TEST METRICS (1,600 IMAGES)")
    print("=" * 80)
    print(f"  MAE: {summary['MAE']:.2f} | RMSE: {summary['RMSE']:.2f} | NAE: {summary['NAE']:.3f} | Bias: {summary['Bias']:+.2f}")

    # Load environmental attributes
    attrs = load_jhu_attributes(Path(args.labels))
    if not attrs:
        print(f"Warning: image_labels.txt not found at {args.labels}. Skipping attribute breakdown.")
        return

    # Map evaluation rows to attributes
    weather_map = {0: "Clear / Normal", 1: "Fog / Haze", 2: "Rain", 3: "Snow"}
    weather_rows = defaultdict(list)
    distractor_rows = defaultdict(list)
    scene_rows = defaultdict(list)

    for r in rows:
        stem = Path(r.get("id", "")).stem.replace("jhu_test_", "")
        if stem in attrs:
            at = attrs[stem]
            weather_rows[at["weather"]].append(r)
            distractor_rows[at["distractor"]].append(r)
            scene_rows[at["scene_type"]].append(r)

    print("\n" + "-" * 80)
    print(" 1. WEATHER CONDITION ROBUSTNESS BREAKDOWN")
    print("-" * 80)
    print(f"{'Weather':20s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s}")
    print("-" * 80)
    weather_results = {}
    for w_code in [0, 1, 2, 3]:
        sub_rows = weather_rows[w_code]
        res = compute_subset_metrics(sub_rows)
        w_name = weather_map[w_code]
        weather_results[w_name] = res
        print(f"{w_name:20s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f}")

    print("\n" + "-" * 80)
    print(" 2. DISTRACTOR NEGATIVE SCENE ROBUSTNESS")
    print("-" * 80)
    print(f"{'Category':20s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s}")
    print("-" * 80)
    distractor_results = {}
    for d_code, d_name in [(0, "Normal Crowd Scenes"), (1, "Distractor Negatives")]:
        sub_rows = distractor_rows[d_code]
        res = compute_subset_metrics(sub_rows)
        distractor_results[d_name] = res
        print(f"{d_name:20s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f}")

    print("\n" + "-" * 80)
    print(" 3. TOP SCENE CATEGORIES BREAKDOWN")
    print("-" * 80)
    print(f"{'Scene Category':20s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s}")
    print("-" * 80)
    scene_results = {}
    top_scenes = sorted(scene_rows.items(), key=lambda kv: len(kv[1]), reverse=True)[:8]
    for s_name, sub_rows in top_scenes:
        res = compute_subset_metrics(sub_rows)
        scene_results[s_name] = res
        print(f"{s_name:20s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f}")

    print("=" * 80)

    if args.output_json:
        out_data = {
            "overall": {"mae": summary["MAE"], "rmse": summary["RMSE"], "nae": summary["NAE"]},
            "weather": weather_results,
            "distractors": distractor_results,
            "scenes": scene_results,
        }
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)
        print(f"\nDetailed attribute breakdown saved to: {args.output_json}")


if __name__ == "__main__":
    main()
