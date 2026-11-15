#!/usr/bin/env python3
"""Fine-Grained Attribute & Environmental Robustness Evaluator for NWPU-Crowd.

Evaluates any RMR-Net checkpoint across:
1. Density Tiers (S0: ==0, S1: 1-100, S2: 101-500, S3: 501-2000, S4: >2000)
2. Luminance Conditions (L0: Night, L1: Low-Light, L2: Normal)
3. Scene Modalities (Negative/Distractor, Outdoor, Indoor)
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


def compute_subset_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"mae": 0.0, "rmse": 0.0, "nae": 0.0, "bias": 0.0, "n": 0}
    aes = [r["abs_err"] for r in rows]
    sqs = [r["sq_err"] for r in rows]
    gts = [r["gt"] for r in rows]
    preds = [r["pred"] for r in rows]

    pos_mask = [g > 0 for g in gts]
    if any(pos_mask):
        nae = float(np.mean([abs(p - g) / g for p, g, m in zip(preds, gts, pos_mask) if m]))
    else:
        nae = 0.0

    return {
        "mae": round(float(np.mean(aes)), 2),
        "rmse": round(float(np.sqrt(np.mean(sqs))), 2),
        "nae": round(nae, 3),
        "bias": round(float(np.mean(np.array(preds) - np.array(gts))), 2),
        "n": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description="NWPU-Crowd Attribute & Robustness Evaluator")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to model checkpoint")
    parser.add_argument("--manifest", default="data/nwpu_val.jsonl", help="Evaluation manifest")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--output-json", default="results/nwpu_val_attribute_breakdown.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"[NWPU EVAL] Loading checkpoint: {args.checkpoint}")
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

    manifest_path = Path(args.manifest)
    print(f"[NWPU EVAL] Loading dataset from: {manifest_path} (stride={eff_stride}, norm={norm_mode})")
    dataset = CrowdManifestDataset(
        manifest=manifest_path,
        train=False,
        data_root="data",
        max_size=args.max_size,
        output_stride=eff_stride,
        norm_mode=norm_mode,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval, num_workers=2)

    print(f"[NWPU EVAL] Running evaluation on {len(dataset)} samples...")
    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=eff_stride,
        density_bins=(100.0, 500.0),
    )

    # Load item metadata from manifest for attribute mapping
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_items = [json.loads(line) for line in f if line.strip()]

    item_meta = {str(it.get("id", idx)): it for idx, it in enumerate(manifest_items)}

    # Group rows
    density_map = {
        "S0: Zero (0)": lambda c: c == 0,
        "S1: Sparse (1-100)": lambda c: 1 <= c <= 100,
        "S2: Moderate (101-500)": lambda c: 101 <= c <= 500,
        "S3: Dense (501-2000)": lambda c: 501 <= c <= 2000,
        "S4: Extreme (>2000)": lambda c: c > 2000,
    }
    lum_labels = {0: "L0: Night", 1: "L1: Low Light", 2: "L2: Normal"}
    scene_labels = {0: "Negative (Scene 0)", 1: "Outdoor (Scene 1)", 2: "Indoor (Scene 2)"}

    density_rows = defaultdict(list)
    lum_rows = defaultdict(list)
    scene_rows = defaultdict(list)

    for r in rows:
        meta = item_meta.get(str(r["id"]), {})
        gt_c = r["gt"]
        for d_name, d_fn in density_map.items():
            if d_fn(gt_c):
                density_rows[d_name].append(r)
                break

        lum_code = meta.get("luminance", -1)
        if lum_code in lum_labels:
            lum_rows[lum_labels[lum_code]].append(r)

        sc_code = meta.get("scene_type", -1)
        if sc_code in scene_labels:
            scene_rows[scene_labels[sc_code]].append(r)

    print("\n" + "=" * 80)
    print(" NWPU-CROWD OVERALL VALIDATION METRICS")
    print("=" * 80)
    print(f"  Samples: {summary['num_samples']} | MAE: {summary['MAE']:.2f} | RMSE: {summary['RMSE']:.2f} | NAE: {summary['NAE']:.3f} | Bias: {summary['Bias']:+.2f}")
    if "GAME0" in summary:
        print(f"  GAME(0): {summary['GAME0']:.2f} | GAME(1): {summary.get('GAME1', 0):.2f} | GAME(2): {summary.get('GAME2', 0):.2f} | GAME(3): {summary.get('GAME3', 0):.2f}")

    print("\n" + "-" * 80)
    print(" 1. DENSITY TIER BREAKDOWN (S0 - S4)")
    print("-" * 80)
    print(f"{'Tier':26s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s} | {'Bias':10s}")
    print("-" * 80)
    density_res = {}
    for d_name in density_map.keys():
        sub_rows = density_rows[d_name]
        res = compute_subset_metrics(sub_rows)
        density_res[d_name] = res
        print(f"{d_name:26s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f} | {res['bias']:+10.2f}")

    print("\n" + "-" * 80)
    print(" 2. LUMINANCE CONDITION BREAKDOWN (L0 - L2)")
    print("-" * 80)
    print(f"{'Condition':26s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s} | {'Bias':10s}")
    print("-" * 80)
    lum_res = {}
    for l_name in lum_labels.values():
        sub_rows = lum_rows[l_name]
        res = compute_subset_metrics(sub_rows)
        lum_res[l_name] = res
        print(f"{l_name:26s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f} | {res['bias']:+10.2f}")

    print("\n" + "-" * 80)
    print(" 3. SCENE MODALITY BREAKDOWN")
    print("-" * 80)
    print(f"{'Scene Type':26s} | {'Samples':8s} | {'MAE':10s} | {'RMSE':10s} | {'Bias':10s}")
    print("-" * 80)
    scene_res = {}
    for sc_name in scene_labels.values():
        sub_rows = scene_rows[sc_name]
        res = compute_subset_metrics(sub_rows)
        scene_res[sc_name] = res
        print(f"{sc_name:26s} | {res['n']:8d} | {res['mae']:10.2f} | {res['rmse']:10.2f} | {res['bias']:+10.2f}")
    print("=" * 80)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "overall": {
                "num_samples": summary["num_samples"],
                "MAE": round(summary["MAE"], 2),
                "RMSE": round(summary["RMSE"], 2),
                "NAE": round(summary["NAE"], 3),
                "Bias": round(summary["Bias"], 2),
                "GAME0": round(summary.get("GAME0", 0), 2),
                "GAME1": round(summary.get("GAME1", 0), 2),
                "GAME2": round(summary.get("GAME2", 0), 2),
                "GAME3": round(summary.get("GAME3", 0), 2),
            },
            "density_tiers": density_res,
            "luminance": lum_res,
            "scenes": scene_res,
        }
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\n[SAVED] Comprehensive NWPU attribute breakdown saved to: {out_p}")


if __name__ == "__main__":
    main()
