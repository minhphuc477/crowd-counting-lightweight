#!/usr/bin/env python3
"""Cross-Dataset Zero-Shot Generalization Benchmark for NWPU-Trained Checkpoint.

Evaluates an NWPU-trained model checkpoint (0-epoch adaptation) across:
1. ShanghaiTech Part A Test (N = 182)
2. ShanghaiTech Part B Test (N = 316)
3. UCF-QNRF Test (N = 334)
4. JHU-CROWD++ Test (N = 1600)

Demonstrates the geometric universality of the continuous Radon measure formulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_v3.eval import load_model_from_ckpt


BENCHMARK_TARGETS = [
    {"name": "ShanghaiTech Part A", "manifest": "data/sha_a_test.jsonl", "bins": (100.0, 500.0)},
    {"name": "ShanghaiTech Part B", "manifest": "data/shb_test.jsonl", "bins": (20.0, 100.0)},
    {"name": "UCF-QNRF",            "manifest": "data/qnrf_test.jsonl", "bins": (200.0, 1000.0)},
    {"name": "JHU-CROWD++",         "manifest": "data/jhu_test.jsonl", "bins": (50.0, 500.0)},
]


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Zero-Shot Generalization Benchmark")
    parser.add_argument("--checkpoint", "--ckpt", required=True, help="Path to NWPU trained model checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default="results/nwpu_zero_shot_generalization.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f" NWPU ZERO-SHOT CROSS-DATASET GENERALIZATION BENCHMARK")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device    : {device}")
    print("=" * 75)

    model, _, _, cfg = load_model_from_ckpt(Path(args.checkpoint), device=device)
    model.eval()

    m_cfg = cfg.get("model", {})
    d_cfg = cfg.get("data", {})
    eff_stride = 2 if bool(m_cfg.get("subpixel_stride2", False)) else int(m_cfg.get("output_stride", 4))
    backbone_name = m_cfg.get("backbone_name", m_cfg.get("backbone"))
    norm_mode = CrowdManifestDataset.resolve_norm(backbone_name, d_cfg.get("norm_mode", "auto"))

    summary_records = []

    for target in BENCHMARK_TARGETS:
        mf_path = Path(_REPO_ROOT) / target["manifest"]
        if not mf_path.is_file():
            print(f"  [SKIP] Manifest not found: {mf_path}")
            continue

        print(f"\n--> Evaluating Zero-Shot on {target['name']} ({mf_path})...")
        dataset = CrowdManifestDataset(
            manifest=mf_path,
            train=False,
            data_root="data",
            max_size=2048,
            output_stride=eff_stride,
            norm_mode=norm_mode,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_eval, num_workers=2)

        rows, summary = evaluate_dataset(
            model=model,
            loader=loader,
            device=device,
            output_stride=eff_stride,
            density_bins=target["bins"],
        )

        record = {
            "target_dataset": target["name"],
            "num_samples": len(dataset),
            "mae": round(summary["mae"], 2),
            "rmse": round(summary["rmse"], 2),
            "nae": round(summary.get("nae", 0.0), 3),
            "bias": round(summary.get("bias", 0.0), 2),
            "sparse_mae": round(summary.get("sparse_mae", 0.0), 2),
            "moderate_mae": round(summary.get("moderate_mae", 0.0), 2),
            "dense_mae": round(summary.get("dense_mae", 0.0), 2),
        }
        summary_records.append(record)
        print(f"  Results: MAE={record['mae']:.2f} | RMSE={record['rmse']:.2f} | NAE={record['nae']:.3f}")

    print("\n" + "=" * 75)
    print(" ZERO-SHOT GENERALIZATION SUMMARY")
    print("=" * 75)
    print(f"{'Target Dataset':<25} | {'Samples':<8} | {'MAE':<8} | {'RMSE':<8} | {'NAE':<8}")
    print("-" * 75)
    for r in summary_records:
        print(f"{r['target_dataset']:<25} | {r['num_samples']:<8} | {r['mae']:<8.2f} | {r['rmse']:<8.2f} | {r['nae']:<8.3f}")
    print("=" * 75)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"[SAVED] Cross-dataset evaluation metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
