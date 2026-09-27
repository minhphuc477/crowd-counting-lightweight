"""Post-Training Evaluation & Ensembling for Sub-60 Benchmark Suite.

Evaluates:
1. Single-model Horizontal Flip Test-Time Augmentation (TTA).
2. Heterogeneous Prediction Blend of the top 3 complementary Pareto models:
   - sub60_count_harmonized (Best Moderate & Overall MAE)
   - sub60_resonant_peak (Best Sparse MAE)
   - sub60_unified_t8 (Best Dense MAE)
3. Comprehensive metrics: MAE, RMSE, NAE, Bias, Sparse, Moderate, Dense, GAME(0..3).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset
from rmr_core.metrics import summarize_predictions
from rmr_v3.engine import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sub-60 Post-Training TTA & Ensemble Evaluation")
    parser.add_argument("--test-manifest", default="data/sha_a_test.jsonl")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model, _ = make_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    print(f"Executing Post-Training Evaluation on: {device}")

    test_ds = CrowdManifestDataset(args.test_manifest, train=False, output_stride=4, data_root=args.data_root)
    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)
    print(f"Loaded test dataset: {len(test_ds)} samples from {args.test_manifest}\n")

    top_runs = {
        "Count-Harmonized (Single Best)": Path("runs/sha_a/sub60_count_harmonized/best_val_mae.pt"),
        "Resonant-Peak (Sparse Champion)": Path("runs/sha_a/sub60_resonant_peak/best_val_mae.pt"),
        "Unified-T8 (Dense Champion)": Path("runs/sha_a/sub60_unified_t8/best_val_mae.pt"),
    }

    models = {}
    for name, path in top_runs.items():
        if path.exists():
            models[name] = load_model_from_ckpt(path, device)
            print(f"Loaded {name} from {path}")
        else:
            print(f"Warning: Checkpoint not found: {path}")

    print("\n" + "=" * 80)
    print("  PHASE 1: INDIVIDUAL MODEL EVALUATION (RAW vs FLIP TTA)")
    print("=" * 80)

    cached_preds: dict[str, list[dict]] = {}

    for name, model in models.items():
        # 1. Raw evaluation
        _, summary_raw = evaluate_dataset(
            model=model, loader=loader, device=device, output_stride=4,
            use_tta=False, enforce_gt_consistency=True
        )
        # 2. TTA evaluation
        rows_tta, summary_tta = evaluate_dataset(
            model=model, loader=loader, device=device, output_stride=4,
            use_tta=True, enforce_gt_consistency=True
        )
        cached_preds[name] = rows_tta
        delta_mae = summary_tta["MAE"] - summary_raw["MAE"]
        print(f"[{name}]")
        print(f"  Raw:  MAE={summary_raw['MAE']:.2f} | RMSE={summary_raw['RMSE']:.2f} | Bias={summary_raw['Bias']:.2f}")
        print(f"  +TTA: MAE={summary_tta['MAE']:.2f} | RMSE={summary_tta['RMSE']:.2f} | Bias={summary_tta['Bias']:.2f} (Delta: {delta_mae:+.2f})")
        print(f"  Subgroups (+TTA): Sparse={summary_tta.get('mae_sparse', 0):.2f} | Mod={summary_tta.get('mae_moderate', 0):.2f} | Dense={summary_tta.get('mae_dense', 0):.2f}")
        print()

    # 3. Multi-Model Ensemble Blend
    if len(models) >= 2:
        print("=" * 80)
        print("  PHASE 2: HETEROGENEOUS PREDICTION BLEND ENSEMBLE")
        print("=" * 80)

        # Weighting: 40% Count-Harmonized, 30% Resonant-Peak, 30% Unified-T8
        weights = {
            "Count-Harmonized (Single Best)": 0.40,
            "Resonant-Peak (Sparse Champion)": 0.30,
            "Unified-T8 (Dense Champion)": 0.30,
        }
        active_weights = {k: weights[k] for k in models.keys()}
        tot_w = sum(active_weights.values())
        norm_weights = {k: v / tot_w for k, v in active_weights.items()}

        ens_rows = []
        n_samples = len(loader)
        first_key = list(models.keys())[0]

        for i in range(n_samples):
            gt = cached_preds[first_key][i]["gt"]
            sid = cached_preds[first_key][i]["id"]
            pred_blend = sum(norm_weights[m_name] * cached_preds[m_name][i]["pred"] for m_name in models.keys())
            ens_rows.append({
                "id": sid,
                "index": i,
                "gt": gt,
                "pred": pred_blend,
                "abs_err": abs(pred_blend - gt),
                "sq_err": (pred_blend - gt) ** 2,
            })

        ens_summary = summarize_predictions(ens_rows)
        print(f"Blend Weights: {norm_weights}")
        print(f"ENSEMBLE RESULTS (+TTA):")
        print(f"  MAE:      {ens_summary['MAE']:.2f}")
        print(f"  RMSE:     {ens_summary['RMSE']:.2f}")
        print(f"  NAE:      {ens_summary['NAE']:.3f}")
        print(f"  Bias:     {ens_summary['Bias']:+.2f}")
        print(f"  Sparse:   {ens_summary.get('mae_sparse', 0):.2f}")
        print(f"  Moderate: {ens_summary.get('mae_moderate', 0):.2f}")
        print(f"  Dense:    {ens_summary.get('mae_dense', 0):.2f}")
        print("=" * 80)


if __name__ == "__main__":
    main()
