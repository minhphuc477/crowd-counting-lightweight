import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import csv
import torch
from rmr_core.training import safe_torch_load

def analyze_v71():
    base_dir = Path("runs/sha_a/rmr_v71")
    runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
    
    print("=" * 80)
    print(f"FOUND {len(runs)} RUNS IN {base_dir}")
    print("=" * 80)

    summary_rows = []

    for r in runs:
        print(f"\nAnalyzing run: {r.name}")
        ckpt_path = r / "best_val_mae.pt"
        if not ckpt_path.exists():
            print(f"  Missing {ckpt_path}")
            continue

        ckpt = safe_torch_load(ckpt_path, map_location="cpu", weights_only=True)
        epoch = ckpt.get("epoch", -1)
        val_mae = ckpt.get("val_mae", float("nan"))
        val_rmse = ckpt.get("val_rmse", float("nan"))
        cfg = ckpt.get("config", {})
        
        # Load summary.json
        summary_path = r / "eval_val" / "summary.json"
        s_data = {}
        if summary_path.exists():
            with open(summary_path) as f:
                s_data = json.load(f)
        
        # Load predictions.csv
        pred_path = r / "eval_val" / "predictions.csv"
        dense_errors = []
        normal_errors = []
        all_errors = []
        under_count = 0
        over_count = 0
        
        if pred_path.exists():
            with open(pred_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gt = float(row["gt"])
                    pred = float(row["pred"])
                    err = abs(pred - gt)
                    diff = pred - gt
                    all_errors.append(err)
                    if gt > 1000:
                        dense_errors.append((row["id"], gt, pred, diff, err))
                    else:
                        normal_errors.append(err)
                    if diff < 0:
                        under_count += 1
                    else:
                        over_count += 1

        mae = s_data.get("MAE", s_data.get("mae", val_mae))
        rmse = s_data.get("RMSE", s_data.get("rmse", val_rmse))
        nae = s_data.get("NAE", s_data.get("nae", float("nan")))
        game0 = s_data.get("GAME0", s_data.get("game0", float("nan")))
        game1 = s_data.get("GAME1", s_data.get("game1", float("nan")))
        game2 = s_data.get("GAME2", s_data.get("game2", float("nan")))
        game3 = s_data.get("GAME3", s_data.get("game3", float("nan")))
        net_bias = s_data.get("Bias", s_data.get("bias", s_data.get("net_bias", float("nan"))))
        sparse_mae = s_data.get("mae_sparse", s_data.get("sparse_mae", float("nan")))
        dense_mae = s_data.get("mae_dense", s_data.get("dense_mae", float("nan")))
        mod_mae = s_data.get("mae_moderate", float("nan"))

        row_info = {
            "run": r.name,
            "epoch": epoch,
            "mae": mae,
            "rmse": rmse,
            "nae": nae,
            "game0": game0,
            "game1": game1,
            "game2": game2,
            "game3": game3,
            "net_bias": net_bias,
            "sparse_mae": sparse_mae,
            "mod_mae": mod_mae,
            "dense_mae": dense_mae,
            "norm_mae": sum(normal_errors) / len(normal_errors) if normal_errors else float("nan"),
            "dense20_mae": sum(e[4] for e in dense_errors) / len(dense_errors) if dense_errors else float("nan"),
            "dense_count": len(dense_errors),
            "under_count": under_count,
            "over_count": over_count,
        }
        summary_rows.append(row_info)

        print(f"  Best Epoch: {epoch}")
        print(f"  MAE: {mae:.2f} | RMSE: {rmse:.2f} | NAE: {nae:.4f} | Net Bias: {net_bias:+.2f}")
        print(f"  GAME 0/1/2/3: {game0:.2f} / {game1:.2f} / {game2:.2f} / {game3:.2f}")
        print(f"  Sparse MAE (<=100): {sparse_mae:.2f} | Mod MAE (100-500): {mod_mae:.2f} | Dense MAE (>500): {dense_mae:.2f}")
        if dense_errors:
            print(f"  Dense Outliers (>1000): {len(dense_errors)} images, Mean MAE: {row_info['dense20_mae']:.2f}")
            print(f"  Normal Images (<=1000): {len(normal_errors)} images, Mean MAE: {row_info['norm_mae']:.2f}")
            print(f"  Under-predicting: {under_count} images | Over-predicting: {over_count} images")
            print("  Top 5 dense outlier samples:")
            sorted_dense = sorted(dense_errors, key=lambda x: x[4], reverse=True)[:5]
            for sid, gt, pred, diff, err in sorted_dense:
                print(f"    [{sid}] GT: {gt:.0f} -> Pred: {pred:.1f} (Diff: {diff:+.1f}, Err: {err:.1f})")

    print("\n" + "=" * 120)
    print("V71 vs V70 EXP 4b COMPREHENSIVE LEADERBOARD")
    print("=" * 120)
    print(f"{'Run':<45} | {'Ep':<5} | {'MAE':<7} | {'RMSE':<7} | {'GAME3':<7} | {'NAE':<6} | {'Bias':<7} | {'Norm':<6} | {'Dense20':<7}")
    print("-" * 120)
    
    # Also load V70 Exp 4b
    v70_path = Path("runs/sha_a/rmr_v70/v70_track_b_exp4b_no_solver/eval_val/summary.json")
    if v70_path.exists():
        with open(v70_path) as f:
            v70_data = json.load(f)
        v70_mae = v70_data.get("MAE", v70_data.get("mae"))
        v70_rmse = v70_data.get("RMSE", v70_data.get("rmse"))
        v70_game3 = v70_data.get("GAME3", v70_data.get("game3"))
        v70_nae = v70_data.get("NAE", v70_data.get("nae"))
        v70_bias = v70_data.get("Bias", v70_data.get("bias"))
        print(f"{'V70 Exp 4b (Pure Radon-OT Baseline Champion)':<45} | {'1495':<5} | {v70_mae:<7.2f} | {v70_rmse:<7.2f} | {v70_game3:<7.2f} | {v70_nae:<6.4f} | {v70_bias:<+7.2f} | {'41.30':<6} | {'263.2':<7}")
        print("-" * 120)

    for row in sorted(summary_rows, key=lambda x: x["mae"]):
        print(f"{row['run']:<45} | {row['epoch']:<5} | {row['mae']:<7.2f} | {row['rmse']:<7.2f} | {row['game3']:<7.2f} | {row['nae']:<6.4f} | {row['net_bias']:<+7.2f} | {row['norm_mae']:<6.2f} | {row['dense20_mae']:<7.2f}")

if __name__ == "__main__":
    analyze_v71()
