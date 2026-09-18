import csv
import json
from pathlib import Path

v25_runs = [
    "rmr_v25_canonical",
    "rmr_v25_ablation_no_bb",
    "rmr_v25_ablation_no_perspective",
    "rmr_v25_ablation_no_curvature",
    "rmr_v25_ablation_no_morozov",
    "rmr_v25_control_no_solver",
]

print("=" * 115)
print("                      RMR-v25 EMPIRICAL BENCHMARK SUITE (DIRECT EVALUATION)")
print("=" * 115)
print(f"{'Run':<32} | {'Epochs':<6} | {'BestEp':<6} | {'MAE':<7} | {'RMSE':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'95% CI MAE':<15}")
print("-" * 115)

results = []
for r in v25_runs:
    run_dir = Path("runs/sha_a") / r
    summary_path = run_dir / "eval_val" / "summary.json"
    train_log_path = run_dir / "train_log.csv"

    total_epochs = 0
    best_ep = 0
    best_mae = 9999.0
    if train_log_path.exists():
        with open(train_log_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_epochs += 1
                try:
                    v = float(row.get("val_mae", 9999.0))
                    ep = int(row.get("epoch", 0))
                    if v < best_mae:
                        best_mae = v
                        best_ep = ep
                except (ValueError, TypeError):
                    pass

    mae = rmse = sparse = mod = dense = ci95 = "N/A"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        mae = f"{s.get('MAE', s.get('mae', 0.0)):.2f}"
        rmse = f"{s.get('RMSE', s.get('rmse', 0.0)):.2f}"
        sparse = f"{s.get('mae_sparse', s.get('density_sparse_mae', 0.0)):.2f}"
        mod = f"{s.get('mae_moderate', s.get('density_medium_mae', 0.0)):.2f}"
        dense = f"{s.get('mae_dense', s.get('density_dense_mae', 0.0)):.2f}"
        ci = s.get("mae_ci95", None)
        if ci:
            ci95 = f"[{ci[0]:.2f}, {ci[1]:.2f}]"

    print(f"{r:<32} | {total_epochs:<6} | {best_ep:<6} | {mae:<7} | {rmse:<7} | {sparse:<7} | {mod:<7} | {dense:<7} | {ci95:<15}")
    results.append({
        "run": r,
        "epochs": total_epochs,
        "best_ep": best_ep,
        "mae": mae,
        "rmse": rmse,
        "sparse": sparse,
        "mod": mod,
        "dense": dense,
        "ci95": ci95,
    })

print("=" * 115)
