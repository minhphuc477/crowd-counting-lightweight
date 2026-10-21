import csv
from pathlib import Path

runs_dir = Path("runs/sha_a/rmr_v63")
results = []

for r_dir in sorted(runs_dir.iterdir()):
    if not r_dir.is_dir():
        continue
    log_path = r_dir / "train_log.csv"
    if not log_path.is_file():
        continue
    best_mae = float("inf")
    best_ep = 0
    best_rmse = 0.0
    final_mae = 0.0
    final_ep = 0
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row.get("epoch", 0))
            final_ep = ep
            val_mae_str = row.get("val_mae")
            val_rmse_str = row.get("val_rmse")
            if val_mae_str:
                v_mae = float(val_mae_str)
                v_rmse = float(val_rmse_str) if val_rmse_str else 0.0
                final_mae = v_mae
                if v_mae < best_mae:
                    best_mae = v_mae
                    best_ep = ep
                    best_rmse = v_rmse
    results.append({
        "name": r_dir.name,
        "total_epochs": final_ep,
        "best_epoch": best_ep,
        "best_mae": best_mae,
        "best_rmse": best_rmse,
        "final_mae": final_mae,
    })

print(f"{'Run Name':<42} | {'Total Ep':<8} | {'Best Ep':<8} | {'Best MAE':<9} | {'Best RMSE':<9} | {'Final MAE':<9}")
print("-" * 95)
for r in results:
    print(
        f"{r['name']:<42} | "
        f"{r['total_epochs']:<8} | "
        f"{r['best_epoch']:<8} | "
        f"{r['best_mae']:<9.2f} | "
        f"{r['best_rmse']:<9.2f} | "
        f"{r['final_mae']:<9.2f}"
    )
