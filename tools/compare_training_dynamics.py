import csv
from pathlib import Path

def load_train_log(p):
    rows = []
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "epoch": int(r["epoch"]),
                    "train_loss": float(r.get("train_loss", 0.0)),
                    "val_mae": float(r.get("val_mae", 9999.0)),
                    "val_rmse": float(r.get("val_rmse", 9999.0)),
                })
            except (ValueError, KeyError):
                pass
    return rows

log19 = load_train_log("runs/sha_a/rmr_v19_canonical_isotropic/train_log.csv")
log25_can = load_train_log("runs/sha_a/rmr_v25_canonical/train_log.csv")
log25_np = load_train_log("runs/sha_a/rmr_v25_ablation_no_perspective/train_log.csv")
log25_nocurv = load_train_log("runs/sha_a/rmr_v25_ablation_no_curvature/train_log.csv")
log25_nobb = load_train_log("runs/sha_a/rmr_v25_ablation_no_bb/train_log.csv")

print("="*80)
print("TRAINING DYNAMICS & BEST EPOCH TRAJECTORY")
print("="*80)

for name, log in [
    ("v19 Canonical", log19),
    ("v25 Canonical", log25_can),
    ("v25 No Perspective", log25_np),
    ("v25 No Curvature", log25_nocurv),
    ("v25 No BB", log25_nobb),
]:
    if not log:
        print(f"{name}: log empty or missing")
        continue
    best = min(log, key=lambda x: x["val_mae"])
    final = log[-1]
    
    # Count epochs where val_mae < 80 and val_mae < 75
    sub80 = sum(1 for x in log if x["val_mae"] < 80.0)
    sub75 = sum(1 for x in log if x["val_mae"] < 75.0)
    
    # Top 5 best epochs
    top5 = sorted(log, key=lambda x: x["val_mae"])[:5]
    top5_str = ", ".join([f"ep{x['epoch']}={x['val_mae']:.2f}" for x in top5])
    
    print(f"\n{name} (Total Epochs: {len(log)}):")
    print(f"  Best:  Epoch {best['epoch']:<4} -> MAE {best['val_mae']:.2f} (RMSE {best['val_rmse']:.2f})")
    print(f"  Final: Epoch {final['epoch']:<4} -> MAE {final['val_mae']:.2f}")
    print(f"  Stability: {sub80} epochs < 80 MAE, {sub75} epochs < 75 MAE")
    print(f"  Top 5: {top5_str}")
