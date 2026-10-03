import json
from pathlib import Path

runs = sorted(Path("runs/sha_a").glob("*/eval_val/summary.json"))
header = f"{'Run Name':<38} | {'MAE':<7} | {'RMSE':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'Bias':<7}"
print(header)
print("-" * len(header))
for p in runs:
    run = p.parent.parent.name
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    mae = d.get("MAE", 0.0)
    rmse = d.get("RMSE", 0.0)
    sparse = d.get("mae_sparse", 0.0)
    mod = d.get("mae_moderate", 0.0)
    dense = d.get("mae_dense", 0.0)
    bias = d.get("Bias", 0.0)
    print(f"{run:<38} | {mae:<7.2f} | {rmse:<7.2f} | {sparse:<7.2f} | {mod:<7.2f} | {dense:<7.2f} | {bias:<7.2f}")
