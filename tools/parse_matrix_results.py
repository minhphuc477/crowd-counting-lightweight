import json
import glob
from pathlib import Path

runs = sorted(glob.glob("runs/sha_a/matrix/*"))
header = f"{'Run Configuration':<30} | {'MAE':>6} | {'RMSE':>6} | {'Sparse':>6} | {'Mod':>6} | {'Dense':>7} | {'Bias':>7} | {'NAE':>6} | {'Help%':>5} | {'Harm%':>5}"
print(header)
print("-" * len(header))

for r in runs:
    p = Path(r)
    sp = p / "eval_val" / "summary.json"
    if not sp.is_file():
        continue
    with open(sp, "r", encoding="utf-8") as f:
        d = json.load(f)
    name = p.name
    mae = d["MAE"]
    rmse = d["RMSE"]
    sparse = d["mae_sparse"]
    mod = d["mae_moderate"]
    dense = d["mae_dense"]
    bias = d["Bias"]
    nae = d["NAE"]
    help_f = d.get("solver_help_fraction", 0.0) * 100
    harm_f = d.get("solver_harm_fraction", 0.0) * 100
    print(f"{name:<30} | {mae:>6.2f} | {rmse:>6.2f} | {sparse:>6.2f} | {mod:>6.2f} | {dense:>7.2f} | {bias:>+7.2f} | {nae:>6.4f} | {help_f:>4.1f}% | {harm_f:>4.1f}%")
