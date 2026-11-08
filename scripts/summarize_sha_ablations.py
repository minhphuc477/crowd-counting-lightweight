import json
from pathlib import Path

runs = sorted(Path("runs/benchmarks/ablations_sha").glob("exp_sha_abl_*")) + sorted(Path("runs/benchmarks").glob("exp_sha_direct_*"))
print(f"{'Experiment':38s} | {'MAE':8s} | {'RMSE':8s} | {'NAE':8s} | {'Bias':8s}")
print("-" * 80)
for r in runs:
    sum_file = r / "eval_val" / "summary.json"
    if sum_file.is_file():
        with open(sum_file, "r", encoding="utf-8") as f:
            d = json.load(f)
        mae = d.get("MAE", d.get("mae", 0.0))
        rmse = d.get("RMSE", d.get("rmse", 0.0))
        nae = d.get("NAE", d.get("nae", 0.0))
        bias = d.get("Bias", d.get("bias", 0.0))
        print(f"{r.name:38s} | {mae:8.2f} | {rmse:8.2f} | {nae:8.3f} | {bias:+8.2f}")
    else:
        print(f"{r.name:38s} | NO SUMMARY")
