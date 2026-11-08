import json
from pathlib import Path

def get_summary(path):
    p = Path(path)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {
            "MAE": d.get("MAE", d.get("mae", 0.0)),
            "RMSE": d.get("RMSE", d.get("rmse", 0.0)),
            "NAE": d.get("NAE", d.get("nae", 0.0)),
            "Bias": d.get("Bias", d.get("bias", 0.0)),
        }
    return None

print("=" * 85)
print(f"{'Experiment':45s} | {'MAE':8s} | {'RMSE':8s} | {'NAE':8s} | {'Bias':8s}")
print("=" * 85)

# 1. JHU Benchmarks & Ablations
print("--- [JHU-CROWD++ BENCHMARKS & ABLATIONS] ---")
jhu_runs = sorted(Path("runs/benchmarks/ablations_jhu").glob("exp_jhu_abl_*")) + sorted(Path("runs/benchmarks").glob("exp_jhu_direct_*"))
for r in jhu_runs:
    s = get_summary(r / "eval_val" / "summary.json")
    if s:
        print(f"{r.name:45s} | {s['MAE']:8.2f} | {s['RMSE']:8.2f} | {s['NAE']:8.3f} | {s['Bias']:+8.2f}")
    else:
        # Check train log
        log_file = r / "train_log.csv"
        if log_file.is_file():
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            print(f"{r.name:45s} | RUNNING ({len(lines)-1} epochs logged, no eval_val yet)")
        else:
            print(f"{r.name:45s} | NO DATA")

# 2. QNRF Updates
print("\n--- [UCF-QNRF BENCHMARKS & ABLATIONS (NEW/UPDATED)] ---")
qnrf_runs = sorted(Path("runs/benchmarks/ablations_qnrf").glob("exp_qnrf_abl_c*"))
for r in qnrf_runs:
    s = get_summary(r / "eval_val" / "summary.json")
    if s:
        print(f"{r.name:45s} | {s['MAE']:8.2f} | {s['RMSE']:8.2f} | {s['NAE']:8.3f} | {s['Bias']:+8.2f}")

