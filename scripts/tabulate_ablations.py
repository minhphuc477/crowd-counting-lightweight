from pathlib import Path
import json

def tabulate(root_dir: str):
    print(f"\n{'='*75}")
    print(f"ABLATION RESULTS TABLE: {root_dir}")
    print(f"{'='*75}")
    print(f"{'Experiment Name':38s} | {'MAE':>6s} | {'RMSE':>6s} | {'Bias':>7s} | {'Dense MAE':>9s}")
    print(f"{'-'*75}")
    paths = sorted(Path(root_dir).glob("*/eval_val/summary.json"))
    for p in paths:
        exp_name = p.parent.parent.name
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        mae = d.get("MAE", d.get("mae", 0.0))
        rmse = d.get("RMSE", d.get("rmse", 0.0))
        bias = d.get("Bias", d.get("bias", 0.0))
        dense_mae = d.get("mae_dense", 0.0)
        print(f"{exp_name:38s} | {mae:6.2f} | {rmse:6.2f} | {bias:+7.2f} | {dense_mae:9.2f}")

if __name__ == "__main__":
    tabulate("runs/benchmarks/ablations_sha")
    tabulate("runs/benchmarks/ablations_shb")
    tabulate("runs/benchmarks/ablations_qnrf")
    tabulate("runs/benchmarks/ablations_jhu")
