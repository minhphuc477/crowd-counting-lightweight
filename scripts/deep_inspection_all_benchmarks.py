import json
import sys
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

def get_best_epoch(log_path):
    if not Path(log_path).is_file():
        return None, None
    try:
        df = pd.read_csv(log_path)
        if "val_mae" not in df.columns or df["val_mae"].dropna().empty:
            return len(df), None
        best_idx = df["val_mae"].idxmin()
        best_row = df.loc[best_idx]
        return len(df), (int(best_row["epoch"]), float(best_row["val_mae"]), float(best_row["val_rmse"]))
    except Exception:
        return None, None

def inspect_group(title, paths):
    print("\n" + "=" * 90)
    print(f"  {title.upper()}")
    print("=" * 90)
    print(f"{'Experiment':42s} | {'Val MAE':8s} | {'Val RMSE':8s} | {'NAE':7s} | {'Bias':8s} | {'Best Ep':8s} | {'Status'}")
    print("-" * 90)
    for p in paths:
        name = p.name
        sum_p = p / "eval_val" / "summary.json"
        log_p = p / "train_log.csv"
        total_eps, best_info = get_best_epoch(log_p)
        
        if sum_p.is_file():
            with open(sum_p, "r", encoding="utf-8") as f:
                d = json.load(f)
            mae = d.get("MAE", d.get("mae", 0.0))
            rmse = d.get("RMSE", d.get("rmse", 0.0))
            nae = d.get("NAE", d.get("nae", 0.0))
            bias = d.get("Bias", d.get("bias", 0.0))
            b_ep_str = f"Ep {best_info[0]}" if best_info else "N/A"
            status = f"Completed ({total_eps} ep)" if total_eps and total_eps >= 500 else f"Eval ({total_eps} ep)"
            print(f"{name:42s} | {mae:8.2f} | {rmse:8.2f} | {nae:7.3f} | {bias:+8.2f} | {b_ep_str:8s} | {status}")
        elif total_eps is not None:
            if best_info:
                print(f"{name:42s} | {best_info[1]:8.2f} | {best_info[2]:8.2f} | {'N/A':7s} | {'N/A':8s} | {f'Ep {best_info[0]}':8s} | Running ({total_eps} ep)")
            else:
                print(f"{name:42s} | {'---':8s} | {'---':8s} | {'---':7s} | {'---':8s} | {'---':8s} | Started ({total_eps} ep)")
        else:
            print(f"{name:42s} | {'---':8s} | {'---':8s} | {'---':7s} | {'---':8s} | {'---':8s} | Queued/Pending")

# 1. JHU-CROWD++
jhu_runs = (
    sorted(Path("runs/benchmarks").glob("exp_jhu_direct_*")) +
    sorted(Path("runs/benchmarks/ablations_jhu").glob("exp_jhu_abl_*"))
)
inspect_group("JHU-CROWD++ (4,372 images, 1.51M heads)", jhu_runs)

# 2. UCF-QNRF
qnrf_runs = (
    sorted(Path("runs/benchmarks").glob("exp_qnrf_direct_*")) +
    sorted(Path("runs/benchmarks/ablations_qnrf").glob("exp_qnrf_abl_*"))
)
inspect_group("UCF-QNRF (1,535 high-res images, 1.25M heads)", qnrf_runs)

# 3. ShanghaiTech Part A
sha_runs = (
    sorted(Path("runs/benchmarks").glob("exp_sha_direct_*")) +
    sorted(Path("runs/benchmarks/ablations_sha").glob("exp_sha_abl_*"))
)
inspect_group("ShanghaiTech Part A (Ultra-dense)", sha_runs)

# 4. ShanghaiTech Part B
shb_runs = (
    sorted(Path("runs/benchmarks").glob("exp_shb_direct_*")) +
    sorted(Path("runs/benchmarks/ablations_shb").glob("exp_shb_abl_*"))
)
inspect_group("ShanghaiTech Part B (Wide perspective)", shb_runs)

# 5. UCF_CC_50
ucf_runs = sorted(Path("runs/benchmarks").glob("exp_ucf_cc_50_*"))
inspect_group("UCF_CC_50 (5-Fold Cross Validation)", ucf_runs)
