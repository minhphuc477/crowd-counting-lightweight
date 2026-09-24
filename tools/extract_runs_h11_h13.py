"""
Extract and analyze experimental results for H11 and H13 runs in runs/sha_a.
"""
import os
import json
import csv
from pathlib import Path

RUNS_DIR = Path("f:/lightweightcrcn/runs/sha_a")

TARGET_RUNS = [
    # H11 Core & Ablations
    "rmr_h11_harmonious_peak",
    "rmr_h11_abl_no_solver",
    "rmr_h11_abl_no_resonant",
    "rmr_h11_abl_no_spectral",
    "rmr_h11_abl_no_ci_cell",
    "rmr_h11_abl_soft_proximal",
    "rmr_h11_abl_no_proximal",
    "rmr_h11_abl_no_tv",
    "rmr_h11_charbonnier_tv",
    # H13 Core & Ablations
    "rmr_h13_harmonious_frontier",
    "rmr_h13_abl_no_cpcm",
    "rmr_h13_abl_no_asam",
    "rmr_h13_abl_no_bb",
]

def analyze_run(run_name):
    run_path = RUNS_DIR / run_name
    if not run_path.exists():
        return {"error": f"Path not found: {run_path}"}
    
    summary_path = run_path / "eval_val" / "summary.json"
    log_path = run_path / "train_log.csv"
    
    data = {"run_name": run_name}
    
    # 1. Summary JSON
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
            data["summary_mae"] = summary.get("mae", summary.get("MAE"))
            data["summary_rmse"] = summary.get("rmse", summary.get("RMSE"))
            data["summary_bias"] = summary.get("bias", summary.get("Bias"))
            data["mae_sparse"] = summary.get("mae_sparse")
            data["mae_moderate"] = summary.get("mae_moderate")
            data["mae_dense"] = summary.get("mae_dense")
            data["game0"] = summary.get("GAME0")
            data["game1"] = summary.get("GAME1")
            data["game2"] = summary.get("GAME2")
            data["game3"] = summary.get("GAME3")
            data["solver_help_fraction"] = summary.get("solver_help_fraction")
            data["solver_delta_e_mean"] = summary.get("solver_delta_e_mean")
            data["energy_monotonic_fraction"] = summary.get("energy_monotonic_fraction")
            data["mae_reg_y0"] = summary.get("mae_reg_y0")
            data["mae_reg_y8"] = summary.get("mae_reg_y8")
    else:
        data["summary_error"] = "eval_val/summary.json not found"

    # 2. Train log CSV
    if log_path.exists():
        best_epoch = None
        best_val_mae = float("inf")
        best_row = None
        total_epochs = 0
        
        with open(log_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_epochs += 1
                val_mae_str = row.get("val_mae", "")
                if val_mae_str and val_mae_str.strip():
                    try:
                        vmae = float(val_mae_str)
                        if vmae < best_val_mae:
                            best_val_mae = vmae
                            best_epoch = int(row.get("epoch", total_epochs))
                            best_row = row
                    except ValueError:
                        pass
                        
        data["total_epochs"] = total_epochs
        data["best_epoch"] = best_epoch
        data["best_log_val_mae"] = best_val_mae
        if best_row:
            data["log_rmse_at_best"] = float(best_row.get("val_rmse", 0)) if best_row.get("val_rmse") else None
            data["log_bias_at_best"] = float(best_row.get("val_bias", 0)) if best_row.get("val_bias") else None
            data["log_sparse_at_best"] = float(best_row.get("val_mae_sparse", 0)) if best_row.get("val_mae_sparse") else None
            data["log_mod_at_best"] = float(best_row.get("val_mae_moderate", 0)) if best_row.get("val_mae_moderate") else None
            data["log_dense_at_best"] = float(best_row.get("val_mae_dense", 0)) if best_row.get("val_mae_dense") else None
    else:
        data["log_error"] = "train_log.csv not found"
        
    return data

def main():
    results = []
    for r in TARGET_RUNS:
        results.append(analyze_run(r))
        
    print(f"{'Run Name':<32} | {'Epoch':<5} | {'Summary MAE':<11} | {'RMSE':<7} | {'Bias':<7} | {'Sparse':<7} | {'Moderate':<8} | {'Dense':<7}")
    print("-" * 105)
    for res in results:
        name = res["run_name"]
        ep = str(res.get("best_epoch", "N/A"))
        s_mae = f"{res.get('summary_mae', 0.0):.2f}" if res.get('summary_mae') is not None else "N/A"
        s_rmse = f"{res.get('summary_rmse', 0.0):.2f}" if res.get('summary_rmse') is not None else "N/A"
        s_bias = f"{res.get('summary_bias', 0.0):.2f}" if res.get('summary_bias') is not None else "N/A"
        sp = f"{res.get('mae_sparse', 0.0):.2f}" if res.get('mae_sparse') is not None else "N/A"
        mod = f"{res.get('mae_moderate', 0.0):.2f}" if res.get('mae_moderate') is not None else "N/A"
        de = f"{res.get('mae_dense', 0.0):.2f}" if res.get('mae_dense') is not None else "N/A"
        print(f"{name:<32} | {ep:<5} | {s_mae:<11} | {s_rmse:<7} | {s_bias:<7} | {sp:<7} | {mod:<8} | {de:<7}")

    # Output full json for precise analysis
    out_file = Path("runs/sha_a/h11_h13_analysis_extracted.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed JSON to {out_file}")

if __name__ == "__main__":
    main()
