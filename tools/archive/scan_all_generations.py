"""
Scan all directories in runs/sha_a to extract all historical generations (v-series and H-series).
"""
import os
import json
import csv
from pathlib import Path

RUNS_DIR = Path("f:/lightweightcrcn/runs/sha_a")

def inspect_all():
    runs = []
    for item in sorted(RUNS_DIR.iterdir()):
        if not item.is_dir():
            continue
        name = item.name
        summary_path = item / "eval_val" / "summary.json"
        log_path = item / "train_log.csv"
        
        mae = None
        rmse = None
        bias = None
        sparse = None
        moderate = None
        dense = None
        game3 = None
        best_ep = None
        min_log_mae = None
        
        if summary_path.exists():
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    mae = s.get("mae", s.get("MAE"))
                    rmse = s.get("rmse", s.get("RMSE"))
                    bias = s.get("bias", s.get("Bias"))
                    sparse = s.get("mae_sparse")
                    moderate = s.get("mae_moderate")
                    dense = s.get("mae_dense")
                    game3 = s.get("GAME3")
            except Exception:
                pass
                
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        vmae_str = row.get("val_mae", "")
                        if vmae_str and vmae_str.strip():
                            v = float(vmae_str)
                            if min_log_mae is None or v < min_log_mae:
                                min_log_mae = v
                                best_ep = int(row.get("epoch", 0))
            except Exception:
                pass
                
        if mae is not None or min_log_mae is not None:
            eff_mae = mae if mae is not None else min_log_mae
            runs.append({
                "name": name,
                "best_epoch": best_ep,
                "mae": eff_mae,
                "rmse": rmse,
                "bias": bias,
                "sparse": sparse,
                "moderate": moderate,
                "dense": dense,
                "game3": game3,
            })
            
    # Sort runs by name
    return runs

def main():
    runs = inspect_all()
    print(f"Total evaluated runs found: {len(runs)}")
    
    # Categorize into V-series and H-series
    v_series = [r for r in runs if "_v" in r["name"]]
    h_series = [r for r in runs if "_h" in r["name"]]
    other = [r for r in runs if r not in v_series and r not in h_series]
    
    print("\n" + "=" * 105)
    print("V-SERIES GENERATIONS:")
    print("=" * 105)
    print(f"{'Run Name':<38} | {'Epoch':<5} | {'Val MAE':<8} | {'RMSE':<7} | {'Bias':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7}")
    print("-" * 105)
    for r in v_series:
        m = f"{r['mae']:.2f}" if r['mae'] is not None else "N/A"
        rm = f"{r['rmse']:.2f}" if r['rmse'] is not None else "N/A"
        b = f"{r['bias']:.2f}" if r['bias'] is not None else "N/A"
        sp = f"{r['sparse']:.2f}" if r['sparse'] is not None else "N/A"
        mo = f"{r['moderate']:.2f}" if r['moderate'] is not None else "N/A"
        de = f"{r['dense']:.2f}" if r['dense'] is not None else "N/A"
        ep = str(r['best_epoch']) if r['best_epoch'] is not None else "N/A"
        print(f"{r['name']:<38} | {ep:<5} | {m:<8} | {rm:<7} | {b:<7} | {sp:<7} | {mo:<7} | {de:<7}")

    print("\n" + "=" * 105)
    print("H-SERIES GENERATIONS:")
    print("=" * 105)
    print(f"{'Run Name':<38} | {'Epoch':<5} | {'Val MAE':<8} | {'RMSE':<7} | {'Bias':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7}")
    print("-" * 105)
    for r in h_series:
        m = f"{r['mae']:.2f}" if r['mae'] is not None else "N/A"
        rm = f"{r['rmse']:.2f}" if r['rmse'] is not None else "N/A"
        b = f"{r['bias']:.2f}" if r['bias'] is not None else "N/A"
        sp = f"{r['sparse']:.2f}" if r['sparse'] is not None else "N/A"
        mo = f"{r['moderate']:.2f}" if r['moderate'] is not None else "N/A"
        de = f"{r['dense']:.2f}" if r['dense'] is not None else "N/A"
        ep = str(r['best_epoch']) if r['best_epoch'] is not None else "N/A"
        print(f"{r['name']:<38} | {ep:<5} | {m:<8} | {rm:<7} | {b:<7} | {sp:<7} | {mo:<7} | {de:<7}")

if __name__ == "__main__":
    main()
