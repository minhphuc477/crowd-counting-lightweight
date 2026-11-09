import json
import os
import sys
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

print("=== 1. NEW JHU ABLATIONS ===")
for name in ["exp_jhu_abl_b3_count_nb_disp10_104k", "exp_jhu_abl_d4_crop384_104k"]:
    sum_path = f"runs/benchmarks/ablations_jhu/{name}/eval_val/summary.json"
    log_path = f"runs/benchmarks/ablations_jhu/{name}/train_log.csv"
    if os.path.exists(sum_path):
        data = json.load(open(sum_path, encoding="utf-8"))
        print(f"{name}: MAE={data.get('MAE'):.2f}, RMSE={data.get('RMSE'):.2f}, Bias={data.get('Bias'):.2f}")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        best_idx = df["val_mae"].idxmin()
        print(f"   Epochs trained: {len(df)}, Best Val MAE in log: {df['val_mae'].min():.2f} at epoch {df.loc[best_idx, 'epoch']}")

print("\n=== 2. UPDATED JHU ABLATIONS ===")
for name in [
    "exp_jhu_abl_b2_no_count_loss_104k",
    "exp_jhu_abl_c2_no_neck_104k",
    "exp_jhu_abl_c3_radon_8proj_104k",
    "exp_jhu_abl_d2_maxsize_1536_104k",
    "exp_jhu_abl_d3_no_weather_aug_104k",
]:
    sum_path = f"runs/benchmarks/ablations_jhu/{name}/eval_val/summary.json"
    log_path = f"runs/benchmarks/ablations_jhu/{name}/train_log.csv"
    if os.path.exists(sum_path):
        data = json.load(open(sum_path, encoding="utf-8"))
        print(f"{name}: MAE={data.get('MAE'):.2f}, RMSE={data.get('RMSE'):.2f}, Bias={data.get('Bias'):.2f}")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        best_idx = df["val_mae"].idxmin()
        print(f"   Epochs trained: {len(df)}, Best Val MAE in log: {df['val_mae'].min():.2f} at epoch {df.loc[best_idx, 'epoch']}")

print("\n=== 3. JHU ATTRIBUTE BREAKDOWN ===")
for mode in ["pretrained", "scratch"]:
    attr_path = f"runs/benchmarks/exp_jhu_direct_{mode}_104k/attribute_breakdown.json"
    if os.path.exists(attr_path):
        data = json.load(open(attr_path, encoding="utf-8"))
        print(f"\n--- JHU {mode.upper()} ---")
        print("Overall:", data.get("overall"))
        print("Weather:")
        for w, m in data.get("weather", {}).items():
            print(f"  {w:20s}: N={m['n']:4d}, MAE={m['mae']:6.2f}, RMSE={m['rmse']:6.2f}")
        print("Distractors:")
        for d, m in data.get("distractors", {}).items():
            print(f"  {d:20s}: N={m['n']:4d}, MAE={m['mae']:6.2f}, RMSE={m['rmse']:6.2f}")

print("\n=== 4. ZERO-SHOT CROSS-DATASET TRANSFER ===")
for path in [
    "runs/eval_qnrf_on_shb/summary.json",
    "runs/eval_sha_on_shb/summary.json",
    "runs/eval_qnrf_on_ucf_cc_50/summary.json",
    "runs/eval_sha_on_ucf_cc_50/summary.json",
]:
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))
        print(f"{path}: MAE={data.get('MAE'):.2f}, RMSE={data.get('RMSE'):.2f}, Bias={data.get('Bias'):.2f}")
