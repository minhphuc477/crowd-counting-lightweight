import json
from pathlib import Path
import pandas as pd
import numpy as np

base_dir = Path("runs/sha_a/rmr_v66")
runs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

print(f"Deep Analysis of {len(runs)} V66 runs:\n")

summary_table = []
outlier_dict = {}

target_outliers = [
    # Extreme dense outliers (undercounted in previous runs)
    "IMG_8", "IMG_90", "IMG_92", "IMG_127", "IMG_150", "IMG_63", "IMG_122", "IMG_36",
    # Sparse texture false positive outliers
    "IMG_157", "IMG_111", "IMG_104", "IMG_175", "IMG_112"
]

for r in runs:
    name = r.name
    raw_sum_file = r / "eval_val" / "summary.json"
    tta_sum_file = r / "eval_sha_a_test_weighted_tta" / "summary.json"
    raw_pred_file = r / "eval_val" / "predictions.csv"
    tta_pred_file = r / "eval_sha_a_test_weighted_tta" / "predictions.csv"
    log_file = r / "train_log.csv"
    
    raw_sum = json.loads(raw_sum_file.read_text(encoding="utf-8")) if raw_sum_file.exists() else {}
    tta_sum = json.loads(tta_sum_file.read_text(encoding="utf-8")) if tta_sum_file.exists() else {}
    
    # Train log analysis
    best_ep = None
    min_val_mae = None
    if log_file.exists():
        df_log = pd.read_csv(log_file)
        if "val_mae" in df_log.columns:
            best_idx = df_log["val_mae"].idxmin()
            best_ep = int(df_log.loc[best_idx, "epoch"]) if "epoch" in df_log.columns else best_idx + 1
            min_val_mae = float(df_log.loc[best_idx, "val_mae"])

    df_raw = pd.read_csv(raw_pred_file) if raw_pred_file.exists() else None
    df_tta = pd.read_csv(tta_pred_file) if tta_pred_file.exists() else None
    
    # Extreme Dense (> 1000)
    ext_raw_mae = None
    ext_tta_mae = None
    if df_raw is not None and "gt" in df_raw.columns:
        ext_mask = df_raw["gt"] > 1000
        ext_raw_mae = float(np.mean(np.abs(df_raw.loc[ext_mask, "pred"] - df_raw.loc[ext_mask, "gt"])))
    if df_tta is not None and "gt" in df_tta.columns:
        ext_mask = df_tta["gt"] > 1000
        ext_tta_mae = float(np.mean(np.abs(df_tta.loc[ext_mask, "pred"] - df_tta.loc[ext_mask, "gt"])))
        
        # Outlier tracking
        for img_id in target_outliers:
            row = df_tta[df_tta["id"] == img_id]
            if not row.empty:
                gt = float(row["gt"].values[0])
                pred = float(row["pred"].values[0])
                diff = pred - gt
                if img_id not in outlier_dict:
                    outlier_dict[img_id] = {"gt": gt}
                short_name = name.replace("v66_", "").replace("_1500ep", "")
                outlier_dict[img_id][short_name] = (pred, diff)

    summary_table.append({
        "Model": name.replace("v66_", "").replace("_1500ep", ""),
        "Best Ep": best_ep,
        "Raw MAE": round(raw_sum.get("mae") or raw_sum.get("MAE") or 0.0, 2),
        "Raw RMSE": round(raw_sum.get("rmse") or raw_sum.get("RMSE") or 0.0, 2),
        "TTA MAE": round(tta_sum.get("mae") or tta_sum.get("MAE") or 0.0, 2),
        "TTA RMSE": round(tta_sum.get("rmse") or tta_sum.get("RMSE") or 0.0, 2),
        "TTA Bias": round(tta_sum.get("bias") or tta_sum.get("Bias") or 0.0, 2),
        "Sparse": round(tta_sum.get("mae_sparse") or 0.0, 2),
        "Moderate": round(tta_sum.get("mae_moderate") or 0.0, 2),
        "Dense": round(tta_sum.get("mae_dense") or 0.0, 2),
        "Ext Dense (>1k)": round(ext_tta_mae or 0.0, 2),
        "GAME3": round(tta_sum.get("GAME3") or 0.0, 2),
    })

df_summary = pd.DataFrame(summary_table)
print("=== V66 PERFORMANCE BENCHMARK MATRIX (182 TEST IMAGES) ===")
print(df_summary.to_string(index=False))

print("\n=== OUTLIER DETAILED AUDIT TABLE (TTA PREDICTIONS VS GROUND TRUTH) ===")
outlier_rows = []
for img_id, d in outlier_dict.items():
    row = {"Image": img_id, "GT": int(d["gt"])}
    for r in runs:
        short_name = r.name.replace("v66_", "").replace("_1500ep", "")
        if short_name in d:
            pred, diff = d[short_name]
            row[short_name] = f"{pred:.0f} ({diff:+.0f})"
    outlier_rows.append(row)
print(pd.DataFrame(outlier_rows).to_string(index=False))
