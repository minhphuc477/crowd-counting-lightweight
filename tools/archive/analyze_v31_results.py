import json
from pathlib import Path
import pandas as pd

runs = {
    "Step 0 (v19 Anchor)": "runs/sha_a/rmr_v31_step0_v19_anchor",
    "H1 (Stride-2 Area Norm)": "runs/sha_a/rmr_v31_h1_stride2_area_norm",
    "H2 (Density-Gated Anscombe)": "runs/sha_a/rmr_v31_h2_density_gated_anscombe",
    "H3 (Anisotropic SIRT)": "runs/sha_a/rmr_v31_h3_anisotropic_sirt",
    "H4 (Composite Sub-50)": "runs/sha_a/rmr_v31_h4_composite_sub50",
}

records = []
for name, p_str in runs.items():
    p = Path(p_str)
    sum_file = p / "eval_val" / "summary.json"
    if not sum_file.exists():
        continue
    with open(sum_file, "r") as f:
        data = json.load(f)
    
    # Also read train_log.csv for best epoch
    csv_file = p / "train_log.csv"
    best_epoch = None
    min_val_mae = None
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        if "val_mae" in df.columns:
            idx = df["val_mae"].idxmin()
            best_epoch = df.loc[idx, "epoch"]
            min_val_mae = df.loc[idx, "val_mae"]
    
    rec = {
        "Variant": name,
        "Val MAE": data.get("MAE", data.get("mae")),
        "RMSE": data.get("RMSE", data.get("rmse")),
        "Bias": data.get("Bias", data.get("bias")),
        "Sparse MAE": data.get("mae_sparse"),
        "Medium MAE": data.get("mae_moderate"),
        "Dense MAE": data.get("mae_dense"),
        "Min Val MAE (CSV)": min_val_mae,
        "Best Epoch": best_epoch,
    }
    records.append(rec)

res_df = pd.DataFrame(records)
print(res_df.to_string(index=False))
