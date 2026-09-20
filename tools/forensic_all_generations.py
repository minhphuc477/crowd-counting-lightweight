import json
from pathlib import Path
import pandas as pd
import yaml

runs_to_check = [
    ("v10 Canonical", "runs/sha_a/rmr_v10_canonical_isotropic"),
    ("v11 Canonical", "runs/sha_a/rmr_v11_canonical_dsr"),
    ("v13 Canonical", "runs/sha_a/rmr_v13_native_reconstruction"),
    ("v14 Canonical", "runs/sha_a/rmr_v14_unified_reconstruction"),
    ("v15 Canonical", "runs/sha_a/rmr_v15_native_geometry"),
    ("v16 Canonical", "runs/sha_a/rmr_v16_canonical"),
    ("v17 Canonical", "runs/sha_a/rmr_v17_canonical"),
    ("v18 Canonical", "runs/sha_a/rmr_v18_canonical"),
    ("v19 Original", "runs/sha_a/rmr_v19_canonical_isotropic"),
    ("v20 Canonical", "runs/sha_a/rmr_v20_canonical"),
    ("v21 Canonical", "runs/sha_a/rmr_v21_canonical"),
    ("v22 Canonical", "runs/sha_a/rmr_v22_canonical"),
    ("v23 Canonical", "runs/sha_a/rmr_v23_canonical"),
    ("v24 Canonical", "runs/sha_a/rmr_v24_canonical"),
    ("v25 Canonical", "runs/sha_a/rmr_v25_canonical"),
    ("v26 Canonical", "runs/sha_a/rmr_v26_canonical"),
    ("v27 Canonical", "runs/sha_a/rmr_v27_canonical_restored"),
    ("v28 Sub50", "runs/sha_a/rmr_v28_sub50_discovery"),
    ("v29 Step0", "runs/sha_a/rmr_v29_step0_v19_anchor"),
    ("v30 Step0", "runs/sha_a/rmr_v30_step0_v19_anchor"),
    ("v31 Step0", "runs/sha_a/rmr_v31_step0_v19_anchor"),
    ("v31 H1 Stride2", "runs/sha_a/rmr_v31_h1_stride2_area_norm"),
    ("v31 H2 Anscombe", "runs/sha_a/rmr_v31_h2_density_gated_anscombe"),
    ("v31 H3 Anisotropic", "runs/sha_a/rmr_v31_h3_anisotropic_sirt"),
    ("v31 H4 Composite", "runs/sha_a/rmr_v31_h4_composite_sub50"),
]

records = []
for name, p_str in runs_to_check:
    p = Path(p_str)
    if not p.exists():
        continue
    
    # Try eval_val summary.json
    sum_file = p / "eval_val" / "summary.json"
    data = {}
    if sum_file.exists():
        with open(sum_file, "r") as f:
            data = json.load(f)
            
    # Try train_log.csv
    csv_file = p / "train_log.csv"
    best_ep = None
    min_mae = None
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
            if "val_mae" in df.columns:
                vdf = df.dropna(subset=["val_mae"])
                if not vdf.empty:
                    idx = vdf["val_mae"].idxmin()
                    best_ep = int(vdf.loc[idx, "epoch"])
                    min_mae = vdf.loc[idx, "val_mae"]
        except Exception:
            pass

    mae = data.get("MAE", data.get("mae", min_mae))
    rmse = data.get("RMSE", data.get("rmse"))
    bias = data.get("Bias", data.get("bias"))
    sp = data.get("mae_sparse")
    mod = data.get("mae_moderate")
    dense = data.get("mae_dense")
    
    records.append({
        "Generation": name,
        "Best Epoch": best_ep,
        "Val MAE": round(mae, 2) if mae is not None else None,
        "RMSE": round(rmse, 2) if rmse is not None else None,
        "Bias": round(bias, 2) if bias is not None else None,
        "Sparse": round(sp, 2) if sp is not None else None,
        "Moderate": round(mod, 2) if mod is not None else None,
        "Dense": round(dense, 2) if dense is not None else None,
    })

res_df = pd.DataFrame(records)
print(res_df.to_string(index=False))
