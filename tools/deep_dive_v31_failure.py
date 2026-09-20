import json
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

runs = {
    "Step 0 (v19 Anchor)": "runs/sha_a/rmr_v31_step0_v19_anchor",
    "H1 (Stride-2 Area Norm)": "runs/sha_a/rmr_v31_h1_stride2_area_norm",
    "H2 (Density-Gated Anscombe)": "runs/sha_a/rmr_v31_h2_density_gated_anscombe",
    "H3 (Anisotropic SIRT)": "runs/sha_a/rmr_v31_h3_anisotropic_sirt",
    "H4 (Composite Sub-50)": "runs/sha_a/rmr_v31_h4_composite_sub50",
}

print("="*90)
print(f"{'Variant':<28} | {'Best Epoch':<10} {'Best Val':<10} {'Final Val':<10} {'Best Dense':<10} {'Final Dense':<11} {'Train Loss (Best)':<16}")
print("-" * 90)

for name, p_str in runs.items():
    p = Path(p_str)
    csv_file = p / "train_log.csv"
    if not csv_file.exists():
        continue
    df = pd.read_csv(csv_file)
    val_df = df.dropna(subset=["val_mae"]).copy()
    if val_df.empty:
        continue
    
    best_idx = val_df["val_mae"].idxmin()
    best_row = val_df.loc[best_idx]
    final_row = val_df.iloc[-1]
    
    b_ep = int(best_row["epoch"])
    b_val = best_row["val_mae"]
    f_val = final_row["val_mae"]
    b_dense = best_row["val_mae_dense"]
    f_dense = final_row["val_mae_dense"]
    t_loss = best_row["train_total"]
    
    print(f"{name:<28} | {b_ep:<10} {b_val:<10.2f} {f_val:<10.2f} {b_dense:<10.2f} {f_dense:<11.2f} {t_loss:<16.4f}")

print("\n" + "="*90)
print("CONFIG DIFFERENCES FROM STEP 0 ANCHOR:")
print("="*90)

with open(Path(runs["Step 0 (v19 Anchor)"]) / "resolved_config.yaml", "r") as f:
    cfg_base = yaml.safe_load(f)

for name, p_str in runs.items():
    if name == "Step 0 (v19 Anchor)":
        continue
    cfg_path = Path(p_str) / "resolved_config.yaml"
    with open(cfg_path, "r") as f:
        cfg_curr = yaml.safe_load(f)
    
    print(f"\n>>> Differences for {name}:")
    for sec in ["model", "loss", "data", "train"]:
        s_base = cfg_base.get(sec, {})
        s_curr = cfg_curr.get(sec, {})
        all_keys = set(s_base.keys()).union(set(s_curr.keys()))
        diffs = []
        for k in sorted(all_keys):
            v_b = s_base.get(k)
            v_c = s_curr.get(k)
            if v_b != v_c:
                diffs.append((k, v_b, v_c))
        if diffs:
            print(f"  [{sec}]")
            for k, vb, vc in diffs:
                print(f"    {k}: {vb} -> {vc}")
