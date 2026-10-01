import json
from pathlib import Path

runs_dir = Path("runs/sha_a")
v_canonicals = [
    "rmr_v7_canonical", "rmr_v9_canonical", "rmr_v10_canonical_isotropic",
    "rmr_v11_canonical_dsr", "rmr_v13_native_reconstruction", "rmr_v14_unified_reconstruction",
    "rmr_v15_native_geometry", "rmr_v16_canonical", "rmr_v17_canonical",
    "rmr_v18_canonical", "rmr_v19_canonical_isotropic", "rmr_v20_canonical",
    "rmr_v21_canonical", "rmr_v22_canonical", "rmr_v23_canonical",
    "rmr_v24_canonical", "rmr_v25_canonical", "rmr_v26_canonical",
    "rmr_v27_canonical_restored", "rmr_v28_sub50_discovery", "rmr_v29_step0_v19_anchor",
    "rmr_v30_step0_v19_anchor", "rmr_v31_step0_v19_anchor", "rmr_v32_step0_v19_anchor"
]

print(f"{'Name':<35} | {'MAE':<7} | {'RMSE':<7} | {'Bias':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7}")
print("-" * 95)
for v in v_canonicals:
    p = runs_dir / v / "eval_val" / "summary.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
            m = s.get("mae", s.get("MAE", 0))
            r = s.get("rmse", s.get("RMSE", 0))
            b = s.get("bias", s.get("Bias", 0))
            sp = s.get("mae_sparse", 0)
            mo = s.get("mae_moderate", 0)
            de = s.get("mae_dense", 0)
            print(f"{v:<35} | {m:<7.2f} | {r:<7.2f} | {b:<7.2f} | {sp:<7.2f} | {mo:<7.2f} | {de:<7.2f}")
    else:
        print(f"{v:<35} | NOT FOUND")
