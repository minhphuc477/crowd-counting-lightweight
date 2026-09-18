import json
from pathlib import Path

runs = [
    ("v19 Canonical", "runs/sha_a/rmr_v19_canonical_isotropic/eval_sha_a_test_weighted_tta/summary.json"),
    ("v24 Canonical (ABB+Diff)", "runs/sha_a/rmr_v24_canonical/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v24 Ablation No ABB", "runs/sha_a/rmr_v24_ablation_no_abb/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Control (No Solver)", "runs/sha_a/rmr_v25_control_no_solver/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Ablation No Persp", "runs/sha_a/rmr_v25_ablation_no_perspective/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Ablation No Morozov", "runs/sha_a/rmr_v25_ablation_no_morozov/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Ablation No BB", "runs/sha_a/rmr_v25_ablation_no_bb/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Canonical (BB1+Persp)", "runs/sha_a/rmr_v25_canonical/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Ablation No Curv", "runs/sha_a/rmr_v25_ablation_no_curvature/eval_sha_a_test_weighted_notiling_tta/summary.json"),
    ("v25 Model Soup (Can+NoBB)", "runs/sha_a/rmr_v25_canonical/eval_sha_a_test_weighted_notiling_tta/summary.json"),
]

print("=" * 115)
print("             RMR MASTER CROSS-GENERATIONAL BENCHMARK SUMMARY (SHANGHAITECH PART A - TTA)")
print("=" * 115)
print(f"{'Model / Configuration':<28} | {'MAE':<7} | {'RMSE':<7} | {'Sparse':<7} | {'Mod':<7} | {'Dense':<7} | {'Bias':<7} | {'95% CI MAE':<16}")
print("-" * 115)

for name, p in runs:
    path = Path(p)
    if not path.exists():
        print(f"{name:<28} | File not found: {p}")
        continue
    with open(path, "r") as f:
        s = json.load(f)
    mae_val = s.get("MAE", s.get("mae", 0.0))
    rmse_val = s.get("RMSE", s.get("rmse", 0.0))
    sp_val = s.get("mae_sparse", s.get("density_sparse_mae", 0.0))
    md_val = s.get("mae_moderate", s.get("density_medium_mae", 0.0))
    ds_val = s.get("mae_dense", s.get("density_dense_mae", 0.0))
    bs_val = s.get("Bias", s.get("bias", 0.0))
    ci = s.get("mae_ci95", None)
    ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "N/A"
    print(f"{name:<28} | {mae_val:<7.2f} | {rmse_val:<7.2f} | {sp_val:<7.2f} | {md_val:<7.2f} | {ds_val:<7.2f} | {bs_val:<+7.2f} | {ci_str:<16}")
print("=" * 115)
