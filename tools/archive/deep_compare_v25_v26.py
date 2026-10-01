"""
Deep comparative analysis between RMR-v25 and RMR-v26 suites on ShanghaiTech Part A.
"""
import json
import csv
from pathlib import Path

def load_run_data(run_name):
    rd = Path("runs/sha_a") / run_name
    res = {
        "run": run_name,
        "exists": rd.exists(),
        "direct_mae": None,
        "direct_rmse": None,
        "direct_sparse": None,
        "direct_mod": None,
        "direct_dense": None,
        "direct_bias": None,
        "tta_mae": None,
        "tta_rmse": None,
        "tta_sparse": None,
        "tta_mod": None,
        "tta_dense": None,
        "tta_bias": None,
        "ci95": None,
        "epochs": 0,
        "best_epoch": 0,
        "best_val_mae_curve": 999.0
    }
    if not rd.exists():
        return res

    # Direct validation summary
    val_json = rd / "eval_val" / "summary.json"
    if val_json.is_file():
        with open(val_json, "r", encoding="utf-8") as f:
            d = json.load(f)
        res["direct_mae"] = d.get("mae", d.get("MAE"))
        res["direct_rmse"] = d.get("rmse", d.get("RMSE"))
        res["direct_sparse"] = d.get("mae_sparse", d.get("density_sparse_mae"))
        res["direct_mod"] = d.get("mae_moderate", d.get("density_medium_mae"))
        res["direct_dense"] = d.get("mae_dense", d.get("density_dense_mae"))
        res["direct_bias"] = d.get("bias", d.get("Bias"))
        res["ci95"] = d.get("mae_ci95")

    # TTA summary
    for cand in rd.iterdir():
        if cand.is_dir() and "tta" in cand.name:
            t_json = cand / "summary.json"
            if t_json.is_file():
                with open(t_json, "r", encoding="utf-8") as f:
                    td = json.load(f)
                res["tta_mae"] = td.get("mae", td.get("MAE"))
                res["tta_rmse"] = td.get("rmse", td.get("RMSE"))
                res["tta_sparse"] = td.get("mae_sparse", td.get("density_sparse_mae"))
                res["tta_mod"] = td.get("mae_moderate", td.get("density_medium_mae"))
                res["tta_dense"] = td.get("mae_dense", td.get("density_dense_mae"))
                res["tta_bias"] = td.get("bias", td.get("Bias"))
                break

    # Training log
    tl = rd / "train_log.csv"
    if tl.is_file():
        with open(tl, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                res["epochs"] += 1
                try:
                    v = float(row.get("val_mae", 999.0))
                    ep = int(row.get("epoch", 0))
                    if v < res["best_val_mae_curve"]:
                        res["best_val_mae_curve"] = v
                        res["best_epoch"] = ep
                except (ValueError, TypeError):
                    pass
    return res

v25_runs = [
    "rmr_v25_canonical",
    "rmr_v25_ablation_no_bb",
    "rmr_v25_ablation_no_perspective",
    "rmr_v25_ablation_no_curvature",
    "rmr_v25_ablation_no_morozov",
    "rmr_v25_control_no_solver",
]

v26_runs = [
    "rmr_v26_canonical",
    "rmr_v26_ablation_no_elevation",
    "rmr_v26_ablation_no_bb",
    "rmr_v26_ablation_no_morozov",
    "rmr_v26_ablation_with_curv01",
    "rmr_v26_control_no_solver",
]

print("=" * 125)
print(f"{'RMR-v25 Run':<32} | {'Direct MAE':<10} | {'TTA MAE':<10} | {'RMSE':<8} | {'Sparse':<8} | {'Mod':<8} | {'Dense':<8} | {'Bias':<8} | {'Best Ep':<7}")
print("-" * 125)
for r in v25_runs:
    d = load_run_data(r)
    d_mae = f"{d['direct_mae']:.2f}" if d['direct_mae'] else "-"
    t_mae = f"{d['tta_mae']:.2f}" if d['tta_mae'] else "-"
    rmse = f"{d['tta_rmse']:.2f}" if d['tta_rmse'] else (f"{d['direct_rmse']:.2f}" if d['direct_rmse'] else "-")
    sp = f"{d['tta_sparse']:.2f}" if d['tta_sparse'] else "-"
    md = f"{d['tta_mod']:.2f}" if d['tta_mod'] else "-"
    ds = f"{d['tta_dense']:.2f}" if d['tta_dense'] else "-"
    b = f"{d['tta_bias']:+.2f}" if d['tta_bias'] else "-"
    print(f"{r:<32} | {d_mae:<10} | {t_mae:<10} | {rmse:<8} | {sp:<8} | {md:<8} | {ds:<8} | {b:<8} | {d['best_epoch']:<7}")

print("=" * 125)
print(f"\n{'RMR-v26 Run':<32} | {'Direct MAE':<10} | {'TTA MAE':<10} | {'RMSE':<8} | {'Sparse':<8} | {'Mod':<8} | {'Dense':<8} | {'Bias':<8} | {'Best Ep':<7}")
print("-" * 125)
for r in v26_runs:
    d = load_run_data(r)
    d_mae = f"{d['direct_mae']:.2f}" if d['direct_mae'] else "-"
    t_mae = f"{d['tta_mae']:.2f}" if d['tta_mae'] else "-"
    rmse = f"{d['tta_rmse']:.2f}" if d['tta_rmse'] else (f"{d['direct_rmse']:.2f}" if d['direct_rmse'] else "-")
    sp = f"{d['tta_sparse']:.2f}" if d['tta_sparse'] else "-"
    md = f"{d['tta_mod']:.2f}" if d['tta_mod'] else "-"
    ds = f"{d['tta_dense']:.2f}" if d['tta_dense'] else "-"
    b = f"{d['tta_bias']:+.2f}" if d['tta_bias'] else "-"
    print(f"{r:<32} | {d_mae:<10} | {t_mae:<10} | {rmse:<8} | {sp:<8} | {md:<8} | {ds:<8} | {b:<8} | {d['best_epoch']:<7}")
print("=" * 125)
