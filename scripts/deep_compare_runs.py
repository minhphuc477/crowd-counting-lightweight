from __future__ import annotations

import csv
import json
from pathlib import Path


def main():
    runs = [
        "rmr_v7_canonical",
        "rmr_v7_t6_tv",
        "rmr_v7_ablation_no_hurdle",
        "rmr_v7_multiscale_dm",
        "rmr_v7_ablation_no_ema",
        "rmr_v7_t2_fast",
        "rmr_v7_ablation_no_photo_aug",
    ]

    short_names = {
        "rmr_v7_canonical": "Canonical (T4)",
        "rmr_v7_t6_tv": "T6+TV (T6,0.02)",
        "rmr_v7_ablation_no_hurdle": "No Hurdle",
        "rmr_v7_multiscale_dm": "Multi-Scale DM",
        "rmr_v7_ablation_no_ema": "No EMA",
        "rmr_v7_t2_fast": "T2 Fast (Ep139)",
        "rmr_v7_ablation_no_photo_aug": "No PhotoAug (Ep149)",
    }

    data = {}
    for r in runs:
        p_sum = Path(f"runs/sha_a/{r}/eval_val/summary.json")
        p_log = Path(f"runs/sha_a/{r}/train_log.csv")
        if not p_sum.exists():
            continue
        with open(p_sum, "r", encoding="utf-8") as f:
            s = json.load(f)

        best_ep, best_val = "-", "-"
        final_ep, final_train_loss, final_val_mae = "-", "-", "-"
        if p_log.exists():
            with open(p_log, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                if rows:
                    final_ep = rows[-1].get("epoch", "-")
                    final_train_loss = rows[-1].get("train_loss", "-")
                    final_val_mae = rows[-1].get("val_mae", "-")
                    v_rows = [x for x in rows if x.get("val_mae")]
                    if v_rows:
                        b = min(v_rows, key=lambda x: float(x["val_mae"]))
                        best_ep = b.get("epoch", "-")
                        best_val = b.get("val_mae", "-")

        data[r] = {
            "summary": s,
            "final_ep": final_ep,
            "best_ep": best_ep,
            "best_val": best_val,
            "final_train_loss": final_train_loss,
            "final_val_mae": final_val_mae,
        }

    header = f"{'Metric':<26} | " + " | ".join(f"{short_names[r]:<16}" for r in runs if r in data)
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    metrics = [
        ("Trained / Best Epoch", lambda d: f"{d['final_ep']} / {d['best_ep']}"),
        ("Final Train Loss", lambda d: f"{float(d['final_train_loss']):.4f}" if d["final_train_loss"] != "-" else "-"),
        ("MAE (Global Mean)", lambda d: f"{float(d['summary']['MAE']):.2f}"),
        ("RMSE (Root Mean Sq)", lambda d: f"{float(d['summary']['RMSE']):.2f}"),
        ("NAE (Normalized AE)", lambda d: f"{float(d['summary']['NAE']):.3f}"),
        ("Bias (Signed Error)", lambda d: f"{float(d['summary']['Bias']):+.2f}"),
        ("Median AE", lambda d: f"{float(d['summary']['MedianAE']):.2f}"),
        ("P90 Absolute Error", lambda d: f"{float(d['summary']['P90AE']):.2f}"),
        ("P95 Absolute Error", lambda d: f"{float(d['summary']['P95AE']):.2f}"),
        ("Max Absolute Error", lambda d: f"{float(d['summary']['MaxAE']):.2f}"),
        ("--- Spatial Partition ---", lambda d: "---"),
        ("GAME-0 (Whole Image)", lambda d: f"{float(d['summary']['GAME0']):.2f}"),
        ("GAME-1 (4 blocks, 2x2)", lambda d: f"{float(d['summary']['GAME1']):.2f}"),
        ("GAME-2 (16 blocks, 4x4)", lambda d: f"{float(d['summary']['GAME2']):.2f}"),
        ("GAME-3 (64 blocks, 8x8)", lambda d: f"{float(d['summary']['GAME3']):.2f}"),
        ("--- Density Strata ---", lambda d: "---"),
        ("Sparse MAE (<=100)", lambda d: f"{float(d['summary']['mae_sparse']):.2f}"),
        ("Moderate MAE (101-500)", lambda d: f"{float(d['summary']['mae_moderate']):.2f}"),
        ("Dense MAE (>500)", lambda d: f"{float(d['summary']['mae_dense']):.2f}"),
        ("--- Solver Dynamics ---", lambda d: "---"),
        ("Solver Help %", lambda d: f"{float(d['summary']['solver_help_fraction']) * 100:.1f}%"),
        ("Solver Harm %", lambda d: f"{float(d['summary']['solver_harm_fraction']) * 100:.1f}%"),
        ("Energy Monotonic %", lambda d: f"{float(d['summary']['energy_monotonic_fraction']) * 100:.1f}%"),
        ("Mean Delta Energy", lambda d: f"{float(d['summary']['solver_delta_e_mean']):.1f}"),
        ("MAE Reg Y0 (Init)", lambda d: f"{float(d['summary']['mae_reg_y0']):.2f}"),
        ("MAE Reg Y_final", lambda d: f"{float(d['summary'].get('mae_reg_y6', d['summary'].get('mae_reg_y4', d['summary'].get('mae_reg_y2', 0)))):.2f}"),
        ("Reg Disagree Y0", lambda d: f"{float(d['summary']['reg_disagreement_y0']):.2f}"),
        ("Reg Disagree Y_final", lambda d: f"{float(d['summary'].get('reg_disagreement_y6', d['summary'].get('reg_disagreement_y4', d['summary'].get('reg_disagreement_y2', 0)))):.2f}"),
        ("--- Uncertainty & Dispersion ---", lambda d: "---"),
        ("Spearman(Var, Error)", lambda d: f"{float(d['summary']['spearman_rate_var_error']):.3f}"),
        ("Spearman(Weight, Err)", lambda d: f"{float(d['summary']['spearman_weight_error']):.3f}"),
        ("95% Coverage Interval", lambda d: f"{float(d['summary']['coverage_95']) * 100:.1f}%"),
        ("95% Calibration Gap", lambda d: f"{float(d['summary']['calib_gap_95']) * 100:+.1f}%"),
        ("Dispersion Low Sat %", lambda d: f"{float(d['summary']['dispersion_sat_low_fraction']) * 100:.1f}%"),
        ("Dispersion High Sat %", lambda d: f"{float(d['summary']['dispersion_sat_high_fraction']) * 100:.1f}%"),
    ]

    for label, fn in metrics:
        vals = []
        for r in runs:
            if r in data:
                try:
                    vals.append(f"{fn(data[r]):<16}")
                except Exception:
                    vals.append(f"{'N/A':<16}")
        print(f"{label:<26} | " + " | ".join(vals))
    print(separator)


if __name__ == "__main__":
    main()
