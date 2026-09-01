"""Summarize R0--R5 evidence plus the final R6 NPAC run."""

from __future__ import annotations

import argparse
import csv
import os


RUNS = [
    ("R0: Exact Multi-Scale Regression Baseline", "ntpc_r0_exact_regression/val.csv"),
    ("R1: S-DCNet Deterministic Allocation", "ntpc_r1_sdc_deterministic/val.csv"),
    ("R2: Flat Dirichlet-Multinomial @ 16", "ntpc_r2_flat_dm16/val.csv"),
    ("R3: Hierarchical Multinomial Tree", "ntpc_r3_hierarchical_multinomial/val.csv"),
    ("R4: Neural DTM Tree Likelihood (Proposed Core)", "ntpc_r4_neural_dtm_tree/val.csv"),
    ("R5: Full NTPC (DTM Tree + Dense 16->8 Auxiliary)", "ntpc_r5_full_adaptive_ntpc/val.csv"),
    ("R6: NPAC (Full C32 + NB/Flat-DM16)", "ntpc_r6_npac/val.csv"),
]


def summarize(runs_root: str = "runs") -> dict[str, float]:
    """Print run status and return the best selection MAE for each available mode."""
    best_results: dict[str, float] = {}
    for name, relative_path in RUNS:
        path = os.path.join(runs_root, relative_path)
        tag = name.split(":")[0]
        if not os.path.exists(path):
            print(f"\n--- {name} : Not started / no log yet ---")
            continue
        with open(path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            print(f"\n--- {name} : Initializing... ---")
            continue
        best = min(rows, key=lambda row: float(row["mae"]))
        last = rows[-1]
        best_results[tag] = float(best["mae"])
        print("\n=======================================================")
        print(f" {name}")
        print("=======================================================")
        print(f" Current Progress      : Epoch {last['epoch']}")
        print(f" Best Selection Epoch  : {best['epoch']}")
        print(f" Best Selection MAE    : {float(best['mae']):.2f}")
        print(f" Best Selection RMSE   : {float(best['rmse']):.2f}")
        print(
            " Breakdown at Best MAE : "
            f"Sparse={float(best.get('bin_sparse_mae', best.get('sparse_mae', float('nan')))):.2f} | "
            f"Med={float(best.get('bin_medium_mae', best.get('medium_mae', float('nan')))):.2f} | "
            f"Dense={float(best.get('bin_dense_mae', best.get('dense_mae', float('nan')))):.2f}"
        )
        print(f" Latest MAE (Ep {last['epoch']}) : {float(last['mae']):.2f}")

    if {"R1", "R2", "R4"} <= best_results.keys():
        print("\n=======================================================")
        print(" DECISION RULE CHECK: R4 MUST beat R1 and R2")
        print("=======================================================")
        r4_mae = best_results["R4"]
        r1_mae = best_results["R1"]
        r2_mae = best_results["R2"]
        beats_r1 = r4_mae < r1_mae
        beats_r2 = r4_mae < r2_mae
        print(f" R4 (DTM Tree) MAE : {r4_mae:.2f}")
        print(f" R1 (S-DC Det) MAE : {r1_mae:.2f} -> R4 Beats R1: {beats_r1}")
        print(f" R2 (Flat DM) MAE  : {r2_mae:.2f} -> R4 Beats R2: {beats_r2}")
        if beats_r1 and beats_r2:
            print(" -> PASSES first-pass falsification; proceed to later confirmation seeds.")
        else:
            print(" -> Falsification condition triggered or training is incomplete.")
    return best_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()
    summarize(args.runs_root)


if __name__ == "__main__":
    main()
