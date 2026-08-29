"""Automated run summary and hypothesis test comparator for NTPC ablations."""

import csv
import os

runs = [
    ("R0: Exact Multi-Scale Regression Baseline", "runs/ntpc_r0_exact_regression/val.csv"),
    ("R1: S-DCNet Deterministic Allocation", "runs/ntpc_r1_sdc_deterministic/val.csv"),
    ("R2: Flat Dirichlet-Multinomial @ 16", "runs/ntpc_r2_flat_dm16/val.csv"),
    ("R3: Hierarchical Multinomial Tree", "runs/ntpc_r3_hierarchical_multinomial/val.csv"),
    ("R4: Neural DTM Tree Likelihood (Proposed Core)", "runs/ntpc_r4_neural_dtm_tree/val.csv"),
    ("R5: Full NTPC (DTM Tree + Dense 16->8 Auxiliary)", "runs/ntpc_r5_full_adaptive_ntpc/val.csv"),
]

best_results = {}

for name, path in runs:
    tag = name.split(":")[0]
    if not os.path.exists(path):
        print(f"\n--- {name} : Not started / no log yet ---")
        continue
    with open(path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    if not reader:
        print(f"\n--- {name} : Initializing... ---")
        continue
    best = min(reader, key=lambda x: float(x["mae"]))
    last = reader[-1]
    best_results[tag] = float(best["mae"])
    print(f"\n=======================================================")
    print(f" {name}")
    print(f"=======================================================")
    print(f" Current Progress      : Epoch {last['epoch']}")
    print(f" Best Validation Epoch : {best['epoch']}")
    print(f" Best Val MAE          : {float(best['mae']):.2f}")
    print(f" Best Val RMSE         : {float(best['rmse']):.2f}")
    print(f" Breakdown at Best MAE : Sparse={float(best.get('sparse_mae', 0.0)):.2f} | Med={float(best.get('medium_mae', 0.0)):.2f} | Dense={float(best.get('dense_mae', 0.0)):.2f}")
    print(f" Latest Val MAE (Ep {last['epoch']}) : {float(last['mae']):.2f}")

# Check Decision Rule (§33) if available
if "R4" in best_results and "R1" in best_results and "R2" in best_results:
    print(f"\n=======================================================")
    print(f" DECISION RULE CHECK (§33): R4 MUST beat R1 and R2")
    print(f"=======================================================")
    r4_mae = best_results["R4"]
    r1_mae = best_results["R1"]
    r2_mae = best_results["R2"]
    beats_r1 = r4_mae < r1_mae
    beats_r2 = r4_mae < r2_mae
    print(f" R4 (DTM Tree) MAE : {r4_mae:.2f}")
    print(f" R1 (S-DC Det) MAE : {r1_mae:.2f} -> R4 Beats R1: {'YES [✓]' if beats_r1 else 'NO [✗]'}")
    print(f" R2 (Flat DM)  MAE : {r2_mae:.2f} -> R4 Beats R2: {'YES [✓]' if beats_r2 else 'NO [✗]'}")
    if beats_r1 and beats_r2:
        print(" -> CONCLUSION: PROBABILISTIC DTM TREE HYPOTHESIS CONFIRMED!")
    else:
        print(" -> CONCLUSION: Falsification condition triggered or training still in progress.")
