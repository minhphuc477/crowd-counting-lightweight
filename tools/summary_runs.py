import csv
import os

runs = [
    ("R0: Multi-Scale Exact Regression Baseline", "runs/ntpc_r0_exact_regression/val.csv"),
    ("R1: S-DCNet Deterministic Allocation", "runs/ntpc_r1_sdc_deterministic/val.csv"),
    ("R2: Flat Dirichlet-Multinomial @ 16", "runs/ntpc_r2_flat_dm16/val.csv"),
    ("R3: Neural DTM Tree (Proposed Core)", "runs/ntpc_r3_neural_dtm_tree/val.csv"),
    ("R4: Full NTPC (DTM Tree + Dense 16->8)", "runs/ntpc_r4_full_adaptive_ntpc/val.csv"),
]

for name, path in runs:
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
    print(f"\n=======================================================")
    print(f" {name}")
    print(f"=======================================================")
    print(f" Current Progress      : Epoch {last['epoch']} / 1000")
    print(f" Best Validation Epoch : {best['epoch']}")
    print(f" Best Val MAE          : {float(best['mae']):.2f}")
    print(f" Best Val RMSE         : {float(best['rmse']):.2f}")
    print(f" Breakdown at Best MAE : Sparse={float(best['sparse_mae']):.2f} | Med={float(best['med_mae']):.2f} | Dense={float(best['dense_mae']):.2f}")
    print(f" Latest Val MAE (Ep {last['epoch']}) : {float(last['mae']):.2f}")
