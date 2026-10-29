import csv
from pathlib import Path
import numpy as np

v70_csv = Path("runs/sha_a/rmr_v70/v70_track_b_exp4b_no_solver/eval_val/predictions.csv")
v71_1_csv = Path("runs/sha_a/rmr_v71/v71_track_b_exp1_pure_radon_ot_l1_sub50/eval_val/predictions.csv")
v71_3_csv = Path("runs/sha_a/rmr_v71/v71_track_b_exp3_peak_concentrated_solver/eval_val/predictions.csv")
v71_4_csv = Path("runs/sha_a/rmr_v71/v71_track_b_exp4_full_synergy_sub50/eval_val/predictions.csv")

def load_preds(p):
    data = {}
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["id"]] = {
                "gt": float(row["gt"]),
                "pred": float(row["pred"]),
                "diff": float(row["pred"]) - float(row["gt"]),
                "ae": abs(float(row["pred"]) - float(row["gt"])),
            }
    return data

p70 = load_preds(v70_csv)
p71_1 = load_preds(v71_1_csv)
p71_3 = load_preds(v71_3_csv)
p71_4 = load_preds(v71_4_csv)

bins = [
    ("Sparse (0-100)", 0, 100),
    ("Low-Mod (100-300)", 100, 300),
    ("High-Mod (300-500)", 300, 500),
    ("Dense (500-1000)", 500, 1000),
    ("Extreme Outlier (>1000)", 1000, 10000),
]

print("=" * 100)
print(f"{'Bracket':<25} | {'Count':<5} | {'V70 Exp 4b':<12} | {'V71 Exp 1':<12} | {'V71 Exp 3':<12} | {'V71 Exp 4':<12}")
print("-" * 100)

for b_name, low, high in bins:
    ids = [k for k, v in p70.items() if low < v["gt"] <= high or (low == 0 and v["gt"] == 0)]
    ae70 = np.mean([p70[i]["ae"] for i in ids])
    ae71_1 = np.mean([p71_1[i]["ae"] for i in ids])
    ae71_3 = np.mean([p71_3[i]["ae"] for i in ids])
    ae71_4 = np.mean([p71_4[i]["ae"] for i in ids])
    print(f"{b_name:<25} | {len(ids):<5} | {ae70:<12.2f} | {ae71_1:<12.2f} | {ae71_3:<12.2f} | {ae71_4:<12.2f}")

print("-" * 100)
ids_all = list(p70.keys())
print(f"{'OVERALL MAE':<25} | {len(ids_all):<5} | {np.mean([p70[i]['ae'] for i in ids_all]):<12.2f} | {np.mean([p71_1[i]['ae'] for i in ids_all]):<12.2f} | {np.mean([p71_3[i]['ae'] for i in ids_all]):<12.2f} | {np.mean([p71_4[i]['ae'] for i in ids_all]):<12.2f}")
print(f"{'NET BIAS':<25} | {len(ids_all):<5} | {np.mean([p70[i]['diff'] for i in ids_all]):<+12.2f} | {np.mean([p71_1[i]['diff'] for i in ids_all]):<+12.2f} | {np.mean([p71_3[i]['diff'] for i in ids_all]):<+12.2f} | {np.mean([p71_4[i]['diff'] for i in ids_all]):<+12.2f}")

# Count how many images regressed
regressed_1 = sum(1 for i in ids_all if p71_1[i]["ae"] > p70[i]["ae"])
regressed_4 = sum(1 for i in ids_all if p71_4[i]["ae"] > p70[i]["ae"])
print(f"\nImages regressed in V71 Exp 1 vs V70 Exp 4b: {regressed_1}/{len(ids_all)} ({regressed_1/len(ids_all)*100:.1f}%)")
print(f"Images regressed in V71 Exp 4 vs V70 Exp 4b: {regressed_4}/{len(ids_all)} ({regressed_4/len(ids_all)*100:.1f}%)")
