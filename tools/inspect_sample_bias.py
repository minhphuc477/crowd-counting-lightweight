import csv
from pathlib import Path

v70_csv = Path("runs/sha_a/rmr_v70/v70_track_b_exp4b_no_solver/eval_val/predictions.csv")
v71_1_csv = Path("runs/sha_a/rmr_v71/v71_track_b_exp1_pure_radon_ot_l1_sub50/eval_val/predictions.csv")
v71_4_csv = Path("runs/sha_a/rmr_v71/v71_track_b_exp4_full_synergy_sub50/eval_val/predictions.csv")

def load_rows(p):
    with open(p) as f:
        return {r["id"]: (float(r["gt"]), float(r["pred"])) for r in csv.DictReader(f)}

r70 = load_rows(v70_csv)
r71_1 = load_rows(v71_1_csv)
r71_4 = load_rows(v71_4_csv)

print(f"{'Image ID':<10} | {'GT':<8} | {'V70 Exp4b':<12} | {'V71 Exp1':<12} | {'V71 Exp4':<12} | {'Diff V71-V70':<12}")
print("-" * 75)

# Pick some sparse/moderate images
sample_ids = ["IMG_1", "IMG_5", "IMG_10", "IMG_20", "IMG_35", "IMG_50", "IMG_75", "IMG_100", "IMG_120", "IMG_150"]
for sid in sample_ids:
    if sid not in r70: continue
    gt = r70[sid][0]
    p70 = r70[sid][1]
    p1 = r71_1[sid][1]
    p4 = r71_4[sid][1]
    print(f"{sid:<10} | {gt:<8.0f} | {p70:<12.1f} | {p1:<12.1f} | {p4:<12.1f} | {p4 - p70:<+12.1f}")
