import csv
from pathlib import Path

runs = [
    "v63_champion_aspp_subpixel_1500ep",
    "v63_champion_occlusion_robust_1500ep",
]

data = {}
for r in runs:
    csv_path = Path("runs/sha_a/rmr_v63") / r / "eval_val" / "predictions.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["id"]
            gt = float(row["gt"])
            pred = float(row["pred"])
            err = float(row["abs_err"])
            if sid not in data:
                data[sid] = {"gt": gt}
            data[sid][r + "_pred"] = pred
            data[sid][r + "_err"] = err

# Sort by GT descending to see dense images
sorted_items = sorted(data.items(), key=lambda x: x[1]["gt"], reverse=True)

print(f"{'Image ID':<10} | {'GT':<6} | {'Subpixel 1500 Pred':<18} | {'Subpixel Err':<12} | {'Occl-Robust Pred':<18} | {'Occl Err':<10} | {'Delta Err':<10}")
print("-" * 95)
for sid, vals in sorted_items[:20]:
    gt = vals["gt"]
    p1 = vals["v63_champion_aspp_subpixel_1500ep_pred"]
    e1 = vals["v63_champion_aspp_subpixel_1500ep_err"]
    p2 = vals["v63_champion_occlusion_robust_1500ep_pred"]
    e2 = vals["v63_champion_occlusion_robust_1500ep_err"]
    delta = e2 - e1  # negative means occl-robust is better
    print(
        f"{sid:<10} | {gt:<6.0f} | {p1:<18.1f} | {e1:<12.1f} | {p2:<18.1f} | {e2:<10.1f} | {delta:<+10.1f}"
    )
