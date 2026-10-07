import json
from pathlib import Path

summary_file = Path("data/dataset_forensics_summary.json")
if not summary_file.exists():
    print("Summary file not found")
    exit(0)

with summary_file.open("r", encoding="utf-8") as f:
    items = json.load(f)

header = f"| {'Dataset Manifest':<22} | {'Images':>6} | {'Heads':>8} | {'Mean':>7} | {'Median':>6} | {'Res Median':>10} | {'<512px':>7} | {'>2048px':>7} | {'Sub-4px':>7} | {'1NN Med':>7} |"
sep = f"|{'-'*24}|{'-'*8}|{'-'*10}|{'-'*9}|{'-'*8}|{'-'*12}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*9}|"
print(header)
print(sep)

for it in items:
    name = it["manifest"].replace(".jsonl", "")
    n_img = it["n_images"]
    tot_h = it["total_heads"]
    mean_c = it["head_count"]["mean"]
    med_c = it["head_count"]["median"]
    res = f"{int(it['resolution']['w_median'])}x{int(it['resolution']['h_median'])}"
    lt_512 = f"{it['resolution']['pct_short_side_lt_512']:.1f}%"
    gt_2048 = f"{it['resolution']['pct_long_side_gt_2048']:.1f}%"
    sub4 = f"{it['spatial_1nn']['sub_4px_fraction_pct']:.1f}%"
    nn_med = f"{it['spatial_1nn']['median_px']:.1f}p"
    print(f"| {name:<22} | {n_img:>6} | {tot_h:>8} | {mean_c:>7.1f} | {med_c:>6.0f} | {res:>10} | {lt_512:>7} | {gt_2048:>7} | {sub4:>7} | {nn_med:>7} |")
