import csv
from pathlib import Path
import numpy as np

runs = [
    "v63_champion_occlusion_robust_1500ep",
    "v63_champion_aspp_subpixel_1500ep",
    "v63_champion_aspp_subpixel_2000ep",
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
            if sid not in data:
                data[sid] = {"gt": gt, "preds": []}
            data[sid]["preds"].append(pred)

gt_list = []
ens2_preds = []  # occl_robust + aspp_1500
ens3_preds = []  # all 3

for sid, d in data.items():
    gt = d["gt"]
    gt_list.append(gt)
    p_occl = d["preds"][0]
    p_1500 = d["preds"][1]
    p_2000 = d["preds"][2]
    ens2_preds.append((p_occl + p_1500) / 2.0)
    ens3_preds.append((p_occl + p_1500 + p_2000) / 3.0)

gt_arr = np.array(gt_list)
p_occl_arr = np.array([d["preds"][0] for d in data.values()])
p_1500_arr = np.array([d["preds"][1] for d in data.values()])
p_2000_arr = np.array([d["preds"][2] for d in data.values()])
ens2_arr = np.array(ens2_preds)
ens3_arr = np.array(ens3_preds)

def metrics(gt, pred):
    mae = np.mean(np.abs(pred - gt))
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    sparse = np.mean(np.abs(pred[gt <= 100] - gt[gt <= 100]))
    moderate = np.mean(np.abs(pred[(gt > 100) & (gt <= 500)] - gt[(gt > 100) & (gt <= 500)]))
    dense = np.mean(np.abs(pred[gt > 500] - gt[gt > 500]))
    return mae, rmse, sparse, moderate, dense

print(f"{'Method':<35} | {'MAE':<6} | {'RMSE':<6} | {'Sparse':<6} | {'Mod':<6} | {'Dense':<6}")
print("-" * 75)
for name, p in [
    ("Occlusion-Robust Champion (Single)", p_occl_arr),
    ("ASPP Subpixel 1500ep (Single)", p_1500_arr),
    ("ASPP Subpixel 2000ep (Single)", p_2000_arr),
    ("Ensemble 2: Occl + ASPP 1500", ens2_arr),
    ("Ensemble 3: Occl + ASPP 1500 + 2000", ens3_arr),
]:
    mae, rmse, s, m, d = metrics(gt_arr, p)
    print(f"{name:<35} | {mae:<6.2f} | {rmse:<6.2f} | {s:<6.2f} | {m:<6.2f} | {d:<6.2f}")
