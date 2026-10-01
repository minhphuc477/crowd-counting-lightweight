import csv
from pathlib import Path

p_v19 = Path("runs/sha_a/rmr_v19_canonical_isotropic/eval_sha_a_test_weighted_tta/predictions.csv")
p_v25 = Path("runs/sha_a/rmr_v25_canonical/eval_sha_a_test_weighted_notiling_tta/predictions.csv")
p_v25_nc = Path("runs/sha_a/rmr_v25_ablation_no_curvature/eval_sha_a_test_weighted_notiling_tta/predictions.csv")

def load_preds(p):
    preds = {}
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_id = r.get("img_name") or r.get("image_name") or r.get("img_id") or r.get("id") or ""
            if not img_id and "image" in r:
                img_id = Path(r["image"]).name
            gt = float(r.get("gt_count", r.get("gt", 0.0)))
            pred = float(r.get("pred_count", r.get("pred", 0.0)))
            ae = abs(pred - gt)
            preds[img_id] = {"gt": gt, "pred": pred, "ae": ae}
    return preds

v19 = load_preds(p_v19)
v25 = load_preds(p_v25)
v25_nc = load_preds(p_v25_nc)

print(f"Total v19: {len(v19)}, Total v25: {len(v25)}, Total v25_NC: {len(v25_nc)}")

diffs = []
for k in v19:
    if k in v25:
        ae_19 = v19[k]["ae"]
        ae_25 = v25[k]["ae"]
        ae_nc = v25_nc.get(k, {}).get("ae", ae_25)
        diff = ae_25 - ae_19  # positive: v25 is worse, negative: v25 is better
        diff_nc = ae_nc - ae_19
        diffs.append({
            "id": k,
            "gt": v19[k]["gt"],
            "v19_pred": v19[k]["pred"],
            "v19_ae": ae_19,
            "v25_pred": v25[k]["pred"],
            "v25_ae": ae_25,
            "v25_diff": diff,
            "nc_pred": v25_nc.get(k, {}).get("pred", 0.0),
            "nc_ae": ae_nc,
            "nc_diff": diff_nc,
        })

diffs.sort(key=lambda x: x["v25_diff"], reverse=True)

print("\n" + "="*110)
print("Top 15 Images Where v25 Canonical is WORSE than v19:")
print("="*110)
print(f"{'Img Name':<20} | {'GT':<6} | {'v19 Pred':<9} | {'v19 AE':<8} | {'v25 Pred':<9} | {'v25 AE':<8} | {'Delta AE':<10} | {'NC AE':<8}")
print("-"*110)
for d in diffs[:15]:
    print(f"{d['id']:<20} | {d['gt']:<6.0f} | {d['v19_pred']:<9.1f} | {d['v19_ae']:<8.1f} | {d['v25_pred']:<9.1f} | {d['v25_ae']:<8.1f} | {d['v25_diff']:<+10.1f} | {d['nc_ae']:<8.1f}")

print("\n" + "="*110)
print("Top 15 Images Where v25 Canonical is BETTER than v19:")
print("="*110)
print(f"{'Img Name':<20} | {'GT':<6} | {'v19 Pred':<9} | {'v19 AE':<8} | {'v25 Pred':<9} | {'v25 AE':<8} | {'Delta AE':<10} | {'NC AE':<8}")
print("-"*110)
for d in diffs[-15:]:
    print(f"{d['id']:<20} | {d['gt']:<6.0f} | {d['v19_pred']:<9.1f} | {d['v19_ae']:<8.1f} | {d['v25_pred']:<9.1f} | {d['v25_ae']:<8.1f} | {d['v25_diff']:<+10.1f} | {d['nc_ae']:<8.1f}")

# Group by density bins
sparse = [d for d in diffs if d["gt"] <= 100]
mod = [d for d in diffs if 100 < d["gt"] <= 500]
dense = [d for d in diffs if d["gt"] > 500]

print("\n" + "="*90)
print("DENSITY REGIME BREAKDOWN: MEAN ABSOLUTE ERROR COMPARISON")
print("="*90)
print(f"{'Regime':<20} | {'N':<4} | {'v19 MAE':<9} | {'v25 MAE':<9} | {'v25_NC MAE':<11} | {'Delta v25-v19':<15}")
print("-"*90)
for name, group in [("Sparse (<=100)", sparse), ("Moderate (101-500)", mod), ("Dense (>500)", dense), ("All (Overall)", diffs)]:
    m19 = sum(d["v19_ae"] for d in group) / len(group)
    m25 = sum(d["v25_ae"] for d in group) / len(group)
    mnc = sum(d["nc_ae"] for d in group) / len(group)
    delta = m25 - m19
    print(f"{name:<20} | {len(group):<4} | {m19:<9.2f} | {m25:<9.2f} | {mnc:<11.2f} | {delta:<+15.2f}")
