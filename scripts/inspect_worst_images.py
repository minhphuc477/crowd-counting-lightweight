import json
import csv
import sys
from pathlib import Path
from PIL import Image

manifest_path = Path("data/sha_a_test.jsonl")
items = {}
with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        sid = Path(d["image"]).stem
        items[sid] = d

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/eval_sha_a_test_weighted_notiling_tta/predictions.csv")
if not csv_path.exists():
    csv_path = Path("runs/sha_a/rmr_v63/v63_champion_occlusion_robust_1500ep/eval_val/predictions.csv")

rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        sid = r.get("id") or Path(r["image_path"]).stem
        gt = float(r.get("gt") or r["gt_count"])
        pred = float(r.get("pred") or r["pred_count"])
        err = abs(pred - gt)
        info = items.get(sid, {})
        img_p = Path(info.get("image", ""))
        if not img_p.is_absolute() and img_p.exists():
            pass
        elif not img_p.exists():
            img_p = Path("data") / img_p
        
        w, h = 0, 0
        if img_p.exists():
            with Image.open(img_p) as im:
                w, h = im.size

        rows.append({
            "id": sid,
            "gt": gt,
            "pred": pred,
            "err": err,
            "signed_err": pred - gt,
            "width": w,
            "height": h,
            "image_path": str(img_p),
        })

rows.sort(key=lambda x: x["err"], reverse=True)
total_err = sum(r["err"] for r in rows)
mae = total_err / len(rows)
print(f"=== WORST PREDICTIONS: {csv_path.name} (Total MAE: {mae:.2f}) ===")
print(f"{'Image ID':<10} | {'GT':<5} | {'Pred':<6} | {'Err':<7} | {'Resolution':<11} | {'Dens/kpx':<9} | {'Aspect':<6}")
print("-" * 68)
for r in rows[:20]:
    area = r["width"] * r["height"]
    density = r["gt"] / area * 1000 if area > 0 else 0
    aspect = r["width"] / r["height"] if r["height"] > 0 else 0
    print(f"{r['id']:<10} | {r['gt']:<5.0f} | {r['pred']:<6.1f} | {r['signed_err']:<+7.1f} | {r['width']:<4}x{r['height']:<5} | {density:<9.3f} | {aspect:<6.2f}")

top10 = sum(r["err"] for r in rows[:10])
print("-" * 68)
print(f"Top 10 total error: {top10:.1f} ({top10/total_err*100:.1f}% of entire test error)")
print(f"MAE on remaining 172 images: {(total_err - top10)/(len(rows)-10):.2f}")

