import json
import numpy as np
from pathlib import Path
from PIL import Image

worst_ids = ["IMG_63", "IMG_90", "IMG_127", "IMG_8", "IMG_104", "IMG_111", "IMG_157", "IMG_2"]

items = {}
with open("data/sha_a_test.jsonl") as f:
    for line in f:
        d = json.loads(line)
        sid = d.get("id") or Path(d["image"]).stem
        items[sid] = d

print("=== DEEP VISUAL AUTOPSY OF TARGET IMAGES ===")
for sid in worst_ids:
    d = items[sid]
    img_p = Path("data") / d["image"] if not Path(d["image"]).exists() else Path(d["image"])
    with Image.open(img_p) as im:
        rgb = np.array(im.convert("RGB"))
        w, h = im.size
    
    pts = np.array(d["points"])
    n = len(pts)
    
    # Check grayscale
    r, g, b = rgb[:, :, 0].astype(float), rgb[:, :, 1].astype(float), rgb[:, :, 2].astype(float)
    is_bw = (np.abs(r - g).mean() < 1.0 and np.abs(r - b).mean() < 1.0)
    
    # Analyze where people are and local image patches
    from scipy.spatial import KDTree
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    min_dist_to_neighbor = dists[:, 1]
    
    # Group people into 3 distance strata: dense (d < 6px), medium (6 <= d < 15px), sparse (d >= 15px)
    dense_pts = (min_dist_to_neighbor < 6.0).sum()
    med_pts = ((min_dist_to_neighbor >= 6.0) & (min_dist_to_neighbor < 15.0)).sum()
    sparse_pts = (min_dist_to_neighbor >= 15.0).sum()
    
    # Average local patch brightness around heads
    head_lum = []
    bg_lum = []
    for pt in pts[:min(100, n)]:
        px, py = int(round(pt[0])), int(round(pt[1]))
        if 5 <= px < w - 5 and 5 <= py < h - 5:
            patch = rgb[py-5:py+6, px-5:px+6].mean()
            head_lum.append(patch)
    
    # Image top vs bottom brightness
    top_half_lum = rgb[:h//2, :].mean()
    bot_half_lum = rgb[h//2:, :].mean()
    
    print(f"\n--- {sid} (GT: {n}, B/W: {is_bw}, Res: {w}x{h}) ---")
    print(f"Distance strata: Dense (<6px): {dense_pts} ({dense_pts/n*100:.1f}%), Med (6-15px): {med_pts} ({med_pts/n*100:.1f}%), Sparse (>=15px): {sparse_pts} ({sparse_pts/n*100:.1f}%)")
    print(f"Top 50% Lum: {top_half_lum:.1f} | Bottom 50% Lum: {bot_half_lum:.1f} | Delta (Bot - Top): {bot_half_lum - top_half_lum:+.1f}")
    if head_lum:
        print(f"Sample Head Patch Mean Luminance: {np.mean(head_lum):.1f} (Whole image: {rgb.mean():.1f})")
