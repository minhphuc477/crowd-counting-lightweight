import json
import numpy as np
from pathlib import Path
from PIL import Image

def analyze_image_properties(img_path, points):
    with Image.open(img_path) as im:
        rgb = np.array(im.convert("RGB"))
        w, h = im.size
    
    # Check if image is grayscale (R == G == B)
    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)
    diff_rg = np.abs(r - g).mean()
    diff_rb = np.abs(r - b).mean()
    is_grayscale = (diff_rg < 1.0 and diff_rb < 1.0)
    
    # Image contrast and brightness
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mean_lum = gray.mean()
    std_lum = gray.std()
    
    pts = np.array(points)
    n_pts = len(pts)
    
    if n_pts > 1:
        # Compute k-nearest neighbor distances
        from scipy.spatial import KDTree
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=min(4, n_pts))
        # dists[:, 0] is distance to self (0)
        knn_1 = dists[:, 1].mean()
        knn_3 = dists[:, -1].mean()
        min_dist = dists[:, 1].min()
        pct_under_4px = (dists[:, 1] < 4.0).mean() * 100
        pct_under_8px = (dists[:, 1] < 8.0).mean() * 100
        
        # Spatial bounding box of points
        min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
        min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
        bbox_coverage = ((max_x - min_x) * (max_y - min_y)) / (w * h) * 100
        # Point center of mass vs image center
        com_y = pts[:, 1].mean() / h
        com_x = pts[:, 0].mean() / w
    else:
        knn_1 = knn_3 = min_dist = pct_under_4px = pct_under_8px = 0
        bbox_coverage = com_y = com_x = 0

    return {
        "width": w,
        "height": h,
        "is_grayscale": is_grayscale,
        "diff_rg": diff_rg,
        "mean_lum": mean_lum,
        "std_lum": std_lum,
        "n_pts": n_pts,
        "knn_1": knn_1,
        "knn_3": knn_3,
        "min_dist": min_dist,
        "pct_under_4px": pct_under_4px,
        "pct_under_8px": pct_under_8px,
        "bbox_cov": bbox_coverage,
        "com_y": com_y,
        "com_x": com_x,
    }

def main():
    # Analyze the 15 worst test images
    worst_ids = [
        "IMG_8", "IMG_63", "IMG_104", "IMG_90", "IMG_127", 
        "IMG_122", "IMG_2", "IMG_111", "IMG_150", "IMG_173", 
        "IMG_157", "IMG_60", "IMG_175", "IMG_14", "IMG_110", "IMG_121", "IMG_143", "IMG_116", "IMG_144", "IMG_28"
    ]
    
    items = {}
    with open("data/sha_a_test.jsonl") as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("id") or Path(d["image"]).stem
            items[sid] = d

    print("=== VISUAL & GEOMETRIC AUDIT OF TOP WORST TEST IMAGES ===")
    print(f"{'ID':<8} | {'Res':<10} | {'B/W?':<5} | {'MeanL':<5} | {'StdL':<5} | {'GT':<5} | {'kNN1':<5} | {'kNN3':<5} | {'<4px':<5} | {'<8px':<5} | {'BBox%':<5} | {'CoM_Y':<5}")
    print("-" * 95)
    for sid in worst_ids:
        if sid not in items:
            continue
        d = items[sid]
        img_p = Path("data") / d["image"] if not Path(d["image"]).exists() else Path(d["image"])
        props = analyze_image_properties(img_p, d["points"])
        bw = "YES" if props["is_grayscale"] else "no"
        print(f"{sid:<8} | {props['width']}x{props['height']:<5} | {bw:<5} | {props['mean_lum']:<5.1f} | {props['std_lum']:<5.1f} | {props['n_pts']:<5} | {props['knn_1']:<5.1f} | {props['knn_3']:<5.1f} | {props['pct_under_4px']:<4.1f}% | {props['pct_under_8px']:<4.1f}% | {props['bbox_cov']:<4.1f}% | {props['com_y']:<5.2f}")

    # Now audit the entire train and test dataset for Grayscale and Lighting statistics!
    print("\n=== DATASET-WIDE AUDIT (Train vs Test) ===")
    for split, path in [("Train", "data/sha_a_train_all.jsonl"), ("Test", "data/sha_a_test.jsonl")]:
        bw_count = 0
        total = 0
        under_4_sum = 0
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                total += 1
                img_p = Path("data") / d["image"] if not Path(d["image"]).exists() else Path(d["image"])
                props = analyze_image_properties(img_p, d["points"])
                if props["is_grayscale"]:
                    bw_count += 1
                under_4_sum += props["pct_under_4px"]
        print(f"{split} ({total} images): {bw_count} Grayscale images ({bw_count/total*100:.1f}%), Avg % points < 4px: {under_4_sum/total:.1f}%")

if __name__ == "__main__":
    main()
