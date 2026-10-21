import json
import numpy as np
from pathlib import Path
from PIL import Image

with open("data/sha_a_test.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if Path(d["image"]).stem == "IMG_127":
            pts = np.array(d["points"])
            img_p = Path("data") / d["image"] if not Path(d["image"]).exists() else Path(d["image"])
            with Image.open(img_p) as im:
                w, h = im.size
                # Save crops of bottom (foreground people with bodies) and middle (dense crowd)
                im_bot = im.crop((0, int(h * 0.65), w, h))
                im_mid = im.crop((0, int(h * 0.35), w, int(h * 0.65)))
                im_top = im.crop((0, 0, w, int(h * 0.35)))
                
                print(f"IMG_127: Size={w}x{h}")
                # Analyze points in bot vs mid
                pts_bot = pts[pts[:, 1] >= h * 0.65]
                pts_mid = pts[(pts[:, 1] >= h * 0.35) & (pts[:, 1] < h * 0.65)]
                pts_top = pts[pts[:, 1] < h * 0.35]
                
                print(f"Points in Top (Y < 0.35H): {len(pts_top)}")
                print(f"Points in Mid (0.35H <= Y < 0.65H): {len(pts_mid)}")
                print(f"Points in Bot (Y >= 0.65H): {len(pts_bot)}")
                
                # Check distances in bot vs mid
                from scipy.spatial import KDTree
                if len(pts_bot) > 1:
                    t_bot = KDTree(pts_bot)
                    d_bot, _ = t_bot.query(pts_bot, k=2)
                    print(f"Bot kNN distance: Mean={d_bot[:, 1].mean():.1f}px, Min={d_bot[:, 1].min():.1f}px, Max={d_bot[:, 1].max():.1f}px")
                if len(pts_mid) > 1:
                    t_mid = KDTree(pts_mid)
                    d_mid, _ = t_mid.query(pts_mid, k=2)
                    print(f"Mid kNN distance: Mean={d_mid[:, 1].mean():.1f}px, Min={d_mid[:, 1].min():.1f}px, Max={d_mid[:, 1].max():.1f}px")
                    print(f"Mid points < 4px: {(d_mid[:, 1] < 4.0).sum()} ({(d_mid[:, 1] < 4.0).mean()*100:.1f}%)")
                    print(f"Mid points < 8px: {(d_mid[:, 1] < 8.0).sum()} ({(d_mid[:, 1] < 8.0).mean()*100:.1f}%)")
