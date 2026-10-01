import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from rmr_v3.eval import load_model_from_ckpt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

img_path = Path('data/part_A_final/test_data/images/IMG_92.jpg')
img = Image.open(img_path).convert('RGB')
img_t = TF.to_tensor(img).unsqueeze(0).to(device)

mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
img_norm = (img_t - mean) / std

for name, p in [
    ('v19', 'runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt'),
    ('v30', 'runs/sha_a/rmr_v30_step0_v19_anchor/best_val_mae.pt'),
    ('v31', 'runs/sha_a/rmr_v31_step0_v19_anchor/best_val_mae.pt'),
]:
    m, uniform, _, _ = load_model_from_ckpt(Path(p), device, use_ema=True)
    with torch.no_grad():
        out = m(img_norm)
        y0_sum = float(out.y0.sum().item())
        y_final_sum = float(out.y.sum().item())
        b_sum = float(out.b_region.sum().item())
        print(f"\nModel {name}:")
        print(f"  y0 sum: {y0_sum:.2f}")
        print(f"  y_final sum: {y_final_sum:.2f}")
        print(f"  Delta solver (y_final - y0): {y_final_sum - y0_sum:+.2f}")
        if hasattr(out, 'iterates') and out.iterates is not None:
            for it_idx, it_y in enumerate(out.iterates):
                print(f"    Iter {it_idx}: {float(it_y.sum().item()):.2f}")
