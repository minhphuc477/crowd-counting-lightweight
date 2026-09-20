import sys
from pathlib import Path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.utils.data import DataLoader
from rmr_v3.eval import load_model_from_ckpt
from rmr_core.data import CrowdManifestDataset, collate_eval
from rmr_core.evaluation import evaluate_dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Running comparative evaluation on {device}...")

ds = CrowdManifestDataset('data/sha_a_test.jsonl', train=False, output_stride=4)
loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

for name, p in [
    ('Original v19', 'runs/sha_a/rmr_v19_canonical_isotropic/best_val_mae.pt'),
    ('v30 Step0', 'runs/sha_a/rmr_v30_step0_v19_anchor/best_val_mae.pt'),
    ('v31 Step0', 'runs/sha_a/rmr_v31_step0_v19_anchor/best_val_mae.pt'),
]:
    model, uniform, cfg, ckpt = load_model_from_ckpt(Path(p), device, use_ema=True)
    rows, summary = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        output_stride=4,
        run_tiling=False,
        forward_kwargs={"uniform_reliability": uniform},
        density_bins=(100.0, 500.0),
    )
    mae = summary['MAE']
    rmse = summary['RMSE']
    bias = summary['Bias']
    sp = summary['mae_sparse']
    mod = summary['mae_moderate']
    dense = summary['mae_dense']
    print(f'{name:<15}: MAE={mae:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}, Sparse={sp:.2f}, Mod={mod:.2f}, Dense={dense:.2f}')
