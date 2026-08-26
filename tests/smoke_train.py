"""
Smoke test: 2 forward+backward passes with the real train dataloader.
Run: python tests/smoke_train.py --config configs/sha.yaml
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import yaml
from torch.utils.data import DataLoader

from hpc.data.sha import ShanghaiTechDataset
from hpc.models.hpc_lite import HPCLiteSR48
from hpc.losses.criterion import HPCLossCriterion

# Import the collate fn from train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train import custom_collate_fn, build_dataset

config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else "configs/sha.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Dataset
print("Building train dataset...")
ds = build_dataset(cfg, is_train=True)
print(f"  Train dataset: {len(ds)} samples")

loader = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=custom_collate_fn,
                    num_workers=0, drop_last=False)

# Model + criterion
model = HPCLiteSR48(pretrained=False).to(device)
model.train()
l_cfg = cfg["loss"]
criterion = HPCLossCriterion(
    block_sizes=cfg["dataset"]["hnb_blocks"],
    allocation_block=cfg["dataset"].get("allocation_block", 16),
    lambda_route=float(l_cfg.get("lambda_route", 0.1)),
).to(device)

print("Running 2 smoke batches...")
for i, batch in enumerate(loader):
    if i >= 2:
        break

    print(f"\n  Batch {i+1} keys: {sorted(batch.keys())}")

    images       = batch["image"].to(device)
    gt_blocks    = {b: batch["gt_blocks"][b].to(device) for b in batch["gt_blocks"]}
    gt_z_alloc   = batch["gt_z_alloc"].to(device)
    gt_count     = batch["gt_count"].to(device)
    gt_spec      = batch.get("gt_special_mask16")
    if gt_spec is not None: gt_spec = gt_spec.to(device)
    gt_rq        = batch.get("gt_route_q")
    gt_rm        = batch.get("gt_route_mask")
    if gt_rq is not None: gt_rq = gt_rq.to(device)
    if gt_rm is not None: gt_rm = gt_rm.to(device)
    img_deg      = batch.get("image_degraded")
    deg_mask     = batch.get("has_degraded")
    if img_deg is not None: img_deg = img_deg.to(device)
    if deg_mask is not None: deg_mask = deg_mask.to(device)

    print(f"  images: {tuple(images.shape)}")
    print(f"  gt_count: {gt_count.tolist()}")
    print(f"  gt_route_q present: {gt_rq is not None}, gt_route_mask present: {gt_rm is not None}")

    d, aux = model(images, return_aux=True)
    routes8 = aux["routes8"]
    d_deg = model(img_deg) if img_deg is not None else None

    loss, ld = criterion(
        d, gt_blocks, gt_z_alloc, gt_count,
        gt_special_mask16=gt_spec,
        d_degraded=d_deg, degraded_mask=deg_mask,
        routes8=routes8, gt_route_q=gt_rq, gt_route_mask=gt_rm,
        progress=0.5,
    )
    print(f"  loss={loss.item():.4f}  route={ld['loss_route'].item():.4f}  finite={loss.isfinite().item()}")
    loss.backward()
    print(f"  backward OK")

print("\nSMOKE TEST PASSED")
