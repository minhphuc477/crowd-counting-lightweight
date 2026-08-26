import math
import yaml
import torch
import numpy as np
from hpc.models.hpc_lite import HPCLiteSR48
from hpc.data.sha import ShanghaiTechDataset

def run_diagnostics():
    with open('configs/sha.yaml') as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cpu')
    model = HPCLiteSR48(
        pretrained=False,
        neck_width=48,
        eps_d=1e-6,
        route_temperature=1.0,
        pool_kernels=(3, 5, 7),
        pool_residual_mix=0.5,
        simam_lambda=1e-4,
    ).to(device)

    ckpt = torch.load('runs/sha/best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_dataset = ShanghaiTechDataset(
        root=cfg['dataset']['root'],
        part=cfg['dataset']['part'],
        split='test_data',
        crop_size=448,
        hnb_blocks=[16, 32, 64],
        allocation_block=16,
        is_train=False,
    )

    preds, gts = [], []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            img = sample['image'].unsqueeze(0).to(device)
            gt = float(sample['gt_count'])
            cnt, _ = model.predict(img, pad_multiple=32)
            preds.append(cnt.item())
            gts.append(gt)

    preds = np.array(preds)
    gts = np.array(gts)
    mae = np.mean(np.abs(preds - gts))
    rmse = np.sqrt(np.mean((preds - gts)**2))
    bias = np.mean(preds - gts)
    saved_epoch = ckpt.get("epoch", "unknown")

    print(f"==================================================")
    print(f" DIAGNOSTIC AUDIT OF runs/sha/best.pt (Epoch {saved_epoch})")
    print(f"==================================================")
    print(f"Overall MAE:  {mae:.2f}")
    print(f"Overall RMSE: {rmse:.2f}")
    print(f"Overall Bias: {bias:+.2f} (Pred Mean: {preds.mean():.1f} vs GT Mean: {gts.mean():.1f})")
    print(f"--------------------------------------------------")
    
    dense_mask = gts > 1000
    if dense_mask.any():
        print(f"Dense (>1000 GT, N={dense_mask.sum()}):")
        print(f"  MAE:  {np.mean(np.abs(preds[dense_mask] - gts[dense_mask])):.2f}")
        print(f"  Bias: {np.mean(preds[dense_mask] - gts[dense_mask]):+.2f}")
    
    med_mask = (gts >= 100) & (gts <= 1000)
    if med_mask.any():
        print(f"Medium (100-1000 GT, N={med_mask.sum()}):")
        print(f"  MAE:  {np.mean(np.abs(preds[med_mask] - gts[med_mask])):.2f}")
        print(f"  Bias: {np.mean(preds[med_mask] - gts[med_mask]):+.2f}")
        
    sparse_mask = gts < 100
    if sparse_mask.any():
        print(f"Sparse (<100 GT, N={sparse_mask.sum()}):")
        print(f"  MAE:  {np.mean(np.abs(preds[sparse_mask] - gts[sparse_mask])):.2f}")
        print(f"  Bias: {np.mean(preds[sparse_mask] - gts[sparse_mask]):+.2f}")
    print(f"==================================================")

if __name__ == "__main__":
    run_diagnostics()
