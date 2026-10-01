from __future__ import annotations

import json
import math
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from rmr_core.data import CrowdManifestDataset, collate_eval, collate_train
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def create_dense_20_manifest():
    manifest_in = Path("data/sha_a_train_all.jsonl")
    manifest_out = Path("data/sha_a_train_dense_20.jsonl")
    
    records = []
    with open(manifest_in, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            records.append((len(item["points"]), line))
    
    # Sort descending by point count
    records.sort(key=lambda x: x[0], reverse=True)
    dense_20 = records[:20]

    with open(manifest_out, "w", encoding="utf-8") as f:
        for count, line in dense_20:
            f.write(line)
            
    print(f"Created {manifest_out} with {len(dense_20)} densest images.")
    print(f"GT range: {dense_20[-1][0]} to {dense_20[0][0]} (Mean GT: {sum(x[0] for x in dense_20)/20:.1f})")
    return manifest_out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Overfit Diagnostic on 20 Dense Train Images on {device}...")

    manifest_path = create_dense_20_manifest()

    output_stride = 4
    # Dataset for training with 512 crops (no scale jitter, fixed scale=1.0)
    train_ds = CrowdManifestDataset(
        str(manifest_path),
        train=True,
        crop_size=512,
        scale_range=(1.0, 1.0),
        hflip_prob=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        output_stride=output_stride,
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_train)

    # Eval dataset for exact full-image evaluation
    eval_ds = CrowdManifestDataset(str(manifest_path), train=False, output_stride=output_stride)
    eval_loader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_eval)

    # Initialize RMRv3 model with canonical v19 configuration
    cfg = RMRv3Config(
        output_stride=4,
        feature_width=32,
        backbone_name="mobilenetv4_conv_small_050.e3000_r224_in1k",
        pretrained=True,
        neck_type="aspp_lite",
        use_aspp_gap=True,
        aspp_dilations=(1, 3, 6),
        region_sizes_px=(32, 64, 128),
        region_overlap=0.5,
        include_full_image=False,
        region_head_hidden=48,
        dynamic_scale_routing=True,
        enable_solver=True,
        solver_mode="additive",
        iterations=6,
        omega=1.0,
        hurdle_head=True,
        use_barzilai_borwein=True,
    )
    model = RMRv3(cfg).to(device)

    loss_cfg = RMRv3LossConfig(
        lambda_count=1.0,
        lambda_flat_dm16=1.0,
        lambda_cell=0.5,
        lambda_region_nb=0.2,
        lambda_hurdle=0.1,
        lambda_trunc_nb=0.2,
        count_loss_mode="nb",
        cell_loss_mode="mass_weighted",
        lambda_scale_align=0.05,
        lambda_curvature=0.5,
        lambda_hard_bg=0.15,
        output_stride=4,
        dm_target="dual",
    )

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)

    print("\n" + "="*70)
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Full-Img MAE':<14} {'Full-Img Bias':<14} {'Rel Error':<12}")
    print("-" * 70)

    num_epochs = 120

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            imgs = batch["image"].to(device)
            target_ys = batch["target_y"].to(device)
            points = [p.to(device) for p in batch["points"]] if "points" in batch else None

            optimizer.zero_grad()
            out = model(imgs)
            losses = compute_rmr_v3_losses(out, target_ys, loss_cfg, points=points)
            loss = losses["total"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate on full images every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                errors = []
                biases = []
                gts = []
                for b in eval_loader:
                    item = b[0]
                    img = item["image"].unsqueeze(0).to(device)
                    target_y = item["target_y"].to(device)
                    gt = float(target_y.sum().item())
                    out = model(img)
                    pred = float(out.y.sum().item())
                    errors.append(abs(pred - gt))
                    biases.append(pred - gt)
                    gts.append(gt)

                mae = float(np.mean(errors))
                bias = float(np.mean(biases))
                mean_gt = float(np.mean(gts))
                rel_err = (mae / mean_gt) * 100

                print(f"{epoch:<8} {avg_loss:<14.4f} {mae:<14.1f} {bias:<+14.1f} {rel_err:<11.2f}%")


if __name__ == "__main__":
    main()
