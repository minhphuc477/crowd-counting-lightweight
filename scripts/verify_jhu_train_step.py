#!/usr/bin/env python3
"""End-to-end verification of JHU-CROWD++ DataLoader, Model forward, and Loss backward pass."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import torch
from torch.utils.data import DataLoader

from rmr_core.data import CrowdManifestDataset, collate_train
from rmr_v3.model import RMRv3, RMRv3Config
from rmr_v3.losses import RMRv3LossConfig, compute_rmr_v3_losses


def main():
    print("[1/5] Loading configuration...")
    with open("configs/benchmarks/exp_jhu_direct_scratch_104k.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[2/5] Initializing CrowdManifestDataset for JHU...")
    dataset = CrowdManifestDataset(
        manifest=cfg["data"]["train_manifest"],
        train=True,
        data_root=cfg["data"]["data_root"],
        crop_size=cfg["data"]["crop_size"],
        max_size=cfg["data"]["max_size"],
        pad_small_images=cfg["data"]["pad_small_images"],
        scale_range=tuple(cfg["data"]["scale_range"]),
        output_stride=cfg["model"]["output_stride"],
    )
    print(f"      Train dataset length: {len(dataset)}")

    print("[3/5] Sampling a batch of 4...")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_train)
    batch = next(iter(loader))
    print(f"      Batch image shape: {batch['image'].shape}")
    print(f"      Batch target_y shape: {batch['target_y'].shape}")
    print(f"      Head counts per sample: {[len(pts) for pts in batch['points']]}")

    print("[4/5] Model forward pass...")
    model_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
    model = RMRv3(model_cfg)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model initialized: {total_params:,} parameters (Budget: <= 104,441)")
    assert total_params <= 104441, f"Parameter budget exceeded: {total_params} > 104441"

    outputs = model(batch["image"])
    print(f"      Output density map shape: {outputs['y'].shape}")

    print("[5/5] Loss calculation and backward pass...")
    loss_cfg = RMRv3LossConfig.from_dict(cfg["loss"])
    losses = compute_rmr_v3_losses(outputs, batch["target_y"], loss_cfg, points=batch["points"])
    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1 and abs(v.item()) > 1e-6:
            print(f"      {k:18s}: {v.item():.4f}")

    losses["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"      Backward successful! Total Grad Norm: {grad_norm:.4f}")
    assert not torch.isnan(losses["total"]), "Loss total is NaN!"
    print("\n[VERIFICATION PASSED] JHU-CROWD++ pipeline is 100% operational!")


if __name__ == "__main__":
    main()
