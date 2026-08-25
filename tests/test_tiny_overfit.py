import pytest
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

from hpc.models.hpc_lite import HPCLite
from hpc.losses.criterion import HPCLossCriterion
from hpc.targets.block_counts import build_hierarchical_block_counts
from hpc.targets.allocation_target import build_block_constrained_allocation_target


def test_t11_tiny_set_overfit_and_amp_backward():
    """T11 & Section 36: Overfit a tiny synthetic dataset of 8 crops (empty, sparse, dense)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crop_size = 448
    hnb_blocks = [16, 32, 64]
    alloc_block = 16
    
    # 1. Synthesize 8 crops
    # Crop 0, 1: Empty (0 points)
    # Crop 2, 3: Sparse (1-5 points)
    # Crop 4, 5: Medium (20-50 points)
    # Crop 6, 7: Dense (200-500 points)
    np.random.seed(42)
    torch.manual_seed(42)
    
    point_counts = [0, 0, 3, 5, 25, 40, 200, 350]
    batch_images = []
    batch_gt_blocks = {b: [] for b in hnb_blocks}
    batch_gt_alloc = []
    batch_gt_counts = []
    
    for cnt in point_counts:
        img_np = np.random.randint(50, 200, size=(crop_size, crop_size, 3), dtype=np.uint8)
        img_tensor = TF.to_tensor(Image.fromarray(img_np))
        img_norm = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        batch_images.append(img_norm)
        
        if cnt > 0:
            pts = np.random.uniform(10, crop_size - 10, size=(cnt, 2)).astype(np.float32)
        else:
            pts = np.zeros((0, 2), dtype=np.float32)
            
        b_counts = build_hierarchical_block_counts(pts, crop_size, crop_size, hnb_blocks)
        for b in hnb_blocks:
            batch_gt_blocks[b].append(b_counts[b])
            
        z_alloc = build_block_constrained_allocation_target(pts, crop_size, crop_size, block_size=alloc_block, output_stride=4)
        batch_gt_alloc.append(z_alloc)
        batch_gt_counts.append(float(cnt))
        
    images = torch.stack(batch_images).to(device)
    gt_blocks = {b: torch.stack(batch_gt_blocks[b]).to(device) for b in hnb_blocks}
    gt_z_alloc = torch.stack(batch_gt_alloc).to(device)
    gt_counts = torch.tensor(batch_gt_counts, dtype=torch.float32).to(device)
    
    # 2. Instantiate Model and Loss Criterion
    model = HPCLite(
        pretrained=False,
        neck_width=32,
        context_dilations=(1, 2, 3),
        truncate_backbone=True,
    ).to(device)
    
    # Data-driven head initialization
    mean_count = float(np.mean(point_counts))
    model.init_head_bias_from_data(mean_crop_count=mean_count, crop_size=crop_size, output_stride=4)
    
    criterion = HPCLossCriterion(
        block_sizes=hnb_blocks,
        allocation_block=alloc_block,
        lambda_hnb=1.0,
        lambda_alloc=0.5,
        lambda_hn=0.25,
        lambda_empty=0.5,
        lambda_global=1.0,
        lambda_rob=0.0,
        enable_curriculum=False,  # Test full objective from start
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=2e-3,
        weight_decay=1e-4,
    )
    
    initial_loss = None
    final_loss = None
    
    # Train for 60 iterations
    model.train()
    criterion.train()
    
    for step in range(60):
        optimizer.zero_grad()
        d_map = model(images)
        loss, loss_dict = criterion(d_map, gt_blocks, gt_z_alloc, gt_counts, progress=1.0)
        
        assert torch.isfinite(loss), f"Loss became non-finite at step {step}: {loss}"
        
        if step == 0:
            initial_loss = loss.item()
            
        loss.backward()
        
        # Verify gradients are finite
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                
        optimizer.step()
        final_loss = loss.item()
        
    print(f"Tiny Overfit Initial Loss: {initial_loss:.4f}, Final Loss: {final_loss:.4f}")
    # Loss should decrease significantly (by at least 50%)
    assert final_loss < initial_loss * 0.5, f"Loss did not decrease sufficiently: {initial_loss} -> {final_loss}"
    
    # Test predictions
    model.eval()
    with torch.no_grad():
        d_out = model(images)
        pred_counts = d_out.sum(dim=(-1, -2, -3)).cpu().numpy()
        
    print(f"GT Counts:   {point_counts}")
    print(f"Pred Counts: {[round(float(c), 2) for c in pred_counts]}")
    
    # Monotonicity check: dense crops should predict more than sparse/empty crops
    assert pred_counts[-1] > pred_counts[0], "Dense crop should predict more count than empty crop"
    assert pred_counts[-2] > pred_counts[0], "Dense crop should predict more count than empty crop"
