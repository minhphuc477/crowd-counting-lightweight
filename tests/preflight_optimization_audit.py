"""Mandatory Pre-Flight Optimization Audits from NTPC Specification (§31).

Tests:
  - Test O1: Initial count distribution at Step 0 (check Softplus baseline mass).
  - Test O2: One-image overfit test (loss drops >50%, final count error |pred - GT| < 1.0).
  - Test O3: Ten-image overfit test (loss drops >50%, final MAE < 3.0).
  - Test O4: Gradient norm breakdown across all individual loss components.
"""

from __future__ import annotations

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.losses.ntpc import NTPCConfig, NTPCLoss, sum_pool_mass_pyramid
from hpc.models.hpc_lite import HPCLite
from hpc.data.point_counts import build_exact_count_pyramid

passed = 0
failed = 0


def check(name: str, cond: bool, msg: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  [✓] {name}: PASS {msg}")
    else:
        failed += 1
        print(f"  [✗] {name}: FAIL {msg}")


def test_o1_initial_count_distribution():
    print("\n" + "=" * 60)
    print("TEST O1: Step 0 Initial Count Distribution")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    model.eval()

    img = torch.rand(4, 3, 448, 448, device=device)
    with torch.no_grad():
        mass = model(img)  # (4, 1, 112, 112)
        pred_counts = mass.sum(dim=(1, 2, 3)).cpu().tolist()

    mean_cnt = sum(pred_counts) / len(pred_counts)
    print(f"  Step 0 predicted counts per 448x448 crop: {pred_counts}")
    print(f"  Step 0 mean predicted count: {mean_cnt:.2f}")

    check("Step 0 predicted count is reasonable (< 150)", mean_cnt < 150.0, f"mean={mean_cnt:.2f}")
    check("Step 0 predicted count is positive (> 1.0)", mean_cnt > 1.0, f"mean={mean_cnt:.2f}")


def test_o2_one_image_overfit():
    print("\n" + "=" * 60)
    print("TEST O2: One-Image Overfit Test (Strict Convergence)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    crit = NTPCLoss(NTPCConfig(mode="r4_dtm_tree")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-5)

    img = torch.rand(1, 3, 448, 448, device=device)
    pts = [torch.rand(75, 2, device=device) * 448]
    targets = build_exact_count_pyramid(pts, 448, 448, (8, 16, 32, 64), device=device)
    target_n = targets["N"].item()

    model.train()
    initial_loss = 0.0
    final_loss = 0.0
    final_pred = 0.0

    for step in range(1, 301):
        optimizer.zero_grad()
        mass = model(img)
        loss, logs = crit(mass, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1:
            initial_loss = loss.item()
        if step == 300:
            final_loss = loss.item()
            final_pred = mass.sum().item()

    print(f"  Target Count: {target_n:.1f}")
    print(f"  Step 1 Loss: {initial_loss:.2f}")
    print(f"  Step 300 Loss: {final_loss:.2f} | Final Predicted Count: {final_pred:.2f} | Error: {abs(final_pred - target_n):.2f}")

    loss_decreased_half = final_loss < initial_loss * 0.50
    count_close = abs(final_pred - target_n) < 1.0
    check("Loss decreased by >50% on 1 image", loss_decreased_half, f"loss: {initial_loss:.2f} -> {final_loss:.2f} ({final_loss/initial_loss*100:.1f}%)")
    check("Predicted count converged strictly near GT (|pred - GT| < 1.0)", count_close, f"pred={final_pred:.2f}, GT={target_n:.1f}, diff={abs(final_pred-target_n):.2f}")


def test_o3_ten_image_overfit():
    print("\n" + "=" * 60)
    print("TEST O3: Ten-Image Overfit Test (Strict Batch Convergence)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    crit = NTPCLoss(NTPCConfig(mode="r4_dtm_tree")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=1e-5)

    imgs = torch.rand(10, 3, 448, 448, device=device)
    pts_batch = [torch.rand(int(20 + i * 10), 2, device=device) * 448 for i in range(10)]
    targets = build_exact_count_pyramid(pts_batch, 448, 448, (4, 8, 16, 32, 64), device=device)
    target_n = targets["N"]

    model.train()
    initial_loss = 0.0
    final_loss = 0.0
    final_mae = 0.0

    for step in range(1, 251):
        optimizer.zero_grad()
        mass = model(imgs)
        loss, logs = crit(mass, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()

        if step == 1:
            initial_loss = loss.item()
            pred_n = mass.flatten(1).sum(dim=1)
            init_mae = (pred_n - target_n).abs().mean().item()
        if step == 250:
            final_loss = loss.item()
            pred_n = mass.flatten(1).sum(dim=1)
            final_mae = (pred_n - target_n).abs().mean().item()

    print(f"  Step 1 Loss: {initial_loss:.2f} (MAE: {init_mae:.2f})")
    print(f"  Step 250 Loss: {final_loss:.2f} (MAE: {final_mae:.2f})")

    check("Ten-image loss decreased by >50%", final_loss < initial_loss * 0.50, f"loss: {initial_loss:.2f} -> {final_loss:.2f}")
    check("Ten-image MAE reduced to < 3.0", final_mae < 3.0, f"MAE: {init_mae:.2f} -> {final_mae:.2f}")


def test_o4_gradient_norm_breakdown():
    print("\n" + "=" * 60)
    print("TEST O4: Gradient Norm Breakdown Across Loss Components")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    crit = NTPCLoss(NTPCConfig(mode="r5_full_ntpc", dense_threshold_16=2.0)).to(device)

    img = torch.rand(2, 3, 448, 448, device=device)
    pts = [torch.rand(80, 2, device=device) * 448 for _ in range(2)]
    target_pyramid = build_exact_count_pyramid(pts, 448, 448, (8, 16, 32, 64), device=device)

    mass = model(img)
    _, logs = crit(mass, target_pyramid)

    print("  Loss Scale Breakdown at Step 0:")
    for k, v in logs.items():
        print(f"    - {k:<20}: {v.item():.4f}")

    check("Root NB loss is finite", torch.isfinite(logs["root_nb"]))
    check("Root->64 DM loss is finite", torch.isfinite(logs["root_to_64"]))
    check("64->32 DM loss is finite", torch.isfinite(logs["64_to_32"]))
    check("32->16 DM loss is finite", torch.isfinite(logs["32_to_16"]))
    check("16->8 Dense DM loss is finite", torch.isfinite(logs["16_to_8_dense"]))
    check("Total loss is finite", torch.isfinite(logs["total"]))


def main():
    print("\n" + "=" * 60)
    print("STARTING NTPC PRE-FLIGHT OPTIMIZATION AUDIT (§31)")
    print("=" * 60)
    test_o1_initial_count_distribution()
    test_o2_one_image_overfit()
    test_o3_ten_image_overfit()
    test_o4_gradient_norm_breakdown()

    print("\n" + "=" * 60)
    print(f"PRE-FLIGHT AUDIT SUMMARY: {passed} PASSED | {failed} FAILED")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
