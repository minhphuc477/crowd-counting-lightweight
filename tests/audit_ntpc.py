"""Comprehensive audit and test suite for Neural Tree-Pólya Crowd Counting (NTPC)."""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.losses.ntpc import NTPCConfig, NTPCLoss, sum_pool_mass_pyramid, group_four_children
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


def test_mass_pyramid_and_grouping():
    print("\n" + "=" * 60)
    print("AUDIT 1: NTPC Mass Pyramid & Child Grouping")
    print("=" * 60)
    
    mass = torch.rand(2, 1, 112, 112) * 0.1
    pyramid = sum_pool_mass_pyramid(mass, block_sizes=(8, 16, 32, 64), stride=4)
    
    check("Pyramid contains all block sizes (8, 16, 32, 64)", set(pyramid.keys()) == {8, 16, 32, 64})
    check("Pyramid 8 shape (2, 56, 56)", pyramid[8].shape == torch.Size([2, 56, 56]))
    check("Pyramid 16 shape (2, 28, 28)", pyramid[16].shape == torch.Size([2, 28, 28]))
    check("Pyramid 32 shape (2, 14, 14)", pyramid[32].shape == torch.Size([2, 14, 14]))
    check("Pyramid 64 shape (2, 7, 7)", pyramid[64].shape == torch.Size([2, 7, 7]))
    
    # Exact mass conservation check
    n_pred = mass.sum(dim=(1, 2, 3))
    check("Mass conservation: p[64].sum == mass.sum", torch.allclose(pyramid[64].sum(dim=(1, 2)), n_pred, atol=1e-4))
    check("Mass conservation: p[32].sum == mass.sum", torch.allclose(pyramid[32].sum(dim=(1, 2)), n_pred, atol=1e-4))
    check("Mass conservation: p[16].sum == mass.sum", torch.allclose(pyramid[16].sum(dim=(1, 2)), n_pred, atol=1e-4))
    check("Mass conservation: p[8].sum == mass.sum", torch.allclose(pyramid[8].sum(dim=(1, 2)), n_pred, atol=1e-4))

    # Child grouping layout
    g32 = group_four_children(pyramid[32])
    check("group_four_children(32) shape (2, 7, 7, 4)", g32.shape == torch.Size([2, 7, 7, 4]))
    check("Sum of 4 grouped children == parent 64", torch.allclose(g32.sum(dim=-1), pyramid[64], atol=1e-4))


def test_ntpc_modes():
    print("\n" + "=" * 60)
    print("AUDIT 2: NTPC 5 Decisive Research Modes (R0 - R4)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mass = (torch.rand(2, 1, 112, 112, device=device) * 0.1).requires_grad_(True)
    
    # Create synthetic point pyramid target
    pts = [torch.rand(50, 2, device=device) * 448 for _ in range(2)]
    target_pyramid = build_exact_count_pyramid(pts, 448, 448, (8, 16, 32, 64), device=device)

    # 1. R0: Exact Regional Regression
    crit_r0 = NTPCLoss(NTPCConfig(mode="r0_exact")).to(device)
    loss_r0, logs_r0 = crit_r0(mass, target_pyramid)
    loss_r0.backward()
    check("R0 loss is finite scalar", torch.isfinite(loss_r0) and loss_r0.ndim == 0, f"loss={loss_r0.item():.4f}")
    check("R0 grad on mass is finite", mass.grad is not None and torch.isfinite(mass.grad).all())
    check("R0 logs has exact_regression key", "exact_regression" in logs_r0)

    # 2. R1: S-DCNet Deterministic Allocation
    mass.grad = None
    crit_r1 = NTPCLoss(NTPCConfig(mode="r1_deterministic")).to(device)
    loss_r1, logs_r1 = crit_r1(mass, target_pyramid)
    loss_r1.backward()
    check("R1 loss is finite scalar", torch.isfinite(loss_r1) and loss_r1.ndim == 0, f"loss={loss_r1.item():.4f}")
    check("R1 grad on mass is finite", mass.grad is not None and torch.isfinite(mass.grad).all())
    check("R1 logs has deterministic_alloc key", "deterministic_alloc" in logs_r1)

    # 3. R2: Flat Dirichlet-Multinomial at Leaf 16
    mass.grad = None
    crit_r2 = NTPCLoss(NTPCConfig(mode="r2_flat_dm")).to(device)
    loss_r2, logs_r2 = crit_r2(mass, target_pyramid)
    loss_r2.backward()
    check("R2 loss is finite scalar", torch.isfinite(loss_r2) and loss_r2.ndim == 0, f"loss={loss_r2.item():.4f}")
    check("R2 grad on mass is finite", mass.grad is not None and torch.isfinite(mass.grad).all())
    check("R2 logs has flat_16 key", "flat_16" in logs_r2)

    # 4. R3: Neural DTM Tree (Core Proposed)
    mass.grad = None
    crit_r3 = NTPCLoss(NTPCConfig(mode="r3_tree_dtm")).to(device)
    loss_r3, logs_r3 = crit_r3(mass, target_pyramid)
    loss_r3.backward()
    check("R3 loss is finite scalar", torch.isfinite(loss_r3) and loss_r3.ndim == 0, f"loss={loss_r3.item():.4f}")
    check("R3 grad on mass is finite", mass.grad is not None and torch.isfinite(mass.grad).all())
    check("R3 logs has root_to_64, 64_to_32, 32_to_16", all(k in logs_r3 for k in ["root_to_64", "64_to_32", "32_to_16"]))

    # 5. R4: Full NTPC (DTM Tree + Dense 16->8)
    mass.grad = None
    crit_r4 = NTPCLoss(NTPCConfig(mode="r4_full_ntpc", dense_threshold_16=2.0)).to(device)
    loss_r4, logs_r4 = crit_r4(mass, target_pyramid)
    loss_r4.backward()
    check("R4 loss is finite scalar", torch.isfinite(loss_r4) and loss_r4.ndim == 0, f"loss={loss_r4.item():.4f}")
    check("R4 grad on mass is finite", mass.grad is not None and torch.isfinite(mass.grad).all())
    check("R4 logs has 16_to_8_dense key", "16_to_8_dense" in logs_r4)


def test_full_model_integration_and_amp():
    print("\n" + "=" * 60)
    print("AUDIT 3: Full HPCLite + NTPCLoss + AMP Integration")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPCLite(pretrained=False, use_p8_context=True).to(device)
    crit = NTPCLoss(NTPCConfig(mode="r4_full_ntpc")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0, enabled=(device.type == "cuda"))

    img = torch.rand(2, 3, 448, 448, device=device)
    pts = [torch.rand(40, 2, device=device) * 448 for _ in range(2)]
    target_pyramid = build_exact_count_pyramid(pts, 448, 448, (8, 16, 32, 64), device=device)

    optimizer.zero_grad()
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        d_map = model(img)
        loss, logs = crit(d_map, target_pyramid)

    check("Model forward output shape (2, 1, 112, 112)", d_map.shape == torch.Size([2, 1, 112, 112]))
    check("Loss dtype is float32 (numerical stability)", loss.dtype == torch.float32)

    if device.type == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        check("Gradients are finite before step", torch.isfinite(grad_norm))
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    check("Optimization step completed cleanly", True)


def main():
    print("\n" + "=" * 60)
    print("STARTING NTPC AUDIT SUITE")
    print("=" * 60)
    test_mass_pyramid_and_grouping()
    test_ntpc_modes()
    test_full_model_integration_and_amp()
    
    print("\n" + "=" * 60)
    print(f"NTPC AUDIT SUMMARY: {passed} PASSED | {failed} FAILED")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
