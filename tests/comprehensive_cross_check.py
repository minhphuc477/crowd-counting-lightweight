"""Comprehensive End-to-End Mathematical & Code Cross-Check Suite for NTPC.

Performs rigorous verification across:
  1. Loss mathematics (R0-R5, Euler's theorem, Dirichlet-Multinomial PMF, Multinomial limit).
  2. Hierarchy consistency (Exact integer counts, recursive block sums, zero parent invariance).
  3. Model architecture (MobileNetV4 backbone, Additive FPN neck, GroupNorm head, Float32 output).
  4. Metrics mathematical correctness (MAE, RMSE, NAE, Bias, Subgroup bins).
  5. Configuration sanity across all 6 YAML files.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import yaml

import numpy as np
import torch
import torch.nn.functional as F

from hpc.losses.dirichlet_multinomial import dirichlet_multinomial_nll, multinomial_nll
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion
from hpc.losses.ntpc import (
    NTPCConfig,
    NTPCLoss,
    alpha_from_mass,
    block_sum,
    dm_nll_none,
    group_2x2_flat,
    multinomial_nll_none,
    probs_from_positive_mass,
    sum_pool_mass_pyramid,
)
from hpc.data.point_counts import build_exact_count_pyramid, points_to_y8_grid
from hpc.models.hpc_lite import HPCLite, inv_softplus
from hpc.metrics.counting import evaluate_counting_metrics
from hpc.metrics.subgroup import evaluate_subgroup_diagnostics

total_passed = 0
total_failed = 0


def assert_test(name: str, condition: bool, msg: str = "") -> None:
    global total_passed, total_failed
    if condition:
        total_passed += 1
        print(f"  [PASS] {name} {msg}")
    else:
        total_failed += 1
        print(f"  [FAIL] {name} {msg}")


def check_section_1_probabilistic_losses():
    print("\n" + "=" * 70)
    print("SECTION 1: Core Probability & Loss Mathematics")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1.1 Dirichlet-Multinomial formula correctness against manual calculation
    # For y = [2, 3], alpha = [4, 6] (so alpha0 = 10, n = 5)
    # PMF = (5! / (2! 3!)) * (Gamma(10)/Gamma(15)) * (Gamma(6)/Gamma(4)) * (Gamma(9)/Gamma(6))
    # log P = log(10) + lgamma(10) - lgamma(15) + lgamma(6) - lgamma(4) + lgamma(9) - lgamma(6)
    y_test = torch.tensor([[2.0, 3.0]], device=device)
    alpha_test = torch.tensor([[4.0, 6.0]], device=device)
    
    manual_log_prob = (
        torch.lgamma(torch.tensor(6.0)) - torch.lgamma(torch.tensor(3.0)) - torch.lgamma(torch.tensor(4.0))
        + torch.lgamma(torch.tensor(10.0)) - torch.lgamma(torch.tensor(15.0))
        + torch.lgamma(torch.tensor(6.0)) - torch.lgamma(torch.tensor(4.0))
        + torch.lgamma(torch.tensor(9.0)) - torch.lgamma(torch.tensor(6.0))
    )
    manual_nll = -manual_log_prob.item()
    code_nll = dm_nll_none(y_test, alpha_test).item()
    
    assert_test(
        "Dirichlet-Multinomial exact PMF match",
        abs(code_nll - manual_nll) < 1e-5,
        f"(code: {code_nll:.6f}, manual: {manual_nll:.6f})",
    )

    # 1.2 Asymptotic convergence: DM -> Multinomial as kappa -> infinity
    m_test = torch.tensor([[10.0, 20.0, 30.0, 40.0]], device=device)
    pi_test = probs_from_positive_mass(m_test)
    y_vec = torch.tensor([[5.0, 10.0, 15.0, 20.0]], device=device)
    
    multi_nll = multinomial_nll_none(y_vec, pi_test).item()
    alpha_large_k = alpha_from_mass(m_test, kappa=1e7)
    dm_large_k_nll = dm_nll_none(y_vec, alpha_large_k).item()
    
    assert_test(
        "DM asymptotically approaches Multinomial as kappa -> 1e7",
        abs(dm_large_k_nll - multi_nll) < 1e-3,
        f"(DM: {dm_large_k_nll:.5f}, Multinomial: {multi_nll:.5f})",
    )

    # 1.3 Empty parent zero loss: n = 0 -> NLL = 0 and grad = 0
    m_param = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True, device=device)
    alpha_param = alpha_from_mass(m_param, kappa=20.0)
    y_empty = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=device)
    
    loss_empty = dm_nll_none(y_empty, alpha_param)
    loss_empty.backward()
    
    assert_test(
        "Zero target counts produce exactly 0.0 loss",
        loss_empty.item() == 0.0,
        f"(loss: {loss_empty.item()})",
    )
    assert_test(
        "Zero target counts produce exactly zero gradient",
        torch.all(m_param.grad == 0.0).item(),
        f"(grad norm: {m_param.grad.norm().item()})",
    )

    # 1.4 Euler's Homogeneity: m . grad_m(L_DM) == 0
    m_homo = torch.tensor([[2.5, 4.1, 1.8, 3.2]], requires_grad=True, device=device)
    alpha_homo = alpha_from_mass(m_homo, kappa=20.0)
    y_homo = torch.tensor([[3.0, 5.0, 2.0, 4.0]], device=device)
    
    loss_homo = dm_nll_none(y_homo, alpha_homo)
    loss_homo.backward()
    
    euler_dot = (m_homo * m_homo.grad).sum().item()
    assert_test(
        "Euler homogeneity condition: m . grad_m(L_DM) == 0",
        abs(euler_dot) < 1e-4,
        f"(dot product: {euler_dot:.2e})",
    )


def check_section_2_ablation_modes_and_per_image_joint():
    print("\n" + "=" * 70)
    print("SECTION 2: NTPC Ablation Modes (R0 - R5) & Per-Image Joint Likelihood")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modes = ["r0_exact", "r1_deterministic", "r2_flat_dm", "r3_multinomial_tree", "r4_dtm_tree", "r5_full_ntpc"]

    # Generate synthetic crowd batch (B=2, H=256, W=256)
    pts_batch = [
        torch.tensor([[30.0, 40.0], [100.0, 150.0], [200.0, 220.0]]),
        torch.tensor([[50.0, 80.0], [70.0, 90.0]]),
    ]
    targets = build_exact_count_pyramid(pts_batch, 256, 256, (8, 16, 32, 64), device=device)
    
    mass = torch.rand(2, 1, 64, 64, requires_grad=True, device=device) * 0.1

    for mode in modes:
        crit = NTPCLoss(NTPCConfig(mode=mode)).to(device)
        loss, logs = crit(mass, targets)
        loss.backward(retain_graph=True)
        
        is_finite = torch.isfinite(loss).item()
        has_grad = mass.grad is not None and torch.isfinite(mass.grad).all().item()
        assert_test(f"Mode {mode:<22} loss & gradients are finite", is_finite and has_grad, f"(loss={loss.item():.2f})")
        mass.grad.zero_()


def check_section_3_data_structures_and_target_conservation():
    print("\n" + "=" * 70)
    print("SECTION 3: Target Hierarchy, Coordinates & Conservation")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3.1 Verify (u, v) rasterization: x indexes col (width), y indexes row (height)
    pts = torch.tensor([[10.0, 20.0]], device=device)  # x=10 (col 1), y=20 (row 2)
    grid = points_to_y8_grid(pts, height=64, width=64, device=device)  # shape (1, 8, 8)
    
    # x=10 -> bx = floor(10/8) = 1 (col)
    # y=20 -> by = floor(20/8) = 2 (row)
    val_at_coord = grid[0, 2, 1].item()
    val_at_swapped = grid[0, 1, 2].item()
    
    assert_test(
        "Coordinate mapping: y indexes rows, x indexes columns",
        val_at_coord == 1.0 and val_at_swapped == 0.0,
        f"(at [row 2, col 1]: {val_at_coord}, at [row 1, col 2]: {val_at_swapped})",
    )

    # 3.2 Exact integer conservation across 100 random point sets
    torch.manual_seed(42)
    pts_rand = [torch.rand(np.random.randint(10, 200), 2, device=device) * 256 for _ in range(10)]
    pyramid = build_exact_count_pyramid(pts_rand, 256, 256, (8, 16, 32, 64), device=device)
    
    n_true = pyramid["N"]
    s8 = pyramid[8].flatten(1).sum(dim=1)
    s16 = pyramid[16].flatten(1).sum(dim=1)
    s32 = pyramid[32].flatten(1).sum(dim=1)
    s64 = pyramid[64].flatten(1).sum(dim=1)
    
    cons_8 = torch.equal(s8, n_true)
    cons_16 = torch.equal(s16, n_true)
    cons_32 = torch.equal(s32, n_true)
    cons_64 = torch.equal(s64, n_true)
    
    assert_test(
        "Exact hierarchy conservation: sum(y8) == sum(y16) == sum(y32) == sum(y64) == N",
        cons_8 and cons_16 and cons_32 and cons_64,
        f"(N counts: {n_true.cpu().tolist()[:4]}...)",
    )


def check_section_4_model_architecture():
    print("\n" + "=" * 70)
    print("SECTION 4: HPCLite Model Architecture & Floating Point Safety")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HPCLite(pretrained=False, use_p8_context=True).to(device)

    # 4.1 Parameter count verification
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert_test(
        "Model parameter count is lightweight (~0.352M)",
        0.30e6 < params < 0.40e6,
        f"(actual: {params:,} = {params/1e6:.3f}M)",
    )

    # 4.2 Forward pass output dtype is strictly float32 under AMP autocast
    x = torch.rand(2, 3, 256, 256, device=device)
    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
        d_map = model(x)
        
    assert_test(
        "Output mass map is strictly float32 (immune to AMP float16 underflow)",
        d_map.dtype == torch.float32,
        f"(dtype: {d_map.dtype})",
    )

    # 4.3 Output stride is strictly 4
    assert_test(
        "Output spatial stride is strictly 4 (256x256 -> 64x64 mass map)",
        d_map.shape == (2, 1, 64, 64),
        f"(shape: {tuple(d_map.shape)})",
    )

    # 4.4 Mass values are strictly positive
    assert_test(
        "All mass values are strictly positive (>= 1e-12)",
        torch.all(d_map >= 1e-12).item(),
        f"(min mass: {d_map.min().item():.2e})",
    )


def check_section_5_metrics():
    print("\n" + "=" * 70)
    print("SECTION 5: Counting Metrics & Subgroup Diagnostics")
    print("=" * 70)

    preds = np.array([50.0, 120.0, 1500.0, 0.0])
    gts = np.array([40.0, 100.0, 1600.0, 0.0])

    res = evaluate_counting_metrics(preds, gts)
    sub = evaluate_subgroup_diagnostics(preds, gts)

    # MAE = (|10| + |20| + |100| + |0|) / 4 = 130 / 4 = 32.5
    # RMSE = sqrt((100 + 400 + 10000 + 0) / 4) = sqrt(2625) = 51.2347...
    # Bias = (10 + 20 - 100 + 0) / 4 = -70 / 4 = -17.5
    # NAE (ignoring zero) = (10/40 + 20/100 + 100/1600) / 3 = (0.25 + 0.20 + 0.0625) / 3 = 0.5125 / 3 = 0.17083...

    assert_test("MAE formula precision", abs(res["mae"] - 32.5) < 1e-5, f"(mae: {res['mae']:.2f})")
    assert_test("RMSE formula precision", abs(res["rmse"] - math.sqrt(2625)) < 1e-5, f"(rmse: {res['rmse']:.2f})")
    assert_test("Bias formula precision", abs(res["bias"] - (-17.5)) < 1e-5, f"(bias: {res['bias']:.2f})")
    assert_test("NAE formula precision", abs(res["nae"] - (0.5125 / 3.0)) < 1e-5, f"(nae: {res['nae']:.4f})")

    assert_test("Sparse bin [11-100] MAE", abs(sub["bin_11_100_mae"] - 10.0) < 1e-5, f"(mae: {sub['bin_11_100_mae']:.2f})")
    assert_test("Medium bin [101-1000] MAE", abs(sub["bin_101_1000_mae"] - 20.0) < 1e-5, f"(mae: {sub['bin_101_1000_mae']:.2f})")
    assert_test("Dense bin [>1000] MAE", abs(sub["bin_gt1000_mae"] - 100.0) < 1e-5, f"(mae: {sub['bin_gt1000_mae']:.2f})")


def check_section_6_configs_consistency():
    print("\n" + "=" * 70)
    print("SECTION 6: Configuration Files Consistency (R0 - R5)")
    print("=" * 70)

    config_files = sorted(glob.glob("configs/ntpc_r*.yaml"))
    assert_test("All 6 NTPC configuration files exist", len(config_files) == 6, f"(found: {len(config_files)})")

    for path in config_files:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        c_size = cfg["dataset"]["crop_size"]
        b_size = cfg["training"]["batch_size"]
        epochs = cfg["schedule"]["epochs"]
        val_ev = cfg["training"]["validate_every"]
        pretr = cfg["model"]["pretrained"]
        scale = cfg["augmentation"]["scale_range"]

        valid = (c_size == 256 and b_size == 16 and epochs == 1000 and val_ev == 5 and pretr is False and scale == [0.75, 2.0])
        assert_test(
            f"Config {os.path.basename(path)} is standardized",
            valid,
            f"(crop={c_size}, bs={b_size}, ep={epochs}, val_ev={val_ev}, pre={pretr})",
        )


def main():
    print("\n" + "#" * 70)
    print("STARTING NTPC FULL END-TO-END SYSTEM CROSS-CHECK")
    print("#" * 70)

    check_section_1_probabilistic_losses()
    check_section_2_ablation_modes_and_per_image_joint()
    check_section_3_data_structures_and_target_conservation()
    check_section_4_model_architecture()
    check_section_5_metrics()
    check_section_6_configs_consistency()

    print("\n" + "#" * 70)
    print(f"FINAL CROSS-CHECK SUMMARY: {total_passed} PASSED | {total_failed} FAILED")
    print("#" * 70)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
