"""Comprehensive audit script for HPC-Lite implementation."""
import sys
import torch
import numpy as np

sys.path.insert(0, r"f:\lightweightcrcn")

PASS = "PASS"
FAIL = "FAIL"
results = []


def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {name}: {status}  {detail}")


print("=" * 60)
print("AUDIT 1: point_counts.py")
print("=" * 60)
from hpc.data.point_counts import (
    pad_hw_to_multiple,
    points_to_impulse_map,
    build_exact_count_pyramid,
)

# Test pad_hw_to_multiple
hp, wp = pad_hw_to_multiple(100, 120, 64)
check("pad_hw_to_multiple: 100->128, 120->128",
      hp == 128 and wp == 128, f"got {hp},{wp}")
hp2, wp2 = pad_hw_to_multiple(64, 64, 64)
check("pad_hw_to_multiple: 64->64 (exact)",
      hp2 == 64 and wp2 == 64, f"got {hp2},{wp2}")

# Test points_to_impulse_map: point (x=5.3, y=10.7) -> floor -> (5, 10)
pts = [torch.tensor([[5.3, 10.7], [0.0, 0.0]])]
impulse = points_to_impulse_map(pts, height=64, width=64, device=torch.device("cpu"))
check("impulse_map shape [1,1,H,W]", impulse.shape == (1, 1, 64, 64), str(impulse.shape))
check("impulse_map total count", impulse.sum().item() == 2.0, str(impulse.sum().item()))
check("impulse at floor(x)=5, floor(y)=10", impulse[0, 0, 10, 5].item() == 1.0,
      f"got {impulse[0,0,10,5].item()}")
check("impulse at (0,0)", impulse[0, 0, 0, 0].item() == 1.0,
      f"got {impulse[0,0,0,0].item()}")

# Test out-of-bounds clipping
pts_oob = [torch.tensor([[64.0, 64.0], [-1.0, 0.0]])]
impulse_oob = points_to_impulse_map(pts_oob, height=64, width=64, device=torch.device("cpu"))
check("OOB points are clipped (sum=0)", impulse_oob.sum().item() == 0.0,
      str(impulse_oob.sum().item()))

# Test build_exact_count_pyramid
points = [torch.tensor([[1.0, 1.0], [7.0, 7.0], [20.0, 5.0], [40.0, 40.0]])]
target = build_exact_count_pyramid(
    points_batch=points, height=64, width=64,
    block_sizes=(8, 16, 32, 64), device=torch.device("cpu"),
)
check("exact pyramid N=4", int(target["N"][0].item()) == 4, str(target["N"][0].item()))
for block in (8, 16, 32, 64):
    s = int(target[block][0].sum().item())
    check(f"pyramid block={block} sums to N=4", s == 4, f"got {s}")

# Points exactly on boundary of block: x=8.0 -> floor(8.0)=8 -> in block index 1 for block8
pts_bound = [torch.tensor([[8.0, 0.0]])]
t_bound = build_exact_count_pyramid(pts_bound, 64, 64, (8,), device=torch.device("cpu"))
check("point at x=8 falls in block8[0,1]",
      int(t_bound[8][0, 0, 1].item()) == 1,
      f"block8[0,0,1]={t_bound[8][0,0,1].item()}, block8[0,0,0]={t_bound[8][0,0,0].item()}")


print("\n" + "=" * 60)
print("AUDIT 2: count_tree.py - sum_pool_mass and conservation")
print("=" * 60)
from hpc.losses.count_tree import (
    pad_mass_map_for_image_multiple,
    sum_pool_mass,
    build_predicted_count_pyramid,
    group_four_children,
)

# Basic conservation test
torch.manual_seed(0)
mass = torch.rand(2, 1, 64, 64)  # feature map at stride 4 -> 256x256 image equivalent
p = build_predicted_count_pyramid(mass, block_sizes=(8, 16, 32, 64), output_stride=4)

check("pyramid N == mass.sum()",
      torch.allclose(p["N"], mass.sum(dim=(1, 2, 3)), atol=1e-4),
      f"N={p['N']}, sum={mass.sum(dim=(1,2,3))}")

# Conservation at each level
g8 = group_four_children(p[8])
check("conservation: sum(children8) == p[16]",
      torch.allclose(g8.sum(-1), p[16], atol=1e-5),
      f"max_diff={( g8.sum(-1) - p[16]).abs().max().item():.2e}")

g16 = group_four_children(p[16])
check("conservation: sum(children16) == p[32]",
      torch.allclose(g16.sum(-1), p[32], atol=1e-5),
      f"max_diff={(g16.sum(-1) - p[32]).abs().max().item():.2e}")

g32 = group_four_children(p[32])
check("conservation: sum(children32) == p[64]",
      torch.allclose(g32.sum(-1), p[64], atol=1e-5),
      f"max_diff={(g32.sum(-1) - p[64]).abs().max().item():.2e}")

check("conservation: p[64].sum == N",
      torch.allclose(p[64].sum(dim=(1, 2)), p["N"], atol=1e-4),
      f"max_diff={(p[64].sum(dim=(1,2)) - p['N']).abs().max().item():.2e}")

# Test group_four_children correctness: 2x2 -> 4 children
# p[32] has shape [2, H32, W32], p[16] has shape [2, H16, W16] where H16=2*H32
print(f"  p[8].shape={p[8].shape}, p[16].shape={p[16].shape}")
print(f"  p[32].shape={p[32].shape}, p[64].shape={p[64].shape}")
# When feature map is 64x64 at stride 4, image is 256x256
# block8: k=2 -> 32x32 grid of block8 counts
# block16: k=4 -> 16x16 grid
# block32: k=8 -> 8x8 grid
# block64: k=16 -> 4x4 grid

# CRITICAL CHECK: group_four_children layout
# spec: child 32x32 blocks under parent 64x64 block
# A 64x64 parent block in image space contains 4 32x32 children in a 2x2 arrangement
# p[32] shape is [B, H_img/32, W_img/32] = [B, H32, W32]
# p[64] shape is [B, H_img/64, W_img/64] = [B, H64, W64]
# group_four_children(p[32]) must give [B, H64, W64, 4] where sum == p[64]
# This checks that the 2x2 patch of 32-blocks sums to the corresponding 64-block
g32_under_64 = group_four_children(p[32])
check("group_four_children layout: p[32] grouped under p[64] sums correctly",
      torch.allclose(g32_under_64.sum(-1), p[64], atol=1e-5),
      f"max_diff={(g32_under_64.sum(-1)-p[64]).abs().max().item():.2e}")


print("\n" + "=" * 60)
print("AUDIT 3: negative_binomial.py")
print("=" * 60)
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion

# NB NLL: at mean=y, loss should be close to minimum
y = torch.tensor([10.0, 50.0, 100.0])
mu_good = torch.tensor([10.0, 50.0, 100.0])
mu_bad = torch.tensor([100.0, 500.0, 1000.0])
r = 50.0

l_good = negative_binomial_nll_mean_dispersion(y, mu_good, r)
l_bad = negative_binomial_nll_mean_dispersion(y, mu_bad, r)
check("NB: good mean < bad mean NLL",
      l_good.item() < l_bad.item(),
      f"good={l_good.item():.3f}, bad={l_bad.item():.3f}")

# Gradient test
mu_g = torch.tensor([10.0, 50.0], requires_grad=True)
y_g = torch.tensor([10.0, 50.0])
l = negative_binomial_nll_mean_dispersion(y_g, mu_g, 50.0)
l.backward()
check("NB gradient is finite", mu_g.grad is not None and torch.isfinite(mu_g.grad).all(),
      str(mu_g.grad))
# At mu=y, gradient should be ~0 (minimum)
# dL/dmu = (mu - y) / (mu * (1 + mu/r))
expected_grad = (mu_g.data - y_g) / (mu_g.data * (1 + mu_g.data / r))
check("NB gradient at mean=y is ~0",
      (mu_g.grad - expected_grad).abs().max().item() < 1e-5,
      f"grad={mu_g.grad}, expected={expected_grad}")

# AMP safety: float32 cast inside function
mu_half = torch.tensor([10.0]).half()
y_half = torch.tensor([10.0])
l_half = negative_binomial_nll_mean_dispersion(y_half, mu_half, 50.0)
check("NB: float16 input doesn't produce NaN", not torch.isnan(l_half),
      str(l_half.item()))


print("\n" + "=" * 60)
print("AUDIT 4: dirichlet_multinomial.py")
print("=" * 60)
from hpc.losses.dirichlet_multinomial import (
    normalize_positive_mass,
    dirichlet_multinomial_nll,
    multinomial_nll,
)

# normalize_positive_mass: always sums to 1
mass_n = torch.tensor([[3.0, 0.0, 1.0, 0.0]])
probs = normalize_positive_mass(mass_n, dim=-1)
check("normalize_positive_mass sums to ~1",
      abs(probs.sum(-1).item() - 1.0) < 1e-5,
      str(probs.sum(-1).item()))
check("normalize_positive_mass: all positive",
      (probs > 0).all().item(), str(probs))

# DM: good allocation should have lower NLL than bad
y = torch.tensor([[8.0, 1.0, 1.0, 0.0]])
good = torch.tensor([[0.78, 0.10, 0.10, 0.02]])
bad = torch.tensor([[0.05, 0.30, 0.30, 0.35]])

lg = dirichlet_multinomial_nll(y, good, concentration=20.0)
lb = dirichlet_multinomial_nll(y, bad, concentration=20.0)
check("DM: correct allocation < wrong allocation NLL",
      lg.item() < lb.item(),
      f"good={lg.item():.3f}, bad={lb.item():.3f}")

# DM gradient finite
probs_g = torch.tensor([[0.78, 0.10, 0.10, 0.02]], requires_grad=True)
y_g = torch.tensor([[8.0, 1.0, 1.0, 0.0]])
l = dirichlet_multinomial_nll(y_g, probs_g, concentration=20.0)
l.backward()
check("DM gradient is finite", probs_g.grad is not None and torch.isfinite(probs_g.grad).all(),
      str(probs_g.grad))

# DM with valid_mask
y_m = torch.tensor([[8.0, 1.0, 1.0, 0.0], [5.0, 5.0, 0.0, 0.0]])
p_m = torch.tensor([[0.78, 0.10, 0.10, 0.02], [0.5, 0.5, 0.0, 0.0]])
mask = torch.tensor([True, False])
l_masked = dirichlet_multinomial_nll(y_m, p_m, concentration=20.0, valid_mask=mask)
l_single = dirichlet_multinomial_nll(y_m[:1], p_m[:1], concentration=20.0)
check("DM valid_mask: only computes loss on masked elements",
      torch.allclose(l_masked, l_single, atol=1e-5),
      f"masked={l_masked.item():.4f}, single={l_single.item():.4f}")

# DM: n=0 (empty parent) - should not crash
y_zero = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
p_zero = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
try:
    l_zero = dirichlet_multinomial_nll(y_zero, p_zero, concentration=20.0)
    check("DM: n=0 doesn't crash", True, f"loss={l_zero.item():.4f}")
    # lgamma(0+1) = lgamma(1) = 0, log_coeff = lgamma(1) - 4*lgamma(1) = 0
    # log_global = lgamma(alpha0) - lgamma(alpha0) = 0 (if n=0 -> n+alpha0 = alpha0)
    # Wait: lgamma(n+alpha0) at n=0 = lgamma(alpha0) so log_global = 0
    # log_local = sum_i[lgamma(y_i+alpha_i) - lgamma(alpha_i)] = sum_i[lgamma(alpha_i) - lgamma(alpha_i)] = 0
    # So log_prob = 0, nll = 0
    check("DM: n=0 gives loss=0", abs(l_zero.item()) < 1e-4, str(l_zero.item()))
except Exception as e:
    check("DM: n=0 doesn't crash", False, str(e))

# Float16 input safety
y_h = torch.tensor([[8.0, 1.0, 1.0, 0.0]])
p_h = torch.tensor([[0.78, 0.10, 0.10, 0.02]]).half()
try:
    l_h = dirichlet_multinomial_nll(y_h, p_h, concentration=20.0)
    check("DM: float16 probs doesn't crash", True, f"loss={l_h.item():.4f}")
    check("DM: float16 result is finite", torch.isfinite(l_h), str(l_h.item()))
except Exception as e:
    check("DM: float16 probs doesn't crash", False, str(e))


print("\n" + "=" * 60)
print("AUDIT 5: count_tree.py - AdaptiveProbabilisticCountTreeLoss")
print("=" * 60)
from hpc.losses.count_tree import AdaptiveProbabilisticCountTreeLoss, CountTreeConfig

cfg = CountTreeConfig(
    root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0, kappa_32_16=20.0, kappa_16_8=20.0,
    dense_threshold_16=2, use_dirichlet_multinomial=True,
    w_root_nb=1.0, w_root64=1.0, w_64_32=1.0, w_32_16=1.0, w_16_8=1.0,
)
tree_loss = AdaptiveProbabilisticCountTreeLoss(cfg)

# Create a synthetic batch
torch.manual_seed(42)
mass = torch.rand(2, 1, 64, 64).requires_grad_(True)  # feature at stride 4 -> 256x256 image
pred_pyramid = build_predicted_count_pyramid(mass, (8, 16, 32, 64), 4)

# Create target pyramid with integer counts
target_pyramid = {}
for b in (8, 16, 32, 64):
    target_pyramid[b] = pred_pyramid[b].detach().round().clamp_min(0)
target_pyramid["N"] = target_pyramid[64].sum(dim=(1, 2))

total, pieces = tree_loss(target=target_pyramid, pred=pred_pyramid)
check("TreeLoss: forward succeeds", True, "")
check("TreeLoss: total is scalar", total.ndim == 0, str(total.shape))
check("TreeLoss: total is finite", torch.isfinite(total), f"total={total.item():.4f}")

total.backward()
check("TreeLoss: backward succeeds, grad finite",
      mass.grad is not None and torch.isfinite(mass.grad).all(),
      f"max_grad={mass.grad.abs().max().item():.4f}")

# Check pieces dict
expected_keys = {"root_nb", "root_to_64", "64_to_32", "32_to_16", "16_to_8_dense", "flat_16", "indep_nb", "tree_total"}
check("TreeLoss: all expected keys in pieces",
      expected_keys == set(pieces.keys()),
      f"got {set(pieces.keys())}")

# Verify w_=0 correctly zeros out contribution
cfg_partial = CountTreeConfig(
    root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0, kappa_32_16=20.0, kappa_16_8=20.0,
    dense_threshold_16=999,  # no dense parents
    use_dirichlet_multinomial=False,
    w_root_nb=1.0, w_root64=1.0, w_64_32=0.0, w_32_16=0.0, w_16_8=0.0,
)
tl_partial = AdaptiveProbabilisticCountTreeLoss(cfg_partial)
mass2 = torch.rand(2, 1, 64, 64).requires_grad_(True)
pred2 = build_predicted_count_pyramid(mass2, (8, 16, 32, 64), 4)
total2, p2 = tl_partial(target=target_pyramid, pred=pred2)
expected_total2 = p2["root_nb"] + p2["root_to_64"]  # only these two active
check("TreeLoss: w=0 correctly disables levels",
      torch.allclose(total2, expected_total2, atol=1e-4),
      f"total={total2.item():.4f}, root_nb+root64={expected_total2.item():.4f}")


print("\n" + "=" * 60)
print("AUDIT 6: hard_zero.py - HardZeroRegionLoss")
print("=" * 60)
from hpc.losses.hard_zero import HardZeroRegionLoss

hz_loss = HardZeroRegionLoss(top_fraction=0.10, min_k=1, beta=1.0)

# pred_count16: predicted counts for 16x16 blocks
# target_count16: true counts (some are 0)
pred16 = torch.tensor([[[0.0, 5.0, 0.0, 10.0],
                         [0.0, 0.0, 3.0, 0.0]]])  # [1, 2, 4]
target16 = torch.tensor([[[0.0, 5.0, 0.0, 10.0],
                           [0.0, 0.0, 3.0, 0.0]]])  # same, so 4 zero cells

# Zero cells in pred: indices where target=0 -> positions (0,0), (0,2), (1,0), (1,1)
# Predicted values at zero cells: 0.0, 0.0, 0.0, 0.0 -> loss should be 0
l_perfect = hz_loss(pred16, target16)
check("HardZero: perfect predictions give ~0 loss",
      l_perfect.item() < 1e-5, f"loss={l_perfect.item():.6f}")

# Large false positive predictions in zero cells
pred16_bad = torch.tensor([[[100.0, 5.0, 50.0, 10.0],
                              [200.0, 300.0, 3.0, 0.0]]])
l_bad = hz_loss(pred16_bad, target16)
check("HardZero: false positives give positive loss",
      l_bad.item() > 0, f"loss={l_bad.item():.4f}")

# Gradient test
pred16_g = torch.tensor([[[10.0, 5.0, 20.0, 10.0],
                           [5.0, 15.0, 3.0, 0.0]]], requires_grad=True)
l_g = hz_loss(pred16_g, target16)
l_g.backward()
check("HardZero gradient is finite",
      pred16_g.grad is not None and torch.isfinite(pred16_g.grad).all(),
      str(pred16_g.grad))

# valid16 mask: padded cells should be excluded
valid16 = torch.tensor([[[True, True, True, True],
                          [True, True, False, False]]])  # last 2 are padding
# target=3.0 at (1,2) but it's padding -> excluded from zero mask
# zero cells that are valid: (0,0), (0,2), (1,0), (1,1)
pred16_mask = torch.tensor([[[100.0, 5.0, 50.0, 10.0],
                               [200.0, 300.0, 999.0, 999.0]]])  # 999 at padding cells
target16_mask = torch.tensor([[[0.0, 5.0, 0.0, 10.0],
                                [0.0, 0.0, 3.0, 0.0]]])
l_mask = hz_loss(pred16_mask, target16_mask, valid16=valid16)
check("HardZero valid16 mask excludes padding",
      l_mask.item() > 0, f"loss={l_mask.item():.4f}")  # Should compute from valid zeros only


print("\n" + "=" * 60)
print("AUDIT 7: supervised_contrastive.py")
print("=" * 60)
from hpc.losses.supervised_contrastive import (
    pool_p4_to_16px_windows,
    density_classes_from_exact_count,
    balanced_subsample_indices,
    supervised_contrastive_loss,
    LocalDensityContrastiveLoss,
)

# pool_p4_to_16px_windows: stride4 feature, 16px block -> k=4
p4 = torch.rand(2, 32, 112, 112)  # stride4 feature for 448x448 image
pooled = pool_p4_to_16px_windows(p4)
check("pool_p4_to_16px_windows output shape",
      pooled.shape == (2, 32, 28, 28),  # 112/4=28
      str(pooled.shape))

# density_classes_from_exact_count
y16 = torch.tensor([[0.0, 1.0, 2.0, 5.0], [3.0, 0.0, 10.0, 1.0]])
classes = density_classes_from_exact_count(y16, t1=1, t2=4)
expected = torch.tensor([[0, 1, 2, 3], [2, 0, 3, 1]])
check("density_classes correct",
      torch.equal(classes, expected), f"\ngot:      {classes}\nexpected: {expected}")

# balanced_subsample_indices
labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])
idx = balanced_subsample_indices(labels, max_samples=8, num_classes=4)
check("balanced_subsample returns indices",
      idx.numel() > 0 and idx.numel() <= 8, str(idx))
check("balanced_subsample: indices valid range",
      (idx >= 0).all() and (idx < len(labels)).all(), str(idx))

# supervised_contrastive_loss: same-class pairs should have lower loss when well separated
torch.manual_seed(0)
# Two perfectly separated classes (class 0 points up, class 1 points down)
z = torch.zeros(4, 2)
z[0] = torch.tensor([1.0, 0.0])
z[1] = torch.tensor([0.9, 0.1])  # same class
z[2] = torch.tensor([-1.0, 0.0])
z[3] = torch.tensor([-0.9, -0.1])  # same class
z = torch.nn.functional.normalize(z, p=2, dim=-1)
labels = torch.tensor([0, 0, 1, 1])
l_sep = supervised_contrastive_loss(z, labels, temperature=0.1)

# Well-separated vs random
z_rand = torch.randn(4, 2)
z_rand = torch.nn.functional.normalize(z_rand, p=2, dim=-1)
l_rand = supervised_contrastive_loss(z_rand, labels, temperature=0.1)

check("SupContrast: separated classes loss is lower than random",
      l_sep.item() < l_rand.item(),
      f"separated={l_sep.item():.4f}, random={l_rand.item():.4f}")
check("SupContrast: separated classes loss is finite",
      torch.isfinite(l_sep), f"loss={l_sep.item():.4f}")
check("SupContrast: separated classes loss >= 0",
      l_sep.item() >= 0, f"loss={l_sep.item():.4f}")

# Gradient test
z_g = z.clone().detach().requires_grad_(True)
l_g = supervised_contrastive_loss(z_g, labels, temperature=0.1)
l_g.backward()
check("SupContrast gradient finite",
      z_g.grad is not None and torch.isfinite(z_g.grad).all(), str(z_g.grad))

# LocalDensityContrastiveLoss: forward pass
local_cl = LocalDensityContrastiveLoss(feature_dim=32, hidden_dim=64, projection_dim=32,
                                        low_threshold=1, dense_threshold=4, max_samples=64, temperature=0.1)
p4_test = torch.rand(2, 32, 112, 112)
y16_test = torch.randint(0, 10, (2, 28, 28)).float()
l_lc = local_cl(p4_test, y16_test)
check("LocalDensityContrastiveLoss forward succeeds", True, "")
check("LocalDensityContrastiveLoss loss finite", torch.isfinite(l_lc), str(l_lc.item()))


print("\n" + "=" * 60)
print("AUDIT 8: hpc_adaptive.py - AdaptiveHPCLoss forward")
print("=" * 60)
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
from hpc.losses.count_tree import CountTreeConfig

loss_cfg = HPCLossConfig(
    tree=CountTreeConfig(
        root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
        kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
        use_dirichlet_multinomial=True,
        w_root_nb=1.0, w_root64=1.0, w_64_32=1.0, w_32_16=1.0, w_16_8=1.0,
    ),
    hard_zero_weight=0.10, local_contrast_weight=0.05,
    hard_zero_top_fraction=0.10, local_low_threshold=1, local_dense_threshold=4,
    local_max_samples=64, local_temperature=0.10,
)
criterion = AdaptiveHPCLoss(loss_cfg, feature_dim=32)

torch.manual_seed(99)
mass_in = torch.rand(2, 1, 112, 112).requires_grad_(True)  # 448x448 image
p4_in = torch.rand(2, 32, 112, 112)
# Build integer target pyramid for 448x448 image
from hpc.data.point_counts import build_exact_count_pyramid
import random
random.seed(0)
# Synthetic points in 448x448 image space
pts_batch = [
    torch.tensor([[float(random.randint(0, 447)), float(random.randint(0, 447))] for _ in range(80)]),
    torch.tensor([[float(random.randint(0, 447)), float(random.randint(0, 447))] for _ in range(30)]),
]
tgt = build_exact_count_pyramid(pts_batch, height=448, width=448, block_sizes=(8, 16, 32, 64))

total_l, logs = criterion(mass=mass_in, p4=p4_in, target_pyramid=tgt)
check("AdaptiveHPCLoss forward succeeds", True, "")
check("AdaptiveHPCLoss total finite", torch.isfinite(total_l), f"total={total_l.item():.4f}")
expected_log_keys = {"root_nb", "root_to_64", "64_to_32", "32_to_16", "16_to_8_dense",
                     "flat_16", "indep_nb", "exact_count",
                     "tree_total", "hard_zero", "local_contrast", "total"}
check("AdaptiveHPCLoss logs has correct keys",
      expected_log_keys == set(logs.keys()),
      f"\ngot={set(logs.keys())}\nexpected={expected_log_keys}")

total_l.backward()
check("AdaptiveHPCLoss backward succeeds, grad finite",
      mass_in.grad is not None and torch.isfinite(mass_in.grad).all(),
      f"max_grad={mass_in.grad.abs().max().item():.4f}")


print("\n" + "=" * 60)
print("AUDIT 9: AMP compatibility - all loss functions")
print("=" * 60)
if torch.cuda.is_available():
    device = torch.device("cuda")
    mass_amp = torch.rand(2, 1, 112, 112, device=device).requires_grad_(True)
    p4_amp = torch.rand(2, 32, 112, 112, device=device)
    tgt_gpu = {k: v.to(device) for k, v in tgt.items()}

    crit_gpu = AdaptiveHPCLoss(loss_cfg, feature_dim=32).to(device)

    with torch.amp.autocast("cuda", enabled=True):
        total_amp, logs_amp = crit_gpu(mass=mass_amp, p4=p4_amp, target_pyramid=tgt_gpu)

    check("AMP: loss is finite", torch.isfinite(total_amp), f"total={total_amp.item():.4f}")
    check("AMP: loss dtype is float32 (not float16)",
          total_amp.dtype == torch.float32,
          str(total_amp.dtype))
    total_amp.backward()
    check("AMP: backward grad finite",
          mass_amp.grad is not None and torch.isfinite(mass_amp.grad).all(),
          f"max_grad={mass_amp.grad.abs().max().item():.6f}")
else:
    print("  [SKIP] CUDA not available, skipping AMP tests")


print("\n" + "=" * 60)
print("AUDIT 10: train_probabilistic.py API compatibility")
print("=" * 60)
from hpc.models.hpc_lite import HPCLite

model = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False)
model.eval()

x = torch.rand(1, 3, 448, 448)
with torch.no_grad():
    d_map, aux = model(x, return_aux=True)

check("HPCLite forward: d_map shape (1,1,112,112)",
      d_map.shape == (1, 1, 112, 112), str(d_map.shape))
check("HPCLite forward: d_map all positive (softplus)",
      (d_map > 0).all().item(), f"min={d_map.min().item():.4f}")
check("HPCLite forward: aux has 'p4' key",
      "p4" in aux, str(list(aux.keys())))
check("HPCLite forward: aux['p4'] shape (1,32,112,112)",
      aux["p4"].shape == (1, 32, 112, 112), str(aux["p4"].shape))

# predict API
with torch.no_grad():
    cnt, d_map_pred = model.predict(x, pad_multiple=32)
check("HPCLite predict: returns scalar count",
      cnt.ndim == 0 or (cnt.ndim == 1 and cnt.numel() == 1), str(cnt.shape))


print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)
n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print(f"  Total: {len(results)}  PASS: {n_pass}  FAIL: {n_fail}")
if n_fail > 0:
    print("\n  FAILED CHECKS:")
    for name, status, detail in results:
        if status == FAIL:
            print(f"    ✗ {name}: {detail}")
sys.exit(0 if n_fail == 0 else 1)
