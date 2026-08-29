"""NB + DM + training flow edge case audit."""
import sys, torch, math
sys.path.insert(0, r"f:\lightweightcrcn")

from hpc.losses.negative_binomial import (
    negative_binomial_nll_mean_dispersion,
    estimate_nb_dispersion_method_of_moments,
)

PASS = "PASS"
FAIL = "FAIL"
results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((name, status, detail))
    icon = "✓" if condition else "✗"
    print(f"  [{icon}] {name}: {status}  {detail}")

print("=" * 60)
print("AUDIT A: NB edge cases")
print("=" * 60)

# Edge case 1: mu very small, y=0
mu_tiny = torch.tensor([1e-8, 1e-6, 1e-4])
y_zero = torch.tensor([0.0, 0.0, 0.0])
l1 = negative_binomial_nll_mean_dispersion(y_zero, mu_tiny, 50.0)
check("NB(y=0, mu~0): finite", torch.isfinite(l1).all().item(), str(l1))

# Edge case 2: large N
mu_large = torch.tensor([2000.0, 5000.0])
y_large = torch.tensor([2000.0, 5000.0])
l2 = negative_binomial_nll_mean_dispersion(y_large, mu_large, 50.0)
check("NB(y=mu=large): finite", torch.isfinite(l2).all().item(), str(l2))

# Edge case 3: various r values
for r_val in [0.1, 1.0, 10.0, 50.0, 1e6]:
    l = negative_binomial_nll_mean_dispersion(
        torch.tensor([100.0]), torch.tensor([100.0]), float(r_val)
    )
    check(f"NB(y=mu=100, r={r_val}): finite", torch.isfinite(l).all().item(), f"{l.item():.4f}")

# Dispersion estimation
counts_varied = torch.tensor([float(500 + i*30) for i in range(-10, 10)])
r_est = estimate_nb_dispersion_method_of_moments(counts_varied)
check("NB MOM dispersion: positive", r_est > 0, f"r={r_est:.4f}")
check("NB MOM dispersion: reasonable range", 0.1 <= r_est <= 1e7, f"r={r_est:.4f}")

counts_same = torch.tensor([100.0] * 20)
r_same = estimate_nb_dispersion_method_of_moments(counts_same)
check("NB MOM with var<=mean: returns large value", r_same >= 1e5, f"r={r_same}")


print("\n" + "=" * 60)
print("AUDIT B: train_probabilistic.py - forward pass with actual data batch")
print("=" * 60)

from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
from hpc.losses.count_tree import CountTreeConfig, build_predicted_count_pyramid
from hpc.models.hpc_lite import HPCLite

model = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False)
model.eval()

# Simulate realistic 448x448 training batch (batch=2)
B, C, H, W = 2, 3, 448, 448
images = torch.rand(B, C, H, W)

with torch.no_grad():
    d_map, aux = model(images, return_aux=True)

check("Model output shape: (B,1,H/4,W/4)",
      d_map.shape == (B, 1, H//4, W//4), str(d_map.shape))
check("p4 shape: (B,32,H/4,W/4)",
      aux["p4"].shape == (B, 32, H//4, W//4), str(aux["p4"].shape))

# Build integer target pyramid
pts_batch = [
    torch.tensor([[float(i % W), float(i % H)] for i in range(80)]),  # 80 points
    torch.tensor([[float(i % W), float(i % H)] for i in range(10)]),  # 10 points
]
tgt = build_exact_count_pyramid(pts_batch, H, W, (8, 16, 32, 64))

check("Target N counts correct",
      int(tgt["N"][0].item()) == 80 and int(tgt["N"][1].item()) == 10,
      f"N={tgt['N'].tolist()}")
check("Shapes match: tgt[16] vs pred[16]",
      tgt[16].shape == build_predicted_count_pyramid(d_map, (8,16,32,64), 4)[16].shape,
      f"tgt={tgt[16].shape}")

# Full criterion forward
loss_cfg = HPCLossConfig(
    tree=CountTreeConfig(
        root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
        kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
        use_dirichlet_multinomial=True,
        w_root_nb=1.0, w_root64=1.0, w_64_32=1.0, w_32_16=1.0, w_16_8=1.0,
    ),
    hard_zero_weight=0.10, local_contrast_weight=0.05,
)
criterion = AdaptiveHPCLoss(loss_cfg, feature_dim=32)

d_map_g = d_map.detach().requires_grad_(True)
total, logs = criterion(mass=d_map_g, p4=aux["p4"], target_pyramid=tgt)
check("Full criterion forward: finite", torch.isfinite(total), f"total={total.item():.4f}")

total.backward()
check("Full criterion backward: finite grad",
      d_map_g.grad is not None and torch.isfinite(d_map_g.grad).all(),
      f"max_grad={d_map_g.grad.abs().max().item():.6f}")

# Log sanity: each piece should be finite
for k, v in logs.items():
    check(f"logs[{k}] finite", torch.isfinite(v), f"{v.item():.4f}")


print("\n" + "=" * 60)
print("AUDIT C: AMP + gradient clipping simulation")
print("=" * 60)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model_cuda = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False).to(device)
    criterion_cuda = AdaptiveHPCLoss(loss_cfg, feature_dim=32).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": list(model_cuda.parameters()), "lr": 5e-6},
         {"params": list(criterion_cuda.local_contrast.projector.parameters()), "lr": 5e-5}],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    
    images_cuda = torch.rand(2, 3, 448, 448).to(device)
    tgt_cuda = {k: v.to(device) for k, v in tgt.items()}
    
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        d_map_cuda, aux_cuda = model_cuda(images_cuda, return_aux=True)
        total_cuda, logs_cuda = criterion_cuda(
            mass=d_map_cuda, p4=aux_cuda["p4"], target_pyramid=tgt_cuda
        )
    
    check("AMP total finite", torch.isfinite(total_cuda), f"{total_cuda.item():.4f}")
    check("AMP total dtype float32", total_cuda.dtype == torch.float32, str(total_cuda.dtype))
    
    scaler.scale(total_cuda).backward()
    scaler.unscale_(optimizer)
    all_params = list(model_cuda.parameters()) + list(criterion_cuda.local_contrast.projector.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
    check("AMP: pre-clip grad_norm is finite", torch.isfinite(grad_norm).item(), f"{float(grad_norm):.4f}")
    post_clip_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in all_params if p.grad is not None]))
    check("AMP: post-clip grad_norm <= 5.0", float(post_clip_norm) <= 5.0 + 1e-3, f"{float(post_clip_norm):.4f}")
    scaler.step(optimizer)
    scaler.update()
    check("AMP: optimizer step completes", True, "")

else:
    print("  [SKIP] CUDA not available")


print("\n" + "=" * 60)
print("AUDIT D: Dataset and collate - gt_blocks key alignment")
print("=" * 60)
# The dataset returns gt_blocks with int keys
# train_probabilistic.py does: {int(k): v ... for k,v in batch['gt_blocks'].items()}
# But the dataset builds gt_blocks with hnb_blocks=[8,16,32,64]
# We need to check that 8 is INCLUDED in hnb_blocks

# Check sha.py default: hnb_blocks=(16,32,64) -- MISSING 8!
import inspect
from hpc.data.sha import ShanghaiTechDataset
sig = inspect.signature(ShanghaiTechDataset.__init__)
default_hnb = sig.parameters["hnb_blocks"].default
print(f"ShanghaiTechDataset default hnb_blocks: {default_hnb}")
check("ShanghaiTechDataset default hnb_blocks includes 8",
      8 in list(default_hnb),
      f"default_hnb={list(default_hnb)}")

# Check train_probabilistic.py config parse:
# hnb_blocks=ds_cfg.get("hnb_blocks", [8, 16, 32, 64]) -- correct
import train_probabilistic, inspect as insp
src = insp.getsource(train_probabilistic.main)
check("train_probabilistic.py default hnb_blocks includes 8",
      "[8, 16, 32, 64]" in src, "")


print("\n" + "=" * 60)
print("AUDIT E: Numerical stability of lgamma under AMP")
print("=" * 60)
# Spec says: Compute lgamma losses in float32 even under AMP
# Our implementation casts to float32 before lgamma. Verify.
from hpc.losses.dirichlet_multinomial import dirichlet_multinomial_nll
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion

# Test with half-precision inputs
y_half = torch.tensor([8.0, 1.0, 1.0, 0.0]).half()
p_half = torch.tensor([[8.0, 1.0, 1.0, 0.0], [0.78, 0.10, 0.10, 0.02]]).half()

# DM loss
l_dm = dirichlet_multinomial_nll(y_half.unsqueeze(0), p_half[:1], concentration=20.0)
check("DM: half-precision input -> float32 output",
      l_dm.dtype == torch.float32, str(l_dm.dtype))
check("DM: half-precision input -> finite", torch.isfinite(l_dm), str(l_dm.item()))

# NB loss
mu_half = torch.tensor([100.0]).half()
y_nb = torch.tensor([100.0])
l_nb = negative_binomial_nll_mean_dispersion(y_nb, mu_half, 50.0)
check("NB: half-precision mean -> float32 output",
      l_nb.dtype == torch.float32, str(l_nb.dtype))
check("NB: half-precision mean -> finite", torch.isfinite(l_nb), str(l_nb.item()))


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
