"""Verify AMP training is self-correcting after a few skipped steps."""
import sys, torch
sys.path.insert(0, r"f:\lightweightcrcn")

device = torch.device("cuda")
from hpc.models.hpc_lite import HPCLite
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
from hpc.losses.count_tree import CountTreeConfig
from hpc.data.point_counts import build_exact_count_pyramid

ckpt = torch.load("runs/mobilenetv4_p8_s2/best.pt", map_location=device)
model = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False).to(device)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.train()

cfg = HPCLossConfig(
    tree=CountTreeConfig(root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
                         kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
                         use_dirichlet_multinomial=True,
                         w_root_nb=1.0, w_root64=1.0, w_64_32=1.0, w_32_16=1.0, w_16_8=1.0),
    hard_zero_weight=0.10, local_contrast_weight=0.05,
)
criterion = AdaptiveHPCLoss(cfg, feature_dim=32).to(device)

# Use Stage A1 config: disable all but root_nb + root->64
cfg_a1 = HPCLossConfig(
    tree=CountTreeConfig(root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
                         kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
                         use_dirichlet_multinomial=True,
                         w_root_nb=1.0, w_root64=1.0, w_64_32=0.0, w_32_16=0.0, w_16_8=0.0),
    hard_zero_weight=0.0, local_contrast_weight=0.0,  # disabled in A1
)
crit_a1 = AdaptiveHPCLoss(cfg_a1, feature_dim=32).to(device)

pts_batch = [
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(80)]),
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(10)]),
]
tgt = build_exact_count_pyramid(pts_batch, 448, 448, (8, 16, 32, 64))

scaler = torch.amp.GradScaler("cuda")
optim = torch.optim.AdamW(
    [{"params": list(model.parameters()), "lr": 5e-6}],
    weight_decay=1e-4,
)

print("=== AMP Self-Recovery Test (5 mini-steps) ===")
n_skipped = 0
n_applied = 0

for step in range(5):
    tgt_cuda = {k: v.to(device) for k, v in tgt.items()}
    images_cuda = torch.rand(2, 3, 448, 448).to(device)

    optim.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=True):
        d_map, aux = model(images_cuda, return_aux=True)
        total, logs = crit_a1(mass=d_map, p4=aux["p4"], target_pyramid=tgt_cuda)

    old_scale = scaler.get_scale()
    scaler.scale(total).backward()
    scaler.unscale_(optim)
    all_params = list(model.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=5.0)
    scaler.step(optim)
    scaler.update()
    new_scale = scaler.get_scale()

    step_applied = new_scale >= old_scale
    if step_applied:
        n_applied += 1
    else:
        n_skipped += 1

    pred_N = d_map.sum(dim=(1, 2, 3)).detach()
    print(f"  Step {step+1}: loss={total.item():.2f} (root_nb={logs['root_nb'].item():.2f}, "
          f"root64={logs['root_to_64'].item():.2f}) | "
          f"pred_N={pred_N.tolist()} | grad_norm={float(grad_norm):.2f} | "
          f"scale={old_scale:.0f}->{new_scale:.0f} | "
          f"{'APPLIED' if step_applied else 'SKIPPED'}")

print(f"\nTotal: {n_applied} applied, {n_skipped} skipped")
print(f"AMP is {'working correctly' if n_skipped < 5 else 'NOT recovering - may need lower init scale'}")
