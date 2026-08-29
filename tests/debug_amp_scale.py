"""Test AMP with lower initial GradScaler scale."""
import sys, torch
sys.path.insert(0, r"f:\lightweightcrcn")

device = torch.device("cuda")
from hpc.models.hpc_lite import HPCLite
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
from hpc.losses.count_tree import CountTreeConfig

ckpt = torch.load("runs/mobilenetv4_p8_s2/best.pt", map_location=device)
model = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False).to(device)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state, strict=False)
model.train()

pts_batch = [
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(80)]),
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(10)]),
]
tgt = build_exact_count_pyramid(pts_batch, 448, 448, (8, 16, 32, 64))
tgt_cuda = {k: v.to(device) for k, v in tgt.items()}
images_cuda = torch.rand(2, 3, 448, 448).to(device)

cfg_a1 = HPCLossConfig(
    tree=CountTreeConfig(root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
                         kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
                         use_dirichlet_multinomial=True,
                         w_root_nb=1.0, w_root64=1.0, w_64_32=0.0, w_32_16=0.0, w_16_8=0.0),
    hard_zero_weight=0.0, local_contrast_weight=0.0,
)
criterion = AdaptiveHPCLoss(cfg_a1, feature_dim=32).to(device)

# Test with lower init scale: 256 instead of 65536
for init_scale in [65536.0, 1024.0, 256.0]:
    print(f"\n=== Testing init_scale={init_scale:.0f} ===")
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)
    optim = torch.optim.AdamW(
        [{"params": list(model.parameters()), "lr": 5e-6}],
        weight_decay=1e-4,
    )
    model_fresh = HPCLite(pretrained=False, use_p8_context=True, use_repblock=False).to(device)
    model_fresh.load_state_dict(state, strict=False)
    model_fresh.train()
    optim = torch.optim.AdamW(
        [{"params": list(model_fresh.parameters()), "lr": 5e-6}],
        weight_decay=1e-4,
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=init_scale)

    n_applied = 0
    for step in range(5):
        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            d_map, aux = model_fresh(images_cuda, return_aux=True)
            total, logs = criterion(mass=d_map, p4=aux["p4"], target_pyramid=tgt_cuda)

        old_scale = scaler.get_scale()
        scaler.scale(total).backward()
        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(model_fresh.parameters(), 5.0)
        scaler.step(optim)
        scaler.update()
        new_scale = scaler.get_scale()
        applied = new_scale >= old_scale
        if applied:
            n_applied += 1
        print(f"  Step {step+1}: loss={total.item():.2f} | scale={old_scale:.0f}->{new_scale:.0f} | {'APPLIED' if applied else 'SKIPPED'}")

    print(f"Applied: {n_applied}/5")
