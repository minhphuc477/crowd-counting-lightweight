"""Trace exactly where NaN appears in AMP backward."""
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

pts_batch = [
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(80)]),
    torch.tensor([[float(i % 448), float(i % 448)] for i in range(10)]),
]
tgt = build_exact_count_pyramid(pts_batch, 448, 448, (8, 16, 32, 64))
tgt_cuda = {k: v.to(device) for k, v in tgt.items()}
images_cuda = torch.rand(2, 3, 448, 448).to(device)

scaler = torch.amp.GradScaler("cuda")
optim = torch.optim.AdamW(
    [{"params": list(model.parameters()), "lr": 5e-6},
     {"params": list(criterion.local_contrast.projector.parameters()), "lr": 5e-5}],
    weight_decay=1e-4,
)

nan_found = {}
hooks = []

def make_hook(name):
    def hook(grad):
        if grad is not None and not torch.isfinite(grad).all():
            if name not in nan_found:
                nan_found[name] = grad.abs().max().item()
        return grad
    return hook

def hook_tensor(tensor, name):
    if tensor.requires_grad:
        h = tensor.register_hook(make_hook(name))
        hooks.append(h)

optim.zero_grad(set_to_none=True)
with torch.amp.autocast("cuda", enabled=True):
    c4, c8, c16 = model.backbone(images_cuda)
    hook_tensor(c4, "backbone_c4")
    hook_tensor(c8, "backbone_c8")
    hook_tensor(c16, "backbone_c16")

    p4, neck_aux = model.neck(c4, c8, c16, return_routes=True)
    hook_tensor(p4, "neck_p4")
    hook_tensor(neck_aux["p8"], "neck_p8")
    hook_tensor(neck_aux["p16"], "neck_p16")

    # Head path
    h = model.head_act(model.head_norm(model.head_dw(p4)))
    hook_tensor(h, "head_h")
    z = model.head_out(h)
    hook_tensor(z, "head_z")
    d_map = torch.nn.functional.softplus(z) + model.eps_d
    hook_tensor(d_map, "d_map")

    total, logs = criterion(mass=d_map, p4=p4, target_pyramid=tgt_cuda)

scaler.scale(total).backward()

for hh in hooks:
    hh.remove()

print("=== NaN/Inf in backward (tensor hooks) ===")
if nan_found:
    for name, val in sorted(nan_found.items()):
        print(f"  {name}: max_grad_magnitude={val}")
else:
    print("  None! All intermediate gradients finite during backward hooks.")

scaler.unscale_(optim)
all_named = list(model.named_parameters()) + [
    (f"proj.{n}", p) for n, p in criterion.local_contrast.projector.named_parameters()
]
inf_params = [(n, p) for n, p in all_named
              if p.grad is not None and not torch.isfinite(p.grad).all()]
print(f"\nParameters with inf/nan grad after unscale: {len(inf_params)}")
for name, p in inf_params[:10]:
    print(f"  {name}: max={p.grad.abs().max().item()}")

old_scale = scaler.get_scale()
scaler.step(optim)
scaler.update()
new_scale = scaler.get_scale()
print(f"\nScale: {old_scale} -> {new_scale}")
print(f"Step {'SKIPPED (inf/nan)' if new_scale < old_scale else 'applied'}")
