"""Trace NaN source in AMP backward pass."""
import sys, torch
sys.path.insert(0, r"f:\lightweightcrcn")

device = torch.device("cuda")
from hpc.models.hpc_lite import HPCLite
from hpc.losses.count_tree import build_predicted_count_pyramid
from hpc.data.point_counts import build_exact_count_pyramid
from hpc.losses.negative_binomial import negative_binomial_nll_mean_dispersion
from hpc.losses.dirichlet_multinomial import dirichlet_multinomial_nll, normalize_positive_mass

# Load best.pt
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

# Step 1: Get mass under AMP (grad will flow through model)
with torch.amp.autocast("cuda", enabled=True):
    d_map, aux = model(images_cuda, return_aux=True)

mass_detach = d_map.detach()
print(f"d_map stats: min={mass_detach.min():.6f}, max={mass_detach.max():.6f}")
print(f"pred N: {mass_detach.sum(dim=(1,2,3))}")
print(f"tgt N: {tgt_cuda['N']}")

# Step 2: Check each loss term individually for nan/inf
pred = build_predicted_count_pyramid(mass_detach, (8,16,32,64), 4)
mu_N = pred["N"]
y_N = tgt_cuda["N"]
l_nb = negative_binomial_nll_mean_dispersion(y_N, mu_N, 50.0)
print(f"NB loss: {l_nb.item():.4f}, finite={torch.isfinite(l_nb).item()}")

mu_flat = pred[64].reshape(2, -1)
probs64 = normalize_positive_mass(mu_flat, dim=-1)
y64_flat = tgt_cuda[64].reshape(2, -1).float()
l_dm64 = dirichlet_multinomial_nll(y64_flat, probs64, concentration=20.0)
print(f"Root->64 DM loss: {l_dm64.item():.4f}, finite={torch.isfinite(l_dm64).item()}")

# Step 3: Check the full loss with gradient enabled on mass, but not model
mass_g = mass_detach.detach().requires_grad_(True)
pred_g = build_predicted_count_pyramid(mass_g, (8, 16, 32, 64), 4)
mu_N_g = pred_g["N"]
l_nb_g = negative_binomial_nll_mean_dispersion(y_N, mu_N_g, 50.0)
l_nb_g.backward()
print(f"NB grad wrt mass: max={mass_g.grad.abs().max():.4f}, finite={torch.isfinite(mass_g.grad).all().item()}")

# Step 4: Now test if this large gradient * fp16 scale = inf
# The GradScaler unscale_: divides scaled grads by scale to get real grads
# If backward produces large fp16 grads that overflow -> inf even before unscale
# Check: model gradient in AMP context
mass_amp = d_map.float().detach().requires_grad_(True)  # float32 detached
pred_amp = build_predicted_count_pyramid(mass_amp, (8, 16, 32, 64), 4)
from hpc.losses.count_tree import CountTreeConfig, AdaptiveProbabilisticCountTreeLoss
from hpc.losses.hpc_adaptive import AdaptiveHPCLoss, HPCLossConfig
cfg = HPCLossConfig(
    tree=CountTreeConfig(root_dispersion=50.0, kappa_root64=20.0, kappa_64_32=20.0,
                         kappa_32_16=20.0, kappa_16_8=20.0, dense_threshold_16=2,
                         use_dirichlet_multinomial=True,
                         w_root_nb=1.0, w_root64=1.0, w_64_32=1.0, w_32_16=1.0, w_16_8=1.0),
    hard_zero_weight=0.10, local_contrast_weight=0.05,
)
criterion = AdaptiveHPCLoss(cfg, feature_dim=32).to(device)
total, logs = criterion(mass=mass_amp, p4=aux["p4"], target_pyramid=tgt_cuda)
print(f"Full criterion total: {total.item():.4f}, finite={torch.isfinite(total).item()}")
total.backward()
print(f"Full grad wrt mass: max={mass_amp.grad.abs().max():.2f}, finite={torch.isfinite(mass_amp.grad).all().item()}")

# Step 5: The model backward THROUGH fp16 layers
# The key insight: d_map is float32 (our loss output), but inside model forward,
# the backbone features ARE fp16. When loss gradient flows backward through the model:
# - loss gradient wrt d_map is float32 (large)
# - softplus backward: sigmoid in float32 -> OK
# - head_out backward: input (float32 features) * grad_output (float32) -> float32
# So gradients should be float32 throughout since d_map is float32
# Wait - let me check: does AMP cast the head_out computation to float16?

print("\n=== Checking dtype of model operations under AMP ===")
model.eval()
with torch.amp.autocast("cuda", enabled=True):
    d_map2, aux2 = model(images_cuda, return_aux=True)
print(f"d_map under AMP: dtype={d_map2.dtype}")
# head_out: 1x1 conv + softplus. Under AMP, conv is fp16 by default.
# But softplus is fp32. Let me check what dtype the head_out conv output is.
model.train()
