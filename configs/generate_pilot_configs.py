"""Generate all 6 MICF pilot configs matching the design doc specifications.

Design doc defaults (sec.39, 49):
- optimizer: AdamW, lr=1e-4, weight_decay=1e-4, grad_clip=5.0
- training: epochs=1000, warmup_epochs=25, seeds=[41,42,43]
- loss: SmoothL1
- augmentation: hflip via dataset (flip_prob=0.5), vflip via training loop (vflip_prob=0.5)
"""
from __future__ import annotations
import os
import yaml

_OUT_DIR = os.path.join(os.path.dirname(__file__), "pilot_micf")
os.makedirs(_OUT_DIR, exist_ok=True)

_COMMON = {
    "dataset": {
        "name": "sha",
        "part": "part_A",
        "root": "./data/ShanghaiTech",
        "crop_size": 256,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std":  [0.5, 0.5, 0.5],
        "coordinate_base": 0,
    },
    "augmentation": {
        "scale_range": [0.7, 1.3],
        "flip_prob": 0.5,       # horizontal flip in dataset
        "vflip_prob": 0.5,      # vertical flip in training loop (orientation balancing)
    },
    "model": {
        "backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "pretrained": True,
        "neck_width": 32,
        "context_dilations": [1, 2, 3],
        "output_stride": 16,
        "eps_d": 1e-8,
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 1e-4,           # design doc sec.39
        "backbone_lr_scale": 0.1,
        "weight_decay": 1e-4,
        "grad_clip": 5.0,     # design doc sec.39
    },
    "schedule": {
        "epochs": 1000,       # design doc sec.39/49
        "warmup_epochs": 25,  # design doc sec.39
    },
    "training": {
        "amp": True,
        "batch_size": 16,
        "num_workers": 2,
        "drop_last": True,
        "evaluate_every": 5,
    },
}

_VARIANTS = [
    {
        "id": "B1",
        "desc": "Local Count Baseline (SmoothL1 on Y)",
        "model": {"head_type": "local",       "use_integral_context": False},
        "loss":  {"mode": "local_smooth_l1"},
    },
    {
        "id": "B2",
        "desc": "Local Output + Integral Loss (SmoothL1 on P(Y_hat) vs P(Y))",
        "model": {"head_type": "local",       "use_integral_context": False},
        "loss":  {"mode": "integral_on_local", "loss_type": "smooth_l1"},
    },
    {
        "id": "B3",
        "desc": "Direct Cumulative MICF Naive (SmoothL1 on C, lambda_valid=0)",
        "model": {"head_type": "cumulative",  "use_integral_context": False},
        "loss":  {"mode": "micf_naive", "field_loss": "smooth_l1", "lambda_valid": 0.0},
    },
    {
        "id": "B4",
        "desc": "Direct Cumulative MICF + Validity (lambda_valid=1.0)",
        "model": {"head_type": "cumulative",  "use_integral_context": False},
        "loss":  {"mode": "micf_valid", "field_loss": "smooth_l1", "lambda_valid": 1.0},
    },
    {
        "id": "B5",
        "desc": "MICF-v2 Full (4-dir Directional Context + Validity)",
        "model": {"head_type": "cumulative",  "use_integral_context": True},
        "loss":  {"mode": "micf_v2_full", "field_loss": "smooth_l1", "lambda_valid": 1.0},
    },
    {
        "id": "B6",
        "desc": "Local Count + Directional Context (Ablation Control)",
        "model": {"head_type": "local",       "use_integral_context": True},
        "loss":  {"mode": "local_smooth_l1"},
    },
]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


for v in _VARIANTS:
    bid = v["id"].lower()
    cfg = _deep_merge(_COMMON, {
        "model": v["model"],
        "loss":  v["loss"],
        "experiment": {
            "name":        f"pilot_micf_{bid}",
            "model_id":    v["id"],
            "description": v["desc"],
            "seed":        42,
            "save_dir":    f"./runs/pilot_micf/{bid}",
        },
    })
    out_path = os.path.join(_OUT_DIR, f"{bid}.yaml")
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
    print(f"Wrote {out_path}")

print("Done.")
