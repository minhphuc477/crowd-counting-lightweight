"""Generate receptive-field sweep configs (design doc sec.25).

Varies context_dilations at the P16 neck stage while holding neck_width
fixed, isolating receptive field from parameter count. Answers: how much
context is needed before cumulative prediction (B5) starts to beat the
local baseline (B1)?
"""
from __future__ import annotations
import os
import yaml

_OUT_DIR = os.path.join(os.path.dirname(__file__), "rf_sweep")
os.makedirs(_OUT_DIR, exist_ok=True)

_DILATION_SCHEDULES = {
    "d111": [1, 1, 1],
    "d123": [1, 2, 3],   # current default
    "d135": [1, 3, 5],
}

_COMMON = {
    "dataset": {
        "name": "sha", "part": "part_A", "root": "./data/ShanghaiTech",
        "crop_size": 256, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
        "coordinate_base": 0,
    },
    "augmentation": {"scale_range": [0.7, 1.3], "flip_prob": 0.5, "vflip_prob": 0.5},
    "model": {
        "backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "pretrained": True, "neck_width": 32,
        "output_stride": 16, "eps_d": 1e-8,
    },
    "optimizer": {"name": "AdamW", "lr": 1e-4, "backbone_lr_scale": 0.1, "weight_decay": 1e-4, "grad_clip": 5.0},
    "schedule": {"epochs": 1000, "warmup_epochs": 25},
    "training": {"amp": True, "batch_size": 16, "num_workers": 2, "drop_last": True, "evaluate_every": 5},
}

_VARIANTS = [
    {"id": "B1", "desc": "Local Count Baseline (SmoothL1 on Y)",
     "model": {"head_type": "local", "use_integral_context": False},
     "loss": {"mode": "local_smooth_l1"}},
    {"id": "B5", "desc": "MICF-v2 Full (4-dir Directional Context + Validity)",
     "model": {"head_type": "cumulative", "use_integral_context": True, "context_type": "directional"},
     "loss": {"mode": "micf_v2_full", "field_loss": "smooth_l1", "lambda_valid": 1.0}},
]

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

for dname, dilations in _DILATION_SCHEDULES.items():
    for v in _VARIANTS:
        bid = f"{v['id'].lower()}_{dname}"
        cfg = _deep_merge(_COMMON, {
            "model": {**v["model"], "context_dilations": dilations},
            "loss": v["loss"],
            "experiment": {
                "name": f"rf_sweep_{bid}", "model_id": v["id"],
                "description": f"{v['desc']} [RF sweep, dilations={dilations}]",
                "seed": 42, "save_dir": f"./runs/rf_sweep/{bid}",
            },
        })
        with open(os.path.join(_OUT_DIR, f"{bid}.yaml"), "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        print(f"Wrote {bid}.yaml")
