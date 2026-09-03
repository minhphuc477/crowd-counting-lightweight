"""Generate configuration YAMLs for the 6-model MICF Pilot Suite (B1 to B6)."""

from pathlib import Path
import yaml

BASE_CFG = {
    "dataset": {
        "name": "sha",
        "part": "part_A",
        "root": "./data/ShanghaiTech",
        "crop_size": 256,
        "coordinate_base": 0,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    },
    "model": {
        "backbone": "mobilenetv4_conv_small_050.e3000_r224_in1k",
        "pretrained": True,
        "neck_width": 32,
        "context_dilations": [1, 2, 3],
        "output_stride": 16,
        "eps_d": 1e-8,
    },
    "augmentation": {
        "scale_range": [0.7, 1.3],
        "flip_prob": 0.5,
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 0.0002,
        "backbone_lr_scale": 0.1,
        "weight_decay": 0.0001,
        "grad_clip": 500.0,
    },
    "schedule": {
        "epochs": 100,
        "warmup_epochs": 5,
    },
    "training": {
        "batch_size": 16,
        "drop_last": True,
        "amp": True,
        "num_workers": 2,
        "evaluate_every": 2,
    },
}

MODELS = {
    "B1": {
        "desc": "Local Count Baseline (L1 loss on Y)",
        "model": {"head_type": "local", "use_integral_context": False},
        "loss": {"mode": "local_l1"},
    },
    "B2": {
        "desc": "Local Output + Integral Loss (L1 loss on P Y)",
        "model": {"head_type": "local", "use_integral_context": False},
        "loss": {"mode": "integral_on_local"},
    },
    "B3": {
        "desc": "Direct Cumulative MICF Naive (L1 loss on C, lambda_valid=0)",
        "model": {"head_type": "cumulative", "use_integral_context": False},
        "loss": {"mode": "micf", "lambda_valid": 0.0, "lambda_count": 0.5},
    },
    "B4": {
        "desc": "Direct Cumulative MICF + Validity (lambda_valid=1.0)",
        "model": {"head_type": "cumulative", "use_integral_context": False},
        "loss": {"mode": "micf", "lambda_valid": 1.0, "lambda_count": 0.5},
    },
    "B5": {
        "desc": "MICF-v2 Full (Directional Context + Validity)",
        "model": {"head_type": "cumulative", "use_integral_context": True},
        "loss": {"mode": "micf", "lambda_valid": 1.0, "lambda_count": 0.5},
    },
    "B6": {
        "desc": "Local Count + Directional Context (Ablation Control)",
        "model": {"head_type": "local", "use_integral_context": True},
        "loss": {"mode": "local_l1"},
    },
}


def main() -> None:
    configs_dir = Path("configs/pilot_micf")
    configs_dir.mkdir(parents=True, exist_ok=True)

    for m_id, spec in MODELS.items():
        cfg = yaml.safe_load(yaml.safe_dump(BASE_CFG))
        cfg["experiment"] = {
            "name": f"pilot_micf_{m_id.lower()}",
            "model_id": m_id,
            "description": spec["desc"],
            "seed": 42,
            "save_dir": f"./runs/pilot_micf/{m_id.lower()}",
        }
        cfg["model"].update(spec["model"])
        cfg["loss"] = spec["loss"]

        out_file = configs_dir / f"{m_id.lower()}.yaml"
        with open(out_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"Generated {out_file} ({spec['desc']})")


if __name__ == "__main__":
    main()
