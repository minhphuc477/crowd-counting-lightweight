#!/usr/bin/env python3
"""Generate and validate all 14 JHU-CROWD++ benchmark & ablation configs."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import copy
import yaml

from rmr_v3.model import RMRv3Config
from rmr_v3.losses import RMRv3LossConfig

OUT_DIR = Path("configs/benchmarks/ablations_jhu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/benchmarks/exp_jhu_direct_scratch_104k.yaml", "r", encoding="utf-8") as f:
    base_cfg = yaml.safe_load(f)

variants = {
    # Pillar 1: Spatial Allocation Supervision (Loss Formulation)
    "exp_jhu_abl_a1_flagship_104k": {},
    "exp_jhu_abl_a2_pure_bayesian_104k": {
        "loss": {
            "allocation_loss_type": "bayesian",
            "lambda_bayesian": 1.0,
            "lambda_radon_ot": 0.0,
        }
    },
    "exp_jhu_abl_a3_pure_radon_ot_104k": {
        "loss": {
            "allocation_loss_type": "radon_ot",
            "lambda_bayesian": 0.0,
            "lambda_radon_ot": 1.0,
        }
    },
    "exp_jhu_abl_a4_dm16_allocation_104k": {
        "loss": {
            "allocation_loss_type": "flat_dm16",
            "lambda_flat_dm16": 0.1,
            "lambda_bayesian": 0.0,
            "lambda_radon_ot": 0.0,
        }
    },
    "exp_jhu_abl_a5_gaussian_mse_104k": {
        "loss": {
            "allocation_loss_type": "none",
            "lambda_cell": 1.0,
            "lambda_count": 0.0,
            "lambda_bayesian": 0.0,
            "lambda_radon_ot": 0.0,
        }
    },

    # Pillar 2: Macro Count & Uncertainty Regularization
    "exp_jhu_abl_b1_count_l1_104k": {
        "loss": {
            "count_loss_mode": "l1",
            "lambda_count": 1.0,
        }
    },
    "exp_jhu_abl_b2_no_count_loss_104k": {
        "loss": {
            "lambda_count": 0.0,
        }
    },
    "exp_jhu_abl_b3_count_nb_disp10_104k": {
        "loss": {
            "count_loss_mode": "nb",
            "count_nb_dispersion": 10.0,
        }
    },

    # Pillar 3: Architecture & Neck (Resolution & Dilations)
    "exp_jhu_abl_c1_stride4_104k": {
        "model": {
            "output_stride": 4,
            "subpixel_stride2": False,
            "init_m0": 0.005610,
        },
        "loss": {
            "output_stride": 4,
        }
    },
    "exp_jhu_abl_c2_no_neck_104k": {
        "model": {
            "neck_type": "aspp_lite",
            "aspp_dilations": [1],
            "use_aspp_gap": False,
        }
    },
    "exp_jhu_abl_c3_radon_8proj_104k": {
        "loss": {
            "radon_ot_projections": 8,
        }
    },

    # Pillar 4: Multi-Scale & Environmental Augmentations (JHU Specific!)
    "exp_jhu_abl_d1_no_crowd_aware_crop_104k": {
        "data": {
            "crowd_aware_crop": False,
        }
    },
    "exp_jhu_abl_d2_maxsize_1536_104k": {
        "data": {
            "max_size": 1536,
        }
    },
    "exp_jhu_abl_d3_no_weather_aug_104k": {
        "data": {
            "grayscale_prob": 0.0,
            "gamma_jitter": [1.0, 1.0],
            "random_erasing_prob": 0.0,
            "saturation_jitter": 0.0,
        }
    },
    "exp_jhu_abl_d4_crop384_104k": {
        "data": {
            "crop_size": 384,
        }
    },
}


def main():
    print(f"Generating {len(variants)} JHU ablation configs...")
    for name, overrides in variants.items():
        cfg = copy.deepcopy(base_cfg)
        for section, kv in overrides.items():
            if isinstance(kv, dict) and section in cfg:
                cfg[section].update(kv)
            else:
                cfg[section] = kv
        cfg["output_dir"] = f"runs/benchmarks/ablations_jhu/{name}"
        
        # Validate schema contracts
        m_cfg = RMRv3Config.from_dict(cfg["model"], pretrained=False)
        l_cfg = RMRv3LossConfig.from_dict(cfg["loss"])
        
        out_path = OUT_DIR / f"{name}.yaml"
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"  [OK] {out_path.name:45s} -> output_dir: {cfg['output_dir']}")

    print("\nAll 14 JHU ablation configs generated and validated successfully!")


if __name__ == "__main__":
    main()
