"""Evaluate NTPC model checkpoints on official crowd counting splits."""

from __future__ import annotations

import argparse
import json
import os
import torch
import yaml

from hpc.data.nwpu import NWPUDataset, resolve_nwpu_split_file
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.sha import ShanghaiTechDataset
from hpc.evaluation.counting import evaluate_counting
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


def build_evaluation_dataset(cfg: dict, split: str | None = None):
    ds_cfg = cfg["dataset"]
    name = str(ds_cfg.get("name", "sha")).lower().replace("-", "_")
    common_args = {
        "crop_size": int(ds_cfg.get("crop_size", 256)),
        "is_train": False,
        "image_mean": ds_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": ds_cfg.get("image_std", [0.229, 0.224, 0.225]),
    }
    if "coordinate_base" in ds_cfg:
        common_args["coordinate_base"] = int(ds_cfg["coordinate_base"])

    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = ds_cfg.get("part", "part_B" if name.endswith("_b") else "part_A")
        eval_split = split or "test_data"
        return ShanghaiTechDataset(
            root=ds_cfg["root"], part=part, split=eval_split, **common_args
        )
    elif name in {"qnrf", "ucf_qnrf"}:
        eval_split = split or "Test"
        return UCFQNRFDataset(root=ds_cfg["root"], split=eval_split, **common_args)
    elif name == "nwpu":
        eval_split = split or "val"
        return NWPUDataset(
            root=ds_cfg["root"],
            split=eval_split,
            split_file=resolve_nwpu_split_file(ds_cfg, eval_split),
            **common_args,
        )
    else:
        raise ValueError(f"Unsupported dataset '{name}'")


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    output_json: str = "eval_results.json",
    split: str | None = None,
) -> dict:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Evaluation checkpoint file not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}")

    # 1. Load Model via Centralized Factory
    model = build_model_from_config(cfg).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert_checkpoint_compatible(ckpt, cfg)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    model.eval()

    # 2. Load Evaluation Dataset
    val_dataset = build_evaluation_dataset(cfg, split=split)
    print(f"Loaded {len(val_dataset)} evaluation samples.")

    if len(val_dataset) == 0:
        print("No evaluation samples found.")
        return {}

    metrics = evaluate_counting(model, val_dataset, device)

    print("\n--- Evaluation Results ---")
    print(f"MAE:  {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"NAE:  {metrics.get('nae', 0.0):.3f}")
    print(f"Bias: {metrics.get('bias', 0.0):.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved evaluation results to: {output_json}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NTPC crowd counter")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to output results JSON")
    parser.add_argument("--split", type=str, default=None, help="Dataset split override (e.g. test_data, val)")
    args = parser.parse_args()

    evaluate_model(args.checkpoint, args.config, args.output, split=args.split)
