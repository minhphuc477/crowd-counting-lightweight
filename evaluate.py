"""Evaluate NTPC model checkpoints on official crowd counting splits."""

from __future__ import annotations

import argparse
import json
import os
import torch
import yaml

from hpc.data.factory import build_evaluation_dataset
from hpc.evaluation.counting import evaluate_counting
from hpc.models.factory import assert_checkpoint_compatible, build_model_from_config


def evaluate_model(
    checkpoint_path: str,
    config_path: str,
    output_json: str = "eval_results.json",
    split: str | None = None,
    tile_size: int | None = None,
    tile_halo: int = 64,
) -> dict:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Evaluation checkpoint file not found: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation on device: {device}")

    # 1. Load Model via Centralized Factory
    # The task checkpoint fully defines weights; avoid redundant pretrained network I/O.
    model = build_model_from_config(cfg, load_pretrained=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    assert_checkpoint_compatible(ckpt, cfg)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    model.eval()

    # 2. Load Evaluation Dataset
    val_dataset, resolved_split = build_evaluation_dataset(cfg, split=split)
    print(f"Loaded {len(val_dataset)} evaluation samples.")

    if len(val_dataset) == 0:
        print("No evaluation samples found.")
        return {}

    metrics = evaluate_counting(
        model, val_dataset, device, tile_size=tile_size, tile_halo=tile_halo
    )
    metrics["selection_split"] = resolved_split
    metrics["inference"] = (
        {"mode": "full_image"}
        if tile_size is None
        else {"mode": "tiled", "tile_size": tile_size, "tile_halo": tile_halo}
    )

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
    parser.add_argument("--tile-size", type=int, help="Explicit tiled-inference core size (multiple of 16)")
    parser.add_argument("--tile-halo", type=int, default=64, help="Tiled-inference context halo (multiple of 16)")
    args = parser.parse_args()

    evaluate_model(
        args.checkpoint,
        args.config,
        args.output,
        split=args.split,
        tile_size=args.tile_size,
        tile_halo=args.tile_halo,
    )
