"""Post-Training Model Souping & Stochastic Weight Averaging for RMR-v22.

Averages weights across top K validation checkpoints:
    theta_soup = (1/K) * sum_{k=1}^K theta_{(k)}
Finds a wider, flatter basin in the loss landscape with zero additional inference parameters.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from rmr_core.evaluation import evaluate_dataset  # noqa: E402
from rmr_v3.config import validate_v3_config  # noqa: E402
from rmr_v3.model import RMRv3, RMRv3Config  # noqa: E402


def compute_model_soup(checkpoint_paths: list[Path]) -> dict[str, Any]:
    """Compute uniform parameter average (model soup) from multiple checkpoints."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths list cannot be empty")

    print(f"Averaging {len(checkpoint_paths)} checkpoints:")
    for p in checkpoint_paths:
        print(f"  - {p}")

    first_ckpt = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    state_key = "model" if "model" in first_ckpt else ("state_dict" if "state_dict" in first_ckpt else None)
    base_state = first_ckpt[state_key] if state_key else first_ckpt

    soup_state = copy.deepcopy(base_state)
    has_ema = isinstance(first_ckpt, dict) and "ema_model" in first_ckpt
    soup_ema = copy.deepcopy(first_ckpt["ema_model"]) if has_ema else None
    k = len(checkpoint_paths)

    for p in checkpoint_paths[1:]:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state = ckpt[state_key] if state_key else ckpt
        for key in soup_state:
            if key in state and soup_state[key].is_floating_point():
                soup_state[key] += state[key]
        if has_ema and "ema_model" in ckpt:
            for key in soup_ema:
                if key in ckpt["ema_model"] and soup_ema[key].is_floating_point():
                    soup_ema[key] += ckpt["ema_model"][key]

    for key in soup_state:
        if soup_state[key].is_floating_point():
            soup_state[key] /= float(k)
    if has_ema:
        for key in soup_ema:
            if soup_ema[key].is_floating_point():
                soup_ema[key] /= float(k)

    soup_ckpt = copy.deepcopy(first_ckpt)
    if state_key:
        soup_ckpt[state_key] = soup_state
    else:
        soup_ckpt = soup_state
    if has_ema:
        soup_ckpt["ema_model"] = soup_ema

    return soup_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model Soup across checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Path to resolved model config YAML")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True, help="Paths to checkpoints to average")
    parser.add_argument("--output", type=str, default="model_soup.pt", help="Path to save averaged checkpoint")
    parser.add_argument("--eval", action="store_true", help="Run canonical test evaluation after souping")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    ckpt_paths = [Path(p) for p in args.checkpoints]
    for p in ckpt_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    soup_ckpt = compute_model_soup(ckpt_paths)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup_ckpt, out_path)
    print(f"Saved model soup checkpoint to: {out_path}")

    if args.eval:
        cfg_path = Path(args.config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)
        validate_v3_config(raw_cfg)

        model_cfg = RMRv3Config.from_dict(raw_cfg["model"], pretrained=False)
        model = RMRv3(model_cfg)
        state = soup_ckpt["model"] if "model" in soup_ckpt else soup_ckpt
        model.load_state_dict(state, strict=True)
        model.to(args.device)
        model.eval()

        val_manifest = raw_cfg["data"]["val_manifest"]
        from rmr_core.data import CrowdManifestDataset, collate_eval
        from torch.utils.data import DataLoader

        stride = int(raw_cfg.get("model", {}).get("output_stride", 4))
        dataset = CrowdManifestDataset(
            Path(val_manifest),
            train=False,
            output_stride=stride,
            data_root=raw_cfg.get("data", {}).get("data_root"),
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_eval)

        rows, metrics = evaluate_dataset(
            model=model,
            loader=loader,
            device=torch.device(args.device),
            output_stride=stride,
        )
        print("\n--- MODEL SOUP EVALUATION SUMMARY ---")
        mae = metrics.get('MAE') or metrics.get('mae', 0.0)
        rmse = metrics.get('RMSE') or metrics.get('rmse', 0.0)
        bias = metrics.get('Bias') or metrics.get('bias', 0.0)
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"Bias: {bias:+.2f}")


if __name__ == "__main__":
    main()
