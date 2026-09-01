"""Reproduce and localize non-finite NTPC gradients on the first training batch."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hpc.data.common import ntpc_collate_fn
from hpc.losses.factory import build_ntpc_criterion_from_config
from hpc.models.factory import build_model_from_config
from hpc.utils.seed import make_generator, seed_everything
from train_ntpc import build_datasets, build_optimizer, estimate_crop_statistics


def _grad_report(model: torch.nn.Module) -> tuple[float, list[str]]:
    squared = torch.zeros((), device=next(model.parameters()).device)
    bad = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not torch.isfinite(grad).all():
            bad.append(name)
        squared += torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0).square().sum()
    return float(torch.sqrt(squared).cpu()), bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-steps", type=int, default=0)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    seed = int(cfg["experiment"].get("seed", 42))
    stats_cfg = cfg.get("statistics", {})
    seed_everything(int(stats_cfg.get("seed", 12345)))
    train_ds, _, _ = build_datasets(cfg)
    stats = estimate_crop_statistics(
        train_ds,
        max_samples=stats_cfg.get("max_samples"),
        crops_per_image=int(stats_cfg.get("crops_per_image", 3)),
    )
    seed_everything(seed)
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
        shuffle=True,
        num_workers=0,
        collate_fn=ntpc_collate_fn,
        drop_last=True,
        generator=make_generator(seed),
    )
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = batch["image"].to(device)
    targets = {int(k): value.to(device) for k, value in batch["gt_blocks"].items()}
    targets["N"] = batch["gt_count"].to(device)
    model = build_model_from_config(cfg).to(device)
    model.init_head_bias_from_data(stats["mean_crop_count"], int(cfg["dataset"].get("crop_size", 256)), 4)
    criterion = build_ntpc_criterion_from_config(cfg, crop_statistics=stats).to(device)

    print(f"device={device} gt_min={float(targets['N'].min())} gt_max={float(targets['N'].max())} stats={stats}")
    reference_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    for amp in (False, True):
        if amp and device.type != "cuda":
            continue
        model.load_state_dict(reference_state, strict=True)
        model.train()
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            mass = model(images)
            loss, logs, components = criterion(mass, targets, return_components=True, validate_targets=True)
        print(f"amp={amp} mass=[{float(mass.min()):.3e},{float(mass.max()):.3e}] loss={float(loss):.6g}")
        for name, component in components.items():
            model.zero_grad(set_to_none=True)
            component.backward(retain_graph=True)
            norm, bad = _grad_report(model)
            print(f"  {name:20s} value={float(logs[name]):12.6g} grad={norm:12.6g} bad={bad[:5]}")
        model.zero_grad(set_to_none=True)
        scale = float(cfg.get("training", {}).get("init_scale", 256.0)) if amp else 1.0
        (loss * scale).backward()
        if amp:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(scale)
        norm, bad = _grad_report(model)
        print(f"  total scaled={scale:g} finite={not bad and math.isfinite(norm)} grad={norm:.6g} bad={bad[:20]}")
        model.zero_grad(set_to_none=True)

    if args.train_steps > 0:
        model.load_state_dict(reference_state, strict=True)
        # Recreate the loader iterator because the first iterator above advances the
        # shared augmentation RNG even though it does not mutate model weights.
        seed_everything(seed)
        loader = DataLoader(
            train_ds,
            batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
            shuffle=True,
            num_workers=0,
            collate_fn=ntpc_collate_fn,
            drop_last=True,
            generator=make_generator(seed),
        )
        optimizer = build_optimizer(model, cfg["optimizer"])
        scaler = torch.amp.GradScaler(
            "cuda",
            init_scale=float(cfg.get("training", {}).get("init_scale", 256.0)),
            enabled=device.type == "cuda",
        )
        clip = float(cfg["optimizer"].get("grad_clip", 5.0))
        model.train()
        for step, train_batch in enumerate(loader, start=1):
            if step > args.train_steps:
                break
            train_images = train_batch["image"].to(device)
            train_targets = {int(k): value.to(device) for k, value in train_batch["gt_blocks"].items()}
            train_targets["N"] = train_batch["gt_count"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                train_mass = model(train_images)
                train_loss, _ = criterion(train_mass, train_targets, validate_targets=False)
            old_scale = float(scaler.get_scale())
            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)
            norm, bad = _grad_report(model)
            if not bad and math.isfinite(norm):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            new_scale = float(scaler.get_scale())
            print(
                f"step={step:03d} loss={float(train_loss):.6g} gt_max={float(train_targets['N'].max()):.0f} "
                f"norm={norm:.6g} bad={bad[:5]} scale={old_scale:g}->{new_scale:g} "
                f"skipped={new_scale < old_scale}"
            )

if __name__ == "__main__":
    main()
