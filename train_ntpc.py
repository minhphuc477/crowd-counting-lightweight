"""Train the Neural Tree-Polya crowd counter with matched R0--R5 protocols."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import time
from typing import Dict, Iterable, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from hpc.data.common import ntpc_collate_fn
from hpc.data.nwpu import NWPUDataset
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.sampler import build_density_luminance_sampler
from hpc.data.sha import ShanghaiTechDataset
from hpc.evaluation.counting import evaluate_counting
from hpc.losses.ntpc import NTPCConfig, NTPCLoss
from hpc.models.factory import build_model_from_config
from hpc.utils.seed import make_generator, seed_everything, seed_worker


def get_runtime_metadata() -> dict:
    """Capture environment and git commit metadata for reproducibility."""
    git_sha = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "timm": str(timm.__version__),
        "cuda": str(torch.version.cuda) if torch.cuda.is_available() else None,
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() and torch.backends.cudnn.version() is not None else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_sha": git_sha,
    }


def _dataset_common(ds_cfg: dict, aug_cfg: dict, is_train: bool) -> dict:
    res = {
        "crop_size": int(ds_cfg.get("crop_size", 256)),
        "is_train": is_train,
        "scale_range": tuple(aug_cfg.get("scale_range", [0.7, 1.3])),
        "flip_prob": float(aug_cfg.get("flip_prob", 0.5)) if is_train else 0.0,
        "image_mean": ds_cfg.get("image_mean", [0.485, 0.456, 0.406]),
        "image_std": ds_cfg.get("image_std", [0.229, 0.224, 0.225]),
    }
    if "coordinate_base" in ds_cfg:
        res["coordinate_base"] = int(ds_cfg["coordinate_base"])
    return res


def build_datasets(cfg: dict):
    """Use official train/evaluation partitions; never create a custom split."""
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    name = str(ds_cfg.get("name", "sha")).lower().replace("-", "_")
    train_args = _dataset_common(ds_cfg, aug_cfg, True)
    eval_args = _dataset_common(ds_cfg, aug_cfg, False)
    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = ds_cfg.get("part", "part_B" if name.endswith("_b") else "part_A")
        train = ShanghaiTechDataset(
            root=ds_cfg["root"], part=part, split="train_data", **train_args
        )
        evaluation = ShanghaiTechDataset(
            root=ds_cfg["root"], part=part, split="test_data", **eval_args
        )
        selection_split = "test_data"
    elif name in {"qnrf", "ucf_qnrf"}:
        train = UCFQNRFDataset(root=ds_cfg["root"], split="Train", **train_args)
        evaluation = UCFQNRFDataset(root=ds_cfg["root"], split="Test", **eval_args)
        selection_split = "Test"
    elif name == "nwpu":
        train = NWPUDataset(
            root=ds_cfg["root"],
            split="train",
            split_file=ds_cfg.get("train_split_file"),
            **train_args,
        )
        evaluation = NWPUDataset(
            root=ds_cfg["root"],
            split="val",
            split_file=ds_cfg.get("val_split_file"),
            **eval_args,
        )
        selection_split = "val"
    else:
        raise ValueError(
            f"Unsupported dataset '{name}'. This trainer supports SHA, UCF-QNRF and NWPU; "
            "JHU and UCF-CC50 require their official dataset/fold loaders."
        )
    return train, evaluation, selection_split


@torch.no_grad()
def estimate_crop_statistics(
    dataset,
    max_samples: int | None = None,
    crops_per_image: int = 3,
) -> dict:
    """Estimate initialization and dense threshold from training crops across the full dataset."""
    count_values = []
    positive_y16 = []
    limit = len(dataset) if max_samples is None else min(len(dataset), int(max_samples))
    if limit <= 0:
        raise ValueError("Training dataset is empty")
    for index in range(limit):
        for _ in range(crops_per_image):
            sample = dataset[index]
            count_values.append(sample["gt_count"].float().reshape(()))
            if 16 in sample["gt_blocks"]:
                cells = sample["gt_blocks"][16].float().reshape(-1)
                positive_y16.append(cells[cells > 0])
    counts = torch.stack(count_values)
    positive = torch.cat(positive_y16) if positive_y16 else torch.empty(0)
    return {
        "samples": len(count_values),
        "mean_crop_count": float(counts.mean()),
        "count_mean": float(counts.mean()),
        "count_variance": float(counts.var(unbiased=counts.numel() > 1)),
        # interpolation="higher" ensures threshold is an integer present in the data,
        # avoiding fractional values like 3.4 for integer-valued count cells.
        "dense_threshold_q85": (
            int(torch.quantile(positive, 0.85, interpolation="higher").item())
            if positive.numel()
            else 1
        ),
    }


def component_gradient_norms(
    components: Dict[str, torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    names: Tuple[str, ...],
) -> Dict[str, float]:
    """Measure true, unscaled full-model gradient norms for objective terms."""
    params = tuple(p for p in parameters if p.requires_grad)
    result: Dict[str, float] = {}
    for name in names:
        value = components[name]
        grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
        squared = sum(
            float(grad.detach().float().square().sum()) for grad in grads if grad is not None
        )
        result[f"grad_{name}"] = math.sqrt(squared)
    return result


def _grad_names_for_mode(mode: str) -> Tuple[str, ...]:
    """Return the component names to audit per gradient for each specific NTPC mode."""
    mapping = {
        "r0_exact": ("root_magnitude", "exact_regression"),
        "r1_deterministic": ("root_magnitude", "deterministic_alloc"),
        "r2_flat_dm": ("root_magnitude", "flat_16"),
        "r3_multinomial_tree": ("root_magnitude", "multinomial_tree"),
        "r4_dtm_tree16": ("root_magnitude", "root_to_64", "64_to_32", "32_to_16"),
        "r4_dtm_tree8": ("root_magnitude", "root_to_64", "64_to_32", "32_to_16", "16_to_8"),
        "r4_dtm_tree4": ("root_magnitude", "root_to_64", "64_to_32", "32_to_16", "16_to_8", "8_to_4"),
        "r5_full_ntpc": ("root_magnitude", "root_to_64", "64_to_32", "32_to_16", "16_to_8_dense"),
    }
    if mode not in mapping:
        raise ValueError(f"Unknown mode '{mode}' for gradient auditing")
    return mapping[mode]


def _append_csv(path: str, row: dict, fieldnames: list[str]) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Tree-Polya Crowd Counting")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # Scratch and config fail-fast invariants
    for forbidden in (
        "resume",
        "teacher_checkpoint",
        "distillation",
        "warm_start",
        "pretrained_checkpoint",
    ):
        if cfg.get(forbidden):
            raise ValueError(
                f"NTPC strict-scratch invariant: '{forbidden}' is forbidden in config"
            )

    training_cfg = cfg.get("training", {})
    if "epochs" in training_cfg:
        raise ValueError(
            "training.epochs is invalid and ambiguous; specify total epochs under schedule.epochs"
        )

    exp_cfg = cfg["experiment"]
    seed = int(exp_cfg.get("seed", 42))
    seed_everything(seed)
    save_dir = os.path.abspath(exp_cfg.get("save_dir", "./runs/ntpc_experiment"))

    if os.path.isdir(save_dir) and any(os.scandir(save_dir)):
        if not args.overwrite:
            raise FileExistsError(
                f"Run directory is not empty: {save_dir}. Use --overwrite explicitly."
            )
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.yaml"), "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, evaluation_ds, selection_split = build_datasets(cfg)
    stats_cfg = cfg.get("statistics", {})

    # Isolated statistics seed so dataset initialization and tau_D are deterministic across runs
    stats_seed = int(stats_cfg.get("seed", 12345))
    seed_everything(stats_seed)
    crop_stats = estimate_crop_statistics(
        train_ds,
        max_samples=stats_cfg.get("max_samples"),
        crops_per_image=int(stats_cfg.get("crops_per_image", 3)),
    )
    # Restore experiment seed before model/loader instantiation
    seed_everything(seed)

    sampler = None
    if cfg.get("sampler", {}).get("weighted", False):
        sampler_cfg = cfg["sampler"]
        sampler, _ = build_density_luminance_sampler(
            train_ds.image_paths,
            train_ds.points_list,
            num_density_bins=int(sampler_cfg.get("density_bins", 5)),
            num_luminance_bins=int(sampler_cfg.get("luminance_bins", 4)),
            power=float(sampler_cfg.get("power", 0.5)),
        )

    num_workers = int(training_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(training_cfg.get("batch_size", 16)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=ntpc_collate_fn,
        pin_memory=device.type == "cuda",
        drop_last=bool(training_cfg.get("drop_last", True)),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=make_generator(seed),
        persistent_workers=num_workers > 0,
    )
    if len(train_loader) == 0:
        raise ValueError("Training loader has zero batches; reduce batch_size or disable drop_last")

    model_cfg = cfg["model"]
    if model_cfg.get("pretrained", False) or model_cfg.get("init_checkpoint"):
        raise ValueError("NTPC matched ablations must start from scratch; pretrained/init_checkpoint is forbidden")
    model = build_model_from_config(cfg).to(device)
    model.init_head_bias_from_data(
        crop_stats["mean_crop_count"], int(cfg["dataset"].get("crop_size", 256)), 4
    )

    loss_cfg = cfg.get("loss", {})
    shared_kappa = loss_cfg.get("kappa_shared")

    def kappa(name: str, default: float = 20.0) -> float:
        return float(loss_cfg.get(name, shared_kappa if shared_kappa is not None else default))

    threshold_value = loss_cfg.get("dense_threshold_16", "auto")
    dense_threshold = (
        crop_stats["dense_threshold_q85"]
        if threshold_value is None or str(threshold_value).lower() == "auto"
        else float(threshold_value)
    )
    criterion = NTPCLoss(NTPCConfig(
        mode=loss_cfg.get("mode", "r4_dtm_tree16"),
        root_loss=loss_cfg.get("root_loss", "nb"),
        root_dispersion=float(stats_cfg.get("root_dispersion", 50.0)),
        kappa_root64=kappa("kappa_root64"),
        kappa_64_32=kappa("kappa_64_32"),
        kappa_32_16=kappa("kappa_32_16"),
        kappa_16_8=kappa("kappa_16_8"),
        kappa_8_4=kappa("kappa_8_4"),
        kappa_flat16=kappa("kappa_flat16"),
        dense_threshold_16=dense_threshold,
        w_root_nb=float(loss_cfg.get("w_root_nb", 1.0)),
        w_root64=float(loss_cfg.get("w_root64", 1.0)),
        w_64_32=float(loss_cfg.get("w_64_32", 1.0)),
        w_32_16=float(loss_cfg.get("w_32_16", 1.0)),
        w_16_8=float(loss_cfg.get("w_16_8", 1.0)),
        w_8_4=float(loss_cfg.get("w_8_4", 1.0)),
        w_flat_16=float(loss_cfg.get("w_flat_16", 1.0)),
        w_exact_regression=float(loss_cfg.get("w_exact_regression", 1.0)),
        w_deterministic_alloc=float(loss_cfg.get("w_deterministic_alloc", 1.0)),
    )).to(device)

    optimizer_cfg = cfg["optimizer"]
    optimizer_name = str(optimizer_cfg.get("name", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer '{optimizer_cfg.get('name')}'. Only 'AdamW' is supported."
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["schedule"]["epochs"])
    warmup_epochs = int(cfg["schedule"].get("warmup_epochs", 25))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.10 + 0.45 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=float(training_cfg.get("init_scale", 256)),
        enabled=amp_enabled,
    )
    grad_clip = float(optimizer_cfg.get("grad_clip", 5.0))
    evaluate_every = int(training_cfg.get("evaluate_every", training_cfg.get("validate_every", 5)))
    gradient_every = int(training_cfg.get("gradient_audit_every", 50))
    grad_names = _grad_names_for_mode(criterion.cfg.mode)

    train_fields = [
        "epoch", "loss", "root_magnitude", "root_to_64", "64_to_32", "32_to_16",
        "16_to_8", "16_to_8_dense", "8_to_4", "flat_16", "multinomial_tree",
        "deterministic_alloc", "exact_regression", *[f"grad_{x}" for x in grad_names], "lr",
    ]
    val_fields = [
        "epoch", "mae", "rmse", "nae", "sparse_mae", "medium_mae", "dense_mae",
        "sparse_bias", "medium_bias", "dense_bias", "lr",
    ]
    train_csv, val_csv = os.path.join(save_dir, "train.csv"), os.path.join(save_dir, "val.csv")
    for path, fields in ((train_csv, train_fields), (val_csv, val_fields)):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    print(
        f"Device={device}; params={sum(p.numel() for p in model.parameters()):,}; "
        f"mode={criterion.cfg.mode}; selection={selection_split}; stats={crop_stats}; "
        f"dense_threshold_16={dense_threshold:.3f}; exact_joint_nll={criterion.is_exact_joint_nll}"
    )
    best_mae, best_epoch = float("inf"), 0
    component_names = train_fields[2:13]
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        running_loss = torch.zeros((), device=device)
        running = {name: torch.zeros((), device=device) for name in component_names}
        epoch_grads = {f"grad_{name}": float("nan") for name in grad_names}
        audit_gradients = epoch == 1 or (gradient_every > 0 and epoch % gradient_every == 0)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            targets = {int(k): value.to(device, non_blocking=True) for k, value in batch["gt_blocks"].items()}
            targets["N"] = batch["gt_count"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                mass = model(images)
                loss, logs, components = criterion(mass, targets, return_components=True)
            if audit_gradients and step == 0:
                epoch_grads = component_gradient_norms(components, model.parameters(), grad_names)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            running_loss += loss.detach()
            for name in component_names:
                running[name] += logs.get(name, mass.new_zeros(()))
        scheduler.step()
        steps = len(train_loader)
        train_row = {
            "epoch": epoch,
            "loss": float((running_loss / steps).cpu()),
            **{name: float((running[name] / steps).cpu()) for name in component_names},
            **epoch_grads,
            "lr": optimizer.param_groups[0]["lr"],
        }
        _append_csv(train_csv, train_row, train_fields)

        if epoch % evaluate_every == 0 or epoch == epochs:
            metrics = evaluate_counting(model, evaluation_ds, device)
            val_row = {
                "epoch": epoch,
                "mae": metrics["mae"], "rmse": metrics["rmse"], "nae": metrics.get("nae", 0.0),
                "sparse_mae": metrics.get("bin_sparse_mae", float("nan")),
                "medium_mae": metrics.get("bin_medium_mae", float("nan")),
                "dense_mae": metrics.get("bin_dense_mae", float("nan")),
                "sparse_bias": metrics.get("bin_sparse_bias", float("nan")),
                "medium_bias": metrics.get("bin_medium_bias", float("nan")),
                "dense_bias": metrics.get("bin_dense_bias", float("nan")),
                "lr": optimizer.param_groups[0]["lr"],
            }
            _append_csv(val_csv, val_row, val_fields)
            if metrics["mae"] < best_mae:
                best_mae, best_epoch = metrics["mae"], epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_res": metrics,
                    "config": cfg,
                    "resolved_crop_statistics": crop_stats,
                    "selection_split": selection_split,
                    "is_exact_joint_nll": criterion.is_exact_joint_nll,
                    "initialization_policy": "scratch",
                    "runtime": get_runtime_metadata(),
                }, os.path.join(save_dir, "best.pt"))
            print(
                f"Epoch {epoch:04d}/{epochs} loss={train_row['loss']:.3f} "
                f"MAE={metrics['mae']:.3f} RMSE={metrics['rmse']:.3f} "
                f"best={best_mae:.3f}@{best_epoch} time={time.time()-started:.1f}s"
            )
        else:
            print(
                f"Epoch {epoch:04d}/{epochs} loss={train_row['loss']:.3f} "
                f"lr={train_row['lr']:.2e} time={time.time()-started:.1f}s"
            )

    print(f"Training complete. Best MAE={best_mae:.3f} at epoch {best_epoch} on {selection_split}.")


if __name__ == "__main__":
    main()
