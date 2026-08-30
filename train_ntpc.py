"""Train the NTPC ablations and the final R6 Neural Polya Allocation Counter."""

from __future__ import annotations

import argparse
import csv
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

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
from hpc.metrics.tree import finalize_tree_diagnostics, tree_allocation_raw_diagnostics
from hpc.models.factory import (
    build_model_from_config,
    validate_pretrained_normalization,
)
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
        squared = value.new_zeros((), dtype=torch.float32)
        for grad in grads:
            if grad is not None:
                squared = squared + grad.detach().float().square().sum()
        # Exactly one device synchronization per audited component.
        result[f"grad_{name}"] = float(torch.sqrt(squared).cpu())
    return result


@torch.no_grad()
def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Return the unscaled global L2 gradient norm without modifying gradients."""
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.zeros((), dtype=torch.float32)
    device = grads[0].device
    squared = torch.zeros((), device=device, dtype=torch.float32)
    for grad in grads:
        squared += grad.float().square().sum()
    return torch.sqrt(squared)


@torch.no_grad()
def nonfinite_gradient_report(
    model: nn.Module,
    max_names: int = 12,
) -> list[str]:
    """Describe parameters containing NaN/Inf gradients for actionable failures."""
    bad: list[str] = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is None or bool(torch.isfinite(grad).all()):
            continue
        bad.append(
            f"{name}: nan={int(torch.isnan(grad).sum())}, "
            f"inf={int(torch.isinf(grad).sum())}, shape={tuple(grad.shape)}"
        )
        if len(bad) >= max_names:
            break
    return bad


def _grad_names_for_mode(mode: str) -> Tuple[str, ...]:
    """Return the component names to audit per gradient for each specific NTPC mode."""
    mapping = {
        "r0_exact": ("root_magnitude", "exact_regression"),
        "r1_deterministic": ("root_magnitude", "deterministic_alloc"),
        "r2_flat_dm": ("root_magnitude", "flat_16"),
        "r6_npac": ("root_magnitude", "flat_16"),
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


def build_optimizer(model: nn.Module, optimizer_cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW with an explicit discriminative LR for a pretrained backbone."""
    optimizer_name = str(optimizer_cfg.get("name", "AdamW")).lower()
    if optimizer_name != "adamw":
        raise ValueError(
            f"Unsupported optimizer '{optimizer_cfg.get('name')}'. Only 'AdamW' is supported."
        )
    base_lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-4))
    backbone_lr_scale = float(optimizer_cfg.get("backbone_lr_scale", 1.0))
    if not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError(f"optimizer.lr must be positive and finite, got {base_lr}")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError(f"optimizer.weight_decay must be non-negative and finite, got {weight_decay}")
    if not math.isfinite(backbone_lr_scale) or not (0.0 < backbone_lr_scale <= 1.0):
        raise ValueError(
            f"optimizer.backbone_lr_scale must be in (0, 1], got {backbone_lr_scale}"
        )

    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    task_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    if not backbone_parameters or not task_parameters:
        raise RuntimeError("Expected non-empty backbone and task parameter groups")
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": base_lr * backbone_lr_scale,
                "name": "backbone",
            },
            {"params": task_parameters, "lr": base_lr, "name": "task"},
        ],
        lr=base_lr,
        weight_decay=weight_decay,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Neural Tree-Polya Crowd Counting")
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    # Only timm backbone pretraining is supported. Resume/warm-start/distillation
    # are intentionally separate experiments and remain forbidden in this trainer.
    FORBIDDEN_INITIALIZATION_KEYS = {
        "resume",
        "teacher_checkpoint",
        "distillation",
        "warm_start",
        "pretrained_checkpoint",
        "init_checkpoint",
    }

    def assert_supported_initialization(obj: Any, path: str = "") -> None:
        if not isinstance(obj, dict):
            return
        for key, value in obj.items():
            full = f"{path}.{key}" if path else key
            if key in FORBIDDEN_INITIALIZATION_KEYS and value not in (None, False, "", 0):
                raise ValueError(
                    f"Unsupported NTPC initialization: '{full}' is forbidden ({value!r})"
                )
            if isinstance(value, dict):
                assert_supported_initialization(value, full)

    assert_supported_initialization(cfg)
    pretrained_spec = validate_pretrained_normalization(cfg)

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
            generator=make_generator(seed),
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
    optimizer = build_optimizer(model, optimizer_cfg)
    epochs = int(cfg["schedule"]["epochs"])
    if epochs <= 0:
        raise ValueError(f"schedule.epochs must be positive, got {epochs}")
    warmup_epochs = int(cfg["schedule"].get("warmup_epochs", 25))
    if warmup_epochs < 0 or warmup_epochs > epochs:
        raise ValueError(f"schedule.warmup_epochs must be in [0, {epochs}], got {warmup_epochs}")

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
    if grad_clip <= 0 or not math.isfinite(grad_clip):
        raise ValueError(f"grad_clip must be positive and finite, got {grad_clip}")
    evaluate_every = int(training_cfg.get("evaluate_every", training_cfg.get("validate_every", 5)))
    if evaluate_every <= 0:
        raise ValueError(f"evaluate_every must be positive, got {evaluate_every}")
    gradient_every = int(training_cfg.get("gradient_audit_every", 50))
    grad_names = _grad_names_for_mode(criterion.cfg.mode)
    tree_levels = {
        "r4_dtm_tree16": ("root_64", "64_32", "32_16"),
        "r4_dtm_tree8": ("root_64", "64_32", "32_16", "16_8"),
        "r4_dtm_tree4": ("root_64", "64_32", "32_16", "16_8", "8_4"),
        "r5_full_ntpc": ("root_64", "64_32", "32_16", "16_8"),
    }.get(criterion.cfg.mode, ())
    tree_fields = []
    for level in tree_levels:
        prefix = f"tree_{level}"
        tree_fields.extend([
            f"{prefix}_active_parents_per_image", f"{prefix}_zero_parent_fraction",
            f"{prefix}_nll_per_active_parent",
        ])
        for group in ("0", "1", "2_4", "5_9", "ge10"):
            tree_fields.extend([
                f"{prefix}_parent_{group}_count", f"{prefix}_parent_{group}_mean_nll",
                f"{prefix}_parent_{group}_mean_prob_l1",
            ])

    train_fields = [
        "epoch", "loss", "root_magnitude", "root_to_64", "64_to_32", "32_to_16",
        "16_to_8", "16_to_8_dense", "8_to_4", "flat_16", "multinomial_tree",
        "deterministic_alloc", "exact_regression", *tree_fields,
        *[f"grad_{x}" for x in grad_names],
        "grad_total_pre_clip", "grad_total_post_clip", "grad_pre_clip_p50",
        "grad_pre_clip_p95", "grad_pre_clip_max", "grad_backbone", "grad_task",
        "clip_fraction", "amp_scale_start", "amp_scale_end", "amp_skipped_steps",
        "amp_skip_fraction", "overflow_gt_mean", "overflow_gt_max",
        "lr", "lr_backbone", "lr_task",
    ]
    overall_diagnostics = [
        "mae", "rmse", "nae", "bias", "gt_mean", "pred_mean", "gt_std", "pred_std",
        "signed_error_mean", "signed_error_median", "under_count_fraction",
        "over_count_fraction", "pred_gt_ratio",
    ]
    density_diagnostics = [
        f"bin_{group}_{metric}"
        for group in ("sparse", "medium", "dense")
        for metric in ("count", "mae", "rmse", "nae", "bias", "pred_gt_ratio")
    ]
    detailed_count_diagnostics = [
        f"bin_{group}_{metric}"
        for group in ("0", "1_10", "11_100", "101_1000", "gt1000")
        for metric in ("count", "mae", "rmse", "nae", "bias")
    ]
    val_metric_fields = [
        *overall_diagnostics, *density_diagnostics, *detailed_count_diagnostics,
        "top10_dense_count", "top10_dense_mae", "top10_dense_rmse",
        "empty_count", "empty_mae", "empty_pred_mean", "empty_pred_p95",
    ]
    val_fields = ["epoch", *val_metric_fields, "lr", "lr_backbone", "lr_task"]
    train_csv, val_csv = os.path.join(save_dir, "train.csv"), os.path.join(save_dir, "val.csv")
    for path, fields in ((train_csv, train_fields), (val_csv, val_fields)):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    print(
        f"Device={device}; params={sum(p.numel() for p in model.parameters()):,}; "
        f"mode={criterion.cfg.mode}; selection={selection_split}; stats={crop_stats}; "
        f"dense_threshold_16={dense_threshold:.3f}; exact_joint_nll={criterion.is_exact_joint_nll}; "
        f"initialization={'pretrained' if pretrained_spec else 'scratch'}",
        flush=True,
    )
    best_mae, best_epoch = float("inf"), 0
    component_names = train_fields[2:13]
    for epoch in range(1, epochs + 1):
        started = time.time()
        learning_rates = {str(group["name"]): float(group["lr"]) for group in optimizer.param_groups}
        lr_used = learning_rates["task"]
        model.train()
        running_loss = torch.zeros((), device=device)
        running = {name: torch.zeros((), device=device) for name in component_names}
        epoch_grads = {f"grad_{name}": float("nan") for name in grad_names}
        tree_raw: dict[str, float] = {}
        pre_clip_norms: list[float] = []
        post_clip_norms: list[float] = []
        backbone_grad_norms: list[float] = []
        task_grad_norms: list[float] = []
        clipped_steps = 0
        amp_skipped_steps = 0
        consecutive_overflows = 0
        overflow_gt_means: list[float] = []
        overflow_gt_maxs: list[float] = []
        amp_scale_start = float(scaler.get_scale()) if amp_enabled else 1.0
        backbone_params = tuple(optimizer.param_groups[0]["params"])
        task_params = tuple(optimizer.param_groups[1]["params"])
        audit_gradients = epoch == 1 or (gradient_every > 0 and epoch % gradient_every == 0)
        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            targets = {int(k): value.to(device, non_blocking=True) for k, value in batch["gt_blocks"].items()}
            targets["N"] = batch["gt_count"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                mass = model(images)
                loss, logs, components = criterion(
                    mass, targets, return_components=True, validate_targets=False
                )
            if tree_levels:
                for name, value in tree_allocation_raw_diagnostics(
                    mass, targets, criterion.cfg, levels=tree_levels
                ).items():
                    tree_raw[name] = tree_raw.get(name, 0.0) + value
            if audit_gradients and step == 0:
                epoch_grads = component_gradient_norms(components, model.parameters(), grad_names)
                bad_components = {k: v for k, v in epoch_grads.items() if not math.isfinite(v)}
                if bad_components:
                    raise FloatingPointError(
                        f"Non-finite unscaled component gradients at epoch={epoch}, step={step}: "
                        f"{bad_components}"
                    )
                print(f"Gradient audit epoch={epoch}: {epoch_grads}", flush=True)
            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                pre_clip = gradient_norm(model.parameters())
                bad_grads = nonfinite_gradient_report(model) if not bool(torch.isfinite(pre_clip)) else []
                if bad_grads:
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    amp_skipped_steps += 1
                    consecutive_overflows += 1
                    gt_n = targets["N"].detach().float()
                    gt_min_val = float(gt_n.min())
                    gt_mean_val = float(gt_n.mean())
                    gt_max_val = float(gt_n.max())
                    overflow_gt_means.append(gt_mean_val)
                    overflow_gt_maxs.append(gt_max_val)
                    print(
                        f"[AMP overflow] epoch={epoch} step={step} "
                        f"loss={float(loss.detach()):.6g} "
                        f"gt=[{gt_min_val:.0f}, {gt_mean_val:.1f}, {gt_max_val:.0f}] "
                        f"mass=[{float(mass.detach().min()):.3e}, {float(mass.detach().max()):.3e}] "
                        f"scale={scale_before:g}->{scale_after:g}; "
                        + "; ".join(bad_grads),
                        flush=True,
                    )
                    if consecutive_overflows >= 3:
                        print(
                            f"[WARNING] consecutive_overflows={consecutive_overflows} >= 3! "
                            f"Current scale={scale_after:g}. Check mixed-precision stability.",
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    running_loss += loss.detach()
                    for name in component_names:
                        running[name] += logs.get(name, mass.new_zeros(()))
                    continue
                else:
                    consecutive_overflows = 0
                pre_value = float(pre_clip.cpu())
                backbone_value = float(gradient_norm(backbone_params).cpu())
                task_value = float(gradient_norm(task_params).cpu())
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                pre_clip = gradient_norm(model.parameters())
                if not bool(torch.isfinite(pre_clip)):
                    raise FloatingPointError(
                        f"Non-finite gradients without AMP at epoch={epoch}, step={step}:\n"
                        + "\n".join(nonfinite_gradient_report(model))
                    )
                pre_value = float(pre_clip.cpu())
                backbone_value = float(gradient_norm(backbone_params).cpu())
                task_value = float(gradient_norm(task_params).cpu())
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
            post_value = min(pre_value, grad_clip)
            pre_clip_norms.append(pre_value)
            post_clip_norms.append(post_value)
            backbone_grad_norms.append(backbone_value)
            task_grad_norms.append(task_value)
            clipped_steps += int(pre_value > grad_clip)
            running_loss += loss.detach()
            for name in component_names:
                running[name] += logs.get(name, mass.new_zeros(()))
        steps = len(train_loader)
        tree_metrics = (
            {k: v for k, v in finalize_tree_diagnostics(tree_raw).items() if k in tree_fields}
            if tree_levels else {}
        )
        tensor_pre_clip = torch.tensor(pre_clip_norms) if pre_clip_norms else None
        amp_skip_frac = amp_skipped_steps / max(1, steps)
        ovf_gt_mean = (sum(overflow_gt_means) / len(overflow_gt_means)) if overflow_gt_means else float("nan")
        ovf_gt_max = max(overflow_gt_maxs) if overflow_gt_maxs else float("nan")
        train_row = {
            "epoch": epoch,
            "loss": float((running_loss / steps).cpu()),
            **{name: float((running[name] / steps).cpu()) for name in component_names},
            **tree_metrics,
            **epoch_grads,
            "grad_total_pre_clip": sum(pre_clip_norms) / max(1, len(pre_clip_norms)) if pre_clip_norms else float("nan"),
            "grad_total_post_clip": sum(post_clip_norms) / max(1, len(post_clip_norms)) if post_clip_norms else float("nan"),
            "grad_pre_clip_p50": float(torch.quantile(tensor_pre_clip, 0.50)) if tensor_pre_clip is not None else float("nan"),
            "grad_pre_clip_p95": float(torch.quantile(tensor_pre_clip, 0.95)) if tensor_pre_clip is not None else float("nan"),
            "grad_pre_clip_max": max(pre_clip_norms, default=float("nan")),
            "grad_backbone": sum(backbone_grad_norms) / max(1, len(backbone_grad_norms)) if backbone_grad_norms else float("nan"),
            "grad_task": sum(task_grad_norms) / max(1, len(task_grad_norms)) if task_grad_norms else float("nan"),
            "clip_fraction": clipped_steps / max(1, len(pre_clip_norms)) if pre_clip_norms else float("nan"),
            "amp_scale_start": amp_scale_start,
            "amp_scale_end": float(scaler.get_scale()) if amp_enabled else 1.0,
            "amp_skipped_steps": amp_skipped_steps,
            "amp_skip_fraction": amp_skip_frac,
            "overflow_gt_mean": ovf_gt_mean,
            "overflow_gt_max": ovf_gt_max,
            "lr": lr_used,
            "lr_backbone": learning_rates["backbone"],
            "lr_task": learning_rates["task"],
        }
        _append_csv(train_csv, train_row, train_fields)

        # Construct informative loss decomposition string
        active_comps = [f"root={train_row.get('root_magnitude', 0.0):.2f}"]
        if "root_to_64" in train_row:
            active_comps.append(f"r->64={train_row['root_to_64']:.2f}")
        if "64_to_32" in train_row:
            active_comps.append(f"64->32={train_row['64_to_32']:.2f}")
        if "32_to_16" in train_row:
            active_comps.append(f"32->16={train_row['32_to_16']:.2f}")
        if "16_to_8" in train_row:
            active_comps.append(f"16->8={train_row['16_to_8']:.2f}")
        if "16_to_8_dense" in train_row:
            active_comps.append(f"16->8_dense={train_row['16_to_8_dense']:.2f}")
        if "flat_16" in train_row:
            active_comps.append(f"flat16={train_row['flat_16']:.2f}")
        if "deterministic_alloc" in train_row:
            active_comps.append(f"det_alloc={train_row['deterministic_alloc']:.2f}")
        if "exact_regression" in train_row:
            active_comps.append(f"exact_reg={train_row['exact_regression']:.2f}")
        loss_decomp_str = " ".join(active_comps)

        skipped_info = (
            f"skipped={amp_skipped_steps}/{steps} ({amp_skip_frac*100:.1f}%) [gt_mean={ovf_gt_mean:.1f} max={ovf_gt_max:.0f}]"
            if amp_skipped_steps > 0
            else f"skipped=0"
        )
        opt_str = (
            f"grad={train_row.get('grad_total_pre_clip', 0.0):.1f} "
            f"(p50={train_row.get('grad_pre_clip_p50', 0.0):.1f} "
            f"p95={train_row.get('grad_pre_clip_p95', 0.0):.1f} "
            f"max={train_row.get('grad_pre_clip_max', 0.0):.1f}) "
            f"clip@{grad_clip:g}={clipped_steps}/{steps} ({train_row.get('clip_fraction', 0.0)*100:.1f}%) "
            f"bb={train_row.get('grad_backbone', 0.0):.1f} "
            f"task={train_row.get('grad_task', 0.0):.1f} "
            f"scale={train_row.get('amp_scale_end', 1.0):g} "
            f"{skipped_info}"
        )

        tree_str = ""
        if "tree_32_16_active_parents_per_image" in train_row:
            tree_str = (
                f"  Tree 32->16: active={train_row['tree_32_16_active_parents_per_image']:.1f}/img "
                f"(zero={train_row.get('tree_32_16_zero_parent_fraction', 0.0)*100:.1f}%) "
                f"nll/active={train_row.get('tree_32_16_nll_per_active_parent', 0.0):.2f}"
            )

        if epoch % evaluate_every == 0 or epoch == epochs:
            metrics = evaluate_counting(model, evaluation_ds, device)
            val_row = {
                "epoch": epoch,
                **{name: metrics.get(name, float("nan")) for name in val_metric_fields},
                "lr": lr_used,
                "lr_backbone": learning_rates["backbone"],
                "lr_task": learning_rates["task"],
            }
            _append_csv(val_csv, val_row, val_fields)
            is_best = metrics["mae"] < best_mae
            if is_best:
                best_mae, best_epoch = metrics["mae"], epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_res": metrics,
                    "config": cfg,
                    "resolved_crop_statistics": crop_stats,
                    "selection_split": selection_split,
                    "is_exact_joint_nll": criterion.is_exact_joint_nll,
                    "initialization_policy": "timm_pretrained" if pretrained_spec else "scratch",
                    "pretrained_spec": pretrained_spec,
                    "runtime": get_runtime_metadata(),
                }, os.path.join(save_dir, "best.pt"))

            print(
                f"Epoch {epoch:04d}/{epochs} loss={train_row['loss']:.2f} [{loss_decomp_str}] "
                f"lr={train_row['lr']:.2e} time={time.time()-started:.1f}s\n"
                f"  Opt: {opt_str}\n"
                + (f"{tree_str}\n" if tree_str else "")
                + f"  >>> EVAL @ {epoch:04d}: MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f} "
                f"NAE={metrics.get('nae', float('nan')):.3f} Bias={metrics.get('bias', float('nan')):.2f} "
                f"{'(NEW BEST!)' if is_best else f'(best={best_mae:.2f}@{best_epoch})'}\n"
                f"      Pred stats: GT={metrics.get('gt_mean', 0.0):.1f}+/-{metrics.get('gt_std', 0.0):.1f} | "
                f"Pred={metrics.get('pred_mean', 0.0):.1f}+/-{metrics.get('pred_std', 0.0):.1f} | "
                f"SignedMed={metrics.get('signed_error_median', 0.0):.1f} | Ratio={metrics.get('pred_gt_ratio', float('nan')):.3f} | "
                f"Under={metrics.get('under_count_fraction', 0.0)*100:.1f}% Over={metrics.get('over_count_fraction', 0.0)*100:.1f}%\n"
                f"      Density: Sparse(<300, n={metrics.get('bin_sparse_count', 0)}): MAE={metrics.get('bin_sparse_mae', float('nan')):.1f} Bias={metrics.get('bin_sparse_bias', float('nan')):.1f} | "
                f"Mid(300-1k, n={metrics.get('bin_medium_count', 0)}): MAE={metrics.get('bin_medium_mae', float('nan')):.1f} Bias={metrics.get('bin_medium_bias', float('nan')):.1f} | "
                f"Dense(>=1k, n={metrics.get('bin_dense_count', 0)}): MAE={metrics.get('bin_dense_mae', float('nan')):.1f} Bias={metrics.get('bin_dense_bias', float('nan')):.1f}\n"
                f"      Extreme & Empty: Top10% Dense (n={metrics.get('top10_dense_count', 0)}): MAE={metrics.get('top10_dense_mae', float('nan')):.1f} | "
                f"Empty (n={metrics.get('empty_count', 0)}): MAE={metrics.get('empty_mae', 0.0):.2f} P95={metrics.get('empty_pred_p95', 0.0):.2f}",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:04d}/{epochs} loss={train_row['loss']:.2f} [{loss_decomp_str}] "
                f"lr={train_row['lr']:.2e} time={time.time()-started:.1f}s\n"
                f"  Opt: {opt_str}"
                + (f"\n{tree_str}" if tree_str else ""),
                flush=True,
            )
        scheduler.step()

    print(f"Training complete. Best MAE={best_mae:.3f} at epoch {best_epoch} on {selection_split}.", flush=True)


if __name__ == "__main__":
    main()
