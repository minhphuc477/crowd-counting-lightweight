from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import spearmanr

from hpc.data.nwpu import NWPUDataset
from hpc.data.point_counts import block_sum
from hpc.data.qnrf import UCFQNRFDataset
from hpc.data.sha import ShanghaiTechDataset
from hpc.models.factory import (
    assert_checkpoint_compatible,
    build_model_from_config,
    validate_pretrained_normalization,
)

from hpc.diagnostics.cardinality_sufficiency_v2 import (
    BIN_NAMES,
    PCAProjector,
    bootstrap_image_mean_difference,
    build_representation_grid,
    fit_predict_mlp,
    fit_predict_ridge,
    pack_child_counts,
    parent_count_bin,
    per_cell_child_mae,
    summarize_prediction,
)


TRANSITIONS = ((4, 8), (8, 16))
REP_BASE = ("pre_pack", "native_post", "avgpool", "blurpool")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def _stable_seed(base: int, *parts: object) -> int:
    payload = ":".join([str(base), *map(str, parts)]).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)


@contextlib.contextmanager
def temporary_rng(seed: int):
    """Make dataset random crop/scale/flip deterministic without leaking RNG changes."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)


def split_image_indices(
    image_paths: Sequence[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if not (0.05 <= val_fraction <= 0.5):
        raise ValueError("val_fraction must be in [0.05, 0.5]")
    indices = list(range(len(image_paths)))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * float(val_fraction))))
    n_val = min(n_val, len(indices) - 1)
    return indices[n_val:], indices[:n_val]


def resolve_dataset_normalization(cfg: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    ds = cfg.get("dataset", {})
    mean = tuple(float(v) for v in ds.get("image_mean", [0.485, 0.456, 0.406]))
    std = tuple(float(v) for v in ds.get("image_std", [0.229, 0.224, 0.225]))
    return mean, std


def build_train_dataset(
    cfg: dict,
    *,
    image_mean: Sequence[float] | None = None,
    image_std: Sequence[float] | None = None,
):
    ds = cfg["dataset"]
    aug = cfg.get("augmentation", {})
    name = str(ds.get("name", "sha")).lower().replace("-", "_")
    mean = image_mean if image_mean is not None else ds.get("image_mean", [0.485, 0.456, 0.406])
    std = image_std if image_std is not None else ds.get("image_std", [0.229, 0.224, 0.225])
    common = dict(
        crop_size=int(ds.get("crop_size", 256)),
        is_train=True,
        scale_range=tuple(float(v) for v in aug.get("scale_range", [0.7, 1.3])),
        flip_prob=float(aug.get("flip_prob", 0.5)),
        image_mean=mean,
        image_std=std,
    )
    if "coordinate_base" in ds:
        common["coordinate_base"] = int(ds["coordinate_base"])

    if name in {"sha", "shanghaitech", "shanghaitech_a", "shanghaitech_b"}:
        part = ds.get("part", "part_B" if name.endswith("_b") else "part_A")
        return ShanghaiTechDataset(root=ds["root"], part=part, split="train_data", **common)
    if name in {"qnrf", "ucf_qnrf"}:
        return UCFQNRFDataset(root=ds["root"], split="Train", **common)
    if name == "nwpu":
        return NWPUDataset(
            root=ds["root"],
            split="train",
            split_file=ds.get("train_split_file"),
            **common,
        )
    raise ValueError(f"Unsupported dataset for E0-v2: {name}")


class GenericTimmReductionExtractor(nn.Module):
    """Diagnostic-only pretrained feature extractor for cross-backbone controls.

    Unlike the production MobileNetV4Backbone, this wrapper does not physically
    truncate a model and therefore works with timm backbones whose stage names
    are not `blocks.*`. It must not replace the production model silently.
    """

    def __init__(self, model_name: str, reductions: Sequence[int] = (4, 8, 16)):
        super().__init__()
        import timm

        with torch.random.fork_rng(devices=[]):
            probe = timm.create_model(model_name, pretrained=False, features_only=True)
            all_reductions = list(probe.feature_info.reduction())
            del probe
        selected = []
        for reduction in reductions:
            matches = [i for i, r in enumerate(all_reductions) if int(r) == int(reduction)]
            if not matches:
                raise ValueError(
                    f"Reduction {reduction} unavailable for {model_name}; got {all_reductions}"
                )
            selected.append(matches[-1])
        self.reductions = tuple(int(v) for v in reductions)
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            features_only=True,
            out_indices=tuple(selected),
        )

    def forward(self, x: torch.Tensor):
        return tuple(self.model(x))


def resolve_timm_normalization(model_name: str) -> tuple[tuple[float, ...], tuple[float, ...], dict]:
    import timm

    pretrained_cfg = timm.get_pretrained_cfg(model_name)
    if pretrained_cfg is None:
        raise ValueError(f"No timm pretrained config for {model_name}")
    source = pretrained_cfg.hf_hub_id or pretrained_cfg.url or pretrained_cfg.file
    if not source:
        raise ValueError(f"No pretrained weights source for {model_name}")
    meta = {
        "architecture": pretrained_cfg.architecture,
        "tag": pretrained_cfg.tag,
        "source": str(source),
        "mean": list(map(float, pretrained_cfg.mean)),
        "std": list(map(float, pretrained_cfg.std)),
    }
    return tuple(meta["mean"]), tuple(meta["std"]), meta


@dataclass
class LoadedSource:
    extractor: nn.Module
    full_model: nn.Module | None
    reductions: tuple[int, ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    provenance: dict


def load_feature_source(
    cfg: dict,
    device: torch.device,
    checkpoint: str | None,
    backbone_override: str | None,
) -> LoadedSource:
    if checkpoint and backbone_override:
        raise ValueError("Use either --checkpoint or --backbone-override, not both")

    if backbone_override:
        mean, std, meta = resolve_timm_normalization(backbone_override)
        extractor = GenericTimmReductionExtractor(backbone_override, reductions=(4, 8, 16)).to(device).eval()
        for p in extractor.parameters():
            p.requires_grad_(False)
        return LoadedSource(
            extractor=extractor,
            full_model=None,
            reductions=(4, 8, 16),
            normalization_mean=mean,
            normalization_std=std,
            provenance={"kind": "generic_timm_pretrained", "model": backbone_override, **meta},
        )

    # Current production carrier: config normalization must match its pretrained weights.
    pretrained_spec = validate_pretrained_normalization(cfg)
    mean, std = resolve_dataset_normalization(cfg)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert_checkpoint_compatible(ckpt, cfg)
        model = build_model_from_config(cfg, load_pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        provenance = {
            "kind": "task_checkpoint",
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": int(ckpt.get("epoch", -1)),
            "checkpoint_best_mae": float(ckpt.get("best_mae", float("nan"))),
            "checkpoint_git_sha": ckpt.get("runtime", {}).get("git_sha"),
        }
    else:
        model = build_model_from_config(cfg, load_pretrained=True)
        provenance = {
            "kind": "current_timm_pretrained",
            "pretrained_spec": pretrained_spec,
        }
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    reductions = tuple(int(v) for v in model.backbone.target_reductions)
    return LoadedSource(
        extractor=model.backbone,
        full_model=model if checkpoint else None,
        reductions=reductions,
        normalization_mean=mean,
        normalization_std=std,
        provenance=provenance,
    )


class TransitionReservoir:
    """Deterministic stratified active-cell reservoir with aligned representations."""

    def __init__(self, max_per_bin: int, seed: int):
        if max_per_bin <= 0:
            raise ValueError("max_per_bin must be positive")
        self.max_per_bin = int(max_per_bin)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.counts = {name: 0 for name in BIN_NAMES}
        self.rep_chunks: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {rep: [] for rep in REP_BASE} for name in BIN_NAMES
        }
        self.y_chunks = {name: [] for name in BIN_NAMES}
        self.id_chunks = {name: [] for name in BIN_NAMES}
        self.downstream_chunks = {name: [] for name in BIN_NAMES}

    def add(
        self,
        reps: dict[str, torch.Tensor],
        y_children: torch.Tensor,
        image_id: int,
        downstream_parent_pred: torch.Tensor | None,
    ) -> None:
        if y_children.ndim != 4 or y_children.shape[0] != 1 or y_children.shape[-1] != 4:
            raise ValueError("Expected y_children [1,H,W,4]")
        y = y_children.reshape(-1, 4).float().cpu()
        n = y.sum(dim=1)
        labels = parent_count_bin(n)
        flat_reps = {
            name: tensor.reshape(-1, tensor.shape[-1]).float().cpu()
            for name, tensor in reps.items()
        }
        for name, tensor in flat_reps.items():
            if tensor.shape[0] != y.shape[0]:
                raise ValueError(f"Representation {name} is not target-aligned")

        if downstream_parent_pred is not None:
            downstream = downstream_parent_pred.reshape(-1).float().cpu()
            if downstream.numel() != y.shape[0]:
                raise ValueError("downstream_parent_pred is not target-aligned")
        else:
            downstream = torch.full((y.shape[0],), float("nan"))

        for bin_idx, bin_name in enumerate(BIN_NAMES):
            remaining = self.max_per_bin - self.counts[bin_name]
            if remaining <= 0:
                continue
            idx = torch.nonzero(labels == bin_idx, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            order = torch.randperm(idx.numel(), generator=self.generator)
            idx = idx[order[:remaining]]
            if idx.numel() == 0:
                continue
            for rep_name in REP_BASE:
                self.rep_chunks[bin_name][rep_name].append(flat_reps[rep_name][idx])
            self.y_chunks[bin_name].append(y[idx])
            self.id_chunks[bin_name].append(
                torch.full((idx.numel(),), int(image_id), dtype=torch.long)
            )
            self.downstream_chunks[bin_name].append(downstream[idx])
            self.counts[bin_name] += int(idx.numel())

    def finalize(self) -> dict:
        reps: dict[str, list[torch.Tensor]] = {rep: [] for rep in REP_BASE}
        y_all, ids_all, downstream_all = [], [], []
        bin_slices: dict[str, list[int]] = {}
        cursor = 0
        for bin_name in BIN_NAMES:
            if not self.y_chunks[bin_name]:
                bin_slices[bin_name] = [cursor, cursor]
                continue
            y = torch.cat(self.y_chunks[bin_name], dim=0)
            ids = torch.cat(self.id_chunks[bin_name], dim=0)
            downstream = torch.cat(self.downstream_chunks[bin_name], dim=0)
            for rep in REP_BASE:
                reps[rep].append(torch.cat(self.rep_chunks[bin_name][rep], dim=0))
            y_all.append(y)
            ids_all.append(ids)
            downstream_all.append(downstream)
            start = cursor
            cursor += y.shape[0]
            bin_slices[bin_name] = [start, cursor]
        if not y_all:
            raise RuntimeError("No active cells were collected")
        return {
            "reps": {rep: torch.cat(chunks, dim=0) for rep, chunks in reps.items()},
            "y": torch.cat(y_all, dim=0),
            "image_ids": torch.cat(ids_all, dim=0),
            "downstream_parent_pred": torch.cat(downstream_all, dim=0),
            "bin_slices": bin_slices,
            "counts": dict(self.counts),
        }


def collect_split(
    dataset,
    indices: Sequence[int],
    source: LoadedSource,
    device: torch.device,
    crops_per_image: int,
    max_cells_per_bin: int,
    seed: int,
) -> dict[tuple[int, int], dict]:
    reservoirs = {
        transition: TransitionReservoir(
            max_per_bin=max_cells_per_bin,
            seed=_stable_seed(seed, "reservoir", *transition),
        )
        for transition in TRANSITIONS
        if transition[0] in source.reductions and transition[1] in source.reductions
    }
    if not reservoirs:
        raise ValueError(f"No supported transitions in reductions={source.reductions}")

    shuffled = list(indices)
    random.Random(_stable_seed(seed, "image_order")).shuffle(shuffled)

    with torch.no_grad():
        for position, image_idx in enumerate(shuffled):
            for crop_id in range(int(crops_per_image)):
                crop_seed = _stable_seed(seed, "crop", image_idx, crop_id)
                with temporary_rng(crop_seed):
                    sample = dataset[image_idx]
                image = sample["image"].unsqueeze(0).to(device, non_blocking=True)
                features = source.extractor(image)
                feature_by_stride = {
                    int(stride): feat.float()
                    for stride, feat in zip(source.reductions, features)
                }

                mass = None
                if source.full_model is not None:
                    mass = source.full_model(image).float()
                    if mass.shape[-2:] != sample["gt_blocks"][4].shape[-2:]:
                        raise RuntimeError(
                            "Task mass map and exact Y4 are not aligned: "
                            f"mass={tuple(mass.shape)}, y4={tuple(sample['gt_blocks'][4].shape)}"
                        )

                for (s, t), reservoir in reservoirs.items():
                    pre = feature_by_stride[s]
                    post = feature_by_stride[t]
                    reps = build_representation_grid(pre, post)
                    y_s = sample["gt_blocks"][s].unsqueeze(0).to(device)
                    y_children = pack_child_counts(y_s).cpu()
                    if reps["native_post"].shape[:3] != y_children.shape[:3]:
                        raise RuntimeError(
                            f"Feature/target geometry mismatch at {s}->{t}: "
                            f"post={tuple(reps['native_post'].shape)}, target={tuple(y_children.shape)}"
                        )

                    downstream_parent_pred = None
                    if mass is not None:
                        factor = t // 4
                        if t % 4 != 0:
                            raise ValueError(f"Transition parent stride {t} not divisible by output stride 4")
                        downstream_parent_pred = block_sum(mass[:, 0], factor).cpu()
                        y_parent = sample["gt_blocks"][t].unsqueeze(0)
                        if downstream_parent_pred.shape != y_parent.shape:
                            raise RuntimeError(
                                f"Downstream local count mismatch at stride {t}: "
                                f"pred={tuple(downstream_parent_pred.shape)}, gt={tuple(y_parent.shape)}"
                            )

                    reservoir.add(
                        {name: tensor.cpu() for name, tensor in reps.items()},
                        y_children,
                        image_id=image_idx,
                        downstream_parent_pred=downstream_parent_pred,
                    )

            if (position + 1) % 20 == 0:
                counts_text = ", ".join(
                    f"{s}->{t}:{reservoir.counts}"
                    for (s, t), reservoir in reservoirs.items()
                )
                print(f"Collected {position + 1}/{len(shuffled)} images | {counts_text}", flush=True)

    return {transition: reservoir.finalize() for transition, reservoir in reservoirs.items()}


def fit_budget_representations(train: dict, val: dict) -> tuple[dict, dict, dict]:
    """Fit train-only PCA controls to the native post-transition channel budget."""
    train_reps = dict(train["reps"])
    val_reps = dict(val["reps"])
    post_dim = int(train_reps["native_post"].shape[1])
    meta = {"native_post_dim": post_dim, "input_dims": {k: int(v.shape[1]) for k, v in train_reps.items()}}

    for source_name, target_name in (
        ("pre_pack", "s2d_pca_budget"),
        ("avgpool", "avgpool_pca_budget"),
        ("blurpool", "blurpool_pca_budget"),
    ):
        projector = PCAProjector.fit(train_reps[source_name], output_dim=post_dim)
        train_reps[target_name] = projector.transform(train_reps[source_name])
        val_reps[target_name] = projector.transform(val_reps[source_name])
        meta[target_name] = {
            "source_dim": int(train_reps[source_name].shape[1]),
            "output_dim": post_dim,
            "effective_rank_cap": min(int(train_reps[source_name].shape[1]), post_dim),
        }
    return train_reps, val_reps, meta


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-12 else float("nan")


def evaluate_transition(
    train: dict,
    val: dict,
    ridge_alpha: float,
    bootstrap: int,
    seed: int,
    min_relative_degradation: float,
    min_linkage_rho: float,
    run_mlp: bool = False,
    mlp_epochs: int = 40,
    probe_device: str = "cpu",
) -> dict:
    train_reps, val_reps, projection_meta = fit_budget_representations(train, val)
    y_train = train["y"]
    y_val = val["y"]
    image_ids = val["image_ids"]
    n_val = y_val.sum(dim=1)
    multi = n_val >= 2

    predictions = {}
    metrics = {}
    for rep_name in (
        "pre_pack",
        "native_post",
        "s2d_pca_budget",
        "avgpool_pca_budget",
        "blurpool_pca_budget",
    ):
        pred = fit_predict_ridge(
            train_reps[rep_name],
            y_train,
            val_reps[rep_name],
            alpha=float(ridge_alpha),
        )
        predictions[rep_name] = pred
        metrics[rep_name] = summarize_prediction(pred, y_val)

    nonlinear_robustness = None
    if run_mlp:
        nonlinear_robustness = {}
        for rep_name in ("pre_pack", "native_post", "s2d_pca_budget"):
            pred_mlp = fit_predict_mlp(
                train_reps[rep_name],
                y_train,
                val_reps[rep_name],
                hidden=64,
                epochs=int(mlp_epochs),
                batch_size=1024,
                seed=_stable_seed(seed, "mlp", rep_name),
                device=probe_device,
            )
            nonlinear_robustness[rep_name] = summarize_prediction(pred_mlp, y_val)

    losses = {name: per_cell_child_mae(pred, y_val) for name, pred in predictions.items()}
    comparisons = {}
    for name, baseline in (
        ("native_minus_pre", "pre_pack"),
        ("native_minus_s2d_pca", "s2d_pca_budget"),
        ("native_minus_avgpool_pca", "avgpool_pca_budget"),
        ("native_minus_blurpool_pca", "blurpool_pca_budget"),
    ):
        comparisons[name] = {
            "all_active": bootstrap_image_mean_difference(
                losses["native_post"], losses[baseline], image_ids,
                n_boot=bootstrap, seed=_stable_seed(seed, name, "all"),
            ),
            "n2plus": bootstrap_image_mean_difference(
                losses["native_post"][multi], losses[baseline][multi], image_ids[multi],
                n_boot=bootstrap, seed=_stable_seed(seed, name, "n2plus"),
            ) if multi.any() else None,
        }

    pre_n2 = metrics["pre_pack"]["n2p_child_mae"]
    native_n2 = metrics["native_post"]["n2p_child_mae"]
    pca_n2 = metrics["s2d_pca_budget"]["n2p_child_mae"]
    rel_native_vs_pre = _safe_ratio(native_n2 - pre_n2, pre_n2)
    rel_native_vs_pca = _safe_ratio(native_n2 - pca_n2, pca_n2)

    native_pre_gap = native_n2 - pre_n2
    closure = {}
    for control in ("s2d_pca_budget", "avgpool_pca_budget", "blurpool_pca_budget"):
        control_n2 = metrics[control]["n2p_child_mae"]
        closure[control] = _safe_ratio(native_n2 - control_n2, native_pre_gap)

    linkage = None
    downstream_pred = val["downstream_parent_pred"]
    if torch.isfinite(downstream_pred).all():
        gt_parent = y_val.sum(dim=1)
        downstream_signed = downstream_pred - gt_parent
        downstream_abs = downstream_signed.abs()
        representation_excess = losses["native_post"] - losses["s2d_pca_budget"]
        if multi.sum() >= 3:
            rho_abs = spearmanr(
                representation_excess[multi].numpy(), downstream_abs[multi].numpy()
            ).statistic
            # Positive means more representation loss is associated with stronger under-counting.
            underestimate = (gt_parent - downstream_pred)
            rho_under = spearmanr(
                representation_excess[multi].numpy(), underestimate[multi].numpy()
            ).statistic
            linkage = {
                "n2plus_cells": int(multi.sum()),
                "spearman_excess_vs_abs_local_count_error": float(rho_abs),
                "spearman_excess_vs_underestimate": float(rho_under),
            }

    cmp_pre = comparisons["native_minus_pre"]["n2plus"]
    cmp_pca = comparisons["native_minus_s2d_pca"]["n2plus"]
    statistical_pre = bool(cmp_pre is not None and cmp_pre["ci95_low"] > 0.0)
    statistical_pca = bool(cmp_pca is not None and cmp_pca["ci95_low"] > 0.0)
    effect = bool(math.isfinite(rel_native_vs_pre) and rel_native_vs_pre >= min_relative_degradation)
    screen_go = statistical_pre and statistical_pca and effect

    linkage_go = None
    if linkage is not None:
        rho_abs = linkage["spearman_excess_vs_abs_local_count_error"]
        rho_under = linkage["spearman_excess_vs_underestimate"]
        linkage_go = bool(
            (math.isfinite(rho_abs) and rho_abs >= min_linkage_rho)
            or (math.isfinite(rho_under) and rho_under >= min_linkage_rho)
        )

    return {
        "projection_controls": projection_meta,
        "metrics": metrics,
        "comparisons": comparisons,
        "relative_n2plus_degradation": {
            "native_vs_pre": rel_native_vs_pre,
            "native_vs_s2d_pca": rel_native_vs_pca,
        },
        "known_control_gap_closure_fraction_n2plus": closure,
        "downstream_linkage": linkage,
        "nonlinear_probe_robustness": nonlinear_robustness,
        "decision": {
            "min_relative_degradation": float(min_relative_degradation),
            "min_linkage_rho": float(min_linkage_rho),
            "native_worse_than_pre_ci95": statistical_pre,
            "native_worse_than_s2d_pca_ci95": statistical_pca,
            "relative_effect_pass": effect,
            "screen_go": screen_go,
            "linkage_go": linkage_go,
            "final_go": bool(screen_go and linkage_go) if linkage_go is not None else None,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0-v2: transition-wise cardinality sufficiency / compression attribution audit"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--backbone-override",
        default=None,
        help="Diagnostic-only timm pretrained backbone for a cross-architecture control.",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--crops-per-image", type=int, default=3)
    p.add_argument("--max-train-cells-per-bin", type=int, default=20000)
    p.add_argument("--max-val-cells-per-bin", type=int, default=10000)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-relative-degradation", type=float, default=0.10)
    p.add_argument("--min-linkage-rho", type=float, default=0.20)
    p.add_argument("--run-mlp", action="store_true", help="Secondary nonlinear-accessibility robustness probe")
    p.add_argument("--mlp-epochs", type=int, default=40)
    p.add_argument("--probe-device", default="cpu", help="Device for optional MLP probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-images", type=int, default=None, help="Smoke/debug only; do not use for final claims")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)

    device = torch.device(args.device)
    source = load_feature_source(
        cfg,
        device=device,
        checkpoint=args.checkpoint,
        backbone_override=args.backbone_override,
    )
    dataset = build_train_dataset(
        cfg,
        image_mean=source.normalization_mean,
        image_std=source.normalization_std,
    )
    train_indices, val_indices = split_image_indices(
        dataset.image_paths,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if args.max_images is not None:
        # Debug-only cap is applied independently so both partitions remain non-empty.
        limit = max(1, int(args.max_images))
        train_indices = train_indices[:limit]
        val_indices = val_indices[: max(1, min(limit, len(val_indices)))]

    print(
        f"E0-v2 source={source.provenance['kind']} device={device} "
        f"images train/val={len(train_indices)}/{len(val_indices)} "
        f"reductions={source.reductions}",
        flush=True,
    )

    train_collected = collect_split(
        dataset,
        train_indices,
        source,
        device,
        crops_per_image=args.crops_per_image,
        max_cells_per_bin=args.max_train_cells_per_bin,
        seed=_stable_seed(args.seed, "train"),
    )
    val_collected = collect_split(
        dataset,
        val_indices,
        source,
        device,
        crops_per_image=args.crops_per_image,
        max_cells_per_bin=args.max_val_cells_per_bin,
        seed=_stable_seed(args.seed, "val"),
    )

    results = {}
    for transition in sorted(train_collected):
        key = f"C{transition[0]}_to_C{transition[1]}"
        results[key] = {
            "train_cell_counts": train_collected[transition]["counts"],
            "val_cell_counts": val_collected[transition]["counts"],
            **evaluate_transition(
                train_collected[transition],
                val_collected[transition],
                ridge_alpha=args.ridge_alpha,
                bootstrap=args.bootstrap,
                seed=_stable_seed(args.seed, key),
                min_relative_degradation=args.min_relative_degradation,
                min_linkage_rho=args.min_linkage_rho,
                run_mlp=args.run_mlp,
                mlp_epochs=args.mlp_epochs,
                probe_device=args.probe_device,
            ),
        }

    payload = {
        "protocol": {
            "name": "E0-v2 Compression Attribution Audit",
            "repo_git_sha": _git_sha(),
            "config": str(args.config),
            "checkpoint": args.checkpoint,
            "backbone_override": args.backbone_override,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "crops_per_image": args.crops_per_image,
            "max_train_cells_per_bin": args.max_train_cells_per_bin,
            "max_val_cells_per_bin": args.max_val_cells_per_bin,
            "ridge_alpha": args.ridge_alpha,
            "bootstrap": args.bootstrap,
            "run_mlp": args.run_mlp,
            "mlp_epochs": args.mlp_epochs,
            "probe_device": args.probe_device,
            "train_image_count": len(train_indices),
            "val_image_count": len(val_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "source": source.provenance,
            "normalization_mean": list(source.normalization_mean),
            "normalization_std": list(source.normalization_std),
            "important_note": (
                "PCA is a budget-matched linear control, not a theoretical information upper bound. "
                "Blur/average pooling controls are diagnostics, not proposed contributions."
            ),
        },
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Wrote {output}", flush=True)
    for key, value in results.items():
        d = value["decision"]
        print(
            f"{key}: screen_go={d['screen_go']} linkage_go={d['linkage_go']} final_go={d['final_go']} "
            f"native/pre n2+={value['relative_n2plus_degradation']['native_vs_pre']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
