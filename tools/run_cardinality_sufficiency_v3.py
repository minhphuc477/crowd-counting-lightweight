from __future__ import annotations

"""E0-v3: Operator-level compression attribution audit.

Protocol corrections over E0-v2:
1. Unified estimand: both effect size and CI use image-weighted averaging.
2. Operator-level hooks: stride-2 conv input/output, not stage endpoints.
3. Uniform per-image sampling: ImageCellCollector replaces TransitionReservoir.
4. Paired multi-seed MLP: identical seeds per representation.
5. Separated metrics: parent_mae_iw and composition_l1_iw are independent.

Primary controls (used in GO/NO-GO):
    op_pre          — representation at operator input (directly before stride-2 conv)
    op_post         — representation at operator output (directly after stride-2 conv)
    s2d_lossless    — lossless 2x2 packing of op_pre (exact SpaceToDepth)
    s2d_pca_matched — S2D + PCA to match op_post channel count (matched-dimensional
                      linear reference, NOT an information upper bound)

Informative decimation controls (NOT used in GO/NO-GO):
    avgpool         — plain 2x average-pooling of op_pre
    blurpool        — anti-aliased 2x decimation of op_pre

For C4→C8:  stride-2 operator is blocks.1.0.conv  (ConvBnAct)
For C8→C16: stride-2 operator is blocks.2.0.dw_mid.conv  (depthwise in UIR)
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence

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

from hpc.diagnostics.cardinality_sufficiency_v3 import (
    PCAProjector,
    ImageCellCollector,
    avgpool2x,
    blurpool2x,
    compute_image_weighted_metrics,
    fit_predict_ridge,
    image_weighted_bootstrap_diff,
    image_weighted_mean,
    pack_2x2_features,
    pack_child_counts,
    paired_seed_mlp_eval,
    per_cell_parent_mae,
    per_cell_composition_l1,
    post_to_cell_vectors,
    summarize_prediction,
)


# ---------------------------------------------------------------------------
# Transition definitions — keyed by (pre_stride, post_stride)
# op_module_path: the stride-2 operator to hook (for features_only=False path)
# These paths are specific to mobilenetv4_conv_small_050.
# ---------------------------------------------------------------------------
TRANSITIONS: tuple[tuple[int, int], ...] = ((4, 8), (8, 16))

# Stride-2 operator paths within the timm backbone (verified via probe)
# Format: (pre_hook_module_path, post_hook_module_path)
# pre_hook_module_path  = module whose INPUT  is X_op_pre
# post_hook_module_path = module whose OUTPUT is X_op_post
STRIDE2_OP_PATHS: dict[tuple[int, int], tuple[str, str]] = {
    (4, 8):  ("blocks.1.0.conv", "blocks.1.0.conv"),
    (8, 16): ("blocks.2.0.dw_mid.conv", "blocks.2.0.dw_mid.conv"),
}

# Stage endpoint paths (for secondary block-level comparison, reported separately)
STAGE_MODULE_PATHS: dict[tuple[int, int], tuple[str, str]] = {
    (4, 8):  ("blocks.0", "blocks.1.0"),
    (8, 16): ("blocks.1", "blocks.2.0"),
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
    raise ValueError(f"Unsupported dataset for E0-v3: {name}")


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


# ---------------------------------------------------------------------------
# Stride-2 operator hook machinery
# ---------------------------------------------------------------------------

class Stride2HookManager:
    """Register forward hooks to capture operator-level pre/post tensors.

    For a given module ``mod``, we:
    - use a forward_pre_hook to capture X_op_pre (input to the operator)
    - use a forward_hook    to capture X_op_post (output of the operator)

    Both hooks are cleaned up on __exit__. This class is a context manager.
    """

    def __init__(self):
        self._handles: list = []
        self._captures: dict[str, dict[str, torch.Tensor | None]] = {}

    def register(self, label: str, module: nn.Module) -> None:
        """Register pre+post hooks on `module` under `label`."""
        self._captures[label] = {"pre": None, "post": None}

        def pre_hook(mod, inp):
            # inp is a tuple of inputs; take the first tensor
            x = inp[0] if isinstance(inp, (tuple, list)) else inp
            if isinstance(x, torch.Tensor):
                self._captures[label]["pre"] = x.detach().float().cpu()

        def post_hook(mod, inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(x, torch.Tensor):
                self._captures[label]["post"] = x.detach().float().cpu()

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(post_hook))

    def get(self, label: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        entry = self._captures.get(label, {})
        return entry.get("pre"), entry.get("post")

    def clear(self) -> None:
        for label in self._captures:
            self._captures[label] = {"pre": None, "post": None}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()


def _get_nested_module(root: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated module path (e.g., 'blocks.2.0.dw_mid.conv')."""
    mod = root
    for part in path.split("."):
        if part.isdigit():
            children = list(mod.children())
            idx = int(part)
            if idx >= len(children):
                raise ValueError(
                    f"Index {idx} out of range for {type(mod).__name__} with {len(children)} children"
                )
            mod = children[idx]
        else:
            if not hasattr(mod, part):
                raise AttributeError(
                    f"Module {type(mod).__name__} has no attribute '{part}'. "
                    f"Available: {[n for n, _ in mod.named_children()]}"
                )
            mod = getattr(mod, part)
    return mod


# ---------------------------------------------------------------------------
# Feature source loader
# ---------------------------------------------------------------------------

class GenericTimmReductionExtractor(nn.Module):
    """Diagnostic-only pretrained feature extractor for cross-backbone controls."""

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


from dataclasses import dataclass


@dataclass
class LoadedSource:
    extractor: nn.Module
    full_model: nn.Module | None
    reductions: tuple[int, ...]
    normalization_mean: tuple[float, ...]
    normalization_std: tuple[float, ...]
    provenance: dict
    supports_op_hooks: bool  # True for MobileNetV4 production backbone


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
            supports_op_hooks=False,
        )

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
        }
    else:
        model = build_model_from_config(cfg, load_pretrained=True)
        provenance = {"kind": "current_timm_pretrained", "pretrained_spec": pretrained_spec}
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
        supports_op_hooks=True,
    )


# ---------------------------------------------------------------------------
# Feature collection with operator-level hooks
# ---------------------------------------------------------------------------

def _resolve_spatial_size(feat: torch.Tensor, reduction: int, input_h: int, input_w: int) -> tuple[int, int]:
    """Best-effort sanity check: feature map should be ~input/reduction."""
    _, _, fh, fw = feat.shape
    expected_h = math.ceil(input_h / reduction)
    expected_w = math.ceil(input_w / reduction)
    if abs(fh - expected_h) > 2 or abs(fw - expected_w) > 2:
        raise RuntimeError(
            f"Unexpected feature spatial size {(fh, fw)} for reduction={reduction} "
            f"on input {(input_h, input_w)}; expected ~{(expected_h, expected_w)}"
        )
    return fh, fw


def build_v3_representation_grid(
    op_pre: torch.Tensor,
    op_post: torch.Tensor,
    include_decimation_controls: bool = True,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Build primary and informative representations from op_pre/op_post.

    Args:
        op_pre:  [B, C_pre, H, W]  — directly before stride-2 op
        op_post: [B, C_post, H/2, W/2]  — directly after stride-2 op
        include_decimation_controls: if True, also return avgpool/blurpool

    Returns:
        (reps, meta)
        reps keys: "op_pre_s2d" (lossless), "op_post"
                   "avgpool", "blurpool" if include_decimation_controls
        meta: dimension info for each representation
    """
    if op_pre.ndim != 4 or op_post.ndim != 4:
        raise ValueError("op_pre and op_post must be [B,C,H,W]")
    b_pre, c_pre, h_pre, w_pre = op_pre.shape
    b_post, c_post, h_post, w_post = op_post.shape
    if b_pre != b_post:
        raise ValueError("Batch size mismatch between op_pre and op_post")
    if h_pre != h_post * 2 or w_pre != w_post * 2:
        raise ValueError(
            f"op_pre spatial {(h_pre, w_pre)} must be exactly 2x op_post {(h_post, w_post)}"
        )

    # Lossless S2D: 4 * C_pre channels, no information loss
    s2d = pack_2x2_features(op_pre)        # [B, H_post, W_post, 4*C_pre]
    # Native operator output
    native = post_to_cell_vectors(op_post)  # [B, H_post, W_post, C_post]

    reps = {
        "s2d_lossless": s2d,
        "op_post": native,
    }
    meta = {
        "c_pre": c_pre,
        "c_post": c_post,
        "s2d_dim": 4 * c_pre,
        "op_post_dim": c_post,
    }

    if include_decimation_controls:
        avg = post_to_cell_vectors(avgpool2x(op_pre))
        blur = post_to_cell_vectors(blurpool2x(op_pre))
        reps["avgpool"] = avg   # [B, H_post, W_post, C_pre]
        reps["blurpool"] = blur
        meta["avgpool_dim"] = c_pre
        meta["blurpool_dim"] = c_pre

    return reps, meta


def fit_matched_linear_reference(
    train_reps: dict[str, torch.Tensor],
    val_reps: dict[str, torch.Tensor],
    op_post_dim: int,
    s2d_key: str = "s2d_lossless",
    out_key: str = "s2d_pca_matched",
) -> tuple[dict, dict, dict]:
    """Fit S2D + PCA to match op_post channel count — matched-dimensional linear reference.

    This is NOT described as an information upper bound.
    ``effective_rank`` is logged; if it < op_post_dim, the comparison is flagged.
    """
    s2d_dim = int(train_reps[s2d_key].shape[1])
    projector = PCAProjector.fit(train_reps[s2d_key], output_dim=op_post_dim)
    train_out = projector.transform(train_reps[s2d_key])
    val_out = projector.transform(val_reps[s2d_key])

    meta = {
        "source_dim": s2d_dim,
        "output_dim": op_post_dim,
        "effective_rank": projector.effective_rank,
        "dimension_matched": projector.effective_rank == op_post_dim,
        "caution": (
            "effective_rank < output_dim: zero-padding applied. "
            "This control is an informative lower bound, not a budget-matched comparison."
        ) if projector.effective_rank < op_post_dim else None,
        "label": "matched-dimensional linear reference (S2D + PCA)",
    }
    return (
        {**train_reps, out_key: train_out},
        {**val_reps, out_key: val_out},
        meta,
    )


def fit_decimation_control_projections(
    train_reps: dict[str, torch.Tensor],
    val_reps: dict[str, torch.Tensor],
    op_post_dim: int,
) -> tuple[dict, dict, dict]:
    """Fit PCA projections for informative decimation controls (avgpool, blurpool).

    These are NOT used in GO/NO-GO decisions. effective_rank is always reported.
    """
    meta = {}
    for src, dst in (("avgpool", "avgpool_pca"), ("blurpool", "blurpool_pca")):
        if src not in train_reps:
            continue
        projector = PCAProjector.fit(train_reps[src], output_dim=op_post_dim)
        train_reps = {**train_reps, dst: projector.transform(train_reps[src])}
        val_reps = {**val_reps, dst: projector.transform(val_reps[src])}
        meta[dst] = {
            "source_dim": int(train_reps[src].shape[1]) if src in train_reps else None,
            "output_dim": op_post_dim,
            "effective_rank": projector.effective_rank,
            "note": "Informative decimation control only — excluded from GO/NO-GO.",
        }
    return train_reps, val_reps, meta


# ---------------------------------------------------------------------------
# Primary evaluation function (image-weighted, separated hypotheses)
# ---------------------------------------------------------------------------

def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if math.isfinite(num) and math.isfinite(den) and abs(den) > 1e-12 else float("nan")


def evaluate_transition_v3(
    train: dict,
    val: dict,
    ridge_alpha: float,
    bootstrap: int,
    seed: int,
    min_relative_degradation: float,
    comparison_mode: str,  # "op_level" or "block_level"
    run_mlp: bool = False,
    mlp_seeds: Sequence[int] = (42, 123, 456, 789, 2024),
    mlp_epochs: int = 40,
    probe_device: str = "cpu",
) -> dict:
    """Evaluate one stride comparison using unified image-weighted estimand.

    Primary representations for GO/NO-GO: op_pre_s2d (lossless S2D), op_post,
    s2d_pca_matched (matched-dimensional linear reference).
    Informative controls: avgpool_pca, blurpool_pca (reported in separate sub-dict,
    NOT used in any GO/NO-GO flag).

    Two independent hypotheses are evaluated:
        H_parent:      op_post carries strictly less parent cardinality information
                       than lossless S2D, beyond the matched linear reference.
        H_composition: op_post carries strictly less child composition information
                       than lossless S2D, beyond the matched linear reference.
    """
    train_reps_raw = dict(train["reps"])
    val_reps_raw = dict(val["reps"])
    op_post_dim_key = "op_post" if comparison_mode == "op_level" else "block_post"
    s2d_key = "s2d_lossless"

    op_post_dim = int(train_reps_raw[op_post_dim_key].shape[1])

    # Fit matched-dimensional linear reference from S2D
    train_reps, val_reps, pca_meta = fit_matched_linear_reference(
        train_reps_raw, val_reps_raw, op_post_dim=op_post_dim,
        s2d_key=s2d_key, out_key="s2d_pca_matched",
    )

    # Fit informative decimation controls (excluded from GO/NO-GO)
    train_reps, val_reps, dec_meta = fit_decimation_control_projections(
        train_reps, val_reps, op_post_dim=op_post_dim,
    )

    y_train = train["y"]
    y_val = val["y"]
    image_ids = val["image_ids"]
    n2plus_mask = y_val.sum(dim=1) >= 2

    # Primary probe: Ridge on primary controls
    primary_keys = [s2d_key, op_post_dim_key, "s2d_pca_matched"]
    predictions: dict[str, torch.Tensor] = {}
    for rep_name in primary_keys:
        predictions[rep_name] = fit_predict_ridge(
            train_reps[rep_name], y_train, val_reps[rep_name], alpha=float(ridge_alpha)
        )

    # Informative decimation controls (ridge only for metric reporting)
    informative_keys = [k for k in ("avgpool_pca", "blurpool_pca") if k in train_reps]
    informative_preds: dict[str, torch.Tensor] = {}
    for rep_name in informative_keys:
        informative_preds[rep_name] = fit_predict_ridge(
            train_reps[rep_name], y_train, val_reps[rep_name], alpha=float(ridge_alpha)
        )

    # --- PRIMARY METRICS: image-weighted, separated ---
    primary_metrics: dict[str, dict] = {}
    for rep_name, pred in predictions.items():
        m_all = compute_image_weighted_metrics(pred, y_val, image_ids)
        m_n2p = compute_image_weighted_metrics(pred, y_val, image_ids, mask=n2plus_mask)
        primary_metrics[rep_name] = {
            "all_active": m_all,
            "n2plus": m_n2p,
        }

    # Bin-stratified diagnostics (cell-averaged, diagnostic only, not used in GO/NO-GO)
    bin_diagnostics: dict[str, dict] = {
        rep_name: summarize_prediction(pred, y_val)
        for rep_name, pred in {**predictions, **informative_preds}.items()
    }

    # --- BOOTSTRAP CI: same estimand as image_weighted_mean ---
    # Evaluate on N>=2 cells only (primary estimand for cardinality hypothesis)
    parent_errors = {
        rep_name: per_cell_parent_mae(pred, y_val)
        for rep_name, pred in predictions.items()
    }
    comp_errors = {
        rep_name: per_cell_composition_l1(pred, y_val)
        for rep_name, pred in predictions.items()
    }

    def _bootstrap_pair(errors_a, errors_b, ids, label):
        ids_n2p = ids[n2plus_mask]
        return {
            "all_active": image_weighted_bootstrap_diff(
                errors_a, errors_b, ids,
                n_boot=bootstrap, seed=_stable_seed(seed, label, "all"),
            ),
            "n2plus": image_weighted_bootstrap_diff(
                errors_a[n2plus_mask], errors_b[n2plus_mask], ids_n2p,
                n_boot=bootstrap, seed=_stable_seed(seed, label, "n2plus"),
            ) if n2plus_mask.any() else None,
        }

    # Comparisons: native op_post vs S2D_lossless, and op_post vs S2D_pca_matched
    # Effect is POSITIVE when op_post is WORSE (higher error) than the reference
    op_key = op_post_dim_key
    parent_ci = {
        "op_post_vs_s2d_lossless": _bootstrap_pair(
            parent_errors[op_key], parent_errors[s2d_key], image_ids, "parent_op_vs_s2d"
        ),
        "op_post_vs_s2d_pca_matched": _bootstrap_pair(
            parent_errors[op_key], parent_errors["s2d_pca_matched"], image_ids, "parent_op_vs_pca"
        ),
    }
    comp_ci = {
        "op_post_vs_s2d_lossless": _bootstrap_pair(
            comp_errors[op_key], comp_errors[s2d_key], image_ids, "comp_op_vs_s2d"
        ),
        "op_post_vs_s2d_pca_matched": _bootstrap_pair(
            comp_errors[op_key], comp_errors["s2d_pca_matched"], image_ids, "comp_op_vs_pca"
        ),
    }

    # --- EFFECT SIZE: same estimand ---
    # Image-weighted mean of (op_post error - reference error) / reference error
    def _rel_deg(errors_target, errors_ref, ids, mask=None):
        if mask is not None:
            errors_target = errors_target[mask]
            errors_ref = errors_ref[mask]
            ids = ids[mask]
        diff_iw = image_weighted_mean(errors_target - errors_ref, ids)
        ref_iw = image_weighted_mean(errors_ref, ids)
        return _safe_ratio(diff_iw, ref_iw)

    n2p_ids = image_ids[n2plus_mask]
    relative_degradation = {
        "parent": {
            "op_vs_s2d_lossless_n2plus": _rel_deg(
                parent_errors[op_key], parent_errors[s2d_key], image_ids, n2plus_mask
            ),
            "op_vs_s2d_pca_matched_n2plus": _rel_deg(
                parent_errors[op_key], parent_errors["s2d_pca_matched"], image_ids, n2plus_mask
            ),
        },
        "composition": {
            "op_vs_s2d_lossless_n2plus": _rel_deg(
                comp_errors[op_key], comp_errors[s2d_key], image_ids, n2plus_mask
            ),
            "op_vs_s2d_pca_matched_n2plus": _rel_deg(
                comp_errors[op_key], comp_errors["s2d_pca_matched"], image_ids, n2plus_mask
            ),
        },
    }

    # Informative decimation metrics (not in GO/NO-GO)
    informative_metrics: dict[str, dict] = {}
    for rep_name, pred in informative_preds.items():
        m_all = compute_image_weighted_metrics(pred, y_val, image_ids)
        m_n2p = compute_image_weighted_metrics(pred, y_val, image_ids, mask=n2plus_mask)
        informative_metrics[rep_name] = {"all_active": m_all, "n2plus": m_n2p}

    # Optional MLP robustness (primary keys only, paired seeds)
    mlp_results = None
    if run_mlp:
        mlp_results = paired_seed_mlp_eval(
            train_reps, y_train, val_reps, y_val, image_ids,
            seeds=mlp_seeds, hidden=64, epochs=mlp_epochs, device=probe_device,
        )

    # --- DOWNSTREAM LINKAGE (image-weighted Spearman) ---
    linkage = None
    downstream_pred = val.get("downstream_parent_pred")
    if downstream_pred is not None and torch.isfinite(downstream_pred).all():
        gt_parent = y_val.sum(dim=1)
        ds_abs_error = (downstream_pred - gt_parent).abs()
        ds_underestimate = gt_parent - downstream_pred
        op_parent_excess = parent_errors[op_key] - parent_errors["s2d_pca_matched"]
        if n2plus_mask.sum() >= 3:
            rho_abs = spearmanr(
                op_parent_excess[n2plus_mask].numpy(),
                ds_abs_error[n2plus_mask].numpy(),
            ).statistic
            rho_under = spearmanr(
                op_parent_excess[n2plus_mask].numpy(),
                ds_underestimate[n2plus_mask].numpy(),
            ).statistic
            linkage = {
                "n2plus_cells": int(n2plus_mask.sum()),
                "spearman_parent_excess_vs_abs_count_error": float(rho_abs),
                "spearman_parent_excess_vs_underestimate": float(rho_under),
            }

    # --- DECISION: two independent hypotheses, unified estimand ---
    # H_parent: CI for parent MAE (op_post vs s2d_lossless) AND (op_post vs s2d_pca_matched)
    #           both have ci95_low > 0 AND relative degradation >= threshold
    def _ci95_low(ci_dict, subset="n2plus"):
        sub = ci_dict.get(subset)
        if sub is None:
            return float("nan")
        return float(sub.get("ci95_low", float("nan")))

    parent_s2d_ci95_low = _ci95_low(parent_ci["op_post_vs_s2d_lossless"])
    parent_pca_ci95_low = _ci95_low(parent_ci["op_post_vs_s2d_pca_matched"])
    comp_s2d_ci95_low = _ci95_low(comp_ci["op_post_vs_s2d_lossless"])
    comp_pca_ci95_low = _ci95_low(comp_ci["op_post_vs_s2d_pca_matched"])

    parent_rel_deg = relative_degradation["parent"]["op_vs_s2d_lossless_n2plus"]
    comp_rel_deg = relative_degradation["composition"]["op_vs_s2d_lossless_n2plus"]

    parent_screen_go = bool(
        math.isfinite(parent_s2d_ci95_low) and parent_s2d_ci95_low > 0.0
        and math.isfinite(parent_pca_ci95_low) and parent_pca_ci95_low > 0.0
        and math.isfinite(parent_rel_deg) and parent_rel_deg >= min_relative_degradation
    )
    composition_screen_go = bool(
        math.isfinite(comp_s2d_ci95_low) and comp_s2d_ci95_low > 0.0
        and math.isfinite(comp_pca_ci95_low) and comp_pca_ci95_low > 0.0
        and math.isfinite(comp_rel_deg) and comp_rel_deg >= min_relative_degradation
    )
    # final_go = either hypothesis passes at op_level
    final_go = parent_screen_go or composition_screen_go

    return {
        "comparison_mode": comparison_mode,
        "projection_controls": {
            "s2d_pca_matched": pca_meta,
            "decimation_controls": dec_meta,
        },
        "primary_metrics_image_weighted": primary_metrics,
        "relative_degradation": relative_degradation,
        "parent_ci": parent_ci,
        "composition_ci": comp_ci,
        "informative_decimation_metrics": informative_metrics,
        "bin_diagnostics": bin_diagnostics,
        "downstream_linkage": linkage,
        "mlp_robustness": mlp_results,
        "decision": {
            "min_relative_degradation": float(min_relative_degradation),
            "parent_screen_go": parent_screen_go,
            "composition_screen_go": composition_screen_go,
            "final_go": final_go,
            "parent_s2d_ci95_low_n2plus": parent_s2d_ci95_low,
            "parent_pca_ci95_low_n2plus": parent_pca_ci95_low,
            "comp_s2d_ci95_low_n2plus": comp_s2d_ci95_low,
            "comp_pca_ci95_low_n2plus": comp_pca_ci95_low,
            "parent_rel_deg_n2plus": parent_rel_deg,
            "comp_rel_deg_n2plus": comp_rel_deg,
        },
    }


# ---------------------------------------------------------------------------
# Data collection (uniform image sampling + operator hooks)
# ---------------------------------------------------------------------------

def collect_split_v3(
    dataset,
    indices: Sequence[int],
    source: LoadedSource,
    device: torch.device,
    crops_per_image: int,
    max_cells_per_image: int,
    seed: int,
) -> dict[tuple[int, int], dict]:
    """Collect cell data with uniform per-image sampling and optional stride-2 hooks.

    When ``source.supports_op_hooks`` is True, operator-level hook pairs are
    registered. Otherwise, stage-level features are used and op_level results
    are marked as unavailable.
    """
    transitions_available = [
        t for t in TRANSITIONS
        if t[0] in source.reductions and t[1] in source.reductions
    ]
    if not transitions_available:
        raise ValueError(f"No supported transitions in reductions={source.reductions}")

    # One collector per transition × comparison_mode
    collectors: dict[tuple[int, int, str], ImageCellCollector] = {}
    for transition in transitions_available:
        for mode in ("op_level", "block_level"):
            key = (*transition, mode)
            collectors[key] = ImageCellCollector(
                max_cells_per_image=max_cells_per_image,
                seed=_stable_seed(seed, "collector", *key),
            )

    # Set up stride-2 hooks if backbone supports it
    hook_manager = Stride2HookManager()
    if source.supports_op_hooks:
        backbone_nn = source.extractor.backbone  # timm model inside MobileNetV4Backbone
        for transition in transitions_available:
            op_path = STRIDE2_OP_PATHS.get(transition)
            if op_path is not None:
                op_mod = _get_nested_module(backbone_nn, op_path[0])  # same module for pre+post hook
                hook_manager.register(f"op_{transition[0]}_{transition[1]}", op_mod)
            # Block-level: hook on whole stride-2 block (first block of next stage)
            blk_path = STAGE_MODULE_PATHS.get(transition)
            if blk_path is not None:
                blk_mod = _get_nested_module(backbone_nn, blk_path[1])
                hook_manager.register(f"block_{transition[0]}_{transition[1]}", blk_mod)
        # Also hook the pre-block boundary (end of previous stage = input to stride block)
        # We reuse the pre_hook on each stride block, which fires before the block runs.

    shuffled = list(indices)
    random.Random(_stable_seed(seed, "image_order")).shuffle(shuffled)

    with torch.no_grad():
        for position, image_idx in enumerate(shuffled):
            for crop_id in range(int(crops_per_image)):
                crop_seed = _stable_seed(seed, "crop", image_idx, crop_id)
                with temporary_rng(crop_seed):
                    sample = dataset[image_idx]
                image = sample["image"].unsqueeze(0).to(device, non_blocking=True)

                # Clear hook captures from previous image
                if source.supports_op_hooks:
                    hook_manager.clear()

                features = source.extractor(image)
                feature_by_stride = {
                    int(stride): feat.float()
                    for stride, feat in zip(source.reductions, features)
                }

                mass = None
                if source.full_model is not None:
                    mass = source.full_model(image).float()

                for transition in transitions_available:
                    s, t = transition
                    y_s = sample["gt_blocks"][s].unsqueeze(0).to(device)
                    y_children = pack_child_counts(y_s).cpu()

                    # downstream task prediction
                    downstream_parent_pred = None
                    if mass is not None:
                        factor = t // 4
                        downstream_parent_pred = block_sum(mass[:, 0], factor).cpu()

                    # ---------- Operator-level representations ----------
                    op_label = f"op_{s}_{t}"
                    if source.supports_op_hooks and op_label in hook_manager._captures:
                        op_pre_tensor, op_post_tensor = hook_manager.get(op_label)
                        if op_pre_tensor is not None and op_post_tensor is not None:
                            # op_pre: immediately before stride-2 conv → need to upsample to
                            # same spatial as op_post? No: op_pre has 2x spatial resolution
                            # of op_post. We S2D-pack op_pre to match op_post spatial size.
                            # Ensure spatial alignment
                            _, cp, hp, wp = op_pre_tensor.shape
                            _, cq, hq, wq = op_post_tensor.shape
                            if hp == hq * 2 and wp == wq * 2:
                                reps_op, _ = build_v3_representation_grid(
                                    op_pre_tensor.to(device), op_post_tensor.to(device),
                                    include_decimation_controls=True,
                                )
                                # Rename op_post to match expected key
                                reps_op["op_post"] = reps_op.pop("op_post")
                                reps_op["s2d_lossless"] = reps_op.pop("s2d_lossless", reps_op.get("s2d_lossless"))

                                if reps_op["op_post"].shape[:3] == y_children.shape[:3]:
                                    collectors[(*transition, "op_level")].add(
                                        {k: v.cpu() for k, v in reps_op.items()},
                                        y_children,
                                        image_id=image_idx,
                                        downstream_parent_pred=downstream_parent_pred,
                                    )

                    # ---------- Block-level representations ----------
                    blk_label = f"block_{s}_{t}"
                    if source.supports_op_hooks and blk_label in hook_manager._captures:
                        blk_pre_tensor, blk_post_tensor = hook_manager.get(blk_label)
                        if blk_pre_tensor is not None and blk_post_tensor is not None:
                            _, cp, hp, wp = blk_pre_tensor.shape
                            _, cq, hq, wq = blk_post_tensor.shape
                            if hp == hq * 2 and wp == wq * 2:
                                reps_blk, _ = build_v3_representation_grid(
                                    blk_pre_tensor.to(device), blk_post_tensor.to(device),
                                    include_decimation_controls=True,
                                )
                                reps_blk["block_post"] = reps_blk.pop("op_post")
                                if reps_blk["block_post"].shape[:3] == y_children.shape[:3]:
                                    collectors[(*transition, "block_level")].add(
                                        {k: v.cpu() for k, v in reps_blk.items()},
                                        y_children,
                                        image_id=image_idx,
                                        downstream_parent_pred=downstream_parent_pred,
                                    )

                    # ---------- Fallback: stage-level (for backbones without hooks) ----------
                    if not source.supports_op_hooks:
                        pre = feature_by_stride[s]
                        post = feature_by_stride[t]
                        reps_stage, _ = build_v3_representation_grid(pre, post, include_decimation_controls=True)
                        reps_stage["op_post"] = reps_stage.pop("op_post")
                        if reps_stage["op_post"].shape[:3] == y_children.shape[:3]:
                            for mode in ("op_level", "block_level"):
                                collectors[(*transition, mode)].add(
                                    {k: v.cpu() for k, v in reps_stage.items()},
                                    y_children,
                                    image_id=image_idx,
                                    downstream_parent_pred=downstream_parent_pred,
                                )

            if (position + 1) % 20 == 0:
                print(f"Collected {position + 1}/{len(shuffled)} images", flush=True)

    hook_manager.remove()

    results = {}
    for (s, t, mode), collector in collectors.items():
        try:
            results[(s, t, mode)] = collector.finalize()
        except RuntimeError:
            results[(s, t, mode)] = None  # no cells collected for this mode

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="E0-v3: Operator-level compression attribution audit (methodologically corrected)"
    )
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument(
        "--backbone-override",
        default=None,
        help="Diagnostic-only timm pretrained backbone (disables operator-level hooks).",
    )
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--crops-per-image", type=int, default=3)
    p.add_argument("--max-cells-per-image", type=int, default=500,
                   help="Max active cells sampled per image (uniform across all images)")
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--min-relative-degradation", type=float, default=0.10)
    p.add_argument("--run-mlp", action="store_true",
                   help="Run paired multi-seed MLP robustness probe")
    p.add_argument("--mlp-seeds", type=int, nargs="+", default=[42, 123, 456, 789, 2024])
    p.add_argument("--mlp-epochs", type=int, default=40)
    p.add_argument("--probe-device", default="cpu")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-images", type=int, default=None,
                   help="Debug cap only — do not use for final claims")
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
        limit = max(1, int(args.max_images))
        train_indices = train_indices[:limit]
        val_indices = val_indices[: max(1, min(limit, len(val_indices)))]

    print(
        f"E0-v3 source={source.provenance['kind']} device={device} "
        f"images train/val={len(train_indices)}/{len(val_indices)} "
        f"reductions={source.reductions} "
        f"op_hooks={'YES' if source.supports_op_hooks else 'NO (stage-level fallback)'}",
        flush=True,
    )

    train_collected = collect_split_v3(
        dataset, train_indices, source, device,
        crops_per_image=args.crops_per_image,
        max_cells_per_image=args.max_cells_per_image,
        seed=_stable_seed(args.seed, "train"),
    )
    val_collected = collect_split_v3(
        dataset, val_indices, source, device,
        crops_per_image=args.crops_per_image,
        max_cells_per_image=args.max_cells_per_image,
        seed=_stable_seed(args.seed, "val"),
    )

    results = {}
    for transition in TRANSITIONS:
        s, t = transition
        if (s, t) not in [(r[0], r[1]) for r in [(s, t)]]:
            continue
        transition_result = {}
        for mode in ("op_level", "block_level"):
            key = (s, t, mode)
            tr = train_collected.get(key)
            vl = val_collected.get(key)
            if tr is None or vl is None:
                transition_result[mode] = {"status": "no_data"}
                continue
            transition_result[mode] = evaluate_transition_v3(
                tr, vl,
                ridge_alpha=args.ridge_alpha,
                bootstrap=args.bootstrap,
                seed=_stable_seed(args.seed, f"C{s}_C{t}", mode),
                min_relative_degradation=args.min_relative_degradation,
                comparison_mode=mode,
                run_mlp=args.run_mlp,
                mlp_seeds=tuple(args.mlp_seeds),
                mlp_epochs=args.mlp_epochs,
                probe_device=args.probe_device,
            )
        results[f"C{s}_to_C{t}"] = transition_result

    payload = {
        "protocol": {
            "name": "E0-v3 Operator-Level Compression Attribution Audit",
            "version": "v3",
            "repo_git_sha": _git_sha(),
            "config": str(args.config),
            "checkpoint": args.checkpoint,
            "backbone_override": args.backbone_override,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "crops_per_image": args.crops_per_image,
            "max_cells_per_image": args.max_cells_per_image,
            "ridge_alpha": args.ridge_alpha,
            "bootstrap": args.bootstrap,
            "run_mlp": args.run_mlp,
            "mlp_seeds": list(args.mlp_seeds),
            "mlp_epochs": args.mlp_epochs,
            "probe_device": args.probe_device,
            "train_image_count": len(train_indices),
            "val_image_count": len(val_indices),
            "train_indices": train_indices,
            "val_indices": val_indices,
            "source": source.provenance,
            "normalization_mean": list(source.normalization_mean),
            "normalization_std": list(source.normalization_std),
            "corrections_over_v2": [
                "unified_image_weighted_estimand",
                "operator_level_stride2_hooks",
                "uniform_per_image_cell_sampling",
                "paired_multi_seed_mlp",
                "separated_parent_mae_and_composition_l1",
                "decimation_controls_excluded_from_go_nogo",
                "s2d_pca_described_as_matched_dimensional_linear_reference",
            ],
        },
        "results": results,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print(f"Wrote {output}", flush=True)

    # Summary printout
    for transition_key, transition_result in results.items():
        for mode, eval_result in transition_result.items():
            if isinstance(eval_result, dict) and "decision" in eval_result:
                d = eval_result["decision"]
                print(
                    f"{transition_key} [{mode}]: "
                    f"parent_go={d['parent_screen_go']} "
                    f"comp_go={d['composition_screen_go']} "
                    f"final_go={d['final_go']} "
                    f"parent_rel_deg={d['parent_rel_deg_n2plus']:.3f} "
                    f"comp_rel_deg={d['comp_rel_deg_n2plus']:.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
