from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rmr_core.data import resolve_manifest_path
from rmr_core.training import compute_file_sha256

from .schema import (
    ALLOWED_LOSS_KEYS,
    ALLOWED_MODEL_KEYS,
    CRITICAL_TRAIN_DEFAULTS,
)
from .validator import validate_v3_config


def _canonicalize_value(val: Any) -> Any:
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        return round(val, 7)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, (list, tuple)):
        return [_canonicalize_value(x) for x in val]
    if isinstance(val, dict):
        return {k: _canonicalize_value(v) for k, v in sorted(val.items())}
    return str(val)


def extract_trajectory_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract only trajectory-critical configuration fields for reproducibility hashing.

    Excludes environment-specific paths and local execution settings:
    `output_dir`, `data_root`, `pin_memory`, etc.
    Includes:
    - seed
    - model (canonicalized hyperparameters, including derived init_m0)
    - loss (all loss weights and parameters)
    - train (learning rates, weight decay, epochs, warmup, solver warmup/ramp, amp, deterministic, grad_clip,
             workers, eval_every, early_stopping, patience)
    - data (crop_size, scale_range, hflip_prob, brightness_jitter, contrast_jitter,
            train_manifest_name, train_manifest_sha256, val_manifest_name, val_manifest_sha256)
    """
    traj: dict[str, Any] = {}
    if "seed" in cfg:
        traj["seed"] = int(cfg["seed"])

    # Model fields
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        m: dict[str, Any] = {}
        for k, v in model_cfg.items():
            if k not in ALLOWED_MODEL_KEYS:
                continue
            norm_k = "backbone" if k in ("backbone", "backbone_name") else (
                "omega" if k in ("omega", "sirt_omega") else k
            )
            m[norm_k] = _canonicalize_value(v)
        traj["model"] = {k: m[k] for k in sorted(m)}

    # Loss fields
    loss_cfg = cfg.get("loss", {})
    if isinstance(loss_cfg, dict):
        traj["loss"] = {
            k: _canonicalize_value(v)
            for k, v in sorted(loss_cfg.items())
            if k in ALLOWED_LOSS_KEYS
        }

    # Train hyperparameters (workers, eval_every, early_stopping, patience are trajectory/outcome critical)
    train_cfg = cfg.get("train", {})
    if isinstance(train_cfg, dict):
        critical_train_keys = {
            "lr",
            "backbone_lr_scale",
            "weight_decay",
            "epochs",
            "warmup_epochs",
            "batch_size",
            "deterministic",
            "grad_clip",
            "amp",
            "solver_warmup_epochs",
            "solver_ramp_epochs",
            "workers",
            "eval_every",
            "early_stopping",
            "patience",
        }
        t: dict[str, Any] = {}
        for k in critical_train_keys:
            if k in train_cfg:
                t[k] = _canonicalize_value(train_cfg[k])
            elif k in CRITICAL_TRAIN_DEFAULTS:
                t[k] = _canonicalize_value(CRITICAL_TRAIN_DEFAULTS[k])
        traj["train"] = {k: t[k] for k in sorted(t)}

    # Data augmentations & manifest identity (manifest name & content SHA256)
    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        d: dict[str, Any] = {}
        for k in ("crop_size", "scale_range", "hflip_prob", "brightness_jitter", "contrast_jitter",
                  "gamma_jitter", "random_invert_prob"):
            if k in data_cfg:
                d[k] = _canonicalize_value(data_cfg[k])
        data_root = data_cfg.get("data_root")
        for k in ("train_manifest", "val_manifest"):
            if k in data_cfg and data_cfg[k]:
                raw_path = str(data_cfg[k])
                d[f"{k}_name"] = Path(raw_path).name
                resolved = resolve_manifest_path(raw_path, data_root=data_root)
                if resolved is not None:
                    d[f"{k}_sha256"] = compute_file_sha256(resolved)
                else:
                    d[f"{k}_sha256"] = "<virtual>"
        traj["data"] = {k: d[k] for k in sorted(d)}

    return traj


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 digest of the trajectory-critical configuration."""
    traj = extract_trajectory_config(cfg)
    normalized_json = json.dumps(traj, sort_keys=True, default=str)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load, parse, and validate an RMR YAML configuration file."""
    import yaml

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration at {p} must parse to a dictionary, got {type(cfg).__name__}")
    validate_v3_config(cfg)
    return cfg
