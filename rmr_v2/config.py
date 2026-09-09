from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


V2_ALLOWED_TOP_LEVEL_KEYS: set[str] = {
    "seed",
    "output_dir",
    "data",
    "model",
    "loss",
    "train",
    "eval",
}

V2_ALLOWED_MODEL_KEYS: set[str] = {
    "variant",
    "output_stride",
    "feature_width",
    "region_sizes_px",
    "region_overlap",
    "include_full_image",
    "iterations",
    "eta_max",
    "eta_init",
    "residual_clip",
    "eps",
    "update_rule",
    "use_jacobian_gate",
    "sirt_omega",
    "learnable_sirt_omega",
    "projected_use_preconditioner",
    "detach_region_evidence",
    "init_m0",
    "backbone_name",
    "backbone",
    "pretrained",
    "backbone_lr_scale",
}

V2_ALLOWED_LOSS_KEYS: set[str] = {
    "lambda_count",
    "lambda_global",
    "lambda_flat_dm16",
    "lambda_cell",
    "lambda_region_head",
    "lambda_region_map",
    "lambda_deep_supervision",
    "count_loss_mode",
    "nb_dispersion",
    "kappa_flat16",
    "cell_beta",
    "region_beta",
    "normalize_flat_dm16",
}

V2_ALLOWED_TRAIN_KEYS: set[str] = {
    "batch_size",
    "workers",
    "pin_memory",
    "lr",
    "backbone_lr_scale",
    "weight_decay",
    "epochs",
    "warmup_epochs",
    "eval_every",
    "grad_clip",
    "amp",
    "solver_warmup_epochs",
    "solver_ramp_epochs",
    "early_stopping",
    "patience",
    "deterministic",
}

V2_ALLOWED_DATA_KEYS: set[str] = {
    "train_manifest",
    "val_manifest",
    "data_root",
    "crop_size",
    "scale_range",
    "hflip_prob",
    "brightness_jitter",
    "contrast_jitter",
}

V2_METHOD_CRITICAL_FIELDS: dict[str, list[str]] = {
    "model": [
        "variant",
        "output_stride",
        "feature_width",
        "region_sizes_px",
        "region_overlap",
        "include_full_image",
        "iterations",
        "update_rule",
        "sirt_omega",
        "learnable_sirt_omega",
        "projected_use_preconditioner",
        "detach_region_evidence",
        "residual_clip",
        "use_jacobian_gate",
        "eta_max",
        "eta_init",
        "backbone",
        "backbone_lr_scale",
    ],
    "loss": [
        "lambda_count",
        "lambda_flat_dm16",
        "lambda_cell",
        "lambda_region_head",
        "lambda_region_map",
        "lambda_deep_supervision",
        "count_loss_mode",
        "nb_dispersion",
        "kappa_flat16",
        "cell_beta",
        "region_beta",
        "normalize_flat_dm16",
    ],
    "train": [
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
    ],
    "data": [
        "crop_size",
        "scale_range",
    ],
}

V2_CRITICAL_TRAIN_DEFAULTS: dict[str, Any] = {
    "workers": 0,
    "eval_every": 10,
    "early_stopping": False,
    "patience": 0,
    "deterministic": False,
    "amp": True,
    "grad_clip": 500.0,
    "solver_warmup_epochs": 5,
    "solver_ramp_epochs": 20,
    "warmup_epochs": 5,
}


def compute_file_sha256(path: Path | str) -> str:
    """Compute deterministic SHA256 hex digest of a file in 64KB blocks."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for SHA256 computation: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def resolve_manifest_path(manifest: str | Path, data_root: str | Path | None = None) -> Path | None:
    """Resolve manifest path on filesystem."""
    p = Path(manifest)
    if p.is_file():
        return p
    if data_root is not None:
        p2 = Path(data_root) / p
        if p2.is_file():
            return p2
    return None


def _canonicalize_value(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return [_canonicalize_value(x) for x in v]
    if isinstance(v, dict):
        return {k: _canonicalize_value(val) for k, val in sorted(v.items())}
    if isinstance(v, float):
        return round(v, 8)
    return v


def _are_values_compatible(v1: Any, v2: Any) -> bool:
    if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
        if isinstance(v1, bool) or isinstance(v2, bool):
            return bool(v1) == bool(v2)
        return abs(float(v1) - float(v2)) < 1e-7
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)):
        if len(v1) != len(v2):
            return False
        return all(_are_values_compatible(x, y) for x, y in zip(v1, v2))
    return str(v1) == str(v2)


def extract_v2_trajectory_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract canonicalized V2 trajectory configuration covering all B0-B5 parameters."""
    traj: dict[str, Any] = {}
    if "seed" in cfg:
        traj["seed"] = int(cfg["seed"])

    # Model fields
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        m: dict[str, Any] = {}
        for k, v in model_cfg.items():
            if k not in V2_ALLOWED_MODEL_KEYS:
                continue
            norm_k = "backbone" if k in ("backbone", "backbone_name") else (
                "sirt_omega" if k in ("omega", "sirt_omega") else k
            )
            m[norm_k] = _canonicalize_value(v)
        traj["model"] = {k: m[k] for k in sorted(m)}

    # Loss fields
    loss_cfg = cfg.get("loss", {})
    if isinstance(loss_cfg, dict):
        l_dict: dict[str, Any] = {}
        for k, v in loss_cfg.items():
            if k not in V2_ALLOWED_LOSS_KEYS:
                continue
            norm_k = "lambda_count" if k == "lambda_global" else k
            l_dict[norm_k] = _canonicalize_value(v)
        traj["loss"] = {k: l_dict[k] for k in sorted(l_dict)}

    # Train fields
    train_cfg = cfg.get("train", {})
    if isinstance(train_cfg, dict):
        t: dict[str, Any] = {}
        for k in V2_ALLOWED_TRAIN_KEYS:
            if k in train_cfg:
                t[k] = _canonicalize_value(train_cfg[k])
            elif k in V2_CRITICAL_TRAIN_DEFAULTS:
                t[k] = _canonicalize_value(V2_CRITICAL_TRAIN_DEFAULTS[k])
        traj["train"] = {k: t[k] for k in sorted(t)}

    # Data fields
    data_cfg = cfg.get("data", {})
    if isinstance(data_cfg, dict):
        d: dict[str, Any] = {}
        for k in ("crop_size", "scale_range", "hflip_prob", "brightness_jitter", "contrast_jitter"):
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


def compute_v2_config_hash(cfg: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 digest of V2 trajectory-critical configuration."""
    traj = extract_v2_trajectory_config(cfg)
    normalized_json = json.dumps(traj, sort_keys=True, default=str)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


def validate_v2_config(cfg: dict[str, Any]) -> None:
    """Strictly validate V2/Stage C configuration dictionary."""
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must be a dictionary, got {type(cfg).__name__}")

    for top_k in cfg.keys():
        if top_k not in V2_ALLOWED_TOP_LEVEL_KEYS:
            raise ValueError(f"Unknown top-level config key '{top_k}' in V2 config")

    m_cfg = cfg.get("model", {})
    if isinstance(m_cfg, dict):
        for k in m_cfg.keys():
            if k not in V2_ALLOWED_MODEL_KEYS:
                raise ValueError(f"Unknown config key '{k}' in section 'model'")
        if "output_stride" in m_cfg and int(m_cfg["output_stride"]) != 4:
            raise ValueError("RMR-v2 only supports output_stride=4")

    l_cfg = cfg.get("loss", {})
    if isinstance(l_cfg, dict):
        for k in l_cfg.keys():
            if k not in V2_ALLOWED_LOSS_KEYS:
                raise ValueError(f"Unknown config key '{k}' in section 'loss'")

    t_cfg = cfg.get("train", {})
    if isinstance(t_cfg, dict):
        for k in t_cfg.keys():
            if k not in V2_ALLOWED_TRAIN_KEYS:
                raise ValueError(f"Unknown config key '{k}' in section 'train'")

    d_cfg = cfg.get("data", {})
    if isinstance(d_cfg, dict):
        for k in d_cfg.keys():
            if k not in V2_ALLOWED_DATA_KEYS:
                raise ValueError(f"Unknown config key '{k}' in section 'data'")

    e_cfg = cfg.get("eval", {})
    if isinstance(e_cfg, dict) and "density_bins" in e_cfg:
        bins = e_cfg["density_bins"]
        if not isinstance(bins, (list, tuple)) or len(bins) != 2:
            raise ValueError(f"eval.density_bins must be a list or tuple of exactly 2 thresholds [low, high], got {bins!r}")
        try:
            b0, b1 = float(bins[0]), float(bins[1])
        except (ValueError, TypeError) as err:
            raise ValueError(f"eval.density_bins thresholds must be numeric, got {bins!r}") from err
        if b0 <= 0 or b1 <= 0:
            raise ValueError(f"eval.density_bins thresholds must be positive numbers, got [{b0}, {b1}]")
        if b0 >= b1:
            raise ValueError(f"eval.density_bins thresholds must satisfy low < high, got [{b0}, {b1}]")


def validate_v2_resume_compatibility(
    ckpt_cfg: dict[str, Any],
    incoming_cfg: dict[str, Any],
    ckpt_hash: str | None = None,
    incoming_hash: str | None = None,
    ckpt_commit: str | None = None,
    current_commit: str | None = None,
    allow_cross_commit: bool = False,
) -> None:
    """Validate that incoming config matches checkpoint across all V2 method-critical fields, hash, and git commit."""
    if not allow_cross_commit and ckpt_commit and current_commit:
        if ckpt_commit != "unknown" and current_commit != "unknown" and ckpt_commit != current_commit:
            raise ValueError(
                f"Resume cross-commit mismatch: checkpoint was created on git commit {ckpt_commit!r}, "
                f"but current environment is at commit {current_commit!r}. "
                f"Resuming across different git commits is disallowed to prevent hybrid code trajectories. "
                f"To override this check intentionally, pass --allow-cross-commit-resume."
            )

    if ckpt_hash is not None and incoming_hash is not None:
        if ckpt_hash != incoming_hash:
            raise ValueError(
                f"Resume config hash mismatch: checkpoint SHA256 {ckpt_hash} != incoming SHA256 {incoming_hash}. "
                f"Resuming requires matching experiment configuration to guarantee trajectory continuity."
            )

    if not isinstance(ckpt_cfg, dict) or not isinstance(incoming_cfg, dict):
        return

    # Check seed immutability
    if "seed" in ckpt_cfg and "seed" in incoming_cfg:
        if int(ckpt_cfg["seed"]) != int(incoming_cfg["seed"]):
            raise ValueError(
                f"Resume config mismatch for 'seed': "
                f"checkpoint has {ckpt_cfg['seed']!r} but incoming config has {incoming_cfg['seed']!r}."
            )

    for section, fields in V2_METHOD_CRITICAL_FIELDS.items():
        ckpt_sec = ckpt_cfg.get(section, {})
        inc_sec = incoming_cfg.get(section, {})
        if not isinstance(ckpt_sec, dict) or not isinstance(inc_sec, dict):
            continue

        for field in fields:
            if field in ("backbone", "backbone_name"):
                v_ckpt = ckpt_sec.get("backbone", ckpt_sec.get("backbone_name"))
                v_inc = inc_sec.get("backbone", inc_sec.get("backbone_name"))
            elif field in ("omega", "sirt_omega"):
                v_ckpt = ckpt_sec.get("omega", ckpt_sec.get("sirt_omega"))
                v_inc = inc_sec.get("omega", inc_sec.get("sirt_omega"))
            elif field in ("lambda_count", "lambda_global"):
                v_ckpt = ckpt_sec.get("lambda_count", ckpt_sec.get("lambda_global"))
                v_inc = inc_sec.get("lambda_count", inc_sec.get("lambda_global"))
            else:
                default_val = V2_CRITICAL_TRAIN_DEFAULTS.get(field) if section == "train" else None
                v_ckpt = ckpt_sec.get(field, default_val)
                v_inc = inc_sec.get(field, default_val)

            if v_ckpt is not None and v_inc is not None:
                if not _are_values_compatible(v_ckpt, v_inc):
                    raise ValueError(
                        f"Resume config mismatch for '{section}.{field}': "
                        f"checkpoint has {v_ckpt!r} but incoming config has {v_inc!r}. "
                        f"Resuming requires matching experiment configuration to guarantee trajectory continuity."
                    )
