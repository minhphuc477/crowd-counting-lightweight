from __future__ import annotations

import hashlib
import json
from typing import Any

ALLOWED_TOP_LEVEL = {
    "seed",
    "output_dir",
    "data",
    "model",
    "loss",
    "train",
    "eval",
}

ALLOWED_DATA_KEYS = {
    "train_manifest",
    "val_manifest",
    "crop_size",
    "scale_range",
    "hflip_prob",
    "brightness_jitter",
    "contrast_jitter",
    "data_root",
}

ALLOWED_MODEL_KEYS = {
    "output_stride",
    "feature_width",
    "backbone",
    "backbone_name",
    "pretrained",
    "backbone_lr_scale",
    "init_m0",
    "region_sizes_px",
    "region_overlap",
    "include_full_image",
    "iterations",
    "omega",
    "sirt_omega",
    "residual_clip",
    "dispersion_init",
    "dispersion_min",
    "dispersion_max",
    "reliability_mode",
    "reliability_rate_std_floor",
    "reliability_weight_min",
    "reliability_weight_max",
    "normalize_reliability_within_scale",
    "detach_region_mean_in_solver",
    "detach_reliability_in_solver",
    "uniform_reliability",
    "eps",
}

ALLOWED_LOSS_KEYS = {
    "lambda_count",
    "lambda_flat_dm16",
    "lambda_cell",
    "lambda_region_nb",
    "count_loss_mode",
    "count_nb_dispersion",
    "kappa_flat16",
    "normalize_flat_dm16",
    "cell_beta",
}

ALLOWED_TRAIN_KEYS = {
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
    "early_stopping",
    "patience",
    "solver_warmup_epochs",
    "solver_ramp_epochs",
    "deterministic",
}

ALLOWED_EVAL_KEYS = {
    "density_bins",
}

SECTION_ALLOWED_KEYS = {
    "data": ALLOWED_DATA_KEYS,
    "model": ALLOWED_MODEL_KEYS,
    "loss": ALLOWED_LOSS_KEYS,
    "train": ALLOWED_TRAIN_KEYS,
    "eval": ALLOWED_EVAL_KEYS,
}


def validate_v3_config(cfg: dict[str, Any]) -> None:
    """Validate RMR-v3 configuration dict and raise ValueError on any unknown/misspelled keys."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(cfg).__name__}")

    # Check top-level keys
    for k in cfg:
        if k not in ALLOWED_TOP_LEVEL:
            raise ValueError(
                f"Unknown top-level config key '{k}'. Allowed keys: {sorted(ALLOWED_TOP_LEVEL)}"
            )

    # Check sections
    for section, allowed in SECTION_ALLOWED_KEYS.items():
        if section in cfg:
            sec_dict = cfg[section]
            if not isinstance(sec_dict, dict):
                raise ValueError(f"Config section '{section}' must be a dictionary, got {type(sec_dict).__name__}")
            for k in sec_dict:
                if k not in allowed:
                    raise ValueError(
                        f"Unknown config key '{k}' in section '{section}'. Allowed keys: {sorted(allowed)}"
                    )

    # Validate logical bounds if present
    m_cfg = cfg.get("model", {})
    if "reliability_weight_min" in m_cfg:
        w_min = float(m_cfg["reliability_weight_min"])
        if w_min <= 0.0:
            raise ValueError(f"reliability_weight_min must be strictly positive, got {w_min}")
    if "dispersion_min" in m_cfg:
        d_min = float(m_cfg["dispersion_min"])
        if d_min <= 0.0:
            raise ValueError(f"dispersion_min must be strictly positive, got {d_min}")
    if "iterations" in m_cfg:
        iters = int(m_cfg["iterations"])
        if iters < 0:
            raise ValueError(f"iterations must be non-negative, got {iters}")


METHOD_CRITICAL_FIELDS: dict[str, list[str]] = {
    "model": [
        "output_stride",
        "feature_width",
        "backbone_name",
        "backbone",
        "region_sizes_px",
        "region_overlap",
        "include_full_image",
        "iterations",
        "omega",
        "sirt_omega",
        "residual_clip",
        "dispersion_init",
        "dispersion_min",
        "dispersion_max",
        "reliability_mode",
        "reliability_rate_std_floor",
        "reliability_weight_min",
        "reliability_weight_max",
        "normalize_reliability_within_scale",
        "detach_region_mean_in_solver",
        "detach_reliability_in_solver",
        "uniform_reliability",
        "eps",
    ],
    "loss": [
        "lambda_count",
        "lambda_flat_dm16",
        "lambda_cell",
        "lambda_region_nb",
        "count_loss_mode",
        "count_nb_dispersion",
        "kappa_flat16",
        "normalize_flat_dm16",
        "cell_beta",
    ],
    "train": [
        "lr",
        "backbone_lr_scale",
        "weight_decay",
        "epochs",
        "warmup_epochs",
        "batch_size",
        "deterministic",
        "solver_warmup_epochs",
        "solver_ramp_epochs",
    ],
    "data": [
        "crop_size",
        "scale_range",
        "hflip_prob",
        "brightness_jitter",
        "contrast_jitter",
    ],
}


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 digest of the configuration dictionary."""
    normalized_json = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(normalized_json.encode("utf-8")).hexdigest()


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


def validate_resume_compatibility(
    ckpt_cfg: dict[str, Any],
    incoming_cfg: dict[str, Any],
    ckpt_hash: str | None = None,
    incoming_hash: str | None = None,
) -> None:
    """Validate that incoming config matches checkpoint across all method-critical fields and hash."""
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
                f"checkpoint has {ckpt_cfg['seed']!r} but incoming config has {incoming_cfg['seed']!r}. "
                f"Resuming requires identical seed for trajectory continuity."
            )

    for section, fields in METHOD_CRITICAL_FIELDS.items():
        ckpt_sec = ckpt_cfg.get(section, {})
        inc_sec = incoming_cfg.get(section, {})
        if not isinstance(ckpt_sec, dict) or not isinstance(inc_sec, dict):
            continue

        for field in fields:
            # Handle backbone alias
            if field in ("backbone", "backbone_name"):
                v_ckpt = ckpt_sec.get("backbone", ckpt_sec.get("backbone_name"))
                v_inc = inc_sec.get("backbone", inc_sec.get("backbone_name"))
            elif field in ("omega", "sirt_omega"):
                v_ckpt = ckpt_sec.get("omega", ckpt_sec.get("sirt_omega"))
                v_inc = inc_sec.get("omega", inc_sec.get("sirt_omega"))
            else:
                v_ckpt = ckpt_sec.get(field)
                v_inc = inc_sec.get(field)

            if v_ckpt is not None and v_inc is not None:
                if not _are_values_compatible(v_ckpt, v_inc):
                    raise ValueError(
                        f"Resume config mismatch for '{section}.{field}': "
                        f"checkpoint has {v_ckpt!r} but incoming config has {v_inc!r}. "
                        f"Resuming requires matching experiment configuration to guarantee trajectory continuity."
                    )

