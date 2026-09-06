from __future__ import annotations

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
