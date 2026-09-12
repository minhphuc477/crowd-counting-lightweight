from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from rmr_core.data import resolve_manifest_path
from rmr_core.training import compute_file_sha256
from .model import RMRv3Config

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
    "gamma_jitter",
    "random_invert_prob",
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
    "neck_type",
    "region_sizes_px",
    "region_overlap",
    "include_full_image",
    "enable_solver",
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
    "native_scale_pooling",
    "regional_feature_stats",
    "context_dilations",
    "use_aspp_gap",
    "aspp_dilations",
    "region_head_hidden",
    # RMR-v7
    "hurdle_head",
    "tv_lambda",
    "ema_decay",
    "temp_softplus",
    # RMR-v8 Stage 2
    "solver_mode",
    "density_gate_rho",
    "density_gate_floor",
    "tv_type",
    "tv_eps_c",
    # RMR-v8 Stage 3
    "use_coord_attn",
    # RMR-v9.1 / AQ-RMR additions
    "proximal_tau",
    # RMR-v10 additions
    "proximal_mode",
    "proximal_mu",
    "dynamic_scale_routing",
    "scale_router_temperature",
}

ALLOWED_LOSS_KEYS = {
    "lambda_count",
    "lambda_flat_dm16",
    "lambda_cell",
    "lambda_region_nb",
    "lambda_hurdle",
    "lambda_trunc_nb",
    "allocation_loss_type",
    "bayesian_sigma",
    "bayesian_background_ratio",
    "ot_reg",
    "ot_num_iters",
    "count_loss_mode",
    "count_nb_dispersion",
    "kappa_flat16",
    "normalize_flat_dm16",
    "cell_beta",
    "use_hierarchical_dm",
    "use_multiscale_dm",
    "dm_block_sizes_px",
    "dm_weights",
    "dm_kappas",
    # RMR-v8 Stage 2 & 3
    "cell_loss_mode",
    "cell_mass_weight_eps",
    "cell_mass_weight_alpha",
    "lambda_kd_spatial",
    "lambda_kd_count",
    # RMR-v9: allocation loss target ("y0" | "y")
    "dm_target",
    "dm_strict",
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
    "ema_decay",
    "teacher_ckpt",
    "min_lr_ratio",
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

    # Validate data manifest invariants
    d_cfg = cfg.get("data", {})
    if isinstance(d_cfg, dict):
        for mk in ("train_manifest", "val_manifest"):
            m_val = str(d_cfg.get(mk, "")).replace("\\", "/")
            if m_val.endswith("sha_a_train.jsonl") or m_val.endswith("sha_a_val.jsonl"):
                raise ValueError(
                    f"Ad-hoc split manifest '{m_val}' in data.{mk} is strictly forbidden under the Zero Ad-hoc Split Policy! "
                    f"For ShanghaiTech Part A, use 'data/sha_a_train_all.jsonl' (300 samples) and 'data/sha_a_test.jsonl' (182 samples)."
                )

    # Validate logical bounds and alias collisions
    m_cfg = cfg.get("model", {})
    if isinstance(m_cfg, dict):
        if "backbone" in m_cfg and "backbone_name" in m_cfg:
            raise ValueError(
                "Conflicting alias keys in model config: cannot declare both 'backbone' and 'backbone_name'."
            )
        if "omega" in m_cfg and "sirt_omega" in m_cfg:
            raise ValueError(
                "Conflicting alias keys in model config: cannot declare both 'omega' and 'sirt_omega'."
            )
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
            if iters < 1:
                raise ValueError(f"iterations must be >= 1, got {iters}")
        if "regional_feature_stats" in m_cfg:
            stats = str(m_cfg["regional_feature_stats"])
            if stats not in ("mean", "mean_std"):
                raise ValueError(f"regional_feature_stats must be 'mean' or 'mean_std', got '{stats}'")
        if "solver_mode" in m_cfg:
            solver_mode = str(m_cfg["solver_mode"])
            if solver_mode not in ("additive", "multiplicative"):
                raise ValueError(
                    f"solver_mode must be 'additive' or 'multiplicative', got '{solver_mode}'"
                )
        if "tv_type" in m_cfg:
            tv_type = str(m_cfg["tv_type"])
            if tv_type not in ("laplacian", "charbonnier"):
                raise ValueError(
                    f"tv_type must be 'laplacian' or 'charbonnier', got '{tv_type}'"
                )
        if "density_gate_rho" in m_cfg:
            rho = float(m_cfg["density_gate_rho"])
            if rho <= 0.0:
                raise ValueError(f"density_gate_rho must be strictly positive, got {rho}")
        if "tv_eps_c" in m_cfg:
            eps_c = float(m_cfg["tv_eps_c"])
            if eps_c <= 0.0:
                raise ValueError(f"tv_eps_c must be strictly positive, got {eps_c}")
        if m_cfg.get("tv_type") == "charbonnier":
            tv_lambda = float(m_cfg.get("tv_lambda", 0.0))
            tv_eps_c = float(m_cfg.get("tv_eps_c", 0.1))
            if tv_lambda > 0.0 and tv_eps_c > 0.0:
                cfl_limit = tv_eps_c / 4.0
                if tv_lambda > cfl_limit:
                    raise ValueError(
                        f"Charbonnier TV CFL stability violation: tv_lambda ({tv_lambda}) must be <= "
                        f"tv_eps_c / 4 ({cfl_limit}) to ensure contractivity and prevent numerical divergence."
                    )
        if "proximal_tau" in m_cfg:
            ptau = float(m_cfg["proximal_tau"])
            if ptau < 0.0:
                raise ValueError(f"proximal_tau must be non-negative, got {ptau}")
        if "proximal_mode" in m_cfg:
            pmode = str(m_cfg["proximal_mode"])
            if pmode not in ("firm", "soft", "none", "clamp"):
                raise ValueError(f"proximal_mode must be 'firm', 'soft', or 'none', got '{pmode}'")
        if "proximal_mu" in m_cfg:
            pmu = float(m_cfg["proximal_mu"])
            if pmu <= 1.0:
                raise ValueError(f"proximal_mu must be > 1.0, got {pmu}")
        if "tv_lambda" in m_cfg:
            tvl = float(m_cfg["tv_lambda"])
            if tvl < 0.0:
                raise ValueError(f"tv_lambda must be non-negative, got {tvl}")
        if "scale_router_temperature" in m_cfg:
            stemp = float(m_cfg["scale_router_temperature"])
            if stemp <= 0.0:
                raise ValueError(f"scale_router_temperature must be strictly positive, got {stemp}")
        if bool(m_cfg.get("use_coord_attn", False)):
            neck = str(m_cfg.get("neck_type", "additive"))
            if neck != "aspp_lite":
                raise ValueError(
                    f"use_coord_attn=True requires neck_type='aspp_lite', got '{neck}'"
                )

    # Validate loss section — Stage 2 extensions
    l_cfg_pre = cfg.get("loss", {})
    if isinstance(l_cfg_pre, dict):
        if "dm_target" in l_cfg_pre:
            dmt = str(l_cfg_pre["dm_target"])
            if dmt not in ("y", "y0", "dual"):
                raise ValueError(f"dm_target must be 'y', 'y0', or 'dual', got '{dmt}'")
        if "dm_strict" in l_cfg_pre:
            if not isinstance(l_cfg_pre["dm_strict"], bool):
                raise ValueError(f"dm_strict must be a boolean, got {type(l_cfg_pre['dm_strict']).__name__}")
        if "count_loss_mode" in l_cfg_pre:
            clm = str(l_cfg_pre["count_loss_mode"])
            if clm not in ("nb", "log1p", "l1"):
                raise ValueError(f"count_loss_mode must be 'nb', 'log1p', or 'l1', got '{clm}'")
        if "allocation_loss_type" in l_cfg_pre:
            alt = str(l_cfg_pre["allocation_loss_type"])
            if alt not in ("flat_dm16", "bayesian", "ot_sinkhorn"):
                raise ValueError(
                    f"allocation_loss_type must be 'flat_dm16', 'bayesian', or 'ot_sinkhorn', got '{alt}'"
                )
        if "cell_loss_mode" in l_cfg_pre:
            clm = str(l_cfg_pre["cell_loss_mode"])
            if clm not in ("balanced", "mass_weighted"):
                raise ValueError(
                    f"cell_loss_mode must be 'balanced' or 'mass_weighted', got '{clm}'"
                )
        if "cell_mass_weight_eps" in l_cfg_pre:
            eps_mw = float(l_cfg_pre["cell_mass_weight_eps"])
            if eps_mw <= 0.0:
                raise ValueError(f"cell_mass_weight_eps must be strictly positive, got {eps_mw}")


    l_cfg = cfg.get("loss", {})
    if isinstance(l_cfg, dict):
        if bool(l_cfg.get("use_multiscale_dm", False)) or bool(l_cfg.get("use_hierarchical_dm", False)):
            b_sizes = l_cfg.get("dm_block_sizes_px", (16, 32, 64))
            weights = l_cfg.get("dm_weights", (0.50, 0.30, 0.20))
            kappas = l_cfg.get("dm_kappas", (20.0, 20.0, 20.0))

            if not b_sizes:
                raise ValueError("dm_block_sizes_px cannot be empty when Multi-Scale DM is enabled")

            if not (len(b_sizes) == len(weights) == len(kappas)):
                raise ValueError(
                    f"Multi-Scale DM length mismatch: dm_block_sizes_px ({len(b_sizes)}), "
                    f"dm_weights ({len(weights)}), dm_kappas ({len(kappas)})"
                )

            stride = int(cfg.get("model", {}).get("output_stride", 4)) if isinstance(cfg.get("model"), dict) else 4
            for b in b_sizes:
                if not isinstance(b, int) or b <= 0:
                    raise ValueError(f"dm_block_sizes_px elements must be positive integers, got {b}")
                if b % stride != 0:
                    raise ValueError(
                        f"dm_block_sizes_px element {b} must be divisible by model output_stride ({stride})"
                    )

            for k in kappas:
                if float(k) <= 0:
                    raise ValueError(f"dm_kappas elements must be > 0, got {k}")

            for w in weights:
                if float(w) < 0:
                    raise ValueError(f"dm_weights elements must be non-negative, got {w}")

            if sum(float(w) for w in weights) <= 0:
                raise ValueError("dm_weights must sum to > 0")

    # Validate train parameter bounds
    t_cfg = cfg.get("train", {})
    if isinstance(t_cfg, dict):
        if "lr" in t_cfg:
            lr = float(t_cfg["lr"])
            if lr <= 0:
                raise ValueError(f"train.lr must be strictly positive, got {lr}")
        if "epochs" in t_cfg:
            epochs = int(t_cfg["epochs"])
            if epochs < 1:
                raise ValueError(f"train.epochs must be >= 1, got {epochs}")
        if "eval_every" in t_cfg:
            eval_every = int(t_cfg["eval_every"])
            if eval_every < 1:
                raise ValueError(f"train.eval_every must be >= 1, got {eval_every}")
        if "batch_size" in t_cfg:
            bs = int(t_cfg["batch_size"])
            if bs < 1:
                raise ValueError(f"train.batch_size must be >= 1, got {bs}")
        if "workers" in t_cfg:
            workers = int(t_cfg["workers"])
            if workers < 0:
                raise ValueError(f"train.workers must be >= 0, got {workers}")
        if "grad_clip" in t_cfg:
            gc = float(t_cfg["grad_clip"])
            if gc <= 0:
                raise ValueError(f"train.grad_clip must be strictly positive, got {gc}")
        if "backbone_lr_scale" in t_cfg:
            scale = float(t_cfg["backbone_lr_scale"])
            if scale <= 0:
                raise ValueError(f"train.backbone_lr_scale must be strictly positive, got {scale}")
        if "weight_decay" in t_cfg:
            wd = float(t_cfg["weight_decay"])
            if wd < 0:
                raise ValueError(f"train.weight_decay must be non-negative, got {wd}")
        if "patience" in t_cfg:
            patience = int(t_cfg["patience"])
            if patience < 0:
                raise ValueError(f"train.patience must be >= 0, got {patience}")
        if "warmup_epochs" in t_cfg:
            warmup = int(t_cfg["warmup_epochs"])
            if warmup < 0:
                raise ValueError(f"train.warmup_epochs must be >= 0, got {warmup}")
        if "solver_warmup_epochs" in t_cfg:
            sw = int(t_cfg["solver_warmup_epochs"])
            if sw < 0:
                raise ValueError(f"train.solver_warmup_epochs must be >= 0, got {sw}")
        if "solver_ramp_epochs" in t_cfg:
            sr = int(t_cfg["solver_ramp_epochs"])
            if sr < 0:
                raise ValueError(f"train.solver_ramp_epochs must be >= 0, got {sr}")

    # Validate eval configuration
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
        "native_scale_pooling",
        "regional_feature_stats",
        "region_head_hidden",
        # Neck architecture fields (changing these changes model weights layout)
        "neck_type",
        "context_dilations",
        "use_aspp_gap",
        "aspp_dilations",
        # RMR-v7 critical fields (changing these invalidates checkpoint weights)
        "hurdle_head",
        "temp_softplus",
        # RMR-v8 Stage 2 solver fields (changing these changes the SIRT update rule)
        "solver_mode",
        "density_gate_rho",
        "density_gate_floor",
        "tv_type",
        "tv_lambda",
        "tv_eps_c",
        # RMR-v8 Stage 3 architecture
        "use_coord_attn",
        # RMR-v9.1 / AQ-RMR additions
        "proximal_tau",
        # RMR-v10 additions
        "proximal_mode",
        "proximal_mu",
        "dynamic_scale_routing",
        "scale_router_temperature",
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
        "use_hierarchical_dm",
        "use_multiscale_dm",
        "dm_block_sizes_px",
        "dm_weights",
        "dm_kappas",
        # RMR-v8 Stage 2 loss fields
        "cell_loss_mode",
        "cell_mass_weight_alpha",
        "cell_mass_weight_eps",
        # RMR-v9 allocation loss fields
        "allocation_loss_type",
        "dm_target",
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
        "workers",
        "eval_every",
        "early_stopping",
        "patience",
    ],
    "data": [
        "crop_size",
        "scale_range",
        "hflip_prob",
        "brightness_jitter",
        "contrast_jitter",
        "gamma_jitter",
        "random_invert_prob",
    ],

}

CRITICAL_TRAIN_DEFAULTS: dict[str, Any] = {
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
    ckpt_commit: str | None = None,
    current_commit: str | None = None,
    allow_cross_commit: bool = False,
) -> None:
    """Validate that incoming config matches checkpoint across all method-critical fields, hash, and git commit."""
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
                default_val = CRITICAL_TRAIN_DEFAULTS.get(field) if section == "train" else None
                v_ckpt = ckpt_sec.get(field, default_val)
                v_inc = inc_sec.get(field, default_val)

            if v_ckpt is not None and v_inc is not None:
                if not _are_values_compatible(v_ckpt, v_inc):
                    raise ValueError(
                        f"Resume config mismatch for '{section}.{field}': "
                        f"checkpoint has {v_ckpt!r} but incoming config has {v_inc!r}. "
                        f"Resuming requires matching experiment configuration to guarantee trajectory continuity."
                    )


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


