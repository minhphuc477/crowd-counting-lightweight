from __future__ import annotations

from typing import Any

from .schema import ALLOWED_TOP_LEVEL, SECTION_ALLOWED_KEYS


def validate_v3_config(cfg: dict[str, Any]) -> None:
    """Validate RMR-v3 configuration dict and raise ValueError on any unknown/misspelled keys."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a dictionary, got {type(cfg).__name__}")

    # Check top-level keys
    for k in list(cfg.keys()):
        clean_k = k.lstrip("\ufeff") if isinstance(k, str) else k
        if clean_k != k:
            cfg[clean_k] = cfg.pop(k)
        if clean_k not in ALLOWED_TOP_LEVEL:
            raise ValueError(
                f"Unknown top-level config key '{clean_k}'. Allowed keys: {sorted(ALLOWED_TOP_LEVEL)}"
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
        if "pad_small_images" in d_cfg and not isinstance(d_cfg["pad_small_images"], bool):
            raise ValueError(f"data.pad_small_images must be a boolean, got {type(d_cfg['pad_small_images']).__name__}")
        for mk in ("train_manifest", "val_manifest"):
            m_val = str(d_cfg.get(mk, "")).replace("\\", "/")
            if m_val.endswith("sha_a_train.jsonl") or m_val.endswith("sha_a_val.jsonl"):
                raise ValueError(
                    f"Ad-hoc split manifest '{m_val}' in data.{mk} is strictly forbidden "
                    "under the Zero Ad-hoc Split Policy! For ShanghaiTech Part A, use "
                    "'data/sha_a_train_all.jsonl' (300 samples) and 'data/sha_a_test.jsonl' (182 samples)."
                )

    # Validate logical bounds and alias collisions
    m_cfg = cfg.get("model", {})
    if isinstance(m_cfg, dict):
        if "hybrid_recovery_alpha" in m_cfg:
            hra = float(m_cfg["hybrid_recovery_alpha"])
            if not (0.0 <= hra <= 1.0):
                raise ValueError(f"hybrid_recovery_alpha must be in [0.0, 1.0], got {hra}")
        if "curvature_alpha_init" in m_cfg:
            try:
                float(m_cfg["curvature_alpha_init"])
            except (ValueError, TypeError) as err:
                raise ValueError(
                    f"curvature_alpha_init must be a float, got {m_cfg['curvature_alpha_init']!r}"
                ) from err
        if "backbone" in m_cfg and "backbone_name" in m_cfg:
            raise ValueError(
                "Conflicting alias keys in model config: cannot declare both 'backbone' and 'backbone_name'."
            )
        if "omega" in m_cfg and "sirt_omega" in m_cfg:
            raise ValueError(
                "Conflicting alias keys in model config: cannot declare both 'omega' and 'sirt_omega'."
            )
        has_subpixel = bool(m_cfg.get("subpixel_stride2", False)) or bool(m_cfg.get("subpixel_dm", False))
        if has_subpixel and int(m_cfg.get("output_stride", 4)) != 2:
            raise ValueError(
                f"Inconsistent config: subpixel_stride2=True or subpixel_dm=True requires output_stride=2, "
                f"got output_stride={m_cfg.get('output_stride')}"
            )
        if int(m_cfg.get("output_stride", 4)) == 2 and not has_subpixel:
            raise ValueError(
                "Inconsistent config: output_stride=2 requires subpixel_stride2=True or subpixel_dm=True."
            )
        for k in (
            "reliability_weight_min", "dispersion_min", "density_gate_rho", "tv_eps_c",
            "pm_kappa", "anscombe_tau_dense", "scale_router_temperature", "trust_region_floor",
            "anscombe_c", "adaptive_tau_rho0", "micro_coord_reduction", "adaptive_relax_sparse",
            "adaptive_relax_threshold", "adaptive_relax_scale", "shifted_carrier_eps",
            "trust_dense_tau", "trust_dense_kappa", "density_scale_tau",
        ):
            if k in m_cfg and float(m_cfg[k]) <= 0.0:
                raise ValueError(f"{k} must be strictly positive, got {m_cfg[k]}")

        for k in (
            "proximal_tau", "tv_lambda", "trust_region_kappa", "morozov_gamma",
            "adaptive_relax_dense_boost", "crest_kappa_0", "crest_eps_seed",
            "morozov_gamma_under", "morozov_rho",
        ):
            if k in m_cfg and float(m_cfg[k]) < 0.0:
                raise ValueError(f"{k} must be non-negative, got {m_cfg[k]}")

        if "iterations" in m_cfg:
            iters = int(m_cfg["iterations"])
            if iters < 1 and bool(m_cfg.get("enable_solver", True)):
                raise ValueError(f"iterations must be >= 1 when enable_solver is True, got {iters}")
        if "regional_feature_stats" in m_cfg:
            stats = str(m_cfg["regional_feature_stats"])
            if stats not in ("mean", "mean_std"):
                raise ValueError(f"regional_feature_stats must be 'mean' or 'mean_std', got '{stats}'")
        if "solver_mode" in m_cfg:
            solver_mode = str(m_cfg["solver_mode"])
            if solver_mode not in ("additive", "multiplicative"):
                raise ValueError(f"solver_mode must be 'additive' or 'multiplicative', got '{solver_mode}'")
        if "hurdle_gating_mode" in m_cfg:
            hgm = str(m_cfg["hurdle_gating_mode"])
            if hgm not in ("occupancy", "product", "disabled", "unbiased"):
                raise ValueError(
                    f"hurdle_gating_mode must be 'occupancy', 'product', 'disabled', or 'unbiased', got '{hgm}'"
                )
        if "tv_type" in m_cfg:
            tv_type = str(m_cfg["tv_type"])
            if tv_type not in ("laplacian", "charbonnier", "perona_malik"):
                raise ValueError(f"tv_type must be 'laplacian', 'charbonnier', or 'perona_malik', got '{tv_type}'")
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
        if "proximal_mode" in m_cfg:
            pmode = str(m_cfg["proximal_mode"])
            if pmode not in ("firm", "soft", "none", "clamp"):
                raise ValueError(f"proximal_mode must be 'firm', 'soft', or 'none', got '{pmode}'")
        if "proximal_mu" in m_cfg:
            pmu = float(m_cfg["proximal_mu"])
            if pmu <= 1.0:
                raise ValueError(f"proximal_mu must be > 1.0, got {pmu}")
        if bool(m_cfg.get("use_coord_attn", False)):
            neck = str(m_cfg.get("neck_type", "additive"))
            if neck != "aspp_lite":
                raise ValueError(f"use_coord_attn=True requires neck_type='aspp_lite', got '{neck}'")
        if "adjoint_mode" in m_cfg:
            adj = str(m_cfg["adjoint_mode"])
            if adj not in ("flat", "radon_nikodym", "anscombe_vst"):
                raise ValueError(
                    f"adjoint_mode must be 'flat', 'radon_nikodym', or 'anscombe_vst', got '{adj}'"
                )
        if "reliability_mode" in m_cfg:
            rmode = str(m_cfg["reliability_mode"])
            if rmode not in ("nb_rate_variance", "rate_variance", "snr", "hybrid_hurdle"):
                raise ValueError(
                    f"reliability_mode must be 'nb_rate_variance', 'snr', or 'hybrid_hurdle', got '{rmode}'"
                )
        if "hybrid_recovery_alpha" in m_cfg:
            hra = float(m_cfg["hybrid_recovery_alpha"])
            if hra < 0.0 or hra > 1.0:
                raise ValueError(f"hybrid_recovery_alpha must be in [0.0, 1.0], got {hra}")
        if bool(m_cfg.get("density_scale_gating", False)) and not bool(m_cfg.get("dynamic_scale_routing", True)):
            raise ValueError("density_scale_gating=True requires dynamic_scale_routing=True.")
        if bool(m_cfg.get("use_park", False)):
            raise ValueError("use_park=True is permanently BANNED. Use dynamic DiAG instead.")
        if bool(m_cfg.get("dynamic_perspective_windows", False)):
            raise ValueError("dynamic_perspective_windows=True is permanently BANNED (Anti-Pattern #8).")


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
            if clm not in ("nb", "log1p", "l1", "smooth_l1", "huber", "anscombe"):
                raise ValueError(
                    f"count_loss_mode must be 'nb', 'log1p', 'l1', 'smooth_l1', or 'anscombe', got '{clm}'"
                )
        if "allocation_loss_type" in l_cfg_pre:
            alt = str(l_cfg_pre["allocation_loss_type"])
            if alt not in ("flat_dm16", "bayesian", "ot_sinkhorn", "fidt", "dual_bayesian_dm16"):
                raise ValueError(
                    "allocation_loss_type must be 'flat_dm16', 'bayesian', 'ot_sinkhorn', "
                    f"'fidt', or 'dual_bayesian_dm16', got '{alt}'"
                )
        if "cell_loss_mode" in l_cfg_pre:
            clm = str(l_cfg_pre["cell_loss_mode"])
            if clm not in ("balanced", "mass_weighted", "count_invariant", "ci_cell", "count_harmonized"):
                raise ValueError(
                    "cell_loss_mode must be 'balanced', 'mass_weighted', 'count_invariant', "
                    f"or 'count_harmonized', got '{clm}'"
                )
        if "cell_tau_head" in l_cfg_pre and float(l_cfg_pre["cell_tau_head"]) <= 0.0:
            raise ValueError(f"cell_tau_head must be strictly positive, got {l_cfg_pre['cell_tau_head']}")
        if "cell_alpha" in l_cfg_pre and float(l_cfg_pre["cell_alpha"]) < 0.0:
            raise ValueError(f"cell_alpha must be non-negative, got {l_cfg_pre['cell_alpha']}")
        if "cell_mass_weight_eps" in l_cfg_pre and float(l_cfg_pre["cell_mass_weight_eps"]) <= 0.0:
            raise ValueError(f"cell_mass_weight_eps must be strictly positive, got {l_cfg_pre['cell_mass_weight_eps']}")
        if "cell_mass_weight_gamma" in l_cfg_pre and float(l_cfg_pre["cell_mass_weight_gamma"]) <= 0.0:
            raise ValueError(
                f"cell_mass_weight_gamma must be strictly positive, got {l_cfg_pre['cell_mass_weight_gamma']}"
            )
        if "hard_bg_ratio" in l_cfg_pre and not (0.0 < float(l_cfg_pre["hard_bg_ratio"]) <= 1.0):
            raise ValueError(f"hard_bg_ratio must be in (0.0, 1.0], got {l_cfg_pre['hard_bg_ratio']}")
        if "cell_fg_ratio" in l_cfg_pre and not (0.0 <= float(l_cfg_pre["cell_fg_ratio"]) <= 1.0):
            raise ValueError(f"cell_fg_ratio must be in [0.0, 1.0], got {l_cfg_pre['cell_fg_ratio']}")
        for key in (
            "lambda_curvature", "lambda_hard_bg", "lambda_fg_gate", "lambda_carrier_cell",
            "lambda_fine_cell", "lambda_spectral", "lambda_spectral_dc", "lambda_scale_align",
            "lambda_kd_spatial", "lambda_kd_count",
        ):
            if key in l_cfg_pre and float(l_cfg_pre[key]) < 0.0:
                raise ValueError(f"{key} must be non-negative, got {l_cfg_pre[key]}")
        if "bayesian_norm_mode" in l_cfg_pre:
            bnm = str(l_cfg_pre["bayesian_norm_mode"])
            if bnm not in ("canonical", "count", "square_root"):
                raise ValueError(f"bayesian_norm_mode must be 'canonical', 'count', or 'square_root', got '{bnm}'")
        for ak in ("adaptive_sigma", "bayesian_adaptive_sigma"):
            if ak in l_cfg_pre and not isinstance(l_cfg_pre[ak], bool):
                raise ValueError(f"{ak} must be a boolean, got {type(l_cfg_pre[ak]).__name__}")
        if "spectral_beta" in l_cfg_pre and float(l_cfg_pre["spectral_beta"]) <= 0.0:
            raise ValueError(f"spectral_beta must be strictly positive, got {l_cfg_pre['spectral_beta']}")
        if "spectral_transform" in l_cfg_pre and str(l_cfg_pre["spectral_transform"]) not in ("fft", "dct"):
            raise ValueError(f"spectral_transform must be 'fft' or 'dct', got '{l_cfg_pre['spectral_transform']}'")
        if "curvature_gate_threshold" in l_cfg_pre:
            cgt = float(l_cfg_pre["curvature_gate_threshold"])
            if cgt < 0.0:
                raise ValueError(f"curvature_gate_threshold must be non-negative, got {cgt}")
        if "curvature_gate_kernel" in l_cfg_pre:
            cgk = int(l_cfg_pre["curvature_gate_kernel"])
            if cgk <= 0 or cgk % 2 == 0:
                raise ValueError(f"curvature_gate_kernel must be a positive odd integer, got {cgk}")
        if "curvature_gate_mode" in l_cfg_pre:
            cgm = str(l_cfg_pre["curvature_gate_mode"])
            if cgm not in ("none", "hard", "soft"):
                raise ValueError(f"curvature_gate_mode must be 'none', 'hard', or 'soft', got '{cgm}'")
        if "scale_align_tau_dense" in l_cfg_pre or "scale_align_tau_sparse" in l_cfg_pre:
            tau_d = float(l_cfg_pre.get("scale_align_tau_dense", 0.12))
            tau_s = float(l_cfg_pre.get("scale_align_tau_sparse", 0.03))
            if tau_s <= 0.0:
                raise ValueError(f"scale_align_tau_sparse must be strictly positive, got {tau_s}")
            if tau_d <= tau_s:
                raise ValueError(f"scale_align_tau_dense ({tau_d}) must be > scale_align_tau_sparse ({tau_s})")
        if "scale_align_kernel" in l_cfg_pre:
            sak = int(l_cfg_pre["scale_align_kernel"])
            if sak <= 0 or sak % 2 == 0:
                raise ValueError(f"scale_align_kernel must be a positive odd integer, got {sak}")
        if "dense_loss_thresh" in l_cfg_pre:
            dlt = float(l_cfg_pre["dense_loss_thresh"])
            if dlt <= 0.0:
                raise ValueError(f"dense_loss_thresh must be strictly positive, got {dlt}")
        if "dense_loss_norm" in l_cfg_pre:
            dln = float(l_cfg_pre["dense_loss_norm"])
            if dln <= 0.0:
                raise ValueError(f"dense_loss_norm must be strictly positive, got {dln}")
        if "dense_loss_alpha" in l_cfg_pre:
            dla = float(l_cfg_pre["dense_loss_alpha"])
            if dla < 0.0:
                raise ValueError(f"dense_loss_alpha must be non-negative, got {dla}")
        if "dense_loss_max_boost" in l_cfg_pre:
            dlmb = float(l_cfg_pre["dense_loss_max_boost"])
            if dlmb < 0.0:
                raise ValueError(f"dense_loss_max_boost must be non-negative, got {dlmb}")
        if "elementwise_dense_scaling" in l_cfg_pre:
            if not isinstance(l_cfg_pre["elementwise_dense_scaling"], bool):
                raise ValueError(
                    "elementwise_dense_scaling must be a boolean, got "
                    f"{type(l_cfg_pre['elementwise_dense_scaling']).__name__}"
                )
        if "output_stride" in l_cfg_pre:
            os_val = int(l_cfg_pre["output_stride"])
            if os_val not in (2, 4):
                raise ValueError(f"output_stride in loss must be 2 or 4, got {os_val}")
        if "regional_mass_weight_alpha" in l_cfg_pre:
            rmwa = float(l_cfg_pre["regional_mass_weight_alpha"])
            if rmwa < 0.0:
                raise ValueError(f"regional_mass_weight_alpha must be non-negative, got {rmwa}")

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
        if "lr" in t_cfg and float(t_cfg["lr"]) <= 0:
            raise ValueError(f"train.lr must be strictly positive, got {t_cfg['lr']}")
        if "epochs" in t_cfg and int(t_cfg["epochs"]) < 1:
            raise ValueError(f"train.epochs must be >= 1, got {t_cfg['epochs']}")
        if "eval_every" in t_cfg and int(t_cfg["eval_every"]) < 1:
            raise ValueError(f"train.eval_every must be >= 1, got {t_cfg['eval_every']}")
        if "batch_size" in t_cfg and int(t_cfg["batch_size"]) < 1:
            raise ValueError(f"train.batch_size must be >= 1, got {t_cfg['batch_size']}")
        if "workers" in t_cfg and int(t_cfg["workers"]) < 0:
            raise ValueError(f"train.workers must be >= 0, got {t_cfg['workers']}")
        if "grad_clip" in t_cfg and float(t_cfg["grad_clip"]) <= 0:
            raise ValueError(f"train.grad_clip must be strictly positive, got {t_cfg['grad_clip']}")
        if "backbone_lr_scale" in t_cfg and float(t_cfg["backbone_lr_scale"]) <= 0:
            raise ValueError(f"train.backbone_lr_scale must be strictly positive, got {t_cfg['backbone_lr_scale']}")
        if "weight_decay" in t_cfg and float(t_cfg["weight_decay"]) < 0:
            raise ValueError(f"train.weight_decay must be non-negative, got {t_cfg['weight_decay']}")
        for k in ("patience", "warmup_epochs", "solver_warmup_epochs", "solver_ramp_epochs"):
            if k in t_cfg and int(t_cfg[k]) < 0:
                raise ValueError(f"train.{k} must be >= 0, got {t_cfg[k]}")
        if "wsd_stable_ratio" in t_cfg and not (0.0 <= float(t_cfg["wsd_stable_ratio"]) <= 1.0):
            raise ValueError(f"train.wsd_stable_ratio must be in [0.0, 1.0], got {t_cfg['wsd_stable_ratio']}")
        if "min_lr_ratio" in t_cfg and not (0.0 < float(t_cfg["min_lr_ratio"]) < 1.0):
            raise ValueError(f"train.min_lr_ratio must be in (0.0, 1.0), got {t_cfg['min_lr_ratio']}")
        if "grad_scaler_init_scale" in t_cfg and float(t_cfg["grad_scaler_init_scale"]) <= 0.0:
            raise ValueError(
                f"train.grad_scaler_init_scale must be strictly positive, got {t_cfg['grad_scaler_init_scale']}"
            )

    # Validate eval configuration
    e_cfg = cfg.get("eval", {})
    if isinstance(e_cfg, dict) and "density_bins" in e_cfg:
        bins = e_cfg["density_bins"]
        if not isinstance(bins, (list, tuple)) or len(bins) != 2:
            raise ValueError(
                f"eval.density_bins must be a list or tuple of exactly 2 thresholds [low, high], got {bins!r}"
            )
        try:
            b0, b1 = float(bins[0]), float(bins[1])
        except (ValueError, TypeError) as err:
            raise ValueError(f"eval.density_bins thresholds must be numeric, got {bins!r}") from err
        if b0 <= 0 or b1 <= 0:
            raise ValueError(f"eval.density_bins thresholds must be positive numbers, got [{b0}, {b1}]")
        if b0 >= b1:
            raise ValueError(f"eval.density_bins thresholds must satisfy low < high, got [{b0}, {b1}]")

    # Cross-section consistency: Solver requires trained regional evidence when both sections are present
    if "model" in cfg and "loss" in cfg:
        m_cfg, l_cfg = cfg.get("model", {}), cfg.get("loss", {})
        solver_active = (
            isinstance(m_cfg, dict) and isinstance(l_cfg, dict)
            and m_cfg.get("enable_solver", False) is True and len(l_cfg) > 0
        )
        if solver_active:
            allow_unsupervised = bool(
                m_cfg.get("allow_unsupervised_solver", False)
                or cfg.get("train", {}).get("allow_unsupervised_solver", False)
            )
            has_regional_supervision = (
                float(l_cfg.get("lambda_region_nb", 0.0)) > 0.0
                or float(l_cfg.get("lambda_trunc_nb", 0.0)) > 0.0
            )
            if not has_regional_supervision and not allow_unsupervised:
                raise ValueError(
                    "Invalid configuration: model.enable_solver=True requires "
                    "loss.lambda_region_nb > 0.0 or loss.lambda_trunc_nb > 0.0. "
                    "When both are <= 0.0, the regional head is never trained. "
                    "Set model.enable_solver=False for standalone carrier, "
                    "or set model.allow_unsupervised_solver=True for unconstrained ablation."
                )
