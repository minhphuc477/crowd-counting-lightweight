from __future__ import annotations

from typing import Any

from .schema import (
    ALLOWED_MODEL_KEYS,
    ALLOWED_TOP_LEVEL,
    SECTION_ALLOWED_KEYS,
)


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
        if bool(m_cfg.get("subpixel_stride2", False)) and int(m_cfg.get("output_stride", 4)) != 2:
            raise ValueError(
                f"Inconsistent config: subpixel_stride2=True requires output_stride=2, "
                f"got output_stride={m_cfg.get('output_stride')}"
            )
        if int(m_cfg.get("output_stride", 4)) == 2 and not bool(m_cfg.get("subpixel_stride2", False)):
            raise ValueError(
                "Inconsistent config: output_stride=2 requires subpixel_stride2=True."
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
            if tv_type not in ("laplacian", "charbonnier", "perona_malik"):
                raise ValueError(
                    f"tv_type must be 'laplacian', 'charbonnier', or 'perona_malik', got '{tv_type}'"
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
        if "pm_kappa" in m_cfg:
            pm_k = float(m_cfg["pm_kappa"])
            if pm_k <= 0.0:
                raise ValueError(f"pm_kappa must be strictly positive, got {pm_k}")
        if "anscombe_tau_dense" in m_cfg:
            at_dense = float(m_cfg["anscombe_tau_dense"])
            if at_dense <= 0.0:
                raise ValueError(f"anscombe_tau_dense must be strictly positive, got {at_dense}")
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
        if "trust_region_kappa" in m_cfg:
            kappa = float(m_cfg["trust_region_kappa"])
            if kappa < 0.0:
                raise ValueError(f"trust_region_kappa must be non-negative, got {kappa}")
        if "trust_region_floor" in m_cfg:
            floor_val = float(m_cfg["trust_region_floor"])
            if floor_val <= 0.0:
                raise ValueError(f"trust_region_floor must be strictly positive, got {floor_val}")
        if bool(m_cfg.get("use_coord_attn", False)):
            neck = str(m_cfg.get("neck_type", "additive"))
            if neck != "aspp_lite":
                raise ValueError(
                    f"use_coord_attn=True requires neck_type='aspp_lite', got '{neck}'"
                )
        if "adjoint_mode" in m_cfg:
            adj = str(m_cfg["adjoint_mode"])
            if adj not in ("flat", "radon_nikodym", "anscombe_vst"):
                raise ValueError(f"adjoint_mode must be 'flat', 'radon_nikodym', or 'anscombe_vst', got '{adj}'")
        if "anscombe_c" in m_cfg:
            ac = float(m_cfg["anscombe_c"])
            if ac <= 0.0:
                raise ValueError(f"anscombe_c must be strictly positive, got {ac}")
        if "adaptive_tau_rho0" in m_cfg:
            at_rho = float(m_cfg["adaptive_tau_rho0"])
            if at_rho <= 0.0:
                raise ValueError(f"adaptive_tau_rho0 must be strictly positive, got {at_rho}")
        if "morozov_gamma" in m_cfg:
            m_gamma = float(m_cfg["morozov_gamma"])
            if m_gamma < 0.0:
                raise ValueError(f"morozov_gamma must be non-negative, got {m_gamma}")
        if "reliability_mode" in m_cfg:
            rmode = str(m_cfg["reliability_mode"])
            if rmode not in ("nb_rate_variance", "rate_variance", "snr", "hybrid_hurdle"):
                raise ValueError(f"reliability_mode must be 'nb_rate_variance', 'snr', or 'hybrid_hurdle', got '{rmode}'")
        if "micro_coord_reduction" in m_cfg:
            mcr = int(m_cfg["micro_coord_reduction"])
            if mcr <= 0:
                raise ValueError(f"micro_coord_reduction must be positive, got {mcr}")
        if "adaptive_relax_sparse" in m_cfg:
            ars = float(m_cfg["adaptive_relax_sparse"])
            if ars <= 0.0:
                raise ValueError(f"adaptive_relax_sparse must be strictly positive, got {ars}")
        if "adaptive_relax_dense_boost" in m_cfg:
            ardb = float(m_cfg["adaptive_relax_dense_boost"])
            if ardb < 0.0:
                raise ValueError(f"adaptive_relax_dense_boost must be non-negative, got {ardb}")
        if "adaptive_relax_threshold" in m_cfg:
            art = float(m_cfg["adaptive_relax_threshold"])
            if art <= 0.0:
                raise ValueError(f"adaptive_relax_threshold must be strictly positive, got {art}")
        if "adaptive_relax_scale" in m_cfg:
            arsc = float(m_cfg["adaptive_relax_scale"])
            if arsc <= 0.0:
                raise ValueError(f"adaptive_relax_scale must be strictly positive, got {arsc}")
        if "hybrid_recovery_alpha" in m_cfg:
            hra = float(m_cfg["hybrid_recovery_alpha"])
            if hra < 0.0 or hra > 1.0:
                raise ValueError(f"hybrid_recovery_alpha must be in [0.0, 1.0], got {hra}")
        if "crest_kappa_0" in m_cfg and float(m_cfg["crest_kappa_0"]) < 0.0:
            raise ValueError(f"crest_kappa_0 must be non-negative, got {m_cfg['crest_kappa_0']}")
        if "crest_eps_seed" in m_cfg and float(m_cfg["crest_eps_seed"]) < 0.0:
            raise ValueError(f"crest_eps_seed must be non-negative, got {m_cfg['crest_eps_seed']}")
        if "morozov_gamma_under" in m_cfg and float(m_cfg["morozov_gamma_under"]) < 0.0:
            raise ValueError(f"morozov_gamma_under must be non-negative, got {m_cfg['morozov_gamma_under']}")
        if "morozov_rho" in m_cfg and float(m_cfg["morozov_rho"]) < 0.0:
            raise ValueError(f"morozov_rho must be non-negative, got {m_cfg['morozov_rho']}")
        if "curvature_alpha_init" in m_cfg:
            float(m_cfg["curvature_alpha_init"])
        if bool(m_cfg.get("use_park", False)):
            h_min, h_max = int(m_cfg.get("park_horizon_h", 16)), int(m_cfg.get("park_foreground_h", 128))
            asp, bands = float(m_cfg.get("park_max_aspect", 2.0)), int(m_cfg.get("park_altitude_bands", 3))
            if h_min <= 0 or h_max <= h_min:
                raise ValueError(f"Invalid park heights: horizon={h_min}, foreground={h_max}")
            if asp <= 0.0 or bands < 1:
                raise ValueError(f"Invalid park config: aspect={asp} (>0.0), bands={bands} (>=1)")

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
            if clm not in ("balanced", "mass_weighted", "count_invariant", "ci_cell", "count_harmonized"):
                raise ValueError(f"cell_loss_mode must be 'balanced', 'mass_weighted', 'count_invariant', or 'count_harmonized', got '{clm}'")
        if "cell_tau_head" in l_cfg_pre and float(l_cfg_pre["cell_tau_head"]) <= 0.0:
            raise ValueError(f"cell_tau_head must be strictly positive, got {l_cfg_pre['cell_tau_head']}")
        if "cell_alpha" in l_cfg_pre and float(l_cfg_pre["cell_alpha"]) < 0.0:
            raise ValueError(f"cell_alpha must be non-negative, got {l_cfg_pre['cell_alpha']}")
        if "cell_mass_weight_eps" in l_cfg_pre and float(l_cfg_pre["cell_mass_weight_eps"]) <= 0.0:
            raise ValueError(f"cell_mass_weight_eps must be strictly positive, got {l_cfg_pre['cell_mass_weight_eps']}")
        if "cell_mass_weight_gamma" in l_cfg_pre and float(l_cfg_pre["cell_mass_weight_gamma"]) <= 0.0:
            raise ValueError(f"cell_mass_weight_gamma must be strictly positive, got {l_cfg_pre['cell_mass_weight_gamma']}")
        if "lambda_curvature" in l_cfg_pre and float(l_cfg_pre["lambda_curvature"]) < 0.0:
            raise ValueError(f"lambda_curvature must be non-negative, got {l_cfg_pre['lambda_curvature']}")
        if "lambda_hard_bg" in l_cfg_pre and float(l_cfg_pre["lambda_hard_bg"]) < 0.0:
            raise ValueError(f"lambda_hard_bg must be non-negative, got {l_cfg_pre['lambda_hard_bg']}")
        if "hard_bg_ratio" in l_cfg_pre and not (0.0 < float(l_cfg_pre["hard_bg_ratio"]) <= 1.0):
            raise ValueError(f"hard_bg_ratio must be in (0.0, 1.0], got {l_cfg_pre['hard_bg_ratio']}")
        if "cell_fg_ratio" in l_cfg_pre and not (0.0 <= float(l_cfg_pre["cell_fg_ratio"]) <= 1.0):
            raise ValueError(f"cell_fg_ratio must be in [0.0, 1.0], got {l_cfg_pre['cell_fg_ratio']}")
        for key in ("lambda_fg_gate", "lambda_carrier_cell", "lambda_fine_cell", "lambda_kd_spatial", "lambda_kd_count"):
            if key in l_cfg_pre and float(l_cfg_pre[key]) < 0.0:
                raise ValueError(f"{key} must be non-negative, got {l_cfg_pre[key]}")
        if "lambda_spectral" in l_cfg_pre and float(l_cfg_pre["lambda_spectral"]) < 0.0:
            raise ValueError(f"lambda_spectral must be non-negative, got {l_cfg_pre['lambda_spectral']}")
        if "lambda_spectral_dc" in l_cfg_pre and float(l_cfg_pre["lambda_spectral_dc"]) < 0.0:
            raise ValueError(f"lambda_spectral_dc must be non-negative, got {l_cfg_pre['lambda_spectral_dc']}")
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
        if "lambda_scale_align" in l_cfg_pre:
            l_sa = float(l_cfg_pre["lambda_scale_align"])
            if l_sa < 0.0:
                raise ValueError(f"lambda_scale_align must be non-negative, got {l_sa}")
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
                    f"elementwise_dense_scaling must be a boolean, got {type(l_cfg_pre['elementwise_dense_scaling']).__name__}"
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
        if "grad_scaler_init_scale" in t_cfg:
            gsis = float(t_cfg["grad_scaler_init_scale"])
            if gsis <= 0.0:
                raise ValueError(f"train.grad_scaler_init_scale must be strictly positive, got {gsis}")

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
