from __future__ import annotations

"""Loss and Diagnostic Tracking for RMR-v3 / RMR-v11 Training.

Tracks multi-task training loss components across mini-batches, and collects
fine-grained solver energy reduction traces, dispersion dynamics, and regional
reliability metrics.
"""

from pathlib import Path
from typing import Any
import numpy as np
import torch

from .model import RMRv3


class LossTracker:
    """Tracks and averages multi-task training loss components across mini-batches."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.count: int = 0

    def update(self, loss_dict: dict[str, Any]) -> None:
        self.count += 1
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                val = float(v.item())
            elif isinstance(v, (float, int)):
                val = float(v)
            else:
                continue
            self.totals[k] = self.totals.get(k, 0.0) + val

    def averages(self) -> dict[str, float]:
        c = max(self.count, 1)
        return {k: v / c for k, v in self.totals.items()}


class DiagnosticTracker:
    """Collects and computes batch-level regional statistics and solver energy traces."""

    def __init__(self, model: RMRv3) -> None:
        self.model = model
        self.mu_means: list[float] = []
        self.disp_means: list[float] = []
        self.all_disps: list[torch.Tensor] = []
        self.all_pred_weights: list[torch.Tensor] = []
        self.all_solver_weights: list[torch.Tensor] = []
        self.e_befores: list[float] = []
        self.e_afters: list[float] = []

        scale_sizes = tuple(model.cfg.region_sizes_px)
        self.scale_map: dict[int, int | str] = {
            sid: (f"{s[0]}_{s[1]}" if isinstance(s, (tuple, list)) else int(s))
            for sid, s in enumerate(scale_sizes)
        }
        self.w_scales: dict[int | str, list[float]] = {s: [] for s in self.scale_map.values()}
        self.pi_scales: dict[int | str, list[float]] = {s: [] for s in self.scale_map.values()}

    @property
    def w_scale_32(self) -> list[float]:
        return self.w_scales.get(32, [])

    @property
    def w_scale_64(self) -> list[float]:
        return self.w_scales.get(64, [])

    @property
    def w_scale_128(self) -> list[float]:
        return self.w_scales.get(128, [])

    @property
    def w_scale_64_32(self) -> list[float]:
        return self.w_scales.get("64_32", [])

    @property
    def scale_pi_32(self) -> list[float]:
        return self.pi_scales.get(32, [])

    @property
    def scale_pi_64(self) -> list[float]:
        return self.pi_scales.get(64, [])

    @property
    def scale_pi_128(self) -> list[float]:
        return self.pi_scales.get(128, [])

    @property
    def scale_pi_64_32(self) -> list[float]:
        return self.pi_scales.get("64_32", [])

    @torch.no_grad()
    def update(self, outputs: dict[str, Any]) -> None:
        self.mu_means.append(float(outputs["b_region"].mean().item()))
        self.disp_means.append(float(outputs["region_dispersion"].mean().item()))
        self.all_disps.append(outputs["region_dispersion"].detach().cpu().float().flatten())
        self.all_pred_weights.append(outputs["region_weight"].detach().cpu().float().flatten())
        self.all_solver_weights.append(outputs["solver_region_weight"].detach().cpu().float().flatten())

        et = outputs.get("energy_trace", [])
        if et:
            self.e_befores.append(float(et[0]["before"].mean().item()))
            self.e_afters.append(float(et[-1]["after"].mean().item()))

        regions = outputs["regions"]
        w = outputs["solver_region_weight"]
        scale_w = outputs.get("scale_weights", None)

        for sid, s_val in self.scale_map.items():
            mask = regions.scale_id == sid
            if mask.any():
                self.w_scales[s_val].append(float(w[..., mask].mean().item()))
            if scale_w is not None and scale_w.shape[1] > sid:
                self.pi_scales[s_val].append(float(scale_w[:, sid].mean().item()))

    def summarize(self) -> dict[str, float]:
        disps_np = torch.cat(self.all_disps).cpu().numpy() if self.all_disps else np.array([50.0])
        pred_weights_np = torch.cat(self.all_pred_weights).cpu().numpy() if self.all_pred_weights else np.array([1.0])
        solver_weights_np = torch.cat(self.all_solver_weights).cpu().numpy() if self.all_solver_weights else np.array([1.0])

        e_b = float(np.mean(self.e_befores)) if self.e_befores else 0.0
        e_a = float(np.mean(self.e_afters)) if self.e_afters else 0.0
        e_red = (e_b - e_a) / max(e_b, 1e-8)

        w_min_val = self.model.cfg.reliability_weight_min
        w_max_val = self.model.cfg.reliability_weight_max
        w_low_frac = float(np.mean(pred_weights_np <= w_min_val + 1e-4))
        w_high_frac = float(np.mean(pred_weights_np >= w_max_val - 1e-4))

        return {
            "region_mu_mean": float(np.mean(self.mu_means)) if self.mu_means else 0.0,
            "region_dispersion_mean": float(np.mean(self.disp_means)) if self.disp_means else 50.0,
            "region_dispersion_p10": float(np.percentile(disps_np, 10)),
            "region_dispersion_p50": float(np.percentile(disps_np, 50)),
            "region_dispersion_p90": float(np.percentile(disps_np, 90)),
            "region_weight_mean": float(np.mean(pred_weights_np)),
            "region_weight_std": float(np.std(pred_weights_np)),
            "region_weight_min": float(np.min(pred_weights_np)),
            "region_weight_max": float(np.max(pred_weights_np)),
            "solver_weight_mean": float(np.mean(solver_weights_np)),
            "solver_weight_std": float(np.std(solver_weights_np)),
            "weight_clip_low_fraction": w_low_frac,
            "weight_clip_high_fraction": w_high_frac,
            "solver_energy_before": e_b,
            "solver_energy_after": e_a,
            "solver_energy_reduction": e_red,
            "weight_mean_16": float(np.mean(self.w_scales[16])) if self.w_scales.get(16) else 1.0,
            "weight_mean_32": float(np.mean(self.w_scales[32])) if self.w_scales.get(32) else 1.0,
            "weight_mean_64": float(np.mean(self.w_scales[64])) if self.w_scales.get(64) else 1.0,
            "weight_mean_128": float(np.mean(self.w_scales[128])) if self.w_scales.get(128) else 1.0,
            "scale_pi_16": float(np.mean(self.pi_scales[16])) if self.pi_scales.get(16) else 0.0,
            "scale_pi_32": float(np.mean(self.pi_scales[32])) if self.pi_scales.get(32) else 0.0,
            "scale_pi_64": float(np.mean(self.pi_scales[64])) if self.pi_scales.get(64) else 0.0,
            "scale_pi_128": float(np.mean(self.pi_scales[128])) if self.pi_scales.get(128) else 0.0,
            "weight_mean_64_32": float(np.mean(self.w_scales["64_32"])) if self.w_scales.get("64_32") else 1.0,
            "scale_pi_64_32": float(np.mean(self.pi_scales["64_32"])) if self.pi_scales.get("64_32") else 0.0,
        }


def format_dynamic_training_banner(
    model: RMRv3,
    cfg: dict[str, Any],
    run_id: str | None,
    n_params: int,
    device: torch.device,
    epochs: int,
    bs: int,
    lr_init: float,
    out_dir: Path,
) -> str:
    """Format a dynamic, attribute-driven training banner without hardcoded version strings."""
    m_cfg = model.cfg

    # 1. Carrier & Backbone
    bb_name = m_cfg.backbone_name.split(".")[0]
    stride = m_cfg.output_stride
    feat_w = m_cfg.feature_width
    neck_desc = m_cfg.neck_type
    if m_cfg.use_coord_attn:
        neck_desc += "+CoordAttn"
    carrier_line = f"Stride-{stride} | Backbone: {bb_name} | Neck: {neck_desc} (C={feat_w})"

    # 2. Geometry & Spatial Pooling
    region_sizes = list(m_cfg.region_sizes_px)
    is_aniso = any(isinstance(s, (tuple, list)) and len(s) == 2 and s[0] != s[1] for s in region_sizes)
    geo_tag = "Anisotropic Perspective" if is_aniso else "Isotropic"

    reg_strs = []
    for s in region_sizes:
        if isinstance(s, (tuple, list)):
            reg_strs.append(f"({s[0]}x{s[1]})")
        else:
            reg_strs.append(str(s))
    overlap = m_cfg.region_overlap
    reg_desc = f"[{', '.join(reg_strs)}] px ({geo_tag}, overlap={overlap:.0%})"

    pool_mode = m_cfg.regional_feature_stats
    if pool_mode == "mean_std":
        pooling_desc = "Spatial Moments (Mean+Std)"
    elif m_cfg.native_scale_pooling:
        pooling_desc = "Native Multiscale Pooling"
    else:
        pooling_desc = "Spatial Average"

    # 3. Inverse Solver Engine
    if not m_cfg.enable_solver:
        solver_desc = "Disabled (Direct Feedforward Carrier Head)"
    else:
        solver_parts = []
        if m_cfg.proximal_tau > 0.0:
            p_mode = m_cfg.proximal_mode.upper()
            solver_parts.append(f"Proximal {p_mode} RW-SIRT (tau={m_cfg.proximal_tau}, mu={m_cfg.proximal_mu})")
        elif m_cfg.solver_mode == "multiplicative":
            solver_parts.append(f"Density-Gated RW-SIRT (rho={m_cfg.density_gate_rho})")
        else:
            solver_parts.append("Additive RW-SIRT")

        solver_parts.append(f"T={m_cfg.iterations}")
        solver_parts.append(f"omega={m_cfg.omega}")

        tv_lam = m_cfg.tv_lambda
        if tv_lam > 0.0:
            tv_t = m_cfg.tv_type.capitalize()
            tv_desc = f"{tv_t} TV (lambda={tv_lam})"
            if m_cfg.density_gated_diffusion:
                tv_desc += f" [Density-Gated tau={m_cfg.diffusion_dense_threshold}]"
            solver_parts.append(tv_desc)

        adj_m = m_cfg.adjoint_mode
        solver_parts.append(f"Adjoint={'Radon-Nikodym' if adj_m == 'radon_nikodym' else 'Flat'}")

        morozov_g = m_cfg.morozov_gamma
        if morozov_g > 0.0:
            solver_parts.append(f"Morozov(gamma={morozov_g})")

        rel_m = m_cfg.reliability_mode
        rel_tag = "SNR" if rel_m == "snr" else "NB-RateVar"
        w_mode = "Uniform (W=I)" if m_cfg.uniform_reliability else f"Weighted({rel_tag})"
        solver_parts.append(f"Weighting={w_mode}")
        if m_cfg.factorized_scale_routing:
            s_num = m_cfg.num_marginal_scales
            a_num = m_cfg.num_aspect_ratios
            solver_parts.append(f"ScaleRouting=Factorized2D(S={s_num}, A={a_num})")
        elif m_cfg.dynamic_scale_routing:
            k_num = len(m_cfg.region_sizes_px)
            solver_parts.append(f"ScaleRouting=Dynamic(K={k_num})")
        else:
            solver_parts.append("ScaleRouting=Isotropic")
        if m_cfg.trust_region_kappa > 0.0:
            solver_parts.append(f"TrustRegion(kappa={m_cfg.trust_region_kappa})")
        solver_desc = " | ".join(solver_parts)

    # 4. Heads & Densities
    guidance_head = "Hurdle-NB (Occupancy Gated)" if m_cfg.hurdle_head else "Negative-Binomial"
    if m_cfg.scale_conditioned_fine_head:
        density_head = "ScaleConditionedFineHead (FiLM Continuous Simplex + 3x3 DW Receptive Field)"
    elif m_cfg.temp_softplus:
        density_head = "FineMeasureHead (Temperature-Calibrated Softplus [tau*softplus(z/tau)])"
    else:
        density_head = "FineMeasureHead (Calibrated Log-Space Softplus)"
    if m_cfg.gated_density_curvature:
        density_head += " + GatedCurvature"
    elif m_cfg.density_curvature:
        density_head += " + QuadraticCurvature"
    if m_cfg.foreground_gate:
        density_head += " + FG-Gate(33p)"

    # 5. Supervision Target & Loss
    dm_target = cfg.get("loss", {}).get("dm_target", "y0")
    if dm_target == "dual":
        target_desc = "Dual-Depth [0.5 Y0 + 0.5 Y] (Anchored Guidance + End-to-End Solver)"
    elif dm_target == "y":
        target_desc = "Post-Solver Y (End-to-End Measure Optimization)"
    else:
        target_desc = "Pre-Solver Y0 (Decoupled Carrier Guidance)"
    loss_type = cfg.get("loss", {}).get("allocation_loss_type", "flat_dm16")
    supervision_desc = f"{target_desc} -> {loss_type.upper()}"

    loss_cfg_dict = cfg.get("loss", {})
    extra_loss_tags = []
    if loss_cfg_dict.get("lambda_scale_align", 0.0) > 0.0:
        extra_loss_tags.append(f"ScaleAln({loss_cfg_dict.get('lambda_scale_align')})")
    if loss_cfg_dict.get("lambda_curvature", 0.0) > 0.0:
        extra_loss_tags.append(f"Curv({loss_cfg_dict.get('lambda_curvature')})")
    if loss_cfg_dict.get("lambda_hard_bg", 0.0) > 0.0:
        extra_loss_tags.append(f"HardBG({loss_cfg_dict.get('lambda_hard_bg')})")
    if loss_cfg_dict.get("lambda_fg_gate", 0.0) > 0.0:
        extra_loss_tags.append(f"FGBce({loss_cfg_dict.get('lambda_fg_gate')})")
    if extra_loss_tags:
        supervision_desc += f" + [{', '.join(extra_loss_tags)}]"

    budget_pct = (n_params / 105_000) * 100
    headroom = 105_000 - n_params
    run_label = run_id if run_id else Path(cfg.get("output_dir", "unspecified")).name

    banner = [
        "=" * 80,
        f"  RMR Training Initialized [Dynamic Architecture Engine]",
        f"  Run: {run_label} | Parameters: {n_params:,} / 105,000 budget ({budget_pct:.1f}% used, +{headroom:,} headroom)",
        f"  Carrier: {carrier_line}",
        f"  Geometry: {reg_desc} | Pooling: {pooling_desc}",
        f"  Solver: {solver_desc}",
        f"  Heads: Regional={guidance_head} | Density={density_head}",
        f"  Supervision: {supervision_desc}",
        f"  Training: Epochs: {epochs} | Batch Size: {bs} | Initial LR: {lr_init:.2e} | Device: {device}",
        f"  Output Directory: {out_dir}",
        "=" * 80,
    ]
    return "\n".join(banner)
