from __future__ import annotations

from typing import Any
from rmr_v3.losses import RMRv3LossConfig

TRAIN_LOG_FIELDNAMES: list[str] = [
    "epoch", "lr_backbone", "lr_main", "solver_strength",
    "train_total", "train_count", "train_flat_dm16", "train_allocation",
    "train_dm16", "train_dm32", "train_dm64",
    "train_cell", "train_region_nb", "train_hurdle_bce", "train_trunc_nb",
    "train_curvature", "train_hard_bg", "train_fg_bce", "train_scale_align",
    "train_spectral", "train_spectral_dc", "train_spectral_ac",
    "train_cell_carrier", "train_cell_fine",
    "train_kd_total", "train_kd_spatial", "train_kd_count",
    "region_mu_mean", "region_dispersion_mean", "region_dispersion_p10", "region_dispersion_p50", "region_dispersion_p90",
    "region_weight_mean", "region_weight_std", "region_weight_min", "region_weight_max",
    "solver_weight_mean", "solver_weight_std",
    "weight_clip_low_fraction", "weight_clip_high_fraction",
    "solver_energy_before", "solver_energy_after", "solver_energy_reduction",
    "weight_mean_16", "weight_mean_32", "weight_mean_64", "weight_mean_64_32", "weight_mean_128",
    "weight_mean_band_0", "weight_mean_band_1", "weight_mean_band_2", "weight_mean_band_3", "weight_mean_band_4",
    "scale_pi_16", "scale_pi_32", "scale_pi_64", "scale_pi_64_32", "scale_pi_128",
    "scale_pi_band_0", "scale_pi_band_1", "scale_pi_band_2", "scale_pi_band_3", "scale_pi_band_4",
    "val_mae", "val_rmse", "val_nae", "val_bias",
    "val_game0", "val_game1", "val_game2", "val_game3",
    "val_mae_sparse", "val_mae_moderate", "val_mae_dense",
    "pearson_rate_var_error", "spearman_rate_var_error", "spearman_weight_error",
    "spearman_pred_weight_error",
    "spearman_rate_var_error_16", "spearman_rate_var_error_32", "spearman_rate_var_error_64", "spearman_rate_var_error_128",
    "spearman_rate_var_error_band_0", "spearman_rate_var_error_band_1", "spearman_rate_var_error_band_2",
    "spearman_rate_var_error_band_3", "spearman_rate_var_error_band_4",
    "mean_std_residual",
    "coverage_50", "coverage_80", "coverage_95",
    "calib_gap_50", "calib_gap_80", "calib_gap_95",
    "dispersion_sat_low_fraction", "dispersion_sat_high_fraction",
    "solver_help_fraction", "solver_harm_fraction", "energy_monotonic_fraction",
    "mae_reg_y0", "mae_reg_y1", "mae_reg_y2",
    "curvature_alpha", "effective_curvature",
]


def format_epoch_row(
    epoch: int,
    epochs: int,
    row_log: dict[str, Any],
    loss_avgs: dict[str, float],
    loss_cfg: RMRv3LossConfig,
    solver_strength: float,
) -> str:
    train_allocation = loss_avgs.get("allocation", 0.0)
    train_dm16 = loss_avgs.get("dm_16", train_allocation if "dm_32" not in loss_avgs else 0.0)

    if loss_cfg.use_multiscale_dm or loss_cfg.use_hierarchical_dm:
        alloc_repr = f"alloc: {train_allocation:.3f} (16:{train_dm16:.3f}, 32:{row_log['train_dm32']:.3f}, 64:{row_log['train_dm64']:.3f})"
    else:
        alloc_repr = f"alloc(dm16): {train_allocation:.3f}"

    hurdle_str = f" | h_bce: {loss_avgs.get('hurdle_bce', 0.0):.4f}" if loss_avgs.get("hurdle_bce", 0.0) > 0 else ""
    v11_str = ""
    if loss_avgs.get("curvature", 0.0) > 0:
        v11_str += f" | curv: {loss_avgs.get('curvature', 0.0):.4f}"
    if loss_avgs.get("hard_bg", 0.0) > 0:
        v11_str += f" | h_bg: {loss_avgs.get('hard_bg', 0.0):.4f}"
    if loss_avgs.get("fg_bce", 0.0) > 0:
        v11_str += f" | fg_bce: {loss_avgs.get('fg_bce', 0.0):.4f}"
    if loss_avgs.get("scale_align", 0.0) > 0:
        v11_str += f" | sc_aln: {loss_avgs.get('scale_align', 0.0):.4f}"
    if loss_avgs.get("cell_carrier", 0.0) > 0:
        v11_str += f" | c_car: {loss_avgs.get('cell_carrier', 0.0):.4f}"
    if loss_avgs.get("cell_fine", 0.0) > 0:
        v11_str += f" | c_fine: {loss_avgs.get('cell_fine', 0.0):.4f}"
    if loss_avgs.get("spectral", 0.0) > 0:
        v11_str += f" | spec: {loss_avgs.get('spectral', 0.0):.4f}"

    return (
        f"[{epoch+1:04d}/{epochs:04d}] "
        f"Loss: {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}{v11_str}] | "
        f"SolvStr: {solver_strength:.2f} | "
        f"Disp: {row_log['region_dispersion_p50']:.1f} | "
        f"W_pred: {row_log['region_weight_mean']:.2f} | W_solv: {row_log['solver_weight_mean']:.2f}"
    )


def format_eval_block(
    epoch: int,
    epochs: int,
    row_log: dict[str, Any],
    val_metrics: dict[str, Any],
    cur_mae: float,
    best_mae: float,
    status_tag: str,
    solver_strength: float,
    diag_summary: dict[str, Any],
    loss_avgs: dict[str, float],
    loss_cfg: RMRv3LossConfig,
) -> str:
    train_allocation = loss_avgs.get("allocation", 0.0)
    train_dm16 = loss_avgs.get("dm_16", train_allocation if "dm_32" not in loss_avgs else 0.0)

    if loss_cfg.use_multiscale_dm or loss_cfg.use_hierarchical_dm:
        alloc_repr = f"alloc: {train_allocation:.3f} (16:{train_dm16:.3f}, 32:{row_log['train_dm32']:.3f}, 64:{row_log['train_dm64']:.3f})"
    else:
        alloc_repr = f"alloc(dm16): {train_allocation:.3f}"

    hurdle_str = f" | h_bce: {loss_avgs.get('hurdle_bce', 0.0):.4f}" if loss_avgs.get("hurdle_bce", 0.0) > 0 else ""
    v11_str = ""
    if loss_avgs.get("curvature", 0.0) > 0:
        v11_str += f" | curv: {loss_avgs.get('curvature', 0.0):.4f}"
    if loss_avgs.get("hard_bg", 0.0) > 0:
        v11_str += f" | h_bg: {loss_avgs.get('hard_bg', 0.0):.4f}"
    if loss_avgs.get("fg_bce", 0.0) > 0:
        v11_str += f" | fg_bce: {loss_avgs.get('fg_bce', 0.0):.4f}"
    if loss_avgs.get("scale_align", 0.0) > 0:
        v11_str += f" | sc_aln: {loss_avgs.get('scale_align', 0.0):.4f}"
    if loss_avgs.get("spectral", 0.0) > 0:
        v11_str += f" | spec: {loss_avgs.get('spectral', 0.0):.4f}"

    sparse_s = f"Sparse(<=100): {val_metrics.get('mae_sparse', 0.0):.2f}" if "mae_sparse" in val_metrics else ""
    mod_s = f"Mod(101-500): {val_metrics.get('mae_moderate', 0.0):.2f}" if "mae_moderate" in val_metrics else ""
    dense_s = f"Dense(>500): {val_metrics.get('mae_dense', 0.0):.2f}" if "mae_dense" in val_metrics else ""
    strata_str = f" | {sparse_s} | {mod_s} | {dense_s}" if sparse_s else ""

    cov_50 = f"{val_metrics.get('coverage_50', 0.0)*100:.1f}%" if "coverage_50" in val_metrics else "N/A"
    cov_80 = f"{val_metrics.get('coverage_80', 0.0)*100:.1f}%" if "coverage_80" in val_metrics else "N/A"
    cov_95 = f"{val_metrics.get('coverage_95', 0.0)*100:.1f}%" if "coverage_95" in val_metrics else "N/A"
    num_samples = int(val_metrics.get("num_samples", 182))

    return (
        f"\n{'='*92}\n"
        f"  EPOCH [{epoch+1:04d}/{epochs:04d}] PERIODIC EVALUATION ({num_samples} test samples)\n"
        f"{'-'*92}\n"
        f"  Train Loss    : {row_log['train_total']:.4f} [cnt: {row_log['train_count']:.2f}, {alloc_repr}, cell: {row_log['train_cell']:.3f}, reg_nb: {row_log['train_region_nb']:.3f}{hurdle_str}{v11_str}]\n"
        f"  Solver / W    : Str: {solver_strength:.2f} | E_red: {diag_summary['solver_energy_reduction']*100:.1f}% | W_pred: {row_log['region_weight_mean']:.2f} (std: {row_log['region_weight_std']:.2f}) | W_solv: {row_log['solver_weight_mean']:.2f}\n"
        f"  Val Metrics   : MAE: {cur_mae:.2f} | RMSE: {float(val_metrics['RMSE']):.2f} | NAE: {float(val_metrics['NAE']):.3f} | Bias: {float(val_metrics['Bias']):+.2f}\n"
        f"  GAME Hierarchy: G0: {float(val_metrics['GAME0']):.2f} | G1: {float(val_metrics['GAME1']):.2f} | G2: {float(val_metrics['GAME2']):.2f} | G3: {float(val_metrics['GAME3']):.2f}{strata_str}\n"
        f"  Uncertainty   : Coverage: 50%={cov_50}, 80%={cov_80}, 95%={cov_95} | Median Disp: {row_log['region_dispersion_p50']:.1f}\n"
        f"  Checkpoint    : Current Val MAE: {cur_mae:.2f} | Best Val MAE: {best_mae:.2f}{status_tag}\n"
        f"{'='*92}\n"
    )
