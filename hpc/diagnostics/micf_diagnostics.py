"""Diagnostic and Evaluation Utilities for MICF (Measure-Consistent Integral Count Fields).

Implements:
1. Count Readout & Consistency (Section 21):
   - N_corner = C_hat[H, W]
   - N_delta = sum(Delta_xy C_hat)
   - E_cons = |N_corner - N_delta|

2. Measure Diagnostics (Section 22):
   - f_- : negative-cell fraction (# {Y_hat < 0} / HW)
   - r_- : negative-mass ratio (sum([-Y_hat]_+) / (sum|Y_hat| + eps))
   - V   : violation magnitude (mean([-Y_hat]_+))

3. Multi-Scale Rectangle Evaluation (Section 3 & 23):
   - N(R) = C(y2, x2) - C(y1, x2) - C(y2, x1) + C(y1, x1) with zero-padded boundary
   - Evaluates count recovery error across normalized area bins {1/64, 1/16, 1/4, 1}

4. 2D Spectral Energy Analysis (Section 36):
   - E_high: fraction of 2D FFT energy above normalized spatial frequency threshold tau
   - Energy retention quantiles: fraction of Fourier coefficients needed to capture 90%, 95%, 99% energy
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from hpc.losses.micf import discrete_mixed_difference


def compute_measure_diagnostics(
    c_pred: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Compute mathematical measure diagnostics for a cumulative field C_hat.

    Args:
        c_pred: Predicted cumulative count field [B, 1, H, W] or [H, W].
        eps: Small epsilon to prevent division by zero.

    Returns:
        Dictionary of diagnostics:
        - 'negative_cell_fraction' (f_-): fraction of cells where Delta_xy C < 0
        - 'negative_mass_ratio' (r_-): ratio of negative mass to total absolute mass
        - 'violation_magnitude' (V): mean negative violation magnitude over all cells
        - 'corner_delta_count_gap' (E_cons): discrepancy between corner and sum-of-differences
        - 'n_corner': count read from bottom-right corner
        - 'n_delta': count read by summing reconstructed discrete cells
    """
    if c_pred.ndim == 2:
        c_pred = c_pred.unsqueeze(0).unsqueeze(0)
    elif c_pred.ndim == 3:
        c_pred = c_pred.unsqueeze(1)

    y_rec = discrete_mixed_difference(c_pred.float())  # [B, 1, H, W]
    neg_violations = F.relu(-y_rec)                    # [-Y]_+

    # f_-: fraction of cells with negative count
    f_minus = float((y_rec < 0).float().mean().item())

    # r_-: negative mass ratio = sum([-Y]_+) / (sum|Y| + eps)
    neg_mass_total = float(neg_violations.sum().item())
    abs_mass_total = float(y_rec.abs().sum().item()) + eps
    r_minus = float(neg_mass_total / abs_mass_total)

    # V: mean violation magnitude across all cells
    v_mag = float(neg_violations.mean().item())

    # Count consistency: N_corner vs N_delta
    n_corner = float(c_pred[:, :, -1, -1].sum().item())
    n_delta = float(y_rec.sum().item())
    e_cons = float(abs(n_corner - n_delta))

    return {
        "negative_cell_fraction": f_minus,
        "negative_mass_ratio": r_minus,
        "violation_magnitude": v_mag,
        "corner_delta_count_gap": e_cons,
        "n_corner": n_corner,
        "n_delta": n_delta,
    }


def query_rectangle_count(
    c: torch.Tensor,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> float:
    """Recover exact count in axis-aligned rectangle R = (x1, x2] x (y1, y2] from C.

    Uses 1-based indexing semantics with zero-padded boundary (Section 3):
        N(R) = C(y2, x2) - C(y1, x2) - C(y2, x1) + C(y1, x1)
    where C(0, x) = C(y, 0) = 0.

    Coordinates x1, y1, x2, y2 are grid indices in [0, W] and [0, H]:
    - x1=0, y1=0 corresponds to the virtual zero boundary (before first column/row).
    - x2=W, y2=H corresponds to the full image.
    """
    if c.ndim == 4:
        c = c.squeeze(0).squeeze(0)
    elif c.ndim == 3:
        c = c.squeeze(0)

    H, W = c.shape
    # Pad top and left by 1 with zeros: shape becomes [H+1, W+1]
    c_pad = F.pad(c.float(), (1, 0, 1, 0), mode="constant", value=0.0)

    # In c_pad, index k corresponds to original coordinate k-1
    val = (
        c_pad[y2, x2]
        - c_pad[y1, x2]
        - c_pad[y2, x1]
        + c_pad[y1, x1]
    )
    return float(val.item())


def evaluate_rectangle_counts(
    c_pred: torch.Tensor,
    c_target: torch.Tensor,
    scale_bins: Tuple[float, ...] = (1 / 64, 1 / 16, 1 / 4, 1.0),
    num_samples_per_bin: int = 40,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate rectangle count error across multiple spatial scale bins (Section 23).

    Tests whether MICF preserves region-level counting across scale hierarchies:
    - 1/64: fine local region
    - 1/16: intermediate region
    - 1/4:  quadrant-scale region
    - 1.0:  global count

    Args:
        c_pred: Predicted cumulative field [H, W] or [1, 1, H, W].
        c_target: Ground truth cumulative field [H, W] or [1, 1, H, W].
        scale_bins: Normalized area fractions to evaluate.
        num_samples_per_bin: Number of random rectangles sampled per scale bin.
        seed: Random seed for deterministic rectangle generation.

    Returns:
        Dictionary mapping bin labels (e.g. 'rectangle_mae_1_64', 'rectangle_mae_1_16', ...) to MAE.
    """
    rng = np.random.default_rng(seed)
    if c_pred.ndim == 4:
        c_pred = c_pred.squeeze(0).squeeze(0)
    if c_target.ndim == 4:
        c_target = c_target.squeeze(0).squeeze(0)

    H, W = c_pred.shape
    results: Dict[str, float] = {}

    bin_names = {
        1 / 64: "rectangle_mae_small",
        1 / 16: "rectangle_mae_medium",
        1 / 4:  "rectangle_mae_large",
        1.0:    "rectangle_mae_full",
    }

    for scale in scale_bins:
        bin_label = bin_names.get(scale, f"rectangle_mae_{scale:.4f}")
        errors: List[float] = []

        if abs(scale - 1.0) < 1e-6:
            # Full crop: exactly one query [0, 0] to [H, W]
            pred_val = query_rectangle_count(c_pred, 0, 0, W, H)
            tgt_val = query_rectangle_count(c_target, 0, 0, W, H)
            errors.append(abs(pred_val - tgt_val))
        else:
            # Target rectangle area in cells
            target_area = max(1.0, scale * H * W)
            aspect_ratio_range = (0.5, 2.0)

            for _ in range(num_samples_per_bin):
                # Generate random aspect ratio
                ar = float(rng.uniform(*aspect_ratio_range))
                # h * w = target_area, h / w = ar -> h^2 = ar * area
                rh = max(1, min(H, int(round(np.sqrt(target_area * ar)))))
                rw = max(1, min(W, int(round(target_area / rh))))

                # Sample top-left corner
                max_y1 = H - rh
                max_x1 = W - rw
                y1 = int(rng.integers(0, max_y1 + 1)) if max_y1 > 0 else 0
                x1 = int(rng.integers(0, max_x1 + 1)) if max_x1 > 0 else 0
                y2 = y1 + rh
                x2 = x1 + rw

                pred_val = query_rectangle_count(c_pred, x1, y1, x2, y2)
                tgt_val = query_rectangle_count(c_target, x1, y1, x2, y2)
                errors.append(abs(pred_val - tgt_val))

        results[bin_label] = float(np.mean(errors)) if errors else 0.0

    return results


def compute_spectral_analysis(
    map_2d: torch.Tensor,
    tau_norm: float = 0.5,
) -> Dict[str, float]:
    """Compute 2D Fourier spectral energy profile (Section 36).

    Measures:
    - High-frequency energy ratio E_high: proportion of energy at ||omega|| > tau
    - Energy retention quantiles: fraction of Fourier coefficients needed to capture 90%, 95%, 99% energy.

    Args:
        map_2d: 2D spatial field [H, W] (e.g. Y or C).
        tau_norm: Normalized frequency radius cutoff in [0, 1].

    Returns:
        Dictionary with spectral diagnostics:
        - 'high_freq_energy_ratio': E_high
        - 'coeff_frac_90': fraction of coefficients for 90% energy
        - 'coeff_frac_95': fraction of coefficients for 95% energy
        - 'coeff_frac_99': fraction of coefficients for 99% energy
    """
    if map_2d.ndim > 2:
        map_2d = map_2d.squeeze()
    H, W = map_2d.shape

    # 2D real FFT
    fft2 = torch.fft.rfft2(map_2d.float())
    power = (fft2.real ** 2 + fft2.imag ** 2)
    total_power = float(power.sum().item()) + 1e-12

    # Frequency coordinates normalized to [0, 1]
    fy = torch.fft.fftfreq(H, device=map_2d.device).unsqueeze(1).repeat(1, fft2.shape[1])
    fx = torch.fft.rfftfreq(W, device=map_2d.device).unsqueeze(0).repeat(H, 1)
    radius = torch.sqrt((fy * 2.0) ** 2 + (fx * 2.0) ** 2)  # [0, ~sqrt(2)]

    high_mask = radius > tau_norm
    high_power = float(power[high_mask].sum().item())
    e_high = high_power / total_power

    # Cumulative energy retention
    flat_power = power.flatten().cpu().numpy()
    sorted_power = np.sort(flat_power)[::-1]  # descending
    cum_energy = np.cumsum(sorted_power) / (np.sum(sorted_power) + 1e-12)
    total_coeffs = len(sorted_power)

    def _frac_at_thresh(th: float) -> float:
        idx = np.searchsorted(cum_energy, th)
        return float(min(idx + 1, total_coeffs) / total_coeffs)

    return {
        "high_freq_energy_ratio": float(e_high),
        "coeff_frac_90": _frac_at_thresh(0.90),
        "coeff_frac_95": _frac_at_thresh(0.95),
        "coeff_frac_99": _frac_at_thresh(0.99),
    }
