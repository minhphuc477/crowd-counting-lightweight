"""Count-Preserving Heavy-Tailed Spectral Loss (Hypothesis H2).

Formulates a Fourier-domain loss for crowd counting density fields:
- At DC frequency (omega = 0): exactly corresponds to total spatial mass integral.
- At AC frequencies (omega > 0): applies a heavy-tailed decaying weight |omega|^-beta
  to emphasize cluster structures while mitigating phase-shift jitter and high-frequency noise.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spectral_weights(
    height: int,
    width_rfft: int,
    full_width: int,
    beta: float = 2.0,
    omega_0: float = 0.05,
    bandpass: bool = False,
    omega_low: float = 0.02,
    omega_high: float = 0.35,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute normalized frequency magnitude grid and decaying weights.

    Args:
        height: Spatial height H.
        width_rfft: RFFT frequency width W // 2 + 1.
        full_width: Original spatial width W.
        beta: Power decay exponent for heavy-tailed spectral weighting.
        omega_0: Base spatial frequency scale (default 0.05).
        bandpass: If True, uses resonant band-pass weighting preserving crowd-wave frequencies.
        omega_low: Low-frequency cutoff for bandpass window.
        omega_high: High-frequency cutoff for bandpass window.
        device: Target torch device.
        dtype: Output tensor dtype.

    Returns:
        Tensor of shape [1, 1, H, W_rfft] containing decaying weights w(omega).
    """
    # Normalized frequency coordinates in [-0.5, 0.5] for H, [0.0, 0.5] for W_rfft
    freq_y = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)  # [H]
    freq_x = torch.fft.rfftfreq(full_width, d=1.0, device=device, dtype=dtype)  # [W_rfft]

    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
    omega_mag = torch.sqrt(grid_y ** 2 + grid_x ** 2)  # [H, W_rfft]

    if bandpass:
        # Resonant crowd-wave bandpass: passes cluster and queue frequencies [omega_low, omega_high]
        high_cut = 1.0 / (1.0 + (omega_mag / float(omega_high)) ** float(beta))
        low_cut = (omega_mag / float(omega_low)) ** 2 / (1.0 + (omega_mag / float(omega_low)) ** 2)
        weights = high_cut * low_cut
    else:
        # Heavy-tailed decay: w(omega) = 1 / (1 + (omega / omega_0))^beta
        weights = 1.0 / (1.0 + (omega_mag / float(omega_0)) ** float(beta))

    # Exclude DC component (omega = 0, 0) from AC weight grid
    weights[0, 0] = 0.0
    return weights.unsqueeze(0).unsqueeze(0)


def count_preserving_spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 2.0,
    lambda_count: float = 1.0,
    lambda_spectral: float = 0.5,
    omega_0: float = 0.05,
    bandpass: bool = False,
    omega_low: float = 0.02,
    omega_high: float = 0.35,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute count-preserving heavy-tailed spectral loss between pred and target.

    Args:
        pred: Predicted density map [B, 1, H, W].
        target: Target density map [B, 1, H, W].
        beta: Heavy-tailed spectral decay exponent.
        lambda_count: Weight for explicit DC mass conservation term.
        lambda_spectral: Weight for weighted AC spectral discrepancy term.
        omega_0: Base spatial frequency scale (default 0.05).
        bandpass: If True, uses resonant band-pass weighting preserving crowd-wave frequencies.
        omega_low: Low-frequency cutoff for bandpass window.
        omega_high: High-frequency cutoff for bandpass window.
        eps: Small positive constant for numerical safety.

    Returns:
        total_loss: Scalar combined loss tensor.
        loss_dict: Dictionary containing 'spectral_ac' and 'spectral_dc' components.
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    b, c, h, w = pred.shape
    pred_f = pred.float()
    target_f = target.float()

    # 1. Real 2D Fast Fourier Transform
    fft_pred = torch.fft.rfft2(pred_f, norm="ortho")    # [B, C, H, W // 2 + 1] complex
    fft_target = torch.fft.rfft2(target_f, norm="ortho")

    # 2. DC component discrepancy (Total Mass Conservation)
    # Under ortho norm, DC = (1 / sqrt(H*W)) * sum(y)
    dc_pred = fft_pred[..., 0, 0].real
    dc_target = fft_target[..., 0, 0].real
    dc_loss = F.l1_loss(dc_pred, dc_target)

    # 3. AC Spectral Discrepancy with Heavy-Tailed or Resonant Weighting
    spec_weights = compute_spectral_weights(
        height=h,
        width_rfft=fft_pred.shape[-1],
        full_width=w,
        beta=beta,
        omega_0=omega_0,
        bandpass=bandpass,
        omega_low=omega_low,
        omega_high=omega_high,
        device=pred.device,
        dtype=torch.float32,
    )

    # Complex difference magnitude: |F(pred) - F(target)|
    diff_complex = fft_pred - fft_target
    diff_mag = torch.abs(diff_complex)  # [B, C, H, W_rfft]

    # Weighted AC loss (excluding DC since spec_weights[0, 0] == 0)
    weighted_diff = diff_mag * spec_weights
    ac_loss = weighted_diff.sum(dim=(-2, -1)).mean()

    total_loss = lambda_count * dc_loss + lambda_spectral * ac_loss

    return total_loss, {
        "spectral_total": total_loss,
        "spectral_dc": dc_loss.detach(),
        "spectral_ac": ac_loss.detach(),
    }


class CountPreservingSpectralLoss(nn.Module):
    """Module wrapper for CountPreservingSpectralLoss."""

    def __init__(
        self,
        beta: float = 2.0,
        lambda_count: float = 1.0,
        lambda_spectral: float = 0.5,
        omega_0: float = 0.05,
        bandpass: bool = False,
        omega_low: float = 0.02,
        omega_high: float = 0.35,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.lambda_count = lambda_count
        self.lambda_spectral = lambda_spectral
        self.omega_0 = omega_0
        self.bandpass = bandpass
        self.omega_low = omega_low
        self.omega_high = omega_high

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loss, _ = count_preserving_spectral_loss(
            pred=pred,
            target=target,
            beta=self.beta,
            lambda_count=self.lambda_count,
            lambda_spectral=self.lambda_spectral,
            omega_0=self.omega_0,
            bandpass=self.bandpass,
            omega_low=self.omega_low,
            omega_high=self.omega_high,
        )
        return loss
