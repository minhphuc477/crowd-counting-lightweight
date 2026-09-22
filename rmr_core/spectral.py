"""Count-Preserving Heavy-Tailed Spectral Loss (Hypothesis H2 & H7/H8).

Supports both:
- Toroidal 2D Fast Fourier Transform (FFT)
- Count-Preserving 2D Discrete Cosine Transform (DCT-II) with Neumann reflection boundary conditions,
  eliminating Gibbs edge ringing while strictly preserving spatial total mass at DC:
      C(0, 0) == N_total / sqrt(HW).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute orthonormal 1D Discrete Cosine Transform (DCT-II) via Makhoul's algorithm."""
    n = x.shape[dim]
    if n == 1:
        return x
    orig_dtype = x.dtype
    x_f = x.float() if orig_dtype in (torch.float16, torch.bfloat16) else x
    x_move = x_f.movedim(dim, -1)
    orig_shape = x_move.shape
    x_2d = x_move.contiguous().reshape(-1, n)

    idx = torch.empty(n, dtype=torch.long, device=x.device)
    n_even = (n + 1) // 2
    n_odd = n // 2
    idx[:n_even] = torch.arange(0, n, 2, device=x.device)
    idx[n_even:] = torch.arange(2 * n_odd - 1, 0, -2, device=x.device)
    v = x_2d[:, idx]

    V = torch.fft.fft(v, dim=-1)
    k = torch.arange(n, device=x.device, dtype=x_f.dtype)
    phase = torch.exp(-1j * float(np.pi) * k / (2.0 * float(n)))
    X = (V * phase).real

    scale = torch.full((n,), math.sqrt(2.0 / float(n)), device=x.device, dtype=x_f.dtype)
    scale[0] = math.sqrt(1.0 / float(n))
    res = (X * scale).to(dtype=orig_dtype)
    return res.reshape(orig_shape).movedim(-1, dim).contiguous()


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Compute orthonormal 2D Discrete Cosine Transform (DCT-II) over the last two dimensions."""
    return dct_1d(dct_1d(x, dim=-1), dim=-2)


def idct_1d(X: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute orthonormal 1D Inverse Discrete Cosine Transform (IDCT-II / DCT-III)."""
    n = X.shape[dim]
    if n == 1:
        return X
    orig_dtype = X.dtype
    X_f = X.float() if orig_dtype in (torch.float16, torch.bfloat16) else X
    k = torch.arange(n, device=X.device, dtype=X_f.dtype).unsqueeze(1)
    idx_n = torch.arange(n, device=X.device, dtype=X_f.dtype).unsqueeze(0)
    M = torch.cos(float(np.pi) * (2 * idx_n + 1) * k / (2.0 * float(n))) * math.sqrt(2.0 / float(n))
    M[0, :] = 1.0 / math.sqrt(float(n))
    X_move = X_f.movedim(dim, -1)
    orig_shape = X_move.shape
    X_2d = X_move.contiguous().reshape(-1, n)
    res = torch.matmul(X_2d, M)
    return res.reshape(orig_shape).movedim(-1, dim).contiguous().to(dtype=orig_dtype)


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Compute orthonormal 2D Inverse Discrete Cosine Transform (IDCT) over the last two dimensions."""
    return idct_1d(idct_1d(x, dim=-1), dim=-2)


def compute_dct_spectral_weights(
    height: int,
    width: int,
    beta: float = 2.0,
    omega_0: float = 0.05,
    bandpass: bool = False,
    omega_low: float = 0.02,
    omega_high: float = 0.35,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Precompute normalized 2D DCT-II frequency magnitude grid and decaying weights."""
    freq_y = torch.arange(height, device=device, dtype=dtype) / (2.0 * float(height))
    freq_x = torch.arange(width, device=device, dtype=dtype) / (2.0 * float(width))

    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
    omega_mag = torch.sqrt(grid_y.square() + grid_x.square())

    if bandpass:
        high_cut = 1.0 / (1.0 + (omega_mag / float(omega_high)) ** float(beta))
        low_cut = (omega_mag / float(omega_low)).square() / (1.0 + (omega_mag / float(omega_low)).square())
        weights = high_cut * low_cut
    else:
        weights = 1.0 / (1.0 + (omega_mag / float(omega_0)) ** float(beta))

    weights[0, 0] = 0.0
    return weights.unsqueeze(0).unsqueeze(0)


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
    """Precompute normalized frequency magnitude grid and decaying weights for RFFT2."""
    freq_y = torch.fft.fftfreq(height, d=1.0, device=device, dtype=dtype)
    freq_x = torch.fft.rfftfreq(full_width, d=1.0, device=device, dtype=dtype)

    grid_y, grid_x = torch.meshgrid(freq_y, freq_x, indexing="ij")
    omega_mag = torch.sqrt(grid_y.square() + grid_x.square())

    if bandpass:
        high_cut = 1.0 / (1.0 + (omega_mag / float(omega_high)) ** float(beta))
        low_cut = (omega_mag / float(omega_low)).square() / (1.0 + (omega_mag / float(omega_low)).square())
        weights = high_cut * low_cut
    else:
        weights = 1.0 / (1.0 + (omega_mag / float(omega_0)) ** float(beta))

    weights[0, 0] = 0.0
    return weights.unsqueeze(0).unsqueeze(0)


def count_preserving_dct2_loss(
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
    """Count-preserving 2D DCT-II spectral loss with even-symmetric Neumann boundary reflection."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    _, _, h, w = pred.shape
    pred_f = pred.float()
    target_f = target.float()

    # 1. Orthonormal 2D DCT-II
    dct_pred = dct_2d(pred_f)
    dct_target = dct_2d(target_f)

    # 2. DC Component Discrepancy: C(0, 0) == sum(y) / sqrt(H*W)
    dc_pred = dct_pred[..., 0, 0]
    dc_target = dct_target[..., 0, 0]
    dc_loss = F.l1_loss(dc_pred, dc_target)

    # 3. AC Spectral Discrepancy (pure real tensors, zero Gibbs edge ringing)
    spec_weights = compute_dct_spectral_weights(
        height=h,
        width=w,
        beta=beta,
        omega_0=omega_0,
        bandpass=bandpass,
        omega_low=omega_low,
        omega_high=omega_high,
        device=pred.device,
        dtype=torch.float32,
    )

    diff_mag = torch.abs(dct_pred - dct_target)
    weighted_diff = diff_mag * spec_weights
    spec_weight_sum = spec_weights.sum().clamp_min(eps)
    ac_loss = (weighted_diff.sum(dim=(-2, -1)) / spec_weight_sum * math.sqrt(h * w)).mean()

    total_loss = lambda_count * dc_loss + lambda_spectral * ac_loss

    return total_loss, {
        "spectral_total": total_loss,
        "spectral_dc": dc_loss.detach(),
        "spectral_ac": ac_loss.detach(),
    }


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
    transform: str = "fft",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute count-preserving spectral loss between pred and target (FFT or DCT-II)."""
    if transform == "dct":
        return count_preserving_dct2_loss(
            pred=pred,
            target=target,
            beta=beta,
            lambda_count=lambda_count,
            lambda_spectral=lambda_spectral,
            omega_0=omega_0,
            bandpass=bandpass,
            omega_low=omega_low,
            omega_high=omega_high,
            eps=eps,
        )

    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

    _, _, h, w = pred.shape
    pred_f = pred.float()
    target_f = target.float()

    fft_pred = torch.fft.rfft2(pred_f, norm="ortho")
    fft_target = torch.fft.rfft2(target_f, norm="ortho")

    dc_pred = fft_pred[..., 0, 0].real
    dc_target = fft_target[..., 0, 0].real
    dc_loss = F.l1_loss(dc_pred, dc_target)

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

    diff_complex = fft_pred - fft_target
    diff_mag = torch.abs(diff_complex)

    weighted_diff = diff_mag * spec_weights
    spec_weight_sum = spec_weights.sum().clamp_min(eps)
    ac_loss = (weighted_diff.sum(dim=(-2, -1)) / spec_weight_sum * math.sqrt(h * w)).mean()

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
        transform: str = "fft",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.lambda_count = lambda_count
        self.lambda_spectral = lambda_spectral
        self.omega_0 = omega_0
        self.bandpass = bandpass
        self.omega_low = omega_low
        self.omega_high = omega_high
        self.transform = transform

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
            transform=self.transform,
        )
        return loss
