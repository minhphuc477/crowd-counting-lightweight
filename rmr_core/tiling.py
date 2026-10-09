"""Tiled inference and Multi-Scale Test-Time Augmentation (TTA) for RMR Core.

Provides:
- Core/halo tiled prediction assembled without double-counting (predict_tiled).
- Multi-Scale TTA with Horizontal Flip and Partition-of-Unity Fusion (predict_multiscale_tta).
- Helper alignment utilities (_aligned_floor, _aligned_ceil).
"""
from __future__ import annotations

import math
from typing import Any

import torch

__all__ = [
    "predict_tiled",
    "predict_multiscale_tta",
]


def _aligned_floor(val: int, s: int) -> int:
    return (val // s) * s


def _aligned_ceil(val: int, s: int) -> int:
    return math.ceil(val / s) * s


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    tile_size: int = 512,
    halo: int = 0,
    forward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Core/halo tiled prediction assembled without double-counting.

    Core boundaries are aligned to output stride except the final image boundary.
    Halo affects context only; only the core prediction is written to the output.

    Args:
        model: Evaluated PyTorch model.
        image: Input tensor [C, H, W].
        output_stride: Stride of the predicted density grid (e.g. 4 or 2).
        tile_size: Size of the square core tile in image pixels.
        halo: Overlapping halo context size in image pixels.
        forward_kwargs: Optional kwargs for model forward pass.

    Returns:
        canvas: Reconstructed continuous Radon density tensor [1, gh, gw].
    """
    forward_kwargs = forward_kwargs or {}
    was_training = model.training
    model.eval()
    try:
        _, h, w = image.shape
        s = output_stride
        tile_size = max(s, _aligned_floor(tile_size, s))
        halo = max(0, _aligned_floor(halo, s))
        gh, gw = math.ceil(h / s), math.ceil(w / s)
        canvas = image.new_zeros((1, gh, gw))

        ys = list(range(0, h, tile_size))
        xs = list(range(0, w, tile_size))
        for y0 in ys:
            y1 = min(h, y0 + tile_size)
            for x0 in xs:
                x1 = min(w, x0 + tile_size)

                sy0 = max(0, _aligned_floor(y0 - halo, s))
                sx0 = max(0, _aligned_floor(x0 - halo, s))
                sy1 = min(h, _aligned_ceil(y1 + halo, s))
                sx1 = min(w, _aligned_ceil(x1 + halo, s))
                patch = image[:, sy0:sy1, sx0:sx1].unsqueeze(0)
                y_patch = model(patch, **forward_kwargs)["y"][0]

                gy0 = y0 // s
                gx0 = x0 // s
                gy1 = math.ceil(y1 / s)
                gx1 = math.ceil(x1 / s)
                ly0 = (y0 - sy0) // s
                lx0 = (x0 - sx0) // s
                hh = gy1 - gy0
                ww = gx1 - gx0
                canvas[:, gy0:gy1, gx0:gx1] = y_patch[:, ly0:ly0 + hh, lx0:lx0 + ww]
        return canvas
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def predict_multiscale_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    tile_size: int = 512,
    halo: int = 64,
    scales: tuple[float, ...] = (1.0,),
    use_hflip: bool = True,
    forward_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Multi-Scale Test-Time Augmentation (TTA) with Horizontal Flip and Partition-of-Unity Fusion.

    Computes:
        Y_final = sum_{s} w_s * 0.5 * (Y_{s, orig} + Flip_H(Y_{s, flip}))
    where Y_{s} is interpolated back to canonical feature grid shape (gh, gw).
    When scales=(1.0,) and use_hflip=False, identically matches standard predict_tiled.
    Adds exactly 0 trainable parameters while reducing test variance.

    Args:
        model: Evaluated PyTorch model.
        image: Input tensor [C, H, W].
        output_stride: Stride of the predicted density grid.
        tile_size: Size of the square core tile in image pixels.
        halo: Overlapping halo context size in image pixels.
        scales: Tuple of relative scaling factors (e.g. (0.8, 1.0, 1.2)).
        use_hflip: Whether to average horizontal mirror flips.
        forward_kwargs: Optional kwargs for model forward pass.

    Returns:
        accum_density: Fused multi-scale Radon measure tensor [1, gh, gw].
    """
    forward_kwargs = forward_kwargs or {}
    was_training = model.training
    model.eval()
    try:
        _, h, w = image.shape
        s = output_stride
        gh, gw = math.ceil(h / s), math.ceil(w / s)

        accum_density = image.new_zeros((1, gh, gw))
        total_weight = 0.0

        for scale in scales:
            scale_w = 1.0
            if abs(scale - 1.0) < 1e-4:
                img_scaled = image
            else:
                sh = max(s, _aligned_floor(int(round(h * scale)), s))
                sw = max(s, _aligned_floor(int(round(w * scale)), s))
                img_scaled = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(sh, sw),
                    mode="bilinear",
                    align_corners=False,
                )[0]

            # 1. Forward original
            d_orig = predict_tiled(
                model=model,
                image=img_scaled,
                output_stride=output_stride,
                tile_size=tile_size,
                halo=halo,
                forward_kwargs=forward_kwargs,
            )

            if use_hflip:
                # 2. Forward horizontally flipped
                img_flip = torch.flip(img_scaled, dims=[-1])
                d_flip = predict_tiled(
                    model=model,
                    image=img_flip,
                    output_stride=output_stride,
                    tile_size=tile_size,
                    halo=halo,
                    forward_kwargs=forward_kwargs,
                )
                d_unflip = torch.flip(d_flip, dims=[-1])
                d_fused = 0.5 * (d_orig + d_unflip)
            else:
                d_fused = d_orig

            # Resize density map back to target (gh, gw) while preserving total count (L1 mass)
            if d_fused.shape[-2:] != (gh, gw):
                orig_count = d_fused.sum()
                d_rescaled = torch.nn.functional.interpolate(
                    d_fused.unsqueeze(0),
                    size=(gh, gw),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                new_count = d_rescaled.sum().clamp_min(1e-8)
                d_fused = d_rescaled * (orig_count / new_count)

            accum_density += scale_w * d_fused
            total_weight += scale_w

        return accum_density / max(total_weight, 1e-8)
    finally:
        if was_training:
            model.train()
