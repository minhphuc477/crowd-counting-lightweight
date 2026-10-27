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


def _hann_ramp_1d(length: int, left_taper: int, right_taper: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct 1D raised-cosine taper window in (0, 1] for Partition-of-Unity tile blending."""
    w = torch.ones(length, device=device, dtype=dtype)
    if left_taper > 0 and length > 1:
        lt = min(left_taper, length)
        t = torch.arange(1, lt + 1, device=device, dtype=dtype) / float(lt + 1)
        w[:lt] = 0.5 * (1.0 - torch.cos(math.pi * t))
    if right_taper > 0 and length > 1:
        rt = min(right_taper, length)
        t = torch.arange(rt, 0, -1, device=device, dtype=dtype) / float(rt + 1)
        w[-rt:] = torch.minimum(w[-rt:], 0.5 * (1.0 - torch.cos(math.pi * t)))
    return w.clamp_min(1e-4)


@torch.no_grad()
def predict_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    tile_size: int = 512,
    halo: int = 0,
    forward_kwargs: dict[str, Any] | None = None,
    blend_mode: str = "core",
) -> torch.Tensor:
    """Core/halo or Hann-blended tiled prediction assembled with exact Partition-of-Unity.

    When blend_mode == 'core' (default):
        Core boundaries are aligned to output stride except the final image boundary.
        Halo affects context only; only the core prediction is written to the output.
    When blend_mode == 'hann' and halo > 0:
        Overlapping halo regions are smoothly blended with a 2D raised-cosine window
        normalized by exact Partition-of-Unity weights, eliminating tile seam steps on 4K/8K images.
    """
    forward_kwargs = forward_kwargs or {}
    was_training = model.training
    model.eval()
    try:
        _, h, w = image.shape
        model_cfg = getattr(model, "cfg", getattr(getattr(model, "module", None), "cfg", None))
        if getattr(model_cfg, "subpixel_stride2", False) or getattr(model_cfg, "subpixel_dm", False):
            s = 2
        elif getattr(model_cfg, "output_stride", None) is not None and getattr(model_cfg, "output_stride") in (1, 2, 4, 8):
            s = int(model_cfg.output_stride)
        else:
            s = output_stride
        tile_size = max(s, _aligned_floor(tile_size, s))
        halo = max(0, _aligned_floor(halo, s))
        gh, gw = math.ceil(h / s), math.ceil(w / s)
        use_hann = (str(blend_mode).lower() == "hann") and (halo > 0)
        canvas = image.new_zeros((1, gh, gw), dtype=torch.float32)
        weight_canvas = image.new_zeros((1, gh, gw), dtype=torch.float32) if use_hann else None

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
                y_patch = model(patch, **forward_kwargs)["y"][0].float()

                if use_hann and weight_canvas is not None:
                    gsy0, gsx0 = sy0 // s, sx0 // s
                    ph, pw = y_patch.shape[-2], y_patch.shape[-1]
                    gsy1, gsx1 = min(gh, gsy0 + ph), min(gw, gsx0 + pw)
                    ph_eff, pw_eff = gsy1 - gsy0, gsx1 - gsx0
                    wy = _hann_ramp_1d(ph_eff, (y0 - sy0) // s, (sy1 - y1) // s, y_patch.device, torch.float32)
                    wx = _hann_ramp_1d(pw_eff, (x0 - sx0) // s, (sx1 - x1) // s, y_patch.device, torch.float32)
                    win2d = (wy[:, None] * wx[None, :]).unsqueeze(0)
                    canvas[:, gsy0:gsy1, gsx0:gsx1] += win2d * y_patch[:, :ph_eff, :pw_eff]
                    weight_canvas[:, gsy0:gsy1, gsx0:gsx1] += win2d
                else:
                    gy0 = y0 // s
                    gx0 = x0 // s
                    gy1 = math.ceil(y1 / s)
                    gx1 = math.ceil(x1 / s)
                    ly0 = (y0 - sy0) // s
                    lx0 = (x0 - sx0) // s
                    hh = gy1 - gy0
                    ww = gx1 - gx0
                    canvas[:, gy0:gy1, gx0:gx1] = y_patch[:, ly0:ly0 + hh, lx0:lx0 + ww]

        if use_hann and weight_canvas is not None:
            canvas = canvas / weight_canvas.clamp_min(1e-6)
        return canvas.to(dtype=image.dtype)
    finally:
        if was_training:
            model.train()


@torch.no_grad()
def predict_multiscale_tta(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    tile_size: int = 512,
    halo: int = 128,
    scales: tuple[float, ...] = (1.0,),
    use_hflip: bool = True,
    forward_kwargs: dict[str, Any] | None = None,
    blend_mode: str = "core",
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
        model_cfg = getattr(model, "cfg", getattr(getattr(model, "module", None), "cfg", None))
        if getattr(model_cfg, "subpixel_stride2", False) or getattr(model_cfg, "subpixel_dm", False):
            s = 2
        elif getattr(model_cfg, "output_stride", None) is not None and getattr(model_cfg, "output_stride") in (1, 2, 4, 8):
            s = int(model_cfg.output_stride)
        else:
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
                output_stride=s,
                tile_size=tile_size,
                halo=halo,
                forward_kwargs=forward_kwargs,
                blend_mode=blend_mode,
            )

            if use_hflip:
                # 2. Forward horizontally flipped
                img_flip = torch.flip(img_scaled, dims=[-1])
                d_flip = predict_tiled(
                    model=model,
                    image=img_flip,
                    output_stride=s,
                    tile_size=tile_size,
                    halo=halo,
                    forward_kwargs=forward_kwargs,
                    blend_mode=blend_mode,
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
