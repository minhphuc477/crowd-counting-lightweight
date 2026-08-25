import torch
from typing import List, Tuple, Union, Optional, Dict
import numpy as np


def build_integer_block_counts(
    points: Union[torch.Tensor, np.ndarray, List[Tuple[float, float]]],
    crop_h: int,
    crop_w: int,
    block_size: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build exact integer block count map Y^(B) of shape (H_B, W_B) from GT points.
    
    Args:
        points: (N, 2) tensor or array of point coordinates (x, y) where x in [0, crop_w), y in [0, crop_h).
        crop_h: Crop height in pixels (must be divisible by block_size).
        crop_w: Crop width in pixels (must be divisible by block_size).
        block_size: Block size B in pixels (e.g. 16, 32, 64, 96).
        device: Target torch device.
        
    Returns:
        Y: Integer count tensor of shape (H // B, W // B) of dtype torch.int64 (or float32 for loss computation).
    """
    assert crop_h % block_size == 0, f"crop_h ({crop_h}) must be divisible by block_size ({block_size})"
    assert crop_w % block_size == 0, f"crop_w ({crop_w}) must be divisible by block_size ({block_size})"
    
    h_b = crop_h // block_size
    w_b = crop_w // block_size
    
    if points is None:
        return torch.zeros((h_b, w_b), dtype=torch.float32, device=device)

    if isinstance(points, np.ndarray):
        pts = torch.from_numpy(points).float()
    elif isinstance(points, (list, tuple)):
        pts = torch.tensor(points, dtype=torch.float32)
    elif isinstance(points, torch.Tensor):
        pts = points.float()
    else:
        raise TypeError(f"Unsupported points type: {type(points)!r}")

    target_device = device if device is not None else pts.device
    if pts.numel() == 0:
        return torch.zeros((h_b, w_b), dtype=torch.float32, device=target_device)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {tuple(pts.shape)}")
    pts = pts.to(target_device)
    y_map = torch.zeros((h_b, w_b), dtype=torch.float32, device=target_device)
        
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    # Filter points within crop bounds
    valid = (xs >= 0) & (xs < crop_w) & (ys >= 0) & (ys < crop_h)
    if not valid.any():
        return y_map
        
    valid_xs = xs[valid]
    valid_ys = ys[valid]
    
    # Integer block indices: b_x = floor(x / B), b_y = floor(y / B)
    b_xs = torch.clamp((valid_xs / block_size).long(), 0, w_b - 1)
    b_ys = torch.clamp((valid_ys / block_size).long(), 0, h_b - 1)
    
    # Flattened linear index for deterministic accumulation
    flat_idx = b_ys * w_b + b_xs
    flat_map = y_map.view(-1)
    flat_map.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
    
    return y_map


def build_hierarchical_block_counts(
    points: Union[torch.Tensor, np.ndarray, List[Tuple[float, float]]],
    crop_h: int,
    crop_w: int,
    block_sizes: List[int],
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """Build exact integer block counts for all hierarchical block sizes."""
    return {
        b: build_integer_block_counts(points, crop_h, crop_w, b, device=device)
        for b in block_sizes
    }
