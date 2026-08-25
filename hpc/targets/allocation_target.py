import torch
from typing import List, Tuple, Union, Optional
import numpy as np


def build_block_constrained_allocation_target(
    points: Union[torch.Tensor, np.ndarray, List[Tuple[float, float]]],
    crop_h: int,
    crop_w: int,
    block_size: int = 16,
    output_stride: int = 4,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build block-constrained soft allocation target Z of shape (H/4, W/4).
    
    Each 16x16 pixel block corresponds to a (16/4)x(16/4) = 4x4 cell region.
    Points are bilinearly splatted strictly within their own 16x16 block cells.
    Any bilinear neighbors falling outside the 16x16 block are trimmed and the
    surviving within-block weights are renormalized so each point contributes exactly 1.0.
    
    This guarantees exact count conservation:
        sum_{k in block b} Z_{bk} == Y_b^{(16)}  (integer block count)
        sum_{all cells} Z == N (total point count)
        
    Args:
        points: (N, 2) tensor or array of point coordinates (x, y).
        crop_h: Height of crop in pixels (divisible by block_size).
        crop_w: Width of crop in pixels (divisible by block_size).
        block_size: Spatial block size B_A in pixels (default 16).
        output_stride: Stride of output mass map (default 4).
        device: Target torch device.
        
    Returns:
        Z: Soft allocation target map of shape (crop_h // output_stride, crop_w // output_stride).
    """
    assert crop_h % block_size == 0, f"crop_h ({crop_h}) must be divisible by {block_size}"
    assert crop_w % block_size == 0, f"crop_w ({crop_w}) must be divisible by {block_size}"
    assert block_size % output_stride == 0
    
    k_dim = block_size // output_stride  # 4
    out_h = crop_h // output_stride       # H/4
    out_w = crop_w // output_stride       # W/4
    
    if points is None:
        return torch.zeros((out_h, out_w), dtype=torch.float32, device=device)

    if isinstance(points, np.ndarray):
        pts = torch.from_numpy(points).float()
    elif isinstance(points, (list, tuple)):
        pts = torch.tensor(points, dtype=torch.float32)
    elif isinstance(points, torch.Tensor):
        pts = points.float()
    else:
        raise TypeError(f"Unsupported points type: {type(points)!r}")

    if pts.numel() == 0:
        target_device = device if device is not None else pts.device
        return torch.zeros((out_h, out_w), dtype=torch.float32, device=target_device)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must have shape (N,2), got {tuple(pts.shape)}")

    target_device = device if device is not None else pts.device
    pts = pts.to(target_device)
    z_map = torch.zeros((out_h, out_w), dtype=torch.float32, device=target_device)
        
    xs = pts[:, 0]
    ys = pts[:, 1]
    
    valid = (xs >= 0) & (xs < crop_w) & (ys >= 0) & (ys < crop_h)
    if not valid.any():
        return z_map
        
    valid_xs = xs[valid]
    valid_ys = ys[valid]
    
    # 1. Determine which 16x16 block each point belongs to
    bx = torch.clamp((valid_xs / block_size).long(), 0, (crop_w // block_size) - 1)
    by = torch.clamp((valid_ys / block_size).long(), 0, (crop_h // block_size) - 1)
    
    # 2. Continuous cell coordinates: u = x / 4 - 0.5, v = y / 4 - 0.5
    u = valid_xs / float(output_stride) - 0.5
    v = valid_ys / float(output_stride) - 0.5
    
    # 3. Local cell coordinates relative to the block's origin cell
    u_loc = u - (bx * k_dim).float()
    v_loc = v - (by * k_dim).float()
    
    # 4. Bilinear neighbor cell indices
    j0 = torch.floor(u_loc).long()
    j1 = j0 + 1
    i0 = torch.floor(v_loc).long()
    i1 = i0 + 1
    
    # Bilinear weights
    du = u_loc - j0.float()
    dv = v_loc - i0.float()
    
    w00 = (1.0 - du) * (1.0 - dv)
    w01 = du * (1.0 - dv)
    w10 = (1.0 - du) * dv
    w11 = du * dv
    
    # 5. Mask out neighbors that fall outside the local [0, k_dim-1] range
    valid00 = (j0 >= 0) & (j0 < k_dim) & (i0 >= 0) & (i0 < k_dim)
    valid01 = (j1 >= 0) & (j1 < k_dim) & (i0 >= 0) & (i0 < k_dim)
    valid10 = (j0 >= 0) & (j0 < k_dim) & (i1 >= 0) & (i1 < k_dim)
    valid11 = (j1 >= 0) & (j1 < k_dim) & (i1 >= 0) & (i1 < k_dim)
    
    w00 = torch.where(valid00, w00, torch.zeros_like(w00))
    w01 = torch.where(valid01, w01, torch.zeros_like(w01))
    w10 = torch.where(valid10, w10, torch.zeros_like(w10))
    w11 = torch.where(valid11, w11, torch.zeros_like(w11))
    
    # Sum of surviving weights
    w_sum = w00 + w01 + w10 + w11
    # Avoid div-by-zero (if all false, which shouldn't happen for valid points inside block)
    w_sum = torch.clamp_min(w_sum, 1e-7)
    
    w00 = w00 / w_sum
    w01 = w01 / w_sum
    w10 = w10 / w_sum
    w11 = w11 / w_sum
    
    # 6. Global cell indices and accumulate
    base_gx = bx * k_dim
    base_gy = by * k_dim
    
    flat_z = z_map.view(-1)
    
    # Accumulate valid 00
    m00 = valid00 & (w00 > 0)
    if m00.any():
        gx = base_gx[m00] + j0[m00]
        gy = base_gy[m00] + i0[m00]
        idx = gy * out_w + gx
        flat_z.index_add_(0, idx, w00[m00])
        
    # Accumulate valid 01
    m01 = valid01 & (w01 > 0)
    if m01.any():
        gx = base_gx[m01] + j1[m01]
        gy = base_gy[m01] + i0[m01]
        idx = gy * out_w + gx
        flat_z.index_add_(0, idx, w01[m01])
        
    # Accumulate valid 10
    m10 = valid10 & (w10 > 0)
    if m10.any():
        gx = base_gx[m10] + j0[m10]
        gy = base_gy[m10] + i1[m10]
        idx = gy * out_w + gx
        flat_z.index_add_(0, idx, w10[m10])
        
    # Accumulate valid 11
    m11 = valid11 & (w11 > 0)
    if m11.any():
        gx = base_gx[m11] + j1[m11]
        gy = base_gy[m11] + i1[m11]
        idx = gy * out_w + gx
        flat_z.index_add_(0, idx, w11[m11])
        
    return z_map
