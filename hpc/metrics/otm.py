"""OT-M: Optimal Transport-based Point Localization Decoder (CVPR 2023).

Given a continuous positive count-mass map D at stride-4:
  1. Set point cardinality m = round(sum(D)).
  2. Optimize point locations to best approximate mass distribution D via log-domain Sinkhorn OT.
  3. Guarantees exact cardinality consistency: len(P_pred) == round(N_pred).
  4. 0-parameter overhead.
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F


def sinkhorn_log(
    a: torch.Tensor,
    b: torch.Tensor,
    cost: torch.Tensor,
    epsilon: float = 0.02,
    iterations: int = 50,
) -> torch.Tensor:
    """Balanced entropy-regularized Optimal Transport in log-domain.
    
    Args:
        a: [N] source mass vector (must sum to same total as b).
        b: [M] target mass vector.
        cost: [N, M] pairwise cost matrix.
        epsilon: entropy regularization coefficient.
        iterations: number of Sinkhorn-Knopp iterations.
        
    Returns:
        P: [N, M] optimal transport plan matrix.
    """
    a = a.float().clamp_min(1e-12)
    b = b.float().clamp_min(1e-12)
    cost = cost.float()

    if not torch.allclose(a.sum(), b.sum(), atol=1e-3, rtol=1e-3):
        raise ValueError(f"Unbalanced OT: sum(a)={a.sum():.4f} != sum(b)={b.sum():.4f}")

    log_a = torch.log(a)
    log_b = torch.log(b)
    log_K = -cost / float(epsilon)

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)

    for _ in range(iterations):
        log_u = log_a - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_b - torch.logsumexp(log_K + log_u[:, None], dim=0)

    log_P = log_u[:, None] + log_K + log_v[None, :]
    return torch.exp(log_P)


def mass_cell_coordinates(Hm: int, Wm: int, device: torch.device) -> torch.Tensor:
    """Compute normalized center coordinates [0, 1]^2 for all grid mass cells.
    
    Returns:
        (Hm*Wm, 2) tensor of (x, y) normalized coordinates.
    """
    yy, xx = torch.meshgrid(
        torch.arange(Hm, device=device, dtype=torch.float32),
        torch.arange(Wm, device=device, dtype=torch.float32),
        indexing="ij",
    )
    # Centers of grid cells
    x = (xx + 0.5) / float(Wm)
    y = (yy + 0.5) / float(Hm)
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def initialize_ot_points(
    mass_flat: torch.Tensor,
    source_xy: torch.Tensor,
    m: int,
    Hm: int,
    Wm: int,
) -> torch.Tensor:
    """Adaptive point initialization via mass density multinomial sampling."""
    prob = mass_flat / mass_flat.sum().clamp_min(1e-12)
    replacement = (m > mass_flat.numel())

    idx = torch.multinomial(prob, num_samples=m, replacement=replacement)
    points = source_xy[idx].clone()

    if replacement:
        jitter = torch.empty_like(points).uniform_(-0.5, 0.5)
        jitter[:, 0] /= float(Wm)
        jitter[:, 1] /= float(Hm)
        points = (points + jitter).clamp(0.0, 1.0)

    return points


@torch.no_grad()
def otm_localize(
    mass: torch.Tensor,
    output_stride: int = 4,
    outer_iterations: int = 8,
    sinkhorn_iterations: int = 50,
    epsilon: float = 0.02,
    mean_stop_px: float = 1.0,
    max_stop_px: float = 4.0,
) -> torch.Tensor:
    """Parameter-free OT-M point localization from continuous mass map D.
    
    Args:
        mass: [1, Hm, Wm] or [Hm, Wm] positive mass map tensor.
        output_stride: stride factor relative to image pixels (default 4).
        outer_iterations: number of point position update steps (M-steps).
        sinkhorn_iterations: number of Sinkhorn balance steps per M-step.
        epsilon: Sinkhorn entropy regularization factor.
        mean_stop_px: early stopping threshold on mean pixel movement.
        max_stop_px: early stopping threshold on maximum pixel movement.
        
    Returns:
        points_xy: [m, 2] in original image pixel space, where m = round(sum(mass)).
    """
    if mass.ndim == 3:
        if mass.shape[0] != 1:
            raise ValueError(f"Expected single-channel mass map, got shape {tuple(mass.shape)}")
        mass = mass[0]

    mass = mass.float().clamp_min(0.0)
    Hm, Wm = mass.shape

    estimated_count = mass.sum().item()
    m = max(0, int(round(estimated_count)))

    if m == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=mass.device)

    source_xy = mass_cell_coordinates(Hm, Wm, device=mass.device)
    mass_flat = mass.reshape(-1).clamp_min(1e-12)

    # Balanced OT requires sum(a) == sum(b) == m
    a = (mass_flat / mass_flat.sum()) * float(m)
    b = torch.ones(m, dtype=torch.float32, device=mass.device)

    points = initialize_ot_points(mass_flat, source_xy, m, Hm, Wm)

    image_w = float(Wm * output_stride)
    image_h = float(Hm * output_stride)

    for _ in range(outer_iterations):
        previous = points.clone()

        # Squared Euclidean cost matrix [Hm*Wm, m]
        cost = torch.cdist(source_xy, points, p=2).square()

        P = sinkhorn_log(
            a=a,
            b=b,
            cost=cost,
            epsilon=epsilon,
            iterations=sinkhorn_iterations,
        )

        # M-step: update target point locations as weighted barycenters
        denom = P.sum(dim=0).clamp_min(1e-12)
        points = (P.T @ source_xy) / denom[:, None]

        # Movement in original image pixels
        dx = (points[:, 0] - previous[:, 0]) * image_w
        dy = (points[:, 1] - previous[:, 1]) * image_h
        movement = torch.sqrt(dx.square() + dy.square())

        if movement.mean().item() < mean_stop_px and movement.max().item() < max_stop_px:
            break

    points_px = points.clone()
    points_px[:, 0] *= image_w
    points_px[:, 1] *= image_h

    return points_px


@torch.no_grad()
def infer_count_and_localization(
    model: torch.nn.Module,
    image: torch.Tensor,
    output_stride: int = 4,
    pad_multiple: int = 32,
) -> Dict[str, Union[torch.Tensor, float]]:
    """Single-image joint counting & OT-M localization inference."""
    model.eval()
    if image.ndim == 3:
        image = image.unsqueeze(0)

    count_tensor, valid_mass = model.predict(image, pad_multiple=pad_multiple)
    points = otm_localize(valid_mass[0, 0], output_stride=output_stride)

    return {
        "mass": valid_mass,
        "count": float(count_tensor[0].item()),
        "points": points,
    }
