from __future__ import annotations

import math
import torch


def bayesian_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    sigma: float = 8.0,
    background_ratio: float = 0.1,
    stride: int = 4,
) -> torch.Tensor:
    """Bayesian Loss for point supervision (Ma et al. ICCV 2019).
    
    Computes continuous spatial allocation loss without artificial block boundaries.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device
    dtype = prob_y0.dtype

    # Create grid of center coordinates in image pixels
    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]

    losses = []
    tau = float(background_ratio)

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()  # [M]
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n = pts.shape[0]

        # Pairwise squared distances: [N, M]
        diff = pts.unsqueeze(1) - grid_xy.unsqueeze(0)  # [N, M, 2]
        dist_sq = (diff ** 2).sum(dim=-1)  # [N, M]

        # Gaussian likelihood: [N, M]
        p_y_given_x = torch.exp(-dist_sq / (2.0 * sigma * sigma))

        # Denominator with background likelihood: [1, M]
        denom = p_y_given_x.sum(dim=0, keepdim=True) + tau

        # Posterior probability: [N, M] and [1, M]
        post_person = p_y_given_x / denom.clamp_min(1e-8)
        post_bg = tau / denom.clamp_min(1e-8)

        # Predicted count assigned to each person and background
        c_hat_person = torch.matmul(post_person, y_flat)  # [N]
        c_hat_bg = torch.matmul(post_bg, y_flat).squeeze(0)  # scalar

        person_err = torch.abs(c_hat_person - 1.0).sum()
        bg_err = c_hat_bg

        sample_loss = (person_err + bg_err) / float(max(n, 1))
        losses.append(sample_loss)

    return torch.stack(losses).mean().to(dtype)


def sinkhorn_ot_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    reg: float = 10.0,
    num_iters: int = 20,
    stride: int = 4,
) -> torch.Tensor:
    """Pure-PyTorch Log-Domain Sinkhorn Optimal Transport Loss.
    
    Measures Wasserstein transportation distance from normalized Y0 to ground truth points.
    Numerically stabilized with bounded dual variables and clamped log-domain scaling.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device
    dtype = prob_y0.dtype

    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]

    losses = []
    eps = 1e-8

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()  # [M]
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n = pts.shape[0]

        # Pairwise cost matrix (Euclidean distance in pixels normalized by image diagonal): [N, M]
        diff = pts.unsqueeze(1) - grid_xy.unsqueeze(0)  # [N, M, 2]
        diag = math.sqrt(float(h * stride) ** 2 + float(w * stride) ** 2)
        c_mat = torch.norm(diff, p=2, dim=-1) / max(diag, 1.0)  # [N, M]

        # Source and target marginal distributions
        mu = torch.ones(n, device=device, dtype=torch.float32) / float(n)  # [N]
        nu = (y_flat / total_pred.clamp_min(eps)).clamp_min(eps)  # [M]
        nu = nu / nu.sum()

        # Log-domain stabilized Sinkhorn iterations
        u = torch.zeros(n, device=device, dtype=torch.float32)
        v = torch.zeros(grid_xy.shape[0], device=device, dtype=torch.float32)
        inv_reg = 1.0 / max(float(reg), 1e-4)

        for _ in range(num_iters):
            # u = log(mu) - logsumexp(-C/reg + v)
            u = torch.log(mu) - torch.logsumexp((-c_mat * inv_reg) + v.unsqueeze(0), dim=1)
            # v = log(nu) - logsumexp(-C/reg + u)
            v = torch.log(nu) - torch.logsumexp((-c_mat * inv_reg) + u.unsqueeze(1), dim=0)

        # Transport plan: P = exp((-C + u + v) / reg)
        log_p = (-c_mat * inv_reg) + u.unsqueeze(1) + v.unsqueeze(0)
        p_mat = torch.exp(torch.clamp(log_p, max=50.0))

        ot_cost = (p_mat * c_mat).sum()
        mass_penalty = torch.abs(total_pred - float(n)) / float(max(n, 1))

        sample_loss = ot_cost + 0.1 * mass_penalty
        losses.append(sample_loss)

    return torch.stack(losses).mean().to(dtype)
