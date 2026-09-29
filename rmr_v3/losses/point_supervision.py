from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def bayesian_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    sigma: float = 8.0,
    background_ratio: float = 0.1,
    stride: int = 4,
) -> torch.Tensor:
    """Bayesian Loss for point supervision (Ma et al. ICCV 2019).
    
    Computes continuous spatial allocation loss without artificial block boundaries.
    Memory-optimized: evaluates coordinate likelihoods under torch.no_grad() and
    avoids [N, M, 2] intermediate tensor duplication. Always returns float32.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    # Create grid of center coordinates in image pixels
    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]

    losses: list[torch.Tensor] = []
    tau = float(background_ratio)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()  # [M]
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        n = pts.shape[0]

        # Compute posterior probabilities under torch.no_grad() without autograd overhead
        with torch.no_grad():
            dx = pts[:, 0:1] - grid_xy[:, 0].unsqueeze(0)  # [N, M]
            dy = pts[:, 1:2] - grid_xy[:, 1].unsqueeze(0)  # [N, M]
            dist_sq = dx.square().add_(dy.square())  # [N, M]

            # Gaussian likelihood
            p_y_given_x = torch.exp(-dist_sq * inv_two_sigma_sq)  # [N, M]

            # Denominator with background likelihood
            denom = p_y_given_x.sum(dim=0, keepdim=True).add_(tau).clamp_min_(1e-8)  # [1, M]

            # Posterior probability
            post_person = p_y_given_x / denom  # [N, M]
            post_bg = (tau / denom).squeeze(0)  # [M]

        # Predicted count assigned to each person and background
        c_hat_person = torch.matmul(post_person, y_flat)  # [N]
        c_hat_bg = torch.dot(post_bg, y_flat)  # scalar

        person_err = torch.abs(c_hat_person - 1.0).sum()
        bg_err = c_hat_bg

        sample_loss = (person_err + bg_err) / float(max(n, 1))
        losses.append(sample_loss)

    # Always return float32 for AMP gradient scaler stability
    return torch.stack(losses).mean().float()


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
    Always returns float32.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2]

    losses: list[torch.Tensor] = []
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

        with torch.no_grad():
            dx = pts[:, 0:1] - grid_xy[:, 0].unsqueeze(0)
            dy = pts[:, 1:2] - grid_xy[:, 1].unsqueeze(0)
            diag = math.sqrt(float(h * stride) ** 2 + float(w * stride) ** 2)
            c_mat = torch.sqrt(dx.square().add_(dy.square())) / max(diag, 1.0)  # [N, M]

            mu = torch.ones(n, device=device, dtype=torch.float32) / float(n)  # [N]
            nu = (y_flat.detach() / total_pred.detach().clamp_min(eps)).clamp_min(eps)  # [M]
            nu = nu / nu.sum().clamp_min(eps)

            u = torch.zeros(n, device=device, dtype=torch.float32)
            v = torch.zeros(grid_xy.shape[0], device=device, dtype=torch.float32)
            inv_reg = 1.0 / max(float(reg), 1e-4)

            for _ in range(num_iters):
                u = torch.log(mu) - torch.logsumexp((-c_mat * inv_reg) + v.unsqueeze(0), dim=1)
                v = torch.log(nu) - torch.logsumexp((-c_mat * inv_reg) + u.unsqueeze(1), dim=0)

            log_p = (-c_mat * inv_reg) + u.unsqueeze(1) + v.unsqueeze(0)
            p_mat = torch.exp(torch.clamp(log_p, max=50.0))  # [N, M]

        # Monge-Kantorovich transport cost against continuous prediction
        p_weights = p_mat.sum(dim=0)  # [M]
        ot_cost = torch.dot(p_weights, y_flat)
        mass_penalty = torch.abs(total_pred - float(n)) / float(max(n, 1))

        sample_loss = ot_cost + 0.1 * mass_penalty
        losses.append(sample_loss)

    # Always return float32 for AMP gradient scaler stability
    return torch.stack(losses).mean().float()
