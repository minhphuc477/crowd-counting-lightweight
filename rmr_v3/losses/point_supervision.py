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
    norm_mode: str = "canonical",
    norm_ref: float = 100.0,
    canonical_background: bool = True,
    targets_list: list[torch.Tensor] | None = None,
    adaptive_sigma: bool = False,
    sigma_min: float = 3.0,
    sigma_max: float = 8.0,
) -> torch.Tensor:
    """Canonical Bayesian Loss for point supervision (Ma et al. ICCV 2019).
    
    Computes continuous spatial allocation loss with Gaussian coordinate posteriors.
    Supports official dynamic virtual background distance and flexible normalization modes.
    Always returns float32.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    # Create grid of center coordinates in image pixels
    y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
    x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    gx = grid_x.flatten().unsqueeze(0)  # [1, M]
    gy = grid_y.flatten().unsqueeze(0)  # [1, M]
    m_total = gx.shape[1]

    losses: list[torch.Tensor] = []
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

        # Compute adaptive per-point sigma if requested (prevents blur overlap in dense clusters)
        inv_sq_pts = None
        if adaptive_sigma and n >= 4:
            with torch.no_grad():
                dist_mat = torch.cdist(pts, pts)
                knn_d = torch.topk(dist_mat, k=4, largest=False).values[:, -1]
                sig_pts = (knn_d * 0.5).clamp(float(sigma_min), float(sigma_max))
                inv_sq_pts = (1.0 / (2.0 * sig_pts.square())).unsqueeze(1)

        # Compute posterior probabilities under torch.no_grad()
        with torch.no_grad():
            points_sum = torch.zeros((m_total,), device=device, dtype=torch.float32)
            if canonical_background:
                st_size = float(min(h, w) * stride)
                min_dist_sq = torch.full((m_total,), float("inf"), device=device, dtype=torch.float32)
                for c_idx in range(0, n, 512):
                    p_chunk = pts[c_idx : c_idx + 512]
                    dx = p_chunk[:, 0:1] - gx
                    dy = p_chunk[:, 1:2] - gy
                    d2 = dx.square().add_(dy.square())
                    min_dist_sq = torch.minimum(min_dist_sq, d2.min(dim=0).values)
                    inv_k = inv_sq_pts[c_idx : c_idx + 512] if inv_sq_pts is not None else inv_two_sigma_sq
                    points_sum.add_(torch.exp(-d2 * inv_k).sum(dim=0))
                bg_dis_sq = ((st_size * float(background_ratio)) ** 2) / (min_dist_sq + 1e-5)
                s_bg = -bg_dis_sq * inv_two_sigma_sq
                exp_bg = torch.exp(s_bg)
            else:
                tau = float(background_ratio)
                exp_bg = torch.full((m_total,), tau, device=device, dtype=torch.float32)
                for c_idx in range(0, n, 512):
                    p_chunk = pts[c_idx : c_idx + 512]
                    dx = p_chunk[:, 0:1] - gx
                    dy = p_chunk[:, 1:2] - gy
                    d2 = dx.square().add_(dy.square())
                    inv_k = inv_sq_pts[c_idx : c_idx + 512] if inv_sq_pts is not None else inv_two_sigma_sq
                    points_sum.add_(torch.exp(-d2 * inv_k).sum(dim=0))

            denom = (exp_bg + points_sum).clamp_min_(1e-8)
            post_bg = exp_bg / denom

        c_hat_bg = torch.dot(post_bg, y_flat)

        u = y_flat / denom
        target_person = (
            targets_list[i].to(device=device, dtype=torch.float32)
            if targets_list is not None and i < len(targets_list) and targets_list[i] is not None and targets_list[i].shape[0] == n
            else torch.ones(n, device=device, dtype=torch.float32)
        )
        person_err = 0.0
        for c_idx in range(0, n, 512):
            p_chunk = pts[c_idx : c_idx + 512]
            inv_k = inv_sq_pts[c_idx : c_idx + 512] if inv_sq_pts is not None else inv_two_sigma_sq
            with torch.no_grad():
                dx = p_chunk[:, 0:1] - gx
                dy = p_chunk[:, 1:2] - gy
                d2 = dx.square().add_(dy.square())
                k_chunk = torch.exp(-d2 * inv_k)
            c_hat_chunk = torch.matmul(k_chunk, u)
            tgt_chunk = target_person[c_idx : c_idx + 512]
            person_err = person_err + torch.abs(c_hat_chunk - tgt_chunk).sum()

        bg_err = c_hat_bg
        raw_sample_loss = person_err + bg_err

        if norm_mode == "canonical":
            sample_loss = raw_sample_loss
        elif norm_mode == "square_root":
            scale_denom = math.sqrt(float(max(n, 1)) / max(float(norm_ref), 1.0))
            sample_loss = raw_sample_loss / max(scale_denom, 1e-4)
        else:
            sample_loss = raw_sample_loss / float(max(n, 1))

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
                u_prev, v_prev = u.clone(), v.clone()
                u = torch.log(mu) - torch.logsumexp((-c_mat * inv_reg) + v.unsqueeze(0), dim=1)
                u = torch.nan_to_num(u, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
                v = torch.log(nu) - torch.logsumexp((-c_mat * inv_reg) + u.unsqueeze(1), dim=0)
                v = torch.nan_to_num(v, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
                if torch.max(torch.abs(u - u_prev)) < 1e-4 and torch.max(torch.abs(v - v_prev)) < 1e-4:
                    break

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


def fidt_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    k: float = 6.0,
    stride: int = 4,
    loss_type: str = "smooth_l1",
    normalize_by_count: bool = True,
) -> torch.Tensor:
    """Focal Inverse Distance Transform (FIDT) Loss (Liang et al. TPAMI 2022).

    Constructs a bounded continuous representation F(x) = 1 / (1 + min_i ||x - x_i||^2 / k^2).
    When normalize_by_count=True, scales F(x) into an exact Radon probability density measure
    whose spatial integral equals N, ensuring zero conflict with count loss and unrolled SIRT iterates.
    Uses foreground-background balanced Smooth L1 loss to prevent gradient starvation over sparse pixels.
    Evaluated under torch.no_grad() for target generation; returns AMP-safe float32.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    y_coords = torch.arange(h, device=device, dtype=torch.float32) + 0.5
    x_coords = torch.arange(w, device=device, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [M, 2] in grid units
    x_norm_sq = (grid_xy ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
    m_total = grid_xy.shape[0]

    k_val = float(max(k, 0.1))
    inv_k_sq = 1.0 / (k_val * k_val)
    losses: list[torch.Tensor] = []

    for i in range(b):
        pred_map = prob_y0[i, 0].float()  # [H, W]
        pts = points_list[i] if points_list is not None and i < len(points_list) else None

        if pts is None or pts.numel() == 0:
            target_map = torch.zeros_like(pred_map)
            l_val = pred_map.sum() if normalize_by_count else F.smooth_l1_loss(pred_map, target_map)
            losses.append(l_val)
            continue

        pts = pts.to(device=device, dtype=torch.float32)
        # Convert points from pixel coordinates to feature grid coordinates
        pts_grid = pts / float(stride)
        n = pts_grid.shape[0]

        with torch.no_grad():
            y_norm_sq = (pts_grid ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
            min_dist_sq = torch.empty((m_total,), device=device, dtype=torch.float32)
            # cuBLAS GEMM distance expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x y^T
            # Processed in 4096-cell spatial blocks to minimize VRAM footprint to < 16MB
            block_sz = 4096
            for m_start in range(0, m_total, block_sz):
                m_end = min(m_start + block_sz, m_total)
                sub_x = grid_xy[m_start:m_end]
                sub_x_norm = x_norm_sq[m_start:m_end]
                d2_block = sub_x_norm + y_norm_sq.t() - 2.0 * torch.mm(sub_x, pts_grid.t())
                min_dist_sq[m_start:m_end] = d2_block.clamp_min(0.0).min(dim=-1).values

            target_flat = 1.0 / (1.0 + min_dist_sq * inv_k_sq)
            raw_field = target_flat.view(h, w)
            if normalize_by_count:
                count_tgt = float(max(n, 1))
                norm_flat = (target_flat / target_flat.sum().clamp_min(1e-6)) * count_tgt
                target_map = norm_flat.view(h, w)
            else:
                target_map = raw_field

        if loss_type == "mse":
            l_val = F.mse_loss(pred_map, target_map)
        else:
            pos = raw_field > 0.05
            neg = ~pos
            per = F.smooth_l1_loss(pred_map, target_map, beta=0.1, reduction="none")
            pos_loss = per[pos].mean() if pos.any() else per.new_tensor(0.0)
            neg_loss = per[neg].mean() if neg.any() else per.new_tensor(0.0)
            l_val = 0.5 * (pos_loss + neg_loss)
        losses.append(l_val)

    return torch.stack(losses).mean().float()
