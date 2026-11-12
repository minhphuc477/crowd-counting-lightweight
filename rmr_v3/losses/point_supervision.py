from __future__ import annotations

import math
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F


from .bayesian_fast import (
    _BAYESIAN_FAST_ELEMS_CAP,
    _BayesianPersonErrorFunction,
    fast_bayesian_sample,
)

_BAYESIAN_GRID_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, float]] = {}


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
    sigma_min: float = 2.0,
    sigma_max: float = 16.0,
) -> torch.Tensor:
    """Canonical Bayesian Loss for point supervision (Ma et al. ICCV 2019).
    
    Computes continuous spatial allocation loss with Gaussian coordinate posteriors.
    Supports official dynamic virtual background distance and flexible normalization modes.
    Always returns float32.
    """
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    # Create grid of center coordinates in image pixels with LRU cache (including centered GEMM coords)
    grid_key = (h, w, stride, str(device))
    if grid_key in _BAYESIAN_GRID_CACHE:
        gx, gy, g_xy_c, g_sq, cx, cy = _BAYESIAN_GRID_CACHE[grid_key]
    else:
        y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * float(stride)
        x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * float(stride)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        gx = grid_x.flatten().unsqueeze(0)  # [1, M]
        gy = grid_y.flatten().unsqueeze(0)  # [1, M]
        cx, cy = float(w * stride) * 0.5, float(h * stride) * 0.5
        g_xy_c = torch.stack([gx[0] - cx, gy[0] - cy], dim=1).contiguous()
        g_sq = g_xy_c.square().sum(dim=1).unsqueeze(0)
        if len(_BAYESIAN_GRID_CACHE) >= 16:
            _BAYESIAN_GRID_CACHE.pop(next(iter(_BAYESIAN_GRID_CACHE)))
        _BAYESIAN_GRID_CACHE[grid_key] = (gx, gy, g_xy_c, g_sq, cx, cy)
    m_total = gx.shape[1]

    # Bounded adaptive chunk size for ultra-dense scenes (<= 64MB temp chunk)
    chunk_size = max(64, min(256, int(8_388_608 // max(m_total, 1))))

    losses: list[torch.Tensor] = []
    inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma)

    for i in range(b):
        y_flat = prob_y0[i, 0].flatten().float()  # [M]
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32, non_blocking=True)
        n = pts.shape[0]

        has_custom_target = (
            targets_list is not None
            and i < len(targets_list)
            and targets_list[i] is not None
            and targets_list[i].shape[0] == n
        )
        target_person = (
            targets_list[i].to(device=device, dtype=torch.float32)
            if has_custom_target
            else torch.ones(n, device=device, dtype=torch.float32)
        )
        if n * m_total <= _BAYESIAN_FAST_ELEMS_CAP:
            raw_sample_loss = fast_bayesian_sample(
                y_flat=y_flat, pts=pts, g_xy_c=g_xy_c, g_sq=g_sq, cx=cx, cy=cy,
                h=h, w=w, stride=stride, sigma=sigma, background_ratio=background_ratio,
                canonical_background=canonical_background, adaptive_sigma=adaptive_sigma,
                sigma_min=sigma_min, sigma_max=sigma_max, target_person=target_person,
            )
            if norm_mode == "canonical":
                sample_loss = raw_sample_loss
            elif norm_mode == "square_root":
                sample_loss = raw_sample_loss / max(math.sqrt(float(max(n, 1)) / max(float(norm_ref), 1.0)), 1e-4)
            else:
                sample_loss = raw_sample_loss / float(max(n, 1))
            losses.append(sample_loss)
            continue

        # Compute adaptive per-point sigma if requested (prevents blur overlap in dense clusters
        # while expanding smoothly to sigma_max on sparse / single-person foreground scenes)
        sig_pts = None
        inv_sq_pts = None
        if adaptive_sigma:
            with torch.no_grad():
                if n >= 2:
                    dist_mat = torch.cdist(pts, pts)
                    k_eff = min(4, n)
                    knn_d = torch.topk(dist_mat, k=k_eff, largest=False).values[:, -1]
                    sig_pts = (knn_d * 0.5).clamp(float(sigma_min), float(sigma_max))
                else:
                    sig_pts = torch.full((1,), float(sigma_max), device=device, dtype=torch.float32)
                inv_sq_pts = (1.0 / (2.0 * sig_pts.square())).unsqueeze(1)

        # Compute exact log-domain max-shifted posterior probabilities under torch.no_grad()
        with torch.no_grad():
            m_fg = torch.full((m_total,), float("-inf"), device=device, dtype=torch.float32)
            l_fg = torch.zeros((m_total,), device=device, dtype=torch.float32)
            min_dist_sq = (
                torch.full((m_total,), float("inf"), device=device, dtype=torch.float32)
                if canonical_background
                else None
            )
            nearest_idx = (
                torch.zeros((m_total,), device=device, dtype=torch.long)
                if (canonical_background and sig_pts is not None)
                else None
            )
            for c_idx in range(0, n, chunk_size):
                p_chunk = pts[c_idx : c_idx + chunk_size]
                dx = p_chunk[:, 0:1] - gx
                dy = p_chunk[:, 1:2] - gy
                d2 = dx.square_().add_(dy.square_())
                if min_dist_sq is not None:
                    if nearest_idx is not None:
                        c_min, c_arg = d2.min(dim=0)
                        closer = c_min < min_dist_sq
                        min_dist_sq = torch.where(closer, c_min, min_dist_sq)
                        nearest_idx = torch.where(closer, c_arg + c_idx, nearest_idx)
                    else:
                        min_dist_sq = torch.minimum(min_dist_sq, d2.min(dim=0).values)
                inv_k = inv_sq_pts[c_idx : c_idx + chunk_size] if inv_sq_pts is not None else inv_two_sigma_sq
                scores = d2.mul_(-inv_k)
                chunk_max = scores.max(dim=0).values
                if c_idx == 0:
                    m_fg = chunk_max
                    l_fg = torch.exp(scores - m_fg.unsqueeze(0)).sum(dim=0)
                else:
                    new_max = torch.maximum(m_fg, chunk_max)
                    l_fg = (
                        l_fg * torch.exp(m_fg - new_max)
                        + torch.exp(scores - new_max.unsqueeze(0)).sum(dim=0)
                    )
                    m_fg = new_max

            if canonical_background and min_dist_sq is not None:
                st_size = float(min(h, w) * stride)
                bg_ratio_eff = max(float(background_ratio), 0.01)
                d_min = torch.sqrt(min_dist_sq.clamp_min(0.0))
                base_bg_dist = st_size * bg_ratio_eff
                if sig_pts is not None and nearest_idx is not None and inv_sq_pts is not None:
                    sig_near = sig_pts[nearest_idx]
                    inv_k_near = inv_sq_pts.squeeze(1)[nearest_idx]
                    d_bg_margin = torch.clamp_min(sig_near * 4.0, base_bg_dist)
                    d_bg = torch.relu(d_bg_margin - d_min)
                    s_bg = -d_bg.square() * inv_k_near
                else:
                    d_bg = torch.relu(base_bg_dist - d_min)
                    s_bg = -d_bg.square() * inv_two_sigma_sq
            else:
                tau = max(float(background_ratio), 1e-4)
                s_bg = torch.full((m_total,), math.log(tau), device=device, dtype=torch.float32)

            max_score = torch.maximum(m_fg, s_bg)
            exp_bg_shifted = torch.exp(s_bg - max_score)
            fg_sum_shifted = l_fg * torch.exp(m_fg - max_score)
            denom_shifted = exp_bg_shifted + fg_sum_shifted  # Guaranteed >= 1.0 algebraically
            post_bg = exp_bg_shifted / denom_shifted
            max_score_row = max_score.unsqueeze(0)

        c_hat_bg = torch.dot(post_bg, y_flat)

        u = (y_flat / denom_shifted).clamp(min=-1e4, max=1e4)
        has_custom_target = (
            targets_list is not None
            and i < len(targets_list)
            and targets_list[i] is not None
            and targets_list[i].shape[0] == n
        )
        if has_custom_target:
            target_person = targets_list[i].to(device=device, dtype=torch.float32)
        else:
            target_person = torch.ones(n, device=device, dtype=torch.float32)

        inv_k_arg = inv_sq_pts if inv_sq_pts is not None else inv_two_sigma_sq
        person_err = _BayesianPersonErrorFunction.apply(
            u, pts, gx, gy, inv_k_arg, target_person, chunk_size, max_score_row
        )

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

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean().float()


def sinkhorn_ot_loss(
    prob_y0: torch.Tensor,
    points_list: list[torch.Tensor] | None,
    reg: float = 10.0,
    num_iters: int = 20,
    stride: int = 4,
    ot_max_dim: int = 128,
) -> torch.Tensor:
    """Log-domain Sinkhorn Optimal Transport Loss with area-preserving pooling when > ot_max_dim."""
    b, _, h, w = prob_y0.shape
    device = prob_y0.device

    scale_f, work_stride, work_y0 = 1, stride, prob_y0
    if h > ot_max_dim or w > ot_max_dim:
        scale_f = max(h // ot_max_dim, w // ot_max_dim)
        work_y0 = F.avg_pool2d(prob_y0, kernel_size=scale_f, stride=scale_f) * float(scale_f ** 2)
        work_stride = stride * scale_f

    _, _, eff_h, eff_w = work_y0.shape
    y_coords = (torch.arange(eff_h, device=device, dtype=torch.float32) + 0.5) * float(work_stride)
    x_coords = (torch.arange(eff_w, device=device, dtype=torch.float32) + 0.5) * float(work_stride)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    grid_xy = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    m_cells = grid_xy.shape[0]

    losses: list[torch.Tensor] = []
    eps = 1e-8
    inv_reg = 1.0 / max(float(reg), 1e-4)
    diag = math.sqrt(float(h * stride) ** 2 + float(w * stride) ** 2)

    for i in range(b):
        y_flat = work_y0[i, 0].flatten().float()
        total_pred = y_flat.sum()

        pts = points_list[i] if points_list is not None and i < len(points_list) else None
        if pts is None or pts.numel() == 0:
            losses.append(total_pred)
            continue

        pts = pts.to(device=device, dtype=torch.float32, non_blocking=True)
        n = pts.shape[0]

        with torch.no_grad():
            dx = pts[:, 0:1] - grid_xy[:, 0].unsqueeze(0)
            dy = pts[:, 1:2] - grid_xy[:, 1].unsqueeze(0)
            c_mat = torch.sqrt(dx.square().add_(dy.square())) / max(diag, 1.0)
            del dx, dy
            c_scaled = c_mat.mul_(-inv_reg)

            mu = torch.full((n,), 1.0 / float(n), device=device, dtype=torch.float32)
            nu = (y_flat.detach() / total_pred.detach().clamp_min(eps)).clamp_min(eps)
            nu = nu / nu.sum().clamp_min(eps)

            log_mu = torch.log(mu)
            log_nu = torch.log(nu)
            u = torch.zeros(n, device=device, dtype=torch.float32)
            v = torch.zeros(m_cells, device=device, dtype=torch.float32)

            for _ in range(num_iters):
                u_prev, v_prev = u.clone(), v.clone()
                u = log_mu - torch.logsumexp(c_scaled + v.unsqueeze(0), dim=1)
                u = torch.nan_to_num(u, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
                v = log_nu - torch.logsumexp(c_scaled + u.unsqueeze(1), dim=0)
                v = torch.nan_to_num(v, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)
                if torch.max(torch.abs(u - u_prev)) < 1e-4 and torch.max(torch.abs(v - v_prev)) < 1e-4:
                    break

            log_p = c_scaled + u.unsqueeze(1) + v.unsqueeze(0)
            p_mat = torch.exp(torch.clamp(log_p, max=50.0))
            p_weights = p_mat.sum(dim=0)
            del c_scaled, log_p, p_mat

        ot_cost = torch.dot(p_weights, y_flat)
        mass_penalty = torch.abs(total_pred - float(n)) / float(max(n, 1))
        losses.append(ot_cost + 0.1 * mass_penalty)

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
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

        pts = pts.to(device=device, dtype=torch.float32, non_blocking=True)
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

    if not losses:
        return torch.zeros((), device=device, dtype=torch.float32)
    return torch.stack(losses).mean().float()
