"""Bounded Point-Neighbor Mass Decomposition Loss (Bias-Safe Person-Level Supervision).

Fixes the background inflation issue of unconstrained Voronoi partitioning:
1. Each person p_i has an adaptive radius:
       R_i = clamp(0.8 * d_NN(p_i), min_radius_px, max_radius_px)
2. Pixels within R_{i^*(k)} contribute to person i's mass:
       m_i = sum_{k in V_i, dist(k, p_i) <= R_i} D_k
3. Pixels outside all head radii belong to Background V_0 and are directly penalized:
       L_bg = sum_{k in V_0} D_k / (|V_0| + 1)
4. Per-person unit mass conservation:
       L_unit = (1/N) * sum_{i=1}^N |m_i - 1.0|

Total Point Loss = L_unit + bg_weight * L_bg.
This prevents background pixels in sparse regions from being inflated to satisfy unit mass.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundedPointMassLoss(nn.Module):
    """Bounded per-person unit mass conservation with strict background isolation.

    Args:
        output_stride: Stride of density map relative to input image (default: 4).
        min_radius_px: Minimum head support radius in input pixels (default: 8.0).
        max_radius_px: Maximum head support radius in input pixels (default: 24.0).
        bg_weight: Penalty weight for mass leaking into background (default: 1.0).
    """

    def __init__(
        self,
        output_stride: int = 4,
        min_radius_px: float = 8.0,
        max_radius_px: float = 24.0,
        bg_weight: float = 1.0,
    ):
        super().__init__()
        self.output_stride = int(output_stride)
        self.min_radius_cells = float(min_radius_px) / float(output_stride)
        self.max_radius_cells = float(max_radius_px) / float(output_stride)
        self.bg_weight = float(bg_weight)

    def forward(
        self,
        d_map: torch.Tensor,
        points_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            d_map: (B, 1, H4, W4) non-negative density map.
            points_list: List of B tensors, each (N_b, 2) with (x, y) coordinates in input crop space.

        Returns:
            total_loss: scalar tensor.
            loss_dict: dictionary with individual terms.
        """
        B, _, H4, W4 = d_map.shape
        device = d_map.device

        # Stride-4 cell grid coordinates: (K, 2) where K = H4 * W4
        y_coords = torch.arange(H4, dtype=torch.float32, device=device) + 0.5
        x_coords = torch.arange(W4, dtype=torch.float32, device=device) + 0.5
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (K, 2)
        K = H4 * W4

        sample_unit_losses = []
        sample_bg_losses = []

        for b in range(B):
            d_flat = d_map[b, 0].reshape(-1)  # (K,)
            pts = points_list[b]

            if pts is None or pts.numel() == 0:
                # No points in crop -> all mass is false positive background
                sample_unit_losses.append(d_map.new_zeros(()))
                sample_bg_losses.append(d_flat.mean() * 10.0)
                continue

            pts = pts.to(device=device, dtype=torch.float32).reshape(-1, 2)
            N = pts.shape[0]

            # Scale points to stride-4 cell space
            pts4 = pts / float(self.output_stride)  # (N, 2)

            # Adaptive radius per point based on nearest neighbor in cell space
            if N > 1:
                p_diff = pts4.unsqueeze(1) - pts4.unsqueeze(0)  # (N, N, 2)
                p_dist = torch.sqrt(torch.sum(p_diff ** 2, dim=-1) + 1e-8)  # (N, N)
                p_dist.fill_diagonal_(float("inf"))
                d_nn_cells = p_dist.min(dim=1).values  # (N,)
                radii_cells = torch.clamp(
                    0.8 * d_nn_cells,
                    min=self.min_radius_cells,
                    max=self.max_radius_cells,
                )
            else:
                radii_cells = torch.full(
                    (1,), self.max_radius_cells, dtype=torch.float32, device=device
                )

            # Pairwise distance: (K, N) between all K cells and N points
            gx = grid[:, 0:1]  # (K, 1)
            gy = grid[:, 1:2]  # (K, 1)
            px = pts4[:, 0:1].t()  # (1, N)
            py = pts4[:, 1:2].t()  # (1, N)
            dist2 = (gx - px) ** 2 + (gy - py) ** 2  # (K, N)

            # Nearest point index and distance for each cell
            min_dist2, nearest_idx = dist2.min(dim=1)  # (K,), (K,)
            min_dist = torch.sqrt(min_dist2.clamp_min(1e-8))  # (K,)

            # Head support mask: cell k is within R_{nearest_idx[k]} of its assigned point
            assigned_radii = radii_cells[nearest_idx]  # (K,)
            head_mask = min_dist <= assigned_radii  # (K,) bool
            bg_mask = ~head_mask

            # 1. Accumulate mass for each head ONLY from cells within its valid radius
            d_head = torch.where(head_mask, d_flat, torch.zeros_like(d_flat))
            m = torch.zeros(N, dtype=d_flat.dtype, device=device)
            m.scatter_add_(0, nearest_idx, d_head)

            l_unit = torch.mean(torch.abs(m - 1.0))
            sample_unit_losses.append(l_unit)

            # 2. Background suppression: strictly penalize mass outside all head radii
            if bg_mask.any():
                l_bg = d_flat[bg_mask].mean()
            else:
                l_bg = d_map.new_zeros(())
            sample_bg_losses.append(l_bg)

        loss_unit = torch.stack(sample_unit_losses).mean()
        loss_bg = torch.stack(sample_bg_losses).mean()

        total_point_loss = loss_unit + self.bg_weight * loss_bg

        loss_dict = {
            "loss_point_unit": loss_unit.detach(),
            "loss_point_bg": loss_bg.detach(),
            "loss_point_total": total_point_loss.detach(),
        }
        return total_point_loss, loss_dict
