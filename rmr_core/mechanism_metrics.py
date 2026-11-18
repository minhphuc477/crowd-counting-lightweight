"""Layer-4 Mechanistic Verification Metrics for SFI-Lite & RMRv3.

Implements quantitative causal diagnostics required for peer-reviewed verification:
  1. hf_preservation_ratio: Laplacian high-frequency energy retention across the FPN neck.
  2. peak_sharpness: Local spatial concentration 1 / (sigma_x * sigma_y) around GT head centers.
  3. spatial_entropy: Normalized Shannon entropy -sum(p log p) measuring density smearing.
  4. gate_response_analysis: Image-level and intra-image local gate4 activation across density regimes.
  5. per_size_mae: Stratified MAE by head size (small <16px, medium 16-32px, large >32px) via Voronoi assignment.
  6. compute_erf: Effective Receptive Field (Luo et al., NeurIPS 2016) radii (R50, R95) & 2D energy map.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _laplacian_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply channel-wise 3x3 discrete Laplacian operator."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    c = x.shape[1]
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=x.dtype,
        device=x.device,
    ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k, groups=c)


@torch.no_grad()
def hf_preservation_ratio(
    f_in: torch.Tensor,
    f_out: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Measure high-frequency energy ratio ||Laplacian(F)||^2 / ||F||^2 and retention across neck."""
    lap_in = _laplacian_2d(f_in.float())
    lap_out = _laplacian_2d(f_out.float())

    e_in = float(f_in.float().square().sum().item())
    e_out = float(f_out.float().square().sum().item())
    hf_in = float(lap_in.square().sum().item()) / max(e_in, eps)
    hf_out = float(lap_out.square().sum().item()) / max(e_out, eps)
    ratio = hf_out / max(hf_in, eps)
    return {
        "hf_ratio_in": hf_in,
        "hf_ratio_out": hf_out,
        "hf_preservation_ratio": ratio,
    }


@torch.no_grad()
def peak_sharpness(
    density_map: torch.Tensor,
    points_xy: torch.Tensor | np.ndarray,
    stride: int = 2,
    radius_px: float = 12.0,
    eps: float = 1e-8,
) -> dict[str, float]:
    """Measure local peak sharpness 1 / (sigma_x * sigma_y) around GT head points."""
    dm = density_map.detach().float()
    while dm.ndim > 2:
        dm = dm.squeeze(0)
    h, w = dm.shape

    if isinstance(points_xy, np.ndarray):
        pts = torch.from_numpy(points_xy).float()
    else:
        pts = points_xy.detach().float().cpu()
    if pts.numel() == 0:
        return {"mean_sharpness": 0.0, "median_sharpness": 0.0, "mean_sigma_r": 0.0, "num_valid_heads": 0}

    pts = pts.reshape(-1, 2)
    r_cells = max(1, int(round(float(radius_px) / float(stride))))

    sharpness_vals: list[float] = []
    sigma_r_vals: list[float] = []
    dm_cpu = dm.cpu()

    for px, py in pts.tolist():
        gx = int(round(px / float(stride) - 0.5))
        gy = int(round(py / float(stride) - 0.5))
        if gx < 0 or gx >= w or gy < 0 or gy >= h:
            continue
        x0, x1 = max(0, gx - r_cells), min(w, gx + r_cells + 1)
        y0, y1 = max(0, gy - r_cells), min(h, gy + r_cells + 1)
        patch = dm_cpu[y0:y1, x0:x1].clamp_min(0.0)
        mass = float(patch.sum().item())
        if mass <= 1e-6:
            continue
        p = patch / mass
        yy, xx = torch.meshgrid(
            torch.arange(y0, y1, dtype=torch.float32) * float(stride),
            torch.arange(x0, x1, dtype=torch.float32) * float(stride),
            indexing="ij",
        )
        mx = float((p * xx).sum().item())
        my = float((p * yy).sum().item())
        vx = float((p * (xx - mx).square()).sum().item())
        vy = float((p * (yy - my).square()).sum().item())
        sx = math.sqrt(max(vx, 0.25))
        sy = math.sqrt(max(vy, 0.25))
        sharpness_vals.append(1.0 / (sx * sy + eps))
        sigma_r_vals.append(math.sqrt(sx * sx + sy * sy))

    if not sharpness_vals:
        return {"mean_sharpness": 0.0, "median_sharpness": 0.0, "mean_sigma_r": 0.0, "num_valid_heads": 0}

    arr = np.asarray(sharpness_vals, dtype=np.float64)
    return {
        "mean_sharpness": float(arr.mean()),
        "median_sharpness": float(np.median(arr)),
        "mean_sigma_r": float(np.mean(sigma_r_vals)),
        "num_valid_heads": len(sharpness_vals),
    }


@torch.no_grad()
def spatial_entropy(
    density_map: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute Shannon entropy -sum(p log p) and resolution-normalized entropy of density map."""
    dm = density_map.detach().float().clamp_min(0.0)
    while dm.ndim > 2:
        dm = dm.squeeze(0)
    total_mass = float(dm.sum().item())
    n_cells = dm.numel()
    if total_mass <= eps or n_cells <= 1:
        return {"spatial_entropy": 0.0, "normalized_entropy": 0.0}
    p = (dm / total_mass).reshape(-1)
    p_nz = p[p > eps]
    ent = float(-(p_nz * torch.log(p_nz)).sum().item())
    max_ent = math.log(float(n_cells))
    return {
        "spatial_entropy": ent,
        "normalized_entropy": ent / max(max_ent, eps),
    }


@torch.no_grad()
def gate_response_analysis(
    model: nn.Module,
    images: list[torch.Tensor],
    points_list: list[torch.Tensor],
    sparse_max: float = 100.0,
    dense_min: float = 500.0,
    stride_gate: int = 4,
) -> dict[str, Any]:
    """Measure gate4 response across image-level and intra-image local density regimes."""
    fusion = getattr(model, "fusion", None)
    if fusion is None or not hasattr(fusion, "_capture_mechanisms"):
        return {"available": False, "reason": "Neck does not expose _capture_mechanisms"}

    model.eval()
    prev_flag = bool(fusion._capture_mechanisms)
    fusion._capture_mechanisms = True

    img_regimes: dict[str, list[float]] = {"sparse": [], "medium": [], "dense": []}
    local_regimes: dict[str, list[float]] = {"bg": [], "sparse_fg": [], "dense_fg": []}
    hf_ratios: list[float] = []

    try:
        for img, pts in zip(images, points_list):
            x = img.unsqueeze(0) if img.ndim == 3 else img
            device = next(model.parameters()).device
            x = x.to(device)
            _ = model(x)

            l4 = getattr(fusion, "_last_l4", None)
            p4 = getattr(fusion, "_last_p4", None)
            if l4 is not None and p4 is not None:
                hf_ratios.append(hf_preservation_ratio(l4, p4)["hf_preservation_ratio"])

            g4 = getattr(fusion, "_last_gate4", None)
            if g4 is None:
                continue

            g_mean_spatial = g4[0].mean(dim=0).cpu()  # [H/4, W/4]
            n_gt = int(pts.shape[0]) if pts is not None and pts.numel() > 0 else 0
            if n_gt <= sparse_max:
                img_regimes["sparse"].append(float(g_mean_spatial.mean().item()))
            elif n_gt <= dense_min:
                img_regimes["medium"].append(float(g_mean_spatial.mean().item()))
            else:
                img_regimes["dense"].append(float(g_mean_spatial.mean().item()))

            # Intra-image local density stratification (count GT heads within 16px of each Stride-4 cell)
            gh, gw = g_mean_spatial.shape
            local_cnt = torch.zeros((gh, gw), dtype=torch.float32)
            if n_gt > 0:
                pts_c = pts.detach().float().cpu().reshape(-1, 2) / float(stride_gate)
                xs = pts_c[:, 0].long().clamp(0, gw - 1)
                ys = pts_c[:, 1].long().clamp(0, gh - 1)
                local_cnt.index_put_((ys, xs), torch.ones_like(xs, dtype=torch.float32), accumulate=True)
                # 9x9 window at stride 4 covers 36x36 px local crowd density
                local_cnt = F.avg_pool2d(local_cnt.unsqueeze(0).unsqueeze(0), kernel_size=9, stride=1, padding=4)[0, 0] * 81.0

            bg_mask = local_cnt < 0.5
            sp_mask = (local_cnt >= 0.5) & (local_cnt < 3.0)
            de_mask = local_cnt >= 3.0
            if bg_mask.any():
                local_regimes["bg"].append(float(g_mean_spatial[bg_mask].mean().item()))
            if sp_mask.any():
                local_regimes["sparse_fg"].append(float(g_mean_spatial[sp_mask].mean().item()))
            if de_mask.any():
                local_regimes["dense_fg"].append(float(g_mean_spatial[de_mask].mean().item()))
    finally:
        fusion._capture_mechanisms = prev_flag

    def _agg(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {"mean": float("nan"), "std": float("nan"), "n": 0}
        arr = np.asarray(vals, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}

    has_gate = any(len(v) > 0 for v in img_regimes.values())
    return {
        "available": has_gate,
        "image_level": {k: _agg(v) for k, v in img_regimes.items()},
        "local_patch_level": {k: _agg(v) for k, v in local_regimes.items()},
        "mean_hf_preservation_ratio": float(np.mean(hf_ratios)) if hf_ratios else float("nan"),
    }


@torch.no_grad()
def per_size_mae(
    pred_maps: list[torch.Tensor],
    points_list: list[torch.Tensor],
    boxes_list: list[torch.Tensor | None] | None = None,
    stride: int = 2,
    small_thresh: float = 16.0,
    medium_thresh: float = 32.0,
    max_assign_dist_px: float = 32.0,
) -> dict[str, Any]:
    """Compute per-head MAE stratified by head size (small <16px, medium 16-32px, large >32px).

    Uses ground-truth head bounding boxes sqrt(w*h) when available (e.g., NWPU-Crowd),
    falling back transparently to 3-NN inter-head distance proxy on point-only datasets.
    """
    tiers: dict[str, list[float]] = {"small": [], "medium": [], "large": []}
    used_gt_boxes = 0
    used_knn_proxy = 0

    for idx, (dm, pts) in enumerate(zip(pred_maps, points_list)):
        if pts is None or pts.numel() == 0:
            continue
        pts_xy = pts.detach().float().cpu().reshape(-1, 2)
        n = pts_xy.shape[0]

        dm_2d = dm.detach().float().cpu().clamp_min(0.0)
        while dm_2d.ndim > 2:
            dm_2d = dm_2d.squeeze(0)
        h, w = dm_2d.shape

        # Determine head diameters
        box = boxes_list[idx] if (boxes_list is not None and idx < len(boxes_list)) else None
        if box is not None and box.numel() > 0 and box.shape[0] == n:
            b_cpu = box.detach().float().cpu()
            bw = (b_cpu[:, 2] - b_cpu[:, 0]).clamp_min(1.0)
            bh = (b_cpu[:, 3] - b_cpu[:, 1]).clamp_min(1.0)
            sizes = torch.sqrt(bw * bh)
            used_gt_boxes += 1
        else:
            if n >= 2:
                dmat = torch.cdist(pts_xy, pts_xy)
                k = min(4, n)
                knn_d = torch.topk(dmat, k=k, largest=False).values[:, 1:]
                sizes = knn_d.mean(dim=1).clamp(4.0, 64.0)
            else:
                sizes = torch.full((n,), 24.0, dtype=torch.float32)
            used_knn_proxy += 1

        # Voronoi assignment of predicted cells to nearest GT head within max_assign_dist_px
        nz_y, nz_x = torch.nonzero(dm_2d > 1e-6, as_tuple=True)
        head_preds = torch.zeros(n, dtype=torch.float32)
        if nz_y.numel() > 0:
            cell_xy = torch.stack(
                [(nz_x.float() + 0.5) * float(stride), (nz_y.float() + 0.5) * float(stride)],
                dim=1,
            )
            cell_mass = dm_2d[nz_y, nz_x]
            # Chunked nearest-neighbor assignment to bound memory on dense images
            chunk = 4096
            for s0 in range(0, cell_xy.shape[0], chunk):
                c_sub = cell_xy[s0 : s0 + chunk]
                m_sub = cell_mass[s0 : s0 + chunk]
                dists = torch.cdist(c_sub, pts_xy)
                min_d, nn_idx = dists.min(dim=1)
                valid = min_d <= float(max_assign_dist_px)
                if valid.any():
                    head_preds.index_add_(0, nn_idx[valid], m_sub[valid])

        abs_errs = (head_preds - 1.0).abs().tolist()
        for sz, err in zip(sizes.tolist(), abs_errs):
            if sz < small_thresh:
                tiers["small"].append(err)
            elif sz <= medium_thresh:
                tiers["medium"].append(err)
            else:
                tiers["large"].append(err)

    size_source = (
        "gt_boxes" if (used_gt_boxes > 0 and used_knn_proxy == 0)
        else ("knn_proxy" if used_gt_boxes == 0 else "mixed")
    )
    return {
        "size_source": size_source,
        "small_lt16_mae": float(np.mean(tiers["small"])) if tiers["small"] else float("nan"),
        "small_count": len(tiers["small"]),
        "medium_16_32_mae": float(np.mean(tiers["medium"])) if tiers["medium"] else float("nan"),
        "medium_count": len(tiers["medium"]),
        "large_gt32_mae": float(np.mean(tiers["large"])) if tiers["large"] else float("nan"),
        "large_count": len(tiers["large"]),
    }


def compute_erf(
    model: nn.Module,
    input_shape: tuple[int, int, int, int] = (1, 3, 256, 256),
    target_layer: str = "p4",
    num_samples: int = 8,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Compute Effective Receptive Field (Luo et al., NeurIPS 2016) at center output neuron."""
    model.eval()
    dev = torch.device(device)
    _, c, h_in, w_in = input_shape
    erf_accum = torch.zeros((h_in, w_in), dtype=torch.float64, device=dev)

    for s in range(max(1, int(num_samples))):
        gen = torch.Generator(device="cpu").manual_seed(1000 + s)
        x = torch.randn((1, c, h_in, w_in), generator=gen).to(dev).requires_grad_(True)
        model.zero_grad(set_to_none=True)

        if target_layer == "p4" and hasattr(model, "_extract_carrier_features"):
            p4, _, _ = model._extract_carrier_features(x)
            feat = p4
        else:
            out = model(x)
            feat = out["y"] if isinstance(out, dict) else out.y

        fh, fw = feat.shape[-2:]
        center_val = feat[0, :, fh // 2, fw // 2].sum()
        center_val.backward()
        if x.grad is not None:
            erf_accum += x.grad.detach()[0].abs().sum(dim=0).double()

    erf_map = erf_accum / max(float(erf_accum.max().item()), 1e-12)
    energy = erf_accum / max(float(erf_accum.sum().item()), 1e-12)

    yy, xx = torch.meshgrid(
        torch.arange(h_in, device=dev, dtype=torch.float64),
        torch.arange(w_in, device=dev, dtype=torch.float64),
        indexing="ij",
    )
    cy, cx = (h_in - 1) * 0.5, (w_in - 1) * 0.5
    rad = torch.sqrt((yy - cy).square() + (xx - cx).square()).reshape(-1)
    e_flat = energy.reshape(-1)
    order = torch.argsort(rad)
    rad_sorted = rad[order]
    cum_e = torch.cumsum(e_flat[order], dim=0)

    r50_idx = int(torch.searchsorted(cum_e, torch.tensor(0.50, device=dev, dtype=torch.float64)).clamp_max(rad_sorted.numel() - 1).item())
    r95_idx = int(torch.searchsorted(cum_e, torch.tensor(0.95, device=dev, dtype=torch.float64)).clamp_max(rad_sorted.numel() - 1).item())

    return {
        "r50_px": float(rad_sorted[r50_idx].item()),
        "r95_px": float(rad_sorted[r95_idx].item()),
        "erf_map": erf_map.cpu().numpy(),
    }
