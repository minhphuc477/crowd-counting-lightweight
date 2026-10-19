from __future__ import annotations

import torch
import torch.nn.functional as F

from .adjoint import regional_adjoint, weighted_coverage
from .prefix_sums import regional_sum
from .regions import RegionSet


def closed_form_radon_nikodym_transport(
    y0: torch.Tensor,
    b_solver: torch.Tensor,
    weight_solver: torch.Tensor | None,
    regions: RegionSet,
    eps: float = 1e-6,
    scale_weights: torch.Tensor | None = None,
    ratio_clamp_max: float = 10.0,
) -> torch.Tensor:
    """Exact 1-step Closed-Form Radon-Nikodym Mass Transport Operator (0 Parameters).

    Projects fine carrier density y0 onto regional evidence b_solver in closed-form:
        b_hat = regional_sum(y0, regions)
        s_R   = b_solver / clamp_min(b_hat, eps)
        S(u)  = (A^T (w * s))(u) / (A^T w)(u)
        y*(u) = y0(u) * S(u)

    Mathematical Guarantees:
    1. Exact 1-step closed-form projection (0 unrolled iterations, O(1) latency).
    2. Strict support preservation (y*(u) = 0 wherever y0(u) = 0, zero false alarms).
    3. Non-negativity preservation (y*(u) >= 0 for all u).
    4. 0 trainable parameters, 0 unrolled computational graph buffers.
    """
    if y0.ndim == 2:
        y0 = y0.unsqueeze(0).unsqueeze(0)
    elif y0.ndim == 3:
        y0 = y0.unsqueeze(1)

    b, c, h, w = y0.shape
    if b_solver.ndim == 2:
        b_solver = b_solver.unsqueeze(1)
    if weight_solver is None:
        weight_solver = torch.ones_like(b_solver)
    elif weight_solver.ndim == 2:
        weight_solver = weight_solver.unsqueeze(1)

    work_dtype = y0.dtype if y0.dtype in (torch.float32, torch.float64) else torch.float32
    y0_f = y0.to(dtype=work_dtype)
    b_s_f = b_solver.to(dtype=work_dtype).clamp_min(0.0)
    w_s_f = weight_solver.to(dtype=work_dtype).clamp_min(0.0)

    # 1. Forward measure projection: integrate fine density over multiscale boxes
    boxes_tensor = regions.boxes if hasattr(regions, "boxes") else regions
    b_hat = regional_sum(y0_f, boxes_tensor)  # [B, 1, M]

    # 2. Regional Radon-Nikodym scaling derivative s_R = d(mu_target) / d(mu_hat)
    scaling_ratio = b_s_f / b_hat.clamp_min(float(eps))
    if float(ratio_clamp_max) > 0.0:
        scaling_ratio = scaling_ratio.clamp(0.0, float(ratio_clamp_max))

    # 3. Adjoint coverage normalization field D_{c,w} = A^T w
    cov = weighted_coverage(
        w_s_f,
        regions,
        height=h,
        width=w,
        eps=eps,
        scale_routing_weights=scale_weights,
    ).to(dtype=work_dtype)

    # 4. Adjoint pull-back of weighted scaling factor
    weighted_s = (w_s_f * scaling_ratio).to(dtype=work_dtype)
    adjoint_s = regional_adjoint(
        weighted_s,
        regions.boxes,
        height=h,
        width=w,
        out_dtype=work_dtype,
    )

    # 5. Continuous Radon-Nikodym spatial density multiplier
    s_field = adjoint_s / cov.clamp_min(float(eps))

    # 6. Push-forward projected measure
    y_projected = (y0_f * s_field).to(dtype=y0.dtype)
    return y_projected
