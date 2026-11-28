from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .necks import ConvGNAct

_DEFAULT_ANCHORS: tuple[float, ...] = (0.0, 0.004, 0.015, 0.05, 0.15, 0.40, 1.0, 2.5)


class LERSubpixelHead(nn.Module):
    """Perspective-Prior Log-Spaced Expectation-Residual Density Head (LER-Head).

    Replaces 1-channel Softplus(z) regression with a differentiable log-spaced density
    distribution over K anchors starting at exact zero (c_0 = 0.0) plus an open-ended
    residual tail on the highest anchor, optionally conditioned by a row-wise perspective
    prior pi(y) from PGRFPNNeck.
    """

    def __init__(
        self,
        width: int = 32,
        init_bias: float = -4.1422,
        subpixel_kernel_size: int = 3,
        head_blocks: int = 1,
        use_persp_prior: bool = False,
        anchors: tuple[float, ...] = _DEFAULT_ANCHORS,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.subpixel_stride2 = True
        self.use_persp_prior = bool(use_persp_prior)
        anchor_t = torch.tensor(anchors, dtype=torch.float32)
        if anchor_t.ndim != 1 or anchor_t.numel() < 2 or float(anchor_t[0].item()) != 0.0:
            raise ValueError(f"anchors must be 1D with >=2 entries starting at 0.0, got {anchors}")
        self.num_bins = int(anchor_t.numel())
        self.register_buffer("anchors", anchor_t, persistent=True)

        blocks = []
        for _ in range(max(1, int(head_blocks))):
            blocks.append(ConvGNAct(self.width, self.width, 3, groups=self.width))
            blocks.append(ConvGNAct(self.width, self.width, 1))
        self.body = nn.Sequential(*blocks)

        pre_k = max(1, int(subpixel_kernel_size))
        self.subpixel_proj = nn.Conv2d(self.width, 4, kernel_size=pre_k, padding=pre_k // 2)
        nn.init.normal_(self.subpixel_proj.weight, std=0.01 / float(pre_k))
        nn.init.zeros_(self.subpixel_proj.bias)

        self.bin_proj = nn.Conv2d(self.width, self.num_bins, kernel_size=1, bias=True)
        nn.init.normal_(self.bin_proj.weight, std=0.01)

        self.res_proj = nn.Conv2d(self.width, 1, kernel_size=1, bias=True)
        nn.init.normal_(self.res_proj.weight, std=0.01)
        nn.init.constant_(self.res_proj.bias, -4.0)

        if self.use_persp_prior:
            self.persp_prior_proj: nn.Conv2d | None = nn.Conv2d(self.width, self.num_bins, kernel_size=1, bias=False)
            nn.init.normal_(self.persp_prior_proj.weight, std=0.01)
            self.beta_persp: nn.Parameter | None = nn.Parameter(torch.full((1,), 0.1))
        else:
            self.persp_prior_proj = None
            self.beta_persp = None

        # Bisection calibration of bin_proj.bias so Step-0 E[y_cell] == m0 / 4.0
        m0 = math.log1p(math.exp(float(init_bias)))
        target_sub = max(m0 / 4.0, 1e-6)
        r0 = math.log1p(math.exp(-4.0))
        eff_anchors = anchor_t.clone()
        eff_anchors[-1] = eff_anchors[-1] + r0
        lo, hi = 0.05, 50.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            logits = -mid * torch.log1p(eff_anchors / target_sub)
            mean_val = float((torch.softmax(logits, dim=0) * eff_anchors).sum().item())
            if mean_val > target_sub:
                lo = mid
            else:
                hi = mid
        tau_opt = 0.5 * (lo + hi)
        with torch.no_grad():
            self.bin_proj.bias.copy_(-tau_opt * torch.log1p(eff_anchors / target_sub))

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward_with_logits(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
        need_full_z0: bool = False,
        force_activate: bool = True,
        row_ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (z0, y0) via perspective-conditioned log-spaced expectation + residual tail."""
        if isinstance(f, tuple):
            f = f[0]
        h4 = self.body(f)
        alpha = 4.0 * F.softmax(self.subpixel_proj(h4), dim=1)
        z_bins = self.bin_proj(h4)

        if self.use_persp_prior:
            if row_ctx is None:
                raise ValueError("LERSubpixelHead(use_persp_prior=True) requires row_ctx from PGRFPNNeck.")
            assert self.persp_prior_proj is not None and self.beta_persp is not None
            if row_ctx.shape[-2] != h4.shape[-2] or row_ctx.shape[-1] != 1:
                row_ctx = F.interpolate(row_ctx, size=(h4.shape[-2], 1), mode="bilinear", align_corners=False)
            z_bins = z_bins + self.beta_persp.view(1, 1, 1, 1) * self.persp_prior_proj(row_ctx)

        probs = F.softmax(z_bins.float(), dim=1)
        r_tail = F.softplus(self.res_proj(h4).float())
        anchors_view = self.anchors.to(device=probs.device, dtype=torch.float32).view(1, self.num_bins, 1, 1)
        y_cell = (probs * anchors_view).sum(dim=1, keepdim=True) + probs[:, -1:] * r_tail

        y0 = self.pixel_shuffle(y_cell * alpha.float()).to(dtype=f.dtype)
        z0_cell = torch.log(y_cell.clamp_min(1e-6))
        z0 = self.pixel_shuffle(z0_cell.expand(-1, 4, -1, -1)) if need_full_z0 else z0_cell
        return z0, y0

    def forward(
        self,
        f: tuple[torch.Tensor, ...] | torch.Tensor,
        scale_weights: torch.Tensor | None = None,
        row_ctx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_with_logits(f, scale_weights=scale_weights, need_full_z0=False, row_ctx=row_ctx)[1]
