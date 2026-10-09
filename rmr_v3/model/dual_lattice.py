from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F


def push_forward_stride2_to_stride4(y_stride2: torch.Tensor) -> torch.Tensor:
    """Exact discrete Radon measure push-forward from Stride 2 to Stride 4 lattice.

    Computes 2x2 cell box summation:
        y^{(4)}_{i,j} = sum_{u=0}^1 sum_{v=0}^1 y^{(2)}_{2i+u, 2j+v}
    Preserves exact total mass: sum(y^{(4)}) == sum(y^{(2)}) to machine precision.
    Trainable parameters: exactly 0.
    """
    orig_ndim = y_stride2.ndim
    if orig_ndim == 2:
        y4d = y_stride2.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        y4d = y_stride2.unsqueeze(1)
    elif orig_ndim == 4:
        y4d = y_stride2
    else:
        raise ValueError(f"push_forward_stride2_to_stride4 expects 2D, 3D, or 4D tensor, got ndim={orig_ndim}")

    h, w = y4d.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h > 0 or pad_w > 0:
        y4d = F.pad(y4d.float(), (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    else:
        y4d = y4d.float()

    # Non-overlapping 2x2 sum pooling: 4.0 * AvgPool2d(2, 2)
    carrier = 4.0 * F.avg_pool2d(y4d, kernel_size=2, stride=2, count_include_pad=False)
    carrier = carrier.to(dtype=y_stride2.dtype)

    if orig_ndim == 2:
        return carrier.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        return carrier.squeeze(1)
    return carrier


def pullback_stride4_to_stride2_rn(
    y_carrier: torch.Tensor,
    y_fine_prior: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Radon-Nikodym measure prolongation from Stride 4 carrier to Stride 2 fine lattice.

    Redistributes coarse carrier mass onto fine cells proportionally to fine prior:
        y^{(2)}_u = y^{(4)}_k * (y^{(2, 0)}_u / sum_{v in sub(k)} y^{(2, 0)}_v)
    with uniform fallback (0.25) when carrier block mass <= eps.
    Guarantees strict local and global mass conservation (Theorem 1) and
    support preservation (Theorem 2).
    """
    y4 = y_carrier.float()
    y2_0 = y_fine_prior.float()

    prior_carrier = push_forward_stride2_to_stride4(y2_0)
    h, w = y2_0.shape[-2:]
    pad_h = h % 2
    pad_w = w % 2
    target_size = (h + pad_h, w + pad_w)

    y4_up = F.interpolate(y4, size=target_size, mode="nearest")[..., :h, :w]
    prior_carrier_up = F.interpolate(prior_carrier, size=target_size, mode="nearest")[..., :h, :w]

    zero_prior_mask = (prior_carrier_up <= float(eps))
    denom = torch.where(zero_prior_mask, torch.ones_like(prior_carrier_up), prior_carrier_up)
    block_cell_counts = push_forward_stride2_to_stride4(torch.ones_like(y2_0))
    block_cell_counts_up = F.interpolate(block_cell_counts, size=target_size, mode="nearest")[..., :h, :w]
    uniform_weight = 1.0 / block_cell_counts_up.clamp_min(1.0)
    weights = torch.where(zero_prior_mask, uniform_weight, y2_0 / denom)
    fine_prolong = y4_up * weights
    return fine_prolong.to(dtype=y_carrier.dtype)


def check_mass_conservation(
    y_fine: torch.Tensor,
    y_carrier: torch.Tensor,
    eps: float = 1e-6,
) -> bool:
    """Verify discrete mass conservation between fine and carrier lattices."""
    m_fine = y_fine.double().sum(dim=(-2, -1))
    m_carrier = y_carrier.double().sum(dim=(-2, -1))
    diff = (m_fine - m_carrier).abs()
    max_abs = diff.max().item()
    max_rel = (diff / m_carrier.clamp_min(1e-6)).max().item()
    return (max_abs < eps) or (max_rel < eps)


def scale_regions_to_stride2(
    regions_feat: Any,
    target_h: int,
    target_w: int,
) -> Any:
    """Scale Stride 4 region boxes to Stride 2 solver grid with zero-stall object-bound caching."""
    cached = getattr(regions_feat, "_stride2_scaled", None)
    if cached is not None and getattr(regions_feat, "_stride2_hw", None) == (int(target_h), int(target_w)):
        return cached

    from rmr_core.operators import RegionSet
    b_s2 = regions_feat.boxes * 2
    b_s2[:, [0, 2]] = b_s2[:, [0, 2]].clamp(0, target_h)
    b_s2[:, [1, 3]] = b_s2[:, [1, 3]].clamp(0, target_w)
    bl_s2 = [
        (min(target_h, 2 * y1), min(target_w, 2 * x1), min(target_h, 2 * y2), min(target_w, 2 * x2))
        for (y1, x1, y2, x2) in regions_feat.boxes_list
    ] if regions_feat.boxes_list is not None else None
    dh = (b_s2[:, 2] - b_s2[:, 0]).clamp_min(1)
    dw = (b_s2[:, 3] - b_s2[:, 1]).clamp_min(1)
    area_s2 = (dh * dw).float()
    res = RegionSet(
        boxes=b_s2,
        scale_id=regions_feat.scale_id,
        area=area_s2,
        boxes_list=bl_s2,
        num_scales=regions_feat.num_scales,
        scale_sizes_px=regions_feat.scale_sizes_px,
    )
    object.__setattr__(regions_feat, "_stride2_scaled", res)
    object.__setattr__(regions_feat, "_stride2_hw", (int(target_h), int(target_w)))
    return res


class SubpixelAllocationHead(torch.nn.Module):
    """Sub-pixel Dirichlet-Multinomial Mass-Preserving Allocation Head (DM-DL).

    Resolves multi-head collisions (up to 4 heads per Stride 4 cell) at Stride 2:
      z_alloc = PW(SiLU(DW(P4)))  in R^{B x 4 x H_4 x W_4}
      pi = Softmax(z_alloc, dim=1)  in Delta^3
      Y_alloc = Y_4 * pi
      Y_2 = PixelShuffle(2)(Y_alloc)  in R^{B x 1 x 2H_4 x 2W_4}

    Guarantees:
      1. Strict Mass Conservation: sum(Y_2) == sum(Y_4) across each 2x2 fine block.
      2. Step 0 Parity: Zero-initialized PW conv ensures exact uniform [0.25, 0.25, 0.25, 0.25]
         allocation at initialization, preserving initial calibrated rate.
      3. Ultra-lightweight: Exactly 452 trainable parameters.
    """

    def __init__(self, in_channels: int = 32) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            torch.nn.SiLU(inplace=True),
            torch.nn.Conv2d(in_channels, 4, kernel_size=1),
        )
        # Step 0 Parity: zero-init guarantees uniform 0.25 allocation
        torch.nn.init.zeros_(self.conv[-1].weight)
        torch.nn.init.zeros_(self.conv[-1].bias)
        self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=2)

    def forward(self, p4: torch.Tensor, y4: torch.Tensor) -> torch.Tensor:
        """Project coarse Stride 4 measure y4 to fine Stride 2 measure y2."""
        orig_ndim = y4.ndim
        if orig_ndim == 2:
            y4_in = y4.unsqueeze(0).unsqueeze(0)
        elif orig_ndim == 3:
            y4_in = y4.unsqueeze(1)
        else:
            y4_in = y4

        logits = self.conv(p4)
        if logits.shape[-2:] != y4_in.shape[-2:]:
            logits = F.interpolate(logits, size=y4_in.shape[-2:], mode="bilinear", align_corners=False)
        pi = F.softmax(logits, dim=1)
        y_alloc = y4_in * pi
        y2 = self.pixel_shuffle(y_alloc)

        if orig_ndim == 2:
            return y2.squeeze(0).squeeze(0)
        if orig_ndim == 3:
            return y2.squeeze(1)
        return y2

    def forward_pair(
        self, p4: torch.Tensor, y4: torch.Tensor, y04: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project both coarse measures (y, y0) simultaneously using a shared allocation partition."""
        logits = self.conv(p4)
        if logits.shape[-2:] != y4.shape[-2:]:
            logits = F.interpolate(logits, size=y4.shape[-2:], mode="bilinear", align_corners=False)
        pi = F.softmax(logits, dim=1)
        y2 = self.pixel_shuffle(y4 * pi)
        y02 = self.pixel_shuffle(y04 * pi)
        return y2, y02



