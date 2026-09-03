"""MICF-Lite lightweight crowd counter.

Unified architecture supporting all pilot models (B1 to B6, plus B7 axial context):
- Backbone: MobileNetV4ConvSmall (C4, C8, C16)
- Neck: Additive FPN Neck with multi-scale routes (P4, P8, P16)
- Stride routing: directly reads P16, P8, or P4 natively from neck routes (no wasted downsampling)
- Context:
  - None / Identity (B1, B2, B3, B4)
  - 4-Directional Normalized Integral Context (B5, B6)
  - Axial Integral Context (B7)
- Head:
  - 'local': Predicts discrete cell counts Y >= 0 via softplus (B1, B2, B6)
  - 'cumulative': Predicts raw linear monotonic integral count field C (B3, B4, B5)
  - 'integrated_local': Valid-by-construction baseline M = softplus(z) -> C = Integral(M) (Section 32)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import MobileNetV4Backbone
from .blocks import make_group_norm
from .integral_context import AxialIntegralContext, DirectionalIntegralContext
from .neck import AdditiveFPNNeck
from hpc.losses.micf import discrete_mixed_difference


def _prefix_area_tl(h: int, w: int, device, dtype) -> torch.Tensor:
    """Prefix area (i+1)(j+1) for the TL cumulative field, shape [H, W]."""
    i = torch.arange(1, h + 1, device=device, dtype=dtype)
    j = torch.arange(1, w + 1, device=device, dtype=dtype)
    return i[:, None] * j[None, :]


def _normalized_coords(h: int, w: int, device, dtype) -> torch.Tensor:
    """Normalized (row, col) prefix-position channels, shape [2, H, W], each in (0, 1]."""
    i = torch.arange(1, h + 1, device=device, dtype=dtype) / h
    j = torch.arange(1, w + 1, device=device, dtype=dtype) / w
    ii = i[:, None].expand(h, w)
    jj = j[None, :].expand(h, w)
    return torch.stack([ii, jj], dim=0)


def _partition_feature_blocks(
    x: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, int, int]:
    """Partition [B,C,H,W] into shared-weight KxK blocks.

    Returns:
        blocks: [B*n_h*n_w, C, K, K]
        n_h: number of block rows
        n_w: number of block columns

    The caller must ensure H and W are divisible by K.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(x.shape)}")

    B, C, H, W = x.shape
    if H % k != 0 or W % k != 0:
        raise ValueError(
            f"FH grid ({H},{W}) must be divisible by finite_horizon={k}. "
            "Use a compatible crop/padding multiple."
        )

    n_h = H // k
    n_w = W // k

    blocks = (
        x.view(B, C, n_h, k, n_w, k)
         .permute(0, 2, 4, 1, 3, 5)
         .contiguous()
         .view(B * n_h * n_w, C, k, k)
    )
    return blocks, n_h, n_w


def compose_batched_fh_cumulative(
    c_blocks: torch.Tensor,
    batch_size: int,
    n_h: int,
    n_w: int,
    k: int,
) -> torch.Tensor:
    """Compose block-local cumulative fields into one global MICF field.

    Args:
        c_blocks: [B*n_h*n_w, 1, K, K]
        batch_size: B
        n_h, n_w: block-grid dimensions
        k: finite horizon K

    Returns:
        c_global: [B, 1, n_h*K, n_w*K]

    Composition is exact and has no learnable parameters.
    """
    if c_blocks.shape != (batch_size * n_h * n_w, 1, k, k):
        raise ValueError(
            "Unexpected c_blocks shape: "
            f"{tuple(c_blocks.shape)} != "
            f"{(batch_size * n_h * n_w, 1, k, k)}"
        )

    # Local cumulative -> local signed mass.
    # Validity is still handled by the existing MICF loss.
    y_blocks = discrete_mixed_difference(c_blocks)

    # [B, nh, nw, 1, K, K]
    y_blocks = y_blocks.view(batch_size, n_h, n_w, 1, k, k)

    # Stitch blocks spatially into [B, 1, nh*K, nw*K].
    y_global = (
        y_blocks.permute(0, 3, 1, 4, 2, 5)
                .contiguous()
                .view(batch_size, 1, n_h * k, n_w * k)
    )

    # Exact global MICF reconstruction.
    c_global = torch.cumsum(torch.cumsum(y_global, dim=-2), dim=-1)
    return c_global


def compose_tiled_cumulative_field(c_local: list[list[torch.Tensor]]) -> torch.Tensor:
    """Exact block-decomposed 2D prefix-sum composition (design doc sec.30, generalized).

    Given a grid of per-tile *local* cumulative count fields c_local[i][j]
    (each computed independently, as if that tile were its own full image),
    returns the single global cumulative field consistent with treating the
    whole tile grid as one image -- equal to computing Y for the full image
    and taking cumsum2d(Y) directly.

    Each tile's own bottom row / right column / bottom-right corner already
    equal the row-prefix, column-prefix, and total-mass summaries its
    down-and-right neighbors need, so composition costs O(n_tiles_h *
    n_tiles_w) vector adds on top of the tiles already computed -- no
    additional model forward passes.
    """
    n_tiles_h = len(c_local)
    n_tiles_w = len(c_local[0])
    out_tile_h, out_tile_w = c_local[0][0].shape
    device = c_local[0][0].device
    dtype = c_local[0][0].dtype

    row_edge = [[c_local[i][j][:, -1] for j in range(n_tiles_w)] for i in range(n_tiles_h)]
    col_edge = [[c_local[i][j][-1, :] for j in range(n_tiles_w)] for i in range(n_tiles_h)]
    corner = [[c_local[i][j][-1, -1] for j in range(n_tiles_w)] for i in range(n_tiles_h)]

    # row_offset[I,J] = sum_{j<J} row_edge[I,j]  (exclusive prefix along J, per tile-row I)
    row_offset = [[torch.zeros(out_tile_h, device=device, dtype=dtype) for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    for i in range(n_tiles_h):
        acc = torch.zeros(out_tile_h, device=device, dtype=dtype)
        for j in range(n_tiles_w):
            row_offset[i][j] = acc
            acc = acc + row_edge[i][j]

    # col_offset[I,J] = sum_{i<I} col_edge[i,J]  (exclusive prefix along I, per tile-col J)
    col_offset = [[torch.zeros(out_tile_w, device=device, dtype=dtype) for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    for j in range(n_tiles_w):
        acc = torch.zeros(out_tile_w, device=device, dtype=dtype)
        for i in range(n_tiles_h):
            col_offset[i][j] = acc
            acc = acc + col_edge[i][j]

    # block_prefix[I,J] = sum_{i<I, j<J} corner[i,j]  -- two clean exclusive-prefix passes:
    # pass 1: exclusive row-prefix of corner along j, for each row i
    corner_row_prefix = [[torch.zeros((), device=device, dtype=dtype) for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    for i in range(n_tiles_h):
        acc = torch.zeros((), device=device, dtype=dtype)
        for j in range(n_tiles_w):
            corner_row_prefix[i][j] = acc
            acc = acc + corner[i][j]
    # pass 2: exclusive col-prefix (along i) of corner_row_prefix, for each J
    block_prefix = [[torch.zeros((), device=device, dtype=dtype) for _ in range(n_tiles_w)] for _ in range(n_tiles_h)]
    for j in range(n_tiles_w):
        acc = torch.zeros((), device=device, dtype=dtype)
        for i in range(n_tiles_h):
            block_prefix[i][j] = acc
            acc = acc + corner_row_prefix[i][j]

    c_global = torch.zeros(n_tiles_h * out_tile_h, n_tiles_w * out_tile_w, device=device, dtype=dtype)
    for i in range(n_tiles_h):
        for j in range(n_tiles_w):
            tile_val = (
                c_local[i][j]
                + row_offset[i][j].unsqueeze(-1)
                + col_offset[i][j].unsqueeze(-2)
                + block_prefix[i][j]
            )
            c_global[
                i * out_tile_h:(i + 1) * out_tile_h,
                j * out_tile_w:(j + 1) * out_tile_w,
            ] = tile_val
    return c_global


class MICFLite(nn.Module):
    """MICF-Lite model supporting local and cumulative count prediction."""

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small_050",
        pretrained: bool = False,
        neck_width: int = 32,
        context_dilations: Tuple[int, ...] = (1, 2, 3),
        use_integral_context: bool = False,
        context_type: str = "directional",   # 'none' | 'directional' | 'axial'
        head_type: str = "cumulative",
        output_stride: int = 16,
        feature_reductions: Tuple[int, ...] = (4, 8, 16),
        eps_d: float = 1e-8,
        extent_aware: bool = False,
        finite_horizon: int | None = None,
    ) -> None:
        super().__init__()
        self.head_type = head_type.lower()
        if self.head_type not in {"local", "cumulative", "integrated_local"}:
            raise ValueError(
                f"head_type must be 'local', 'cumulative', or 'integrated_local', got {head_type}"
            )

        self.extent_aware = bool(extent_aware)
        if self.extent_aware and self.head_type != "cumulative":
            raise ValueError("extent_aware=True is only meaningful for head_type='cumulative'")

        self.finite_horizon = None if finite_horizon is None else int(finite_horizon)
        if self.finite_horizon is not None:
            if self.finite_horizon <= 0:
                raise ValueError("finite_horizon must be a positive integer or None")
            if self.head_type != "cumulative":
                raise ValueError("finite_horizon is only supported for cumulative heads")
            if not self.extent_aware:
                raise ValueError(
                    "FH-CMICF requires extent_aware=True so local cumulative fields "
                    "use local extent-aware parameterization"
                )

        self.output_stride = int(output_stride)
        if self.output_stride not in {4, 8, 16}:
            raise ValueError(f"output_stride must be 4, 8, or 16, got {self.output_stride}")

        # use_integral_context kept for backward compatibility: True maps to
        # context_type='directional' unless context_type was explicitly overridden.
        self.context_type = context_type.lower() if use_integral_context else "none"
        self.use_integral_context = self.context_type != "none"
        if self.context_type not in {"none", "directional", "axial"}:
            raise ValueError(
                f"context_type must be 'none', 'directional', or 'axial', got {self.context_type}"
            )

        self.neck_width = int(neck_width)
        self.context_dilations = tuple(int(d) for d in context_dilations)
        self.eps_d = float(eps_d)
        self.feature_reductions = tuple(int(r) for r in feature_reductions)

        # 1. Backbone
        self.backbone = MobileNetV4Backbone(
            model_name=backbone_name,
            pretrained=pretrained,
            target_reductions=self.feature_reductions,
        )

        # 2. Additive FPN Neck
        self.neck = AdditiveFPNNeck(
            in_channels=self.backbone.out_channels,
            width=self.neck_width,
            context_dilations=context_dilations,
            target_stride=self.output_stride,
        )

        # 3. Context Decoder: Directional / Axial Integral Context vs Identity
        if self.context_type == "directional":
            self.context_module = DirectionalIntegralContext(channels=self.neck_width)
        elif self.context_type == "axial":
            self.context_module = AxialIntegralContext(channels=self.neck_width)
        else:
            self.context_module = nn.Identity()

        # Extent-aware coordinate injection (fixes normalized-mean-context vs
        # extensive-target mismatch demonstrated by the uniform-density
        # counterexample: TL-mean context is ~constant even though C must
        # grow with prefix area).
        if self.extent_aware:
            self.coord_proj = nn.Conv2d(self.neck_width + 2, self.neck_width, kernel_size=1, bias=False)
        else:
            self.coord_proj = None

        # 4. Mass Head (depthwise-separable 1-channel projection)
        self.head_dw = nn.Conv2d(
            self.neck_width, self.neck_width, kernel_size=3, padding=1, groups=self.neck_width, bias=False
        )
        self.head_norm = make_group_norm(self.neck_width)
        self.head_act = nn.SiLU()
        self.head_out = nn.Conv2d(self.neck_width, 1, kernel_size=1)

    def _predict_from_context(self, p_context: torch.Tensor) -> torch.Tensor:
        """Apply the existing task head to one spatial scope.

        For extent-aware cumulative prediction, coordinates and prefix area are
        computed from p_context.shape[-2:], so when p_context is KxK they are
        automatically local to the FH block.
        """
        if self.extent_aware:
            h_o, w_o = p_context.shape[-2:]
            coords = _normalized_coords(h_o, w_o, p_context.device, p_context.dtype)
            coords = coords.unsqueeze(0).expand(p_context.shape[0], -1, -1, -1)
            p_context = self.coord_proj(torch.cat([p_context, coords], dim=1))

        h = self.head_dw(p_context)
        h = self.head_norm(h)
        h = self.head_act(h)
        z = self.head_out(h).float()

        if self.head_type == "local":
            return F.softplus(z).clamp_min(self.eps_d)
        elif self.head_type == "integrated_local":
            m = F.softplus(z).clamp_min(self.eps_d)
            return torch.cumsum(torch.cumsum(m, dim=-2), dim=-1)
        elif self.extent_aware:
            rho = F.softplus(z)
            area = _prefix_area_tl(z.shape[-2], z.shape[-1], z.device, z.dtype)
            return area.unsqueeze(0).unsqueeze(0) * rho
        else:
            return z

    def forward_field(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass extracting predicted field (Y for local, C for cumulative).

        Stride selection: reads native feature P4, P8, or P16 directly from the neck
        routes, avoiding wasted upsampling + redundant downsampling convolutions.

        Head behaviors:
        - Local head (B1, B2, B6):
            Applies softplus to enforce Y_hat >= 0.
        - Cumulative head (B3, B4, B5):
            Returns raw linear output. Validity Delta_xy C_hat >= 0 is enforced
            via the validity loss penalty, NOT by elementwise softplus.
        - Integrated Local head (Section 32):
            Predicts local M = softplus(z) >= 0, then computes C = Integral(M).
            Guarantees Delta_xy C >= 0 by construction.
        - Extent-aware Cumulative head (B5b):
            Predicts local average prefix intensity rho = softplus(z) >= 0,
            then computes C_hat = area(i, j) * rho(i, j).
        - FH-CMICF (B8):
            Partitions feature map into KxK blocks, runs block-scoped integral context,
            block-scoped extent-aware cumulative head, and exact global composition.
        """
        features = self.backbone(x)
        # Extract native multi-scale routes: p4, p8, p16 (with early-return at target_stride)
        _, routes = self.neck(*features, return_routes=True, target_stride=self.output_stride)
        route_key = f"p{self.output_stride}"
        p_feat = routes[route_key]

        # Standard MICF/local path: preserve existing B1-B7 semantics.
        if self.finite_horizon is None:
            p_context = self.context_module(p_feat)
            return self._predict_from_context(p_context)

        # FH-CMICF path.
        k = self.finite_horizon
        B, _, H_o, W_o = p_feat.shape

        blocks, n_h, n_w = _partition_feature_blocks(p_feat, k)

        # Critical: context is applied AFTER partitioning.
        # Each KxK block therefore gets its own local prefix origin.
        block_context = self.context_module(blocks)

        # Extent-aware coordinates/area are also local because
        # _predict_from_context() sees tensors of spatial size KxK.
        c_blocks = self._predict_from_context(block_context)

        # Non-learned exact local-to-global composition.
        c_global = compose_batched_fh_cumulative(
            c_blocks,
            batch_size=B,
            n_h=n_h,
            n_w=n_w,
            k=k,
        )
        return c_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_field(x)

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        pad_multiple: int | None = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference method returning (pred_count, pred_map).

        For cumulative & integrated_local heads: pred_count = pred_c[..., -1, -1].
        For local head: pred_count = pred_y.sum().
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")

        _, _, h, w = x.shape
        if self.finite_horizon is not None:
            fh_multiple = self.output_stride * self.finite_horizon
            if pad_multiple is None:
                pad_multiple = fh_multiple
            else:
                pad_multiple = math.lcm(int(pad_multiple), int(fh_multiple))

        if pad_multiple is not None:
            pad_h = (pad_multiple - (h % pad_multiple)) % pad_multiple
            pad_w = (pad_multiple - (w % pad_multiple)) % pad_multiple
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        field = self.forward_field(x)
        out_h = math.ceil(h / self.output_stride)
        out_w = math.ceil(w / self.output_stride)
        field_valid = field[..., :out_h, :out_w]

        if self.head_type in {"cumulative", "integrated_local"}:
            count = field_valid[..., -1, -1]
        else:
            count = field_valid.sum(dim=(-3, -2, -1))
        if count.numel() == 1:
            count = count.squeeze()
        else:
            count = count.view(-1)
        return count, field_valid

    @torch.no_grad()
    def predict_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 256,
        halo: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tiled inference for the full-image regime (design doc sec.29-30).

        For head_type == 'local': tiles image into non-overlapping tiles,
        crops halo context, and assembles the global local count map.

        For head_type in {'cumulative', 'integrated_local'}: computes each
        tile's local cumulative field independently (optionally with `halo`
        pixels of surrounding context, cropped back out via
        discrete_mixed_difference before composition), then stitches them with
        compose_tiled_cumulative_field -- exact block-decomposed 2D prefix
        composition (design doc sec.30, generalized from a raster chain to a
        full tile grid).

        Caveat: exact composition of the per-tile *predictions*, not identical
        to one full-image forward pass -- each pixel only sees `halo` pixels of
        cross-tile context, not the network's full receptive field. Only B=1
        is supported.
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(x.shape)}")
        if x.shape[0] != 1:
            raise NotImplementedError("predict_tiled currently supports batch size 1 only")

        device = x.device
        _, _, H, W = x.shape
        s = self.output_stride
        out_h_full = math.ceil(H / s)
        out_w_full = math.ceil(W / s)

        req_multiple = s if self.finite_horizon is None else (s * self.finite_horizon)
        if tile_size % req_multiple != 0:
            raise ValueError(
                f"tile_size ({tile_size}) must be a multiple of {req_multiple} "
                f"(output_stride={s}" + (f", finite_horizon={self.finite_horizon})" if self.finite_horizon else ")")
            )
        if halo % req_multiple != 0:
            raise ValueError(
                f"halo ({halo}) must be a multiple of {req_multiple} "
                f"(output_stride={s}" + (f", finite_horizon={self.finite_horizon})" if self.finite_horizon else ")")
            )
        out_tile = tile_size // s

        n_tiles_h = math.ceil(H / tile_size)
        n_tiles_w = math.ceil(W / tile_size)
        padded_h = n_tiles_h * tile_size
        padded_w = n_tiles_w * tile_size
        x_pad = F.pad(x, (0, padded_w - W, 0, padded_h - H), mode="constant", value=0.0)

        if self.head_type == "local":
            y_global = torch.zeros((1, 1, n_tiles_h * out_tile, n_tiles_w * out_tile), device=device, dtype=x.dtype)
            for i in range(n_tiles_h):
                for j in range(n_tiles_w):
                    y0, y1 = i * tile_size, (i + 1) * tile_size
                    x0, x1 = j * tile_size, (j + 1) * tile_size
                    hy0, hy1 = max(0, y0 - halo), min(padded_h, y1 + halo)
                    hx0, hx1 = max(0, x0 - halo), min(padded_w, x1 + halo)
                    crop = x_pad[..., hy0:hy1, hx0:hx1]

                    field = self.forward_field(crop)
                    if halo > 0:
                        ry0 = (y0 - hy0) // s
                        rx0 = (x0 - hx0) // s
                        y_core = field[..., ry0: ry0 + out_tile, rx0: rx0 + out_tile]
                    else:
                        y_core = field[..., :out_tile, :out_tile]
                    y_global[..., i * out_tile:(i + 1) * out_tile, j * out_tile:(j + 1) * out_tile] = y_core

            field_valid = y_global[..., :out_h_full, :out_w_full]
            count = field_valid.sum(dim=(-1, -2, -3)).squeeze()
            return count, field_valid

        if self.head_type not in {"cumulative", "integrated_local"}:
            raise ValueError(f"Unsupported head_type for predict_tiled: {self.head_type}")

        c_local: list[list[torch.Tensor]] = [[None] * n_tiles_w for _ in range(n_tiles_h)]
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                y0, y1 = i * tile_size, (i + 1) * tile_size
                x0, x1 = j * tile_size, (j + 1) * tile_size
                hy0, hy1 = max(0, y0 - halo), min(padded_h, y1 + halo)
                hx0, hx1 = max(0, x0 - halo), min(padded_w, x1 + halo)
                crop = x_pad[..., hy0:hy1, hx0:hx1]

                field = self.forward_field(crop)
                if halo > 0:
                    y_full = discrete_mixed_difference(field)
                    ry0 = (y0 - hy0) // s
                    rx0 = (x0 - hx0) // s
                    y_core = y_full[..., ry0: ry0 + out_tile, rx0: rx0 + out_tile]
                    c_tile = torch.cumsum(torch.cumsum(y_core, dim=-2), dim=-1)
                else:
                    c_tile = field[..., :out_tile, :out_tile]
                c_local[i][j] = c_tile.squeeze(0).squeeze(0)

        c_global = compose_tiled_cumulative_field(c_local)
        field_valid = c_global[:out_h_full, :out_w_full].unsqueeze(0).unsqueeze(0)
        count = field_valid[..., -1, -1].squeeze()
        return count, field_valid
