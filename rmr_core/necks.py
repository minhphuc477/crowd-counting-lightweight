from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        stride: int = 1,
        groups: int = 1,
        act: bool = True,
    ):
        pad = k // 2
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=False),
            _gn(cout),
        ]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class TinyIR(nn.Module):
    """Small inverted residual block."""

    def __init__(self, cin: int, cout: int, stride: int = 1, expand: float = 2.0):
        super().__init__()
        mid = max(cin, int(round(cin * expand)))
        self.use_res = stride == 1 and cin == cout
        self.expand = ConvGNAct(cin, mid, k=1) if mid != cin else nn.Identity()
        self.dw = ConvGNAct(mid, mid, k=3, stride=stride, groups=mid)
        self.proj = nn.Sequential(nn.Conv2d(mid, cout, 1, bias=False), _gn(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj(self.dw(self.expand(x)))
        return x + y if self.use_res else y


class DSResidual(nn.Module):
    """Depthwise-Separable Residual Refinement Block.

    DW 3x3 -> GN -> SiLU -> PW 1x1 -> GN -> Residual Add -> SiLU.
    """

    def __init__(self, c: int = 32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.n1 = _gn(c)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.n2 = _gn(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.dw(x)))
        y = self.n2(self.pw(y))
        return self.act(x + y)


class DepthwiseDilated(nn.Module):
    """Depthwise 3x3 convolution with configurable dilation."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=False,
        )
        self.norm = _gn(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AdditiveFusion(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.p4 = ConvGNAct(24, width, 1)
        self.p8 = ConvGNAct(40, width, 1)
        self.p16 = ConvGNAct(64, width, 1)
        self.out = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
        )

    def forward(self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        size = c4.shape[-2:]
        p4 = self.p4(c4)
        p8 = self.p8(c8)
        p16 = self.p16(c16)
        p = p4 + F.interpolate(p8, size=size, mode="bilinear", align_corners=False)
        p = p + F.interpolate(p16, size=size, mode="bilinear", align_corners=False)
        return self.out(p), p8, p16


class AdditiveFPNNeck(nn.Module):
    """Additive depthwise-separable FPN neck fusing reductions 4, 8, 16 into (P4, P8, P16)."""

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        context_dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks)
        p16 = self.ref16(l16 + ctx_sum)

        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)
        return p4, p8, p16


class RepDWBlock7x7(nn.Module):
    """Structural Re-parameterization 7x7 Depthwise Refinement Block.
    
    Training: Multi-branch (7x7 DW + 1x1 DW + Identity with BatchNorms).
    Deployment: Fused algebraically into a single standard 7x7 DW Conv (0 extra params/FLOPs).
    """

    def __init__(self, channels: int = 32, act: bool = True):
        super().__init__()
        self.channels = channels
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_identity = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.is_deployed = False

        # Zero-init secondary branches for safe continuation
        nn.init.zeros_(self.rbr_1x1[1].weight)
        nn.init.zeros_(self.rbr_1x1[1].bias)
        nn.init.zeros_(self.rbr_identity.weight)
        nn.init.zeros_(self.rbr_identity.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.act(self.rbr_reparam(x))
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / var_sqrt
        return w_fused, b_fused

    def switch_to_deploy(self) -> None:
        """Fuse multi-branch convolutions into a single 7x7 depthwise convolution."""
        if self.is_deployed:
            return
        w_7, b_7 = self._fuse_bn(self.rbr_dense[0], self.rbr_dense[1])
        w_1, b_1 = self._fuse_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        # Pad 1x1 weight to 7x7: pad (3, 3, 3, 3)
        w_1_padded = F.pad(w_1, (3, 3, 3, 3))

        ident_kernel = torch.zeros(self.channels, 1, 7, 7, device=w_7.device, dtype=w_7.dtype)
        ident_kernel[:, 0, 3, 3] = 1.0
        mean = self.rbr_identity.running_mean
        var_sqrt = torch.sqrt(self.rbr_identity.running_var + self.rbr_identity.eps)
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        w_id = ident_kernel * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_id = beta - mean * gamma / var_sqrt

        w_fused = w_7 + w_1_padded + w_id
        b_fused = b_7 + b_1 + b_id

        self.rbr_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=7, padding=3, groups=self.channels, bias=True
        ).to(device=w_7.device, dtype=w_7.dtype)
        self.rbr_reparam.weight.data.copy_(w_fused)
        self.rbr_reparam.bias.data.copy_(b_fused)

        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.is_deployed = True


class RepWeightedFPNNeck(nn.Module):
    """FPN neck with Softmax-normalized weighted fusion and RepDWBlock (7x7 depthwise)."""

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        context_dilations: tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        self.context_blocks = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in context_dilations
        ])

        # Learnable fusion weights: 1 scalar per edge, softmax-normalized
        # Zero-initialized -> equal initial weighting [0.5, 0.5]
        self.weight_16 = nn.Parameter(torch.zeros(2))
        self.weight_8 = nn.Parameter(torch.zeros(2))
        self.weight_4 = nn.Parameter(torch.zeros(2))

        self.ref16 = RepDWBlock7x7(width)
        self.ref8 = RepDWBlock7x7(width)
        self.ref4 = RepDWBlock7x7(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # Scale 16 fusion
        w16 = F.softmax(self.weight_16, dim=0)
        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks)
        p16 = self.ref16(w16[0] * l16 + w16[1] * ctx_sum)

        # Scale 8 fusion
        w8 = F.softmax(self.weight_8, dim=0)
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(w8[0] * l8 + w8[1] * up16_to_8)

        # Scale 4 fusion
        w4 = F.softmax(self.weight_4, dim=0)
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(w4[0] * l4 + w4[1] * up8_to_4)
        return p4, p8, p16

    def switch_to_deploy(self) -> None:
        self.ref16.switch_to_deploy()
        self.ref8.switch_to_deploy()
        self.ref4.switch_to_deploy()


class ASPPLiteFPNNeck(nn.Module):
    """Static Additive FPN neck with a true ASPP-lite module on the P16 feature map.

    Architecture:
        - Lateral projections: C4/C8/C16 → width channels via 1×1 Conv+GN+SiLU.
        - ASPP-lite on L16 (four parallel branches, all static):
            • Branch 0: DW 3×3, dilation=1  (local context)
            • Branch 1: DW 3×3, dilation=3  (medium context)
            • Branch 2: DW 3×3, dilation=6  (wide context, RF=13 cells × stride16 = 208 px)
            • Branch 3: Global Average Pooling → Linear (width→width) → broadcast
          All four branches are summed (no learnable fusion weights) then projected by
          a 1×1 Conv+GN back to ``width`` channels before the DSResidual refinement.
        - Top-down FPN (static additive):
            P16 → upsample + L8 lateral → DSResidual → P8
            P8  → upsample + L4 lateral → DSResidual → P4
        - Returns (P4, P8, P16) matching the AdditiveFPNNeck interface exactly.

    Constraint:
        NO dynamic weights, NO softmax, NO attention. Fusion is strictly 1:1 addition.
        This avoids the "double non-stationarity" that collapsed RMR-v5 (MAE 104.24).
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        aspp_dilations: tuple[int, ...] = (1, 3, 6),
        use_aspp_gap: bool = True,
    ):
        super().__init__()
        c4, c8, c16 = in_channels
        self.width = width
        self.use_aspp_gap = bool(use_aspp_gap)

        # Lateral 1×1 projections
        self.lat4 = ConvGNAct(c4, width, 1)
        self.lat8 = ConvGNAct(c8, width, 1)
        self.lat16 = ConvGNAct(c16, width, 1)

        # ASPP-lite on L16
        # Dilated depthwise branches (output: width channels each)
        self.aspp_dw = nn.ModuleList([
            DepthwiseDilated(width, dilation=d) for d in aspp_dilations
        ])
        # Global Average Pooling branch: pool → Linear → broadcast
        # Uses a small bottleneck (width → width) to keep params low.
        if self.use_aspp_gap:
            self.aspp_gap: nn.Sequential | None = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),           # (B, width, 1, 1)
                nn.Flatten(1),                      # (B, width)
                nn.Linear(width, width, bias=True), # global context vector
                nn.SiLU(inplace=False),
            )
        else:
            self.aspp_gap = None
        # After summing all (len(aspp_dilations) + 1) branches, project back to width.
        # num_branches = len(aspp_dilations) + 1 (GAP) — but since every branch already
        # outputs `width` channels and we sum (not concat), no projection is needed
        # for channel count. We add a 1×1 PW conv purely as a mixing layer.
        self.aspp_proj = ConvGNAct(width, width, k=1)

        # Top-down FPN refinement blocks (Static DSResidual)
        self.ref16 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref4 = DSResidual(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # --- ASPP-lite on L16 ---
        # Dilated depthwise branches
        aspp_out = sum(branch(l16) for branch in self.aspp_dw)  # (B, width, H16, W16)

        # GAP branch: project global vector and broadcast back to spatial dims
        if self.aspp_gap is not None:
            gap_vec = self.aspp_gap(l16)                            # (B, width)
            gap_broadcast = gap_vec.unsqueeze(-1).unsqueeze(-1)     # (B, width, 1, 1)
            aspp_out = aspp_out + gap_broadcast                     # broadcast add

        # Mix: add residual from l16 then project
        p16_pre = self.aspp_proj(l16 + aspp_out)
        p16 = self.ref16(p16_pre)

        # --- Top-down FPN ---
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.ref8(l8 + up16_to_8)

        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.ref4(l4 + up8_to_4)

        return p4, p8, p16


class CoordinateAttention(nn.Module):
    """Coordinate Attention for density-map feature refinement (RMR-v8 Stage 3).

    Applies spatial-aware channel attention by separately encoding horizontal
    and vertical spatial context via strip pooling, then projecting to per-axis
    attention maps gating the input features element-wise.

    Architecture (width=32, reduction=4):
        1. H-pool: AdaptiveAvgPool2d((None, 1)) -> [B, C, H, 1]
        2. V-pool: AdaptiveAvgPool2d((1, None)) -> [B, C, 1, W]
        3. Concat on spatial dim (after transpose) -> [B, C, H+W, 1]
        4. Shared reduction conv: C -> C//reduction via 1x1 Conv + GN + SiLU
        5. Split into H-half and V-half
        6. Two parallel 1x1 Convs (C//reduction -> C) + Sigmoid -> attention gates
        7. Gate: out = x * gate_h.expand_as(x) * gate_v.expand_as(x)

    Parameter count for width=32, reduction=4 (8 mid channels):
        Shared conv: 32 * 8 * 1 + 8 (bias) = 264 + 8 = 272 (but bias=False here)
          -> Conv2d(32, 8, 1, bias=False): 256 params
          -> GroupNorm(1, 8): 16 params (weight + bias)
        H-attention conv: Conv2d(8, 32, 1, bias=False): 256 params
        V-attention conv: Conv2d(8, 32, 1, bias=False): 256 params
        Total: 256 + 16 + 256 + 256 = 784 params

    Note: GroupNorm instead of BatchNorm — BatchNorm is incompatible with
    batch_size=1 inference on variable-resolution images (eval mode).
    GroupNorm(1, channels) is equivalent to LayerNorm over spatial dims and
    works correctly at any batch size.
    """

    def __init__(self, channels: int = 32, reduction: int = 4) -> None:
        super().__init__()
        if channels % reduction != 0:
            raise ValueError(
                f"CoordinateAttention: channels ({channels}) must be divisible by reduction ({reduction})"
            )
        mid = channels // reduction
        self.channels = channels
        self.mid = mid

        # Shared projection: (concat of H-pool and V-pool) -> mid channels
        self.shared_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.GroupNorm(1, mid),  # LayerNorm-equivalent, works at batch_size=1
            nn.SiLU(inplace=True),
        )

        # Per-axis attention projections: mid -> channels
        self.h_conv = nn.Conv2d(mid, channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(mid, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Strip pooling: pool along each axis
        x_h = F.adaptive_avg_pool2d(x, (H, 1))   # [B, C, H, 1]
        x_v = F.adaptive_avg_pool2d(x, (1, W))   # [B, C, 1, W]

        # Transpose x_v to match H-dimension for shared projection
        x_v_t = x_v.permute(0, 1, 3, 2)          # [B, C, W, 1]

        # Concatenate along spatial dimension and project
        x_cat = torch.cat([x_h, x_v_t], dim=2)   # [B, C, H+W, 1]
        z = self.shared_conv(x_cat)                # [B, mid, H+W, 1]

        # Split back into H and W halves
        z_h, z_v = z[:, :, :H, :], z[:, :, H:, :]  # [B, mid, H, 1], [B, mid, W, 1]

        # Per-axis attention gates
        gate_h = torch.sigmoid(self.h_conv(z_h))   # [B, C, H, 1]
        gate_v = torch.sigmoid(self.v_conv(z_v.permute(0, 1, 3, 2)))  # [B, C, 1, W]

        # Apply attention: broadcast H-gate and V-gate over spatial dims
        return x * gate_h * gate_v
