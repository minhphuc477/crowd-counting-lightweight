"""Non-ASPP bidirectional and partial-convolution FPN necks under tight sub-104k budgets.

Designed to avoid three failure modes diagnosed in ASPP-Lite / legacy necks:
1. High-dilation gridding and boundary zero-padding on coarse Stride-16 grids (d=8 hits 75% border at 512px).
2. Non-translation-equivariant Global Average Pooling (GAP) crop drift between 512px crops and full-res eval.
3. Unnormalized top-down addition (l4 + up(p8)) allowing coarse features to attenuate fine C4 spatial contrast,
   as well as BatchNorm2d train/eval statistics mismatch inside GroupNorm pipelines.

Provides:
- ConvexFastNormalizedFusion: Non-negative convex weights summing to <= 1 in FP32 before AMP cast.
- StripMultiKernelContext: Dense hole-free (3x3, 1x5, 5x1) strip depthwise context with GroupNorm.
- PConvBlock: FasterNet-style partial 3x3 convolution on 1/4 channels + GroupNorm + 1x1 mix + residual.
- PAFPNBiNormNeck (neck_type="pafpn_binorm"): Top-down + bottom-up bidirectional FPN with convex fusion
  and direct C4 fine-anchor skip into P4_out.
- HighResPConvFPNNeck (neck_type="pconv_fpn"): High-resolution partial-convolution FPN shifting spatial
  capacity to Stride-8/Stride-4 with bidirectional convex fusion under <= 10.8k neck parameters.
"""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, _gn


class ConvexFastNormalizedFusion(nn.Module):
    """Multi-input fast normalized convex fusion (sum of weights < 1, strictly non-negative)."""

    def __init__(self, num_inputs: int = 2, eps: float = 1e-4) -> None:
        super().__init__()
        if int(num_inputs) < 2:
            raise ValueError(f"num_inputs must be >= 2, got {num_inputs}")
        self.num_inputs = int(num_inputs)
        self.eps = float(eps)
        self.weights = nn.Parameter(torch.ones(self.num_inputs, dtype=torch.float32))

    def normalized_weights(self) -> torch.Tensor:
        """Return convex weights computed stably in FP32."""
        w = F.relu(self.weights.float())
        return w / (w.sum() + self.eps)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        """Fuse equal-shape tensors via convex weighted sum."""
        if len(inputs) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} tensors, got {len(inputs)}")
        w = self.normalized_weights().to(device=inputs[0].device, dtype=inputs[0].dtype)
        out = w[0] * inputs[0]
        for i in range(1, self.num_inputs):
            out = out + w[i] * inputs[i]
        return out


class StripMultiKernelContext(nn.Module):
    """Dense hole-free multi-scale strip depthwise context (3x3, 1x5, 5x1) at coarse stride."""

    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = int(width)
        self.dw3 = nn.Conv2d(w, w, kernel_size=3, padding=1, groups=w, bias=False)
        self.dw1x5 = nn.Conv2d(w, w, kernel_size=(1, 5), padding=(0, 2), groups=w, bias=False)
        self.dw5x1 = nn.Conv2d(w, w, kernel_size=(5, 1), padding=(2, 0), groups=w, bias=False)
        self.branch_fuse = ConvexFastNormalizedFusion(num_inputs=3)
        self.post = nn.Sequential(
            _gn(w),
            nn.SiLU(inplace=True),
            ConvGNAct(w, w, k=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convex-fused isotropic/strip context with identity residual."""
        ctx = self.branch_fuse([self.dw3(x), self.dw1x5(x), self.dw5x1(x)])
        return x + self.post(ctx)


class PConvBlock(nn.Module):
    """GroupNorm-pure partial 3x3 spatial convolution on 1/4 channels + 1x1 mixing residual."""

    def __init__(self, width: int = 32, split_div: int = 4, k: int = 3) -> None:
        super().__init__()
        w = int(width)
        self.width = w
        self.c_p = max(1, w // int(split_div))
        self.pconv = nn.Conv2d(self.c_p, self.c_p, kernel_size=k, padding=k // 2, bias=False)
        self.mix = nn.Sequential(
            _gn(w),
            nn.SiLU(inplace=True),
            ConvGNAct(w, w, k=1, act=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve partial active subspace while preserving remaining identity channels."""
        x1, x2 = torch.split(x, [self.c_p, self.width - self.c_p], dim=1)
        mixed = torch.cat([self.pconv(x1), x2], dim=1)
        return x + self.mix(mixed)


def _make_dw_downsample(width: int) -> nn.Sequential:
    """Stride-2 3x3 depthwise downsampler with GroupNorm (no BatchNorm train/eval drift)."""
    w = int(width)
    return nn.Sequential(
        nn.Conv2d(w, w, kernel_size=3, stride=2, padding=1, groups=w, bias=False),
        _gn(w),
        nn.SiLU(inplace=True),
    )


def _match_size(src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Align spatial dimensions cleanly for arbitrary odd/non-power-of-2 resolutions."""
    if src.shape[-2:] == target.shape[-2:]:
        return src
    return F.interpolate(src, size=target.shape[-2:], mode="bilinear", align_corners=False)


class _BidirectionalConvexFPNBase(nn.Module):
    """Shared top-down + bottom-up convex FPN skeleton avoiding code duplication."""

    def __init__(
        self,
        in_channels: tuple[int, ...],
        width: int,
        ctx16: nn.Module,
        td8_refine: nn.Module,
        td4_refine: nn.Module,
        bu8_refine: nn.Module,
        bu16_refine: nn.Module,
        p4_refine: nn.Module,
    ) -> None:
        super().__init__()
        c4, c8, c16 = (int(c) for c in in_channels)
        w = int(width)

        self.lat4 = ConvGNAct(c4, w, k=1)
        self.lat8 = ConvGNAct(c8, w, k=1)
        self.lat16 = ConvGNAct(c16, w, k=1)

        self.ctx16 = ctx16
        self.fuse_td8 = ConvexFastNormalizedFusion(num_inputs=2)
        self.td8_refine = td8_refine
        self.fuse_td4 = ConvexFastNormalizedFusion(num_inputs=2)
        self.td4_refine = td4_refine

        self.down4_to_8 = _make_dw_downsample(w)
        self.fuse_bu8 = ConvexFastNormalizedFusion(num_inputs=3)
        self.bu8_refine = bu8_refine

        self.down8_to_16 = _make_dw_downsample(w)
        self.fuse_bu16 = ConvexFastNormalizedFusion(num_inputs=3)
        self.bu16_refine = bu16_refine

        self.fuse_p4 = ConvexFastNormalizedFusion(num_inputs=3)
        self.p4_refine = p4_refine

    def forward(
        self,
        c4: torch.Tensor,
        c8: torch.Tensor,
        c16: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute bidirectional convex-fused pyramid (p4, p8, p16) without GAP or ASPP holes."""
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        p16_ctx = self.ctx16(l16)

        p8_td = self.td8_refine(self.fuse_td8([l8, _match_size(p16_ctx, l8)]))
        p4_td = self.td4_refine(self.fuse_td4([l4, _match_size(p8_td, l4)]))

        down4 = _match_size(self.down4_to_8(p4_td), l8)
        p8_out = self.bu8_refine(self.fuse_bu8([l8, p8_td, down4]))

        down8 = _match_size(self.down8_to_16(p8_out), l16)
        p16_out = self.bu16_refine(self.fuse_bu16([l16, p16_ctx, down8]))

        # Direct C4 fine skip + top-down + bottom-up consensus prevents high-freq contrast damping
        p4_out = self.p4_refine(self.fuse_p4([l4, p4_td, _match_size(p8_out, l4)]))
        return p4_out, p8_out, p16_out


class PAFPNBiNormNeck(_BidirectionalConvexFPNBase):
    """Bidirectional Strip-Context PAFPN with GroupNorm and Convex Fast-Normalized Fusion.

    Removes dilated ASPP gridding/border artifacts and non-equivariant GAP, while adding a
    bottom-up N4->N8->N16 pathway plus direct C4 fine-edge re-injection into P4.
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (16, 32, 48),
        width: int = 32,
    ) -> None:
        w = int(width)
        td4_dw = _ResidualWrap(
            nn.Sequential(
                nn.Conv2d(w, w, kernel_size=3, padding=1, groups=w, bias=False),
                _gn(w),
            )
        )
        super().__init__(
            in_channels=in_channels,
            width=w,
            ctx16=StripMultiKernelContext(w),
            td8_refine=DSResidual(w),
            td4_refine=td4_dw,
            bu8_refine=DSResidual(w),
            bu16_refine=nn.Identity(),
            p4_refine=DSResidual(w),
        )


class HighResPConvFPNNeck(_BidirectionalConvexFPNBase):
    """High-Resolution Partial-Convolution FPN concentrating spatial capacity at C8/C4.

    Uses dense 5x5 depthwise refinement at coarse Stride-16 and FasterNet-style PConvBlocks
    on Stride-8/Stride-4 with bidirectional convex fusion (<= 11.2k neck parameters at width=32).
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (16, 32, 48),
        width: int = 32,
        split_div: int = 4,
    ) -> None:
        w = int(width)
        ctx16 = nn.Sequential(
            nn.Conv2d(w, w, kernel_size=5, padding=2, groups=w, bias=False),
            _gn(w),
            nn.SiLU(inplace=True),
            ConvGNAct(w, w, k=1, act=False),
        )
        # Keep residual around L16 context cleanly
        super().__init__(
            in_channels=in_channels,
            width=w,
            ctx16=_ResidualWrap(ctx16),
            td8_refine=PConvBlock(w, split_div=split_div, k=3),
            td4_refine=nn.Identity(),
            bu8_refine=PConvBlock(w, split_div=split_div, k=3),
            bu16_refine=nn.Identity(),
            p4_refine=PConvBlock(w, split_div=split_div, k=3),
        )


class _ResidualWrap(nn.Module):
    """Apply residual connection x + fn(x)."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)
