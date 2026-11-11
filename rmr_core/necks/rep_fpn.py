from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DepthwiseDilated


class RepDWBlock7x7(nn.Module):
    """Structural Re-parameterization Depthwise Refinement Block (default 7x7).

    Training: Multi-branch (kxk DW + 3x3 DW + Identity with BatchNorms).
    Note: Replaces a scale-invariant 1x1 DW+BN branch (which receives zero gradient under
    per-channel BN) with a 3x3 DW+BN branch so every parameter contributes non-zero gradients.
    Deployment: Fused algebraically into a single standard kxk DW Conv (0 extra params/FLOPs).
    """

    def __init__(self, channels: int = 32, act: bool = True, kernel_size: int = 7):
        super().__init__()
        k = int(kernel_size)
        if k < 3 or k % 2 == 0:
            raise ValueError(f"RepDWBlock7x7 kernel_size must be odd >= 3, got {kernel_size}")
        self.channels = channels
        self.kernel_size = k
        pad_k = k // 2

        self.rbr_dense = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k, padding=pad_k, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=0.01),
        )
        self.rbr_3x3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels, momentum=0.01),
        )
        self.rbr_identity = nn.BatchNorm2d(channels, momentum=0.01)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.is_deployed = False

        # Zero-init identity skip for stable continuation while keeping rbr_dense/rbr_3x3 active
        nn.init.zeros_(self.rbr_identity.weight)
        nn.init.zeros_(self.rbr_identity.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.act(self.rbr_reparam(x))
        return self.act(self.rbr_dense(x) + self.rbr_3x3(x) + self.rbr_identity(x))

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight.float()
        mean = bn.running_mean.float()
        var_sqrt = torch.sqrt(bn.running_var.float() + bn.eps)
        gamma = bn.weight.float()
        beta = bn.bias.float()
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / var_sqrt
        return w_fused, b_fused

    def switch_to_deploy(self) -> None:
        """Fuse multi-branch convolutions into a single kxk depthwise convolution."""
        if self.is_deployed:
            return
        k = self.kernel_size
        pad_k = k // 2
        orig_dtype = self.rbr_dense[0].weight.dtype
        w_dense, b_dense = self._fuse_bn(self.rbr_dense[0], self.rbr_dense[1])
        w_3, b_3 = self._fuse_bn(self.rbr_3x3[0], self.rbr_3x3[1])
        pad_3 = (k - 3) // 2
        w_3_padded = F.pad(w_3, (pad_3, pad_3, pad_3, pad_3)) if pad_3 > 0 else w_3

        ident_kernel = torch.zeros(self.channels, 1, k, k, device=w_dense.device, dtype=torch.float32)
        ident_kernel[:, 0, pad_k, pad_k] = 1.0
        mean = self.rbr_identity.running_mean.float()
        var_sqrt = torch.sqrt(self.rbr_identity.running_var.float() + self.rbr_identity.eps)
        gamma = self.rbr_identity.weight.float()
        beta = self.rbr_identity.bias.float()
        w_id = ident_kernel * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_id = beta - mean * gamma / var_sqrt

        w_fused = (w_dense + w_3_padded + w_id).to(dtype=orig_dtype)
        b_fused = (b_dense + b_3 + b_id).to(dtype=orig_dtype)

        self.rbr_reparam = nn.Conv2d(
            self.channels, self.channels, kernel_size=k, padding=pad_k, groups=self.channels, bias=True
        ).to(device=w_dense.device, dtype=orig_dtype)
        self.rbr_reparam.weight.data.copy_(w_fused)
        self.rbr_reparam.bias.data.copy_(b_fused)

        del self.rbr_dense
        del self.rbr_3x3
        del self.rbr_identity
        self.is_deployed = True


class _ChannelMixPW(nn.Module):
    """Cross-channel 1x1 grouped pointwise mixer with GroupNorm, SiLU, and channel shuffle."""

    def __init__(self, channels: int = 32, groups: int = 4) -> None:
        super().__init__()
        g = groups if (channels % groups == 0 and channels >= groups) else 1
        self.groups = g
        ng = 8 if channels % 8 == 0 else (4 if channels % 4 == 0 else 1)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, groups=g, bias=False)
        self.norm = nn.GroupNorm(ng, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm(self.pw(x)))
        if self.groups > 1:
            b, c, h, w = y.shape
            y = y.view(b, self.groups, c // self.groups, h, w).transpose(1, 2).contiguous().view(b, c, h, w)
        return y


class RepWeightedFPNNeck(nn.Module):
    """FPN neck with Softmax-normalized weighted fusion, RepDW spatial refinement, and 1x1 PW mixing."""

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

        self.ref16 = RepDWBlock7x7(width, kernel_size=5)
        self.pw16 = _ChannelMixPW(width)
        self.ref8 = RepDWBlock7x7(width, kernel_size=5)
        self.pw8 = _ChannelMixPW(width)
        self.ref4 = RepDWBlock7x7(width, kernel_size=5)
        self.pw4 = _ChannelMixPW(width)

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        # Scale 16 fusion
        w16 = F.softmax(self.weight_16, dim=0)
        ctx_sum = sum(ctx(l16) for ctx in self.context_blocks)
        p16 = self.pw16(self.ref16(w16[0] * l16 + w16[1] * ctx_sum))

        # Scale 8 fusion
        w8 = F.softmax(self.weight_8, dim=0)
        up16_to_8 = F.interpolate(
            p16, size=l8.shape[-2:], mode="bilinear", align_corners=False
        )
        p8 = self.pw8(self.ref8(w8[0] * l8 + w8[1] * up16_to_8))

        # Scale 4 fusion
        w4 = F.softmax(self.weight_4, dim=0)
        up8_to_4 = F.interpolate(
            p8, size=l4.shape[-2:], mode="bilinear", align_corners=False
        )
        p4 = self.pw4(self.ref4(w4[0] * l4 + w4[1] * up8_to_4))
        return p4, p8, p16

    def switch_to_deploy(self) -> None:
        self.ref16.switch_to_deploy()
        self.ref8.switch_to_deploy()
        self.ref4.switch_to_deploy()
