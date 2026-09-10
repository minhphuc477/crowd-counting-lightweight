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

