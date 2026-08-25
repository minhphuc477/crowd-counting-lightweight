import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    """1x1 or 3x3 Conv + GroupNorm + SiLU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # GroupNorm: standard 8 groups (or gcd if channels < 8)
        num_groups = min(8, out_channels)
        while out_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DepthwiseDilated(nn.Module):
    """Depthwise 3x3 Conv with configurable dilation + GroupNorm + SiLU."""
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
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DSResidual(nn.Module):
    """Depthwise-Separable Residual Refinement Block.
    DW 3x3 -> GN -> SiLU -> PW 1x1 -> GN -> Residual Add -> SiLU.
    """
    def __init__(self, c: int = 32):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        num_groups = min(8, c)
        while c % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.n1 = nn.GroupNorm(num_groups, c)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.n2 = nn.GroupNorm(num_groups, c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.dw(x)))
        y = self.n2(self.pw(y))
        return self.act(x + y)
