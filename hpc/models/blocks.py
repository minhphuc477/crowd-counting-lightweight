from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_group_norm(channels: int) -> nn.GroupNorm:
    """Return a GroupNorm with up to 8 groups (falls back for small channel counts)."""
    num_groups = min(8, channels)
    while channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, channels)


class ConvGNAct(nn.Module):
    """Conv (1×1 or 3×3) + GroupNorm + SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        groups: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.norm = make_group_norm(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DSResidual(nn.Module):
    """Depthwise-Separable Residual Refinement Block.

    DW 3×3 → GN → SiLU → PW 1×1 → GN → Residual Add → SiLU.
    """

    def __init__(self, c: int = 48):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c, bias=False)
        self.n1 = make_group_norm(c)
        self.pw = nn.Conv2d(c, c, kernel_size=1, bias=False)
        self.n2 = make_group_norm(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.n1(self.dw(x)))
        y = self.n2(self.pw(y))
        return self.act(x + y)


class RepDWBlock(nn.Module):
    """Structural Re-parameterization Depthwise Refinement Block.
    
    Training: Multi-branch (3×3 DW + 1×1 DW + Identity with BatchNorms).
    Deployment: Fused algebraically into a single standard 3×3 DW Conv (0 extra params/FLOPs).
    """

    def __init__(self, channels: int = 32, act: bool = True):
        super().__init__()
        self.channels = channels
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.rbr_identity = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
        self.is_deployed = False

        # Zero-init secondary branches for safe continuation from pre-trained weights
        nn.init.zeros_(self.rbr_1x1[1].weight)
        nn.init.zeros_(self.rbr_1x1[1].bias)
        nn.init.zeros_(self.rbr_identity.weight)
        nn.init.zeros_(self.rbr_identity.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_deployed:
            return self.act(self.rbr_reparam(x))
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x))

    def _fuse_bn(self, conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_fused = w * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / var_sqrt
        return w_fused, b_fused

    def switch_to_deploy(self):
        """Fuse multi-branch convolutions into a single 3x3 depthwise convolution."""
        if self.is_deployed:
            return
        w_3, b_3 = self._fuse_bn(self.rbr_dense[0], self.rbr_dense[1])
        w_1, b_1 = self._fuse_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        w_1_padded = F.pad(w_1, (1, 1, 1, 1))

        ident_kernel = torch.zeros(self.channels, 1, 3, 3, device=w_3.device)
        ident_kernel[:, 0, 1, 1] = 1.0
        mean = self.rbr_identity.running_mean
        var_sqrt = torch.sqrt(self.rbr_identity.running_var + self.rbr_identity.eps)
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        w_id = ident_kernel * (gamma / var_sqrt).reshape(-1, 1, 1, 1)
        b_id = beta - mean * gamma / var_sqrt

        w_fused = w_3 + w_1_padded + w_id
        b_fused = b_3 + b_1 + b_id

        self.rbr_reparam = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels, bias=True).to(w_3.device)
        self.rbr_reparam.weight.data.copy_(w_fused)
        self.rbr_reparam.bias.data.copy_(b_fused)

        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.is_deployed = True


class MultiPoolContext(nn.Module):
    """Parameter-free coarse spatial mixer via averaged multi-scale pooling.

    Residual mixing prevents pure average-pool oversmoothing.
    Default kernels (3, 5, 7) expose very broad input-space context at /32
    while adding zero trainable parameters.
    """

    def __init__(self, kernels=(3, 5, 7), residual_mix: float = 0.5):
        super().__init__()
        self.kernels = tuple(int(k) for k in kernels)
        self.residual_mix = float(residual_mix)
        for k in self.kernels:
            if k <= 0 or k % 2 == 0:
                raise ValueError("pool kernels must be positive odd integers")
        if not 0.0 <= self.residual_mix <= 1.0:
            raise ValueError("residual_mix must be in [0, 1]")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.kernels:
            return x
        pooled = [
            F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
            for k in self.kernels
        ]
        p = torch.stack(pooled, dim=0).mean(dim=0)
        return x + self.residual_mix * (p - x)


class SimAM(nn.Module):
    """Parameter-free 3D attention (SimAM, ICML 2021).

    lambda_e is a fixed constant, NOT an nn.Parameter.
    Verified: sum(p.numel() for p in SimAM().parameters()) == 0.
    """

    def __init__(self, lambda_e: float = 1e-4):
        super().__init__()
        self.lambda_e = float(lambda_e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        n = h * w - 1
        if n <= 0:
            return x
        d = (x - x.mean(dim=(2, 3), keepdim=True)).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / float(n)
        e_inv = d / (4.0 * (v + self.lambda_e)) + 0.5
        return x * torch.sigmoid(e_inv)


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
        self.norm = make_group_norm(channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))
