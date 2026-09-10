from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .necks import ConvGNAct

# Data-driven prior initialization (RMR-v2/v3):
# Computed empirically from ShanghaiTech Part A training manifest:
# m0 = total_points / total_stride4_cells = 0.015763 count/cell.
# Softplus inverse: softplus(x) = log(exp(x)-1), softplus^{-1}(m0) = log(exp(m0)-1) ≈ -4.1422.
_M0_INIT: float = 0.015763
_FINE_HEAD_BIAS_INIT: float = math.log(math.exp(_M0_INIT) - 1.0)  # ≈ -4.1422


class FineMeasureHead(nn.Module):
    """Fine-grained density head with calibrated initial rate.

    The final Conv2d bias is initialized so that:
        softplus(bias) ≈ mu0 = 0.015763 count/cell
    i.e. bias ≈ log(exp(0.015763) - 1) ≈ -4.1422.
    Weights of the last conv are initialized to small values (std=0.01).

    When temp_softplus=True (RMR-v7+), the output activation uses a learnable
    temperature τ (initialized to 1):
        output = τ * softplus(z / τ)
    This prevents saturation at high density, fixing the systematic negative bias
    observed in RMR-v6 (Bias=-14.72). τ is clamped to ≥ 0.1 during forward pass.
    """

    def __init__(
        self,
        width: int = 32,
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        temp_softplus: bool = False,
    ):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
            nn.Conv2d(width, 1, 1),
        )
        final_conv: nn.Conv2d = self.body[-1]  # type: ignore[assignment]
        nn.init.normal_(final_conv.weight, std=0.01)
        nn.init.constant_(final_conv.bias, init_bias)  # type: ignore[arg-type]

        self.temp_softplus = bool(temp_softplus)
        if self.temp_softplus:
            # Learnable temperature τ; initialized to 1.0 (identical to vanilla softplus)
            self.tau = nn.Parameter(torch.ones(1))

    def forward(self, f: tuple[torch.Tensor, ...] | torch.Tensor) -> torch.Tensor:
        if isinstance(f, tuple):
            f = f[0]
        z = self.body(f)
        if self.temp_softplus:
            tau = self.tau.clamp_min(0.1)
            return tau * F.softplus(z / tau)
        return F.softplus(z)
