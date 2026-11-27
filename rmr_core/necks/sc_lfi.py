from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual
from .pafpn import ConvexFastNormalizedFusion, StripMultiKernelContext, _match_size


class SCLFINeck(nn.Module):
    """Strip-Context Convex Laplacian Frequency-Inheritance FPN Neck (SC-LFI).

    Combines:
      1. Hole-free StripMultiKernelContext (3x3, 1x5, 5x1) on L16.
      2. Direct 3-level convex context fusion into P4 without slow bottom-up loops.
      3. Norm-free linear Laplacian high-frequency residual at L4 with WD-exempt gamma_hf.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int] = (16, 32, 48),
        width: int = 32,
        decomp_mode: str = "laplacian",
        pad_aware_laplacian: bool = True,
        hf_act_mode: str = "linear",
    ) -> None:
        super().__init__()
        c4, c8, c16 = (int(c) for c in in_channels)
        w = int(width)
        self.width = w
        self.decomp_mode = str(decomp_mode).lower().strip()
        self.pad_aware_laplacian = bool(pad_aware_laplacian)
        self.hf_act_mode = str(hf_act_mode).lower().strip()

        if self.decomp_mode not in ("laplacian", "direct_dw", "none"):
            raise ValueError(f"Unsupported SC-LFI decomp_mode: '{self.decomp_mode}'")
        if self.hf_act_mode not in ("linear", "tanh", "silu"):
            raise ValueError(f"Unsupported SC-LFI hf_act_mode: '{self.hf_act_mode}'")

        self.lat4 = ConvGNAct(c4, w, 1)
        self.lat8 = ConvGNAct(c8, w, 1)
        self.lat16 = ConvGNAct(c16, w, 1)

        self.ctx16 = StripMultiKernelContext(w)
        self.fuse8 = ConvexFastNormalizedFusion(num_inputs=2)
        self.ref8 = DSResidual(w)
        self.fuse4 = ConvexFastNormalizedFusion(num_inputs=3)
        self.ref4 = DSResidual(w)

        if self.decomp_mode != "none":
            self.hf_enhance: nn.Module | None = nn.Conv2d(w, w, kernel_size=3, padding=1, groups=w, bias=False)
            self.gamma_hf: nn.Parameter | None = nn.Parameter(torch.full((1, w, 1, 1), 0.5))
        else:
            self.hf_enhance = None
            self.gamma_hf = None

    def _act_hf(self, x: torch.Tensor) -> torch.Tensor:
        if self.hf_act_mode == "tanh":
            return torch.tanh(x)
        if self.hf_act_mode == "silu":
            return F.silu(x)
        return x

    def forward(
        self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        l4 = self.lat4(c4)
        l8 = self.lat8(c8)
        l16 = self.lat16(c16)

        p16 = self.ctx16(l16)
        p8 = self.ref8(self.fuse8([l8, _match_size(p16, l8)]))
        p4_ctx = self.fuse4([l4, _match_size(p8, l4), _match_size(p16, l4)])

        if self.decomp_mode == "laplacian" and self.hf_enhance is not None and self.gamma_hf is not None:
            l4_lf = F.avg_pool2d(
                l4, kernel_size=3, stride=1, padding=1, count_include_pad=(not self.pad_aware_laplacian)
            )
            f4 = p4_ctx + self.gamma_hf * self._act_hf(self.hf_enhance(l4 - l4_lf))
        elif self.decomp_mode == "direct_dw" and self.hf_enhance is not None and self.gamma_hf is not None:
            f4 = p4_ctx + self.gamma_hf * self._act_hf(self.hf_enhance(l4))
        else:
            f4 = p4_ctx

        p4 = self.ref4(f4)
        return p4, p8, p16
