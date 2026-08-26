import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, MultiPoolContext, SimAM


class ScaleRoutedFusionNeck(nn.Module):
    """Four-scale feature fusion neck with per-location scale routing.

    Architecture (§7 of SR48 proposal):
      1. Four lateral 1×1 Conv+GN projections: C4/C8/C16/C32 → 48 ch.
      2. Independent DS-Residual refinements at each scale.
         At /32: zero-parameter multi-pool context before refinement.
      3. Router computed at /8 resolution:
           concat [↓R4, R8, ↑R16, ↑R32] → Conv1×1 192→4 → softmax (temperature τ).
      4. Routing weights upsample to /4.
      5. Weighted sum: F = Σ_s α_s(x,y) * R_s(x,y).
      6. SimAM (0 params) applied to fused features.
      Output: fused feature map at /4, 48 ch.

    Router initialized to zero weight/bias so initial routing is exactly uniform.
    """

    def __init__(
        self,
        in_channels=(24, 48, 96, 192),
        width: int = 48,
        route_temperature: float = 1.0,
        pool_kernels=(3, 5, 7),
        pool_residual_mix: float = 0.5,
        simam_lambda: float = 1e-4,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("Expected four backbone feature scales")
        self.width = int(width)
        self.route_temperature = float(route_temperature)

        c4, c8, c16, c32 = in_channels

        # Lateral 1×1 projections (bias=False per spec §4)
        self.lat4 = ConvGNAct(c4, width, kernel_size=1)
        self.lat8 = ConvGNAct(c8, width, kernel_size=1)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)
        self.lat32 = ConvGNAct(c32, width, kernel_size=1)

        # Independent scale refinements
        self.ref4 = DSResidual(width)
        self.ref8 = DSResidual(width)
        self.ref16 = DSResidual(width)
        self.context32 = MultiPoolContext(pool_kernels, residual_mix=pool_residual_mix)
        self.ref32 = DSResidual(width)

        # Scale router: 4×width inputs → 4 logits per spatial location at /8
        # Initialized to zeros → initial routing is exactly uniform softmax
        self.router = nn.Conv2d(4 * width, 4, kernel_size=1, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

        self.attn = SimAM(lambda_e=simam_lambda)

    @staticmethod
    def _resize(x: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
        if x.shape[-2:] == torch.Size(size) if isinstance(size, (list, tuple)) else x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode=mode, align_corners=False)

    def forward(self, c4, c8, c16, c32, return_routes: bool = False):
        # Step 1 & 2: lateral + refinement
        r4 = self.ref4(self.lat4(c4))
        r8 = self.ref8(self.lat8(c8))
        r16 = self.ref16(self.lat16(c16))
        r32 = self.ref32(self.context32(self.lat32(c32)))

        # Step 3: compute routing logits at /8 (r8 resolution)
        route_size = r8.shape[-2:]
        # Use area interpolation for downsampling r4 → /8 (parameter-free, deterministic)
        g4 = F.interpolate(r4, size=route_size, mode="area")
        g8 = r8
        g16 = self._resize(r16, route_size)
        g32 = self._resize(r32, route_size)

        logits = self.router(torch.cat([g4, g8, g16, g32], dim=1))
        routes8 = torch.softmax(logits / self.route_temperature, dim=1)  # B×4×H/8×W/8

        # Step 4: upsample routing weights to /4
        routes4 = self._resize(routes8, r4.shape[-2:])  # B×4×H/4×W/4

        # Step 5: weighted sum at /4
        u4 = r4
        u8 = self._resize(r8, r4.shape[-2:])
        u16 = self._resize(r16, r4.shape[-2:])
        u32 = self._resize(r32, r4.shape[-2:])

        scales = (u4, u8, u16, u32)
        fused = sum(routes4[:, i : i + 1] * scales[i] for i in range(4))

        # Step 6: SimAM (0 params)
        fused = self.attn(fused)

        if return_routes:
            return fused, {
                "routes8": routes8,
                "routes4": routes4,
                "scale_features": scales,
            }
        return fused
