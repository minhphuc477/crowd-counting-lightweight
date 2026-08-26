import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvGNAct, DSResidual, MultiPoolContext, SimAM


class ScaleRoutedFusionNeck(nn.Module):
    """Four-scale feature fusion neck with Shared Scale-Evidence Routing (SSER).

    Architecture (SR48 spec §7 + SSER upgrade):
      1. Four lateral 1×1 Conv+GN projections: C4/C8/C16/C32 → 48 ch.
      2. Independent DS-Residual refinements at each scale.
         At /32: zero-parameter multi-pool context before refinement.
      3. SSER computed at /8 resolution:
           For each scale s ∈ {4, 8, 16, 32}:
             e_s(x,y) = shared_scorer(R_s_at_/8) + b_s   [52 params total]
           α = softmax([e_4, e_8, e_16, e_32] / τ)        [sum-to-1 per location]
      4. Routing weights upsample to /4.
      5. Weighted sum: F = Σ_s α_s(x,y) * R_s(x,y).
      6. SimAM (0 params) applied to fused features.
      Output: fused feature map at /4, 48 ch.

    Router initialised to zero weight/zero bias → initial routing is exactly
    uniform (same as spec §7 requirement, but now only 52 params instead of 772).

    Training-only: the neck can optionally return routes8 for routing supervision
    via KL(q_scale ‖ α) loss. This signal teaches the router to be geometry-aware.
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
        self.lat4  = ConvGNAct(c4,  width, kernel_size=1)
        self.lat8  = ConvGNAct(c8,  width, kernel_size=1)
        self.lat16 = ConvGNAct(c16, width, kernel_size=1)
        self.lat32 = ConvGNAct(c32, width, kernel_size=1)

        # Independent scale refinements
        self.ref4    = DSResidual(width)
        self.ref8    = DSResidual(width)
        self.ref16   = DSResidual(width)
        self.context32 = MultiPoolContext(pool_kernels, residual_mix=pool_residual_mix)
        self.ref32   = DSResidual(width)

        # ── Shared Scale-Evidence Router (SSER, 52 params) ───────────────────
        # One Conv2d(48→1) shared across all 4 scales = 48 params.
        # Four learnable per-scale bias scalars                = 4  params.
        # Total: 52 params (vs 772 for the concat-concat router).
        # Init to zero → initial routing is exactly uniform softmax.
        self.shared_scorer = nn.Conv2d(width, 1, kernel_size=1, bias=False)
        nn.init.zeros_(self.shared_scorer.weight)
        self.scale_bias = nn.Parameter(torch.zeros(4))   # [b_/4, b_/8, b_/16, b_/32]
        # ─────────────────────────────────────────────────────────────────────

        self.attn = SimAM(lambda_e=simam_lambda)

    @staticmethod
    def _resize(x: torch.Tensor, size, mode: str = "bilinear") -> torch.Tensor:
        target = torch.Size(size) if not isinstance(size, torch.Size) else size
        if x.shape[-2:] == target:
            return x
        return F.interpolate(x, size=target, mode=mode, align_corners=False)

    def forward(self, c4, c8, c16, c32, return_routes: bool = False):
        """Forward pass.

        Args:
            c4, c8, c16, c32: multi-scale backbone features at /4, /8, /16, /32.
            return_routes: if True, also return a dict with ``routes8`` and
                ``routes4`` for routing supervision and diagnostics.

        Returns:
            fused: (B, 48, H/4, W/4) fused feature map.
            route_info (dict, only when return_routes=True):
                routes8: (B, 4, H/8, W/8) per-location scale weights at /8.
                routes4: (B, 4, H/4, W/4) upsampled weights at /4.
        """
        # Step 1 & 2: lateral projection + independent refinement
        r4  = self.ref4(self.lat4(c4))
        r8  = self.ref8(self.lat8(c8))
        r16 = self.ref16(self.lat16(c16))
        r32 = self.ref32(self.context32(self.lat32(c32)))

        # Step 3: SSER — compute per-scale evidence at /8 resolution
        route_size = r8.shape[-2:]  # (H/8, W/8)

        # Project each refined scale feature to /8 (parameter-free interpolation)
        g4  = F.interpolate(r4,  size=route_size, mode="area")   # /4  → /8  (avg pool)
        g8  = r8                                                   # /8  as-is
        g16 = self._resize(r16, route_size)                        # /16 → /8  (bilinear)
        g32 = self._resize(r32, route_size)                        # /32 → /8  (bilinear)

        # Shared scorer applied independently to each scale (broadcasting the same weights)
        e4  = self.shared_scorer(g4)   # (B, 1, H/8, W/8)
        e8  = self.shared_scorer(g8)
        e16 = self.shared_scorer(g16)
        e32 = self.shared_scorer(g32)

        # Stack and add per-scale learnable bias
        logits = torch.cat([e4, e8, e16, e32], dim=1)              # (B, 4, H/8, W/8)
        logits = logits + self.scale_bias.view(1, 4, 1, 1)

        routes8 = torch.softmax(logits / self.route_temperature, dim=1)

        # Step 4: upsample routing weights to /4
        routes4 = self._resize(routes8, r4.shape[-2:])             # (B, 4, H/4, W/4)

        # Step 5: weighted sum at /4 (upsample all scales first)
        u4  = r4
        u8  = self._resize(r8,  r4.shape[-2:])
        u16 = self._resize(r16, r4.shape[-2:])
        u32 = self._resize(r32, r4.shape[-2:])

        scales = (u4, u8, u16, u32)
        fused  = sum(routes4[:, i : i + 1] * scales[i] for i in range(4))

        # Step 6: SimAM (0 params)
        fused = self.attn(fused)

        if return_routes:
            return fused, {
                "routes8": routes8,
                "routes4": routes4,
            }
        return fused
