from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    RegionSet,
    build_multiscale_regions,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)

Variant = Literal[
    "direct",
    "region_loss",
    "region_aux",
    "local_refine",
    "learned_project",
    "rmr",
]


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


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
    """Small inverted residual block. Enabling component, not a novelty claim."""

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


class TinyLocalEncoder(nn.Module):
    """Native local-first encoder exposing stride-4/8/16 features."""

    def __init__(self):
        super().__init__()
        self.stem = ConvGNAct(3, 16, 3, stride=2)
        self.s4 = nn.Sequential(TinyIR(16, 24, stride=2), TinyIR(24, 24))
        self.s8 = nn.Sequential(TinyIR(24, 40, stride=2), TinyIR(40, 40), TinyIR(40, 40))
        self.s16 = nn.Sequential(TinyIR(40, 64, stride=2), TinyIR(64, 64))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        c4 = self.s4(x)
        c8 = self.s8(c4)
        c16 = self.s16(c8)
        return c4, c8, c16


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

    def forward(self, c4: torch.Tensor, c8: torch.Tensor, c16: torch.Tensor) -> torch.Tensor:
        size = c4.shape[-2:]
        p = self.p4(c4)
        p = p + F.interpolate(self.p8(c8), size=size, mode="bilinear", align_corners=False)
        p = p + F.interpolate(self.p16(c16), size=size, mode="bilinear", align_corners=False)
        return self.out(p)


class FineMeasureHead(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
            nn.Conv2d(width, 1, 1),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.body(f)


class RegionalEvidenceHead(nn.Module):
    """Shared regional count regressor over exact integral-feature pooled descriptors."""

    def __init__(self, feature_dim: int = 32, hidden: int = 48, geom_dim: int = 6):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + geom_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, f: torch.Tensor, regions: RegionSet) -> torch.Tensor:
        b, _, h, w = f.shape
        pooled = region_average_features(f, regions.boxes)  # [B,M,C]
        geom = region_geometry(regions.boxes, h, w).to(dtype=f.dtype)
        geom = geom.unsqueeze(0).expand(b, -1, -1)
        raw = self.mlp(torch.cat([pooled, geom], dim=-1)).squeeze(-1)
        return F.softplus(raw).unsqueeze(1)  # [B,1,M]


class LocalPreconditioner(nn.Module):
    """Small bounded local preconditioner applied after the exact adjoint field."""

    def __init__(
        self,
        feature_dim: int = 32,
        hidden: int = 32,
        m_min: float = 0.25,
        m_max: float = 1.75,
    ):
        super().__init__()
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.net = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f: torch.Tensor, y: torch.Tensor, residual_field: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.net(torch.cat([f, y, residual_field], dim=1)))
        return self.m_min + (self.m_max - self.m_min) * gate


class LocalCNNRefiner(nn.Module):
    """B3a control: purely local learned refinement.

    It has no regional count input. Spatial scope is deliberately limited to local 3x3 mixing.
    This asks whether simply spending extra local neural capacity can explain RMR's gain.
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.in_proj = ConvGNAct(feature_dim + 1, hidden, 1)
        self.dw = ConvGNAct(hidden, hidden, 3, groups=hidden)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.out(self.dw(self.in_proj(torch.cat([f, y], dim=1))))


class LearnedMembershipProjector(nn.Module):
    """B3b control: same regional residuals and same region memberships, learned allocation.

    For each rectangle R, the raw regional count residual delta_R = (A Y - b)_R is
    redistributed over cells p in R using a learned region-normalized visual weighting:

        pi_{R,p} = softmax_{p in R}(s_theta(F)_p)
        r_p = mean_{R contains p} delta_R * pi_{R,p}

    Exact RMR corresponds to a fixed uniform allocation delta_R / |R| before overlap
    averaging, followed by a small local preconditioner. This B3b control therefore has
    access to the same regional information and scope but is allowed to learn the
    region-to-grid allocation geometry.

    The explicit region loop is intentionally used for correctness in the causal control.
    Its measured latency must be reported; it is not proposed as the deployment model.
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.score = nn.Sequential(
            ConvGNAct(feature_dim + 1, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )
        self.post = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 1),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            nn.Conv2d(hidden, 1, 1),
        )

    def project(
        self,
        f: torch.Tensor,
        y: torch.Tensor,
        raw_delta: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        b, _, h, w = y.shape
        score = self.score(torch.cat([f, y], dim=1))
        out = y.new_zeros((b, 1, h, w))
        coverage = y.new_zeros((b, 1, h, w))

        # Strong, fair control: each residual can influence every cell in its own region.
        for m, box in enumerate(regions.boxes.tolist()):
            y1, x1, y2, x2 = map(int, box)
            logits = score[:, :, y1:y2, x1:x2]
            flat = logits.flatten(-2)
            pi = torch.softmax(flat, dim=-1).view_as(logits)
            delta = raw_delta[:, :, m].view(b, 1, 1, 1)
            out[:, :, y1:y2, x1:x2] += delta * pi
            coverage[:, :, y1:y2, x1:x2] += 1.0

        field = out / coverage.clamp_min(1.0)
        # The post-net is allowed to shape the learned projection further.
        return self.post(torch.cat([f, y, field], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    # Pilot starts at 32 px; 16 px is added only if oracle/predicted-scale diagnostics justify it.
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = True
    iterations: int = 2

    # Stability: bounded step size with small initialization.
    eta_max: float = 0.20
    eta_init: float = 0.05
    residual_clip: float = 5.0

    eps: float = 1e-6


class RMRCount(nn.Module):
    """Regional Measure Reconciliation crowd counter and registered controls."""

    def __init__(self, cfg: RMRConfig = RMRConfig(), variant: Variant = "rmr"):
        super().__init__()
        self.cfg = cfg
        self.variant = variant
        self.encoder = TinyLocalEncoder()
        self.fusion = AdditiveFusion(cfg.feature_width)
        self.fine_head = FineMeasureHead(cfg.feature_width)

        needs_region_head = variant in {"region_aux", "learned_project", "rmr"}
        self.region_head = RegionalEvidenceHead(cfg.feature_width) if needs_region_head else None

        self.preconditioner = LocalPreconditioner(cfg.feature_width) if variant == "rmr" else None
        self.local_refiner = LocalCNNRefiner(cfg.feature_width) if variant == "local_refine" else None
        self.learned_projector = (
            LearnedMembershipProjector(cfg.feature_width) if variant == "learned_project" else None
        )

        n_steps = max(1, cfg.iterations)
        frac = cfg.eta_init / max(cfg.eta_max, 1e-8)
        init = _logit(frac)
        self.eta_logits = nn.Parameter(torch.full((n_steps,), init))

        # Training script ramps this from 0 -> 1 after the direct prediction has stabilized.
        self.solver_strength: float = 1.0

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

    def _regions(self, h: int, w: int, device: torch.device) -> RegionSet:
        return build_multiscale_regions(
            height=h,
            width=w,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=self.cfg.include_full_image,
            device=device,
        )

    def _eta(self, t: int) -> torch.Tensor:
        idx = min(t, self.eta_logits.numel() - 1)
        return self.cfg.eta_max * torch.sigmoid(self.eta_logits[idx])

    def _raw_region_delta(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        return regional_sum(y, regions.boxes) - b_region

    def _rmr_field(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        """Coverage-normalized exact adjoint of regional count residual density."""
        _, _, h, w = y.shape
        delta = self._raw_region_delta(y, b_region, regions)  # [B,1,M]
        area = regions.area.to(y.dtype).view(1, 1, -1)
        residual_density = delta / area.clamp_min(1.0)

        back = regional_adjoint(residual_density, regions.boxes, h, w)
        coverage = regional_adjoint(
            torch.ones_like(residual_density), regions.boxes, h, w
        )
        r = back / coverage.clamp_min(1.0)
        if self.cfg.residual_clip > 0:
            r = r.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
        return r

    def forward(
        self,
        x: torch.Tensor,
        *,
        b_region_override: torch.Tensor | None = None,
        shuffle_region: bool = False,
    ) -> dict:
        c4, c8, c16 = self.encoder(x)
        f = self.fusion(c4, c8, c16)
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]
        regions = self._regions(h, w, x.device)

        out: dict = {
            "features": f,
            "z0": z0,
            "y0": y0,
            "regions": regions,
        }

        if self.region_head is not None:
            b_region = self.region_head(f, regions)
            if b_region_override is not None:
                if b_region_override.shape != b_region.shape:
                    raise ValueError(
                        f"b_region_override shape {tuple(b_region_override.shape)} "
                        f"!= {tuple(b_region.shape)}"
                    )
                b_region = b_region_override
            elif shuffle_region:
                # Shuffle only within each scale family to avoid a trivial scale mismatch artifact.
                pieces = []
                b_region = b_region.clone()
                for sid in torch.unique(regions.scale_id):
                    mask = regions.scale_id == sid
                    idx = torch.where(mask)[0]
                    if idx.numel() > 1:
                        perm = idx[torch.randperm(idx.numel(), device=idx.device)]
                        b_region[..., idx] = b_region[..., perm]
            out["b_region"] = b_region
        else:
            b_region = None

        if self.variant in {"direct", "region_loss", "region_aux"}:
            out["y"] = y0
            out["z"] = z0
            out["iterates"] = [y0]
            out["residual_fields"] = []
            return out

        z = z0
        y = y0
        iterates = [y0]
        residual_fields: list[torch.Tensor] = []

        if self.variant in {"learned_project", "rmr"} and b_region is None:
            raise RuntimeError(f"variant {self.variant} requires regional evidence")

        for t in range(self.cfg.iterations):
            eta = self._eta(t) * self.solver_strength
            if self.variant == "rmr":
                assert b_region is not None and self.preconditioner is not None
                r = self._rmr_field(y, b_region, regions)
                residual_fields.append(r)
                m = self.preconditioner(f, y, r)
                z = z - eta * m * torch.sigmoid(z) * r

            elif self.variant == "learned_project":
                assert b_region is not None and self.learned_projector is not None
                delta = self._raw_region_delta(y, b_region, regions)
                learned_field = self.learned_projector.project(f, y, delta, regions)
                if self.cfg.residual_clip > 0:
                    learned_field = learned_field.clamp(
                        -self.cfg.residual_clip, self.cfg.residual_clip
                    )
                residual_fields.append(learned_field)
                z = z - eta * learned_field

            elif self.variant == "local_refine":
                assert self.local_refiner is not None
                dz = self.local_refiner(f, y)
                if self.cfg.residual_clip > 0:
                    dz = dz.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
                residual_fields.append(dz)
                z = z - eta * dz

            else:
                raise RuntimeError(f"Unknown variant {self.variant}")

            y = F.softplus(z)
            iterates.append(y)

        out["y"] = y
        out["z"] = z
        out["iterates"] = iterates
        out["residual_fields"] = residual_fields
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
