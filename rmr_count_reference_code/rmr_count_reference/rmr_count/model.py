from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    region_average_features,
    region_geometry,
    regional_adjoint,
    regional_sum,
)

Variant = Literal[
    "direct",
    "region_loss",
    "region_aux",
    "learned_project",
    "rmr",
]


def _gn(channels: int) -> nn.GroupNorm:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1, act: bool = True):
        pad = k // 2
        layers: list[nn.Module] = [nn.Conv2d(cin, cout, k, stride=stride, padding=pad, groups=groups, bias=False), _gn(cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class TinyIR(nn.Module):
    """Small inverted residual block using depthwise spatial mixing."""

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
    """Local-first encoder exposing stride-4/8/16 features."""

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
    """Shared region-count regressor over integral-feature pooled descriptors."""

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
        b, c, h, w = f.shape
        pooled = region_average_features(f, regions.boxes)  # [B,M,C]
        geom = region_geometry(regions.boxes, h, w).to(dtype=f.dtype)
        geom = geom.unsqueeze(0).expand(b, -1, -1)
        raw = self.mlp(torch.cat([pooled, geom], dim=-1)).squeeze(-1)
        return F.softplus(raw).unsqueeze(1)  # [B,1,M]


class LocalPreconditioner(nn.Module):
    def __init__(self, feature_dim: int = 32, hidden: int = 32, m_min: float = 0.25, m_max: float = 1.75):
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


class LearnedRegionProjector(nn.Module):
    """Control that must learn how sparse region-center residuals affect the fine grid."""

    def __init__(self, feature_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(feature_dim + 2, hidden, 3),
            ConvGNAct(hidden, hidden, 3, groups=hidden),
            ConvGNAct(hidden, hidden, 3),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, f: torch.Tensor, y: torch.Tensor, sparse_residual: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f, y, sparse_residual], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    region_sizes_px: tuple[int, ...] = (16, 32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = True
    iterations: int = 2
    eta_max: float = 1.0
    eps: float = 1e-6


class RMRCount(nn.Module):
    """Regional Measure Reconciliation crowd counter.

    Core variant `rmr`:
        image -> fine non-negative measure Y0
              -> separately inferred region counts b
              -> q = A Y
              -> exact residual back-projection A^T[(q-b)/area]
              -> small learned local preconditioner
              -> positive latent update

    Baseline variants are implemented in the same class for matched experiments.
    """

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
        self.learned_projector = LearnedRegionProjector(cfg.feature_width, hidden=16) if variant == "learned_project" else None

        n_steps = max(1, cfg.iterations)
        self.eta_logits = nn.Parameter(torch.zeros(n_steps))

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

    def _normalized_region_error(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        q = regional_sum(y, regions.boxes)  # [B,1,M]
        area = regions.area.to(y.dtype).view(1, 1, -1)
        return (q - b_region) / area.clamp_min(1.0)

    def _rmr_field(self, y: torch.Tensor, b_region: torch.Tensor, regions: RegionSet) -> torch.Tensor:
        bsz, _, h, w = y.shape
        e = self._normalized_region_error(y, b_region, regions)
        back = regional_adjoint(e, regions.boxes, h, w)
        ones = torch.ones_like(e)
        coverage = regional_adjoint(ones, regions.boxes, h, w)
        return back / coverage.clamp_min(1.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | RegionSet | list[torch.Tensor]]:
        c4, c8, c16 = self.encoder(x)
        f = self.fusion(c4, c8, c16)
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]
        regions = self._regions(h, w, x.device)

        out: dict[str, torch.Tensor | RegionSet | list[torch.Tensor]] = {
            "features": f,
            "z0": z0,
            "y0": y0,
            "regions": regions,
        }

        if self.region_head is not None:
            b_region = self.region_head(f, regions)
            out["b_region"] = b_region
        else:
            b_region = None

        if self.variant in {"direct", "region_loss", "region_aux"}:
            out["y"] = y0
            out["iterates"] = [y0]
            return out

        z = z0
        y = y0
        iterates = [y0]
        residual_fields: list[torch.Tensor] = []

        if b_region is None:
            raise RuntimeError(f"variant {self.variant} requires regional evidence")

        for t in range(self.cfg.iterations):
            if self.variant == "rmr":
                r = self._rmr_field(y, b_region, regions)
                residual_fields.append(r)
                assert self.preconditioner is not None
                m = self.preconditioner(f, y, r)
                # Chain rule for Y=softplus(z): dY/dz = sigmoid(z).
                z = z - self._eta(t) * m * torch.sigmoid(z) * r
            elif self.variant == "learned_project":
                e = self._normalized_region_error(y, b_region, regions)
                sparse = center_scatter(e, regions.boxes, h, w)
                residual_fields.append(sparse)
                assert self.learned_projector is not None
                dz = self.learned_projector(f, y, sparse)
                z = z - self._eta(t) * dz
            else:
                raise RuntimeError(f"Unknown variant {self.variant}")
            y = F.softplus(z)
            iterates.append(y)

        out["y"] = y
        out["iterates"] = iterates
        out["residual_fields"] = residual_fields
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
