from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import MobileNetV4Backbone
from rmr_core.heads import _FINE_HEAD_BIAS_INIT, _M0_INIT, FineMeasureHead
from rmr_core.necks import (
    AdditiveFPNNeck,
    AdditiveFusion,
    ConvGNAct,
    DepthwiseDilated,
    DSResidual,
    TinyIR,
    _gn,
)
from rmr_core.operators import (
    RegionSet,
    build_multiscale_regions,
    center_scatter,
    region_average_features,
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

RMRUpdate = Literal["latent", "jacobian", "projected_sirt"]


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


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


class ScaleMatchedRegionalEvidenceHead(nn.Module):
    """Scale-matched regional count regressor observing feature pyramid (P4, P8, P16).

    Architecture (RMR-v2):
        32px regions  -> observed on P4 (stride 4, 8x8 window)
        64px regions  -> observed on P8 (stride 8, 8x8 window)
        128px regions -> observed on P16 (stride 16, 8x8 window)

    Feature representation per region:
        [u_R, log(s_R / 32.0)] in R^33 (32 feature channels + 1 scale feature)

    Shared MLP:
        Linear(33, 48) -> SiLU -> Linear(48, 48) -> SiLU -> Linear(48, 1)

    Rate & Measure:
        rho_R = softplus(raw_R)       [count / stride-4 cell]
        b_R   = |R| * rho_R           [total count in region R]
    where |R| is the area in stride-4 cells (64, 256, 1024).
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden: int = 48,
        init_bias: float = _FINE_HEAD_BIAS_INIT,
        region_sizes_px: tuple[int, ...] | Sequence[int] = (32, 64, 128),
    ):
        super().__init__()
        self.region_sizes_px = tuple(int(s) for s in region_sizes_px)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        final_linear: nn.Linear = self.mlp[-1]  # type: ignore[assignment]
        nn.init.normal_(final_linear.weight, std=0.01)
        nn.init.constant_(final_linear.bias, init_bias)

    def forward(
        self,
        pyramid_or_f: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        if isinstance(pyramid_or_f, tuple):
            p4, p8, p16 = pyramid_or_f
        else:
            p4 = pyramid_or_f
            p8 = F.interpolate(p4, scale_factor=0.5, mode="bilinear", align_corners=False)
            p16 = F.interpolate(p8, scale_factor=0.5, mode="bilinear", align_corners=False)

        b = p4.shape[0]
        device = p4.device
        dtype = p4.dtype

        size4 = p4.shape[-2:]
        feat_p4 = p4
        feat_p8_at_4 = (
            p8 if p8.shape[-2:] == size4
            else F.interpolate(p8, size=size4, mode="bilinear", align_corners=False)
        )
        feat_p16_at_4 = (
            p16 if p16.shape[-2:] == size4
            else F.interpolate(p16, size=size4, mode="bilinear", align_corners=False)
        )

        m_total = regions.boxes.shape[0]
        in_dim = self.mlp[0].in_features  # 33
        feat_all = torch.zeros((b, m_total, in_dim), device=device, dtype=dtype)

        for sid, size_px in enumerate(self.region_sizes_px):
            mask = regions.scale_id == sid
            if not mask.any():
                continue
            if size_px <= 48:
                feat_s = feat_p4
            elif size_px <= 96:
                feat_s = feat_p8_at_4
            else:
                feat_s = feat_p16_at_4

            boxes_s = regions.boxes[mask]
            u_s = region_average_features(feat_s, boxes_s)  # [B, M_s, C]
            m_s = u_s.shape[1]
            scale_feat = torch.full(
                (1, m_s, 1),
                math.log(float(size_px) / 32.0),
                device=device,
                dtype=dtype,
            ).expand(b, -1, -1)
            f_s = torch.cat([u_s, scale_feat], dim=-1)   # [B, M_s, 33]
            feat_all[:, mask] = f_s

        mask_full = regions.scale_id == -1
        if mask_full.any():
            boxes_full = regions.boxes[mask_full]
            u_full = region_average_features(feat_p16_at_4, boxes_full)
            m_f = u_full.shape[1]
            full_scale = max(size4[0], size4[1]) * 4.0
            scale_feat = torch.full(
                (1, m_f, 1),
                math.log(max(full_scale, 32.0) / 32.0),
                device=device,
                dtype=dtype,
            ).expand(b, -1, -1)
            feat_all[:, mask_full] = torch.cat([u_full, scale_feat], dim=-1)

        raw = self.mlp(feat_all).squeeze(-1)            # [B, M]
        rate = F.softplus(raw)                          # [B, M]
        area = regions.area.to(dtype=dtype).view(1, -1) # [1, M]
        b_region = rate * area                          # [B, M]
        return b_region.unsqueeze(1)                    # [B, 1, M]


RegionalEvidenceHead = ScaleMatchedRegionalEvidenceHead


class LocalPreconditioner(nn.Module):
    """Small bounded local preconditioner applied after the normalized adjoint field."""

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
    """B3a control: purely local learned refinement."""

    def __init__(self, feature_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.in_proj = ConvGNAct(feature_dim + 1, hidden, 1)
        self.dw = ConvGNAct(hidden, hidden, 3, groups=hidden)
        self.out = nn.Conv2d(hidden, 1, 1)

    def forward(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.out(self.dw(self.in_proj(torch.cat([f, y], dim=1))))


class LearnedMembershipProjector(nn.Module):
    """B3b control: same regional residuals and same region memberships, learned allocation."""

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

        boxes_list = regions.boxes_list if regions.boxes_list is not None else regions.boxes.tolist()
        for m, box in enumerate(boxes_list):
            y1, x1, y2, x2 = map(int, box)
            logits = score[:, :, y1:y2, x1:x2]
            flat = logits.flatten(-2)
            pi = torch.softmax(flat, dim=-1).view_as(logits)
            delta = raw_delta[:, :, m].view(b, 1, 1, 1)
            out[:, :, y1:y2, x1:x2] += delta * pi
            coverage[:, :, y1:y2, x1:x2] += 1.0

        field = out / coverage.clamp_min(1.0)
        return self.post(torch.cat([f, y, field], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False
    iterations: int = 2

    eta_max: float = 0.20
    eta_init: float = 0.05
    residual_clip: float = 5.0

    eps: float = 1e-6
    update_rule: RMRUpdate = "projected_sirt"
    use_jacobian_gate: bool = False

    sirt_omega: float = 1.0
    learnable_sirt_omega: bool = False
    projected_use_preconditioner: bool = False
    detach_region_evidence: bool = True

    init_m0: float = 0.015763
    backbone_name: str = "mobilenetv4_conv_small_050.e3000_r224_in1k"
    pretrained: bool = False
    backbone_lr_scale: float = 0.1


class RMRCount(nn.Module):
    r"""Regional Measure Reconciliation crowd counter and registered controls (RMR-v2)."""

    def __init__(self, cfg: RMRConfig = RMRConfig(), variant: Variant = "rmr"):
        super().__init__()
        if cfg.output_stride != 4:
            raise ValueError(
                f"RMRCount only supports output_stride=4. Got output_stride={cfg.output_stride}."
            )
        self.cfg = cfg
        self.variant = variant

        needs_region_head = variant in {
            "region_aux",
            "learned_project",
            "rmr",
        }
        init_m0 = float(getattr(cfg, "init_m0", _M0_INIT))
        init_bias = math.log(math.exp(init_m0) - 1.0)
        self.region_head = (
            RegionalEvidenceHead(
                cfg.feature_width,
                init_bias=init_bias,
                region_sizes_px=cfg.region_sizes_px,
            )
            if needs_region_head
            else None
        )

        effective_rule = cfg.update_rule
        if cfg.use_jacobian_gate:
            effective_rule = "jacobian"
        self.rmr_update_rule: RMRUpdate = effective_rule
        self.update_rule: RMRUpdate = effective_rule

        if cfg.backbone_name == "tiny":
            self.encoder = TinyLocalEncoder()
            self.fusion = AdditiveFusion(cfg.feature_width)
        else:
            self.encoder = MobileNetV4Backbone(
                model_name=cfg.backbone_name,
                pretrained=cfg.pretrained,
            )
            self.fusion = AdditiveFPNNeck(
                in_channels=self.encoder.out_channels,
                width=cfg.feature_width,
            )
        self.fine_head = FineMeasureHead(cfg.feature_width, init_bias=init_bias)

        need_preconditioner = (
            variant == "rmr"
            and (
                self.rmr_update_rule in {"latent", "jacobian"}
                or cfg.projected_use_preconditioner
            )
        )
        self.preconditioner = (
            LocalPreconditioner(cfg.feature_width)
            if need_preconditioner
            else None
        )

        self.local_refiner = (
            LocalCNNRefiner(cfg.feature_width)
            if variant == "local_refine"
            else None
        )

        self.learned_projector = (
            LearnedMembershipProjector(cfg.feature_width)
            if variant == "learned_project"
            else None
        )

        need_eta = (variant == "local_refine") or (
            variant == "rmr" and self.rmr_update_rule in {"latent", "jacobian"}
        )
        if need_eta:
            n_steps = max(1, cfg.iterations)
            frac = cfg.eta_init / max(cfg.eta_max, 1e-8)
            init = _logit(frac)
            self.eta_logits = nn.Parameter(
                torch.full((n_steps,), init)
            )
        else:
            self.register_parameter("eta_logits", None)

        if (
            variant == "rmr"
            and self.rmr_update_rule == "projected_sirt"
            and cfg.learnable_sirt_omega
        ):
            if cfg.sirt_omega <= 0:
                raise ValueError("sirt_omega must be > 0")
            self.log_sirt_omega = nn.Parameter(
                torch.tensor(math.log(cfg.sirt_omega), dtype=torch.float32)
            )
        else:
            self.register_parameter("log_sirt_omega", None)

        self.solver_strength: float = 1.0
        self._cache: OrderedDict[
            tuple,
            tuple[RegionSet, torch.Tensor]
        ] = OrderedDict()

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

    def _regions_and_coverage(self, h: int, w: int, device: torch.device) -> tuple[RegionSet, torch.Tensor]:
        key = (
            h, w,
            self.cfg.output_stride,
            self.cfg.region_sizes_px,
            self.cfg.region_overlap,
            self.cfg.include_full_image,
            device.type,
            device.index if device.type == "cuda" else None,
        )
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        if len(self._cache) >= 32:
            self._cache.popitem(last=False)

        regions = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=self.cfg.output_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=self.cfg.include_full_image,
            device=device,
        )
        ones_m = torch.ones((1, 1, regions.boxes.shape[0]), device=device, dtype=torch.float32)
        cov = regional_adjoint(ones_m, regions.boxes, h, w).clamp_min(1.0)
        self._cache[key] = (regions, cov)
        return self._cache[key]

    def _regions(self, h: int, w: int, device: torch.device) -> RegionSet:
        return self._regions_and_coverage(h, w, device)[0]

    def _eta(self, t: int) -> torch.Tensor:
        if self.eta_logits is None:
            raise RuntimeError(
                f"_eta is not defined for variant={self.variant} / "
                f"update_rule={self.rmr_update_rule} (eta_logits is None)"
            )
        idx = min(t, self.eta_logits.numel() - 1)
        return self.cfg.eta_max * torch.sigmoid(self.eta_logits[idx])

    def _sirt_omega(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self.log_sirt_omega is not None:
            return torch.exp(self.log_sirt_omega)
        if self.eta_logits is not None:
            return self.eta_logits.new_tensor(float(self.cfg.sirt_omega))
        if device is None:
            try:
                p = next(self.parameters())
                device = p.device
                if dtype is None:
                    dtype = p.dtype
            except StopIteration:
                device = torch.device("cpu")
                if dtype is None:
                    dtype = torch.float32
        if dtype is None:
            dtype = torch.float32
        return torch.tensor(float(self.cfg.sirt_omega), device=device, dtype=dtype)

    def _projected_sirt_step(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
        coverage: torch.Tensor,
        *,
        features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self._normalized_adjoint_field(
            y,
            b_region,
            regions,
            coverage=coverage,
        )

        if self.cfg.projected_use_preconditioner:
            if self.preconditioner is None:
                raise RuntimeError(
                    "projected_use_preconditioner=True but no LocalPreconditioner was constructed"
                )
            if features is None:
                raise ValueError("features required for learned preconditioner")
            m = self.preconditioner(features, y, r)
        else:
            m = torch.ones_like(y)

        omega = self._sirt_omega(device=y.device, dtype=y.dtype)
        omega_eff = omega * float(self.solver_strength)

        y_next = torch.clamp_min(
            y.float() - (omega_eff * m).float() * r.float(),
            0.0,
        ).to(y.dtype)

        return y_next, r, m, omega_eff

    def _raw_region_delta(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        return regional_sum(y, regions.boxes, out_dtype=torch.float32) - b_region.float()

    def _normalized_adjoint_field(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
        coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, h, w = y.shape
        work_y = y.float()
        work_b = b_region.float()
        work_area = regions.area.float().view(1, 1, -1)

        delta = self._raw_region_delta(work_y, work_b, regions)
        residual_density = delta / work_area.clamp_min(1.0)

        back = regional_adjoint(residual_density, regions.boxes, h, w, out_dtype=torch.float32)
        if coverage is None:
            coverage = self._regions_and_coverage(h, w, y.device)[1]
        r = back / coverage.float()
        if self.cfg.residual_clip > 0:
            r = r.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
        return r.to(y.dtype)

    _rmr_field = _normalized_adjoint_field

    def forward(
        self,
        x: torch.Tensor,
        *,
        b_region_override: torch.Tensor | None = None,
        shuffle_region: bool = False,
    ) -> dict[str, torch.Tensor | RegionSet | None | list]:
        c4, c8, c16 = self.encoder(x)
        p4, p8, p16 = self.fusion(c4, c8, c16)
        f = p4
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]

        needs_regions = self.region_head is not None or self.variant in {
            "region_loss", "learned_project", "rmr"
        }
        if needs_regions:
            regions, coverage = self._regions_and_coverage(h, w, x.device)
        else:
            regions, coverage = None, None

        out: dict = {
            "features": f,
            "pyramid": (p4, p8, p16),
            "z0": z0,
            "y0": y0,
            "regions": regions,
            "preconditioner_fields": [],
            "step_sizes": [],
        }

        if self.region_head is not None:
            assert regions is not None
            b_region = self.region_head((p4, p8, p16), regions)
            if b_region_override is not None:
                if b_region_override.shape != b_region.shape:
                    raise ValueError(
                        f"b_region_override shape {tuple(b_region_override.shape)} != {tuple(b_region.shape)}"
                    )
                b_region = b_region_override
            elif shuffle_region:
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
        preconditioner_fields: list[torch.Tensor] = []
        step_sizes: list[float] = []

        if self.variant in {"learned_project", "rmr"} and b_region is None:
            raise RuntimeError(f"variant {self.variant} requires regional evidence")

        if (
            (
                (self.variant == "rmr" and self.rmr_update_rule == "projected_sirt")
                or self.variant == "learned_project"
            )
            and b_region is not None
            and self.cfg.detach_region_evidence
        ):
            b_solver = b_region.detach()
        else:
            b_solver = b_region

        for t in range(self.cfg.iterations):
            if (
                self.variant == "rmr"
                and self.rmr_update_rule == "projected_sirt"
            ):
                assert b_solver is not None
                assert regions is not None
                assert coverage is not None

                y, r, m, omega_eff = self._projected_sirt_step(
                    y,
                    b_solver,
                    regions,
                    coverage,
                    features=f,
                )

                residual_fields.append(r)
                preconditioner_fields.append(m)
                step_sizes.append(float(omega_eff.detach().item()))
                iterates.append(y)
                continue

            if self.variant == "learned_project":
                assert b_solver is not None
                assert regions is not None
                assert self.learned_projector is not None

                delta = self._raw_region_delta(y, b_solver, regions)
                learned_field = self.learned_projector.project(f, y, delta, regions)

                if self.cfg.residual_clip > 0:
                    learned_field = learned_field.clamp(
                        -self.cfg.residual_clip,
                        self.cfg.residual_clip,
                    )

                residual_fields.append(learned_field)

                omega = self._sirt_omega(device=y.device, dtype=y.dtype)
                omega_eff = omega * float(self.solver_strength)
                step_sizes.append(float(omega_eff.detach().item()))

                y = torch.clamp_min(
                    y.float() - omega_eff.float() * learned_field.float(),
                    0.0,
                ).to(y.dtype)

                iterates.append(y)
                continue

            eta = self._eta(t) * self.solver_strength
            step_sizes.append(float(eta.detach().item()))

            if self.variant == "rmr":
                assert b_solver is not None
                assert regions is not None
                assert coverage is not None
                assert self.preconditioner is not None

                r = self._normalized_adjoint_field(
                    y,
                    b_solver,
                    regions,
                    coverage=coverage,
                )
                residual_fields.append(r)

                m = self.preconditioner(f, y, r)
                preconditioner_fields.append(m)

                if self.rmr_update_rule == "jacobian":
                    z = z - eta * m * torch.sigmoid(z) * r
                elif self.rmr_update_rule == "latent":
                    z = z - eta * m * r
                else:
                    raise RuntimeError(f"unsupported RMR update rule {self.rmr_update_rule}")

                y = F.softplus(z)

            elif self.variant == "local_refine":
                assert self.local_refiner is not None
                dz = self.local_refiner(f, y)

                if self.cfg.residual_clip > 0:
                    dz = dz.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)

                residual_fields.append(dz)
                z = z - eta * dz
                y = F.softplus(z)

            else:
                raise RuntimeError(f"Unknown variant {self.variant}")

            iterates.append(y)

        out["y"] = y

        if (
            (self.variant == "rmr" and self.rmr_update_rule == "projected_sirt")
            or self.variant == "learned_project"
        ):
            out["z"] = None
        else:
            out["z"] = z

        out["iterates"] = iterates
        out["residual_fields"] = residual_fields
        out["preconditioner_fields"] = preconditioner_fields
        out["step_sizes"] = step_sizes

        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
