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

RMRUpdate = Literal["latent", "jacobian", "projected_sirt"]

# Softplus inverse: softplus(x) = log(exp(x)-1), softplus^{-1}(y) = log(exp(y)-1).
# For target initial rate mu0 = 0.01 count/cell:
# softplus^{-1}(0.01) = log(exp(0.01)-1) ≈ -4.595
_FINE_HEAD_BIAS_INIT: float = math.log(math.exp(0.01) - 1.0)   # ≈ -4.595


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
    """Fine-grained density head with calibrated initial rate.

    P0 fix: The final Conv2d bias is initialized so that:
        softplus(bias) ≈ mu0 = 0.01 count/cell
    i.e. bias ≈ log(exp(0.01) - 1) ≈ -4.595.
    Weights of the last conv are also scaled down to avoid large initial variance.
    This prevents the "initialization MAE ≈ 0.693 × N_cells" problem.
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(width, width, 3, groups=width),
            ConvGNAct(width, width, 1),
            nn.Conv2d(width, 1, 1),
        )
        # Initialize final conv: small weights + calibrated bias.
        final_conv: nn.Conv2d = self.body[-1]  # type: ignore[assignment]
        nn.init.normal_(final_conv.weight, std=0.01)
        nn.init.constant_(final_conv.bias, _FINE_HEAD_BIAS_INIT)  # type: ignore[arg-type]

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.body(f)


class RegionalEvidenceHead(nn.Module):
    """Shared regional count regressor: predicts rate rho_R then scales by area.

    P0 fix (rate × area formulation):
        rho_R = softplus(MLP(avg_feat_R, geom_R))   [counts/cell]
        b_R   = |R| * rho_R                          [total counts in region]

    This enforces extensivity by construction: same visual crowd density in a
    larger region correctly predicts a proportionally larger count.

    P0 #1 fix (bias calibration):
        Final Linear bias initialized to _FINE_HEAD_BIAS_INIT ≈ -4.595 so that
        softplus(bias) ≈ 0.01 count/cell at initialization — same density prior
        as FineMeasureHead. Without this, softplus(0) = 0.693 per cell, which
        causes b_128 ≈ 710 while Y_0 ≈ 164 → solver tries to inject mass.

    P1 #1 fix (position-free geometry, geom_dim=4):
        geom = [log(h), log(w), log(|R|), log(w/h)]
        Positional terms cy/H, cx/W are dropped because they change value for the
        SAME physical region depending on whether it sits inside a random 512-crop
        (training), a full-resolution image (direct eval), or a tile (tiled eval).
        This mismatch creates three different "geometric identities" for the same
        window and contaminates the regional evidence head with spurious position cues.
        Positional ablation can be added later with globally-consistent coordinates.
    """

    def __init__(self, feature_dim: int = 32, hidden: int = 48, geom_dim: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + geom_dim, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # P0 #1 fix: calibrate final Linear so initial rate ≈ 0.01 count/cell.
        # softplus^{-1}(0.01) = log(exp(0.01) - 1) ≈ -4.595 = _FINE_HEAD_BIAS_INIT.
        final_linear: nn.Linear = self.mlp[-1]  # type: ignore[assignment]
        nn.init.normal_(final_linear.weight, std=0.01)
        nn.init.constant_(final_linear.bias, _FINE_HEAD_BIAS_INIT)

    def forward(self, f: torch.Tensor, regions: RegionSet) -> torch.Tensor:
        b, _, h, w = f.shape
        pooled = region_average_features(f, regions.boxes)  # [B,M,C]
        geom = region_geometry(regions.boxes, h, w).to(dtype=f.dtype)
        geom = geom.unsqueeze(0).expand(b, -1, -1)
        raw = self.mlp(torch.cat([pooled, geom], dim=-1)).squeeze(-1)  # [B,M]
        # rate per cell (positive), then scale by region area for total count
        area = regions.area.to(dtype=f.dtype).view(1, -1)  # [1,M]
        rate = F.softplus(raw)                              # [B,M] rate per cell
        b_region = rate * area                              # [B,M] total count
        return b_region.unsqueeze(1)                        # [B,1,M]


class LocalPreconditioner(nn.Module):
    """Small bounded local preconditioner applied after the normalized adjoint field.

    This implements M^(t) in [0.25, 1.75] as described in the paper.
    The update formula is the diagonally preconditioned adjoint (not raw gradient):
        r = D_c^{-1} A^T D_a^{-1} (AY - b)
    where D_a = region-area normalization, D_c = overlap-coverage normalization.
    The preconditioner M further shapes this coverage-normalized adjoint field.
    """

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

    Exact RMR uses the normalized adjoint (D_c^{-1} A^T D_a^{-1}), i.e. uniform
    allocation delta_R / |R| before overlap averaging, followed by a small local
    preconditioner. B3b therefore has access to the same regional information and scope
    but is allowed to learn the region-to-grid allocation geometry.

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
        # The post-net is allowed to shape the learned projection further.
        return self.post(torch.cat([f, y, field], dim=1))


@dataclass
class RMRConfig:
    output_stride: int = 4
    feature_width: int = 32
    # Pilot: {32, 64, 128} px only. include_full_image=False avoids:
    #  (a) extent extrapolation bug: full-image descriptor is resolution-invariant by design
    #      but count is proportional to area — they conflict.
    #  (b) direct/tiled inconsistency: full-image region changes between whole-image and tiled inference.
    #  (c) scale imbalance in loss: one full-image term dominates scale-balanced loss.
    # Global supervision is handled separately by global_count_loss.
    region_sizes_px: tuple[int, ...] = (32, 64, 128)
    region_overlap: float = 0.5
    include_full_image: bool = False   # P0 fix: False for registered pilot
    iterations: int = 2

    # Stability: bounded step size with small initialization.
    eta_max: float = 0.20
    eta_init: float = 0.05
    residual_clip: float = 5.0

    eps: float = 1e-6
    # Update rule selection:
    #   "latent":         z^{t+1} = z^t - eta * M * r  (legacy default)
    #   "jacobian":       z^{t+1} = z^t - eta * M * sigma(z) * r  (ablation)
    #   "projected_sirt": Y^{t+1} = max(0, Y^t - omega * M * r)  (RMR-P registered)
    update_rule: RMRUpdate = "latent"
    # Legacy ablation flag: if True and update_rule=="latent", sets update_rule to "jacobian"
    use_jacobian_gate: bool = False

    # Projected SIRT parameters (RMR-P):
    sirt_omega: float = 1.0
    learnable_sirt_omega: bool = False
    projected_use_preconditioner: bool = False
    detach_region_evidence: bool = True


class RMRCount(nn.Module):
    """Regional Measure Reconciliation crowd counter and registered controls.

    The core RMR residual field uses the diagonally preconditioned adjoint:
        r = D_c^{-1} A^T D_a^{-1} (AY - b)
    where:
        A:   rectangular region summation operator (exact)
        A^T: exact adjoint (implemented via 2D difference arrays, O(M+HW))
        D_a: diagonal matrix of region areas (converts count residual -> rate residual)
        D_c: diagonal matrix of overlap coverage counts (averages overlapping regions)

    Update rules:

        RMR-P (update_rule="projected_sirt", REGISTERED MAIN METHOD):
            Y^{t+1} = \\Pi_+ [Y^t - \\omega M r_t] = \\max(0, Y^t - \\omega M r_t)
            Direct measure-space Nonnegative Projected SIRT with fixed \\omega=1 and M=1.
            No learned spatial preconditioner, no sigmoid Jacobian factor, no extra solver
            parameters. Exact parameter parity with B2 (region_aux).

        RMR-Latent (update_rule="latent", legacy mechanism ablation):
            z^{t+1} = z^t - eta * M * r
            y^{t+1} = softplus(z^{t+1})
            Historical latent-space preconditioned reconciliation step.

        RMR-Jacobian (update_rule="jacobian", legacy mechanism ablation):
            z^{t+1} = z^t - eta * M * sigma(z) * r
            y^{t+1} = softplus(z^{t+1})
            Historical Jacobian-gated latent rule.

    P0 checkpoint guard: best_val_mae.pt is only updated after solver_strength
    reaches 1.0, preventing reproducing-when-loaded inconsistency.
    """

    def __init__(self, cfg: RMRConfig = RMRConfig(), variant: Variant = "rmr"):
        super().__init__()
        # P0 guard: TinyLocalEncoder is hardwired to output_stride=4 (stem=s2, body=s4 total).
        # Any other value in the config would silently produce mismatched tensor shapes.
        if cfg.output_stride != 4:
            raise ValueError(
                f"RMRCount only supports output_stride=4 "
                f"(TinyLocalEncoder is hardwired to stride 4). "
                f"Got output_stride={cfg.output_stride}."
            )
        self.cfg = cfg
        self.variant = variant

        needs_region_head = variant in {
            "region_aux",
            "learned_project",
            "rmr",
        }
        self.region_head = (
            RegionalEvidenceHead(cfg.feature_width)
            if needs_region_head
            else None
        )

        effective_rule = cfg.update_rule
        if cfg.use_jacobian_gate and effective_rule == "latent":
            effective_rule = "jacobian"
        self.rmr_update_rule: RMRUpdate = effective_rule
        self.update_rule: RMRUpdate = effective_rule

        self.encoder = TinyLocalEncoder()
        self.fusion = AdditiveFusion(cfg.feature_width)
        self.fine_head = FineMeasureHead(cfg.feature_width)

        # Legacy RMR-Latent / RMR-Jacobian always use the learned positive
        # local preconditioner.
        # Registered Projected-SIRT uses M=1 unless explicitly ablated.
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

        # P1 fix: Only allocate eta_logits for variants and rules that actually use it.
        # Registered Projected-SIRT uses fixed omega=1 (or optional log_sirt_omega ablation)
        # and has zero extra trainable solver parameters, achieving exact parameter parity with B2.
        need_eta = (variant in {"local_refine", "learned_project"}) or (
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

        # Optional learnable relaxation for an ablation only.
        # Main method uses fixed omega=1 and has no extra solver parameter.
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
        self._cache: dict[
            tuple,
            tuple[RegionSet, torch.Tensor]
        ] = {}

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
        if key not in self._cache:
            regions = build_multiscale_regions(
                height=h,
                width=w,
                output_stride=self.cfg.output_stride,
                region_sizes_px=self.cfg.region_sizes_px,
                overlap=self.cfg.region_overlap,
                include_full_image=self.cfg.include_full_image,
                device=device,
            )
            # Coverage is D_c = A^T 1_M on this grid
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
            # Positive relaxation ablation.
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
        """One nonnegative projected RMR/SIRT reconciliation step.

        r = Dc^-1 A^T Da^-1 (AY - b)

        Main registered method:
            M = 1
            omega = 1
            y_next = clamp_min(y - omega * r, 0)

        solver_strength multiplies omega during warm-up/ramp.

        Returns:
            y_next
            residual field r
            preconditioner field M
            effective omega
        """
        r = self._normalized_adjoint_field(
            y,
            b_region,
            regions,
            coverage=coverage,
        )

        if self.cfg.projected_use_preconditioner:
            if self.preconditioner is None:
                raise RuntimeError(
                    "projected_use_preconditioner=True "
                    "but no LocalPreconditioner was constructed"
                )
            if features is None:
                raise ValueError(
                    "features required for learned preconditioner"
                )
            m = self.preconditioner(
                features,
                y,
                r,
            )
        else:
            # Exact registered solver: no learned spatial gate.
            m = torch.ones_like(y)

        omega = self._sirt_omega(device=y.device, dtype=y.dtype)
        omega_eff = omega * float(self.solver_strength)

        y_next = torch.clamp_min(
            y - omega_eff * m * r,
            0.0,
        )

        return y_next, r, m, omega_eff

    def _raw_region_delta(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
    ) -> torch.Tensor:
        return regional_sum(y, regions.boxes) - b_region

    def _normalized_adjoint_field(
        self,
        y: torch.Tensor,
        b_region: torch.Tensor,
        regions: RegionSet,
        coverage: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Diagonally preconditioned adjoint: D_c^{-1} A^T D_a^{-1} (AY - b).

        D_a: area normalization  — converts raw count residual to per-cell residual density.
        D_c: coverage normalization — averages overlapping region contributions per cell.

        This is NOT the raw gradient A^T (AY - b). The diagonal preconditioning makes
        the update scale-invariant with respect to region size and overlap density.
        """
        _, _, h, w = y.shape
        delta = self._raw_region_delta(y, b_region, regions)   # [B,1,M]
        area = regions.area.to(y.dtype).view(1, 1, -1)
        residual_density = delta / area.clamp_min(1.0)          # D_a^{-1} (AY-b)

        back = regional_adjoint(residual_density, regions.boxes, h, w)
        if coverage is None:
            coverage = self._regions_and_coverage(h, w, y.device)[1]
        r = back / coverage.to(y.dtype)                      # D_c^{-1} A^T
        if self.cfg.residual_clip > 0:
            r = r.clamp(-self.cfg.residual_clip, self.cfg.residual_clip)
        return r

    # Keep legacy name for backward compatibility in tests.
    _rmr_field = _normalized_adjoint_field

    def forward(
        self,
        x: torch.Tensor,
        *,
        b_region_override: torch.Tensor | None = None,
        shuffle_region: bool = False,
    ) -> dict[str, torch.Tensor | RegionSet | None | list]:
        c4, c8, c16 = self.encoder(x)
        f = self.fusion(c4, c8, c16)
        z0 = self.fine_head(f)
        y0 = F.softplus(z0)
        h, w = y0.shape[-2:]

        # True Lazy Region Construction:
        # Only construct multiscale regions if the variant actually uses them.
        # Direct (B0) and local_refine (B3a) variants do not need regions at all.
        # Skipping region construction for them:
        #  - saves training memory and computation
        #  - avoids inflating baseline latency
        #  - makes RMR overhead appear genuine rather than hidden in all variants
        needs_regions = self.region_head is not None or self.variant in {
            "region_loss", "learned_project", "rmr"
        }
        if needs_regions:
            regions, coverage = self._regions_and_coverage(h, w, x.device)
        else:
            regions, coverage = None, None

        out: dict = {
            "features": f,
            "z0": z0,
            "y0": y0,
            "regions": regions,
            "preconditioner_fields": [],
            "step_sizes": [],
        }

        if self.region_head is not None:
            assert regions is not None
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
            raise RuntimeError(
                f"variant {self.variant} requires regional evidence"
            )

        # --------------------------------------------------------------
        # Solver measurement.
        #
        # Keep out["b_region"] connected to the regional head so that
        # L_region_head trains it normally.
        #
        # Only the copy used inside RMR is detached for the registered
        # causal B2-vs-B5 comparison.
        # --------------------------------------------------------------
        if (
            self.variant == "rmr"
            and b_region is not None
            and self.cfg.detach_region_evidence
        ):
            b_solver = b_region.detach()
        else:
            b_solver = b_region

        for t in range(self.cfg.iterations):

            # ==========================================================
            # Registered upgraded RMR-P
            # ==========================================================
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
                step_sizes.append(
                    float(omega_eff.detach().item())
                )

                iterates.append(y)
                continue

            # ==========================================================
            # Legacy / learned-control latent rules
            # ==========================================================
            eta = self._eta(t) * self.solver_strength
            step_sizes.append(
                float(eta.detach().item())
            )

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

                m = self.preconditioner(
                    f,
                    y,
                    r,
                )
                preconditioner_fields.append(m)

                if self.rmr_update_rule == "jacobian":
                    z = (
                        z
                        - eta
                        * m
                        * torch.sigmoid(z)
                        * r
                    )
                elif self.rmr_update_rule == "latent":
                    z = z - eta * m * r
                else:
                    raise RuntimeError(
                        f"unsupported RMR update rule "
                        f"{self.rmr_update_rule}"
                    )

                y = F.softplus(z)

            elif self.variant == "learned_project":
                assert b_region is not None
                assert regions is not None
                assert self.learned_projector is not None

                delta = self._raw_region_delta(
                    y,
                    b_region,
                    regions,
                )
                learned_field = self.learned_projector.project(
                    f,
                    y,
                    delta,
                    regions,
                )

                if self.cfg.residual_clip > 0:
                    learned_field = learned_field.clamp(
                        -self.cfg.residual_clip,
                        self.cfg.residual_clip,
                    )

                residual_fields.append(
                    learned_field
                )

                z = z - eta * learned_field
                y = F.softplus(z)

            elif self.variant == "local_refine":
                assert self.local_refiner is not None

                dz = self.local_refiner(
                    f,
                    y,
                )

                if self.cfg.residual_clip > 0:
                    dz = dz.clamp(
                        -self.cfg.residual_clip,
                        self.cfg.residual_clip,
                    )

                residual_fields.append(dz)

                z = z - eta * dz
                y = F.softplus(z)

            else:
                raise RuntimeError(
                    f"Unknown variant {self.variant}"
                )

            iterates.append(y)

        out["y"] = y

        # Projected-SIRT optimizes Y directly.
        # There is no meaningful final latent z after Y0.
        if (
            self.variant == "rmr"
            and self.rmr_update_rule == "projected_sirt"
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
