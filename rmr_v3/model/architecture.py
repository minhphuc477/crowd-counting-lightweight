from __future__ import annotations

from collections import OrderedDict
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from rmr_core.backbones import TimmPyramidBackbone
from rmr_core.necks import CoordinateAttention
from rmr_core.types import RMRModelOutput
from rmr_core.operators import RegionSet, build_multiscale_regions
from .config import (
    RMRv3Config,
    _deep_tuple,
    validate_architecture_contract,
)
from .builder import (
    build_neck_module,
    build_scale_router,
    build_1x1_gate,
    build_fine_carrier_head,
    build_regional_evidence_head,
    package_subpixel_output,
)
from .evidence import extract_regional_evidence
from .perspective_geometry import (
    DynamicCameraAnglePredictor,
    DiAGScaleRoutingHead,
    DiAGFactorizedRoutingHead,
)
from .solver_step import solve_inverse_measure
from .dual_lattice import scale_regions_to_stride2, SubpixelAllocationHead


class RMRv3(nn.Module):
    """Reliability-Weighted Regional Measure Reconciliation."""

    def __init__(
        self,
        cfg: RMRv3Config | None = None,
    ) -> None:
        super().__init__()

        if cfg is None:
            cfg = RMRv3Config()
        validate_architecture_contract(cfg)
        self.cfg = cfg

        self.encoder = TimmPyramidBackbone(
            model_name=cfg.backbone_name,
            pretrained=cfg.pretrained,
            target_reductions=(4, 8, 16),
        )
        self.fusion = build_neck_module(self.encoder.out_channels, cfg)
        self.fine_head = build_fine_carrier_head(cfg)
        self.region_head = build_regional_evidence_head(cfg)
        self.coord_attn = CoordinateAttention(channels=cfg.feature_width, reduction=4) if cfg.use_coord_attn else None

        # Dynamic Scale Routing & DiAG (100% Feature-Driven, Zero Coordinate Linspace)
        self.dcap, self.scale_router = build_scale_router(cfg)
        self.fg_gate = build_1x1_gate(cfg.foreground_gate, cfg.feature_width)
        self.tdsg = build_1x1_gate(cfg.use_top_down_semantic_gate, cfg.feature_width)

        self.trust_gate = nn.Linear(cfg.feature_width, 1) if cfg.dynamic_trust_gate else None
        if self.trust_gate is not None:
            nn.init.zeros_(self.trust_gate.weight)
            nn.init.constant_(self.trust_gate.bias, float(cfg.trust_gate_init_bias))

        self.solver_strength: float = 1.0
        self.subpixel_allocator = (
            SubpixelAllocationHead(cfg.feature_width) if getattr(cfg, "subpixel_dm", False) else None
        )
        lap_k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("_laplace_kernel", lap_k, persistent=False)
        self._region_cache: OrderedDict[tuple, RegionSet] = OrderedDict()

        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.cfg.max_trainable_params > 0 and total_trainable > self.cfg.max_trainable_params:
            raise ValueError(
                f"Strict parameter ceiling exceeded: {total_trainable} > {self.cfg.max_trainable_params} parameters. "
                "Check architecture configuration."
            )

    def set_solver_strength(self, strength: float) -> None:
        self.solver_strength = float(min(max(strength, 0.0), 1.0))

    def _regions(self, h: int, w: int, device: torch.device, stride: int | None = None) -> RegionSet:
        effective_stride = self.cfg.output_stride if stride is None else stride
        dev_idx = (device.index if device.index is not None else 0) if device.type == "cuda" else None
        key = (
            h, w, effective_stride,
            _deep_tuple(self.cfg.region_sizes_px), self.cfg.region_overlap,
            device.type, dev_idx,
        )

        if key in self._region_cache:
            self._region_cache.move_to_end(key)
            return self._region_cache[key]

        if len(self._region_cache) >= 32:
            self._region_cache.popitem(last=False)

        region_set = build_multiscale_regions(
            height=h,
            width=w,
            output_stride=effective_stride,
            region_sizes_px=self.cfg.region_sizes_px,
            overlap=self.cfg.region_overlap,
            include_full_image=False,
            device=device,
        )
        self._region_cache[key] = region_set
        return region_set

    def _extract_carrier_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c4, c8, c16 = self.encoder(x)
        p4, p8, p16 = self.fusion(c4, c8, c16)

        if self.tdsg is not None:
            sem_logit = self.tdsg(p16)
            sem_gate = F.interpolate(
                sem_logit, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )
            sem_floor = float(self.cfg.tdsg_floor)
            sem_mask = sem_floor + (1.0 - sem_floor) * torch.sigmoid(sem_gate)
            p4 = p4 * sem_mask

        if self.coord_attn is not None:
            p4 = self.coord_attn(p4)

        return p4, p8, p16

    def _route_scales(
        self,
        p4: torch.Tensor,
        delta_scale: torch.Tensor | None = None,
        scene_tilt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.scale_router is None:
            return None, None, None
        if isinstance(self.scale_router, (DiAGScaleRoutingHead, DiAGFactorizedRoutingHead)):
            router_out = self.scale_router(p4, delta_scale=delta_scale, scene_tilt=scene_tilt)
        else:
            router_out = self.scale_router(p4)
        if isinstance(router_out, tuple):
            return router_out[0], router_out[1], router_out[2]
        return router_out, None, None

    def _predict_fine_density(
        self, p4: torch.Tensor, scale_weights: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        z0 = self.fine_head.forward_logits(p4, scale_weights=scale_weights)
        y0 = self.fine_head.activate(z0, scale_weights=scale_weights)

        fg_logit = None
        if self.fg_gate is not None:
            fg_logit = self.fg_gate(p4)
            floor = float(self.cfg.fg_gate_floor)
            fg_mask = floor + (1.0 - floor) * torch.sigmoid(fg_logit)
            if fg_mask.shape[-2:] != y0.shape[-2:]:
                fg_mask = F.interpolate(
                    fg_mask, size=y0.shape[-2:], mode="bilinear", align_corners=False
                )
            y0 = y0 * fg_mask

        return z0, y0, fg_logit

    def _extract_regional_evidence(
        self, p4: torch.Tensor, p8: torch.Tensor, p16: torch.Tensor,
        regions: RegionSet, scale_weights: torch.Tensor | None,
        uniform_reliability: bool, grid_h: int,
    ) -> dict[str, Any]:
        return extract_regional_evidence(
            cfg=self.cfg, region_head=self.region_head, p4=p4, p8=p8, p16=p16,
            regions=regions, scale_weights=scale_weights,
            uniform_reliability=uniform_reliability, grid_h=grid_h,
        )

    def _solve_inverse_measure(
        self,
        y0: torch.Tensor,
        z0: torch.Tensor,
        regional_evidence: dict[str, Any],
        regions: RegionSet,
        scale_weights: torch.Tensor | None,
        solver_strength: float | None,
        uniform_reliability: bool,
        p16: torch.Tensor,
        fg_logit: torch.Tensor | None,
        pi_scale: torch.Tensor | None,
        pi_aspect: torch.Tensor | None,
        carrier_energy: torch.Tensor | None = None,
        compute_energy: bool = True,
    ) -> RMRModelOutput:
        return solve_inverse_measure(
            cfg=self.cfg, y0=y0, z0=z0, regional_evidence=regional_evidence,
            regions=regions, scale_weights=scale_weights, solver_strength=solver_strength,
            default_solver_strength=self.solver_strength, uniform_reliability=uniform_reliability,
            p16=p16, fg_logit=fg_logit, pi_scale=pi_scale, pi_aspect=pi_aspect,
            trust_gate=self.trust_gate, laplace_kernel=self._laplace_kernel,
            carrier_energy=carrier_energy, compute_energy=compute_energy,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        uniform_reliability: bool = False,
        solver_strength: float | None = None,
        compute_energy: bool = True,
    ) -> RMRModelOutput:
        h_in, w_in = x.shape[-2:]
        divisor = 16  # FPN reductions (4, 8, 16) require resolution divisibility
        pad_h = (divisor - h_in % divisor) % divisor
        pad_w = (divisor - w_in % divisor) % divisor
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0) if (pad_h > 0 or pad_w > 0) else x

        p4, p8, p16 = self._extract_carrier_features(x_in)
        scene_tilt, delta_scale = None, None
        if self.dcap is not None:
            scene_tilt, delta_scale = self.dcap(p16)
        scale_weights, pi_scale, pi_aspect = self._route_scales(
            p4, delta_scale=delta_scale, scene_tilt=scene_tilt
        )
        z0, y0, fg_logit = self._predict_fine_density(p4, scale_weights)

        target_h4, target_w4 = (h_in + 3) // 4, (w_in + 3) // 4
        if pad_h > 0 or pad_w > 0 or p4.shape[-2] != target_h4 or p4.shape[-1] != target_w4:
            p4 = p4[..., :target_h4, :target_w4]
            p8 = p8[..., :(h_in + 7) // 8, :(w_in + 7) // 8]
            p16 = p16[..., :(h_in + 15) // 16, :(w_in + 15) // 16]
            if scale_weights is not None:
                scale_weights = scale_weights[..., :target_h4, :target_w4]
            if pi_scale is not None:
                pi_scale = pi_scale[..., :target_h4, :target_w4]
            if pi_aspect is not None:
                pi_aspect = pi_aspect[..., :target_h4, :target_w4]
            if fg_logit is not None:
                fg_logit = fg_logit[..., :target_h4, :target_w4]

        regions_feat = self._regions(target_h4, target_w4, x.device, stride=4)
        if self.cfg.subpixel_stride2:
            target_h, target_w = (h_in + 1) // 2, (w_in + 1) // 2
            if y0.shape[-2] != target_h or y0.shape[-1] != target_w:
                y0, z0 = y0[..., :target_h, :target_w], z0[..., :target_h, :target_w]
            regions_solver = scale_regions_to_stride2(regions_feat, target_h, target_w)
        else:
            if y0.shape[-2] != target_h4 or y0.shape[-1] != target_w4:
                y0, z0 = y0[..., :target_h4, :target_w4], z0[..., :target_h4, :target_w4]
            regions_solver = regions_feat

        regional_evidence = self._extract_regional_evidence(
            p4=p4,
            p8=p8,
            p16=p16,
            regions=regions_feat,
            scale_weights=scale_weights,
            uniform_reliability=uniform_reliability,
            grid_h=target_h4,
        )

        carrier_energy = None
        if self.cfg.resonant_adjoint or getattr(self.cfg, "crest_discovery_flux", False):
            b_c, c_p4, h_p4, w_p4 = p4.shape
            p4_flat_f32 = p4.float().view(b_c * c_p4, 1, h_p4, w_p4)
            kernel_f32 = self._laplace_kernel.to(device=p4.device, dtype=torch.float32)
            lap_p4 = F.conv2d(p4_flat_f32, kernel_f32, padding=1).view(b_c, c_p4, h_p4, w_p4)
            carrier_energy = (lap_p4 ** 2).mean(dim=1, keepdim=True)

        out = self._solve_inverse_measure(
            y0=y0, z0=z0, regional_evidence=regional_evidence, regions=regions_solver,
            scale_weights=scale_weights, solver_strength=solver_strength,
            uniform_reliability=uniform_reliability,
            p16=p16, fg_logit=fg_logit, pi_scale=pi_scale, pi_aspect=pi_aspect,
            carrier_energy=carrier_energy, compute_energy=compute_energy,
        )

        package_subpixel_output(
            out=out,
            p4=p4,
            allocator=self.subpixel_allocator,
            subpixel_stride2=self.cfg.subpixel_stride2,
            h_in=h_in,
            w_in=w_in,
            regions_solver=regions_solver,
        )

        if delta_scale is not None:
            out["delta_scale"] = delta_scale
        if scene_tilt is not None:
            out["scene_tilt"] = scene_tilt

        return out

    def switch_to_deploy(self) -> None:
        if hasattr(self.fusion, "switch_to_deploy"):
            self.fusion.switch_to_deploy()
