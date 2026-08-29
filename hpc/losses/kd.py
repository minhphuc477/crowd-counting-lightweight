from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_groups(channels: int, max_groups: int = 8) -> int:
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return g


class Projector(nn.Module):
    def __init__(self, in_ch: int, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(_gn_groups(out_ch), out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resize_like(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if src.shape[-2:] == ref.shape[-2:]:
        return src
    return F.interpolate(
        src,
        size=ref.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def spatial_energy(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    e = x.float().pow(2).mean(dim=1, keepdim=True)
    denom = e.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
    return e / denom


def normalized_mass(d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    d = d.float().clamp_min(0)
    denom = d.sum(dim=(-1, -2), keepdim=True).clamp_min(eps)
    return d / denom


class MultiLevelKDLoss(nn.Module):
    """
    Multi-Level Knowledge Distillation Loss.
    All projectors are training-only and must not be exported with the student.

    Default channel assumptions for the planned setup:
      student P4/P8/P16 = 48/48/48 (ShuffleNet) or 32/32/32 (MobileNetV4)
      teacher P4/P8/P16 = 96/96/96
    """

    def __init__(
        self,
        student_channels: Dict[str, int] = {"p4": 48, "p8": 48, "p16": 48},
        teacher_channels: Dict[str, int] = {"p4": 96, "p8": 96, "p16": 96},
        kd_dim: int = 64,
        lambda_feat: float = 0.10,
        lambda_energy: float = 0.05,
        lambda_relation: float = 0.05,
        lambda_map: float = 0.20,
        lambda_count: float = 0.05,
        reliability_gate: bool = True,
        ramp_start: float = 0.05,
        ramp_full: float = 0.20,
    ):
        super().__init__()
        self.scales = ("p4", "p8", "p16")
        self.lambda_feat = lambda_feat
        self.lambda_energy = lambda_energy
        self.lambda_relation = lambda_relation
        self.lambda_map = lambda_map
        self.lambda_count = lambda_count
        self.reliability_gate = reliability_gate
        self.ramp_start = ramp_start
        self.ramp_full = ramp_full

        self.student_proj = nn.ModuleDict({
            k: Projector(student_channels[k], kd_dim)
            for k in self.scales
        })
        self.teacher_proj = nn.ModuleDict({
            k: Projector(teacher_channels[k], kd_dim)
            for k in self.scales
        })

    def _project_pair(
        self,
        key: str,
        student_feat: torch.Tensor,
        teacher_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.student_proj[key](student_feat)
        t = self.teacher_proj[key](teacher_feat.detach())
        t = _resize_like(t, s)
        return s, t

    def feature_loss(self, student_out: Dict[str, torch.Tensor], teacher_out: Dict[str, torch.Tensor]):
        losses = []
        projected = {}
        for key in self.scales:
            s, t = self._project_pair(key, student_out[key], teacher_out[key])
            projected[key] = (s, t)
            cos = F.cosine_similarity(s.float(), t.float(), dim=1)
            losses.append(1.0 - cos.mean())
        return torch.stack(losses).mean(), projected

    def energy_loss(self, student_out: Dict[str, torch.Tensor], teacher_out: Dict[str, torch.Tensor]):
        losses = []
        for key in self.scales:
            s = student_out[key]
            t = _resize_like(teacher_out[key].detach(), s)
            a_s = spatial_energy(s)
            a_t = spatial_energy(t)
            h, w = a_s.shape[-2:]
            losses.append(F.l1_loss(a_s, a_t) * float(h * w))
        return torch.stack(losses).mean()

    def relation_loss(self, projected: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        def relation(side: int):
            vecs = []
            for key in self.scales:
                x = projected[key][side]
                z = F.adaptive_avg_pool2d(x, 1).flatten(1).float()
                z = F.normalize(z, dim=1)
                vecs.append(z)
            z = torch.stack(vecs, dim=1)  # B x 3 x D
            return torch.bmm(z, z.transpose(1, 2))

        r_s = relation(0)
        r_t = relation(1).detach()
        return F.l1_loss(r_s, r_t)

    def map_loss(self, student_density: torch.Tensor, teacher_density: torch.Tensor):
        t = _resize_like(teacher_density.detach(), student_density)
        p_s = normalized_mass(student_density)
        p_t = normalized_mass(t)
        h, w = p_s.shape[-2:]
        return F.l1_loss(p_s, p_t) * float(h * w)

    def count_loss(
        self,
        student_density: torch.Tensor,
        teacher_count: torch.Tensor,
        gt_count: torch.Tensor,
    ):
        student_count = student_density.sum(dim=(-1, -2, -3))
        teacher_count = teacher_count.detach().to(student_count.device).float()
        gt_count = gt_count.to(student_count.device).float()

        residual = torch.abs(student_count - teacher_count) / torch.sqrt(gt_count + 1.0)

        if self.reliability_gate:
            teacher_error = torch.abs(teacher_count - gt_count)
            gate = torch.exp(-teacher_error / (gt_count + 10.0))
            residual = gate * residual

        return residual.mean()

    def kd_ramp(self, progress: float) -> float:
        if progress < self.ramp_start:
            return 0.0
        if progress < self.ramp_full:
            return float((progress - self.ramp_start) / max(self.ramp_full - self.ramp_start, 1e-6))
        return 1.0

    def forward(
        self,
        student_out: Dict[str, torch.Tensor],
        teacher_out: Dict[str, torch.Tensor],
        gt_count: torch.Tensor,
        progress: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_feat, projected = self.feature_loss(student_out, teacher_out)
        l_energy = self.energy_loss(student_out, teacher_out)
        l_rel = self.relation_loss(projected)
        l_map = self.map_loss(student_out["density"], teacher_out["density"])
        
        teacher_cnt = teacher_out.get("count_map", teacher_out.get("density").sum(dim=(-1, -2, -3)))
        l_count = self.count_loss(
            student_out["density"],
            teacher_cnt,
            gt_count,
        )

        raw = (
            self.lambda_feat * l_feat
            + self.lambda_energy * l_energy
            + self.lambda_relation * l_rel
            + self.lambda_map * l_map
            + self.lambda_count * l_count
        )

        ramp = self.kd_ramp(progress)
        total = ramp * raw

        details = {
            "kd_total": total.detach(),
            "kd_raw": raw.detach(),
            "kd_ramp": student_out["density"].new_tensor(ramp),
            "kd_feat": l_feat.detach(),
            "kd_energy": l_energy.detach(),
            "kd_relation": l_rel.detach(),
            "kd_map": l_map.detach(),
            "kd_count": l_count.detach(),
        }
        return total, details
