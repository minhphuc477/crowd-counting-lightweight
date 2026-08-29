from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hpc.models.local_projection import LocalProjectionHead


def pool_p4_to_16px_windows(
    p4: torch.Tensor,
    output_stride: int = 4,
    image_block_size: int = 16,
) -> torch.Tensor:
    k = image_block_size // output_stride
    return F.avg_pool2d(p4, kernel_size=k, stride=k)


def density_classes_from_exact_count(
    y16: torch.Tensor,
    t1: int,
    t2: int,
) -> torch.Tensor:
    label = torch.zeros_like(y16, dtype=torch.long)

    label[(y16 > 0) & (y16 <= t1)] = 1
    label[(y16 > t1) & (y16 <= t2)] = 2
    label[y16 > t2] = 3

    return label


def balanced_subsample_indices(
    labels: torch.Tensor,
    max_samples: int = 256,
    num_classes: int = 4,
) -> torch.Tensor:
    per_class = max(1, max_samples // num_classes)
    selected = []

    for c in range(num_classes):
        idx = torch.where(labels == c)[0]
        if idx.numel() == 0:
            continue

        if idx.numel() > per_class:
            perm = torch.randperm(
                idx.numel(), device=labels.device
            )[:per_class]
            idx = idx[perm]

        selected.append(idx)

    if not selected:
        return torch.empty(
            0, dtype=torch.long, device=labels.device
        )

    out = torch.cat(selected, dim=0)

    if out.numel() > max_samples:
        perm = torch.randperm(
            out.numel(), device=labels.device
        )[:max_samples]
        out = out[perm]

    return out


def supervised_contrastive_loss(
    z: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.10,
    eps: float = 1e-12,
) -> torch.Tensor:
    if z.shape[0] <= 1:
        return z.sum() * 0.0

    orig_dtype = z.dtype
    z = z.to(dtype=torch.float32)
    m = z.shape[0]

    logits = (z @ z.t()) / float(temperature)
    logits = logits - logits.max(
        dim=1, keepdim=True
    ).values.detach()

    eye = torch.eye(
        m, device=z.device, dtype=torch.bool
    )

    positive_mask = (
        labels[:, None] == labels[None, :]
    ) & (~eye)

    denominator_mask = ~eye

    exp_logits = torch.exp(logits) * denominator_mask

    log_prob = logits - torch.log(
        exp_logits.sum(dim=1, keepdim=True) + eps
    )

    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0

    if not valid_anchor.any():
        return z.sum() * 0.0

    mean_log_prob_pos = (
        (positive_mask * log_prob).sum(dim=1)
        / positive_count.clamp_min(1)
    )

    return -mean_log_prob_pos[valid_anchor].mean()


class LocalDensityContrastiveLoss(nn.Module):
    def __init__(
        self,
        feature_dim=32,
        hidden_dim=64,
        projection_dim=32,
        low_threshold=1,
        dense_threshold=4,
        max_samples=256,
        temperature=0.10,
    ):
        super().__init__()

        self.projector = LocalProjectionHead(
            feature_dim,
            hidden_dim,
            projection_dim,
        )

        self.low_threshold = low_threshold
        self.dense_threshold = dense_threshold
        self.max_samples = max_samples
        self.temperature = temperature

    def forward(self, p4, y16, valid16=None):
        pooled = pool_p4_to_16px_windows(p4)

        feat = pooled.permute(
            0, 2, 3, 1
        ).contiguous().view(
            -1, pooled.shape[1]
        )

        labels = density_classes_from_exact_count(
            y16,
            t1=self.low_threshold,
            t2=self.dense_threshold,
        ).reshape(-1)

        if valid16 is not None:
            valid = valid16.reshape(-1).bool()
            feat = feat[valid]
            labels = labels[valid]

        idx = balanced_subsample_indices(
            labels,
            max_samples=self.max_samples,
        )

        if idx.numel() <= 1:
            return feat.sum() * 0.0

        z = self.projector(feat[idx])

        return supervised_contrastive_loss(
            z,
            labels[idx],
            temperature=self.temperature,
        )
