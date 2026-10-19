from __future__ import annotations

import torch
import torch.nn.functional as F


def balanced_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
    stride: int = 4,
) -> torch.Tensor:
    """Equalize empty and non-empty cell contributions with strict per-sample isolation."""
    if pred.numel() == 0 or target.numel() == 0:
        return (pred.sum() + target.sum()) * 0.0

    if pred.ndim == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
    elif pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if target.ndim == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.ndim == 3:
        target = target.unsqueeze(1)

    per = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=beta)
    b_sz = pred.shape[0]

    sample_losses = []
    for i in range(b_sz):
        per_i = per[i]
        tgt_i = target[i]
        pos_i = tgt_i > 0
        neg_i = ~pos_i
        has_pos = pos_i.any()
        has_neg = neg_i.any()
        if has_pos and has_neg:
            pos_loss_i = per_i[pos_i].mean()
            neg_loss_i = per_i[neg_i].mean()
            sample_losses.append(0.5 * (pos_loss_i + neg_loss_i))
        elif has_pos:
            sample_losses.append(per_i[pos_i].mean())
        elif has_neg:
            sample_losses.append(per_i[neg_i].mean())
        else:
            sample_losses.append(per_i.new_tensor(0.0))

    return torch.stack(sample_losses).mean() if sample_losses else (pred.sum() + target.sum()) * 0.0


__all__ = ["balanced_smooth_l1"]
