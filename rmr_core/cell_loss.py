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
    per = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=beta)
    b_sz = pred.shape[0] if pred.ndim >= 3 else 1
    if b_sz <= 1:
        pos = target > 0
        neg = ~pos
        pos_loss = per[pos].mean() if pos.any() else per.new_tensor(0.0)
        neg_loss = per[neg].mean() if neg.any() else per.new_tensor(0.0)
        return 0.5 * (pos_loss + neg_loss)

    sample_losses = []
    for i in range(b_sz):
        per_i = per[i]
        tgt_i = target[i]
        pos_i = tgt_i > 0
        neg_i = ~pos_i
        pos_loss_i = per_i[pos_i].mean() if pos_i.any() else per_i.new_tensor(0.0)
        neg_loss_i = per_i[neg_i].mean() if neg_i.any() else per_i.new_tensor(0.0)
        sample_losses.append(0.5 * (pos_loss_i + neg_loss_i))
    return torch.stack(sample_losses).mean()


__all__ = ["balanced_smooth_l1"]
