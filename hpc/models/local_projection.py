import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalProjectionHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        projection_dim: int = 32,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)
