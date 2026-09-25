from __future__ import annotations

import torch
from torch import nn

class EFRegressionHead(nn.Module):
    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings).squeeze(-1)

def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False

def trainable_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
