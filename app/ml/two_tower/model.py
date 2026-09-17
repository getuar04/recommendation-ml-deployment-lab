"""Small, explainable Two-Tower model (PyTorch, CPU-only).

Two independent towers -- Linear -> ReLU -> Linear -> L2-normalize -- projecting the
(different-dimensioned) user and content input vectors into the SAME embedding space, so
similarity (dot product == cosine similarity, since both sides are unit-normalized) is
meaningful between a user embedding and any content embedding. Deliberately shallow (spec:
"do NOT build a huge deep-learning architecture") -- two linear layers per tower is enough to
learn a nonlinear projection without needing GPU/large-batch training infrastructure.
"""
from __future__ import annotations

import torch
from torch import nn


class Tower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), p=2, dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, user_input_dim: int, content_input_dim: int, *,
                 hidden_dim: int = 64, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.user_tower = Tower(user_input_dim, hidden_dim, embedding_dim)
        self.content_tower = Tower(content_input_dim, hidden_dim, embedding_dim)

    def embed_user(self, user_x: torch.Tensor) -> torch.Tensor:
        return self.user_tower(user_x)

    def embed_content(self, content_x: torch.Tensor) -> torch.Tensor:
        return self.content_tower(content_x)

    def similarity(self, user_embedding: torch.Tensor, content_embedding: torch.Tensor) -> torch.Tensor:
        """Dot product of two unit-normalized vectors == cosine similarity, in [-1, 1]."""
        return (user_embedding * content_embedding).sum(dim=-1)

    def forward(self, user_x: torch.Tensor, content_x: torch.Tensor) -> torch.Tensor:
        return self.similarity(self.embed_user(user_x), self.embed_content(content_x))
