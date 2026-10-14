"""SignNet: a small 1D-convolution + Transformer classifier over feature sequences.

Input  "features" [B, T, F] raw features from features.py (normalisation is inside the model).
Output "logits"   [B, C].
Only ops that export cleanly to ONNX opset 17 and run in Unity Sentis.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        inner = dim * expand
        self.norm = nn.LayerNorm(dim)
        self.expand = nn.Linear(dim, inner)
        self.depthwise = nn.Conv1d(inner, inner, kernel, padding=kernel // 2, groups=inner)
        self.bn = nn.BatchNorm1d(inner)
        self.project = nn.Linear(inner, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.expand(self.norm(x)))
        h = self.act(self.bn(self.depthwise(h.transpose(1, 2)))).transpose(1, 2)
        return x + self.drop(self.project(h))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * expand), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * expand, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        return x + self.drop(self.mlp(self.norm2(x)))


class SignNet(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dim: int = 192, stages: int = 3, conv_per_stage: int = 2,
                 kernel_size: int = 11, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("inv_std", torch.ones(in_dim))
        self.stem = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        blocks: list[nn.Module] = []
        for _ in range(stages):
            blocks += [ConvBlock(dim, kernel_size, dropout=dropout) for _ in range(conv_per_stage)]
            blocks.append(TransformerBlock(dim, heads, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.inv_std.copy_(1.0 / torch.as_tensor(np.maximum(std, 1e-3), dtype=torch.float32))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = (features - self.mean) * self.inv_std
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=1))
