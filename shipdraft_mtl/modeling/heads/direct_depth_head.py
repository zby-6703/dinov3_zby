from __future__ import annotations

"""Direct ROI-to-draft regression head.

The main task deliberately consumes image features only. Detection points,
character classes, and the auxiliary heatmap are supervised separately and do
not participate in the final depth computation.
"""

from typing import Dict

import torch
from torch import nn


class DirectDepthHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 256,
        hidden_dim: int = 256,
        min_depth: float = 0.0,
        max_depth: float = 5.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_depth <= min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.feature_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32 if hidden_dim % 32 == 0 else 8, hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.context = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.depth = nn.Linear(hidden_dim, 1)
        self.valid = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"DirectDepthHead expects [B,C,H,W], got {tuple(features.shape)}")
        projected = self.feature_proj(features)
        batch, channels, height, width = projected.shape
        flat = projected.flatten(2)
        weights = self.attention(projected).flatten(1).softmax(dim=-1)
        weighted = (flat * weights.unsqueeze(1)).sum(dim=-1)
        mean = flat.mean(dim=-1)
        maximum = flat.amax(dim=-1)
        context = self.context(torch.cat([weighted, mean, maximum], dim=-1))

        normalized_depth = torch.sigmoid(self.depth(context).squeeze(-1))
        depth = self.min_depth + (self.max_depth - self.min_depth) * normalized_depth
        return {
            "pred_depth": depth,
            "pred_depth_valid_logits": self.valid(context).squeeze(-1),
            "pred_depth_attention": weights.reshape(batch, 1, height, width),
        }


def build_direct_depth_head(config=None, in_channels: int = 256) -> DirectDepthHead:
    cfg = dict(config or {})
    cfg.pop("name", None)
    allowed = {"hidden_dim", "min_depth", "max_depth", "dropout"}
    return DirectDepthHead(
        in_channels=in_channels,
        **{key: value for key, value in cfg.items() if key in allowed},
    )
