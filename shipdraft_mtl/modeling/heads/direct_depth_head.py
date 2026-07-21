from __future__ import annotations

"""Neural aggregation of dual decoder queries into draft depth."""

from typing import Dict

import torch
from torch import nn


class DirectDepthHead(nn.Module):
    """Predict depth from character and waterline query representations."""

    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 256,
        num_character_queries: int = 100,
        min_depth: float = 0.0,
        max_depth: float = 5.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_depth <= min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.num_character_queries = int(num_character_queries)

        self.query_projection = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.GELU(),
        )
        self.task_projection = nn.Sequential(
            nn.Linear(2, hidden_dim // 8),
            nn.GELU(),
        )
        fused_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 8
        self.query_fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.waterline_attention = nn.Linear(hidden_dim, 1)
        self.scale_attention = nn.Linear(hidden_dim, 1)
        self.depth_context = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.depth = nn.Linear(hidden_dim, 1)
        self.valid = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _soft_pool(features: torch.Tensor, logits: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(logits + torch.log(prior.clamp_min(1e-6)), dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1), weights

    def forward(self, query_features: torch.Tensor, pred_points: torch.Tensor) -> Dict[str, torch.Tensor]:
        if query_features.ndim != 3:
            raise ValueError(f"query_features must be [B,Q,C], got {tuple(query_features.shape)}")
        if pred_points.shape[:2] != query_features.shape[:2] or pred_points.shape[-1] != 2:
            raise ValueError("pred_points must have shape [B,Q,2] matching query_features")

        batch, queries, _ = query_features.shape
        character_count = min(self.num_character_queries, queries)
        query_features = self.query_projection(query_features)
        coordinates = self.coordinate_projection(pred_points.clamp(0.0, 1.0))

        task_type = query_features.new_zeros(batch, queries, 2)
        task_type[:, :character_count, 0] = 1.0
        task_type[:, character_count:, 1] = 1.0
        tasks = self.task_projection(task_type)
        fused = self.query_fusion(torch.cat([query_features, coordinates, tasks], dim=-1))

        scale_prior = fused.new_zeros(batch, queries)
        waterline_prior = fused.new_zeros(batch, queries)
        scale_prior[:, :character_count] = 1.0
        waterline_prior[:, character_count:] = 1.0
        waterline_context, waterline_weights = self._soft_pool(
            fused,
            self.waterline_attention(fused).squeeze(-1),
            waterline_prior,
        )
        scale_context, scale_weights = self._soft_pool(
            fused,
            self.scale_attention(fused).squeeze(-1),
            scale_prior,
        )
        global_context = fused.mean(dim=1)
        relation_context = waterline_context - scale_context
        context = self.depth_context(
            torch.cat([waterline_context, scale_context, relation_context, global_context], dim=-1)
        )

        normalized_depth = torch.sigmoid(self.depth(context).squeeze(-1))
        depth = self.min_depth + (self.max_depth - self.min_depth) * normalized_depth
        return {
            "pred_depth": depth,
            "pred_depth_valid_logits": self.valid(context).squeeze(-1),
            "pred_depth_waterline_weights": waterline_weights,
            "pred_depth_scale_weights": scale_weights,
        }


def build_direct_depth_head(config=None, query_dim: int = 256, num_character_queries: int = 100):
    cfg = dict(config or {})
    cfg.pop("name", None)
    allowed = {"hidden_dim", "num_character_queries", "min_depth", "max_depth", "dropout"}
    return DirectDepthHead(
        query_dim=query_dim,
        num_character_queries=int(cfg.get("num_character_queries", num_character_queries)),
        **{key: value for key, value in cfg.items() if key in allowed and key != "num_character_queries"},
    )
