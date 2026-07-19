from __future__ import annotations

"""Neural aggregation of decoder queries into the final draft depth."""

from typing import Dict

import torch
from torch import nn


class DirectDepthHead(nn.Module):
    """Predict depth from continuous decoder query representations.

    The head never decodes a character string or fits a geometric line. Point
    coordinates and class probabilities are continuous inputs to the MLP, so
    the depth loss can improve the same queries used by the auxiliary point and
    classification tasks.
    """

    def __init__(
        self,
        query_dim: int = 256,
        hidden_dim: int = 256,
        num_classes: int = 10,
        waterline_class_id: int = 9,
        min_depth: float = 0.0,
        max_depth: float = 5.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if max_depth <= min_depth:
            raise ValueError("max_depth must be greater than min_depth")
        if not 0 <= waterline_class_id < num_classes:
            raise ValueError("waterline_class_id must be a valid class index")
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.num_classes = int(num_classes)
        self.waterline_class_id = int(waterline_class_id)

        self.query_projection = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.GELU(),
        )
        self.class_projection = nn.Sequential(
            nn.Linear(num_classes, hidden_dim // 2),
            nn.GELU(),
        )
        fused_dim = hidden_dim + hidden_dim // 4 + hidden_dim // 2
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

    def forward(
        self,
        query_features: torch.Tensor,
        pred_points: torch.Tensor,
        pred_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if query_features.ndim != 3:
            raise ValueError(f"query_features must be [B,Q,C], got {tuple(query_features.shape)}")
        if pred_points.shape[:2] != query_features.shape[:2] or pred_points.shape[-1] != 2:
            raise ValueError("pred_points must have shape [B,Q,2] matching query_features")
        if pred_logits.shape[:2] != query_features.shape[:2] or pred_logits.shape[-1] != self.num_classes:
            raise ValueError("pred_logits must have shape [B,Q,num_classes] matching query_features")

        class_probabilities = pred_logits.sigmoid()
        objectness = class_probabilities.amax(dim=-1)
        waterline_prior = class_probabilities[..., self.waterline_class_id] * objectness
        scale_prior = (1.0 - class_probabilities[..., self.waterline_class_id]) * objectness

        query_features = self.query_projection(query_features)
        coordinates = pred_points.clamp(0.0, 1.0)
        coordinates = self.coordinate_projection(coordinates)
        classes = self.class_projection(class_probabilities)
        fused = self.query_fusion(torch.cat([query_features, coordinates, classes], dim=-1))

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


def build_direct_depth_head(config=None, query_dim: int = 256, num_classes: int = 10, waterline_class_id: int = 9):
    cfg = dict(config or {})
    cfg.pop("name", None)
    allowed = {"hidden_dim", "min_depth", "max_depth", "dropout"}
    return DirectDepthHead(
        query_dim=query_dim,
        num_classes=num_classes,
        waterline_class_id=waterline_class_id,
        **{key: value for key, value in cfg.items() if key in allowed},
    )
