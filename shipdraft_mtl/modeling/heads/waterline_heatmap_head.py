from __future__ import annotations

"""Training-only waterline heatmap auxiliary task."""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def render_gaussian_heatmap(points_xy_norm, height: int, width: int, sigma: float) -> torch.Tensor:
    device = points_xy_norm.device
    heatmap = torch.zeros(height, width, device=device)
    if points_xy_norm.numel() == 0:
        return heatmap
    ys = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
    denominator = 2.0 * sigma * sigma
    for point in points_xy_norm:
        x = point[0].clamp(0, 1) * (width - 1)
        y = point[1].clamp(0, 1) * (height - 1)
        gaussian = torch.exp(-((xs - x).square() + (ys - y).square()) / denominator)
        heatmap = torch.maximum(heatmap, gaussian)
    return heatmap


class WaterlineHeatmapHead(nn.Module):
    def __init__(self, in_channels: int = 256, sigma: float = 2.0, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.sigma = float(sigma)
        hidden = max(in_channels // 2, 32)
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)

    def build_targets(self, targets: List[Dict], feature_size: Tuple[int, int]) -> torch.Tensor:
        height, width = feature_size
        heatmaps = []
        for target in targets:
            points = target.get("points")
            waterline = target.get("is_waterline")
            if points is None or waterline is None or points.numel() == 0:
                selected = torch.zeros(0, 2, device=next(self.parameters()).device)
            else:
                selected = points[waterline.bool()]
            heatmaps.append(render_gaussian_heatmap(selected, height, width, self.sigma))
        return torch.stack(heatmaps).unsqueeze(1)


class WaterlineHeatmapLoss(nn.Module):
    def __init__(self, weight: float = 1.5, enabled: bool = True) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.weight_dict = {"loss_waterline_heatmap": float(weight)}

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        if not self.enabled:
            return {"loss_waterline_heatmap": logits.new_zeros(())}
        if logits.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(target, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        prediction = logits.sigmoid()
        positive = target.gt(0.1).float()
        negative = 1.0 - positive
        positive_loss = (prediction - target).square() * positive
        negative_loss = prediction.square() * negative
        normalizer = positive.sum().clamp_min(1.0)
        loss = (positive_loss.sum() + 0.1 * negative_loss.sum()) / normalizer
        return {"loss_waterline_heatmap": loss}


def build_waterline_heatmap_head(config=None, in_channels: int = 256):
    cfg = dict(config or {})
    return WaterlineHeatmapHead(
        in_channels=in_channels,
        sigma=float(cfg.get("sigma", 2.0)),
        enabled=bool(cfg.get("enabled", True)),
    )


def build_waterline_heatmap_loss(config=None):
    cfg = dict(config or {})
    return WaterlineHeatmapLoss(
        weight=float(cfg.get("weight", 1.5)),
        enabled=bool(cfg.get("enabled", True)),
    )
