from __future__ import annotations

"""Losses for the direct image-to-draft regression task."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class DirectDepthLoss(nn.Module):
    def __init__(
        self,
        depth_weight: float = 5.0,
        valid_weight: float = 0.25,
        beta: float = 0.05,
    ) -> None:
        super().__init__()
        self.depth_weight = float(depth_weight)
        self.valid_weight = float(valid_weight)
        self.beta = float(beta)
        self.weight_dict = {
            "loss_depth_direct": self.depth_weight,
            "loss_depth_valid_direct": self.valid_weight,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        prediction = outputs["pred_depth"]
        valid_logits = outputs["pred_depth_valid_logits"]
        depth_values = []
        valid_values = []
        for target in targets:
            raw_depth = target.get("draft_depth")
            if torch.is_tensor(raw_depth):
                depth_values.append(raw_depth.to(device=prediction.device, dtype=prediction.dtype).reshape(()))
            elif raw_depth is None:
                depth_values.append(prediction.new_zeros(()))
            else:
                depth_values.append(prediction.new_tensor(float(raw_depth)))
            raw_valid = target.get("draft_depth_valid", raw_depth is not None)
            if torch.is_tensor(raw_valid):
                valid_values.append(raw_valid.to(device=prediction.device).bool().reshape(()))
            else:
                valid_values.append(torch.tensor(bool(raw_valid), device=prediction.device))
        target_depth = torch.stack(depth_values).reshape(-1)
        target_valid = torch.stack(valid_values).reshape(-1).bool()

        validity = F.binary_cross_entropy_with_logits(valid_logits, target_valid.float())
        if target_valid.any():
            depth = F.smooth_l1_loss(
                prediction[target_valid],
                target_depth[target_valid],
                beta=self.beta,
                reduction="mean",
            )
        else:
            depth = prediction.new_zeros(())
        return {
            "loss_depth_direct": depth,
            "loss_depth_valid_direct": validity,
        }


def build_direct_depth_loss(config=None) -> DirectDepthLoss:
    cfg = dict(config or {})
    cfg.pop("name", None)
    allowed = {"depth_weight", "valid_weight", "beta"}
    return DirectDepthLoss(**{key: value for key, value in cfg.items() if key in allowed})
