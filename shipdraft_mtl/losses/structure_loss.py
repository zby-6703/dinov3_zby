from __future__ import annotations

"""Geometry-only auxiliary constraints for detected scale points."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn


class StructureConstraintLoss(nn.Module):
    """Improve point localization without participating in depth regression.

    Matching uses the target class and nearest predicted point. The constraints
    then preserve the target vertical order and local spacing. No hard-coded
    digit semantics or meter interval is used here; those belong to neither
    the detection objective nor the direct depth head.
    """

    def __init__(
        self,
        order_weight: float = 0.5,
        space_weight: float = 0.5,
        margin: float = 0.005,
        match_score_threshold: float = 0.05,
        num_character_queries: int = 100,
        num_character_classes: int = 9,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.order_weight = float(order_weight)
        self.space_weight = float(space_weight)
        self.margin = float(margin)
        self.match_score_threshold = float(match_score_threshold)
        self.num_character_queries = int(num_character_queries)
        self.num_character_classes = int(num_character_classes)
        self.enabled = bool(enabled)
        self.weight_dict = {
            "loss_struct_order": self.order_weight,
            "loss_struct_space": self.space_weight,
        }

    def _match_points(self, pred_points, pred_logits, gt_points, gt_labels):
        probabilities = pred_logits.sigmoid()
        predicted_labels = probabilities.argmax(dim=-1)
        scores = probabilities.amax(dim=-1)
        used = torch.zeros(pred_points.shape[0], dtype=torch.bool, device=pred_points.device)
        matches = []
        # Match in target order so repeated glyphs are assigned consistently.
        for gt_index, (gt_point, gt_label) in enumerate(zip(gt_points, gt_labels)):
            candidates = (predicted_labels == gt_label) & (~used) & (scores >= self.match_score_threshold)
            if not candidates.any():
                candidates = (~used) & (scores >= self.match_score_threshold)
            if not candidates.any():
                continue
            candidate_ids = candidates.nonzero(as_tuple=False).flatten()
            distances = torch.norm(pred_points[candidate_ids] - gt_point.unsqueeze(0), dim=-1)
            selected = candidate_ids[distances.argmin()]
            used[selected] = True
            matches.append((gt_index, selected))
        if not matches:
            return torch.zeros(0, 2, dtype=torch.long, device=pred_points.device)
        return torch.tensor(
            [[gt_index, int(pred_index)] for gt_index, pred_index in matches],
            dtype=torch.long,
            device=pred_points.device,
        )

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict]) -> Dict[str, torch.Tensor]:
        zero = outputs["pred_logits"].new_zeros(())
        if not self.enabled or "pred_points" not in outputs:
            return {"loss_struct_order": zero, "loss_struct_space": zero}

        order_losses = []
        spacing_losses = []
        pred_points = outputs["pred_points"]
        pred_logits = outputs["pred_logits"]
        for batch_index, target in enumerate(targets):
            gt_points = target.get("points")
            gt_labels = target.get("labels")
            if gt_points is None or gt_labels is None or len(gt_points) < 2:
                continue
            character_mask = gt_labels < self.num_character_classes
            gt_points = gt_points[character_mask]
            gt_labels = gt_labels[character_mask]
            if len(gt_points) < 2:
                continue
            order = torch.argsort(gt_points[:, 1])
            gt_points = gt_points[order]
            gt_labels = gt_labels[order]
            matched_pairs = self._match_points(
                pred_points[batch_index, : self.num_character_queries],
                pred_logits[batch_index, : self.num_character_queries],
                gt_points,
                gt_labels,
            )
            if len(matched_pairs) < 2:
                continue
            matched_gt = gt_points[matched_pairs[:, 0]]
            matched_pred = pred_points[batch_index][matched_pairs[:, 1]]
            gt_dy = (matched_gt[1:, 1] - matched_gt[:-1, 1]).clamp_min(0.0)
            pred_dy = matched_pred[1:, 1] - matched_pred[:-1, 1]
            order_losses.append(F.relu(self.margin - pred_dy).mean())
            spacing_losses.append(F.smooth_l1_loss(pred_dy, gt_dy, beta=0.01))

        return {
            "loss_struct_order": torch.stack(order_losses).mean() if order_losses else zero,
            "loss_struct_space": torch.stack(spacing_losses).mean() if spacing_losses else zero,
        }


def build_structure_loss(config=None, class_names=None):
    cfg = dict(config or {})
    cfg.pop("name", None)
    allowed = {
        "order_weight", "space_weight", "margin", "match_score_threshold",
        "num_character_queries", "num_character_classes", "enabled",
    }
    return StructureConstraintLoss(**{key: value for key, value in cfg.items() if key in allowed})
