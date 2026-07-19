from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from shipdraft_mtl.modeling.backbones import build_backbone
from shipdraft_mtl.modeling.decoders import build_decoder
from shipdraft_mtl.modeling.encoders import build_encoder
from shipdraft_mtl.modeling.heads import build_head
from shipdraft_mtl.modeling.heads.waterline_heatmap_head import (
    build_waterline_heatmap_head,
    build_waterline_heatmap_loss,
)
from shipdraft_mtl.losses import build_loss
from shipdraft_mtl.losses.direct_depth_loss import build_direct_depth_loss
from shipdraft_mtl.modeling.heads.direct_depth_head import build_direct_depth_head
from shipdraft_mtl.losses.structure_loss import build_structure_loss
from shipdraft_mtl.postprocess import build_post_process
from shipdraft_mtl.utils import box_ops


class DraftFormerModel(nn.Module):
    def __init__(
        self,
        backbone,
        encoder,
        decoder,
        head,
        loss,
        post_process,
        pixel_mean,
        pixel_std,
        direct_depth_head=None,
        direct_depth_loss=None,
        structure_loss=None,
        waterline_heatmap_head=None,
        waterline_heatmap_loss=None,
        point_mode=False,
        auxiliary_inference=False,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.loss = loss
        self.post_process = post_process
        self.direct_depth_head = direct_depth_head
        self.direct_depth_loss = direct_depth_loss
        self.structure_loss = structure_loss
        self.waterline_heatmap_head = waterline_heatmap_head
        self.waterline_heatmap_loss = waterline_heatmap_loss
        self.point_mode = bool(point_mode)
        self.auxiliary_inference = bool(auxiliary_inference)
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_image(self, batched_inputs: List[Dict]):
        images = [sample["image"].to(self.device) for sample in batched_inputs]
        image_sizes = [(int(image.shape[-2]), int(image.shape[-1])) for image in images]
        images = [(image - self.pixel_mean) / self.pixel_std for image in images]
        return torch.stack(images, dim=0), image_sizes

    def prepare_targets(self, batched_inputs: List[Dict], image_sizes: List[Tuple[int, int]]):
        targets = []
        for sample, (height, width) in zip(batched_inputs, image_sizes):
            labels = sample.get("labels")
            points = sample.get("points")
            boxes = sample.get("boxes")
            masks = sample.get("masks")
            is_waterline = sample.get("is_waterline")

            if labels is None:
                labels = torch.zeros(0, dtype=torch.int64, device=self.device)
            else:
                labels = labels.to(self.device)

            if points is not None and len(points):
                points = points.to(self.device).float()
                points_norm = points / torch.tensor([width, height], device=self.device, dtype=points.dtype)
            else:
                points_norm = torch.zeros(0, 2, device=self.device)

            if boxes is None or (hasattr(boxes, "__len__") and len(boxes) == 0):
                if len(points_norm):
                    # default pseudo size in normalized coords
                    wh = torch.full((len(points_norm), 2), 0.05, device=self.device)
                    boxes_cxcywh = torch.cat([points_norm, wh], dim=-1)
                else:
                    boxes_cxcywh = torch.zeros(0, 4, device=self.device)
            else:
                boxes = boxes.to(self.device).float()
                boxes_normalized = boxes / torch.tensor(
                    [width, height, width, height], dtype=boxes.dtype, device=boxes.device
                )
                boxes_cxcywh = box_ops.box_xyxy_to_cxcywh(boxes_normalized)
                if self.point_mode:
                    # force centers from points when available
                    if len(points_norm) == len(boxes_cxcywh):
                        boxes_cxcywh = boxes_cxcywh.clone()
                        boxes_cxcywh[:, :2] = points_norm

            if masks is None:
                masks = torch.zeros(len(labels), height, width, device=self.device)
            else:
                masks = masks.to(self.device).float()

            if is_waterline is None:
                is_waterline = torch.zeros(len(labels), dtype=torch.bool, device=self.device)
            else:
                is_waterline = is_waterline.to(self.device).bool()

            draft_depth = sample.get("draft_depth")
            if draft_depth is None:
                draft_depth_t = torch.tensor(0.0, device=self.device)
                draft_valid = torch.tensor(False, device=self.device)
            else:
                draft_depth_t = draft_depth.to(self.device).float().reshape(())
                if "draft_depth_valid" in sample:
                    draft_valid = sample["draft_depth_valid"].to(self.device).bool().reshape(())
                else:
                    draft_valid = torch.tensor(True, device=self.device)

            targets.append(
                {
                    "boxes": boxes_cxcywh,
                    "points": points_norm,
                    "labels": labels,
                    "masks": masks,
                    "is_waterline": is_waterline,
                    "draft_depth": draft_depth_t,
                    "draft_depth_valid": draft_valid,
                }
            )
        return targets

    def forward(self, batched_inputs: List[Dict]):
        images, image_sizes = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        encoded = self.encoder(features)
        if self.direct_depth_head is not None:
            outputs = self.direct_depth_head(encoded["mask_features"])
        else:
            outputs = {}
        if self.training or self.auxiliary_inference:
            decoded = self.decoder(
                encoded["multi_scale_features"],
                proposal_class_head=self.head.get_proposal_class_head(),
                proposal_box_head=self.head.get_proposal_box_head(),
                bbox_refine_heads=self.head.get_bbox_refine_heads(),
            )
            outputs.update(self.head(decoded, encoded["mask_features"], self.decoder))
            outputs["pred_points"] = outputs["pred_boxes"][..., :2]

        if self.training and self.waterline_heatmap_head is not None:
            outputs["pred_waterline_heatmap_logits"] = self.waterline_heatmap_head(encoded["mask_features"])

        if not self.training:
            return self.post_process(outputs, image_sizes, batched_inputs)

        targets = self.prepare_targets(batched_inputs, image_sizes)
        losses = self.loss(outputs, targets)
        if self.direct_depth_loss is not None and self.direct_depth_head is not None:
            losses.update(self.direct_depth_loss(outputs, targets))
        if self.structure_loss is not None:
            losses.update(self.structure_loss(outputs, targets))
        if self.waterline_heatmap_head is not None and self.waterline_heatmap_loss is not None:
            heatmap_logits = outputs.get("pred_waterline_heatmap_logits")
            if heatmap_logits is not None:
                heatmap_target = self.waterline_heatmap_head.build_targets(targets, heatmap_logits.shape[-2:])
                losses.update(self.waterline_heatmap_loss(heatmap_logits, heatmap_target))
        return self._format_losses(losses)

    def _format_losses(self, losses: Dict[str, torch.Tensor]):
        weighted_losses = {}
        task_losses = {}
        total_loss = None
        weight_dict = dict(getattr(self.loss, "weight_dict", {}))
        if self.direct_depth_loss is not None:
            weight_dict.update(getattr(self.direct_depth_loss, "weight_dict", {}))
        if self.structure_loss is not None:
            weight_dict.update(getattr(self.structure_loss, "weight_dict", {}))
        if self.waterline_heatmap_loss is not None:
            weight_dict.update(getattr(self.waterline_heatmap_loss, "weight_dict", {}))

        for key, value in losses.items():
            if key in weight_dict:
                weighted = value * weight_dict[key]
            else:
                base_key = key.split("_aux")[0]
                if base_key not in weight_dict:
                    # keep unweighted draft keys if any missed
                    if key.startswith("loss_"):
                        weighted = value
                    else:
                        continue
                else:
                    weighted = value * weight_dict[base_key]
            weighted_losses[key] = weighted
            total_loss = weighted if total_loss is None else total_loss + weighted
            weighted_losses[f"stat_{key}"] = value.detach()
            if key.endswith("_det") or "_det_" in key or "point" in key:
                task_losses["det"] = task_losses.get("det", 0.0) + weighted
            elif key.endswith("_seg") or "_seg_" in key:
                task_losses["seg"] = task_losses.get("seg", 0.0) + weighted
            elif "draft" in key or "depth" in key:
                task_losses["draft"] = task_losses.get("draft", 0.0) + weighted
            elif "struct" in key or "heatmap" in key:
                task_losses["det"] = task_losses.get("det", 0.0) + weighted

        if total_loss is None:
            raise ValueError("DraftFormer produced no valid weighted losses")

        if getattr(self.loss, "task_balancer", None) is not None and task_losses:
            balanced_losses = self.loss.task_balancer(task_losses)
            formatted = {"loss": balanced_losses["total_loss"]}
            for key, value in balanced_losses.items():
                if key == "total_loss":
                    continue
                formatted[key] = value
            formatted.update(weighted_losses)
            return formatted

        weighted_losses["loss"] = total_loss
        return weighted_losses


def build_model(cfg):
    arch_cfg = cfg["Architecture"]
    point_mode = bool(arch_cfg.get("point_mode", arch_cfg.get("model_type") in {"point_seg_e2e", "point_seg", "e2e_point"}))
    backbone = build_backbone(arch_cfg["Backbone"], in_channels=arch_cfg.get("in_channels", 3))
    encoder = build_encoder(arch_cfg["Encoder"], input_shape=backbone.output_shape())
    decoder = build_decoder(
        arch_cfg["Decoder"],
        in_channels=encoder.hidden_dim,
        num_feature_levels=encoder.num_feature_levels,
    )
    head = build_head(arch_cfg["Head"], num_decoder_layers=decoder.num_layers)
    loss = build_loss(cfg["Loss"], arch_cfg)
    post_process = build_post_process(cfg["PostProcess"], arch_cfg)

    direct_depth_head = None
    direct_depth_loss = None
    depth_cfg = arch_cfg.get("DirectDepth") or cfg.get("DirectDepth")
    if not depth_cfg or not depth_cfg.get("enabled", True):
        raise ValueError("DirectDepth must be enabled for the ROI-to-draft model")
    direct_depth_head = build_direct_depth_head(depth_cfg, in_channels=encoder.hidden_dim)
    direct_depth_loss = build_direct_depth_loss(
        cfg.get("DirectDepthLoss") or depth_cfg.get("loss") or {}
    )

    structure_loss = None
    struct_cfg = cfg.get("StructureLoss") or arch_cfg.get("StructureLoss")
    if struct_cfg and struct_cfg.get("enabled", True):
        class_names = None
        if cfg.get("Data", {}).get("detection_classes"):
            class_names = list(cfg["Data"]["detection_classes"])
        elif cfg.get("Metric", {}).get("det_class_names"):
            class_names = list(cfg["Metric"]["det_class_names"])
        structure_loss = build_structure_loss(struct_cfg, class_names=class_names)

    waterline_heatmap_head = None
    waterline_heatmap_loss = None
    heat_cfg = arch_cfg.get("WaterlineHeatmap") or cfg.get("WaterlineHeatmap")
    if heat_cfg and heat_cfg.get("enabled", True):
        waterline_heatmap_head = build_waterline_heatmap_head(heat_cfg, in_channels=encoder.hidden_dim)
        waterline_heatmap_loss = build_waterline_heatmap_loss(heat_cfg)

    return DraftFormerModel(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        head=head,
        loss=loss,
        post_process=post_process,
        pixel_mean=tuple(arch_cfg.get("pixel_mean", [103.530, 116.280, 123.675])),
        pixel_std=tuple(arch_cfg.get("pixel_std", [1.0, 1.0, 1.0])),
        direct_depth_head=direct_depth_head,
        direct_depth_loss=direct_depth_loss,
        structure_loss=structure_loss,
        waterline_heatmap_head=waterline_heatmap_head,
        waterline_heatmap_loss=waterline_heatmap_loss,
        point_mode=point_mode,
        auxiliary_inference=arch_cfg.get("auxiliary_inference", False),
    )
