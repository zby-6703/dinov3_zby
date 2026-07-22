from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from shipdraft_mtl.modeling.backbones import build_backbone
from shipdraft_mtl.modeling.decoders import build_decoder
from shipdraft_mtl.modeling.encoders import build_encoder
from shipdraft_mtl.modeling.heads import build_head
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
        point_mode=False,
        keypoint_mode=False,
        return_auxiliary_outputs=False,
        train_character=True,
        train_waterline=True,
        train_depth=True,
        train_structure=True,
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
        self.point_mode = bool(point_mode)
        self.keypoint_mode = bool(keypoint_mode)
        self.return_auxiliary_outputs = bool(return_auxiliary_outputs)
        self.train_character = bool(train_character)
        self.train_waterline = bool(train_waterline)
        self.train_depth = bool(train_depth)
        self.train_structure = bool(train_structure)
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self):
        return self.pixel_mean.device

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for name in getattr(self, "_task_frozen_modules", ()):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
        return self

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
            waterline_mask = sample.get("waterline_mask")

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
            if waterline_mask is None:
                waterline_mask = torch.zeros(height, width, device=self.device)
            else:
                waterline_mask = waterline_mask.to(self.device).float()

            # Ordered waterline curve points (pixel -> normalized), dynamic length via validity.
            waterline_curve = sample.get("waterline_curve_points")
            waterline_valid = sample.get("waterline_curve_valid")
            has_waterline = sample.get("has_waterline")
            if waterline_curve is not None and len(waterline_curve):
                waterline_curve = waterline_curve.to(self.device).float()
                waterline_curve_norm = waterline_curve / torch.tensor(
                    [width, height], device=self.device, dtype=waterline_curve.dtype
                )
            else:
                waterline_curve_norm = torch.zeros(0, 2, device=self.device)
            if waterline_valid is not None:
                waterline_valid_t = waterline_valid.to(self.device).bool()
            elif len(waterline_curve_norm):
                waterline_valid_t = torch.ones(len(waterline_curve_norm), dtype=torch.bool, device=self.device)
            else:
                waterline_valid_t = torch.zeros(0, dtype=torch.bool, device=self.device)
            if has_waterline is None:
                has_waterline_t = torch.tensor(
                    bool(waterline_valid_t.any().item()) if len(waterline_valid_t) else False,
                    device=self.device,
                )
            else:
                has_waterline_t = has_waterline.to(self.device).bool().reshape(())

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
                    "waterline_mask": waterline_mask,
                    "waterline_curve_points": waterline_curve_norm,
                    "waterline_curve_valid": waterline_valid_t,
                    "has_waterline": has_waterline_t,
                    "draft_depth": draft_depth_t,
                    "draft_depth_valid": draft_valid,
                }
            )
        return targets

    def forward(self, batched_inputs: List[Dict]):
        images, image_sizes = self.preprocess_image(batched_inputs)
        features = self.backbone(images)
        encoded = self.encoder(features)
        decoded = self.decoder(
            encoded["multi_scale_features"],
            proposal_class_head=self.head.get_proposal_class_head(),
            proposal_box_head=self.head.get_proposal_box_head(),
            bbox_refine_heads=self.head.get_bbox_refine_heads(),
        )
        query_features = self.decoder.normalize_queries(decoded["decoder_states"][-1]).transpose(0, 1)
        reference_points = decoded["references"][-1][..., :2]
        if self.training or self.return_auxiliary_outputs:
            outputs = self.head(decoded, encoded["mask_features"], self.decoder)
            if "pred_points" not in outputs or outputs["pred_points"] is None:
                outputs["pred_points"] = outputs["pred_boxes"][..., :2]
            depth_points = outputs["pred_points"]
        else:
            outputs = {}
            depth_points = reference_points

        run_depth = self.direct_depth_head is not None and (
            self.train_depth or (not self.training and self.return_auxiliary_outputs) or not self.training
        )
        # Always run depth head at inference if it exists; training respects TaskTrain.
        if self.training:
            run_depth = self.direct_depth_head is not None and self.train_depth
        elif self.direct_depth_head is not None:
            run_depth = True

        if run_depth:
            outputs.update(
                self.direct_depth_head(
                    query_features,
                    depth_points,
                )
            )

        if not self.training:
            if not self.return_auxiliary_outputs:
                if "pred_depth" in outputs:
                    outputs = {
                        "pred_depth": outputs["pred_depth"],
                        "pred_depth_valid_logits": outputs["pred_depth_valid_logits"],
                    }
                # else keep character/waterline outputs only
            return self.post_process(outputs, image_sizes, batched_inputs)

        targets = self.prepare_targets(batched_inputs, image_sizes)
        losses = {}
        if self.train_character or self.train_waterline:
            losses.update(self.loss(outputs, targets))
        if (
            self.train_depth
            and self.direct_depth_loss is not None
            and self.direct_depth_head is not None
            and "pred_depth" in outputs
        ):
            losses.update(self.direct_depth_loss(outputs, targets))
        if self.train_structure and self.train_character and self.structure_loss is not None:
            losses.update(self.structure_loss(outputs, targets))
        if not losses:
            raise ValueError(
                "No active training losses. Enable at least one of "
                "TaskTrain.train_character / train_waterline / train_depth."
            )
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
            if key.endswith("_det") or "_det_" in key or ("point" in key and "curve" not in key):
                task_losses["det"] = task_losses.get("det", 0.0) + weighted
            elif key.endswith("_seg") or "_seg_" in key or "curve" in key:
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
    keypoint_mode = bool(
        arch_cfg.get(
            "keypoint_mode",
            arch_cfg.get("model_type") in {"direct_depth_mtl", "keypoint_e2e", "e2e_keypoint"}
            or point_mode,
        )
    )
    # Propagate keypoint mode into head so pure 2D point heads are built.
    arch_cfg.setdefault("Head", {})
    arch_cfg["Head"].setdefault("keypoint_mode", keypoint_mode)
    if keypoint_mode:
        arch_cfg["Decoder"].setdefault("query_dim", 2)
        arch_cfg["Head"].setdefault("predict_masks", False)
    backbone = build_backbone(arch_cfg["Backbone"], in_channels=arch_cfg.get("in_channels", 3))
    encoder = build_encoder(arch_cfg["Encoder"], input_shape=backbone.output_shape())
    decoder = build_decoder(
        arch_cfg["Decoder"],
        in_channels=encoder.hidden_dim,
        num_feature_levels=encoder.num_feature_levels,
    )
    # Propagate multi-stage task switches into the loss builder.
    task_train = dict(cfg.get("TaskTrain") or {})
    arch_cfg["TaskTrain"] = task_train

    head = build_head(arch_cfg["Head"], num_decoder_layers=decoder.num_layers)
    loss = build_loss(cfg["Loss"], arch_cfg)
    post_process = build_post_process(cfg["PostProcess"], arch_cfg)

    direct_depth_head = None
    direct_depth_loss = None
    depth_cfg = arch_cfg.get("DirectDepth") or cfg.get("DirectDepth") or {}
    depth_enabled = bool(depth_cfg.get("enabled", True))
    # Stage-1 can disable depth entirely; stage-2/3 re-enable it.
    train_depth_flag = bool(task_train.get("train_depth", depth_enabled))
    if depth_enabled:
        direct_depth_head = build_direct_depth_head(
            depth_cfg,
            query_dim=encoder.hidden_dim,
            num_character_queries=depth_cfg.get(
                "num_character_queries",
                arch_cfg["Decoder"].get("num_detection_queries", 100),
            ),
        )
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
        point_mode=point_mode,
        keypoint_mode=keypoint_mode,
        return_auxiliary_outputs=arch_cfg.get("return_auxiliary_outputs", False),
        train_character=bool(task_train.get("train_character", True)),
        train_waterline=bool(task_train.get("train_waterline", True)),
        train_depth=bool(train_depth_flag and depth_enabled),
        train_structure=bool(task_train.get("train_structure", True)),
    )
