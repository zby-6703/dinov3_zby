from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from shipdraft_mtl.modeling.backbones import build_backbone
from shipdraft_mtl.modeling.decoders import build_decoder
from shipdraft_mtl.modeling.encoders import build_encoder
from shipdraft_mtl.modeling.heads import build_head
from shipdraft_mtl.losses import build_loss
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
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.head = head
        self.loss = loss
        self.post_process = post_process
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
            boxes = sample.get("boxes")
            labels = sample.get("labels")
            masks = sample.get("masks")
            if boxes is None or labels is None or masks is None:
                targets.append(
                    {
                        "boxes": torch.zeros(0, 4, device=self.device),
                        "labels": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "masks": torch.zeros(0, height, width, device=self.device),
                    }
                )
                continue

            boxes = boxes.to(self.device)
            labels = labels.to(self.device)
            masks = masks.to(self.device).float()
            boxes_normalized = boxes / torch.tensor(
                [width, height, width, height],
                dtype=boxes.dtype,
                device=boxes.device,
            )
            targets.append(
                {
                    "boxes": box_ops.box_xyxy_to_cxcywh(boxes_normalized),
                    "labels": labels,
                    "masks": masks,
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
        outputs = self.head(decoded, encoded["mask_features"], self.decoder)

        if not self.training:
            return self.post_process(outputs, image_sizes, batched_inputs)

        targets = self.prepare_targets(batched_inputs, image_sizes)
        losses = self.loss(outputs, targets)
        return self._format_losses(losses)

    def _format_losses(self, losses: Dict[str, torch.Tensor]):
        weighted_losses = {}
        task_losses = {}
        total_loss = None
        for key, value in losses.items():
            if key in self.loss.weight_dict:
                weighted = value * self.loss.weight_dict[key]
            else:
                base_key = key.split("_aux")[0]
                if base_key not in self.loss.weight_dict:
                    continue
                weighted = value * self.loss.weight_dict[base_key]
            weighted_losses[key] = weighted
            total_loss = weighted if total_loss is None else total_loss + weighted
            weighted_losses[f"stat_{key}"] = value.detach()
            if key.endswith("_det") or "_det_" in key:
                task_losses["det"] = task_losses.get("det", 0.0) + weighted
            elif key.endswith("_seg") or "_seg_" in key:
                task_losses["seg"] = task_losses.get("seg", 0.0) + weighted

        if total_loss is None:
            raise ValueError("DraftFormer produced no valid weighted losses")

        if self.loss.task_balancer is not None and task_losses:
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
    return DraftFormerModel(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        head=head,
        loss=loss,
        post_process=post_process,
        pixel_mean=tuple(arch_cfg.get("pixel_mean", [103.530, 116.280, 123.675])),
        pixel_std=tuple(arch_cfg.get("pixel_std", [1.0, 1.0, 1.0])),
    )
