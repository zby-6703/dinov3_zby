from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms, nms

from shipdraft_mtl.utils.box_ops import box_cxcywh_to_xyxy


class DraftFormerPostProcess:
    def __init__(
        self,
        score_thresh: float = 0.1,
        topk: int = 100,
        num_detection_queries: int = 100,
        num_det_classes: int = 9,
        num_seg_classes: int = 1,
        nms_thresh: float = 0.5,
        nms_type: str = "class_aware",
    ):
        self.score_thresh = score_thresh
        self.topk = topk
        self.num_detection_queries = num_detection_queries
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes
        self.nms_thresh = nms_thresh
        self.nms_type = nms_type.lower()

    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        batched_inputs: List[Dict],
    ) -> List[Dict[str, torch.Tensor]]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_masks = outputs["pred_masks"]
        results = []

        for batch_idx, ((height, width), sample) in enumerate(zip(image_sizes, batched_inputs)):
            det_logits = pred_logits[batch_idx, : self.num_detection_queries, : self.num_det_classes]
            det_boxes = pred_boxes[batch_idx, : self.num_detection_queries]
            seg_logits = pred_logits[batch_idx, self.num_detection_queries :, : self.num_seg_classes]
            seg_masks = pred_masks[batch_idx, self.num_detection_queries :]

            boxes, scores, labels = self._decode_detections(det_logits, det_boxes, width, height, sample)
            sem_seg = self._decode_segmentation(seg_logits, seg_masks, width, height, sample)
            results.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "sem_seg": sem_seg,
                    "image_size": (sample.get("orig_height", height), sample.get("orig_width", width)),
                }
            )

        return results

    def _decode_detections(self, logits, boxes, width, height, sample):
        prob = logits.sigmoid()
        flat_scores = prob.flatten()
        numel = flat_scores.numel()
        if numel == 0:
            return (
                torch.zeros((0, 4), device=boxes.device),
                torch.zeros((0,), device=boxes.device),
                torch.zeros((0,), dtype=torch.int64, device=boxes.device),
            )

        topk = min(self.topk, numel)
        scores, topk_indices = flat_scores.topk(topk, sorted=True)
        labels = topk_indices % self.num_det_classes
        query_indices = topk_indices // self.num_det_classes
        keep = scores > self.score_thresh
        scores = scores[keep]
        labels = labels[keep]
        query_indices = query_indices[keep]
        boxes = boxes[query_indices]

        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        boxes_xyxy = boxes_xyxy * torch.tensor([width, height, width, height], device=boxes.device)
        boxes_xyxy = self._restore_boxes_from_padding(boxes_xyxy, sample)

        if len(scores) > 0 and self.nms_type != "none" and self.nms_thresh is not None and self.nms_thresh > 0:
            if self.nms_type == "class_aware":
                keep = batched_nms(boxes_xyxy, scores, labels, self.nms_thresh)
            elif self.nms_type == "class_agnostic":
                keep = nms(boxes_xyxy, scores, self.nms_thresh)
            else:
                raise ValueError(f"Unsupported nms_type: {self.nms_type}")
            boxes_xyxy = boxes_xyxy[keep]
            scores = scores[keep]
            labels = labels[keep]
        return boxes_xyxy, scores, labels

    def _decode_segmentation(self, logits, masks, width, height, sample):
        orig_h = sample.get("orig_height", height)
        orig_w = sample.get("orig_width", width)
        if masks.numel() == 0:
            return torch.zeros((self.num_seg_classes, orig_h, orig_w), device=logits.device)

        upsampled_masks = F.interpolate(
            masks.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        sem_seg = torch.einsum("qc,qhw->chw", logits.sigmoid(), upsampled_masks.sigmoid())
        return self._restore_segmentation_from_padding(sem_seg, sample)

    def _restore_boxes_from_padding(self, boxes_xyxy, sample):
        orig_h = sample.get("orig_height", sample.get("height"))
        orig_w = sample.get("orig_width", sample.get("width"))
        scale = sample.get("resize_scale", 1.0)
        pad_left = sample.get("pad_left", 0)
        pad_top = sample.get("pad_top", 0)
        boxes_xyxy[:, 0::2] = (boxes_xyxy[:, 0::2] - pad_left) / scale
        boxes_xyxy[:, 1::2] = (boxes_xyxy[:, 1::2] - pad_top) / scale
        boxes_xyxy[:, 0::2] = boxes_xyxy[:, 0::2].clamp(min=0, max=orig_w)
        boxes_xyxy[:, 1::2] = boxes_xyxy[:, 1::2].clamp(min=0, max=orig_h)
        return boxes_xyxy

    def _restore_segmentation_from_padding(self, sem_seg, sample):
        orig_h = sample.get("orig_height", sample.get("height"))
        orig_w = sample.get("orig_width", sample.get("width"))
        target_h = sample.get("height")
        target_w = sample.get("width")
        pad_left = sample.get("pad_left", 0)
        pad_top = sample.get("pad_top", 0)
        pad_right = sample.get("pad_right", 0)
        pad_bottom = sample.get("pad_bottom", 0)

        h_start = pad_top
        h_end = target_h - pad_bottom
        w_start = pad_left
        w_end = target_w - pad_right
        sem_seg = sem_seg[:, h_start:h_end, w_start:w_end]
        sem_seg = F.interpolate(
            sem_seg.unsqueeze(0),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return sem_seg
