from __future__ import annotations

import torch
import torch.nn.functional as F

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
        point_mode: bool = False,
        point_nms_dist: float = 0.03,
    ):
        self.score_thresh = score_thresh
        self.topk = topk
        self.num_detection_queries = num_detection_queries
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes
        self.nms_thresh = nms_thresh
        self.nms_type = nms_type.lower()
        self.point_mode = bool(point_mode)
        self.point_nms_dist = float(point_nms_dist)

    def __call__(self, outputs, image_sizes, batched_inputs):
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        pred_masks = outputs.get("pred_masks")
        results = []
        for batch_idx, ((height, width), sample) in enumerate(zip(image_sizes, batched_inputs)):
            if pred_logits is not None and pred_boxes is not None:
                det_logits = pred_logits[batch_idx, : self.num_detection_queries, : self.num_det_classes]
                det_boxes = pred_boxes[batch_idx, : self.num_detection_queries]
                points, scores, labels = self._decode_points(det_logits, det_boxes, width, height, sample)
                boxes = self._points_to_tiny_boxes(points) if self.point_mode else self._decode_boxes_fallback(det_logits, det_boxes, width, height, sample)[0]
            else:
                device = outputs["pred_depth"].device
                det_logits = None
                points = torch.zeros(0, 2, device=device)
                scores = torch.zeros(0, device=device)
                labels = torch.zeros(0, dtype=torch.int64, device=device)
                boxes = torch.zeros(0, 4, device=device)
            result = {
                "points": points,
                "scores": scores,
                "labels": labels,
                "boxes": boxes,
                "image_size": (sample.get("orig_height", height), sample.get("orig_width", width)),
            }
            if pred_masks is not None and self.num_seg_classes > 0 and pred_logits is not None and pred_logits.shape[1] > self.num_detection_queries:
                seg_logits = pred_logits[batch_idx, self.num_detection_queries :, : max(self.num_seg_classes, 1)]
                seg_masks = pred_masks[batch_idx, self.num_detection_queries :]
                result["sem_seg"] = self._decode_segmentation(seg_logits, seg_masks, width, height, sample)
            else:
                result["sem_seg"] = torch.zeros(1, sample.get("orig_height", height), sample.get("orig_width", width), device=points.device)
            if "pred_depth" in outputs:
                result["draft_depth"] = outputs["pred_depth"][batch_idx].detach()
                result["draft_valid"] = torch.sigmoid(outputs["pred_depth_valid_logits"][batch_idx]).detach()
            results.append(result)
        return results

    def _decode_points(self, logits, boxes, width, height, sample):
        prob = logits.sigmoid()
        scores, labels = prob.max(dim=-1)
        keep = scores > self.score_thresh
        scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
        if scores.numel() == 0:
            dev = logits.device
            return torch.zeros(0, 2, device=dev), torch.zeros(0, device=dev), torch.zeros(0, dtype=torch.int64, device=dev)
        # topk
        if scores.numel() > self.topk:
            scores, idx = scores.topk(self.topk)
            labels = labels[idx]
            boxes = boxes[idx]
        points = boxes[:, :2] * torch.tensor([width, height], device=boxes.device)
        points = self._restore_points_from_padding(points, sample)
        # simple distance nms in original coords
        if self.point_mode and len(scores) > 1:
            order = scores.argsort(descending=True)
            keep_idx = []
            suppressed = torch.zeros(len(order), dtype=torch.bool, device=scores.device)
            pts = points[order]
            oh = float(sample.get("orig_height", height))
            ow = float(sample.get("orig_width", width))
            thr = self.point_nms_dist * max(oh, ow)
            for i in range(len(order)):
                if suppressed[i]:
                    continue
                keep_idx.append(order[i].item())
                if i + 1 < len(order):
                    d = torch.norm(pts[i + 1 :] - pts[i], dim=-1)
                    suppressed[i + 1 :] |= d < thr
            keep_idx = torch.tensor(keep_idx, device=scores.device, dtype=torch.long)
            points, scores, labels = points[keep_idx], scores[keep_idx], labels[keep_idx]
        return points, scores, labels

    def _points_to_tiny_boxes(self, points):
        if len(points) == 0:
            return torch.zeros(0, 4, device=points.device)
        r = 3.0
        return torch.stack([points[:, 0] - r, points[:, 1] - r, points[:, 0] + r, points[:, 1] + r], dim=-1)

    def _decode_boxes_fallback(self, logits, boxes, width, height, sample):
        # legacy path for non-point mode
        prob = logits.sigmoid()
        flat = prob.flatten()
        topk = min(self.topk, flat.numel())
        scores, topk_indices = flat.topk(topk)
        labels = topk_indices % self.num_det_classes
        query_indices = topk_indices // self.num_det_classes
        keep = scores > self.score_thresh
        scores, labels, query_indices = scores[keep], labels[keep], query_indices[keep]
        boxes = boxes[query_indices]
        boxes_xyxy = box_cxcywh_to_xyxy(boxes) * torch.tensor([width, height, width, height], device=boxes.device)
        boxes_xyxy = self._restore_boxes_from_padding(boxes_xyxy, sample)
        return boxes_xyxy, scores, labels

    def _decode_segmentation(self, logits, masks, width, height, sample):
        orig_h = sample.get("orig_height", height)
        orig_w = sample.get("orig_width", width)
        if masks.numel() == 0:
            return torch.zeros(1, orig_h, orig_w, device=logits.device)
        prob = logits.sigmoid().reshape(-1, 1, 1) * masks.sigmoid()
        sem = prob.max(dim=0).values
        if sem.ndim == 2:
            sem = sem.unsqueeze(0)
        sem = F.interpolate(sem.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False)[0]
        pad_left = int(sample.get("pad_left", 0))
        pad_top = int(sample.get("pad_top", 0))
        pad_right = int(sample.get("pad_right", 0))
        pad_bottom = int(sample.get("pad_bottom", 0))
        y_end = height - pad_bottom if pad_bottom > 0 else height
        x_end = width - pad_right if pad_right > 0 else width
        sem = sem[:, pad_top:y_end, pad_left:x_end]
        sem = F.interpolate(sem.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False)[0]
        return sem

    def _restore_points_from_padding(self, points, sample):
        scale = float(sample.get("resize_scale", 1.0)) or 1.0
        pad_left = float(sample.get("pad_left", 0))
        pad_top = float(sample.get("pad_top", 0))
        pts = points.clone()
        pts[:, 0] = (pts[:, 0] - pad_left) / scale
        pts[:, 1] = (pts[:, 1] - pad_top) / scale
        return pts

    def _restore_boxes_from_padding(self, boxes, sample):
        scale = float(sample.get("resize_scale", 1.0)) or 1.0
        pad_left = float(sample.get("pad_left", 0))
        pad_top = float(sample.get("pad_top", 0))
        out = boxes.clone()
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_left) / scale
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_top) / scale
        return out
