from __future__ import annotations

import cv2
import numpy as np
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
        keypoint_mode: bool = False,
        point_nms_dist: float = 0.03,
        waterline_score_thresh: float = 0.3,
        waterline_keep_all: bool = True,
    ):
        self.score_thresh = score_thresh
        self.topk = topk
        self.num_detection_queries = num_detection_queries
        self.num_det_classes = num_det_classes
        self.num_seg_classes = num_seg_classes
        self.nms_thresh = nms_thresh
        self.nms_type = nms_type.lower()
        self.point_mode = bool(point_mode)
        self.keypoint_mode = bool(keypoint_mode or point_mode)
        self.point_nms_dist = float(point_nms_dist)
        self.waterline_score_thresh = float(waterline_score_thresh)
        self.waterline_keep_all = bool(waterline_keep_all)

    def __call__(self, outputs, image_sizes, batched_inputs):
        pred_logits = outputs.get("pred_logits")
        pred_boxes = outputs.get("pred_boxes")
        pred_points = outputs.get("pred_points")
        pred_masks = outputs.get("pred_masks")
        results = []
        for batch_idx, ((height, width), sample) in enumerate(zip(image_sizes, batched_inputs)):
            if pred_logits is not None and (pred_boxes is not None or pred_points is not None):
                det_logits = pred_logits[batch_idx, : self.num_detection_queries, : self.num_det_classes]
                if pred_points is not None:
                    det_points_norm = pred_points[batch_idx, : self.num_detection_queries]
                else:
                    det_points_norm = pred_boxes[batch_idx, : self.num_detection_queries, :2]
                det_boxes = (
                    pred_boxes[batch_idx, : self.num_detection_queries]
                    if pred_boxes is not None
                    else None
                )
                points, scores, labels = self._decode_character_points(
                    det_logits, det_points_norm, width, height, sample
                )
                boxes = (
                    self._points_to_tiny_boxes(points)
                    if self.point_mode or self.keypoint_mode
                    else self._decode_boxes_fallback(det_logits, det_boxes, width, height, sample)[0]
                )
            else:
                device = outputs["pred_depth"].device
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

            # ---- Waterline: true curve points in keypoint mode ----
            total_q = pred_logits.shape[1] if pred_logits is not None else 0
            if (
                self.keypoint_mode
                and pred_logits is not None
                and total_q > self.num_detection_queries
            ):
                wl_logits = pred_logits[batch_idx, self.num_detection_queries :, 0]
                if pred_points is not None:
                    wl_pts_norm = pred_points[batch_idx, self.num_detection_queries :]
                else:
                    wl_pts_norm = pred_boxes[batch_idx, self.num_detection_queries :, :2]
                wl_points, wl_scores, sem = self._decode_waterline_curve(
                    wl_logits, wl_pts_norm, width, height, sample
                )
                result["waterline_points"] = wl_points
                result["waterline_scores"] = wl_scores
                result["sem_seg"] = sem
            elif (
                pred_masks is not None
                and self.num_seg_classes > 0
                and pred_logits is not None
                and pred_logits.shape[1] > self.num_detection_queries
            ):
                seg_logits = pred_logits[batch_idx, self.num_detection_queries :, : max(self.num_seg_classes, 1)]
                seg_masks = pred_masks[batch_idx, self.num_detection_queries :]
                result["sem_seg"] = self._decode_segmentation(seg_logits, seg_masks, width, height, sample)
                result["waterline_points"] = torch.zeros(0, 2, device=points.device)
                result["waterline_scores"] = torch.zeros(0, device=points.device)
            else:
                device = points.device
                result["sem_seg"] = torch.zeros(
                    1, sample.get("orig_height", height), sample.get("orig_width", width), device=device
                )
                result["waterline_points"] = torch.zeros(0, 2, device=device)
                result["waterline_scores"] = torch.zeros(0, device=device)

            if "pred_depth" in outputs:
                result["draft_depth"] = outputs["pred_depth"][batch_idx].detach()
                result["draft_valid"] = torch.sigmoid(outputs["pred_depth_valid_logits"][batch_idx]).detach()
            else:
                result["draft_depth"] = torch.tensor(0.0, device=points.device)
                result["draft_valid"] = torch.tensor(0.0, device=points.device)
            results.append(result)
        return results

    def _decode_character_points(self, logits, points_norm, width, height, sample):
        prob = logits.sigmoid()
        scores, labels = prob.max(dim=-1)
        keep = scores > self.score_thresh
        scores, labels, points_norm = scores[keep], labels[keep], points_norm[keep]
        if scores.numel() == 0:
            dev = logits.device
            return (
                torch.zeros(0, 2, device=dev),
                torch.zeros(0, device=dev),
                torch.zeros(0, dtype=torch.int64, device=dev),
            )
        if scores.numel() > self.topk:
            scores, idx = scores.topk(self.topk)
            labels = labels[idx]
            points_norm = points_norm[idx]
        points = points_norm * torch.tensor([width, height], device=points_norm.device, dtype=points_norm.dtype)
        points = self._restore_points_from_padding(points, sample)
        if (self.point_mode or self.keypoint_mode) and len(scores) > 1:
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
                    ordered_labels = labels[order]
                    suppressed[i + 1 :] |= (d < thr) & (ordered_labels[i + 1 :] == ordered_labels[i])
            keep_idx = torch.tensor(keep_idx, device=scores.device, dtype=torch.long)
            points, scores, labels = points[keep_idx], scores[keep_idx], labels[keep_idx]
        return points, scores, labels

    def _decode_waterline_curve(self, logits, points_norm, width, height, sample):
        """Decode dynamic-length ordered waterline points (existence-gated)."""
        scores = logits.sigmoid()
        points = points_norm * torch.tensor(
            [width, height], device=points_norm.device, dtype=points_norm.dtype
        )
        points = self._restore_points_from_padding(points, sample)

        # Training pads invalid slots at the end; decode as a left-to-right prefix
        # of high-confidence points (true dynamic length, typically 2..max_slots).
        above = scores >= self.waterline_score_thresh
        if above.any():
            # Longest prefix: count leading Trues after filling tiny holes of length 1.
            valid = above.clone()
            if valid.numel() >= 3:
                for i in range(1, valid.numel() - 1):
                    if (not valid[i]) and valid[i - 1] and valid[i + 1]:
                        valid[i] = True
            # Prefer a prefix starting at slot 0 (matches padded-GT layout).
            end = 0
            while end < valid.numel() and valid[end]:
                end += 1
            if end == 0:
                # Use the longest contiguous run; do not include low-confidence gaps.
                runs = []
                start = None
                for index, enabled in enumerate(valid.tolist() + [False]):
                    if enabled and start is None:
                        start = index
                    elif not enabled and start is not None:
                        runs.append((start, index))
                        start = None
                start, end = max(runs, key=lambda run: run[1] - run[0])
                points_out = points[start:end]
                scores_out = scores[start:end]
            else:
                points_out = points[:end]
                scores_out = scores[:end]
        else:
            points_out = points[:0]
            scores_out = scores[:0]

        orig_h = int(sample.get("orig_height", height))
        orig_w = int(sample.get("orig_width", width))
        sem = self._rasterize_polyline(points_out, orig_h, orig_w, scores_out)
        return points_out, scores_out, sem

    def _rasterize_polyline(self, points, height, width, scores=None):
        device = points.device if hasattr(points, "device") else "cpu"
        sem = np.zeros((height, width), dtype=np.float32)
        if points is None or len(points) == 0:
            return torch.from_numpy(sem).unsqueeze(0).to(device)
        pts = np.rint(points.detach().float().cpu().numpy()).astype(np.int32)
        thickness = max(1, int(round(height / 256.0 * 2.0)))
        if len(pts) >= 2:
            cv2.polylines(sem, [pts], isClosed=False, color=1.0, thickness=thickness)
        else:
            cv2.circle(sem, tuple(pts[0]), thickness, color=1.0, thickness=-1)
        return torch.from_numpy(sem).unsqueeze(0).to(device)

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
        pts[:, 0].clamp_(0, max(float(sample.get("orig_width", 1)) - 1.0, 0.0))
        pts[:, 1].clamp_(0, max(float(sample.get("orig_height", 1)) - 1.0, 0.0))
        return pts

    def _restore_boxes_from_padding(self, boxes, sample):
        scale = float(sample.get("resize_scale", 1.0)) or 1.0
        pad_left = float(sample.get("pad_left", 0))
        pad_top = float(sample.get("pad_top", 0))
        out = boxes.clone()
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_left) / scale
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_top) / scale
        return out
