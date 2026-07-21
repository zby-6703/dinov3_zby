from __future__ import annotations

import copy
import numpy as np

from utils.metrics.detection_metrics import DetectionMetrics
from utils.metrics.segmentation_metrics import SegmentationMetrics
from utils.metrics.waterline_metrics import WaterlineMetrics
from utils.metrics.depth_metrics import DepthMetrics

__all__ = ["DraftFormerMetric", "build_metric"]


class DraftFormerMetric:
    def __init__(self, config):
        self.main_indicator = config.get("main_indicator", "mAP@0.5")
        self.det_class_names = list(config.get("det_class_names", []))
        if not self.det_class_names:
            raise ValueError("Metric.det_class_names must be populated from dataset metadata")
        self.hybrid_det_metric = config.get("hybrid_det_metric", "mAP@0.5")
        self.hybrid_seg_metric = config.get("hybrid_seg_metric", "water_iou")
        self.hybrid_det_weight = float(config.get("hybrid_det_weight", 0.5))
        self.hybrid_seg_weight = float(config.get("hybrid_seg_weight", 0.5))
        self.hybrid_draft_weight = float(config.get("hybrid_draft_weight", 0.0))
        self.point_mode = bool(config.get("point_mode", False))
        self.evaluate_auxiliary = bool(config.get("evaluate_auxiliary", False))
        self.det_metric = DetectionMetrics(num_classes=len(self.det_class_names), class_names=self.det_class_names)
        self.seg_metric = SegmentationMetrics(num_classes=2, class_names=["background", "water"])
        self.waterline_metric = WaterlineMetrics(water_class_id=1)
        self.depth_metric = DepthMetrics(epsilon_list=[0.05, 0.1, 0.2, 0.5])
        self._pck_hits = 0
        self._pck_class_hits = 0
        self._pck_total = 0
        self._point_predictions = 0
        self._point_hits_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self._point_gt_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self._point_predictions_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self.pck_threshold = float(config.get("pck_threshold", 0.05))

    @staticmethod
    def _restore_points(points, sample):
        points = np.asarray(points, dtype=np.float32).copy()
        if points.size == 0:
            return points.reshape(-1, 2)
        scale = float(sample.get("resize_scale", 1.0)) or 1.0
        points[:, 0] = (points[:, 0] - float(sample.get("pad_left", 0))) / scale
        points[:, 1] = (points[:, 1] - float(sample.get("pad_top", 0))) / scale
        return points

    @classmethod
    def _restore_boxes(cls, boxes, sample):
        boxes = np.asarray(boxes, dtype=np.float32).copy().reshape(-1, 4)
        if boxes.size == 0:
            return boxes
        points = boxes.reshape(-1, 2)
        return cls._restore_points(points, sample).reshape(-1, 4)

    def __call__(self, outputs, batch):
        for output, sample in zip(outputs, batch):
            if self.evaluate_auxiliary:
                if not self.point_mode:
                    self._update_detection(output, sample)
                self._update_segmentation(output, sample)
            elif not self.point_mode:
                self._update_segmentation(output, sample)
            self._update_draft(output, sample)
            if self.point_mode and self.evaluate_auxiliary:
                self._update_point_pck(output, sample)

    def _update_detection(self, output, sample):
        if "boxes" in output and len(output.get("boxes", [])) >= 0:
            pred_boxes = output["boxes"].detach().cpu().numpy() if len(output["boxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_scores = output["scores"].detach().cpu().numpy() if len(output.get("scores", [])) > 0 else np.zeros((0,), dtype=np.float32)
        pred_labels = output["labels"].detach().cpu().numpy() if len(output.get("labels", [])) > 0 else np.zeros((0,), dtype=np.int64)

        # In point mode convert points to tiny boxes for existing det metric
        if self.point_mode and "points" in output and len(output["points"]) > 0:
            pts = output["points"].detach().cpu().numpy()
            r = 3.0
            pred_boxes = np.stack([pts[:, 0] - r, pts[:, 1] - r, pts[:, 0] + r, pts[:, 1] + r], axis=-1).astype(np.float32)

        gt_boxes = sample.get("gt_boxes_original")
        gt_labels = sample.get("gt_classes")
        if gt_boxes is None or gt_labels is None:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
        else:
            gt_boxes = self._restore_boxes(gt_boxes.detach().cpu().numpy(), sample)
            gt_labels = gt_labels.detach().cpu().numpy()
            keep = gt_labels < len(self.det_class_names)
            gt_boxes = gt_boxes[keep]
            gt_labels = gt_labels[keep]

        self.det_metric.add_image(
            image_id=sample.get("image_id", len(self.det_metric.image_ids)),
            pred_boxes=pred_boxes,
            pred_scores=pred_scores,
            pred_labels=pred_labels,
            gt_boxes=gt_boxes,
            gt_labels=gt_labels,
        )

    def _update_segmentation(self, output, sample):
        gt_sem_seg = sample.get("gt_sem_seg")
        if gt_sem_seg is None or "sem_seg" not in output:
            return
        gt_sem_seg = gt_sem_seg.detach().cpu().numpy().astype(np.uint8)
        pred_sem_seg = output["sem_seg"]
        if pred_sem_seg.ndim == 3:
            pred_sem_seg = pred_sem_seg[0]
        pred_sem_seg = (pred_sem_seg.detach().cpu().numpy() > 0.5).astype(np.uint8)
        self.seg_metric.add_image(pred_sem_seg, gt_sem_seg)
        self.waterline_metric.add_image(
            pred_sem_seg,
            gt_sem_seg,
            filename=str(sample.get("file_name", sample.get("image_id", "unknown"))),
        )

    def _update_draft(self, output, sample):
        if "draft_depth" not in output:
            return
        pred = output["draft_depth"]
        pred = float(pred.detach().cpu().item() if hasattr(pred, "detach") else pred)
        gt = sample.get("draft_depth")
        valid = sample.get("draft_depth_valid")
        if gt is None:
            self.depth_metric.add_sample(pred, None, filename=str(sample.get("file_name", "unk")))
            return
        gt_v = float(gt.detach().cpu().item() if hasattr(gt, "detach") else gt)
        is_valid = True
        if valid is not None:
            is_valid = bool(valid.detach().cpu().item() if hasattr(valid, "detach") else valid)
        if not is_valid:
            self.depth_metric.add_sample(pred, None, filename=str(sample.get("file_name", "unk")))
        else:
            self.depth_metric.add_sample(pred, gt_v, filename=str(sample.get("file_name", "unk")))

    def _update_point_pck(self, output, sample):
        if "points" not in output or "points" not in sample:
            return
        pred = output["points"].detach().cpu().numpy()
        pred_labels = output["labels"].detach().cpu().numpy()
        gt = self._restore_points(sample["points"].detach().cpu().numpy(), sample)
        gt_labels_t = sample.get("gt_classes", sample.get("labels"))
        if gt_labels_t is None:
            return
        gt_labels = gt_labels_t.detach().cpu().numpy()
        character_mask = gt_labels < len(self.det_class_names)
        gt = gt[character_mask]
        gt_labels = gt_labels[character_mask]
        self._point_predictions += len(pred)
        for label in pred_labels:
            if 0 <= int(label) < len(self.det_class_names):
                self._point_predictions_by_class[int(label)] += 1
        for label in gt_labels:
            self._point_gt_by_class[int(label)] += 1
        if len(gt) == 0:
            return
        oh = float(sample.get("orig_height", sample.get("height", 1)))
        ow = float(sample.get("orig_width", sample.get("width", 1)))
        thr = self.pck_threshold * max(oh, ow)
        # greedy match by distance ignoring class for rough PCK
        used = np.zeros(len(pred), dtype=bool)
        used_class = np.zeros(len(pred), dtype=bool)
        for g, gt_label in zip(gt, gt_labels):
            if len(pred) == 0:
                self._pck_total += 1
                continue
            d = np.linalg.norm(pred - g[None, :], axis=1)
            d[used] = 1e9
            j = int(d.argmin())
            self._pck_total += 1
            if d[j] <= thr:
                self._pck_hits += 1
                used[j] = True
            class_distance = np.linalg.norm(pred - g[None, :], axis=1)
            class_distance[used_class | (pred_labels != gt_label)] = 1e9
            class_index = int(class_distance.argmin())
            if class_distance[class_index] <= thr:
                self._pck_class_hits += 1
                used_class[class_index] = True
                self._point_hits_by_class[int(gt_label)] += 1

    def get_metric(self):
        results = {}
        if self.evaluate_auxiliary and not self.point_mode:
            results.update(self.det_metric.compute())

        if self.evaluate_auxiliary:
            seg_results = self.seg_metric.compute()
            water_results = self.waterline_metric.compute()
            results["mIoU"] = seg_results["mIoU"]
            results["pixel_accuracy"] = seg_results["pixel_accuracy"]
            results["mean_accuracy"] = seg_results["mean_accuracy"]
            results["FWIoU"] = seg_results["FWIoU"]
            results["water_iou"] = seg_results["per_class"]["water"]["IoU"]
            results["water_accuracy"] = seg_results["per_class"]["water"]["accuracy"]
            results.update(water_results)
        elif self.point_mode:
            results["water_iou"] = 0.0

        depth_results = self.depth_metric.compute()
        # DepthMetrics may use MADDE key
        results.update(depth_results)
        madde = float(depth_results.get("MADDE", depth_results.get("madde", 0.0)) or 0.0)
        results["MADDE"] = madde
        results["draft_score"] = 1.0 / (1.0 + max(madde, 0.0))
        if self._pck_total > 0:
            results["PCK"] = self._pck_hits / self._pck_total
            results["PCK_class_aware"] = self._pck_class_hits / self._pck_total
        else:
            results["PCK"] = 0.0
            results["PCK_class_aware"] = 0.0
        point_precision = self._pck_class_hits / max(self._point_predictions, 1)
        point_recall = self._pck_class_hits / max(self._pck_total, 1)
        results["point_precision"] = point_precision
        results["point_recall"] = point_recall
        results["point_f1"] = 2.0 * point_precision * point_recall / max(point_precision + point_recall, 1e-12)
        results["point_per_class"] = {
            class_name: {
                "precision": float(self._point_hits_by_class[index] / max(self._point_predictions_by_class[index], 1)),
                "recall": float(self._point_hits_by_class[index] / max(self._point_gt_by_class[index], 1)),
                "num_gt": int(self._point_gt_by_class[index]),
                "num_predictions": int(self._point_predictions_by_class[index]),
            }
            for index, class_name in enumerate(self.det_class_names)
        }

        self.det_metric.reset()
        self.seg_metric.reset()
        self.waterline_metric.reset()
        self.depth_metric.reset()
        self._pck_hits = 0
        self._pck_class_hits = 0
        self._pck_total = 0
        self._point_predictions = 0
        self._point_hits_by_class.fill(0)
        self._point_gt_by_class.fill(0)
        self._point_predictions_by_class.fill(0)

        results["hybrid_score"] = self._hybrid_score(results)
        return results

    def _hybrid_score(self, results):
        det_score = float(results.get(self.hybrid_det_metric, 0.0) or 0.0)
        seg_score = float(results.get(self.hybrid_seg_metric, 0.0) or 0.0)
        draft_score = float(results.get("draft_score", 0.0) or 0.0)
        weight_sum = self.hybrid_det_weight + self.hybrid_seg_weight + self.hybrid_draft_weight
        if weight_sum <= 0:
            return draft_score
        return (
            self.hybrid_det_weight * det_score
            + self.hybrid_seg_weight * seg_score
            + self.hybrid_draft_weight * draft_score
        ) / weight_sum


def build_metric(config):
    config = copy.deepcopy(config)
    name = config.pop("name", "DraftFormerMetric")
    if name not in {"DraftFormerMetric", "MultitaskMetric"}:
        raise ValueError(f"Unsupported metric: {name}")
    return DraftFormerMetric(config)
