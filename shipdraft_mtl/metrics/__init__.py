from __future__ import annotations

import copy
import numpy as np

from utils.metrics.detection_metrics import DetectionMetrics
from utils.metrics.segmentation_metrics import SegmentationMetrics
from utils.metrics.waterline_metrics import WaterlineMetrics

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
        self.det_metric = DetectionMetrics(num_classes=len(self.det_class_names), class_names=self.det_class_names)
        self.seg_metric = SegmentationMetrics(num_classes=2, class_names=["background", "water"])
        self.waterline_metric = WaterlineMetrics(water_class_id=1)

    def __call__(self, outputs, batch):
        for output, sample in zip(outputs, batch):
            self._update_detection(output, sample)
            self._update_segmentation(output, sample)

    def _update_detection(self, output, sample):
        pred_boxes = output["boxes"].detach().cpu().numpy() if len(output["boxes"]) > 0 else np.zeros((0, 4), dtype=np.float32)
        pred_scores = output["scores"].detach().cpu().numpy() if len(output["scores"]) > 0 else np.zeros((0,), dtype=np.float32)
        pred_labels = output["labels"].detach().cpu().numpy() if len(output["labels"]) > 0 else np.zeros((0,), dtype=np.int64)

        gt_boxes = sample.get("gt_boxes_original")
        gt_labels = sample.get("gt_classes")
        if gt_boxes is None or gt_labels is None:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
        else:
            gt_boxes = gt_boxes.detach().cpu().numpy()
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
        if gt_sem_seg is None:
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

    def get_metric(self):
        det_results = self.det_metric.compute()
        seg_results = self.seg_metric.compute()
        water_results = self.waterline_metric.compute()

        self.det_metric.reset()
        self.seg_metric.reset()
        self.waterline_metric.reset()

        results = dict(det_results)
        results["mIoU"] = seg_results["mIoU"]
        results["pixel_accuracy"] = seg_results["pixel_accuracy"]
        results["mean_accuracy"] = seg_results["mean_accuracy"]
        results["FWIoU"] = seg_results["FWIoU"]
        results["water_iou"] = seg_results["per_class"]["water"]["IoU"]
        results["water_accuracy"] = seg_results["per_class"]["water"]["accuracy"]
        results.update(water_results)
        results["hybrid_score"] = self._hybrid_score(results)
        return results

    def _hybrid_score(self, results):
        det_score = float(results.get(self.hybrid_det_metric, 0.0))
        seg_score = float(results.get(self.hybrid_seg_metric, 0.0))
        weight_sum = self.hybrid_det_weight + self.hybrid_seg_weight
        if weight_sum <= 0:
            return 0.0
        return (
            self.hybrid_det_weight * det_score
            + self.hybrid_seg_weight * seg_score
        ) / weight_sum


def build_metric(config):
    config = copy.deepcopy(config)
    name = config.pop("name", "DraftFormerMetric")
    if name not in {"DraftFormerMetric", "MultitaskMetric"}:
        raise ValueError(f"Unsupported metric: {name}")
    return DraftFormerMetric(config)
