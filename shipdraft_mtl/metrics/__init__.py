from __future__ import annotations

import copy
import numpy as np

from utils.metrics.detection_metrics import DetectionMetrics
from utils.metrics.segmentation_metrics import SegmentationMetrics
from utils.metrics.waterline_metrics import WaterlineMetrics
from utils.metrics.depth_metrics import DepthMetrics

__all__ = ["DraftFormerMetric", "build_metric"]


class DraftFormerMetric:
    """Stage-aware metrics for character points, waterline curves, and draft depth.

    Best-checkpoint selection always uses ``main_indicator`` (higher is better):
      - stage1: ``keypoint_score``  (characters + waterline)
      - stage2: ``draft_score``     (draft depth / MADDE)
      - stage3: ``hybrid_score``    (joint, draft-primary)
    """

    def __init__(self, config):
        self.main_indicator = config.get("main_indicator", "hybrid_score")
        self.det_class_names = list(config.get("det_class_names", []))
        if not self.det_class_names:
            raise ValueError("Metric.det_class_names must be populated from dataset metadata")

        # Backward-compatible aliases + new explicit names.
        self.hybrid_char_metric = config.get(
            "hybrid_char_metric",
            config.get("hybrid_det_metric", "PCK_class_aware"),
        )
        self.hybrid_waterline_metric = config.get(
            "hybrid_waterline_metric",
            config.get("hybrid_seg_metric", "waterline_curve_score"),
        )
        self.hybrid_draft_metric = config.get("hybrid_draft_metric", "draft_score")

        self.hybrid_char_weight = float(
            config.get("hybrid_char_weight", config.get("hybrid_det_weight", 0.5))
        )
        self.hybrid_waterline_weight = float(
            config.get("hybrid_waterline_weight", config.get("hybrid_seg_weight", 0.5))
        )
        self.hybrid_draft_weight = float(config.get("hybrid_draft_weight", 0.0))

        # Which tasks to score this stage (controls updates + score composition).
        self.evaluate_character = bool(config.get("evaluate_character", True))
        self.evaluate_waterline = bool(config.get("evaluate_waterline", True))
        self.evaluate_depth = bool(config.get("evaluate_depth", True))

        self.point_mode = bool(config.get("point_mode", False))
        self.keypoint_mode = bool(config.get("keypoint_mode", self.point_mode))
        self.evaluate_auxiliary = bool(config.get("evaluate_auxiliary", False))
        # Tighter default for character keypoints (fraction of max image side).
        self.pck_threshold = float(config.get("pck_threshold", 0.02))
        # Normalize curve / y-L1 into higher-is-better scores.
        self.curve_score_ref_px = float(config.get("curve_score_ref_px", 10.0))
        self.curve_y_score_ref_px = float(config.get("curve_y_score_ref_px", self.curve_score_ref_px))
        self.curve_pck_threshold = float(
            config.get("curve_pck_threshold", self.pck_threshold)
        )
        # keypoint_score = w_char * char_score + w_wl * waterline_score
        # char_score = 0.5 * PCK_class_aware + 0.5 * point_f1
        # waterline_score = 0.5 * y_L1_score + 0.5 * y_L1_median_score
        self.keypoint_char_weight = float(config.get("keypoint_char_weight", 0.5))
        self.keypoint_waterline_weight = float(config.get("keypoint_waterline_weight", 0.5))
        self.keypoint_char_pck_weight = float(config.get("keypoint_char_pck_weight", 0.5))
        self.keypoint_char_f1_weight = float(config.get("keypoint_char_f1_weight", 0.5))
        self.keypoint_wl_mean_weight = float(config.get("keypoint_wl_mean_weight", 0.5))
        self.keypoint_wl_median_weight = float(config.get("keypoint_wl_median_weight", 0.5))

        # Legacy keys still used by some configs.
        self.hybrid_det_metric = self.hybrid_char_metric
        self.hybrid_seg_metric = self.hybrid_waterline_metric
        self.hybrid_det_weight = self.hybrid_char_weight
        self.hybrid_seg_weight = self.hybrid_waterline_weight

        self.det_metric = DetectionMetrics(
            num_classes=len(self.det_class_names), class_names=self.det_class_names
        )
        self.seg_metric = SegmentationMetrics(num_classes=2, class_names=["background", "water"])
        self.waterline_metric = WaterlineMetrics(water_class_id=1)
        self.depth_metric = DepthMetrics(epsilon_list=[0.05, 0.1, 0.2, 0.5])

        self._curve_l1_sum = 0.0
        self._curve_l1_count = 0
        self._curve_y_l1_sum = 0.0
        self._curve_y_l1_images: list = []  # per-image mean |dy| for median
        self._curve_pck_hits = 0
        self._curve_pck_total = 0
        self._curve_presence_tp = 0
        self._curve_presence_fp = 0
        self._curve_presence_fn = 0
        self._curve_cardinality_score_sum = 0.0
        self._curve_cardinality_count = 0
        self._pck_hits = 0
        self._pck_class_hits = 0
        self._pck_total = 0
        self._point_predictions = 0
        self._point_hits_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self._point_gt_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self._point_predictions_by_class = np.zeros(len(self.det_class_names), dtype=np.int64)
        self._depth_samples = 0
        self._depth_valid_correct = 0
        self._depth_valid_total = 0

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
                if not self.point_mode and self.evaluate_character:
                    self._update_detection(output, sample)
                if self.evaluate_waterline:
                    self._update_segmentation(output, sample)
                    if self.keypoint_mode:
                        self._update_waterline_curve(output, sample)
            elif not self.point_mode and self.evaluate_waterline:
                self._update_segmentation(output, sample)

            if self.evaluate_depth:
                self._update_draft(output, sample)
            if self.evaluate_character and self.point_mode and self.evaluate_auxiliary:
                self._update_point_pck(output, sample)

    def _update_detection(self, output, sample):
        if "boxes" in output and len(output.get("boxes", [])) >= 0:
            pred_boxes = (
                output["boxes"].detach().cpu().numpy()
                if len(output["boxes"]) > 0
                else np.zeros((0, 4), dtype=np.float32)
            )
        else:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
        pred_scores = (
            output["scores"].detach().cpu().numpy()
            if len(output.get("scores", [])) > 0
            else np.zeros((0,), dtype=np.float32)
        )
        pred_labels = (
            output["labels"].detach().cpu().numpy()
            if len(output.get("labels", [])) > 0
            else np.zeros((0,), dtype=np.int64)
        )

        if self.point_mode and "points" in output and len(output["points"]) > 0:
            pts = output["points"].detach().cpu().numpy()
            r = 3.0
            pred_boxes = np.stack(
                [pts[:, 0] - r, pts[:, 1] - r, pts[:, 0] + r, pts[:, 1] + r], axis=-1
            ).astype(np.float32)

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
        # Stage-1 (no depth head) emits placeholder 0 depth; do not score it.
        valid_conf = None
        if "draft_valid" in output:
            valid_conf = output["draft_valid"]
            valid_conf = float(
                valid_conf.detach().cpu().item() if hasattr(valid_conf, "detach") else valid_conf
            )
            # If model has no depth head, postprocess sets draft_valid=0 constant;
            # still score when GT is valid and we are in a depth stage.
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
        if valid_conf is not None:
            self._depth_valid_total += 1
            self._depth_valid_correct += int((valid_conf >= 0.5) == is_valid)
        if not is_valid:
            self.depth_metric.add_sample(pred, None, filename=str(sample.get("file_name", "unk")))
        else:
            self.depth_metric.add_sample(pred, gt_v, filename=str(sample.get("file_name", "unk")))
            self._depth_samples += 1

    def _update_waterline_curve(self, output, sample):
        """Curve L1 / PCK between ordered predicted and GT waterline points."""
        pred = output.get("waterline_points")
        gt = sample.get("waterline_curve_points_original")
        if gt is None:
            gt = sample.get("waterline_curve_points")
        valid = sample.get("waterline_curve_valid")
        has_wl = sample.get("has_waterline")
        if has_wl is not None:
            has_wl = bool(has_wl.detach().cpu().item() if hasattr(has_wl, "detach") else has_wl)
        if pred is None or gt is None:
            return
        pred = pred.detach().cpu().numpy().reshape(-1, 2)
        gt = np.asarray(gt.detach().cpu().numpy() if hasattr(gt, "detach") else gt, dtype=np.float32).reshape(
            -1, 2
        )
        if valid is not None:
            valid_np = np.asarray(
                valid.detach().cpu().numpy() if hasattr(valid, "detach") else valid, dtype=bool
            ).reshape(-1)
            if valid_np.size == gt.shape[0]:
                gt = gt[valid_np]
            elif valid_np.any():
                gt = gt[: int(valid_np.sum())]
        gt_present = has_wl is not False and len(gt) > 0
        pred_present = len(pred) > 0
        if gt_present and pred_present:
            self._curve_presence_tp += 1
        elif pred_present:
            self._curve_presence_fp += 1
        elif gt_present:
            self._curve_presence_fn += 1
        if not gt_present:
            return
        if sample.get("waterline_curve_points_original") is None and "resize_scale" in sample:
            # GT still in resized/padded coords.
            gt = self._restore_points(gt, sample)

        oh = float(sample.get("orig_height", sample.get("height", 1)))
        ow = float(sample.get("orig_width", sample.get("width", 1)))
        if len(pred) == 0:
            # Completely missed waterline: count full curve as misses.
            self._curve_cardinality_score_sum += 0.0
            self._curve_cardinality_count += 1
            self._curve_pck_total += len(gt)
            penalty = float(max(oh, ow))
            self._curve_l1_sum += penalty
            self._curve_l1_count += 1
            self._curve_y_l1_sum += penalty
            self._curve_y_l1_images.append(penalty)
            return

        cardinality_score = 1.0 - abs(len(pred) - len(gt)) / max(len(pred), len(gt), 1)
        self._curve_cardinality_score_sum += cardinality_score
        self._curve_cardinality_count += 1

        # Compare on the larger cardinality so missing curve extent is not discarded.
        n = max(len(pred), len(gt))
        n = max(n, 1)

        def _resample(arr, count):
            if len(arr) == count:
                return arr
            if len(arr) == 1:
                return np.repeat(arr, count, axis=0)
            idx = np.linspace(0, len(arr) - 1, count)
            low = np.floor(idx).astype(int)
            high = np.minimum(low + 1, len(arr) - 1)
            w = (idx - low)[:, None]
            return arr[low] * (1 - w) + arr[high] * w

        pred_r = _resample(pred, n)
        gt_r = _resample(gt, n)
        dists = np.linalg.norm(pred_r - gt_r, axis=1)
        dy = np.abs(pred_r[:, 1] - gt_r[:, 1])
        mean_l1 = float(np.mean(dists))
        mean_y = float(np.mean(dy))

        self._curve_l1_sum += mean_l1
        self._curve_l1_count += 1
        self._curve_y_l1_sum += mean_y
        self._curve_y_l1_images.append(mean_y)

        thr = self.curve_pck_threshold * max(oh, ow)
        self._curve_pck_hits += int((dists <= thr).sum())
        self._curve_pck_total += int(len(dists))

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
        is_waterline = sample.get("is_waterline")
        if is_waterline is not None:
            wl = is_waterline.detach().cpu().numpy().astype(bool)
            if wl.shape[0] == character_mask.shape[0]:
                character_mask = character_mask & (~wl)
        gt = gt[character_mask]
        gt_labels = gt_labels[character_mask]
        self._point_predictions += len(pred)
        for label in pred_labels:
            if 0 <= int(label) < len(self.det_class_names):
                self._point_predictions_by_class[int(label)] += 1
        for label in gt_labels:
            if 0 <= int(label) < len(self.det_class_names):
                self._point_gt_by_class[int(label)] += 1
        if len(gt) == 0:
            return
        oh = float(sample.get("orig_height", sample.get("height", 1)))
        ow = float(sample.get("orig_width", sample.get("width", 1)))
        thr = self.pck_threshold * max(oh, ow)
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
                if 0 <= int(gt_label) < len(self.det_class_names):
                    self._point_hits_by_class[int(gt_label)] += 1

    def get_metric(self):
        results = {
            "main_indicator": self.main_indicator,
            "evaluate_character": float(self.evaluate_character),
            "evaluate_waterline": float(self.evaluate_waterline),
            "evaluate_depth": float(self.evaluate_depth),
        }

        if self.evaluate_auxiliary and not self.point_mode and self.evaluate_character:
            results.update(self.det_metric.compute())

        if self.evaluate_auxiliary and self.evaluate_waterline:
            seg_results = self.seg_metric.compute()
            water_results = self.waterline_metric.compute()
            results["mIoU"] = seg_results["mIoU"]
            results["pixel_accuracy"] = seg_results["pixel_accuracy"]
            results["mean_accuracy"] = seg_results["mean_accuracy"]
            results["FWIoU"] = seg_results["FWIoU"]
            results["water_iou"] = seg_results["per_class"]["water"]["IoU"]
            results["water_accuracy"] = seg_results["per_class"]["water"]["accuracy"]
            results.update(water_results)
        else:
            results["water_iou"] = 0.0

        # ---- Character keypoint metrics ----
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
        results["point_f1"] = (
            2.0 * point_precision * point_recall / max(point_precision + point_recall, 1e-12)
        )
        results["point_per_class"] = {
            class_name: {
                "precision": float(
                    self._point_hits_by_class[index]
                    / max(self._point_predictions_by_class[index], 1)
                ),
                "recall": float(
                    self._point_hits_by_class[index] / max(self._point_gt_by_class[index], 1)
                ),
                "num_gt": int(self._point_gt_by_class[index]),
                "num_predictions": int(self._point_predictions_by_class[index]),
            }
            for index, class_name in enumerate(self.det_class_names)
        }

        # ---- Waterline curve metrics (primary for keypoint_mode) ----
        if self._curve_l1_count > 0:
            curve_l1 = self._curve_l1_sum / self._curve_l1_count
            curve_y_l1 = self._curve_y_l1_sum / self._curve_l1_count
            results["waterline_curve_L1"] = curve_l1
            results["waterline_y_L1"] = curve_y_l1
            results["waterline_curve_score"] = 1.0 / (
                1.0 + max(curve_l1, 0.0) / max(self.curve_score_ref_px, 1e-6)
            )
            results["waterline_y_score"] = 1.0 / (
                1.0 + max(curve_y_l1, 0.0) / max(self.curve_y_score_ref_px, 1e-6)
            )
        else:
            results["waterline_curve_L1"] = 0.0
            results["waterline_y_L1"] = 0.0
            results["waterline_curve_score"] = 0.0
            results["waterline_y_score"] = 0.0

        if self._curve_y_l1_images:
            y_median = float(np.median(np.asarray(self._curve_y_l1_images, dtype=np.float64)))
            results["waterline_y_L1_median"] = y_median
            results["waterline_y_median_score"] = 1.0 / (
                1.0 + max(y_median, 0.0) / max(self.curve_y_score_ref_px, 1e-6)
            )
        else:
            results["waterline_y_L1_median"] = 0.0
            results["waterline_y_median_score"] = 0.0

        if self._curve_pck_total > 0:
            results["waterline_curve_PCK"] = self._curve_pck_hits / self._curve_pck_total
        else:
            results["waterline_curve_PCK"] = 0.0

        presence_precision = self._curve_presence_tp / max(
            self._curve_presence_tp + self._curve_presence_fp, 1
        )
        presence_recall = self._curve_presence_tp / max(
            self._curve_presence_tp + self._curve_presence_fn, 1
        )
        presence_f1 = (
            2.0 * presence_precision * presence_recall
            / max(presence_precision + presence_recall, 1e-12)
        )
        cardinality_score = self._curve_cardinality_score_sum / max(
            self._curve_cardinality_count, 1
        )
        results["waterline_presence_precision"] = presence_precision
        results["waterline_presence_recall"] = presence_recall
        results["waterline_presence_f1"] = presence_f1
        results["waterline_cardinality_score"] = cardinality_score

        # Stage-1 selection score:
        #   char = 0.5 * PCK_class_aware + 0.5 * point_f1   (tighter PCK threshold)
        #   wl   = 0.5 * waterline_y_score + 0.5 * waterline_y_median_score
        #   keypoint_score = 0.5 * char + 0.5 * wl
        pck_cls = float(results.get("PCK_class_aware", 0.0) or 0.0)
        point_f1 = float(results.get("point_f1", 0.0) or 0.0)
        char_w = max(self.keypoint_char_pck_weight, 0.0) + max(self.keypoint_char_f1_weight, 0.0)
        if char_w > 0:
            char_score = (
                max(self.keypoint_char_pck_weight, 0.0) * pck_cls
                + max(self.keypoint_char_f1_weight, 0.0) * point_f1
            ) / char_w
        else:
            char_score = pck_cls
        results["char_score"] = char_score

        y_score = float(results.get("waterline_y_score", 0.0) or 0.0)
        y_med_score = float(results.get("waterline_y_median_score", 0.0) or 0.0)
        wl_w = max(self.keypoint_wl_mean_weight, 0.0) + max(self.keypoint_wl_median_weight, 0.0)
        if wl_w > 0:
            wl_score = (
                max(self.keypoint_wl_mean_weight, 0.0) * y_score
                + max(self.keypoint_wl_median_weight, 0.0) * y_med_score
            ) / wl_w
        else:
            wl_score = y_score
        results["waterline_score"] = wl_score * presence_f1 * cardinality_score

        if self.evaluate_character and self.evaluate_waterline:
            total_w = max(self.keypoint_char_weight, 0.0) + max(self.keypoint_waterline_weight, 0.0)
            if total_w <= 0:
                results["keypoint_score"] = 0.5 * char_score + 0.5 * wl_score
            else:
                results["keypoint_score"] = (
                    max(self.keypoint_char_weight, 0.0) * char_score
                    + max(self.keypoint_waterline_weight, 0.0) * wl_score
                ) / total_w
        elif self.evaluate_character:
            results["keypoint_score"] = char_score
        elif self.evaluate_waterline:
            results["keypoint_score"] = wl_score
        else:
            results["keypoint_score"] = 0.0

        # ---- Draft depth ----
        if self.evaluate_depth:
            depth_results = self.depth_metric.compute()
            results.update(depth_results)
            madde = float(depth_results.get("MADDE", depth_results.get("madde", 0.0)) or 0.0)
            results["MADDE"] = madde
            results["draft_score"] = (
                1.0 / (1.0 + max(madde, 0.0)) if self._depth_samples > 0 else 0.0
            )
            results["depth_valid_samples"] = float(self._depth_samples)
            results["draft_valid_accuracy"] = (
                self._depth_valid_correct / self._depth_valid_total
                if self._depth_valid_total > 0
                else 0.0
            )
        else:
            results["MADDE"] = 0.0
            results["draft_score"] = 0.0
            results["depth_valid_samples"] = 0.0
            results["draft_valid_accuracy"] = 0.0

        # Joint score used by stage-3 (draft-primary by default weights).
        results["hybrid_score"] = self._hybrid_score(results)

        # Selection score alias for logging.
        results["selection_score"] = float(results.get(self.main_indicator, 0.0) or 0.0)
        results["selection_valid"] = not self.evaluate_depth or self._depth_samples > 0

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
        self._curve_l1_sum = 0.0
        self._curve_l1_count = 0
        self._curve_y_l1_sum = 0.0
        self._curve_y_l1_images = []
        self._curve_pck_hits = 0
        self._curve_pck_total = 0
        self._curve_presence_tp = 0
        self._curve_presence_fp = 0
        self._curve_presence_fn = 0
        self._curve_cardinality_score_sum = 0.0
        self._curve_cardinality_count = 0
        self._depth_samples = 0
        self._depth_valid_correct = 0
        self._depth_valid_total = 0

        return results

    def _metric_value(self, results, key):
        # Convert lower-is-better L1 keys to higher-is-better scores for hybrid selection.
        lower_is_better = {
            "waterline_curve_L1": "waterline_curve_score",
            "waterline_y_L1": "waterline_y_score",
            "waterline_y_L1_median": "waterline_y_median_score",
        }
        if key in lower_is_better:
            return float(results.get(lower_is_better[key], 0.0) or 0.0)
        return float(results.get(key, 0.0) or 0.0)

    def _hybrid_score(self, results):
        parts = []
        weights = []
        if self.hybrid_char_weight > 0 and self.evaluate_character:
            parts.append(self._metric_value(results, self.hybrid_char_metric))
            weights.append(self.hybrid_char_weight)
        if self.hybrid_waterline_weight > 0 and self.evaluate_waterline:
            parts.append(self._metric_value(results, self.hybrid_waterline_metric))
            weights.append(self.hybrid_waterline_weight)
        if self.hybrid_draft_weight > 0 and self.evaluate_depth:
            parts.append(self._metric_value(results, self.hybrid_draft_metric))
            weights.append(self.hybrid_draft_weight)
        if not parts:
            # Fallbacks by stage intent.
            if self.evaluate_depth:
                return float(results.get("draft_score", 0.0) or 0.0)
            return float(results.get("keypoint_score", 0.0) or 0.0)
        weight_sum = sum(weights)
        return float(sum(w * s for w, s in zip(weights, parts)) / max(weight_sum, 1e-12))


def build_metric(config):
    config = copy.deepcopy(config)
    name = config.pop("name", "DraftFormerMetric")
    if name not in {"DraftFormerMetric", "MultitaskMetric"}:
        raise ValueError(f"Unsupported metric: {name}")
    return DraftFormerMetric(config)
