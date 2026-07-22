from __future__ import annotations

"""ShipDraft e2e dataset: unified point-segmentation labels."""

import hashlib
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

from .common import (
    IMAGE_EXTENSIONS,
    DatasetMetadata,
    MultitaskDataset,
    natural_label_key,
    normalize_path,
)
from shipdraft_mtl.preprocess import resize_with_padding

WATERLINE_LABEL = "waterline"
MARK_POINT_TYPES = {"point"}
WATERLINE_TYPES = {"linestrip", "line", "linestring"}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid LabelMe JSON: {path}: {error}") from error
    if not isinstance(data.get("shapes"), list):
        raise ValueError(f"LabelMe JSON has no shapes list: {path}")
    return data


def _json_files(directory: str) -> Iterable[str]:
    for root, _, filenames in os.walk(directory):
        for filename in sorted(filenames):
            if filename.lower().endswith(".json"):
                yield os.path.join(root, filename)


def _resolve_split_dir(data_root: str, split: str) -> str:
    aliases = [split, "valid" if split == "val" else "val"] if split in {"val", "valid"} else [split]
    for name in aliases:
        candidate = os.path.join(data_root, name)
        if os.path.isdir(candidate):
            return candidate
    raise ValueError(f"Dataset split directory not found: {os.path.join(data_root, split)}")


def _resolve_image(label_path: str, data: Mapping[str, Any]) -> str:
    directory = os.path.dirname(label_path)
    image_path = data.get("imagePath")
    if image_path:
        candidate = image_path if os.path.isabs(image_path) else os.path.join(directory, image_path)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    stem = os.path.splitext(label_path)[0]
    for extension in IMAGE_EXTENSIONS:
        candidate = stem + extension
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise ValueError(f"No image found for label file: {label_path}")


def sample_polyline_points(
    points: Sequence[Sequence[float]],
    *,
    spacing_px: float = 12.0,
) -> List[List[float]]:
    pts = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [pts[0]]
    sampled: List[List[float]] = [pts[0]]
    for i in range(1, len(pts)):
        x0, y0 = sampled[-1]
        x1, y1 = pts[i]
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist < 1e-6:
            continue
        n_seg = max(1, int(math.floor(dist / max(spacing_px, 1e-3))))
        for s in range(1, n_seg + 1):
            t = s / n_seg
            sampled.append([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t])
    return sampled


def _polyline_arc_length(pts: Sequence[Sequence[float]]) -> float:
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total


def choose_waterline_point_count(
    points: Sequence[Sequence[float]],
    *,
    max_points: int = 10,
    min_points: int = 2,
    spacing_px: float = 48.0,
) -> int:
    """Adaptive curve cardinality from arc length, clamped to [min_points, max_points]."""
    pts = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
    max_points = max(int(max_points), 1)
    min_points = max(1, min(int(min_points), max_points))
    if len(pts) == 0:
        return 0
    if len(pts) == 1:
        return 1
    length = _polyline_arc_length(pts)
    if length < 1e-6:
        return 1
    # +1 so endpoints are covered when spacing tiles the segment.
    estimated = int(round(length / max(float(spacing_px), 1e-3))) + 1
    return int(max(min_points, min(max_points, estimated)))


def resample_polyline_fixed(
    points: Sequence[Sequence[float]],
    num_points: int,
) -> List[List[float]]:
    """Arc-length uniform resampling to a fixed number of ordered curve points."""
    pts = [[float(p[0]), float(p[1])] for p in points if len(p) >= 2]
    n = int(num_points)
    if n <= 0:
        return []
    if len(pts) == 0:
        return [[0.0, 0.0] for _ in range(n)]
    if len(pts) == 1:
        return [pts[0][:] for _ in range(n)]
    if n == 1:
        return [pts[0][:]]

    # Ensure left-to-right order for waterline-like curves.
    if pts[0][0] > pts[-1][0]:
        pts = list(reversed(pts))

    lengths = [0.0]
    for i in range(1, len(pts)):
        lengths.append(lengths[-1] + math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]))
    total = lengths[-1]
    if total < 1e-6:
        return [pts[0][:] for _ in range(n)]

    targets = [total * i / (n - 1) for i in range(n)]
    resampled: List[List[float]] = []
    seg = 0
    for t in targets:
        while seg + 1 < len(lengths) and lengths[seg + 1] < t:
            seg += 1
        if seg + 1 >= len(lengths):
            resampled.append(pts[-1][:])
            continue
        seg_len = lengths[seg + 1] - lengths[seg]
        alpha = 0.0 if seg_len < 1e-6 else (t - lengths[seg]) / seg_len
        x = pts[seg][0] + alpha * (pts[seg + 1][0] - pts[seg][0])
        y = pts[seg][1] + alpha * (pts[seg + 1][1] - pts[seg][1])
        resampled.append([x, y])
    return resampled


def build_dynamic_waterline_curve(
    points: Sequence[Sequence[float]],
    *,
    max_points: int = 10,
    min_points: int = 2,
    spacing_px: float = 48.0,
) -> Tuple[List[List[float]], List[bool]]:
    """Return padded curve coords + validity mask (True for real points)."""
    max_points = max(int(max_points), 1)
    n = choose_waterline_point_count(
        points, max_points=max_points, min_points=min_points, spacing_px=spacing_px
    )
    coords = [[0.0, 0.0] for _ in range(max_points)]
    valid = [False] * max_points
    if n <= 0:
        return coords, valid
    curve = resample_polyline_fixed(points, n)
    for i, pt in enumerate(curve[:max_points]):
        coords[i] = [float(pt[0]), float(pt[1])]
        valid[i] = True
    return coords, valid


def infer_e2e_metadata(data_root: str) -> DatasetMetadata:
    mark_labels = set()
    has_waterline = False
    for label_path in _json_files(data_root):
        for shape in _read_json(label_path)["shapes"]:
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).lower()
            if not label:
                continue
            if shape_type in MARK_POINT_TYPES and label.lower() != WATERLINE_LABEL:
                mark_labels.add(label)
            elif shape_type in WATERLINE_TYPES or label.lower() == WATERLINE_LABEL:
                has_waterline = True
    if not mark_labels:
        raise ValueError(f"No mark point labels found under {data_root}")
    detection = tuple(sorted(mark_labels, key=natural_label_key))
    if has_waterline:
        detection = detection + (WATERLINE_LABEL,)
    return DatasetMetadata(detection, ("__unused_point_seg__",))


def _parse_draft_depth(data: Mapping[str, Any]) -> Tuple[Optional[float], bool]:
    if "draft_depth_valid" in data and data["draft_depth_valid"] is not None:
        valid_flag = bool(data["draft_depth_valid"])
    else:
        valid_flag = None
    raw = data.get("draft depth", data.get("draft_depth"))
    if raw is None:
        return None, False if valid_flag is None else bool(valid_flag)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(value) or value <= 0.0:
        return None, False
    if valid_flag is False:
        return None, False
    return value, True


def parse_e2e_points(
    data: Mapping[str, Any],
    label_path: str,
    class_to_id: Mapping[str, int],
    *,
    waterline_spacing_px: float,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    annotations: List[Dict[str, Any]] = []
    waterline_id = class_to_id.get(WATERLINE_LABEL)
    for shape in data["shapes"]:
        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "")).lower()
        raw_points = shape.get("points") or []
        if not label or not raw_points:
            continue
        if shape_type in MARK_POINT_TYPES and label.lower() != WATERLINE_LABEL:
            if label not in class_to_id:
                raise ValueError(f"Unknown mark label {label!r} in {label_path}")
            x, y = float(raw_points[0][0]), float(raw_points[0][1])
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(f"Non-finite point coordinate in {label_path}: {raw_points[0]!r}")
            x = float(np.clip(x, 0, max(width - 1e-3, 0)))
            y = float(np.clip(y, 0, max(height - 1e-3, 0)))
            annotations.append(
                {
                    "point": [x, y],
                    "category_id": int(class_to_id[label]),
                    "is_waterline": False,
                    "label_name": label,
                }
            )
            continue
        if shape_type in WATERLINE_TYPES or label.lower() == WATERLINE_LABEL:
            if waterline_id is None:
                continue
            for point in raw_points:
                if len(point) < 2 or not all(math.isfinite(float(v)) for v in point[:2]):
                    raise ValueError(f"Invalid waterline coordinate in {label_path}: {point!r}")
            poly = sample_polyline_points(raw_points, spacing_px=waterline_spacing_px)
            for x, y in poly:
                x = float(np.clip(x, 0, max(width - 1e-3, 0)))
                y = float(np.clip(y, 0, max(height - 1e-3, 0)))
                annotations.append(
                    {
                        "point": [x, y],
                        "category_id": int(waterline_id),
                        "is_waterline": True,
                        "label_name": WATERLINE_LABEL,
                    }
                )
    return annotations


def load_e2e_records(
    data_root: str,
    split: str,
    metadata: DatasetMetadata,
    *,
    waterline_spacing_px: float = 12.0,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> List[Dict[str, Any]]:
    split_dir = _resolve_split_dir(data_root, split)
    label_paths = sorted(_json_files(split_dir))
    if not label_paths:
        raise ValueError(f"No JSON labels found in {split_dir}")
    class_to_id = {name: index for index, name in enumerate(metadata.detection_classes)}
    records: List[Dict[str, Any]] = []
    for label_path in label_paths:
        data = _read_json(label_path)
        image_path = _resolve_image(label_path, data)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        relative_id = os.path.relpath(label_path, data_root).replace("\\", "/")
        annotations = parse_e2e_points(
            data,
            label_path,
            class_to_id,
            waterline_spacing_px=waterline_spacing_px,
            width=width,
            height=height,
        )
        draft_depth, draft_valid = _parse_draft_depth(data)
        if draft_valid and draft_depth is not None:
            if min_depth is not None and draft_depth < min_depth:
                raise ValueError(f"Draft depth {draft_depth} below model minimum {min_depth}: {label_path}")
            if max_depth is not None and draft_depth > max_depth:
                raise ValueError(f"Draft depth {draft_depth} above model maximum {max_depth}: {label_path}")
        records.append(
            {
                "file_name": image_path,
                "label_file": os.path.abspath(label_path),
                "image_id": int(hashlib.sha1(relative_id.encode("utf-8")).hexdigest()[:15], 16),
                "height": height,
                "width": width,
                "annotations": annotations,
                "draft_depth": draft_depth,
                "draft_depth_valid": draft_valid,
            }
        )
    return records


class PointSegDatasetMapper:
    def __init__(
        self,
        *,
        image_format: str,
        target_size_wh: Tuple[int, int],
        pad_value: int,
        pseudo_box_size: float = 0.05,
        num_waterline_points: int = 10,
        min_waterline_points: int = 2,
        waterline_target_spacing_px: float = 48.0,
    ) -> None:
        image_format = image_format.upper()
        if image_format not in {"BGR", "RGB"}:
            raise ValueError(f"image_format must be BGR or RGB, got {image_format!r}")
        self.image_format = image_format
        self.target_size_wh = tuple(int(v) for v in target_size_wh)
        self.pad_value = int(pad_value)
        self.pseudo_box_size = float(pseudo_box_size)
        self.num_waterline_points = int(num_waterline_points)
        self.min_waterline_points = int(min_waterline_points)
        self.waterline_target_spacing_px = float(waterline_target_spacing_px)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sample = dict(record)
        image = cv2.imread(sample["file_name"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {sample['file_name']}")
        if self.image_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        image, resize = resize_with_padding(image, self.target_size_wh, self.pad_value)
        target_h, target_w = image.shape[:2]
        scale = float(resize["scale"])
        pad_left = int(resize["pad_left"])
        pad_top = int(resize["pad_top"])
        points, labels, is_waterline = [], [], []
        original_waterline_points = []
        for ann in sample.get("annotations", []):
            x, y = ann["point"]
            tx = float(np.clip(x * scale + pad_left, 0, target_w - 1e-3))
            ty = float(np.clip(y * scale + pad_top, 0, target_h - 1e-3))
            points.append([tx, ty])
            labels.append(int(ann["category_id"]))
            point_is_waterline = bool(ann.get("is_waterline", False))
            is_waterline.append(point_is_waterline)
            if point_is_waterline:
                original_waterline_points.append([float(x), float(y)])
        points_t = torch.tensor(points, dtype=torch.float32).reshape(-1, 2)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        waterline_t = torch.tensor(is_waterline, dtype=torch.bool)
        waterline_mask = np.zeros((target_h, target_w), dtype=np.uint8)
        waterline_points = points_t[waterline_t]
        if len(waterline_points) >= 2:
            cv2.polylines(
                waterline_mask,
                [np.rint(waterline_points.numpy()).astype(np.int32)],
                isClosed=False,
                color=1,
                thickness=max(1, int(round(target_h / 256.0 * 2.0))),
            )
        elif len(waterline_points) == 1:
            x, y = np.rint(waterline_points[0].numpy()).astype(np.int32).tolist()
            cv2.circle(waterline_mask, (int(x), int(y)), max(1, int(round(target_h / 256.0))), 1, -1)
        gt_waterline_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if len(original_waterline_points) >= 2:
            cv2.polylines(
                gt_waterline_mask,
                [np.rint(np.asarray(original_waterline_points)).astype(np.int32)],
                isClosed=False,
                color=1,
                thickness=max(1, int(round(orig_h / 256.0 * 2.0))),
            )
        elif len(original_waterline_points) == 1:
            x, y = np.rint(original_waterline_points[0]).astype(np.int32).tolist()
            cv2.circle(gt_waterline_mask, (int(x), int(y)), max(1, int(round(orig_h / 256.0))), 1, -1)

        # Dynamic-length ordered curve (padded to max slots). Validity marks real points.
        has_waterline = len(original_waterline_points) > 0
        if has_waterline:
            # Spacing is defined in original pixels; scale for the resized canvas.
            spacing_resized = max(self.waterline_target_spacing_px * scale, 1.0)
            curve_orig, valid_list = build_dynamic_waterline_curve(
                original_waterline_points,
                max_points=self.num_waterline_points,
                min_points=self.min_waterline_points,
                spacing_px=self.waterline_target_spacing_px,
            )
            curve_resized = []
            for (x, y), is_valid in zip(curve_orig, valid_list):
                if not is_valid:
                    curve_resized.append([0.0, 0.0])
                    continue
                tx = float(np.clip(x * scale + pad_left, 0, target_w - 1e-3))
                ty = float(np.clip(y * scale + pad_top, 0, target_h - 1e-3))
                curve_resized.append([tx, ty])
            waterline_curve_points = torch.tensor(curve_resized, dtype=torch.float32).reshape(-1, 2)
            waterline_curve_points_original = torch.tensor(curve_orig, dtype=torch.float32).reshape(-1, 2)
            waterline_curve_valid = torch.tensor(valid_list, dtype=torch.bool)
            # Keep spacing_resized referenced so future density tweaks are easy to wire.
            _ = spacing_resized
        else:
            waterline_curve_points = torch.zeros((self.num_waterline_points, 2), dtype=torch.float32)
            waterline_curve_points_original = torch.zeros((self.num_waterline_points, 2), dtype=torch.float32)
            waterline_curve_valid = torch.zeros((self.num_waterline_points,), dtype=torch.bool)

        if len(points_t):
            pw = max(self.pseudo_box_size * target_w, 2.0)
            ph = max(self.pseudo_box_size * target_h, 2.0)
            boxes = torch.stack(
                [
                    points_t[:, 0] - pw * 0.5,
                    points_t[:, 1] - ph * 0.5,
                    points_t[:, 0] + pw * 0.5,
                    points_t[:, 1] + ph * 0.5,
                ],
                dim=-1,
            )
            boxes[:, 0::2] = boxes[:, 0::2].clamp(0, target_w)
            boxes[:, 1::2] = boxes[:, 1::2].clamp(0, target_h)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        draft_depth = sample.get("draft_depth")
        draft_valid = bool(sample.get("draft_depth_valid", draft_depth is not None))
        if draft_depth is None:
            draft_valid = False
            draft_depth_value = 0.0
        else:
            draft_depth_value = float(draft_depth)
        sample.update(
            {
                "orig_height": int(orig_h),
                "orig_width": int(orig_w),
                "height": int(target_h),
                "width": int(target_w),
                "resize_scale": scale,
                "pad_left": pad_left,
                "pad_top": pad_top,
                "pad_right": int(resize["pad_right"]),
                "pad_bottom": int(resize["pad_bottom"]),
                "resize_padding": {**resize, "orig_h": int(orig_h), "orig_w": int(orig_w)},
                "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)),
                "points": points_t,
                "labels": labels_t,
                "is_waterline": waterline_t,
                "waterline_mask": torch.from_numpy(waterline_mask),
                "waterline_curve_points": waterline_curve_points,
                "waterline_curve_points_original": waterline_curve_points_original,
                "waterline_curve_valid": waterline_curve_valid,
                "has_waterline": torch.tensor(has_waterline, dtype=torch.bool),
                "boxes": boxes,
                "masks": torch.zeros((len(points_t), target_h, target_w), dtype=torch.uint8),
                "gt_boxes_original": boxes.clone(),
                "gt_classes": labels_t.clone(),
                "gt_sem_seg": torch.from_numpy(gt_waterline_mask),
                "draft_depth": torch.tensor(draft_depth_value, dtype=torch.float32),
                "draft_depth_valid": torch.tensor(draft_valid, dtype=torch.bool),
            }
        )
        return sample


class JsonDraftE2EDataset(MultitaskDataset):
    def __init__(self, config, mode, logger, seed=None, epoch=1, task="multitask") -> None:
        dataset_cfg = config[mode]["dataset"]
        data_root = normalize_path(dataset_cfg.get("data_root"))
        if not os.path.isdir(data_root):
            raise ValueError(f"JSON e2e dataset root not found: {data_root}")
        split = dataset_cfg.get("split", "train" if mode == "Train" else "val")
        metadata = infer_e2e_metadata(data_root)
        waterline_spacing = float(dataset_cfg.get("waterline_spacing_px", 12.0))
        depth_cfg = config.get("Architecture", {}).get("DirectDepth", {})
        records = load_e2e_records(
            data_root,
            split,
            metadata,
            waterline_spacing_px=waterline_spacing,
            min_depth=float(depth_cfg["min_depth"]) if depth_cfg.get("min_depth") is not None else None,
            max_depth=float(depth_cfg["max_depth"]) if depth_cfg.get("max_depth") is not None else None,
        )
        if dataset_cfg.get("filter_empty", False):
            records = [r for r in records if r.get("annotations")]
        if not records:
            raise ValueError(f"No usable e2e samples in {data_root} split={split}")
        self.config = config
        self.mode = mode
        self.data_root = data_root
        self.split = split
        self.records = records
        self.metadata = metadata
        self.class_names = list(metadata.detection_classes)
        self.segmentation_class_names = []
        decoder_cfg = config.get("Architecture", {}).get("Decoder", {})
        num_queries = int(decoder_cfg.get("num_queries", 110))
        num_det = int(decoder_cfg.get("num_detection_queries", 100))
        num_waterline_points = int(
            dataset_cfg.get("num_waterline_points", max(num_queries - num_det, 1))
        )
        self.mapper = PointSegDatasetMapper(
            image_format=dataset_cfg.get("image_format", config.get("Input", {}).get("format", "BGR")),
            target_size_wh=tuple(dataset_cfg.get("image_size", config.get("Input", {}).get("image_size", [256, 640]))),
            pad_value=dataset_cfg.get("pad_value", 0),
            pseudo_box_size=float(dataset_cfg.get("point_pseudo_box_size", 0.05)),
            num_waterline_points=num_waterline_points,
            min_waterline_points=int(dataset_cfg.get("min_waterline_points", 2)),
            waterline_target_spacing_px=float(
                dataset_cfg.get("waterline_target_spacing_px", dataset_cfg.get("waterline_spacing_px", 48.0))
            ),
        )
        self.need_reset = False
        logger.info(
            "%s dataset: format=%s, root=%s, split=%s, point_classes=%s, samples=%d",
            mode, self.__class__.__name__, data_root, split, self.class_names, len(records),
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.mapper(self.records[index])
