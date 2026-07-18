from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List

import cv2
import numpy as np

from .common import IMAGE_EXTENSIONS, DatasetMetadata, MultitaskDataset, normalize_path


def _class_names(raw: Any, task: str) -> tuple[str, ...]:
    if isinstance(raw, Mapping):
        try:
            items = sorted((int(class_id), str(name)) for class_id, name in raw.items())
        except (TypeError, ValueError) as error:
            raise ValueError(f"YOLO {task} class ids must be integers") from error
        expected = list(range(len(items)))
        if [class_id for class_id, _ in items] != expected:
            raise ValueError(f"YOLO {task} class ids must be contiguous from 0")
        names = tuple(name for _, name in items)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        names = tuple(str(name) for name in raw)
    else:
        raise ValueError(
            "YOLO dataset requires dataset.class_names.detection and "
            "dataset.class_names.segmentation as ID-to-name mappings or lists"
        )
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError(f"YOLO {task} class names must be non-empty and unique")
    return names


def yolo_metadata(dataset_cfg: Mapping[str, Any]) -> DatasetMetadata:
    class_names = dataset_cfg.get("class_names")
    if not isinstance(class_names, Mapping):
        raise ValueError("YOLO dataset requires dataset.class_names with detection and segmentation mappings")
    return DatasetMetadata(
        _class_names(class_names.get("detection"), "detection"),
        _class_names(class_names.get("segmentation"), "segmentation"),
    )


def _parse_detection_labels(path: str, width: int, height: int, num_classes: int):
    annotations = []
    if not os.path.isfile(path):
        return annotations
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            values = line.split()
            if not values or values[0].startswith("#"):
                continue
            if len(values) != 5:
                raise ValueError(f"Expected 5 YOLO detection values at {path}:{line_no}")
            class_id = int(values[0])
            cx, cy, box_w, box_h = (float(value) for value in values[1:])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"Detection class id {class_id} is not mapped at {path}:{line_no}")
            x1 = np.clip((cx - box_w / 2) * width, 0, width)
            y1 = np.clip((cy - box_h / 2) * height, 0, height)
            x2 = np.clip((cx + box_w / 2) * width, 0, width)
            y2 = np.clip((cy + box_h / 2) * height, 0, height)
            if x2 <= x1 or y2 <= y1:
                continue
            polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
            annotations.append(
                {"bbox": [x1, y1, x2, y2], "segmentation": [polygon], "category_id": class_id}
            )
    return annotations


def _parse_segmentation_labels(
    path: str, width: int, height: int, num_classes: int, category_offset: int
):
    annotations = []
    if not os.path.isfile(path):
        return annotations
    with open(path, "r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            values = line.split()
            if not values or values[0].startswith("#"):
                continue
            if len(values) < 7 or len(values) % 2 == 0:
                raise ValueError(f"Invalid YOLO segmentation polygon at {path}:{line_no}")
            class_id = int(values[0])
            if not 0 <= class_id < num_classes:
                raise ValueError(f"Segmentation class id {class_id} is not mapped at {path}:{line_no}")
            points = np.asarray([float(value) for value in values[1:]], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            if x2 <= x1 or y2 <= y1:
                continue
            annotations.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "segmentation": [points.reshape(-1).tolist()],
                    "category_id": category_offset + class_id,
                }
            )
    return annotations


def load_yolo_records(data_root: str, split: str, metadata: DatasetMetadata) -> List[Dict[str, Any]]:
    image_dir = os.path.join(data_root, "images", split)
    detection_dir = os.path.join(data_root, "labels", "detection", split)
    segmentation_dir = os.path.join(data_root, "labels", "segmentation", split)
    for directory in (image_dir, detection_dir, segmentation_dir):
        if not os.path.isdir(directory):
            raise ValueError(f"Required YOLO dataset directory not found: {directory}")

    image_paths = sorted(
        os.path.join(image_dir, filename)
        for filename in os.listdir(image_dir)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    )
    records = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        annotations = _parse_detection_labels(
            os.path.join(detection_dir, stem + ".txt"),
            width,
            height,
            metadata.num_detection_classes,
        )
        annotations.extend(
            _parse_segmentation_labels(
                os.path.join(segmentation_dir, stem + ".txt"),
                width,
                height,
                metadata.num_segmentation_classes,
                metadata.num_detection_classes,
            )
        )
        relative_id = os.path.relpath(image_path, data_root).replace("\\", "/")
        records.append(
            {
                "file_name": os.path.abspath(image_path),
                "image_id": int(hashlib.sha1(relative_id.encode("utf-8")).hexdigest()[:15], 16),
                "height": height,
                "width": width,
                "annotations": annotations,
            }
        )
    return records


class YoloMultitaskDataset(MultitaskDataset):
    """YOLO dataset with separate detection and segmentation TXT directories."""

    def __init__(self, config, mode, logger, seed=None, epoch=1, task="multitask") -> None:
        dataset_cfg = config[mode]["dataset"]
        data_root = normalize_path(dataset_cfg.get("data_root"))
        if not os.path.isdir(data_root):
            raise ValueError(f"YOLO dataset root not found: {data_root}")
        split = dataset_cfg.get("split", "train" if mode == "Train" else "val")
        metadata = yolo_metadata(dataset_cfg)
        records = load_yolo_records(data_root, split, metadata)
        super().__init__(config, mode, logger, records, metadata, data_root, split)
