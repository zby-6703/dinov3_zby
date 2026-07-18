from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2

from .common import (
    IMAGE_EXTENSIONS,
    DatasetMetadata,
    MultitaskDataset,
    natural_label_key,
    normalize_path,
)

DETECTION_SHAPES = {"rectangle"}
SEGMENTATION_SHAPES = {"polygon"}
# LabelMe point annotations are auxiliary marks, not DraftFormer targets.
IGNORED_SHAPES = {"point"}


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid LabelMe JSON: {path}: {error}") from error
    if not isinstance(data.get("shapes"), list):
        raise ValueError(f"LabelMe JSON has no shapes list: {path}")
    return data


def _json_files(data_root: str) -> Iterable[str]:
    for directory, _, filenames in os.walk(data_root):
        for filename in sorted(filenames):
            if filename.lower().endswith(".json"):
                yield os.path.join(directory, filename)


def infer_json_metadata(data_root: str) -> DatasetMetadata:
    detection_labels = set()
    segmentation_labels = set()
    for label_path in _json_files(data_root):
        for shape in _read_json(label_path)["shapes"]:
            label = str(shape.get("label", "")).strip()
            shape_type = str(shape.get("shape_type", "")).lower()
            if shape_type in IGNORED_SHAPES:
                continue
            if not label:
                raise ValueError(f"Empty shape label in {label_path}")
            if shape_type in DETECTION_SHAPES:
                detection_labels.add(label)
            elif shape_type in SEGMENTATION_SHAPES:
                segmentation_labels.add(label)
            else:
                raise ValueError(
                    f"Unsupported LabelMe shape_type={shape_type!r} in {label_path}; "
                    "DraftFormer JSON labels must use rectangle for detection or polygon for segmentation"
                )
    return DatasetMetadata(
        tuple(sorted(detection_labels, key=natural_label_key)),
        tuple(sorted(segmentation_labels, key=natural_label_key)),
    )


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


def _flatten_points(points: Sequence[Sequence[float]], path: str) -> List[float]:
    try:
        polygon = [float(value) for point in points for value in point[:2]]
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid shape points in {path}") from error
    if len(polygon) < 4 or len(polygon) % 2:
        raise ValueError(f"Invalid shape points in {path}")
    return polygon


def _parse_shapes(
    data: Mapping[str, Any],
    label_path: str,
    metadata: DatasetMetadata,
    width: int,
    height: int,
) -> List[Dict[str, Any]]:
    det_ids = {name: index for index, name in enumerate(metadata.detection_classes)}
    seg_ids = {
        name: metadata.num_detection_classes + index
        for index, name in enumerate(metadata.segmentation_classes)
    }
    annotations = []
    for shape in data["shapes"]:
        label = str(shape.get("label", "")).strip()
        shape_type = str(shape.get("shape_type", "")).lower()
        if shape_type in IGNORED_SHAPES:
            continue
        points = _flatten_points(shape.get("points", []), label_path)
        xs = points[0::2]
        ys = points[1::2]
        x1, x2 = max(0.0, min(xs)), min(float(width), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(height), max(ys))
        if x2 <= x1 or y2 <= y1:
            continue
        if shape_type == "rectangle":
            category_id = det_ids[label]
            polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
        elif shape_type == "polygon":
            if len(points) < 6:
                continue
            category_id = seg_ids[label]
            polygon = points
        else:
            raise ValueError(f"Unsupported shape_type={shape_type!r} in {label_path}")
        annotations.append(
            {"bbox": [x1, y1, x2, y2], "segmentation": [polygon], "category_id": category_id}
        )
    return annotations


def load_json_records(data_root: str, split: str, metadata: DatasetMetadata) -> List[Dict[str, Any]]:
    split_dir = _resolve_split_dir(data_root, split)
    label_paths = sorted(_json_files(split_dir))
    if not label_paths:
        raise ValueError(f"No JSON labels found in {split_dir}")

    records = []
    for label_path in label_paths:
        data = _read_json(label_path)
        image_path = _resolve_image(label_path, data)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        height, width = image.shape[:2]
        relative_id = os.path.relpath(label_path, data_root).replace("\\", "/")
        record = {
            "file_name": image_path,
            "label_file": os.path.abspath(label_path),
            "image_id": int(hashlib.sha1(relative_id.encode("utf-8")).hexdigest()[:15], 16),
            "height": height,
            "width": width,
            "annotations": _parse_shapes(data, label_path, metadata, width, height),
        }
        if data.get("draft depth") is not None:
            record["draft_depth"] = float(data["draft depth"])
        records.append(record)
    return records


class JsonMultitaskDataset(MultitaskDataset):
    """LabelMe dataset: rectangles/polygons are targets; points are ignored."""

    def __init__(self, config, mode, logger, seed=None, epoch=1, task="multitask") -> None:
        dataset_cfg = config[mode]["dataset"]
        data_root = normalize_path(dataset_cfg.get("data_root"))
        if not os.path.isdir(data_root):
            raise ValueError(f"JSON dataset root not found: {data_root}")
        split = dataset_cfg.get("split", "train" if mode == "Train" else "val")
        metadata = infer_json_metadata(data_root)
        records = load_json_records(data_root, split, metadata)
        super().__init__(config, mode, logger, records, metadata, data_root, split)
