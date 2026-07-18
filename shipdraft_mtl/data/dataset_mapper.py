from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, Tuple

import cv2
import numpy as np
import torch

from shipdraft_mtl.preprocess import resize_with_padding


def _polygon_mask(
    polygons: Iterable[Iterable[float]],
    height: int,
    width: int,
    *,
    scale: float = 1.0,
    pad_left: int = 0,
    pad_top: int = 0,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32)
        if points.size < 6 or points.size % 2:
            continue
        points = points.reshape(-1, 2)
        points[:, 0] = points[:, 0] * scale + pad_left
        points[:, 1] = points[:, 1] * scale + pad_top
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
    return mask


class MultitaskDatasetMapper:
    """Convert a format-independent record into DraftFormer's tensor contract."""

    def __init__(
        self,
        *,
        image_format: str,
        target_size_wh: Tuple[int, int],
        pad_value: int,
        num_detection_classes: int,
    ) -> None:
        image_format = image_format.upper()
        if image_format not in {"BGR", "RGB"}:
            raise ValueError(f"image_format must be BGR or RGB, got {image_format!r}")
        if len(target_size_wh) != 2 or min(target_size_wh) <= 0:
            raise ValueError(f"image_size must be [width, height], got {target_size_wh!r}")
        self.image_format = image_format
        self.target_size_wh = tuple(int(value) for value in target_size_wh)
        self.pad_value = int(pad_value)
        self.num_detection_classes = int(num_detection_classes)

    def __call__(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sample = copy.deepcopy(record)
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

        boxes = []
        labels = []
        masks = []
        original_boxes = []
        original_labels = []
        semantic_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

        for annotation in sample.get("annotations", []):
            x1, y1, x2, y2 = (float(value) for value in annotation["bbox"])
            transformed_box = [
                np.clip(x1 * scale + pad_left, 0, target_w),
                np.clip(y1 * scale + pad_top, 0, target_h),
                np.clip(x2 * scale + pad_left, 0, target_w),
                np.clip(y2 * scale + pad_top, 0, target_h),
            ]
            if transformed_box[2] <= transformed_box[0] or transformed_box[3] <= transformed_box[1]:
                continue

            polygons = annotation["segmentation"]
            mask = _polygon_mask(
                polygons,
                target_h,
                target_w,
                scale=scale,
                pad_left=pad_left,
                pad_top=pad_top,
            )
            if not mask.any():
                continue

            category_id = int(annotation["category_id"])
            boxes.append(transformed_box)
            labels.append(category_id)
            masks.append(mask)
            original_boxes.append([x1, y1, x2, y2])
            original_labels.append(category_id)

            if category_id >= self.num_detection_classes:
                semantic_mask = np.maximum(
                    semantic_mask,
                    _polygon_mask(polygons, orig_h, orig_w),
                )

        mask_tensor = torch.from_numpy(np.stack(masks)) if masks else torch.zeros(
            (0, target_h, target_w), dtype=torch.uint8
        )
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
                "image": torch.from_numpy(
                    np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
                ),
                "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(labels, dtype=torch.int64),
                "masks": mask_tensor,
                "gt_boxes_original": torch.tensor(original_boxes, dtype=torch.float32).reshape(-1, 4),
                "gt_classes": torch.tensor(original_labels, dtype=torch.int64),
                "gt_sem_seg": torch.from_numpy(semantic_mask),
            }
        )
        return sample
