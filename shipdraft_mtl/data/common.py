from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from torch.utils.data import Dataset

from .dataset_mapper import MultitaskDatasetMapper

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def normalize_path(path: str) -> str:
    if not path:
        raise ValueError("dataset.data_root is required")
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))


def natural_label_key(label: str):
    try:
        return 0, int(label)
    except ValueError:
        return 1, label.casefold(), label


@dataclass(frozen=True)
class DatasetMetadata:
    detection_classes: tuple[str, ...]
    segmentation_classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.detection_classes:
            raise ValueError("The dataset contains no detection classes")
        if not self.segmentation_classes:
            raise ValueError("The dataset contains no segmentation classes")
        overlap = set(self.detection_classes) & set(self.segmentation_classes)
        if overlap:
            raise ValueError(f"Labels cannot belong to both tasks: {sorted(overlap)}")

    @property
    def num_detection_classes(self) -> int:
        return len(self.detection_classes)

    @property
    def num_segmentation_classes(self) -> int:
        return len(self.segmentation_classes)


class MultitaskDataset(Dataset):
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str,
        logger,
        records: List[Dict[str, Any]],
        metadata: DatasetMetadata,
        data_root: str,
        split: str,
    ) -> None:
        dataset_cfg = config[mode]["dataset"]
        if dataset_cfg.get("filter_empty", False):
            records = [record for record in records if record.get("annotations")]
        if not records:
            raise ValueError(f"No usable samples found in {data_root} split={split}")

        self.config = config
        self.mode = mode
        self.data_root = data_root
        self.split = split
        self.records = records
        self.metadata = metadata
        self.class_names = list(metadata.detection_classes)
        self.segmentation_class_names = list(metadata.segmentation_classes)
        self.mapper = MultitaskDatasetMapper(
            image_format=dataset_cfg.get("image_format", config.get("Input", {}).get("format", "BGR")),
            target_size_wh=tuple(
                dataset_cfg.get("image_size", config.get("Input", {}).get("image_size", [256, 640]))
            ),
            pad_value=dataset_cfg.get("pad_value", 0),
            num_detection_classes=metadata.num_detection_classes,
        )
        self.need_reset = False
        logger.info(
            "%s dataset: format=%s, root=%s, split=%s, detection_classes=%s, "
            "segmentation_classes=%s, samples=%d",
            mode,
            self.__class__.__name__,
            data_root,
            split,
            self.class_names,
            self.segmentation_class_names,
            len(records),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.mapper(self.records[index])

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return list(batch)
