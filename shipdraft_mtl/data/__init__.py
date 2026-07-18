from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from .common import DatasetMetadata
from .json_multitask_dataset import JsonMultitaskDataset
from .yolo_multitask_dataset import YoloMultitaskDataset

DATASET_TYPES = {
    "JsonMultitaskDataset": JsonMultitaskDataset,
    "YoloMultitaskDataset": YoloMultitaskDataset,
}

__all__ = [
    "DatasetMetadata",
    "JsonMultitaskDataset",
    "YoloMultitaskDataset",
    "build_dataloader",
]


def _apply_metadata(config, metadata: DatasetMetadata) -> None:
    data_cfg = config.setdefault("Data", {})
    detection_classes = list(metadata.detection_classes)
    segmentation_classes = list(metadata.segmentation_classes)
    existing_detection = data_cfg.get("detection_classes")
    existing_segmentation = data_cfg.get("segmentation_classes")
    if existing_detection is not None and list(existing_detection) != detection_classes:
        raise ValueError(
            f"Train/Eval detection classes differ: {existing_detection} != {detection_classes}"
        )
    if existing_segmentation is not None and list(existing_segmentation) != segmentation_classes:
        raise ValueError(
            f"Train/Eval segmentation classes differ: {existing_segmentation} != {segmentation_classes}"
        )

    data_cfg["detection_classes"] = detection_classes
    data_cfg["segmentation_classes"] = segmentation_classes
    head_cfg = config["Architecture"]["Head"]
    head_cfg["num_classes"] = metadata.num_detection_classes
    head_cfg["num_classes_seg"] = metadata.num_segmentation_classes
    config.setdefault("Metric", {})["det_class_names"] = detection_classes


def build_dataloader(config, mode, logger, seed=None, epoch=1, task="multitask"):
    mode = mode.capitalize()
    if mode not in config:
        raise ValueError(f"Missing {mode} configuration")
    dataset_cfg = config[mode].get("dataset", {})
    dataset_name = dataset_cfg.get("name")
    try:
        dataset_class = DATASET_TYPES[dataset_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dataset type {dataset_name!r}; choose one of {sorted(DATASET_TYPES)}"
        ) from error

    dataset = dataset_class(config, mode, logger, seed, epoch=epoch, task=task)
    _apply_metadata(config, dataset.metadata)

    loader_cfg = config[mode]["loader"]
    sampler = None
    if config["Global"].get("distributed", False) and mode == "Train":
        sampler = DistributedSampler(dataset, shuffle=loader_cfg.get("shuffle", True))
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=int(loader_cfg["batch_size_per_card"]),
        shuffle=False if sampler is not None else loader_cfg.get("shuffle", False),
        drop_last=loader_cfg.get("drop_last", False),
        num_workers=int(loader_cfg.get("num_workers", 0)),
        pin_memory=loader_cfg.get("pin_memory", False),
        collate_fn=dataset.collate_fn,
    )
    if not len(loader):
        raise ValueError(f"No batches in {mode} dataloader")
    return loader
