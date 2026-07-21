from __future__ import annotations

from torch.utils.data import DataLoader, DistributedSampler

from .common import DatasetMetadata
from .json_multitask_dataset import JsonMultitaskDataset
from .json_draft_e2e_dataset import JsonDraftE2EDataset
from .yolo_multitask_dataset import YoloMultitaskDataset

DATASET_TYPES = {
    "JsonMultitaskDataset": JsonMultitaskDataset,
    "JsonDraftE2EDataset": JsonDraftE2EDataset,
    "YoloMultitaskDataset": YoloMultitaskDataset,
}

__all__ = [
    "DatasetMetadata",
    "JsonMultitaskDataset",
    "JsonDraftE2EDataset",
    "YoloMultitaskDataset",
    "build_dataloader",
]


def _apply_metadata(config, metadata: DatasetMetadata) -> None:
    data_cfg = config.setdefault("Data", {})
    detection_classes = list(metadata.detection_classes)
    segmentation_classes = [c for c in metadata.segmentation_classes if not str(c).startswith("__unused")]
    existing_detection = data_cfg.get("detection_classes")
    existing_segmentation = data_cfg.get("segmentation_classes")
    if existing_detection is not None and list(existing_detection) != detection_classes:
        raise ValueError(
            f"Train/Eval detection classes differ: {existing_detection} != {detection_classes}"
        )
    if existing_segmentation is not None and list(existing_segmentation) != segmentation_classes:
        # Allow empty e2e seg list vs placeholder-only metadata
        if not (not existing_segmentation and not segmentation_classes):
            raise ValueError(
                f"Train/Eval segmentation classes differ: {existing_segmentation} != {segmentation_classes}"
            )

    data_cfg["detection_classes"] = detection_classes
    data_cfg["segmentation_classes"] = segmentation_classes
    head_cfg = config["Architecture"]["Head"]
    # Direct-depth model uses character queries for detection and dedicated
    # waterline queries for segmentation.
    if config.get("Architecture", {}).get("point_mode", False):
        character_classes = [name for name in detection_classes if name.lower() != "waterline"]
        head_cfg["num_classes"] = len(character_classes)
        head_cfg["num_classes_seg"] = 1 if "waterline" in [name.lower() for name in detection_classes] else 0
    else:
        head_cfg["num_classes"] = metadata.num_detection_classes
        head_cfg["num_classes_seg"] = max(metadata.num_segmentation_classes if segmentation_classes else 0, head_cfg.get("num_classes_seg", 1) if segmentation_classes else 0)
        if segmentation_classes:
            head_cfg["num_classes_seg"] = len(segmentation_classes)
    metric_classes = character_classes if config.get("Architecture", {}).get("point_mode", False) else detection_classes
    config.setdefault("Metric", {})["det_class_names"] = metric_classes


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
