from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from shipdraft_mtl.modeling import build_model
from shipdraft_mtl.preprocess import resize_with_padding
from shipdraft_mtl.utils.ckpt import load_pretrained_params

from .config import Config

logger = logging.getLogger("shipdraft_mtl.engine.predictor")
logging.basicConfig(level=logging.INFO)


def setup_cfg(
    config_file: str,
    weights: Optional[str] = None,
    device: str = "cuda",
    opts: Optional[Dict] = None,
):
    cfg = Config(config_file)
    cfg.merge_dict(
        {
            "Global": {
                "device": "gpu" if device.startswith("cuda") else "cpu",
                "pretrained_model": weights,
                "checkpoints": None,
            }
        }
    )
    if opts:
        cfg.merge_dict(opts)
    return cfg


def resolve_torch_device(device_cfg: str) -> torch.device:
    if device_cfg == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False in this process. "
                "Run the command with GPU access instead of falling back to CPU."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def get_image_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        return []
    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        files.extend(glob.glob(os.path.join(input_path, ext)))
        files.extend(glob.glob(os.path.join(input_path, ext.upper())))
    return sorted(files)


@dataclass
class BoundingBoxPrediction:
    xyxy: np.ndarray
    score: float
    label: int
    class_name: str


@dataclass
class DetectionPrediction:
    boxes: List[BoundingBoxPrediction]


@dataclass
class SegmentationPrediction:
    sem_seg: np.ndarray
    orig_shape: Tuple[int, int]

    def get_binary_mask(self, threshold: float = 0.5) -> np.ndarray:
        sem_seg = self.sem_seg
        if sem_seg.ndim == 3:
            sem_seg = sem_seg[0]
        if threshold == 0.5:
            max_value = float(sem_seg.max()) if sem_seg.size > 0 else 0.0
            if max_value > 1.0:
                threshold = max(0.1, max_value * 0.3)
        return (sem_seg > threshold).astype(np.uint8)


@dataclass
class MultiTaskPrediction:
    detection: DetectionPrediction
    segmentation: SegmentationPrediction
    raw_output: Dict[str, torch.Tensor]
    image_path: Optional[str] = None


class DraftFormerPredictor:
    def __init__(
        self,
        cfg=None,
        *,
        config_file: Optional[str] = None,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        opts: Optional[Dict] = None,
    ):
        if cfg is None:
            if config_file is None:
                raise ValueError("DraftFormerPredictor requires cfg or config_file")
            cfg = setup_cfg(config_file, weights=weights_path, device=device, opts=opts)

        self.cfg = cfg.cfg if hasattr(cfg, "cfg") else cfg
        self.device = resolve_torch_device(self.cfg["Global"].get("device", "gpu"))
        backbone_name = self.cfg.get("Architecture", {}).get("Backbone", {}).get("name")
        if self.device.type == "cpu" and backbone_name == "HybridBackboneV2":
            raise RuntimeError(
                "DraftFormer HybridBackboneV2 currently requires CUDA for vmamba-backed operators. "
                "Use device='cuda' for inference with this backbone."
            )
        self.model = build_model(self.cfg).to(self.device)
        weights = (
            weights_path
            or self.cfg["Global"].get("pretrained_model")
            or self.cfg["Global"].get("checkpoints")
        )
        if weights:
            load_pretrained_params(self.model, weights, logger)
        self.model.eval()

        self.input_size = tuple(
            self.cfg.get("Input", {}).get(
                "image_size",
                self.cfg["Train"]["dataset"].get("image_size", [256, 640]),
            )
        )
        self.image_format = self.cfg.get("Input", {}).get(
            "format",
            self.cfg["Train"]["dataset"].get("image_format", "BGR"),
        )
        self.class_names = list(self.cfg.get("Metric", {}).get("det_class_names", []))
        if not self.class_names:
            self.class_names = list(self.cfg.get("Data", {}).get("detection_classes", []))
        if not self.class_names:
            raise ValueError("No detection class names in Metric.det_class_names or Data.detection_classes")

    def preprocess(self, image: np.ndarray) -> Dict:
        orig_h, orig_w = image.shape[:2]
        model_image = image
        if self.image_format.upper() == "RGB":
            model_image = cv2.cvtColor(model_image, cv2.COLOR_BGR2RGB)
        padded, params = resize_with_padding(model_image, self.input_size, 0)
        image_tensor = torch.as_tensor(padded.transpose(2, 0, 1).astype("float32"))
        return {
            "image": image_tensor,
            "height": params["target_h"],
            "width": params["target_w"],
            "orig_height": orig_h,
            "orig_width": orig_w,
            "resize_scale": float(params["scale"]),
            "pad_left": int(params["pad_left"]),
            "pad_top": int(params["pad_top"]),
            "pad_right": int(params["pad_right"]),
            "pad_bottom": int(params["pad_bottom"]),
        }

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        input_dict = self.preprocess(image)
        return self.model([input_dict])[0]

    @torch.no_grad()
    def predict(self, image_or_path: Union[str, os.PathLike, np.ndarray]) -> MultiTaskPrediction:
        image_path = None
        if isinstance(image_or_path, (str, os.PathLike, Path)):
            image_path = str(image_or_path)
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")
        else:
            image = image_or_path

        outputs = self(image)
        return self._to_prediction(outputs, image_path=image_path)

    def _to_prediction(
        self,
        outputs: Dict[str, torch.Tensor],
        image_path: Optional[str] = None,
    ) -> MultiTaskPrediction:
        boxes = outputs["boxes"].detach().cpu().numpy()
        scores = outputs["scores"].detach().cpu().numpy()
        labels = outputs["labels"].detach().cpu().numpy()
        sem_seg = outputs["sem_seg"].detach().cpu().numpy()
        orig_shape = tuple(outputs.get("image_size", sem_seg.shape[-2:]))

        det_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            class_name = self.class_names[int(label)] if int(label) < len(self.class_names) else f"class_{int(label)}"
            det_boxes.append(
                BoundingBoxPrediction(
                    xyxy=box.astype(np.float32),
                    score=float(score),
                    label=int(label),
                    class_name=class_name,
                )
            )

        return MultiTaskPrediction(
            detection=DetectionPrediction(boxes=det_boxes),
            segmentation=SegmentationPrediction(sem_seg=sem_seg, orig_shape=orig_shape),
            raw_output=outputs,
            image_path=image_path,
        )
