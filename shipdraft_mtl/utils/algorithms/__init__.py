"""船舶吃水深度算法模块。"""

from .structure_constrained_vessel_draft_depth_estimation import (
    StructureConstrainedVesselDraftDepthEstimation,
    DepthEstimationResult,
    ScaleReading,
    DetBox,
    GridRow,
    GridPatternMatcher,
    GlobalTemplate,
    NMS_IOU_THRESHOLD,
    quick_estimate_depth,
    batch_estimate_depth,
    extract_waterline,
)
from .baseline_depth_estimation import BaselineDepthEstimator, BaselineScale

__all__ = [
    "StructureConstrainedVesselDraftDepthEstimation",
    "DepthEstimationResult",
    "ScaleReading",
    "DetBox",
    "GridRow",
    "GridPatternMatcher",
    "GlobalTemplate",
    "NMS_IOU_THRESHOLD",
    "quick_estimate_depth",
    "batch_estimate_depth",
    "extract_waterline",
    "BaselineDepthEstimator",
    "BaselineScale",
]
