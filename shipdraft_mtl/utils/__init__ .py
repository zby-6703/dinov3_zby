"""utils 统一入口。

主结构：
- algorithms: 深度估计算法
- metrics: 精度评估指标
"""

from .algorithms import (
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
    BaselineDepthEstimator,
    BaselineScale,
)
from .metrics import (
    DetectionMetrics,
    SegmentationMetrics,
    WaterlineMetrics,
    WaterlineSample,
    DepthMetrics,
    DepthSample,
    UnifiedEvaluator,
    EvaluationRecorder,
    SampleRecord,
    PerformanceMetrics,
    ParameterStats,
    FLOPsStats,
    FPSStats,
    count_model_parameters,
    calculate_model_flops,
    measure_model_fps,
)

__all__ = [
    "DetectionMetrics",
    "SegmentationMetrics",
    "WaterlineMetrics",
    "WaterlineSample",
    "DepthMetrics",
    "DepthSample",
    "UnifiedEvaluator",
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
    "EvaluationRecorder",
    "SampleRecord",
    "PerformanceMetrics",
    "ParameterStats",
    "FLOPsStats",
    "FPSStats",
    "count_model_parameters",
    "calculate_model_flops",
    "measure_model_fps",
]
