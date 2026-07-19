"""
逐样本精度评估记录器 (Per-Sample Evaluation Recorder)
=====================================================

在评估流水线中逐图像记录所有任务的详细数据，评估结束后持久化为 CSV 文件。
不同模型各自生成独立的记录文件，后续绘图脚本统一从这些文件中提取绘图数据。

记录的字段:
    通用: image_id, filename, dataset (pl=自制/clear, VDR=公开/stained)
    检测: n_pred, n_gt, det_matched (TP@0.5)
    分割: iou_per_image, pixel_acc_per_image, water_iou_per_image
    水线: mavd (px), mavd_valid, mavd_failure_reason
    深度(Ours): pred_depth, gt_depth, depth_error, depth_valid, depth_failure_reason
    深度(Baseline): pred_depth_baseline, depth_error_baseline, depth_baseline_valid, depth_baseline_failure_reason

使用方式:
    from utils.metrics import EvaluationRecorder

    recorder = EvaluationRecorder(
        model_name='DraftFormer_v2',
        output_dir='./evaluation_results'
    )

    for image in test_images:
        recorder.add_sample(
            filename='pl_103.jpg',
            pred_boxes=..., gt_boxes=..., pred_labels=..., gt_labels=...,
            pred_mask=..., gt_mask=...,
            pred_depth=..., gt_depth=...,
            pred_depth_baseline=...
        )

    recorder.save()  # → evaluation_results/DraftFormer_v2_per_sample.csv
"""

import csv
import os
import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class SampleRecord:
    """单个样本的完整评估记录"""
    # ---- 通用 ----
    image_id: int = 0
    filename: str = ''
    dataset: str = ''            # 'pl' (自制/clear) 或 'VDR' (公开/stained)
    model_name: str = ''         # 模型名称 (方便多模型合并后绘图区分)

    # ---- 检测 ----
    n_pred: int = 0              # 预测框数量
    n_gt: int = 0                # 真值框数量
    det_matched: int = 0         # TP@IoU=0.5 数量

    # ---- 分割 ----
    iou_per_image: float = 0.0   # 该图的 mIoU
    pixel_acc_per_image: float = 0.0  # 像素准确率
    water_iou_per_image: float = 0.0  # 水体类别 IoU

    # ---- 水线 ----
    mavd: float = 0.0            # MAVD (px)
    mavd_valid: bool = True
    mavd_failure_reason: str = ''

    # ---- 深度 (Ours) ----
    pred_depth: float = 0.0
    gt_depth: float = 0.0
    depth_error: float = 0.0     # |pred - gt|
    depth_valid: bool = True
    depth_failure_reason: str = ''

    # ---- 深度 (Baseline) ----
    pred_depth_baseline: float = 0.0
    depth_error_baseline: float = 0.0
    depth_baseline_valid: bool = True
    depth_baseline_failure_reason: str = ''


# CSV 列顺序 (固定, 确保不同模型的文件可合并)
CSV_COLUMNS = [
    'image_id', 'filename', 'dataset', 'model_name',
    'n_pred', 'n_gt', 'det_matched',
    'iou_per_image', 'pixel_acc_per_image', 'water_iou_per_image',
    'mavd', 'mavd_valid', 'mavd_failure_reason',
    'pred_depth', 'gt_depth', 'depth_error', 'depth_valid', 'depth_failure_reason',
    'pred_depth_baseline', 'depth_error_baseline', 'depth_baseline_valid', 'depth_baseline_failure_reason',
]


# =============================================================================
# 辅助计算函数
# =============================================================================

def _classify_dataset(filename: str) -> str:
    """根据文件名前缀判断数据集来源"""
    basename = os.path.splitext(os.path.basename(filename))[0]
    if basename.startswith('pl_'):
        return 'pl'
    elif basename.startswith('VDR'):
        return 'VDR'
    return 'unknown'


def _compute_per_image_iou(pred_mask: np.ndarray, gt_mask: np.ndarray,
                           num_classes: int = 2) -> Dict[str, float]:
    """
    计算单张图像的逐类 IoU 和像素准确率。

    Args:
        pred_mask: (H, W) 预测掩码
        gt_mask: (H, W) 真值掩码
        num_classes: 类别数

    Returns:
        {'mIoU': float, 'pixel_acc': float, 'water_iou': float}
    """
    pred = np.asarray(pred_mask).flatten()
    gt = np.asarray(gt_mask).flatten()

    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    pred = pred[valid]
    gt = gt[valid]

    if len(gt) == 0:
        return {'mIoU': 0.0, 'pixel_acc': 0.0, 'water_iou': 0.0}

    # 混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    indices = num_classes * gt.astype(np.int64) + pred.astype(np.int64)
    cm_flat = np.bincount(indices, minlength=num_classes ** 2)
    cm = cm_flat.reshape(num_classes, num_classes)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    iou = tp / (tp + fp + fn + 1e-10)
    pixel_acc = float(tp.sum() / (cm.sum() + 1e-10))

    # water_iou: 水体类别(id=1)
    water_iou = float(iou[1]) if num_classes > 1 else 0.0

    # mIoU: 只对有 GT 的类别取均值
    present = cm.sum(axis=1) > 0
    miou = float(np.nanmean(iou[present])) if present.any() else 0.0

    return {'mIoU': miou, 'pixel_acc': pixel_acc, 'water_iou': water_iou}


def _compute_det_matched(pred_boxes: np.ndarray, pred_labels: np.ndarray,
                         gt_boxes: np.ndarray, gt_labels: np.ndarray,
                         iou_thresh: float = 0.5) -> int:
    """
    简易匹配: 计算 TP@IoU 数量 (贪心匹配, 不区分类别内/外).

    Args:
        pred_boxes: (N,4) [x1,y1,x2,y2]
        pred_labels: (N,)
        gt_boxes: (M,4)
        gt_labels: (M,)
        iou_thresh: IoU 阈值

    Returns:
        matched: TP 数量
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0

    pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)
    gt_boxes = np.asarray(gt_boxes).reshape(-1, 4)
    pred_labels = np.asarray(pred_labels).flatten()
    gt_labels = np.asarray(gt_labels).flatten()

    matched = 0
    gt_matched = set()

    for pi in range(len(pred_boxes)):
        best_iou = 0.0
        best_gi = -1
        for gi in range(len(gt_boxes)):
            if gi in gt_matched:
                continue
            if pred_labels[pi] != gt_labels[gi]:
                continue
            iou = _box_iou(pred_boxes[pi], gt_boxes[gi])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matched += 1
            gt_matched.add(best_gi)

    return matched


def _box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个框的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-10)


# =============================================================================
# 核心类
# =============================================================================

class EvaluationRecorder:
    """
    逐样本精度评估记录器

    在评估循环中逐图像记录所有任务的详细数据, 评估完成后保存为 CSV 文件。
    不同模型各自生成独立的记录文件, 后续绘图脚本可按需提取数据。

    特性:
    - 记录每个样本的检测 / 分割 / 水线 / 深度等所有任务指标
    - 自动根据文件名前缀 (pl_ / VDR_) 标注 dataset 字段
    - 保存为标准 CSV, 列顺序固定, 方便多文件合并
    - 向后兼容: 不影响现有 UnifiedEvaluator 的调用接口
    """

    def __init__(
        self,
        model_name: str = 'model',
        output_dir: str = '.',
        enabled: bool = True,
        num_seg_classes: int = 2,
    ):
        """
        初始化记录器

        Args:
            model_name: 模型名称, 用于输出文件命名
            output_dir: 输出目录
            enabled: 是否启用记录 (False 则所有操作为空)
            num_seg_classes: 分割类别数 (用于逐图 IoU计算)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.enabled = enabled
        self.num_seg_classes = num_seg_classes
        self.records: List[SampleRecord] = []

        if enabled:
            os.makedirs(output_dir, exist_ok=True)

    def reset(self):
        """清空所有记录"""
        self.records = []

    # -----------------------------------------------------------------
    # 逐样本添加
    # -----------------------------------------------------------------

    def add_sample(
        self,
        image_id: int = 0,
        filename: str = '',
        # 检测
        pred_boxes: Optional[np.ndarray] = None,
        pred_scores: Optional[np.ndarray] = None,
        pred_labels: Optional[np.ndarray] = None,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
        # 分割
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        # 水线 (已由 WaterlineMetrics 计算)
        mavd: Optional[float] = None,
        mavd_valid: bool = True,
        mavd_failure_reason: str = '',
        # 深度 (Ours)
        pred_depth: Optional[float] = None,
        gt_depth: Optional[float] = None,
        depth_valid: bool = True,
        depth_failure_reason: str = '',
        # 深度 (Baseline)
        pred_depth_baseline: Optional[float] = None,
        depth_baseline_valid: bool = True,
        depth_baseline_failure_reason: str = '',
    ):
        """
        记录单个样本的评估数据

        Note:
            - 检测: 如果传入 pred_boxes / gt_boxes, 自动计算匹配数
            - 分割: 如果传入 pred_mask / gt_mask, 自动计算逐图 IoU
            - 水线: 需要外部传入已计算的 mavd 值 (因为水线计算逻辑在 WaterlineMetrics 中)
            - 深度: 需要外部传入已计算的 pred_depth 和 gt_depth
        """
        if not self.enabled:
            return

        rec = SampleRecord(
            image_id=image_id,
            filename=filename,
            dataset=_classify_dataset(filename),
            model_name=self.model_name,
        )

        # ---- 检测 ----
        if pred_boxes is not None and gt_boxes is not None:
            n_pred = len(pred_boxes) if hasattr(pred_boxes, '__len__') else 0
            n_gt = len(gt_boxes) if hasattr(gt_boxes, '__len__') else 0
            pred_boxes_arr = np.asarray(pred_boxes).reshape(-1, 4) if n_pred > 0 else np.zeros((0, 4))
            gt_boxes_arr = np.asarray(gt_boxes).reshape(-1, 4) if n_gt > 0 else np.zeros((0, 4))
            pred_labels_arr = np.asarray(pred_labels).flatten() if pred_labels is not None else np.zeros(n_pred, dtype=int)
            gt_labels_arr = np.asarray(gt_labels).flatten() if gt_labels is not None else np.zeros(n_gt, dtype=int)

            rec.n_pred = n_pred
            rec.n_gt = n_gt
            rec.det_matched = _compute_det_matched(
                pred_boxes_arr, pred_labels_arr,
                gt_boxes_arr, gt_labels_arr,
                iou_thresh=0.5
            )

        # ---- 分割 ----
        if pred_mask is not None and gt_mask is not None:
            seg_result = _compute_per_image_iou(
                pred_mask, gt_mask, num_classes=self.num_seg_classes
            )
            rec.iou_per_image = seg_result['mIoU']
            rec.pixel_acc_per_image = seg_result['pixel_acc']
            rec.water_iou_per_image = seg_result['water_iou']

        # ---- 水线 ----
        if mavd is not None:
            rec.mavd = mavd
            rec.mavd_valid = mavd_valid
            rec.mavd_failure_reason = mavd_failure_reason or ''

        # ---- 深度 (Ours) ----
        if gt_depth is not None:
            rec.gt_depth = float(gt_depth)
        if pred_depth is not None:
            rec.pred_depth = float(pred_depth)
        rec.depth_valid = depth_valid
        rec.depth_failure_reason = depth_failure_reason or ''
        if depth_valid and pred_depth is not None and gt_depth is not None:
            rec.depth_error = abs(float(pred_depth) - float(gt_depth))
        else:
            rec.depth_error = float('inf')

        # ---- 深度 (Baseline) ----
        if pred_depth_baseline is not None:
            rec.pred_depth_baseline = float(pred_depth_baseline)
        rec.depth_baseline_valid = depth_baseline_valid
        rec.depth_baseline_failure_reason = depth_baseline_failure_reason or ''
        if depth_baseline_valid and pred_depth_baseline is not None and gt_depth is not None:
            rec.depth_error_baseline = abs(float(pred_depth_baseline) - float(gt_depth))
        else:
            rec.depth_error_baseline = float('inf')

        self.records.append(rec)

    # -----------------------------------------------------------------
    # 保存
    # -----------------------------------------------------------------

    def save(self, filename: Optional[str] = None) -> Optional[str]:
        """
        保存所有记录到 CSV 文件

        Args:
            filename: 自定义文件名 (不含目录), None 则自动生成

        Returns:
            保存的文件路径, 或 None (已禁用)
        """
        if not self.enabled or not self.records:
            return None

        if filename is None:
            filename = f'{self.model_name}_per_sample.csv'

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for rec in self.records:
                row = {}
                for col in CSV_COLUMNS:
                    val = getattr(rec, col, '')
                    # bool → str
                    if isinstance(val, bool):
                        val = str(val)
                    # inf → 'inf'
                    elif isinstance(val, float) and not np.isfinite(val):
                        val = 'inf'
                    row[col] = val
                writer.writerow(row)

        print(f"[EvaluationRecorder] 逐样本记录已保存: {filepath} ({len(self.records)} 条记录)")
        return filepath

    # -----------------------------------------------------------------
    # 加载 (静态方法, 供绘图脚本使用)
    # -----------------------------------------------------------------

    @staticmethod
    def load(filepath: str) -> List[SampleRecord]:
        """
        从 CSV 文件加载逐样本记录

        Args:
            filepath: CSV 文件路径

        Returns:
            SampleRecord 列表
        """
        records = []
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rec = SampleRecord()
                rec.image_id = int(row.get('image_id', 0))
                rec.filename = row.get('filename', '')
                rec.dataset = row.get('dataset', '')
                rec.model_name = row.get('model_name', '')

                rec.n_pred = int(row.get('n_pred', 0))
                rec.n_gt = int(row.get('n_gt', 0))
                rec.det_matched = int(row.get('det_matched', 0))

                rec.iou_per_image = float(row.get('iou_per_image', 0))
                rec.pixel_acc_per_image = float(row.get('pixel_acc_per_image', 0))
                rec.water_iou_per_image = float(row.get('water_iou_per_image', 0))

                rec.mavd = float(row.get('mavd', 0))
                rec.mavd_valid = row.get('mavd_valid', 'True') == 'True'
                rec.mavd_failure_reason = row.get('mavd_failure_reason', '')

                rec.pred_depth = float(row.get('pred_depth', 0))
                rec.gt_depth = float(row.get('gt_depth', 0))
                err_str = row.get('depth_error', '0')
                rec.depth_error = float('inf') if err_str == 'inf' else float(err_str)
                rec.depth_valid = row.get('depth_valid', 'True') == 'True'
                rec.depth_failure_reason = row.get('depth_failure_reason', '')

                rec.pred_depth_baseline = float(row.get('pred_depth_baseline', 0))
                err_str_b = row.get('depth_error_baseline', '0')
                rec.depth_error_baseline = float('inf') if err_str_b == 'inf' else float(err_str_b)
                rec.depth_baseline_valid = row.get('depth_baseline_valid', 'True') == 'True'
                rec.depth_baseline_failure_reason = row.get('depth_baseline_failure_reason', '')

                records.append(rec)
        return records

    @staticmethod
    def load_multiple(filepaths: List[str]) -> Dict[str, List[SampleRecord]]:
        """
        加载多个模型的记录文件

        Args:
            filepaths: CSV 文件路径列表

        Returns:
            {模型名 → SampleRecord 列表}
        """
        result = {}
        for fp in filepaths:
            records = EvaluationRecorder.load(fp)
            # 优先使用 CSV 内的 model_name 字段; 若为空则从文件名推断
            if records and records[0].model_name:
                model_name = records[0].model_name
            else:
                basename = os.path.splitext(os.path.basename(fp))[0]
                model_name = basename.replace('_per_sample', '')
            result[model_name] = records
        return result

    # -----------------------------------------------------------------
    # 统计概览
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """返回记录概览"""
        if not self.records:
            return "[EvaluationRecorder] 无记录"

        n = len(self.records)
        n_pl = sum(1 for r in self.records if r.dataset == 'pl')
        n_vdr = sum(1 for r in self.records if r.dataset == 'VDR')
        n_depth_valid = sum(1 for r in self.records if r.depth_valid)
        n_baseline_valid = sum(1 for r in self.records if r.depth_baseline_valid)

        lines = [
            f"[EvaluationRecorder] Model: {self.model_name}",
            f"  Total Records:   {n}",
            f"  pl (clear):      {n_pl}",
            f"  VDR (stained):   {n_vdr}",
            f"  Depth valid (Ours):     {n_depth_valid}/{n}",
            f"  Depth valid (Baseline): {n_baseline_valid}/{n}",
        ]
        return "\n".join(lines)
