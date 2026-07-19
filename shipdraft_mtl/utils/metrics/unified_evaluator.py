"""
统一评估器 (Unified Evaluator)

提供一个统一的接口，同时评估多个任务的精度指标

支持的任务组合:
- 多任务模型 (YOLOP, HybridNet): detection + segmentation + waterline + depth
- 分割模型 (Mask2Former, MaskDINO, UNet): segmentation + waterline
- 检测模型 (DINO, Faster R-CNN): detection

使用示例:
    from utils.metrics import UnifiedEvaluator
    
    # 创建统一评估器
    evaluator = UnifiedEvaluator(
        tasks=['detection', 'segmentation', 'waterline', 'depth'],
        detection_config={'num_classes': 8, 'class_names': ['0','1','2','3','4','6','8','M']},
        segmentation_config={'num_classes': 2, 'class_names': ['background', 'water']}
    )
    
    # 添加数据
    evaluator.add_detection(image_id, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    evaluator.add_segmentation(pred_mask, gt_mask)
    evaluator.add_depth(pred_depth, gt_depth)
    
    # 计算所有指标
    results = evaluator.compute()
    print(evaluator.summary())
    
    # 保存结果
    evaluator.save_results('results.json')
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Union

from .detection_metrics import DetectionMetrics
from .segmentation_metrics import SegmentationMetrics
from .waterline_metrics import WaterlineMetrics
from .depth_metrics import DepthMetrics


class UnifiedEvaluator:
    """
    统一评估器
    
    可以根据需要选择评估的任务，支持:
    - detection: 目标检测 (mAP)
    - segmentation: 语义分割 (mIoU)
    - waterline: 水线检测 (MAVD)
    - depth: 深度估计 (MADDE) - 支持双算法对比评估
    """
    
    SUPPORTED_TASKS = ['detection', 'segmentation', 'waterline', 'depth', 'depth_baseline']
    
    def __init__(
        self,
        tasks: List[str],
        detection_config: Optional[Dict] = None,
        segmentation_config: Optional[Dict] = None,
        waterline_config: Optional[Dict] = None,
        depth_config: Optional[Dict] = None
    ):
        """
        初始化统一评估器
        
        Args:
            tasks: 需要评估的任务列表
            detection_config: 检测任务配置 {'num_classes': int, 'class_names': List[str]}
            segmentation_config: 分割任务配置 {'num_classes': int, 'class_names': List[str]}
            waterline_config: 水线任务配置 {'water_class_id': int}
            depth_config: 深度任务配置 {'epsilon_list': List[float]}
        """
        # 验证任务
        for task in tasks:
            if task not in self.SUPPORTED_TASKS:
                raise ValueError(f"Unsupported task: {task}. Supported: {self.SUPPORTED_TASKS}")
        
        self.tasks = tasks
        self.evaluators = {}
        
        # 初始化各任务评估器
        if 'detection' in tasks:
            config = detection_config or {}
            self.evaluators['detection'] = DetectionMetrics(
                num_classes=config.get('num_classes', 8),
                class_names=config.get('class_names', None)
            )
        
        if 'segmentation' in tasks:
            config = segmentation_config or {}
            self.evaluators['segmentation'] = SegmentationMetrics(
                num_classes=config.get('num_classes', 2),
                class_names=config.get('class_names', None),
                ignore_index=config.get('ignore_index', -1)
            )
        
        if 'waterline' in tasks:
            config = waterline_config or {}
            self.evaluators['waterline'] = WaterlineMetrics(
                water_class_id=config.get('water_class_id', 1)
            )
        
        if 'depth' in tasks:
            config = depth_config or {}
            self.evaluators['depth'] = DepthMetrics(
                epsilon_list=config.get('epsilon_list', None)
            )
        
        # Baseline深度估计（论文复现对比算法）
        if 'depth_baseline' in tasks:
            config = depth_config or {}
            self.evaluators['depth_baseline'] = DepthMetrics(
                epsilon_list=config.get('epsilon_list', None)
            )
        
        self.sample_count = 0
    
    def reset(self):
        """重置所有评估器"""
        for evaluator in self.evaluators.values():
            evaluator.reset()
        self.sample_count = 0
    
    # ==================== 添加数据接口 ====================
    
    def add_detection(
        self,
        image_id: Union[int, str],
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        """添加检测数据"""
        if 'detection' not in self.evaluators:
            return
        self.evaluators['detection'].add_image(
            image_id, pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
        )
    
    def add_segmentation(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """添加分割数据"""
        if 'segmentation' not in self.evaluators:
            return
        self.evaluators['segmentation'].add_image(pred_mask, gt_mask)
        self.sample_count += 1
    
    def add_waterline(self, pred_mask: np.ndarray, gt_mask: np.ndarray, 
                      filename: Optional[str] = None):
        """添加水线数据（使用分割掩码）
        
        Args:
            pred_mask: 预测分割掩码
            gt_mask: 真实分割掩码
            filename: 图像文件名（用于失败样本追踪和Top K Worst）
        """
        if 'waterline' not in self.evaluators:
            return
        self.evaluators['waterline'].add_image(pred_mask, gt_mask, filename=filename)
    
    def add_depth(self, pred_depth: float, gt_depth: float, valid: bool = True,
                  filename: Optional[str] = None, failure_reason: Optional[str] = None):
        """添加深度数据 (Ours: Global Grid Matching)
        
        Args:
            pred_depth: 预测深度值
            gt_depth: 真实深度值
            valid: 是否为有效样本
            filename: 图像文件名（用于失败样本追踪和Top K Worst）
            failure_reason: 失败原因（当valid=False时）
        """
        if 'depth' not in self.evaluators:
            return
        self.evaluators['depth'].add_sample(pred_depth, gt_depth, 
                                            filename=filename, valid=valid,
                                            failure_reason=failure_reason)
    
    def add_depth_baseline(self, pred_depth: float, gt_depth: float, valid: bool = True,
                          filename: Optional[str] = None, failure_reason: Optional[str] = None):
        """添加Baseline深度数据 (MTL-VDR论文算法)
        
        Args:
            pred_depth: 预测深度值
            gt_depth: 真实深度值
            valid: 是否为有效样本
            filename: 图像文件名（用于失败样本追踪和Top K Worst）
            failure_reason: 失败原因（当valid=False时）
        """
        if 'depth_baseline' not in self.evaluators:
            return
        self.evaluators['depth_baseline'].add_sample(pred_depth, gt_depth, 
                                                     filename=filename, valid=valid,
                                                     failure_reason=failure_reason)
    
    def add_sample(
        self,
        image_id: Optional[Union[int, str]] = None,
        filename: Optional[str] = None,  # 新增：用于追踪失败样本
        # 检测数据
        pred_boxes: Optional[np.ndarray] = None,
        pred_scores: Optional[np.ndarray] = None,
        pred_labels: Optional[np.ndarray] = None,
        gt_boxes: Optional[np.ndarray] = None,
        gt_labels: Optional[np.ndarray] = None,
        # 分割数据
        pred_mask: Optional[np.ndarray] = None,
        gt_mask: Optional[np.ndarray] = None,
        # 深度数据 (Ours)
        pred_depth: Optional[float] = None,
        gt_depth: Optional[float] = None,
        depth_valid: bool = True,
        depth_failure_reason: Optional[str] = None,
        # 深度数据 (Baseline)
        pred_depth_baseline: Optional[float] = None,
        depth_baseline_valid: bool = True,
        depth_baseline_failure_reason: Optional[str] = None
    ):
        """
        添加一个完整样本的所有数据
        
        可以根据提供的数据自动判断要更新哪些评估器
        
        Args:
            image_id: 图像ID
            filename: 文件名（用于失败样本追踪和Top K Worst）
            pred_depth: 我们算法的预测深度
            pred_depth_baseline: Baseline算法的预测深度
            其他参数同各add_xxx方法
        """
        # 检测
        if all(v is not None for v in [pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels]):
            self.add_detection(image_id or self.sample_count, 
                             pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
        
        # 分割
        if pred_mask is not None and gt_mask is not None:
            self.add_segmentation(pred_mask, gt_mask)
            # 水线复用分割掩码
            self.add_waterline(pred_mask, gt_mask, filename=filename)
        
        # 深度 (Ours)
        if pred_depth is not None and gt_depth is not None:
            self.add_depth(pred_depth, gt_depth, valid=depth_valid, 
                          filename=filename, failure_reason=depth_failure_reason)
        
        # 深度 (Baseline)
        if pred_depth_baseline is not None and gt_depth is not None:
            self.add_depth_baseline(pred_depth_baseline, gt_depth, valid=depth_baseline_valid,
                                   filename=filename, failure_reason=depth_baseline_failure_reason)
    
    # ==================== 计算指标 ====================
    
    def compute(self) -> Dict[str, Any]:
        """
        计算所有任务的指标
        
        Returns:
            {
                'sample_count': int,
                'timestamp': str,
                'tasks': {
                    'detection': {...},
                    'segmentation': {...},
                    'waterline': {...},
                    'depth': {...}
                }
            }
        """
        results = {
            'sample_count': self.sample_count,
            'timestamp': datetime.now().isoformat(),
            'tasks': {}
        }
        
        for task_name, evaluator in self.evaluators.items():
            results['tasks'][task_name] = evaluator.compute()
        
        return results
    
    def compute_single_task(self, task: str) -> Dict[str, Any]:
        """计算单个任务的指标"""
        if task not in self.evaluators:
            raise ValueError(f"Task '{task}' not configured")
        return self.evaluators[task].compute()
    
    # ==================== 输出结果 ====================
    
    def summary(self, show_top_k: int = 5, show_failed: bool = True) -> str:
        """
        返回评估结果摘要字符串
        
        Args:
            show_top_k: 显示Top K个表现最差的样本，0表示不显示
            show_failed: 是否显示失败样本列表
        """
        results = self.compute()
        
        lines = [
            "=" * 60,
            "Unified Evaluation Results",
            "=" * 60,
            f"Sample Count: {results['sample_count']}",
            f"Timestamp: {results['timestamp']}",
            ""
        ]
        
        # 检测结果
        if 'detection' in results['tasks']:
            det = results['tasks']['detection']
            lines.extend([
                "【Detection】",
                "-" * 40,
                f"  mAP@0.5:      {det['mAP@0.5']:.4f}",
                f"  mAP@0.75:     {det['mAP@0.75']:.4f}",
                f"  mAP@0.5:0.95: {det['mAP@0.5:0.95']:.4f}",
                f"  Precision:    {det['precision']:.4f}",
                f"  Recall:       {det['recall']:.4f}",
                ""
            ])
        
        # 分割结果
        if 'segmentation' in results['tasks']:
            seg = results['tasks']['segmentation']
            lines.extend([
                "【Segmentation】",
                "-" * 40,
                f"  mIoU:           {seg['mIoU']:.4f}",
                f"  Pixel Accuracy: {seg['pixel_accuracy']:.4f}",
                f"  FWIoU:          {seg['FWIoU']:.4f}",
                ""
            ])
        
        # 水线结果
        if 'waterline' in results['tasks']:
            wl = results['tasks']['waterline']
            lines.extend([
                "【Waterline】",
                "-" * 40,
                f"  MAVD:         {wl['MAVD']:.2f} pixels",
                f"  MAVD Median:  {wl['MAVD_median']:.2f} pixels",
                f"  Valid:        {wl['valid_samples']}/{wl['total_samples']}",
            ])
            # Top K Worst
            if show_top_k > 0 and 'waterline' in self.evaluators:
                top_worst = self.evaluators['waterline'].get_top_k_worst(show_top_k)
                if top_worst:
                    lines.append(f"  Top {len(top_worst)} Worst:")
                    for sample in top_worst:
                        lines.append(f"    - {sample.filename}: MAVD={sample.mavd:.2f}px")
            # Failed samples
            if show_failed and 'waterline' in self.evaluators:
                failed = self.evaluators['waterline'].get_failed_samples()
                if failed:
                    lines.append(f"  Failed Samples ({len(failed)}):")
                    for sample in failed[:5]:  # 最多显示5个
                        lines.append(f"    - {sample.filename}: {sample.failure_reason}")
                    if len(failed) > 5:
                        lines.append(f"    ... and {len(failed)-5} more")
            lines.append("")
        
        # 深度结果 (Ours: Global Grid Matching)
        if 'depth' in results['tasks']:
            depth = results['tasks']['depth']
            lines.extend([
                "【Depth Estimation (Ours)】",
                "-" * 40,
                f"  MADDE:        {depth['MADDE']:.4f} m",
                f"  RMSE:         {depth['RMSE']:.4f} m",
                f"  Median:       {depth['Median']:.4f} m",
            ])
            for key, value in depth.items():
                if key.startswith('P@'):
                    lines.append(f"  {key}:        {value:.2%}")
            lines.append(f"  Valid:        {depth['valid_samples']}/{depth['total_samples']}")
            # Top K Worst
            if show_top_k > 0 and 'depth' in self.evaluators:
                top_worst = self.evaluators['depth'].get_top_k_worst(show_top_k)
                if top_worst:
                    lines.append(f"  Top {len(top_worst)} Worst:")
                    for sample in top_worst:
                        lines.append(f"    - {sample.filename}: pred={sample.pred_depth:.2f}m, gt={sample.gt_depth:.2f}m, err={sample.error:.4f}m")
            # Failed samples
            if show_failed and 'depth' in self.evaluators:
                failed = self.evaluators['depth'].get_failed_samples()
                if failed:
                    lines.append(f"  Failed Samples ({len(failed)}):")
                    for sample in failed[:5]:  # 最多显示5个
                        lines.append(f"    - {sample.filename}: {sample.failure_reason}")
                    if len(failed) > 5:
                        lines.append(f"    ... and {len(failed)-5} more")
            lines.append("")
        
        # 深度结果 (Baseline: MTL-VDR论文算法)
        if 'depth_baseline' in results['tasks']:
            depth = results['tasks']['depth_baseline']
            lines.extend([
                "【Depth Estimation (Baseline: MTL-VDR)】",
                "-" * 40,
                f"  MADDE:        {depth['MADDE']:.4f} m",
                f"  RMSE:         {depth['RMSE']:.4f} m",
                f"  Median:       {depth['Median']:.4f} m",
            ])
            for key, value in depth.items():
                if key.startswith('P@'):
                    lines.append(f"  {key}:        {value:.2%}")
            lines.append(f"  Valid:        {depth['valid_samples']}/{depth['total_samples']}")
            # Top K Worst
            if show_top_k > 0 and 'depth_baseline' in self.evaluators:
                top_worst = self.evaluators['depth_baseline'].get_top_k_worst(show_top_k)
                if top_worst:
                    lines.append(f"  Top {len(top_worst)} Worst:")
                    for sample in top_worst:
                        lines.append(f"    - {sample.filename}: pred={sample.pred_depth:.2f}m, gt={sample.gt_depth:.2f}m, err={sample.error:.4f}m")
            # Failed samples
            if show_failed and 'depth_baseline' in self.evaluators:
                failed = self.evaluators['depth_baseline'].get_failed_samples()
                if failed:
                    lines.append(f"  Failed Samples ({len(failed)}):")
                    for sample in failed[:5]:  # 最多显示5个
                        lines.append(f"    - {sample.filename}: {sample.failure_reason}")
                    if len(failed) > 5:
                        lines.append(f"    ... and {len(failed)-5} more")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str, format: str = 'json'):
        """
        保存评估结果
        
        Args:
            filepath: 保存路径
            format: 保存格式 ('json' 或 'txt')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            results = self.compute()
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        elif format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.summary())
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to: {filepath}")
    
    # ==================== 便捷方法 ====================
    
    def get_primary_metrics(self) -> Dict[str, float]:
        """
        获取主要指标（用于快速对比）
        
        Returns:
            {
                'mAP@0.5': float,
                'mAP@0.5:0.95': float,
                'mIoU': float,
                'MAVD': float,
                'MADDE': float,
                'MADDE_baseline': float  # Baseline算法
            }
        """
        results = self.compute()
        primary = {}
        
        if 'detection' in results['tasks']:
            primary['mAP@0.5'] = results['tasks']['detection']['mAP@0.5']
            primary['mAP@0.5:0.95'] = results['tasks']['detection']['mAP@0.5:0.95']
        
        if 'segmentation' in results['tasks']:
            primary['mIoU'] = results['tasks']['segmentation']['mIoU']
        
        if 'waterline' in results['tasks']:
            primary['MAVD'] = results['tasks']['waterline']['MAVD']
        
        if 'depth' in results['tasks']:
            primary['MADDE'] = results['tasks']['depth']['MADDE']
        
        if 'depth_baseline' in results['tasks']:
            primary['MADDE_baseline'] = results['tasks']['depth_baseline']['MADDE']
        
        return primary
