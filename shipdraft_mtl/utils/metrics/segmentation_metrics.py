"""
分割任务精度评估模块 (Segmentation Metrics)

计算语义分割的 mIoU (Mean Intersection over Union)

输入格式:
    - pred_mask: np.ndarray, shape (H, W) 或 (N, H, W), 值为类别索引
    - gt_mask: np.ndarray, shape (H, W) 或 (N, H, W), 值为类别索引

输出格式:
    {
        'mIoU': float,
        'pixel_accuracy': float,
        'mean_accuracy': float,
        'FWIoU': float,
        'per_class': {
            'class_name': {'IoU': float, 'accuracy': float}
        }
    }

使用示例:
    from utils.metrics import SegmentationMetrics
    
    # 方式1: 逐图添加
    metrics = SegmentationMetrics(num_classes=2, class_names=['background', 'water'])
    for pred, gt in zip(predictions, ground_truths):
        metrics.add_image(pred, gt)
    results = metrics.compute()
    
    # 方式2: 批量添加
    metrics = SegmentationMetrics(num_classes=2)
    metrics.add_batch(pred_masks, gt_masks)  # shape: (N, H, W)
    results = metrics.compute()
"""

import numpy as np
from typing import List, Dict, Optional, Any


class SegmentationMetrics:
    """
    语义分割精度评估器
    
    支持计算:
    - mIoU: 平均交并比
    - Pixel Accuracy: 像素准确率
    - Mean Accuracy: 类别平均准确率
    - FWIoU: 频率加权IoU
    """
    
    def __init__(
        self, 
        num_classes: int,
        class_names: Optional[List[str]] = None,
        ignore_index: int = -1
    ):
        """
        初始化分割评估器
        
        Args:
            num_classes: 类别数量
            class_names: 类别名称列表，可选
            ignore_index: 忽略的类别索引，默认-1（不忽略）
        """
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.ignore_index = ignore_index
        
        # 混淆矩阵
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.sample_count = 0
        
    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.sample_count = 0
        
    def add_image(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """
        添加单张图片的预测和真值
        
        Args:
            pred_mask: (H, W) 预测掩码，值为类别索引
            gt_mask: (H, W) 真值掩码，值为类别索引
        """
        pred_mask = np.asarray(pred_mask)
        gt_mask = np.asarray(gt_mask)
        
        assert pred_mask.shape == gt_mask.shape, \
            f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}"
        
        # 处理忽略索引
        if self.ignore_index >= 0:
            valid_mask = gt_mask != self.ignore_index
            pred_mask = pred_mask[valid_mask]
            gt_mask = gt_mask[valid_mask]
        else:
            pred_mask = pred_mask.flatten()
            gt_mask = gt_mask.flatten()
        
        # 更新混淆矩阵
        mask = (gt_mask >= 0) & (gt_mask < self.num_classes) & \
               (pred_mask >= 0) & (pred_mask < self.num_classes)
        
        indices = self.num_classes * gt_mask[mask].astype(np.int64) + pred_mask[mask].astype(np.int64)
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)
        self.sample_count += 1
    
    def add_batch(self, pred_masks: np.ndarray, gt_masks: np.ndarray):
        """
        批量添加多张图片的预测和真值
        
        Args:
            pred_masks: (N, H, W) 预测掩码
            gt_masks: (N, H, W) 真值掩码
        """
        pred_masks = np.asarray(pred_masks)
        gt_masks = np.asarray(gt_masks)
        
        if pred_masks.ndim == 2:
            pred_masks = pred_masks[np.newaxis, ...]
            gt_masks = gt_masks[np.newaxis, ...]
        
        for pred, gt in zip(pred_masks, gt_masks):
            self.add_image(pred, gt)
    
    def compute(self) -> Dict[str, Any]:
        """
        计算所有分割指标
        
        Returns:
            包含 mIoU, pixel_accuracy, mean_accuracy, FWIoU, per_class 的字典
        """
        cm = self.confusion_matrix
        
        # 各类别的TP, FP, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        
        # 各类别IoU
        iou = tp / (tp + fp + fn + 1e-10)
        
        # 各类别准确率
        class_accuracy = tp / (cm.sum(axis=1) + 1e-10)
        
        # 像素准确率
        pixel_accuracy = tp.sum() / (cm.sum() + 1e-10)
        
        # 平均类别准确率
        mean_accuracy = np.nanmean(class_accuracy)
        
        # mIoU
        valid_classes = cm.sum(axis=1) > 0
        miou = np.nanmean(iou[valid_classes]) if valid_classes.any() else 0.0
        
        # FWIoU (频率加权IoU)
        freq = cm.sum(axis=1) / (cm.sum() + 1e-10)
        fwiou = np.sum(freq[valid_classes] * iou[valid_classes]) if valid_classes.any() else 0.0
        
        # 构建结果
        results = {
            'mIoU': float(miou),
            'pixel_accuracy': float(pixel_accuracy),
            'mean_accuracy': float(mean_accuracy),
            'FWIoU': float(fwiou),
            'per_class': {}
        }
        
        for i in range(self.num_classes):
            class_name = self.class_names[i] if i < len(self.class_names) else str(i)
            results['per_class'][class_name] = {
                'IoU': float(iou[i]),
                'accuracy': float(class_accuracy[i])
            }
        
        return results
    
    def get_confusion_matrix(self) -> np.ndarray:
        """返回混淆矩阵"""
        return self.confusion_matrix.copy()
    
    def summary(self) -> str:
        """返回评估结果摘要字符串"""
        results = self.compute()
        
        lines = [
            "=" * 50,
            "Segmentation Metrics Summary",
            "=" * 50,
            f"mIoU:           {results['mIoU']:.4f}",
            f"Pixel Accuracy: {results['pixel_accuracy']:.4f}",
            f"Mean Accuracy:  {results['mean_accuracy']:.4f}",
            f"FWIoU:          {results['FWIoU']:.4f}",
            "-" * 50,
            "Per-class IoU:"
        ]
        
        for class_name, class_metrics in results['per_class'].items():
            lines.append(f"  {class_name}: {class_metrics['IoU']:.4f}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
