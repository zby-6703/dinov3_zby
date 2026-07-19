"""
水线检测精度评估模块 (Waterline Metrics)

计算水线检测的 MAVD (Mean Absolute Vertical Distance)

MAVD定义:
    水线是预测水体分割掩码上边缘与真值水体分割掩码上边缘之间的垂直距离（像素）的平均值

输入格式:
    - pred_mask: np.ndarray, shape (H, W), 二值掩码，1表示水体，0表示背景
    - gt_mask: np.ndarray, shape (H, W), 二值掩码，1表示水体，0表示背景

输出格式:
    {
        'MAVD': float,          # 平均绝对垂直距离（像素）
        'MAVD_std': float,      # 标准差
        'MAVD_median': float,   # 中位数
        'valid_samples': int,   # 有效样本数
        'total_samples': int    # 总样本数
    }

使用示例:
    from utils.metrics import WaterlineMetrics
    
    metrics = WaterlineMetrics()
    for pred, gt in zip(pred_masks, gt_masks):
        metrics.add_image(pred, gt, filename="image.jpg")
    results = metrics.compute()
    print(f"MAVD: {results['MAVD']:.2f} pixels")
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Any, Union, Tuple
from dataclasses import dataclass


@dataclass
class WaterlineSample:
    """水线评估样本数据"""
    filename: str
    mavd: float
    valid: bool
    failure_reason: Optional[str] = None


class WaterlineMetrics:
    """
    水线检测精度评估器
    
    计算预测水线与真值水线之间的平均绝对垂直距离(MAVD)
    支持记录失败样本和Top K Worst样本
    
    改进：使用形态学操作和连通区域分析过滤离散噪点，避免因为分割中的
    小噪点导致水线位置错误偏移。
    """
    
    def __init__(self, water_class_id: int = 1, min_contour_area_ratio: float = 0.01,
                 use_morphology: bool = True, morphology_kernel_size: int = 5):
        """
        初始化水线评估器
        
        Args:
            water_class_id: 水体类别ID，默认为1
            min_contour_area_ratio: 最小连通区域面积比例（相对于图像面积），默认0.01(1%)
            use_morphology: 是否使用形态学操作去除噪点，默认True
            morphology_kernel_size: 形态学操作核大小，默认5
        """
        self.water_class_id = water_class_id
        self.min_contour_area_ratio = min_contour_area_ratio
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.samples: List[WaterlineSample] = []
        self.sample_count = 0
        
    def reset(self):
        """重置所有数据"""
        self.samples = []
        self.sample_count = 0
    
    def _clean_mask(self, water_mask: np.ndarray) -> np.ndarray:
        """
        清理水体掩码，去除离散噪点
        
        使用形态学开运算和连通区域分析，只保留主要的水体区域
        
        Args:
            water_mask: (H, W) 二值水体掩码
            
        Returns:
            清理后的水体掩码
        """
        H, W = water_mask.shape
        cleaned_mask = water_mask.copy()
        
        # 步骤1: 形态学开运算去除小噪点
        if self.use_morphology:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, 
                (self.morphology_kernel_size, self.morphology_kernel_size)
            )
            cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # 步骤2: 连通区域分析，只保留足够大的区域
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            # 如果形态学操作后没有轮廓，返回原始掩码
            return water_mask
        
        # 计算最小面积阈值
        total_area = H * W
        min_area = total_area * self.min_contour_area_ratio
        
        # 创建新的掩码，只包含足够大的连通区域
        filtered_mask = np.zeros_like(cleaned_mask)
        valid_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                valid_contours.append(contour)
                cv2.fillPoly(filtered_mask, [contour], 1)
        
        # 如果过滤后没有有效区域，保留最大的连通区域
        if filtered_mask.sum() == 0 and len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.fillPoly(filtered_mask, [largest_contour], 1)
        
        return filtered_mask
    
    def _extract_waterline(self, mask: np.ndarray) -> Tuple[np.ndarray, bool, str]:
        """
        从水体掩码中提取水线（水体上边缘）
        
        改进：先对掩码进行清理，过滤掉离散噪点，避免因为小噪点
        导致水线位置错误偏移。
        
        Args:
            mask: (H, W) 水体掩码
            
        Returns:
            (waterline, valid, reason): 水线数组，是否有效，失败原因
        """
        mask = np.asarray(mask)
        H, W = mask.shape
        
        # 提取水体区域
        water_mask = (mask == self.water_class_id).astype(np.uint8)
        
        # 检查是否有水体
        if water_mask.sum() == 0:
            return np.full(W, -1, dtype=np.int32), False, "no_water_pixels"
        
        # 清理掩码，去除离散噪点
        water_mask_clean = self._clean_mask(water_mask)
        
        # 如果清理后没有水体，使用原始掩码
        if water_mask_clean.sum() == 0:
            water_mask_clean = water_mask
        
        # 找到每列最上方的水体像素
        waterline = np.full(W, -1, dtype=np.int32)
        
        for col in range(W):
            water_rows = np.where(water_mask_clean[:, col] > 0)[0]
            if len(water_rows) > 0:
                waterline[col] = water_rows.min()
        
        return waterline, True, ""
    
    def _compute_mavd_single(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray
    ) -> Tuple[float, bool, str]:
        """
        计算单张图片的MAVD
        
        Args:
            pred_mask: 预测掩码
            gt_mask: 真值掩码
            
        Returns:
            (mavd, valid, reason): MAVD值，是否有效，失败原因
        """
        pred_waterline, pred_valid, pred_reason = self._extract_waterline(pred_mask)
        gt_waterline, gt_valid, gt_reason = self._extract_waterline(gt_mask)
        
        if not pred_valid:
            return -1.0, False, f"pred_{pred_reason}"
        if not gt_valid:
            return -1.0, False, f"gt_{gt_reason}"
        
        # 找到两者都有效的列
        valid_cols = (pred_waterline >= 0) & (gt_waterline >= 0)
        
        if not valid_cols.any():
            return -1.0, False, "no_overlapping_valid_columns"
        
        # 计算垂直距离
        distances = np.abs(pred_waterline[valid_cols] - gt_waterline[valid_cols])
        
        return float(np.mean(distances)), True, ""
    
    def add_image(
        self, 
        pred_mask: np.ndarray, 
        gt_mask: np.ndarray,
        filename: Optional[str] = None
    ):
        """
        添加单张图片的预测和真值
        
        Args:
            pred_mask: (H, W) 预测掩码
            gt_mask: (H, W) 真值掩码
            filename: 文件名（可选，用于记录失败样本）
        """
        self.sample_count += 1
        if filename is None:
            filename = f"sample_{self.sample_count}"
        
        mavd, valid, reason = self._compute_mavd_single(pred_mask, gt_mask)
        
        sample = WaterlineSample(
            filename=filename,
            mavd=mavd if valid else float('inf'),
            valid=valid,
            failure_reason=reason if not valid else None
        )
        self.samples.append(sample)
    
    def add_batch(
        self, 
        pred_masks: np.ndarray, 
        gt_masks: np.ndarray,
        filenames: Optional[List[str]] = None
    ):
        """
        批量添加多张图片
        
        Args:
            pred_masks: (N, H, W) 预测掩码
            gt_masks: (N, H, W) 真值掩码
            filenames: 文件名列表（可选）
        """
        pred_masks = np.asarray(pred_masks)
        gt_masks = np.asarray(gt_masks)
        
        if pred_masks.ndim == 2:
            pred_masks = pred_masks[np.newaxis, ...]
            gt_masks = gt_masks[np.newaxis, ...]
        
        if filenames is None:
            filenames = [None] * len(pred_masks)
        
        for pred, gt, fname in zip(pred_masks, gt_masks, filenames):
            self.add_image(pred, gt, fname)
    
    def get_valid_samples(self) -> List[WaterlineSample]:
        """获取所有有效样本"""
        return [s for s in self.samples if s.valid]
    
    def get_failed_samples(self) -> List[WaterlineSample]:
        """获取所有失败样本"""
        return [s for s in self.samples if not s.valid]
    
    def get_top_k_worst(self, k: int = 5) -> List[WaterlineSample]:
        """获取MAVD最大的Top K样本（表现最差的样本）"""
        valid_samples = self.get_valid_samples()
        sorted_samples = sorted(valid_samples, key=lambda x: x.mavd, reverse=True)
        return sorted_samples[:k]
    
    def compute(self) -> Dict[str, Any]:
        """
        计算MAVD指标
        
        Returns:
            包含 MAVD, MAVD_std, MAVD_median, valid_samples, total_samples 的字典
        """
        valid_samples = self.get_valid_samples()
        mavd_values = [s.mavd for s in valid_samples]
        
        results = {
            'MAVD': 0.0,
            'MAVD_std': 0.0,
            'MAVD_median': 0.0,
            'valid_samples': len(valid_samples),
            'total_samples': self.sample_count
        }
        
        if mavd_values:
            values = np.array(mavd_values)
            results['MAVD'] = float(np.mean(values))
            results['MAVD_std'] = float(np.std(values))
            results['MAVD_median'] = float(np.median(values))
        
        return results
    
    def summary(self, show_top_k: int = 5, show_failed: bool = True) -> str:
        """
        返回评估结果摘要字符串
        
        Args:
            show_top_k: 显示Top K Worst样本数量
            show_failed: 是否显示失败样本
        """
        results = self.compute()
        
        lines = [
            "=" * 60,
            "Waterline Metrics Summary (MAVD)",
            "=" * 60,
            f"MAVD:          {results['MAVD']:.2f} pixels",
            f"MAVD Std:      {results['MAVD_std']:.2f} pixels",
            f"MAVD Median:   {results['MAVD_median']:.2f} pixels",
            f"Valid Samples: {results['valid_samples']}/{results['total_samples']}",
        ]
        
        # 显示失败样本
        failed_samples = self.get_failed_samples()
        if show_failed and failed_samples:
            lines.append("")
            lines.append(f"❌ Failed Samples ({len(failed_samples)}):")
            for sample in failed_samples:
                lines.append(f"  - {sample.filename}: {sample.failure_reason}")
        
        # 显示Top K Worst样本
        if show_top_k > 0:
            top_worst = self.get_top_k_worst(show_top_k)
            if top_worst:
                lines.append("")
                lines.append(f"🔍 Top {len(top_worst)} Worst Performing Samples (Outliers):")
                for i, sample in enumerate(top_worst, 1):
                    lines.append(f"  {i}. File: {sample.filename:<25} MAVD: {sample.mavd:.2f} pixels")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compute_mavd(
    pred_masks: Union[np.ndarray, List[np.ndarray]], 
    gt_masks: Union[np.ndarray, List[np.ndarray]],
    water_class_id: int = 1,
    filenames: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    便捷函数：一次性计算MAVD
    
    Args:
        pred_masks: 预测掩码，(N, H, W) 或列表
        gt_masks: 真值掩码，(N, H, W) 或列表
        water_class_id: 水体类别ID
        filenames: 文件名列表
        
    Returns:
        MAVD评估结果字典
    """
    metrics = WaterlineMetrics(water_class_id=water_class_id)
    
    if isinstance(pred_masks, list):
        if filenames is None:
            filenames = [None] * len(pred_masks)
        for pred, gt, fname in zip(pred_masks, gt_masks, filenames):
            metrics.add_image(pred, gt, fname)
    else:
        metrics.add_batch(pred_masks, gt_masks, filenames)
    
    return metrics.compute()
