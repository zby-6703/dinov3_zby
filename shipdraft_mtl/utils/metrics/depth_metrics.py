"""
船舶吃水深度估计精度评估模块 (Depth Estimation Metrics)

计算 MADDE (Mean Absolute Draft Depth Error)

MADDE定义:
    预测吃水深度值与真实吃水深度值之间的平均绝对误差（米）

输入格式:
    - pred_depths: List[float] 或 np.ndarray, 预测深度值（米）
    - gt_depths: List[float] 或 np.ndarray, 真实深度值（米）

输出格式:
    {
        'MADDE': float,         # 平均绝对深度误差（米）
        'RMSE': float,          # 均方根误差（米）
        'Median': float,        # 中位数误差（米）
        'Std': float,           # 标准差
        'P@0.1': float,         # 误差<0.1m的比例
        'P@0.2': float,         # 误差<0.2m的比例
        'P@0.5': float,         # 误差<0.5m的比例
        'valid_samples': int,   # 有效样本数
        'total_samples': int    # 总样本数
    }

使用示例:
    from utils.metrics import DepthMetrics
    
    metrics = DepthMetrics()
    for pred, gt in zip(pred_depths, gt_depths):
        metrics.add_sample(pred, gt, filename="image.jpg")
    results = metrics.compute()
    print(f"MADDE: {results['MADDE']:.4f} m")
    print(metrics.summary())
"""

import numpy as np
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class DepthSample:
    """深度评估样本数据"""
    filename: str
    pred_depth: Optional[float]
    gt_depth: Optional[float]
    error: float
    valid: bool
    failure_reason: Optional[str] = None


class DepthMetrics:
    """
    吃水深度估计精度评估器
    
    计算预测深度与真实深度之间的误差指标
    支持记录失败样本和Top K Worst样本
    """
    
    def __init__(self, epsilon_list: Optional[List[float]] = None):
        """
        初始化深度评估器
        
        Args:
            epsilon_list: 用于计算P@ε的阈值列表，默认 [0.05, 0.1, 0.2, 0.5]
        """
        self.epsilon_list = epsilon_list or [0.05, 0.1, 0.2, 0.5]
        self.samples: List[DepthSample] = []
        self.sample_count = 0
        
    def reset(self):
        """重置所有数据"""
        self.samples = []
        self.sample_count = 0
    
    def add_sample(
        self, 
        pred_depth: Optional[float], 
        gt_depth: Optional[float],
        filename: Optional[str] = None,
        valid: bool = True,
        failure_reason: Optional[str] = None
    ):
        """
        添加单个样本
        
        Args:
            pred_depth: 预测深度值（米）
            gt_depth: 真实深度值（米）
            filename: 文件名（可选，用于记录失败样本）
            valid: 是否为有效样本
            failure_reason: 失败原因（当valid=False时，可由调用方提供）
        """
        self.sample_count += 1
        if filename is None:
            filename = f"sample_{self.sample_count}"
        
        # 判断有效性和失败原因
        sample_valid = valid
        sample_failure_reason = failure_reason  # 使用调用方提供的原因
        error = float('inf')
        
        if pred_depth is None:
            sample_valid = False
            if sample_failure_reason is None:
                sample_failure_reason = "pred_depth_is_none"
        elif gt_depth is None:
            sample_valid = False
            if sample_failure_reason is None:
                sample_failure_reason = "gt_depth_is_none"
        elif np.isnan(pred_depth):
            sample_valid = False
            if sample_failure_reason is None:
                sample_failure_reason = "pred_depth_is_nan"
        elif np.isnan(gt_depth):
            sample_valid = False
            if sample_failure_reason is None:
                sample_failure_reason = "gt_depth_is_nan"
        elif not valid:
            sample_valid = False
            if sample_failure_reason is None:
                sample_failure_reason = "marked_as_invalid"
        else:
            error = abs(float(pred_depth) - float(gt_depth))
        
        sample = DepthSample(
            filename=filename,
            pred_depth=pred_depth,
            gt_depth=gt_depth,
            error=error,
            valid=sample_valid,
            failure_reason=sample_failure_reason
        )
        self.samples.append(sample)
    
    def add_batch(
        self, 
        pred_depths: Union[List[float], np.ndarray],
        gt_depths: Union[List[float], np.ndarray],
        filenames: Optional[List[str]] = None,
        valid_mask: Optional[Union[List[bool], np.ndarray]] = None
    ):
        """
        批量添加样本
        
        Args:
            pred_depths: 预测深度值列表
            gt_depths: 真实深度值列表
            filenames: 文件名列表
            valid_mask: 有效性掩码，可选
        """
        pred_depths = list(pred_depths) if isinstance(pred_depths, np.ndarray) else pred_depths
        gt_depths = list(gt_depths) if isinstance(gt_depths, np.ndarray) else gt_depths
        
        if filenames is None:
            filenames = [None] * len(pred_depths)
        if valid_mask is None:
            valid_mask = [True] * len(pred_depths)
        else:
            valid_mask = list(valid_mask) if isinstance(valid_mask, np.ndarray) else valid_mask
        
        for pred, gt, fname, valid in zip(pred_depths, gt_depths, filenames, valid_mask):
            self.add_sample(pred, gt, fname, valid)
    
    def get_valid_samples(self) -> List[DepthSample]:
        """获取所有有效样本"""
        return [s for s in self.samples if s.valid]
    
    def get_failed_samples(self) -> List[DepthSample]:
        """获取所有失败样本"""
        return [s for s in self.samples if not s.valid]
    
    def get_top_k_worst(self, k: int = 5) -> List[DepthSample]:
        """获取误差最大的Top K样本（表现最差的样本）"""
        valid_samples = self.get_valid_samples()
        sorted_samples = sorted(valid_samples, key=lambda x: x.error, reverse=True)
        return sorted_samples[:k]
    
    def compute(self) -> Dict[str, Any]:
        """
        计算所有深度估计指标
        
        Returns:
            包含 MADDE, RMSE, Median, Std, P@ε, valid_samples, total_samples 的字典
        """
        valid_samples = self.get_valid_samples()
        
        results = {
            'MADDE': 0.0,
            'RMSE': 0.0,
            'Median': 0.0,
            'Std': 0.0,
            'valid_samples': len(valid_samples),
            'total_samples': self.sample_count
        }
        
        # 添加P@ε
        for eps in self.epsilon_list:
            results[f'P@{eps}'] = 0.0
        
        if not valid_samples:
            return results
        
        errors = np.array([s.error for s in valid_samples])
        
        # MADDE (Mean Absolute Draft Depth Error)
        results['MADDE'] = float(np.mean(errors))
        
        # RMSE
        results['RMSE'] = float(np.sqrt(np.mean(errors ** 2)))
        
        # Median
        results['Median'] = float(np.median(errors))
        
        # Std
        results['Std'] = float(np.std(errors))
        
        # P@ε (误差小于ε的比例)
        for eps in self.epsilon_list:
            results[f'P@{eps}'] = float(np.mean(errors < eps))
        
        return results
    
    def get_errors(self) -> np.ndarray:
        """返回所有有效样本的误差值"""
        valid_samples = self.get_valid_samples()
        if not valid_samples:
            return np.array([])
        return np.array([s.error for s in valid_samples])
    
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
            "Depth Estimation Metrics Summary (MADDE)",
            "=" * 60,
            f"MADDE:         {results['MADDE']:.4f} m",
            f"RMSE:          {results['RMSE']:.4f} m",
            f"Median Error:  {results['Median']:.4f} m",
            f"Std:           {results['Std']:.4f} m",
            "-" * 60,
            "P@ε (Percentage of samples with error < ε):"
        ]
        
        for eps in self.epsilon_list:
            lines.append(f"  P@{eps}m: {results[f'P@{eps}']:.2%}")
        
        lines.append("-" * 60)
        lines.append(f"Valid Samples: {results['valid_samples']}/{results['total_samples']}")
        
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
                    lines.append(f"  {i}. File: {sample.filename:<25} MADDE: {sample.error:.4f} meters")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def compute_madde(
    pred_depths: Union[List[float], np.ndarray],
    gt_depths: Union[List[float], np.ndarray],
    filenames: Optional[List[str]] = None,
    epsilon_list: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    便捷函数：一次性计算MADDE及相关指标
    
    Args:
        pred_depths: 预测深度值列表
        gt_depths: 真实深度值列表
        filenames: 文件名列表
        epsilon_list: P@ε阈值列表
        
    Returns:
        深度估计评估结果字典
    """
    metrics = DepthMetrics(epsilon_list=epsilon_list)
    metrics.add_batch(pred_depths, gt_depths, filenames)
    return metrics.compute()
