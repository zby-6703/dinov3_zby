# ------------------------------------------------------------------------
# Dynamic Task Balancer for Multi-task Learning
# Copyright (c) 2025. All Rights Reserved.
# ------------------------------------------------------------------------
"""
动态任务平衡器，用于自动平衡多任务学习中的损失权重

支持三种平衡策略：
1. Uncertainty Weighting (不确定性加权) - 基于任务不确定性自动调整权重
2. GradNorm - 基于梯度范数平衡任务训练速度
3. DWA (Dynamic Weight Average) - 基于任务损失变化率动态调整权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class UncertaintyWeighting(nn.Module):
    """
    基于任务不确定性的权重自动学习
    
    参考论文: "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., CVPR 2018)
    
    原理：
    - 每个任务有一个可学习的log(σ²)参数
    - 损失项会被 1/(2σ²) 加权，同时添加 log(σ²) 正则项
    - 网络自动学习最优的任务权重
    """
    
    def __init__(self, task_names: List[str], initial_log_var: float = 0.0):
        """
        Args:
            task_names: 任务名称列表，如 ['detection', 'segmentation', 'depth']
            initial_log_var: log(σ²)的初始值，默认0表示初始σ²=1
        """
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        
        # 为每个任务创建可学习的log variance参数
        self.log_vars = nn.Parameter(
            torch.ones(self.num_tasks, dtype=torch.float32) * initial_log_var
        )
        
        logger.info(f"🎯 初始化 Uncertainty Weighting，任务数: {self.num_tasks}")
        logger.info(f"   任务列表: {task_names}")
        logger.info(f"   初始 log_var: {initial_log_var}")
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        应用不确定性加权
        
        Args:
            losses: 原始损失字典，如 {'loss_det': x, 'loss_seg': y, 'loss_depth': z}
        
        Returns:
            weighted_losses: 加权后的损失字典
        """
        weighted_losses = {}
        total_loss = 0.0
        
        for i, task_name in enumerate(self.task_names):
            # 查找匹配的损失
            task_loss = None
            for loss_key, loss_value in losses.items():
                if task_name in loss_key.lower():
                    if task_loss is None:
                        task_loss = 0.0
                    task_loss = task_loss + loss_value
            
            if task_loss is not None:
                # 应用不确定性加权: L_weighted = L / (2*σ²) + log(σ²)
                precision = torch.exp(-self.log_vars[i])  # 1/σ²
                weighted_loss = 0.5 * precision * task_loss + 0.5 * self.log_vars[i]
                
                weighted_losses[f"weighted_{task_name}"] = weighted_loss
                total_loss = total_loss + weighted_loss
                
                # 记录权重信息（用于监控）
                weighted_losses[f"weight_{task_name}"] = precision.detach()
                weighted_losses[f"sigma_{task_name}"] = torch.exp(0.5 * self.log_vars[i]).detach()
        
        weighted_losses["total_loss"] = total_loss
        return weighted_losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """返回当前任务权重（precision = 1/σ²）"""
        weights = {}
        precisions = torch.exp(-self.log_vars).detach().cpu()
        for i, task_name in enumerate(self.task_names):
            weights[task_name] = precisions[i].item()
        return weights


class GradNormBalancer(nn.Module):
    """
    基于梯度范数的动态任务平衡
    
    参考论文: "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    (Chen et al., ICML 2018)
    
    原理：
    - 通过平衡不同任务的梯度范数来调整权重
    - 确保所有任务以相似的速度训练
    - 使用可学习的权重参数，通过梯度更新调整
    """
    
    def __init__(
        self, 
        task_names: List[str], 
        alpha: float = 1.5,
        update_frequency: int = 10,
        initial_weight: float = 1.0
    ):
        """
        Args:
            task_names: 任务名称列表
            alpha: 恢复速率超参数，控制任务重要性的相对权重
            update_frequency: 多少次迭代更新一次权重
            initial_weight: 初始权重
        """
        super().__init__()
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.alpha = alpha
        self.update_frequency = update_frequency
        self.iter_count = 0
        
        # 可学习的任务权重
        self.task_weights = nn.Parameter(
            torch.ones(self.num_tasks, dtype=torch.float32) * initial_weight,
            requires_grad=True
        )
        
        # 记录初始损失（用于计算相对损失率）
        self.register_buffer("initial_losses", torch.zeros(self.num_tasks))
        self.register_buffer("loss_ratios", torch.ones(self.num_tasks))
        
        logger.info(f"🎯 初始化 GradNorm Balancer，任务数: {self.num_tasks}")
        logger.info(f"   Alpha: {alpha}, Update Frequency: {update_frequency}")
    
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[torch.nn.Parameter]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        应用GradNorm权重
        
        Args:
            losses: 原始损失字典
            shared_params: 共享参数列表（用于计算梯度范数）
        
        Returns:
            weighted_losses: 加权后的损失字典
        """
        weighted_losses = {}
        task_losses = []
        
        # 收集任务损失
        for i, task_name in enumerate(self.task_names):
            task_loss = None
            for loss_key, loss_value in losses.items():
                if task_name in loss_key.lower():
                    if task_loss is None:
                        task_loss = 0.0
                    task_loss = task_loss + loss_value
            
            if task_loss is not None:
                task_losses.append(task_loss)
            else:
                task_losses.append(torch.tensor(0.0, device=self.task_weights.device))
        
        task_losses_tensor = torch.stack(task_losses)
        
        # 初始化初始损失
        if self.initial_losses.sum() == 0:
            self.initial_losses.copy_(task_losses_tensor.detach())
        
        # 计算加权损失
        weighted_task_losses = self.task_weights * task_losses_tensor
        total_loss = weighted_task_losses.sum()
        
        weighted_losses["total_loss"] = total_loss
        
        # 记录每个任务的加权损失
        for i, task_name in enumerate(self.task_names):
            weighted_losses[f"weighted_{task_name}"] = weighted_task_losses[i]
            weighted_losses[f"weight_{task_name}"] = self.task_weights[i].detach()
        
        return weighted_losses
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: List[torch.nn.Parameter]
    ):
        """
        更新任务权重（需要在backward后调用）
        
        Args:
            losses: 原始损失字典
            shared_params: 共享参数列表
        """
        self.iter_count += 1
        
        if self.iter_count % self.update_frequency != 0:
            return
        
        # 计算当前损失率
        task_losses = []
        for task_name in self.task_names:
            task_loss = None
            for loss_key, loss_value in losses.items():
                if task_name in loss_key.lower():
                    if task_loss is None:
                        task_loss = 0.0
                    task_loss = task_loss + loss_value
            task_losses.append(task_loss if task_loss is not None else torch.tensor(0.0))
        
        task_losses_tensor = torch.stack(task_losses).detach()
        self.loss_ratios = task_losses_tensor / (self.initial_losses + 1e-8)
        
        logger.info(f"🔄 GradNorm weights updated at iter {self.iter_count}")
        logger.info(f"   Loss ratios: {self.loss_ratios.cpu().numpy()}")
        logger.info(f"   Task weights: {self.task_weights.detach().cpu().numpy()}")
    
    def get_task_weights(self) -> Dict[str, float]:
        """返回当前任务权重"""
        weights = {}
        weight_values = self.task_weights.detach().cpu()
        for i, task_name in enumerate(self.task_names):
            weights[task_name] = weight_values[i].item()
        return weights


class DynamicWeightAverage:
    """
    动态权重平均 (Dynamic Weight Average)
    
    参考论文: "End-to-End Multi-Task Learning with Attention"
    (Liu et al., CVPR 2019)
    
    原理：
    - 根据任务损失的变化率动态调整权重
    - 损失下降快的任务权重降低，损失下降慢的任务权重提高
    - 使用指数移动平均平滑权重变化
    """
    
    def __init__(
        self, 
        task_names: List[str],
        temperature: float = 2.0,
        window_size: int = 20
    ):
        """
        Args:
            task_names: 任务名称列表
            temperature: softmax温度参数，控制权重分布的锐利程度
            window_size: 计算损失变化率的窗口大小
        """
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.temperature = temperature
        self.window_size = window_size
        
        # 存储历史损失
        self.loss_history = {task: [] for task in task_names}
        self.task_weights = {task: 1.0 for task in task_names}
        
        logger.info(f"🎯 初始化 Dynamic Weight Average，任务数: {self.num_tasks}")
        logger.info(f"   Temperature: {temperature}, Window: {window_size}")
    
    def __call__(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        应用DWA权重
        
        Args:
            losses: 原始损失字典
        
        Returns:
            weighted_losses: 加权后的损失字典
        """
        weighted_losses = {}
        total_loss = 0.0
        
        # 收集任务损失并更新历史
        task_losses = {}
        for task_name in self.task_names:
            task_loss = None
            for loss_key, loss_value in losses.items():
                if task_name in loss_key.lower():
                    if task_loss is None:
                        task_loss = 0.0
                    task_loss = task_loss + loss_value.item()
            
            if task_loss is not None:
                task_losses[task_name] = task_loss
                self.loss_history[task_name].append(task_loss)
                
                # 保持固定窗口大小
                if len(self.loss_history[task_name]) > self.window_size:
                    self.loss_history[task_name].pop(0)
        
        # 计算损失变化率并更新权重
        if all(len(hist) >= 2 for hist in self.loss_history.values()):
            loss_ratios = []
            for task_name in self.task_names:
                if task_name in task_losses:
                    history = self.loss_history[task_name]
                    # 计算最近两次的损失比率
                    ratio = history[-1] / (history[-2] + 1e-8)
                    loss_ratios.append(ratio)
                else:
                    loss_ratios.append(1.0)
            
            # 使用softmax计算权重
            loss_ratios_tensor = torch.tensor(loss_ratios, dtype=torch.float32)
            weights = F.softmax(loss_ratios_tensor / self.temperature, dim=0) * self.num_tasks
            
            for i, task_name in enumerate(self.task_names):
                self.task_weights[task_name] = weights[i].item()
        
        # 应用权重
        for task_name in self.task_names:
            task_loss = None
            for loss_key, loss_value in losses.items():
                if task_name in loss_key.lower():
                    if task_loss is None:
                        task_loss = 0.0
                    task_loss = task_loss + loss_value
            
            if task_loss is not None:
                weight = self.task_weights[task_name]
                weighted_loss = weight * task_loss
                weighted_losses[f"weighted_{task_name}"] = weighted_loss
                weighted_losses[f"weight_{task_name}"] = torch.tensor(weight, device=task_loss.device)
                total_loss = total_loss + weighted_loss
        
        weighted_losses["total_loss"] = total_loss
        return weighted_losses
    
    def get_task_weights(self) -> Dict[str, float]:
        """返回当前任务权重"""
        return self.task_weights.copy()


class DynamicTaskBalancer(nn.Module):
    """
    统一的动态任务平衡器接口
    
    支持三种策略：
    - 'uncertainty': 基于不确定性的自动权重学习
    - 'gradnorm': 基于梯度范数的动态平衡
    - 'dwa': 基于损失变化率的动态权重平均
    """
    
    def __init__(
        self,
        task_names: List[str],
        method: str = 'uncertainty',
        **kwargs
    ):
        """
        Args:
            task_names: 任务名称列表
            method: 平衡方法 ('uncertainty', 'gradnorm', 'dwa')
            **kwargs: 各方法的特定参数
        """
        super().__init__()
        self.task_names = task_names
        self.method = method
        
        if method == 'uncertainty':
            self.balancer = UncertaintyWeighting(
                task_names=task_names,
                initial_log_var=kwargs.get('initial_log_var', 0.0)
            )
        elif method == 'gradnorm':
            self.balancer = GradNormBalancer(
                task_names=task_names,
                alpha=kwargs.get('alpha', 1.5),
                update_frequency=kwargs.get('update_frequency', 10),
                initial_weight=kwargs.get('initial_weight', 1.0)
            )
        elif method == 'dwa':
            self.balancer = DynamicWeightAverage(
                task_names=task_names,
                temperature=kwargs.get('temperature', 2.0),
                window_size=kwargs.get('window_size', 20)
            )
        else:
            raise ValueError(f"未知的平衡方法: {method}")
        
        logger.info(f"✅ 动态任务平衡器初始化完成")
        logger.info(f"   方法: {method}")
        logger.info(f"   任务: {task_names}")
    
    def forward(
        self, 
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[torch.nn.Parameter]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        应用动态平衡
        
        Args:
            losses: 原始损失字典
            shared_params: 共享参数（仅GradNorm需要）
        
        Returns:
            balanced_losses: 平衡后的损失字典
        """
        if self.method == 'gradnorm':
            return self.balancer(losses, shared_params)
        else:
            return self.balancer(losses)
    
    def get_task_weights(self) -> Dict[str, float]:
        """获取当前任务权重"""
        return self.balancer.get_task_weights()
    
    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        shared_params: Optional[List[torch.nn.Parameter]] = None
    ):
        """更新权重（仅GradNorm需要）"""
        if self.method == 'gradnorm' and hasattr(self.balancer, 'update_weights'):
            self.balancer.update_weights(losses, shared_params)

