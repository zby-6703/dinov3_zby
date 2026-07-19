"""
模型性能指标评估器 (Model Performance Metrics)

计算模型的性能指标：
- 参数量 (Parameters)
- GFLOPs (Giga Floating Point Operations)
- FPS (Frames Per Second)
  - Model FPS: 仅模型推理时间
  - Postprocess FPS: 后处理时间（如深度估计）
  - Total FPS: 总体端到端时间

使用示例:
    from utils.metrics import PerformanceMetrics
    
    # 创建评估器
    perf = PerformanceMetrics()
    
    # 计算参数量
    params = perf.count_parameters(model)
    
    # 计算GFLOPs
    gflops = perf.calculate_flops(model, input_size=(3, 512, 512))
    
    # 测量FPS（需要提供推理函数）
    fps_results = perf.measure_fps(
        inference_fn=lambda img: model(img),
        postprocess_fn=lambda outputs: depth_estimator(outputs),
        test_input=image,
        num_warmup=10,
        num_iterations=100
    )
    
    # 获取完整报告
    print(perf.summary())
"""

import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Callable, Any, Tuple, Union, List
from dataclasses import dataclass, field


@dataclass
class ParameterStats:
    """参数统计信息"""
    total_params: int = 0
    trainable_params: int = 0
    non_trainable_params: int = 0
    total_params_M: float = 0.0  # 百万
    trainable_params_M: float = 0.0
    
    # 按模块统计
    module_params: Dict[str, int] = field(default_factory=dict)


@dataclass 
class FLOPsStats:
    """FLOPs统计信息"""
    total_flops: int = 0
    total_gflops: float = 0.0
    
    # 按模块统计
    module_flops: Dict[str, int] = field(default_factory=dict)
    
    # 错误信息（如果计算失败）
    error: Optional[str] = None


@dataclass
class FPSStats:
    """FPS统计信息"""
    # 模型推理FPS
    model_fps: float = 0.0
    model_latency_ms: float = 0.0
    model_latency_std_ms: float = 0.0
    
    # 后处理FPS
    postprocess_fps: float = 0.0
    postprocess_latency_ms: float = 0.0
    postprocess_latency_std_ms: float = 0.0
    
    # 总体FPS
    total_fps: float = 0.0
    total_latency_ms: float = 0.0
    total_latency_std_ms: float = 0.0
    
    # 测试配置
    num_warmup: int = 0
    num_iterations: int = 0
    device: str = "cuda"


class PerformanceMetrics:
    """
    模型性能指标评估器
    
    独立封装的性能评估工具，可用于评估任意PyTorch模型的：
    - 参数量
    - GFLOPs  
    - FPS（模型推理、后处理、总体）
    """
    
    def __init__(self, device: str = "cuda"):
        """
        初始化性能评估器
        
        Args:
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.param_stats: Optional[ParameterStats] = None
        self.flops_stats: Optional[FLOPsStats] = None
        self.fps_stats: Optional[FPSStats] = None
        
    def reset(self):
        """重置所有统计数据"""
        self.param_stats = None
        self.flops_stats = None
        self.fps_stats = None
    
    # ==================== 参数量计算 ====================
    
    def count_parameters(
        self, 
        model: nn.Module,
        include_module_breakdown: bool = False
    ) -> ParameterStats:
        """
        计算模型参数量
        
        Args:
            model: PyTorch模型
            include_module_breakdown: 是否包含按模块统计
            
        Returns:
            ParameterStats: 参数统计信息
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        stats = ParameterStats(
            total_params=total_params,
            trainable_params=trainable_params,
            non_trainable_params=non_trainable_params,
            total_params_M=total_params / 1e6,
            trainable_params_M=trainable_params / 1e6,
        )
        
        # 按模块统计
        if include_module_breakdown:
            for name, module in model.named_children():
                module_params = sum(p.numel() for p in module.parameters())
                if module_params > 0:
                    stats.module_params[name] = module_params
        
        self.param_stats = stats
        return stats
    
    # ==================== FLOPs计算 ====================
    
    def calculate_flops(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 512, 512),
        input_constructor: Optional[Callable] = None,
        include_module_breakdown: bool = False
    ) -> FLOPsStats:
        """
        计算模型FLOPs
        
        Args:
            model: PyTorch模型
            input_size: 输入尺寸 (C, H, W)
            input_constructor: 自定义输入构造函数，用于特殊模型（如Detectron2）
            include_module_breakdown: 是否包含按模块统计
            
        Returns:
            FLOPsStats: FLOPs统计信息
        """
        stats = FLOPsStats()
        
        try:
            from fvcore.nn import FlopCountAnalysis, flop_count_table
            
            # 构造输入
            if input_constructor is not None:
                dummy_input = input_constructor(input_size)
            else:
                dummy_input = torch.randn(1, *input_size).to(self.device)
            
            # 确保模型在正确设备上
            model = model.to(self.device)
            model.eval()
            
            # 计算FLOPs
            with torch.no_grad():
                flop_analysis = FlopCountAnalysis(model, dummy_input)
                total_flops = flop_analysis.total()
                
                stats.total_flops = total_flops
                stats.total_gflops = total_flops / 1e9
                
                # 按模块统计
                if include_module_breakdown:
                    by_module = flop_analysis.by_module()
                    for name, flops in by_module.items():
                        if flops > 0 and '.' not in name:  # 只取顶层模块
                            stats.module_flops[name] = flops
                            
        except ImportError:
            stats.error = "fvcore未安装，请运行: pip install fvcore"
        except Exception as e:
            stats.error = str(e)
        
        self.flops_stats = stats
        return stats
    
    def calculate_flops_thop(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int] = (3, 512, 512),
    ) -> FLOPsStats:
        """
        使用thop库计算FLOPs（备选方法）
        
        Args:
            model: PyTorch模型
            input_size: 输入尺寸 (C, H, W)
            
        Returns:
            FLOPsStats: FLOPs统计信息
        """
        stats = FLOPsStats()
        
        try:
            from thop import profile, clever_format
            
            dummy_input = torch.randn(1, *input_size).to(self.device)
            model = model.to(self.device)
            model.eval()
            
            with torch.no_grad():
                macs, params = profile(model, inputs=(dummy_input,), verbose=False)
                # MACs ≈ FLOPs / 2，但通常报告时直接用MACs近似FLOPs
                stats.total_flops = int(macs * 2)  # 转换为FLOPs
                stats.total_gflops = stats.total_flops / 1e9
                
        except ImportError:
            stats.error = "thop未安装，请运行: pip install thop"
        except Exception as e:
            stats.error = str(e)
        
        self.flops_stats = stats
        return stats
    
    # ==================== FPS测量 ====================
    
    def measure_fps(
        self,
        inference_fn: Callable[[Any], Any],
        test_input: Any,
        postprocess_fn: Optional[Callable[[Any], Any]] = None,
        num_warmup: int = 10,
        num_iterations: int = 100,
        sync_cuda: bool = True
    ) -> FPSStats:
        """
        测量FPS（模型推理、后处理、总体）
        
        Args:
            inference_fn: 模型推理函数，接受输入返回输出
            test_input: 测试输入数据
            postprocess_fn: 后处理函数（可选），接受模型输出返回最终结果
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            sync_cuda: 是否同步CUDA（GPU测试时需要）
            
        Returns:
            FPSStats: FPS统计信息
        """
        stats = FPSStats(
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            device=self.device
        )
        
        # 同步函数
        def sync():
            if sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # ========== 预热 ==========
        print(f"  FPS测量: 预热中 ({num_warmup}次)...")
        for _ in range(num_warmup):
            outputs = inference_fn(test_input)
            if postprocess_fn is not None:
                _ = postprocess_fn(outputs)
        sync()
        
        # ========== 测量模型推理时间 ==========
        print(f"  FPS测量: 测量模型推理时间 ({num_iterations}次)...")
        model_latencies = []
        model_outputs_cache = []
        
        for i in range(num_iterations):
            sync()
            start_time = time.perf_counter()
            outputs = inference_fn(test_input)
            sync()
            end_time = time.perf_counter()
            
            model_latencies.append((end_time - start_time) * 1000)  # ms
            model_outputs_cache.append(outputs)
        
        model_latencies = np.array(model_latencies)
        stats.model_latency_ms = np.mean(model_latencies)
        stats.model_latency_std_ms = np.std(model_latencies)
        stats.model_fps = 1000.0 / stats.model_latency_ms
        
        # ========== 测量后处理时间 ==========
        if postprocess_fn is not None:
            print(f"  FPS测量: 测量后处理时间 ({num_iterations}次)...")
            postprocess_latencies = []
            
            for i in range(num_iterations):
                outputs = model_outputs_cache[i]
                sync()
                start_time = time.perf_counter()
                _ = postprocess_fn(outputs)
                sync()
                end_time = time.perf_counter()
                
                postprocess_latencies.append((end_time - start_time) * 1000)  # ms
            
            postprocess_latencies = np.array(postprocess_latencies)
            stats.postprocess_latency_ms = np.mean(postprocess_latencies)
            stats.postprocess_latency_std_ms = np.std(postprocess_latencies)
            stats.postprocess_fps = 1000.0 / stats.postprocess_latency_ms if stats.postprocess_latency_ms > 0 else float('inf')
        
        # ========== 测量总体时间 ==========
        print(f"  FPS测量: 测量总体时间 ({num_iterations}次)...")
        total_latencies = []
        
        for _ in range(num_iterations):
            sync()
            start_time = time.perf_counter()
            outputs = inference_fn(test_input)
            if postprocess_fn is not None:
                _ = postprocess_fn(outputs)
            sync()
            end_time = time.perf_counter()
            
            total_latencies.append((end_time - start_time) * 1000)  # ms
        
        total_latencies = np.array(total_latencies)
        stats.total_latency_ms = np.mean(total_latencies)
        stats.total_latency_std_ms = np.std(total_latencies)
        stats.total_fps = 1000.0 / stats.total_latency_ms
        
        self.fps_stats = stats
        return stats
    
    def measure_fps_simple(
        self,
        inference_fn: Callable[[Any], Any],
        test_input: Any,
        num_warmup: int = 10,
        num_iterations: int = 100,
        sync_cuda: bool = True
    ) -> FPSStats:
        """
        简化的FPS测量（仅测量模型推理）
        
        Args:
            inference_fn: 模型推理函数
            test_input: 测试输入
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            sync_cuda: 是否同步CUDA
            
        Returns:
            FPSStats: FPS统计信息
        """
        return self.measure_fps(
            inference_fn=inference_fn,
            test_input=test_input,
            postprocess_fn=None,
            num_warmup=num_warmup,
            num_iterations=num_iterations,
            sync_cuda=sync_cuda
        )
    
    # ==================== 综合评估 ====================
    
    def evaluate(
        self,
        model: nn.Module,
        inference_fn: Callable[[Any], Any],
        test_input: Any,
        input_size: Tuple[int, int, int] = (3, 512, 512),
        postprocess_fn: Optional[Callable[[Any], Any]] = None,
        num_warmup: int = 10,
        num_iterations: int = 100,
        calculate_flops: bool = True
    ) -> Dict[str, Any]:
        """
        综合评估模型性能
        
        Args:
            model: PyTorch模型
            inference_fn: 推理函数
            test_input: 测试输入
            input_size: 输入尺寸（用于FLOPs计算）
            postprocess_fn: 后处理函数
            num_warmup: 预热次数
            num_iterations: 测试迭代次数
            calculate_flops: 是否计算FLOPs
            
        Returns:
            Dict: 完整性能指标
        """
        results = {}
        
        # 参数量
        print("计算参数量...")
        param_stats = self.count_parameters(model)
        results['parameters'] = {
            'total': param_stats.total_params,
            'total_M': param_stats.total_params_M,
            'trainable': param_stats.trainable_params,
            'trainable_M': param_stats.trainable_params_M,
        }
        
        # FLOPs
        if calculate_flops:
            print("计算GFLOPs...")
            flops_stats = self.calculate_flops(model, input_size)
            results['flops'] = {
                'total': flops_stats.total_flops,
                'gflops': flops_stats.total_gflops,
                'error': flops_stats.error,
            }
        
        # FPS
        print("测量FPS...")
        fps_stats = self.measure_fps(
            inference_fn=inference_fn,
            test_input=test_input,
            postprocess_fn=postprocess_fn,
            num_warmup=num_warmup,
            num_iterations=num_iterations
        )
        results['fps'] = {
            'model_fps': fps_stats.model_fps,
            'model_latency_ms': fps_stats.model_latency_ms,
            'postprocess_fps': fps_stats.postprocess_fps,
            'postprocess_latency_ms': fps_stats.postprocess_latency_ms,
            'total_fps': fps_stats.total_fps,
            'total_latency_ms': fps_stats.total_latency_ms,
        }
        
        return results
    
    # ==================== 输出报告 ====================
    
    def compute(self) -> Dict[str, Any]:
        """
        获取所有已计算的指标
        
        Returns:
            Dict: 所有性能指标
        """
        results = {}
        
        if self.param_stats is not None:
            results['parameters'] = {
                'total': self.param_stats.total_params,
                'total_M': self.param_stats.total_params_M,
                'trainable': self.param_stats.trainable_params,
                'trainable_M': self.param_stats.trainable_params_M,
                'non_trainable': self.param_stats.non_trainable_params,
            }
            if self.param_stats.module_params:
                results['parameters']['by_module'] = self.param_stats.module_params
        
        if self.flops_stats is not None:
            results['flops'] = {
                'total': self.flops_stats.total_flops,
                'gflops': self.flops_stats.total_gflops,
            }
            if self.flops_stats.error:
                results['flops']['error'] = self.flops_stats.error
            if self.flops_stats.module_flops:
                results['flops']['by_module'] = self.flops_stats.module_flops
        
        if self.fps_stats is not None:
            results['fps'] = {
                'model': {
                    'fps': self.fps_stats.model_fps,
                    'latency_ms': self.fps_stats.model_latency_ms,
                    'latency_std_ms': self.fps_stats.model_latency_std_ms,
                },
                'postprocess': {
                    'fps': self.fps_stats.postprocess_fps,
                    'latency_ms': self.fps_stats.postprocess_latency_ms,
                    'latency_std_ms': self.fps_stats.postprocess_latency_std_ms,
                },
                'total': {
                    'fps': self.fps_stats.total_fps,
                    'latency_ms': self.fps_stats.total_latency_ms,
                    'latency_std_ms': self.fps_stats.total_latency_std_ms,
                },
                'config': {
                    'num_warmup': self.fps_stats.num_warmup,
                    'num_iterations': self.fps_stats.num_iterations,
                    'device': self.fps_stats.device,
                }
            }
        
        return results
    
    def summary(self) -> str:
        """
        生成性能指标摘要字符串
        
        Returns:
            str: 格式化的摘要报告
        """
        lines = [
            "=" * 60,
            "模型性能指标 (Model Performance Metrics)",
            "=" * 60,
        ]
        
        # 参数量
        if self.param_stats is not None:
            lines.extend([
                "",
                "【参数量 Parameters】",
                "-" * 40,
                f"  总参数量:        {self.param_stats.total_params:,}",
                f"  总参数量 (M):    {self.param_stats.total_params_M:.2f} M",
                f"  可训练参数:      {self.param_stats.trainable_params:,}",
                f"  不可训练参数:    {self.param_stats.non_trainable_params:,}",
            ])
            if self.param_stats.module_params:
                lines.append("  按模块统计:")
                for name, params in sorted(self.param_stats.module_params.items(), 
                                          key=lambda x: -x[1])[:5]:
                    lines.append(f"    - {name}: {params:,} ({params/1e6:.2f}M)")
        
        # FLOPs
        if self.flops_stats is not None:
            lines.extend([
                "",
                "【计算量 FLOPs】",
                "-" * 40,
            ])
            if self.flops_stats.error:
                lines.append(f"  GFLOPs:          N/A ({self.flops_stats.error})")
            else:
                lines.extend([
                    f"  总FLOPs:         {self.flops_stats.total_flops:,}",
                    f"  GFLOPs:          {self.flops_stats.total_gflops:.2f}",
                ])
        
        # FPS
        if self.fps_stats is not None:
            lines.extend([
                "",
                "【速度 FPS】",
                "-" * 40,
                f"  模型推理 FPS:    {self.fps_stats.model_fps:.2f}",
                f"    - 延迟:        {self.fps_stats.model_latency_ms:.2f} ± {self.fps_stats.model_latency_std_ms:.2f} ms",
            ])
            if self.fps_stats.postprocess_latency_ms > 0:
                lines.extend([
                    f"  后处理 FPS:      {self.fps_stats.postprocess_fps:.2f}",
                    f"    - 延迟:        {self.fps_stats.postprocess_latency_ms:.2f} ± {self.fps_stats.postprocess_latency_std_ms:.2f} ms",
                ])
            lines.extend([
                f"  总体 FPS:        {self.fps_stats.total_fps:.2f}",
                f"    - 延迟:        {self.fps_stats.total_latency_ms:.2f} ± {self.fps_stats.total_latency_std_ms:.2f} ms",
                f"  测试配置:        {self.fps_stats.num_iterations}次迭代, {self.fps_stats.num_warmup}次预热",
            ])
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于JSON保存）"""
        return self.compute()
    
    def save(self, filepath: str):
        """保存结果到JSON文件"""
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.compute(), f, indent=2, ensure_ascii=False)
        print(f"性能指标已保存到: {filepath}")


# ==================== 便捷函数 ====================

def count_model_parameters(model: nn.Module) -> Dict[str, Any]:
    """
    快速计算模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        Dict: {'total': int, 'total_M': float, 'trainable': int}
    """
    perf = PerformanceMetrics()
    stats = perf.count_parameters(model)
    return {
        'total': stats.total_params,
        'total_M': stats.total_params_M,
        'trainable': stats.trainable_params,
    }


def calculate_model_flops(
    model: nn.Module, 
    input_size: Tuple[int, int, int] = (3, 512, 512),
    device: str = 'cuda'
) -> Dict[str, Any]:
    """
    快速计算模型FLOPs
    
    Args:
        model: PyTorch模型
        input_size: 输入尺寸
        device: 设备
        
    Returns:
        Dict: {'gflops': float, 'error': Optional[str]}
    """
    perf = PerformanceMetrics(device=device)
    stats = perf.calculate_flops(model, input_size)
    return {
        'gflops': stats.total_gflops,
        'error': stats.error,
    }


def measure_model_fps(
    inference_fn: Callable,
    test_input: Any,
    postprocess_fn: Optional[Callable] = None,
    num_iterations: int = 100,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    快速测量模型FPS
    
    Args:
        inference_fn: 推理函数
        test_input: 测试输入
        postprocess_fn: 后处理函数
        num_iterations: 迭代次数
        device: 设备
        
    Returns:
        Dict: {'model_fps': float, 'postprocess_fps': float, 'total_fps': float}
    """
    perf = PerformanceMetrics(device=device)
    stats = perf.measure_fps(
        inference_fn=inference_fn,
        test_input=test_input,
        postprocess_fn=postprocess_fn,
        num_iterations=num_iterations
    )
    return {
        'model_fps': stats.model_fps,
        'postprocess_fps': stats.postprocess_fps,
        'total_fps': stats.total_fps,
    }
