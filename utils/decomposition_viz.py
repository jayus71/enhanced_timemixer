"""
季节性-趋势分解可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Tuple, Optional, Union
import os


def plot_decomposition(original: Union[np.ndarray, torch.Tensor],
                       seasonal: Union[np.ndarray, torch.Tensor],
                       trend: Union[np.ndarray, torch.Tensor],
                       scale_name: str = "Scale",
                       time_steps: Optional[List[int]] = None,
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: Optional[str] = None,
                       show_plot: bool = True):
    """
    绘制时间序列的季节性-趋势分解图
    
    Args:
        original: 原始序列，形状为 [seq_len, feature_dim] 或 [batch_size, seq_len, feature_dim]
        seasonal: 季节性成分，形状同original
        trend: 趋势成分，形状同original
        scale_name: 尺度名称，用于标题
        time_steps: 时间步列表，用于x轴
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
    """
    # 转换为numpy数组
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(seasonal, torch.Tensor):
        seasonal = seasonal.detach().cpu().numpy()
    if isinstance(trend, torch.Tensor):
        trend = trend.detach().cpu().numpy()
    
    # 处理维度
    if len(original.shape) == 3:
        # 选择第一个样本
        original = original[0]
        seasonal = seasonal[0]
        trend = trend[0]
    
    # 如果特征维度大于1，选择第一个特征
    if original.shape[1] > 1:
        original = original[:, 0]
        seasonal = seasonal[:, 0]
        trend = trend[:, 0]
    
    seq_len = len(original)
    if time_steps is None:
        time_steps = np.arange(seq_len)
    
    # 绘图
    plt.figure(figsize=figsize)
    
    # 原始序列
    plt.subplot(3, 1, 1)
    plt.plot(time_steps, original, 'b-', label='Original')
    plt.title(f"{scale_name} - Original Series")
    plt.grid(True)
    plt.legend()
    
    # 季节性成分
    plt.subplot(3, 1, 2)
    plt.plot(time_steps, seasonal, 'g-', label='Seasonal')
    plt.title(f"{scale_name} - Seasonal Component")
    plt.grid(True)
    plt.legend()
    
    # 趋势成分
    plt.subplot(3, 1, 3)
    plt.plot(time_steps, trend, 'r-', label='Trend')
    plt.title(f"{scale_name} - Trend Component")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_multiscale_decomposition(originals: List[Union[np.ndarray, torch.Tensor]],
                                 seasonals: List[Union[np.ndarray, torch.Tensor]],
                                 trends: List[Union[np.ndarray, torch.Tensor]],
                                 scale_names: Optional[List[str]] = None,
                                 figsize: Tuple[int, int] = (15, 12),
                                 save_path: Optional[str] = None,
                                 show_plot: bool = True):
    """
    绘制多尺度时间序列的季节性-趋势分解图
    
    Args:
        originals: 多尺度原始序列列表
        seasonals: 多尺度季节性成分列表
        trends: 多尺度趋势成分列表
        scale_names: 尺度名称列表，用于标题
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
    """
    num_scales = len(originals)
    
    # 设置默认尺度名称
    if scale_names is None:
        scale_names = [f"Scale {i+1}" for i in range(num_scales)]
    
    # 创建图形
    fig, axes = plt.subplots(num_scales, 3, figsize=figsize)
    
    # 遍历各个尺度
    for i in range(num_scales):
        # 获取当前尺度的数据
        original = originals[i].detach().cpu().numpy() if isinstance(originals[i], torch.Tensor) else originals[i]
        seasonal = seasonals[i].detach().cpu().numpy() if isinstance(seasonals[i], torch.Tensor) else seasonals[i]
        trend = trends[i].detach().cpu().numpy() if isinstance(trends[i], torch.Tensor) else trends[i]
        
        # 处理维度
        if len(original.shape) == 3:
            original = original[0]
            seasonal = seasonal[0]
            trend = trend[0]
        
        # 如果特征维度大于1，选择第一个特征
        if original.shape[1] > 1:
            original = original[:, 0]
            seasonal = seasonal[:, 0]
            trend = trend[:, 0]
        
        seq_len = len(original)
        time_steps = np.arange(seq_len)
        
        # 绘制原始序列
        axes[i, 0].plot(time_steps, original, 'b-')
        axes[i, 0].set_title(f"{scale_names[i]} - Original")
        axes[i, 0].grid(True)
        
        # 绘制季节性成分
        axes[i, 1].plot(time_steps, seasonal, 'g-')
        axes[i, 1].set_title(f"{scale_names[i]} - Seasonal")
        axes[i, 1].grid(True)
        
        # 绘制趋势成分
        axes[i, 2].plot(time_steps, trend, 'r-')
        axes[i, 2].set_title(f"{scale_names[i]} - Trend")
        axes[i, 2].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_predictor_outputs(predictions: List[Union[np.ndarray, torch.Tensor]],
                          ground_truth: Union[np.ndarray, torch.Tensor],
                          predictor_names: Optional[List[str]] = None,
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None,
                          show_plot: bool = True):
    """
    绘制不同预测器的输出结果
    
    Args:
        predictions: 预测结果列表，每个预测器一个
        ground_truth: 真实值
        predictor_names: 预测器名称列表
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
    """
    num_predictors = len(predictions)
    
    # 转换为numpy数组
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    
    predictions_np = []
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            predictions_np.append(pred.detach().cpu().numpy())
        else:
            predictions_np.append(pred)
    
    # 设置默认预测器名称
    if predictor_names is None:
        predictor_names = [f"Predictor {i+1}" for i in range(num_predictors)]
    
    # 处理维度
    if len(ground_truth.shape) == 3:
        # 选择第一个样本和第一个特征
        ground_truth = ground_truth[0, :, 0]
        predictions_np = [pred[0, :, 0] for pred in predictions_np]
    
    pred_len = len(ground_truth)
    time_steps = np.arange(pred_len)
    
    # 绘图
    plt.figure(figsize=figsize)
    
    # 绘制真实值
    plt.plot(time_steps, ground_truth, 'k-', linewidth=2, label='Ground Truth')
    
    # 绘制各预测器的预测结果
    colors = plt.cm.tab10(np.linspace(0, 1, num_predictors))
    for i, (pred, name) in enumerate(zip(predictions_np, predictor_names)):
        plt.plot(time_steps, pred, '-', color=colors[i], linewidth=1.5, label=name)
    
    plt.title("Comparison of Different Predictor Outputs")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()