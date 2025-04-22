"""
可视化模块
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple
import torch
import os


def plot_predictions(y_true: Union[np.ndarray, torch.Tensor], 
                     y_pred: Union[np.ndarray, torch.Tensor], 
                     sample_indices: List[int] = None,
                     feature_indices: List[int] = None,
                     labels: List[str] = None,
                     figsize: Tuple[int, int] = (15, 8),
                     save_path: Optional[str] = None,
                     show_plot: bool = True,
                     title: str = "预测结果可视化"):
    """
    绘制预测结果与真实值对比图
    
    Args:
        y_true: 真实值，形状为 [batch_size, pred_len, output_dim] 或 [batch_size, pred_len]
        y_pred: 预测值，形状与y_true相同
        sample_indices: 要绘制的样本索引列表，默认绘制前3个样本
        feature_indices: 要绘制的特征索引列表，默认绘制第一个特征
        labels: 特征名称列表
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
        title: 图像标题
    """
    # 将输入转换为numpy数组
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # 调整维度，确保输入为3D: [batch_size, pred_len, output_dim]
    if len(y_true.shape) == 2:
        y_true = y_true[:, :, np.newaxis]
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, :, np.newaxis]
    
    batch_size, pred_len, output_dim = y_true.shape
    
    # 设置默认样本和特征索引
    if sample_indices is None:
        sample_indices = list(range(min(3, batch_size)))
    if feature_indices is None:
        feature_indices = [0]
    
    # 设置默认标签
    if labels is None:
        labels = [f"特征 {i+1}" for i in range(output_dim)]
    
    # 创建时间步序列
    time_steps = np.arange(pred_len)
    
    # 绘制图形
    num_samples = len(sample_indices)
    num_features = len(feature_indices)
    
    if num_samples * num_features > 1:
        fig, axes = plt.subplots(num_samples, num_features, figsize=figsize)
        
        # 确保axes是二维数组
        if num_samples == 1 and num_features == 1:
            axes = np.array([[axes]])
        elif num_samples == 1:
            axes = axes.reshape(1, -1)
        elif num_features == 1:
            axes = axes.reshape(-1, 1)
            
        for i, sample_idx in enumerate(sample_indices):
            for j, feat_idx in enumerate(feature_indices):
                ax = axes[i, j]
                ax.plot(time_steps, y_true[sample_idx, :, feat_idx], 'b-', label='真实值')
                ax.plot(time_steps, y_pred[sample_idx, :, feat_idx], 'r--', label='预测值')
                ax.set_title(f"样本 {sample_idx+1}, {labels[feat_idx]}")
                ax.set_xlabel('时间步')
                ax.set_ylabel('值')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
    else:
        plt.figure(figsize=figsize)
        sample_idx = sample_indices[0]
        feat_idx = feature_indices[0]
        plt.plot(time_steps, y_true[sample_idx, :, feat_idx], 'b-', label='真实值')
        plt.plot(time_steps, y_pred[sample_idx, :, feat_idx], 'r--', label='预测值')
        plt.title(f"{title} - 样本 {sample_idx+1}, {labels[feat_idx]}")
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_metrics(metrics_list: List[Dict[str, float]], 
                 metric_names: List[str] = None,
                 figsize: Tuple[int, int] = (15, 8),
                 save_path: Optional[str] = None,
                 show_plot: bool = True):
    """
    绘制训练过程中的评估指标变化曲线
    
    Args:
        metrics_list: 包含每个epoch评估指标的字典列表
        metric_names: 要绘制的指标名称列表，默认绘制所有指标
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
    """
    if not metrics_list:
        print("指标列表为空，无法绘制。")
        return
    
    # 如果没有指定指标名称，则使用第一个字典中的所有指标
    if metric_names is None:
        metric_names = list(metrics_list[0].keys())
    
    # 提取所有指定的指标数据
    epochs = list(range(1, len(metrics_list) + 1))
    metrics_data = {name: [] for name in metric_names}
    
    for metrics in metrics_list:
        for name in metric_names:
            if name in metrics:
                metrics_data[name].append(metrics[name])
            else:
                metrics_data[name].append(float('nan'))
    
    # 计算需要的行列数
    num_metrics = len(metric_names)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # 确保axes是二维数组
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个指标的曲线
    for i, metric_name in enumerate(metric_names):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        ax.plot(epochs, metrics_data[metric_name], 'o-')
        ax.set_title(f'{metric_name.upper()}')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Value')
        ax.grid(True)
    
    # 隐藏空白子图
    for i in range(num_metrics, rows * cols):
        row = i // cols
        col = i % cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_loss_curves(train_loss: List[float], 
                     val_loss: List[float] = None,
                     figsize: Tuple[int, int] = (10, 6),
                     save_path: Optional[str] = None,
                     show_plot: bool = True,
                     title: str = "损失函数曲线"):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_loss: 训练损失列表
        val_loss: 验证损失列表，如果为None则不绘制
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
        title: 图像标题
    """
    plt.figure(figsize=figsize)
    epochs = list(range(1, len(train_loss) + 1))
    
    plt.plot(epochs, train_loss, 'b-', label='训练损失')
    if val_loss is not None:
        plt.plot(epochs, val_loss, 'r--', label='验证损失')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_feature_importance(model, feature_names=None, figsize=(10, 6), save_path=None, show_plot=True):
    """
    绘制特征重要性图
    
    Args:
        model: 模型实例，必须有特征重要性属性
        feature_names: 特征名称列表
        figsize: 图像大小
        save_path: 保存路径（如果为None则不保存）
        show_plot: 是否显示图像
    """
    try:
        # 尝试获取模型的特征重要性
        if hasattr(model, 'feature_importance_'):
            importance = model.feature_importance_
        else:
            print("模型没有特征重要性属性")
            return
    except:
        print("无法获取模型的特征重要性")
        return
    
    # 设置特征名称
    if feature_names is None:
        feature_names = [f"特征 {i+1}" for i in range(len(importance))]
    
    # 绘图
    plt.figure(figsize=figsize)
    importance_indices = np.argsort(importance)
    plt.barh(range(len(importance_indices)), importance[importance_indices])
    plt.yticks(range(len(importance_indices)), [feature_names[i] for i in importance_indices])
    plt.xlabel('重要性')
    plt.title('特征重要性')
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    # 显示图像
    if show_plot:
        plt.show()
    else:
        plt.close()