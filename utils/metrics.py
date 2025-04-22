"""
评估指标模块
"""
import numpy as np
from typing import Dict, Union, Callable
import torch


def MAE(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算平均绝对误差(Mean Absolute Error)
    
    Args:
        pred: 预测值
        true: 真实值
        mask: 可选的掩码，用于过滤无效值
    
    Returns:
        MAE值
    """
    if mask is None:
        return np.mean(np.abs(pred - true))
    else:
        return np.sum(np.abs(pred - true) * mask) / (np.sum(mask) + 1e-10)


def MSE(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算均方误差(Mean Squared Error)
    
    Args:
        pred: 预测值
        true: 真实值
        mask: 可选的掩码，用于过滤无效值
    
    Returns:
        MSE值
    """
    if mask is None:
        return np.mean(np.square(pred - true))
    else:
        return np.sum(np.square(pred - true) * mask) / (np.sum(mask) + 1e-10)


def RMSE(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None) -> float:
    """
    计算均方根误差(Root Mean Squared Error)
    
    Args:
        pred: 预测值
        true: 真实值
        mask: 可选的掩码，用于过滤无效值
    
    Returns:
        RMSE值
    """
    return np.sqrt(MSE(pred, true, mask))


def MAPE(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None, epsilon: float = 1e-8) -> float:
    """
    计算平均绝对百分比误差(Mean Absolute Percentage Error)
    
    Args:
        pred: 预测值
        true: 真实值
        mask: 可选的掩码，用于过滤无效值
        epsilon: 小值，防止除以零
    
    Returns:
        MAPE值
    """
    abs_percentage_error = np.abs((pred - true) / (np.abs(true) + epsilon))
    
    if mask is None:
        # 只考虑真实值非零的部分
        valid_mask = np.abs(true) > epsilon
        return np.sum(abs_percentage_error * valid_mask) / (np.sum(valid_mask) + 1e-10)
    else:
        # 结合提供的掩码和真实值非零的掩码
        valid_mask = mask * (np.abs(true) > epsilon)
        return np.sum(abs_percentage_error * valid_mask) / (np.sum(valid_mask) + 1e-10)


def SMAPE(pred: np.ndarray, true: np.ndarray, mask: np.ndarray = None, epsilon: float = 1e-8) -> float:
    """
    计算对称平均绝对百分比误差(Symmetric Mean Absolute Percentage Error)
    
    Args:
        pred: 预测值
        true: 真实值
        mask: 可选的掩码，用于过滤无效值
        epsilon: 小值，防止除以零
    
    Returns:
        SMAPE值
    """
    smape_value = 2.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + epsilon)
    
    if mask is None:
        return np.mean(smape_value)
    else:
        return np.sum(smape_value * mask) / (np.sum(mask) + 1e-10)


def compute_metrics(pred: Union[np.ndarray, torch.Tensor], 
                    true: Union[np.ndarray, torch.Tensor],
                    mask: Union[np.ndarray, torch.Tensor] = None,
                    metric_names: list = None) -> Dict[str, float]:
    """
    计算多个评估指标
    
    Args:
        pred: 预测值，形状为 [batch_size, pred_len, output_dim] 或 [batch_size, pred_len]
        true: 真实值，形状为 [batch_size, pred_len, output_dim] 或 [batch_size, pred_len]
        mask: 可选的掩码，用于过滤无效值
        metric_names: 要计算的指标名称列表，默认计算所有指标
    
    Returns:
        评估指标字典
    """
    # 将输入转换为numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # 默认计算所有指标
    if metric_names is None:
        metric_names = ['mae', 'mse', 'rmse', 'mape', 'smape']
    
    # 初始化结果字典
    metrics = {}
    
    # 计算指定的指标
    for metric in metric_names:
        if metric.lower() == 'mae':
            metrics['mae'] = MAE(pred, true, mask)
        elif metric.lower() == 'mse':
            metrics['mse'] = MSE(pred, true, mask)
        elif metric.lower() == 'rmse':
            metrics['rmse'] = RMSE(pred, true, mask)
        elif metric.lower() == 'mape':
            metrics['mape'] = MAPE(pred, true, mask)
        elif metric.lower() == 'smape':
            metrics['smape'] = SMAPE(pred, true, mask)
    
    return metrics