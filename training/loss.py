"""
损失函数模块
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class MSEWithRegLoss(nn.Module):
    """
    带有正则化的MSE损失函数
    """
    
    def __init__(self, reg_lambda: float = 0.001, reduction: str = 'mean'):
        """
        初始化MSE损失函数
        
        Args:
            reg_lambda: 正则化系数
            reduction: 损失聚合方式，'none'|'mean'|'sum'
        """
        super(MSEWithRegLoss, self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.reg_lambda = reg_lambda
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor, model=None) -> torch.Tensor:
        """
        计算MSE损失和L2正则化
        
        Args:
            pred: 预测值，形状为 [batch_size, pred_len, output_dim]
            target: 真实值，形状为 [batch_size, pred_len, output_dim]
            model: 需要进行正则化的模型，如果为None则不进行参数正则化
        
        Returns:
            带有正则化的MSE损失
        """
        mse_loss = self.mse(pred, target)
        
        # L2正则化项，只在提供了模型的情况下计算
        l2_reg = 0.0
        if model is not None:
            for param in model.parameters():
                if param.requires_grad:
                    l2_reg = l2_reg + torch.sum(torch.square(param))
            
        # 使用加法而不是原地操作
        total_loss = torch.add(mse_loss, torch.mul(self.reg_lambda, l2_reg))
        
        return total_loss


class MAPELoss(nn.Module):
    """
    平均绝对百分比误差损失函数
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        初始化MAPE损失函数
        
        Args:
            epsilon: 小数值，防止除以零
        """
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算MAPE损失
        
        Args:
            pred: 预测值，形状为 [batch_size, pred_len, output_dim]
            target: 真实值，形状为 [batch_size, pred_len, output_dim]
        
        Returns:
            MAPE损失
        """
        # 避免除以零
        mask = torch.abs(target) > self.epsilon
        
        # 计算有效预测点的MAPE
        mape = torch.abs((pred - target) / (torch.abs(target) + self.epsilon))
        
        # 只考虑有效点的损失
        valid_mape = mape * mask.float()
        
        # 计算平均损失
        num_valid = torch.sum(mask.float())
        if num_valid > 0:
            mape_loss = torch.sum(valid_mape) / num_valid
        else:
            mape_loss = torch.sum(mape) / torch.numel(mape)
            
        return mape_loss


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数，结合不同的损失函数
    """
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        """
        初始化多任务损失函数
        
        Args:
            loss_weights: 各损失函数的权重字典
        """
        super(MultiTaskLoss, self).__init__()
        
        if loss_weights is None:
            loss_weights = {'mse': 1.0, 'mape': 0.5}
            
        self.loss_weights = loss_weights
        self.mse_loss = nn.MSELoss()
        self.mape_loss = MAPELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务损失
        
        Args:
            pred: 预测值，形状为 [batch_size, pred_len, output_dim]
            target: 真实值，形状为 [batch_size, pred_len, output_dim]
        
        Returns:
            总损失和各子损失的字典
        """
        losses = {}
        
        # 计算MSE损失
        if 'mse' in self.loss_weights:
            losses['mse'] = self.mse_loss(pred, target)
        
        # 计算MAPE损失
        if 'mape' in self.loss_weights:
            losses['mape'] = self.mape_loss(pred, target)
        
        # 计算加权总损失
        total_loss = 0.0
        for loss_name, loss_value in losses.items():
            if loss_name in self.loss_weights:
                total_loss += self.loss_weights[loss_name] * loss_value
                
        return total_loss, losses