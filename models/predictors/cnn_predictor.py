"""
CNN预测器实现
"""
import torch
import torch.nn as nn
from typing import Optional


class CNNPredictor(nn.Module):
    """
    基于卷积神经网络的预测器
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 seq_len: int, 
                 pred_len: int,
                 kernel_size: int = 3,
                 num_layers: int = 3, 
                 dropout: float = 0.1,
                 device: torch.device = None):
        """
        初始化CNN预测器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            kernel_size: 卷积核大小
            num_layers: 卷积层数量
            dropout: Dropout比例
            device: 运行设备
        """
        super(CNNPredictor, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 确保卷积核大小为奇数，便于保持序列长度不变
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # 卷积层
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=padding)
        )
        
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=padding)
            )
        
        # 激活函数
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 用于从卷积特征生成预测的全连接层
        self.forecast_mlp = nn.Sequential(
            nn.Linear(seq_len, pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len, pred_len)
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        
        Returns:
            预测输出，形状为 [batch_size, pred_len, output_dim]
        """
        batch_size = x.size(0)
        
        # 转换维度以适应卷积操作 [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 应用卷积层
        for conv in self.conv_layers:
            residual = x
            x = self.activation(conv(x))
            x = self.dropout(x)
            x = x + residual  # 残差连接
        
        # 使用全连接层生成预测 [batch_size, input_dim, pred_len]
        x_forecast = self.forecast_mlp(x)
        
        # 转换回原始维度顺序 [batch_size, pred_len, input_dim]
        output = x_forecast.transpose(1, 2)
        
        # 输出投影
        output = self.output_projection(output)
        
        return output