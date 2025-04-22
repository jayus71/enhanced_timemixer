"""
过去可分解混合模块 (Past Decomposable Mixing)
增强版本：使用多核移动平均分解和特殊边界处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MovingAvg(nn.Module):
    """
    具有特殊边界处理的移动平均模块
    """
    
    def __init__(self, kernel_size, stride=1):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        # 对时间序列两端进行填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # 应用平均池化
        x_avg = self.avg(x_padded.transpose(1, 2))
        
        return x_avg.transpose(1, 2)


class SeriesDecompMulti(nn.Module):
    """
    多核移动平均分解模块，来自FEDformer
    """
    
    def __init__(self, kernel_sizes=[25, 49, 97]):
        super(SeriesDecompMulti, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.moving_avgs = nn.ModuleList([MovingAvg(kernel_size) for kernel_size in kernel_sizes])
        
    def forward(self, x):
        moving_means = []
        
        # 应用多个不同核大小的移动平均
        for moving_avg in self.moving_avgs:
            moving_means.append(moving_avg(x))
        
        # 计算平均移动均值 - 使用torch.stack和torch.mean代替sum/len
        moving_means_stacked = torch.stack(moving_means, dim=0)
        moving_mean = torch.mean(moving_means_stacked, dim=0)
        
        # 季节性成分 = 原始序列 - 趋势成分
        # 使用torch.sub避免可能的原地操作
        seasonal = torch.sub(x, moving_mean)
        
        return seasonal, moving_mean


class MyLayerNorm(nn.Module):
    """
    针对季节性部分的特殊设计层归一化
    """
    
    def __init__(self, channels):
        super(MyLayerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)
        
    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class BottomUpMixing(nn.Module):
    """自下而上混合模块，用于季节性组件"""
    
    def __init__(self, input_len, output_len, hidden_dim):
        super(BottomUpMixing, self).__init__()
        
        # 从较长序列映射到较短序列的线性层
        self.temporal_linear = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.GELU()
        )
        
        # 特殊层归一化
        self.norm = MyLayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, T_in, C]
        x = self.norm(x)  # 应用特殊层归一化
        x_transposed = x.transpose(1, 2)  # [B, C, T_in]
        output = self.temporal_linear(x_transposed)  # [B, C, T_out]
        return output.transpose(1, 2)  # [B, T_out, C]


class TopDownMixing(nn.Module):
    """自上而下混合模块，用于趋势组件"""
    
    def __init__(self, input_len, output_len, hidden_dim):
        super(TopDownMixing, self).__init__()
        
        # 从较短序列映射到较长序列的线性层
        self.temporal_linear = nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.GELU()
        )
        
        # 标准层归一化
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, T_in, C]
        x = self.norm(x)  # 应用标准层归一化
        x_transposed = x.transpose(1, 2)  # [B, C, T_in]
        output = self.temporal_linear(x_transposed)  # [B, C, T_out]
        return output.transpose(1, 2)  # [B, T_out, C]


class PastDecomposableMixer(nn.Module):
    """
    过去可分解混合模块，使用增强的分解和混合方法
    """
    
    def __init__(self, 
                 input_lens: List[int],  # 不同尺度的序列长度列表
                 hidden_dim: int, 
                 kernel_sizes: List[int] = [25, 49, 97],
                 dropout: float = 0.1,
                 device: torch.device = None):
        super(PastDecomposableMixer, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.input_lens = input_lens
        self.hidden_dim = hidden_dim
        self.num_scales = len(input_lens)
        
        # 序列分解模块 - 使用多核移动平均
        self.decomp = SeriesDecompMulti(kernel_sizes)
        
        # 季节性混合（自下而上）
        self.seasonal_mixing = nn.ModuleList([
            BottomUpMixing(input_lens[i-1], input_lens[i], hidden_dim)
            for i in range(1, self.num_scales)
        ])
        
        # 趋势混合（自上而下）
        self.trend_mixing = nn.ModuleList([
            TopDownMixing(input_lens[i+1], input_lens[i], hidden_dim)
            for i in range(self.num_scales-2, -1, -1)
        ])
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
    
    def forward(self, x_multi_scale: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            x_multi_scale: 多尺度输入张量列表，每个元素形状为 [batch_size, seq_len_i, hidden_dim]
        
        Returns:
            混合后的多尺度表示列表
            分解结果列表（用于可视化），每个元素为(seasonal, trend)元组
        """
        # 分解每个尺度的输入
        seasonal_parts = []
        trend_parts = []
        decomp_results = []  # 存储分解结果用于可视化
        
        for x in x_multi_scale:
            season, trend = self.decomp(x)
            seasonal_parts.append(season)
            trend_parts.append(trend)
            decomp_results.append((season.detach(), trend.detach()))  # 分离梯度，仅用于可视化
            
        # 自下而上混合季节性部分 (fine-to-coarse)
        for i in range(1, self.num_scales):
            # 使用torch.add避免原地操作
            seasonal_parts[i] = torch.add(seasonal_parts[i], self.seasonal_mixing[i-1](seasonal_parts[i-1]))
            
        # 自上而下混合趋势部分 (coarse-to-fine)
        for i in range(self.num_scales-2, -1, -1):
            # 使用torch.add避免原地操作
            trend_parts[i] = torch.add(trend_parts[i], self.trend_mixing[self.num_scales-2-i](trend_parts[i+1]))
            
        # 重组并通过前馈网络
        output_multi_scale = []
        for i in range(self.num_scales):
            combined = torch.add(seasonal_parts[i], trend_parts[i])
            # 使用torch.add避免原地操作
            output = torch.add(x_multi_scale[i], self.feed_forward(combined))
            output_multi_scale.append(output)
            
        return output_multi_scale, decomp_results