"""
LSTM预测器实现
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMPredictor(nn.Module):
    """
    基于LSTM的预测器
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 output_dim: int, 
                 seq_len: int, 
                 pred_len: int,
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 bidirectional: bool = True,
                 device: torch.device = None):
        """
        初始化LSTM预测器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            output_dim: 输出特征维度
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            num_layers: LSTM层数
            dropout: Dropout比例
            bidirectional: 是否使用双向LSTM
            device: 运行设备
        """
        super(LSTMPredictor, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # 用于生成预测的全连接层
        lstm_output_dim = hidden_dim * self.num_directions
        self.forecast_fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 用于生成多步预测的解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, pred_len * hidden_dim)
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        
        Returns:
            预测输出，形状为 [batch_size, pred_len, output_dim]
        """
        batch_size = x.size(0)
        
        # 初始化隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(self.device)
        
        # LSTM前向传播
        output, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 如果是双向LSTM，合并前向和后向的最终隐藏状态
        if self.bidirectional:
            # 取最后一层的前向和后向隐藏状态并拼接
            h_last_layer = hn[-2:].transpose(0, 1).contiguous()
            h_last_layer = h_last_layer.view(batch_size, -1)
        else:
            h_last_layer = hn[-1]
        
        # 应用全连接层生成表示
        h_fc = self.forecast_fc(h_last_layer)
        
        # 解码生成多步预测
        h_decoded = self.decoder(h_fc)
        predictions = h_decoded.view(batch_size, self.pred_len, -1)
        
        # 输出投影
        output = self.output_projection(predictions)
        
        return output