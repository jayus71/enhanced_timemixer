"""
Transformer预测器实现
"""
import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    位置编码模块，为Transformer提供序列位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        初始化位置编码模块
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 计算sin和cos位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 将位置编码添加为缓冲区（不作为模型参数）
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，将位置编码添加到输入中
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        
        Returns:
            添加位置编码后的张量，形状为 [batch_size, seq_len, d_model]
        """
        # 创建新张量而不是修改原有张量
        return x.clone() + self.pe[:, :x.size(1)]


class TransformerPredictor(nn.Module):
    """
    基于Transformer的预测器
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 seq_len: int, 
                 pred_len: int,
                 d_model: int = None,
                 nhead: int = 8, 
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1,
                 device: torch.device = None):
        """
        初始化Transformer预测器
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            seq_len: 输入序列长度
            pred_len: 预测序列长度
            d_model: Transformer模型维度，如果为None则使用input_dim
            nhead: 多头注意力机制中的头数
            num_encoder_layers: 编码器层数
            num_decoder_layers: 解码器层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout比例
            device: 运行设备
        """
        super(TransformerPredictor, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if d_model is None:
            d_model = input_dim
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # 输入投影层，如果输入维度与模型维度不同
        self.input_projection = nn.Linear(input_dim, d_model) if input_dim != d_model else nn.Identity()
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 处理encoder输出并生成decoder输入序列
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # 用于生成预测的MLP层
        self.forecast_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        生成下三角掩码矩阵，用于解码器中的自注意力
        
        Args:
            sz: 序列长度
        
        Returns:
            掩码矩阵
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_dim]
        
        Returns:
            预测输出，形状为 [batch_size, pred_len, output_dim]
        """
        batch_size = x.size(0)
        
        # 输入投影
        x_projected = self.input_projection(x)
        
        # 添加位置编码 - 避免原地操作
        x_with_pos = self.positional_encoding(x_projected)
        
        # Transformer编码器
        memory = self.transformer_encoder(x_with_pos)
        
        # 创建新的解码器输入序列，而不是修改现有的
        start_tokens = self.start_token.expand(batch_size, -1, -1)
        decoder_input = torch.zeros(batch_size, self.pred_len, self.d_model, device=x.device)
        
        # 使用切片赋值而不是原地修改
        decoder_input_updated = decoder_input.clone()
        decoder_input_updated[:, 0:1, :] = start_tokens
        
        # 创建解码器掩码
        tgt_mask = self._generate_square_subsequent_mask(self.pred_len)
        
        # 自回归预测 - 避免修改需要梯度的张量
        for i in range(1, self.pred_len):
            # 对当前输入创建副本，避免视图共享
            curr_input = decoder_input_updated[:, :i, :].clone()
            curr_mask = tgt_mask[:i, :i]
            
            # Transformer解码器
            decoder_output = self.transformer_decoder(
                curr_input, 
                memory, 
                tgt_mask=curr_mask
            )
            
            # 处理解码器输出
            next_pred = self.forecast_mlp(decoder_output[:, -1:, :])
            
            # 创建新张量而不是修改原有张量
            next_decoder_input = decoder_input_updated.clone()
            next_decoder_input[:, i:i+1, :] = next_pred
            decoder_input_updated = next_decoder_input
        
        # 最终预测结果
        output = self.output_projection(decoder_input_updated)
        
        return output