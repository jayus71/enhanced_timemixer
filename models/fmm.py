"""
未来多预测器混合模块 (Future Multipredictor Mixing)
增强版本：使用多头注意力机制动态融合不同预测器结果
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

# 导入预测器
from models.predictors.cnn_predictor import CNNPredictor
from models.predictors.lstm_predictor import LSTMPredictor
from models.predictors.transformer_predictor import TransformerPredictor


class MultiHeadPredictorAttention(nn.Module):
    """
    多头预测器注意力机制，用于动态加权融合不同预测器的输出
    """
    
    def __init__(self, 
                 num_predictors: int, 
                 pred_len: int, 
                 hidden_dim: int,
                 num_heads: int = 4):
        """
        初始化多头预测器注意力机制
        
        Args:
            num_predictors: 预测器数量
            pred_len: 预测长度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
        """
        super(MultiHeadPredictorAttention, self).__init__()
        
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_predictors = num_predictors
        self.hidden_dim = hidden_dim
        
        # 检查隐藏层维度是否可以被头数整除
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, 1)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, num_predictors, pred_len, hidden_dim]
        
        Returns:
            注意力权重，形状为 [batch_size, num_predictors, pred_len, 1]
        """
        batch_size, num_predictors, pred_len, hidden_dim = x.size()
        
        # 应用层归一化 - 创建新的张量而不是原地修改
        x_norm = self.layer_norm(x.reshape(-1, hidden_dim)).reshape(batch_size, num_predictors, pred_len, hidden_dim)
        
        # 计算查询、键、值 - 避免使用原地操作
        q = self.q_proj(x_norm).reshape(batch_size, num_predictors, pred_len, self.num_heads, self.head_dim)
        k = self.k_proj(x_norm).reshape(batch_size, num_predictors, pred_len, self.num_heads, self.head_dim)
        v = self.v_proj(x_norm).reshape(batch_size, num_predictors, pred_len, self.num_heads, self.head_dim)
        
        # 调整维度顺序，便于计算注意力
        q = q.permute(0, 3, 1, 2, 4)  # [batch_size, num_heads, num_predictors, pred_len, head_dim]
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)
        
        # 计算注意力分数 - 使用matmul代替手动计算
        attn_scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=x.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) / attn_scale
        
        # 对预测器维度计算softmax
        attention_weights = F.softmax(scores, dim=2)
        
        # 计算加权和 - 使用matmul代替手动计算
        context = torch.matmul(attention_weights, v)
        
        # 重新调整维度顺序 - 使用reshape代替view以避免潜在问题
        context = context.permute(0, 2, 3, 1, 4).contiguous()
        context = context.reshape(batch_size, num_predictors, pred_len, hidden_dim)
        
        # 计算最终权重
        attn_output = self.out_proj(context)
        
        # 对预测器维度归一化
        final_weights = F.softmax(attn_output, dim=1)
        
        return final_weights


class ContextualPredictorFusion(nn.Module):
    """
    上下文感知的预测器融合模块，考虑时间和特征的上下文信息
    """
    
    def __init__(self, 
                 num_predictors: int, 
                 pred_len: int, 
                 hidden_dim: int,
                 dropout: float = 0.1):
        """
        初始化上下文感知的预测器融合模块
        
        Args:
            num_predictors: 预测器数量
            pred_len: 预测长度
            hidden_dim: 隐藏层维度
            dropout: Dropout比例
        """
        super(ContextualPredictorFusion, self).__init__()
        
        # 预测器注意力机制
        self.predictor_attention = MultiHeadPredictorAttention(
            num_predictors=num_predictors,
            pred_len=pred_len,
            hidden_dim=hidden_dim
        )
        
        # 时间维度注意力，捕捉时间步之间的依赖关系
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Gate机制，用于自适应融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 融合后的特征投影
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, predictors_output: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            predictors_output: 所有预测器的输出，形状为 [batch_size, num_predictors, pred_len, hidden_dim]
        
        Returns:
            融合后的预测结果，形状为 [batch_size, pred_len, hidden_dim]
        """
        batch_size, num_predictors, pred_len, hidden_dim = predictors_output.size()
        
        # 使用预测器注意力计算权重
        predictor_weights = self.predictor_attention(predictors_output)
        
        # 加权融合预测器输出
        weighted_output = torch.sum(predictors_output * predictor_weights, dim=1)  # [batch_size, pred_len, hidden_dim]
        
        # 应用时间维度注意力
        temporal_output, _ = self.temporal_attention(
            weighted_output, weighted_output, weighted_output
        )
        temporal_output = self.dropout(temporal_output)
        
        # 残差连接和归一化 - 使用torch.add避免原地操作
        norm_input = torch.add(weighted_output, temporal_output)
        weighted_output = self.norm1(norm_input)
        
        # 使用门控机制决定原始融合结果和时间注意力结果的比例
        gate_value = self.gate(weighted_output)
        
        # 最终融合 - 使用torch.add避免原地操作
        gated_output = gate_value * weighted_output
        proj_output = (1 - gate_value) * self.fusion_proj(weighted_output)
        output = torch.add(gated_output, proj_output)
        output = self.norm2(output)
        
        return output


class FutureMixingModule(nn.Module):
    """
    未来多预测器混合模块，结合多种预测器的输出
    使用增强的注意力融合机制
    """
    
    def __init__(self, 
                 input_lens: List[int], 
                 pred_len: int,
                 input_dim: int,
                 output_dim: int, 
                 predictor_types: List[str] = None,
                 dropout: float = 0.1,
                 device: torch.device = None):
        """
        初始化未来多预测器混合模块
        
        Args:
            input_lens: 不同尺度的序列长度列表
            pred_len: 预测长度
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            predictor_types: 预测器类型列表，可选值: 'cnn', 'lstm', 'transformer'
            dropout: Dropout比例
            device: 运行设备
        """
        super(FutureMixingModule, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        if predictor_types is None:
            # 对每个尺度使用不同预测器类型
            predictor_types = ['cnn', 'lstm', 'transformer'][:len(input_lens)]
            # 如果尺度数量大于预测器类型，循环使用
            predictor_types = predictor_types + predictor_types * (len(input_lens) - len(predictor_types))
            predictor_types = predictor_types[:len(input_lens)]
        
        # 存储预测器类型为类属性，以便外部访问
        self.predictor_types = predictor_types
        
        self.input_lens = input_lens
        self.num_scales = len(input_lens)
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 为每个尺度创建单独的预测器
        self.predictors = nn.ModuleList()
        for i, (scale_len, p_type) in enumerate(zip(input_lens, predictor_types)):
            if p_type.lower() == 'cnn':
                self.predictors.append(CNNPredictor(
                    input_dim=input_dim,
                    output_dim=input_dim,  # 保持维度不变，后续再投影
                    seq_len=scale_len,
                    pred_len=pred_len,
                    device=self.device
                ))
            elif p_type.lower() == 'lstm':
                self.predictors.append(LSTMPredictor(
                    input_dim=input_dim,
                    hidden_dim=input_dim*2,
                    output_dim=input_dim,  # 保持维度不变，后续再投影
                    seq_len=scale_len,
                    pred_len=pred_len,
                    device=self.device
                ))
            elif p_type.lower() == 'transformer':
                self.predictors.append(TransformerPredictor(
                    input_dim=input_dim,
                    output_dim=input_dim,  # 保持维度不变，后续再投影
                    seq_len=scale_len,
                    pred_len=pred_len,
                    device=self.device
                ))
        
        # 上下文感知的预测器融合模块
        self.fusion_module = ContextualPredictorFusion(
            num_predictors=self.num_scales,
            pred_len=pred_len,
            hidden_dim=input_dim,
            dropout=dropout
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(input_dim, output_dim)
        
    def forward(self, x_multi_scale: List[torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x_multi_scale: 多尺度输入张量列表，每个元素形状为 [batch_size, seq_len_i, input_dim]
        
        Returns:
            预测结果，形状为 [batch_size, pred_len, output_dim]
        """
        batch_size = x_multi_scale[0].size(0)
        
        # 对每个尺度使用相应预测器生成预测
        predictions = []
        for i, predictor in enumerate(self.predictors):
            if i < len(x_multi_scale):  # 安全检查
                pred = predictor(x_multi_scale[i])
                predictions.append(pred)
        
        # 堆叠预测结果，形状为 [batch_size, num_predictors, pred_len, hidden_dim]
        stacked_predictions = torch.stack(predictions, dim=1)
        
        # 使用融合模块计算最终预测
        fused_output = self.fusion_module(stacked_predictions)
        
        # 输出投影
        output = self.output_projection(fused_output)
        
        return output, predictions  # 同时返回各个预测器的单独预测，便于可视化