"""
TimeMixer主模型架构，增强版本
"""
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple

from models.pdm import PastDecomposableMixer
from models.fmm import FutureMixingModule


class TimeMixer(nn.Module):
    """
    增强版TimeMixer: 具有高级分解方法和注意力融合机制
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 seq_len: int,
                 pred_len: int,
                 hidden_dim: int = 512,
                 num_layers: int = 3,
                 scale_factors: List[int] = None,  # 多尺度下采样因子
                 kernel_sizes: List[int] = [25, 49, 97],  # 移动平均核大小
                 dropout: float = 0.1,
                 predictor_types: List[str] = None,
                 device: torch.device = None):
        super(TimeMixer, self).__init__()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        if scale_factors is None:
            scale_factors = [1, 2, 4]  # 默认尺度因子
            
        if predictor_types is None:
            predictor_types = ['cnn', 'lstm', 'transformer'][:len(scale_factors)]
            # 如果尺度数量大于预测器类型，循环使用
            predictor_types = predictor_types + predictor_types * (len(scale_factors) - len(predictor_types))
            predictor_types = predictor_types[:len(scale_factors)]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = hidden_dim
        self.scale_factors = scale_factors
        
        # 计算每个尺度的序列长度
        self.input_lens = [seq_len // sf for sf in scale_factors]
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # PDM层堆叠
        self.pdm_layers = nn.ModuleList([
            PastDecomposableMixer(
                input_lens=self.input_lens,
                hidden_dim=hidden_dim,
                kernel_sizes=kernel_sizes,
                dropout=dropout,
                device=self.device
            ) for _ in range(num_layers)
        ])
        
        # FMM层
        self.fmm = FutureMixingModule(
            input_lens=self.input_lens,
            pred_len=pred_len,
            input_dim=hidden_dim,
            output_dim=output_dim,
            predictor_types=predictor_types,
            dropout=dropout,
            device=self.device
        )
        
    def forward(self, x, return_decomp: bool = False, return_predictions: bool = False):
        """
        前向传播
        
        Args:
            x: 输入数据，可以是单个张量 [batch_size, seq_len, input_dim] 
               或者是多尺度输入的张量列表 [batch_size, seq_len_i, input_dim]
            return_decomp: 是否返回分解结果（用于可视化）
            return_predictions: 是否返回各预测器的预测结果（用于可视化）
        
        Returns:
            预测输出，形状为 [batch_size, pred_len, output_dim]
            可选: 分解结果
            可选: 各预测器的预测结果
        """
        # 检查输入是张量还是列表
        if isinstance(x, list):
            # 如果已经是多尺度输入列表，直接使用
            # print(f"接收到多尺度输入列表，数量: {len(x)}")
            batch_size = x[0].size(0)
            
            # 对每个尺度的输入进行投影
            x_multi_scale = []
            originals = [] if return_decomp else None
            
            for i, x_scale in enumerate(x):
                # 保存原始尺度序列
                if return_decomp:
                    originals.append(x_scale.detach())
                
                # 输入投影
                projected = self.input_projection(x_scale)
                x_multi_scale.append(projected)
        else:
            # 使用原始方法处理单个张量
            batch_size = x.size(0)
            
            # 生成多尺度输入
            x_multi_scale = []
            originals = [] if return_decomp else None
            
            for i, sf in enumerate(self.scale_factors):
                if sf == 1:
                    # 原始尺度，直接使用
                    downsampled = x
                else:
                    # 平均下采样
                    seq_len = (self.seq_len // sf) * sf
                    temp = x[:, :seq_len, :].reshape(batch_size, -1, sf, self.input_dim)
                    downsampled = temp.mean(dim=2)
                
                # 保存原始尺度序列
                if return_decomp:
                    originals.append(downsampled.detach())
                    
                # 输入投影
                projected = self.input_projection(downsampled)
                x_multi_scale.append(projected)
        
        # 存储最后一层的分解结果
        decomp_results = None
        
        # 通过PDM层
        for i, pdm in enumerate(self.pdm_layers):
            if i == len(self.pdm_layers) - 1 and return_decomp:  # 只保存最后一层的分解结果
                x_multi_scale, decomp_results = pdm(x_multi_scale)
            else:
                x_multi_scale, _ = pdm(x_multi_scale)
        
        # 通过FMM层进行预测
        if return_predictions:
            output, predictor_outputs = self.fmm(x_multi_scale)
        else:
            output, _ = self.fmm(x_multi_scale)
            predictor_outputs = None
        
        # 根据需要返回结果
        if return_decomp and return_predictions:
            return output, (originals, decomp_results), predictor_outputs
        elif return_decomp:
            return output, (originals, decomp_results)
        elif return_predictions:
            return output, predictor_outputs
        else:
            return output