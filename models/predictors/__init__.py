"""
预测器模块初始化文件
"""
from .cnn_predictor import CNNPredictor
from .lstm_predictor import LSTMPredictor
from .transformer_predictor import TransformerPredictor

__all__ = ['CNNPredictor', 'LSTMPredictor', 'TransformerPredictor']