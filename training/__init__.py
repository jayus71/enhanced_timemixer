"""
训练模块初始化文件
"""
from .trainer import Trainer
from .loss import MSEWithRegLoss, MAPELoss, MultiTaskLoss

__all__ = ['Trainer', 'MSEWithRegLoss', 'MAPELoss', 'MultiTaskLoss']