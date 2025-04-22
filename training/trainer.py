"""
训练器模块实现
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Tuple, Callable
from tqdm import tqdm

from utils.metrics import compute_metrics


class Trainer:
    """
    模型训练器类，负责模型的训练、验证和测试
    """
    
    def __init__(self, 
                 model: nn.Module, 
                 criterion: nn.Module,
                 optimizer: optim.Optimizer = None,
                 learning_rate: float = 0.001,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: torch.device = None,
                 checkpoint_dir: str = './checkpoints'):
        """
        初始化训练器
        
        Args:
            model: 待训练的模型
            criterion: 损失函数
            optimizer: 优化器，如果为空则创建Adam优化器
            learning_rate: 学习率，当optimizer为空时使用
            scheduler: 学习率调度器
            device: 运行设备
            checkpoint_dir: 模型检查点保存目录
        """
        self.model = model
        self.criterion = criterion
        
        # 初始化优化器（如果没有提供）
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        # 设置学习率调度器
        if scheduler is None:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, 
                verbose=True, min_lr=1e-6
            )
        else:
            self.scheduler = scheduler
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 将模型移至指定设备
        self.model.to(self.device)
        
        # 设置检查点目录
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # 初始化训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0
        
        # 使用tqdm包装数据加载器，显示进度条
        progress_bar = tqdm(train_loader, desc="训练批次", leave=False)
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            # 检查data是否为列表（多尺度输入的情况）
            if isinstance(data, list):
                # 将每个尺度的数据都移到设备上
                data = [x.to(self.device) for x in data]
            else:
                data = data.to(self.device)
                
            target = target.to(self.device)
            
            # 梯度清零
            self.optimizer.zero_grad()
            
            # 前向传播
            output = self.model(data)
            
            # 计算损失 - 检查是否是MSEWithRegLoss类型，如果是则传递模型参数
            if hasattr(self.criterion, 'reg_lambda'):  # 检查是否是MSEWithRegLoss类型
                loss = self.criterion(output, target, self.model)
            else:
                loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条显示当前批次的损失
            progress_bar.set_postfix(loss=loss.item())
            
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        在验证集上评估模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            平均验证损失和评估指标
        """
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # 检查data是否为列表（多尺度输入的情况）
                if isinstance(data, list):
                    # 将每个尺度的数据都移到设备上
                    data = [x.to(self.device) for x in data]
                else:
                    data = data.to(self.device)
                    
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失 - 检查是否是MSEWithRegLoss类型，如果是则传递模型参数
                if hasattr(self.criterion, 'reg_lambda'):  # 检查是否是MSEWithRegLoss类型
                    loss = self.criterion(output, target, self.model)
                else:
                    loss = self.criterion(output, target)
                
                # 累计损失
                total_loss += loss.item()
                
                # 保存预测结果和真实值，用于计算指标
                all_outputs.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # 合并所有批次的预测和真实值
        all_outputs = np.vstack(all_outputs)
        all_targets = np.vstack(all_targets)
        
        # 计算评估指标
        metrics = compute_metrics(all_outputs, all_targets)
        
        return avg_loss, metrics
    
    def test(self, test_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        在测试集上评估模型
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            平均测试损失和评估指标
        """
        # 测试逻辑与验证相同
        return self.validate(test_loader)
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader, 
              num_epochs: int, 
              patience: int = 20,
              verbose: bool = True,
              save_best: bool = True,
              save_interval: int = 10,
              log_interval: int = 1,
              callback: Optional[Callable[[int, float, float, Dict[str, float]], None]] = None) -> Dict:
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            patience: 早停的耐心值
            verbose: 是否打印详细信息
            save_best: 是否保存最佳模型
            save_interval: 保存模型的间隔轮数
            log_interval: 日志打印的间隔轮数
            callback: 每个epoch后的回调函数
        
        Returns:
            训练历史记录
        """
        # 初始化训练记录
        history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': [],
            'time_per_epoch': []
        }
        
        # 初始化早停计数器
        patience_counter = 0
        
        print(f"开始训练，设备：{self.device}")
        
        # 使用tqdm创建epochs的进度条
        epochs_progress = tqdm(range(num_epochs), desc="训练进度", unit="epoch")
        
        for epoch in epochs_progress:
            start_time = time.time()
            
            # 训练一个epoch
            train_loss = self.train_epoch(train_loader)
            
            # 在验证集上评估
            val_loss, val_metrics = self.validate(val_loader)
            
            # 更新学习率调度器
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # 记录当前epoch的训练信息
            epoch_time = time.time() - start_time
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['metrics'].append(val_metrics)
            history['time_per_epoch'].append(epoch_time)
            
            # 更新进度条显示当前的损失值
            epochs_progress.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'MSE': f'{val_metrics["mse"]:.4f}',
                'MAE': f'{val_metrics["mae"]:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # 打印详细训练信息(如果需要)
            if verbose and (epoch % log_interval == 0 or epoch == num_epochs - 1):
                tqdm.write(f"Epoch {epoch+1}/{num_epochs} - "
                           f"训练损失: {train_loss:.6f}, "
                           f"验证损失: {val_loss:.6f}, "
                           f"MSE: {val_metrics['mse']:.6f}, "
                           f"MAE: {val_metrics['mae']:.6f}, "
                           f"耗时: {epoch_time:.2f}秒")
            
            # 回调函数
            if callback is not None:
                callback(epoch, train_loss, val_loss, val_metrics)
            
            # 保存模型
            if save_interval > 0 and (epoch % save_interval == 0 or epoch == num_epochs - 1):
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pth", epoch)
            
            # 保存最佳模型
            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth", epoch)
                patience_counter = 0
                if verbose:
                    tqdm.write(f"发现最佳模型，已保存! 验证损失: {val_loss:.6f}")
            else:
                patience_counter += 1
            
            # 早停
            if patience > 0 and patience_counter >= patience:
                tqdm.write(f"早停! {patience} 个epoch内验证损失未改善。")
                break
        
        print(f"训练完成! 最佳验证损失: {self.best_val_loss:.6f}")
        
        return history
    
    def save_checkpoint(self, filename: str, epoch: int = None):
        """
        保存模型检查点
        
        Args:
            filename: 检查点文件名
            epoch: 当前epoch数
        """
        # 准备保存的状态
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        # 保存模型
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True) -> int:
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            load_optimizer: 是否加载优化器状态
        
        Returns:
            加载的epoch数（如果可用）
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态（如果需要）
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        
        # 返回epoch数（如果有）
        return checkpoint.get('epoch', -1)
    
    def plot_losses(self, figsize: Tuple[int, int] = (10, 6), save_path: Optional[str] = None):
        """
        绘制训练和验证损失曲线
        
        Args:
            figsize: 图像大小
            save_path: 保存路径（如果为None则只显示不保存）
        """
        plt.figure(figsize=figsize)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        
        plt.show()