"""
TimeMixer 主入口脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import numpy as np
import random
import time
from datetime import datetime
import matplotlib.pyplot as plt

from data.dataloader import DataLoader
from data.preprocessing import DataPreprocessor, MultiScaleGenerator, TimeFeatureGenerator
from models.timemixer import TimeMixer
from training.trainer import Trainer
from training.loss import MSEWithRegLoss, MAPELoss, MultiTaskLoss
from utils.metrics import compute_metrics
from utils.visualization import plot_predictions, plot_loss_curves


def set_seed(seed):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        配置字典
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config, device):
    """
    创建模型实例
    
    Args:
        config: 配置字典
        device: 运行设备
    
    Returns:
        模型实例
    """
    # 设置尺度因子
    scale_factors = config.get('scale_factors', [1, 2, 4])
    
    model = TimeMixer(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        seq_len=config['data']['seq_len'],
        pred_len=config['data']['pred_len'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        scale_factors=scale_factors,
        dropout=config['model']['dropout'],
        predictor_types=config['model']['predictor_types'],
        device=device
    )
    return model


def create_optimizer(model, config):
    """
    创建优化器
    
    Args:
        model: 模型实例
        config: 配置字典
    
    Returns:
        优化器实例
    """
    optimizer_type = config['training'].get('optimizer', 'adam').lower()
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training'].get('weight_decay', 0))
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=0.9, 
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    return optimizer


def create_scheduler(optimizer, config):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器实例
        config: 配置字典
    
    Returns:
        学习率调度器实例
    """
    scheduler_type = config['training'].get('scheduler', 'reduce_lr_on_plateau').lower()
    
    if scheduler_type == 'reduce_lr_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            verbose=True, min_lr=1e-6
        )
    elif scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['training']['epochs'], eta_min=1e-6
        )
    else:
        raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    return scheduler


def create_loss_function(config):
    """
    创建损失函数
    
    Args:
        config: 配置字典
    
    Returns:
        损失函数实例
    """
    loss_type = config['training'].get('loss_type', 'mse').lower()
    
    if loss_type == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_type == 'mae':
        criterion = torch.nn.L1Loss()
    elif loss_type == 'mse_with_reg':
        reg_lambda = config['training'].get('reg_lambda', 0.001)
        criterion = MSEWithRegLoss(reg_lambda=reg_lambda)
    elif loss_type == 'multi_task':
        loss_weights = config['training'].get('loss_weights', {'mse': 1.0, 'mape': 0.5})
        criterion = MultiTaskLoss(loss_weights=loss_weights)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
    
    return criterion


def train_model(config):
    """
    训练模型的主要流程
    
    Args:
        config: 配置字典
    """
    # 设置运行设备
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据加载
    print(f"加载数据...")
    data_loader = DataLoader(
        data_path=config['data']['data_path'],
        batch_size=config['data']['batch_size'],
        seq_len=config['data']['seq_len'],
        pred_len=config['data']['pred_len'],
        train_ratio=config['data']['train_ratio'],
        valid_ratio=config['data']['valid_ratio'],
        scale=config['data']['scale'],
        num_workers=config['data']['num_workers'],
        target_col=config['data']['target_col'],
        time_col=config['data']['time_col']
    )
    
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_valid_loader()
    test_loader = data_loader.get_test_loader()
    
    # 创建模型
    print(f"创建模型...")
    model = create_model(config, device)
    
    # 创建优化器
    optimizer = create_optimizer(model, config)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)
    
    # 创建损失函数
    criterion = create_loss_function(config)
    torch.autograd.set_detect_anomaly(True)
    # 创建训练器
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=config['saving']['checkpoint_dir']
    )
    
    # 训练模型
    print(f"开始训练...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['epochs'],
        patience=config['training']['patience'],
        save_best=True,
        save_interval=config['saving']['save_interval'],
        log_interval=config['saving']['log_interval']
    )
    
    # 绘制损失曲线
    print(f"绘制损失曲线...")
    os.makedirs(config['saving']['visualization_dir'], exist_ok=True)
    plot_loss_curves(
        history['train_loss'],
        history['val_loss'],
        save_path=os.path.join(config['saving']['visualization_dir'], 'loss_curve.png')
    )
    
    # 在测试集上评估模型
    print(f"在测试集上评估模型...")
    test_loss, test_metrics = trainer.test(test_loader)
    print(f"测试集损失: {test_loss:.6f}")
    for metric_name, metric_value in test_metrics.items():
        print(f"测试集 {metric_name.upper()}: {metric_value:.6f}")
    
    print("正在生成季节性-趋势分解可视化...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        x, y = batch
        
        # 修复x是列表的情况
        if isinstance(x, list):
            # 如果x是列表（多尺度输入），把每个元素都移到设备上
            x = [x_scale.to(device) for x_scale in x]
        else:
            x = x.to(device)
            
        y = y.to(device)
        
        # 使用带有可视化信息的前向传播
        output, (originals, decomp_results), predictor_outputs = model(x, return_decomp=True, return_predictions=True)
        
        # 创建可视化目录
        viz_dir = os.path.join(config['saving']['visualization_dir'], 'decomposition')
        os.makedirs(viz_dir, exist_ok=True)
        
        # 生成尺度名称
        scale_names = [f"Scale {sf}x" for sf in model.scale_factors]
        
        # 绘制多尺度分解图
        from utils.decomposition_viz import plot_multiscale_decomposition
        
        # 提取原始序列和分解结果
        original_tensors = originals
        seasonal_tensors = [res[0] for res in decomp_results]
        trend_tensors = [res[1] for res in decomp_results]
        
        # 绘制分解图
        plot_multiscale_decomposition(
            originals=original_tensors,
            seasonals=seasonal_tensors,
            trends=trend_tensors,
            scale_names=scale_names,
            save_path=os.path.join(viz_dir, 'multiscale_decomposition.png')
        )
        
        # 绘制预测器输出比较图
        from utils.decomposition_viz import plot_predictor_outputs
        
        # 生成预测器名称
        predictor_names = [f"{ptype.capitalize()} (Scale {sf}x)" 
                        for ptype, sf in zip(model.fmm.predictor_types, model.scale_factors)]
        
        # 绘制预测器输出比较图
        plot_predictor_outputs(
            predictions=predictor_outputs,
            ground_truth=y,
            predictor_names=predictor_names,
            save_path=os.path.join(viz_dir, 'predictor_outputs_comparison.png')
        )
        
        print(f"可视化结果已保存到 {viz_dir}")
    
    # 保存一些测试集的预测结果
    print(f"保存预测可视化结果...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        x, y = batch
        
        # 修复x是列表的情况
        if isinstance(x, list):
            # 如果x是列表（多尺度输入），把每个元素都移到设备上
            x = [x_scale.to(device) for x_scale in x]
        else:
            x = x.to(device)
            
        y = y.to(device)
        output = model(x)
        
        # 反标准化（如果需要）
        if config['data']['scale']:
            output_np = output.cpu().numpy()
            output_np = data_loader.inverse_transform(output_np)
            y_np = y.cpu().numpy()
            y_np = data_loader.inverse_transform(y_np)
        else:
            output_np = output.cpu().numpy()
            y_np = y.cpu().numpy()
        
        # 绘制预测结果
        plot_predictions(
            y_true=y_np,
            y_pred=output_np,
            sample_indices=[0, 1, 2],
            save_path=os.path.join(config['saving']['visualization_dir'], 'prediction_results.png')
        )
    
    print(f"训练完成!")
    return model, test_metrics


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='TimeMixer 时间序列预测')
    parser.add_argument('--config', type=str, default='./configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 确保目录存在
    os.makedirs(config['saving']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['saving']['visualization_dir'], exist_ok=True)
    
    # 训练模型
    model, test_metrics = train_model(config)
    
    # 打印最终结果
    print("\n最终评估结果:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.6f}")


if __name__ == '__main__':
    main()