"""
TimeMixer 预测脚本
"""
import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from data.dataloader import DataLoader
from data.preprocessing import DataPreprocessor, MultiScaleGenerator
from models.timemixer import TimeMixer
from utils.metrics import compute_metrics
from utils.visualization import plot_predictions


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


def load_model(model_path, config, device):
    """
    加载预训练模型
    
    Args:
        model_path: 模型检查点路径
        config: 配置字典
        device: 运行设备
    
    Returns:
        加载的模型
    """
    # 创建模型实例
    model = TimeMixer(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        seq_len=config['data']['seq_len'],
        pred_len=config['data']['pred_len'],
        hidden_dim=config['model']['hidden_dim'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        predictor_types=config['model']['predictor_types'],
        use_time_features=config['data'].get('time_feat', True),
        use_multi_scale=config['data'].get('use_multi_scale', True),
        device=device
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置为评估模式
    model.eval()
    
    return model


def predict(model, data_loader, device, config):
    """
    使用加载的模型进行预测
    
    Args:
        model: 加载的模型
        data_loader: 数据加载器
        device: 运行设备
        config: 配置字典
    
    Returns:
        预测结果、真实值和评估指标
    """
    model.eval()
    test_loader = data_loader.get_test_loader()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # 预测
            output = model(data)
            
            # 保存结果
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    # 合并所有批次的结果
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    # 如果数据已标准化，则需要反标准化
    if config['data']['scale']:
        predictions = data_loader.inverse_transform(predictions)
        targets = data_loader.inverse_transform(targets)
    
    # 计算评估指标
    metrics = compute_metrics(predictions, targets)
    
    return predictions, targets, metrics


def save_predictions(predictions, targets, output_path):
    """
    保存预测结果到CSV文件
    
    Args:
        predictions: 预测值
        targets: 真实值
        output_path: 输出路径
    """
    # 创建结果目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 重塑数组以便保存
    batch_size, pred_len, output_dim = predictions.shape
    preds_reshaped = predictions.reshape(-1, output_dim)
    targets_reshaped = targets.reshape(-1, output_dim)
    
    # 创建索引和列名
    indices = np.repeat(np.arange(batch_size), pred_len)
    time_steps = np.tile(np.arange(pred_len), batch_size)
    
    # 创建DataFrame
    result_df = pd.DataFrame()
    result_df['样本索引'] = indices
    result_df['时间步'] = time_steps
    
    # 添加预测值和真实值列
    for i in range(output_dim):
        result_df[f'预测值_{i+1}'] = preds_reshaped[:, i]
        result_df[f'真实值_{i+1}'] = targets_reshaped[:, i]
    
    # 保存到CSV
    result_df.to_csv(output_path, index=False)
    print(f"预测结果已保存到 {output_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='TimeMixer 模型预测')
    parser.add_argument('--config', type=str, default='./configs/default.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--output', type=str, default='./predictions/results.csv',
                        help='预测结果输出路径')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化预测结果')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置设备
    device_str = config['training'].get('device', 'cuda')
    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    data_loader = DataLoader(
        data_path=config['data']['data_path'],
        batch_size=config['evaluation'].get('test_batch_size', 64),
        seq_len=config['data']['seq_len'],
        pred_len=config['data']['pred_len'],
        train_ratio=config['data']['train_ratio'],
        valid_ratio=config['data']['valid_ratio'],
        scale=config['data']['scale'],
        num_workers=config['data']['num_workers']
    )
    
    # 加载模型
    print(f"加载模型：{args.model}")
    model = load_model(args.model, config, device)
    
    # 进行预测
    print("正在预测...")
    predictions, targets, metrics = predict(model, data_loader, device, config)
    
    # 打印评估指标
    print("\n预测结果评估指标:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper()}: {metric_value:.6f}")
    
    # 保存预测结果
    save_predictions(predictions, targets, args.output)
    
    # 可视化预测结果
    if args.visualize:
        print("可视化预测结果...")
        vis_dir = os.path.join(os.path.dirname(args.output), 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 绘制多个样本的预测结果
        for sample_idx in range(min(5, predictions.shape[0])):
            plot_predictions(
                y_true=targets[sample_idx:sample_idx+1],
                y_pred=predictions[sample_idx:sample_idx+1],
                save_path=os.path.join(vis_dir, f'prediction_sample_{sample_idx+1}.png'),
                title=f"样本 {sample_idx+1} 的预测结果",
                show_plot=False
            )
        
        # 绘制综合图
        plot_predictions(
            y_true=targets[:5],
            y_pred=predictions[:5],
            sample_indices=list(range(5)),
            save_path=os.path.join(vis_dir, 'predictions_combined.png'),
            show_plot=True
        )


if __name__ == '__main__':
    main()