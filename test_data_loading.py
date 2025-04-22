"""
测试数据加载功能
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data.dataloader import DataLoader, MultiDatasetLoader

# 设置数据目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_root = os.path.join(current_dir, 'data')

def test_single_dataset_loading():
    """测试单个数据集的加载"""
    print("=" * 60)
    print("测试单个数据集加载")
    print("=" * 60)
    
    # 1. 测试AIIA小时级数据集
    aiia_path = os.path.join(data_root, 'AIIA_hour')
    if os.path.exists(aiia_path):
        try:
            print("\n加载AIIA小时级数据集...")
            aiia_loader = DataLoader(
                data_path=aiia_path,
                batch_size=64,
                seq_len=96,
                pred_len=24,
                scale=True
            )
            print(f"AIIA数据形状: {aiia_loader.data.shape}")
            print(f"训练样本数: {len(aiia_loader.train_dataset)}")
            print(f"验证样本数: {len(aiia_loader.valid_dataset)}")
            print(f"测试样本数: {len(aiia_loader.test_dataset)}")
        except Exception as e:
            print(f"加载AIIA数据集失败: {e}")
    else:
        print(f"AIIA数据集目录不存在: {aiia_path}")
    
    # 2. 测试中国联通数据集
    unicom_path = os.path.join(data_root, 'China_Unicom_One_Cell_Data')
    if os.path.exists(unicom_path):
        try:
            print("\n加载中国联通数据集...")
            unicom_loader = DataLoader(
                data_path=unicom_path,
                batch_size=64,
                seq_len=96,
                pred_len=24,
                scale=True
            )
            print(f"联通数据形状: {unicom_loader.data.shape}")
            print(f"训练样本数: {len(unicom_loader.train_dataset)}")
            print(f"验证样本数: {len(unicom_loader.valid_dataset)}")
            print(f"测试样本数: {len(unicom_loader.test_dataset)}")
        except Exception as e:
            print(f"加载联通数据集失败: {e}")
    else:
        print(f"联通数据集目录不存在: {unicom_path}")
    
    # 3. 测试LTE网络数据集
    lte_path = os.path.join(data_root, 'lte_network')
    if os.path.exists(lte_path):
        try:
            print("\n加载LTE网络数据集...")
            lte_loader = DataLoader(
                data_path=lte_path,
                batch_size=64,
                seq_len=96,
                pred_len=24,
                scale=True
            )
            print(f"LTE数据形状: {lte_loader.data.shape}")
            print(f"训练样本数: {len(lte_loader.train_dataset)}")
            print(f"验证样本数: {len(lte_loader.valid_dataset)}")
            print(f"测试样本数: {len(lte_loader.test_dataset)}")
        except Exception as e:
            print(f"加载LTE数据集失败: {e}")
    else:
        print(f"LTE数据集目录不存在: {lte_path}")

def test_multi_dataset_loading():
    """测试多数据集加载器"""
    print("=" * 60)
    print("测试多数据集加载器")
    print("=" * 60)
    
    try:
        # 创建多数据集加载器
        multi_loader = MultiDatasetLoader(
            data_root=data_root,
            batch_size=64,
            seq_len=96,
            pred_len=24,
            scale=True
        )
        
        # 加载所有可用的数据集
        dataset_loaders = multi_loader.load_all_datasets()
        
        print(f"\n成功加载的数据集: {list(dataset_loaders.keys())}")
        
        # 打印每个数据集的信息
        for name, loader in dataset_loaders.items():
            print(f"\n数据集: {name}")
            print(f"数据形状: {loader.data.shape}")
            print(f"特征数量: {loader.data.shape[1]}")
            print(f"样本数量: {len(loader.train_dataset) + len(loader.valid_dataset) + len(loader.test_dataset)}")
    
    except Exception as e:
        print(f"多数据集加载失败: {e}")

def plot_dataset_samples():
    """绘制各数据集的样例数据"""
    print("=" * 60)
    print("绘制数据集样例")
    print("=" * 60)
    
    # 创建多数据集加载器
    multi_loader = MultiDatasetLoader(data_root=data_root)
    dataset_loaders = multi_loader.load_all_datasets()
    
    # 为每个数据集创建一个图表
    for name, loader in dataset_loaders.items():
        try:
            # 获取一些样本数据
            data = loader.data
            
            # 选择前500个时间点和第一个特征进行可视化
            samples = min(500, data.shape[0])
            
            plt.figure(figsize=(12, 6))
            
            # 如果有多个特征，我们最多绘制前3个特征
            features = min(3, data.shape[1])
            
            for i in range(features):
                plt.plot(data[:samples, i], label=f'Feature {i+1}')
            
            plt.title(f'{name} 数据集样例')
            plt.xlabel('时间')
            plt.ylabel('值')
            plt.legend()
            plt.grid(True)
            
            # 保存图片
            save_dir = os.path.join(current_dir, 'visualizations', 'dataset_samples')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f'{name}_samples.png'))
            plt.close()
            
            print(f"已保存 {name} 数据集样例图")
        except Exception as e:
            print(f"绘制 {name} 数据集样例失败: {e}")

if __name__ == "__main__":
    # 测试单个数据集加载
    test_single_dataset_loading()
    
    # 测试多数据集加载
    test_multi_dataset_loading()
    
    # 绘制样本数据
    plot_dataset_samples()