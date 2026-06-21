import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import csv
import time
import os
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns


def visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels, epoch, args):
    """可视化特征中心"""
    # 合并所有特征
    features = torch.cat([f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar], dim=0)
    labels = labels.repeat(4)  # 每个样本有4个特征

    # 创建特征类型标记
    feature_types = []
    n_samples = len(f_sh_hs)
    feature_types.extend(['sh_hs'] * n_samples)
    feature_types.extend(['sh_radar'] * n_samples)
    feature_types.extend(['sp_hs'] * n_samples)
    feature_types.extend(['sp_radar'] * n_samples)

    # TSNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.cpu().numpy())

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=labels.cpu().numpy(),
        style=feature_types,
        palette=sns.color_palette("hls", args.class_num),
        s=100,
        alpha=0.7
    )

    plt.title(f'Feature Centers at Epoch {epoch}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.join(args.log_root, 'feature_vis'), exist_ok=True)
    save_path = os.path.join(args.log_root, 'feature_vis', f'epoch_{epoch}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_features_t():
    """测试特征可视化函数"""
    import argparse

    # 创建模拟参数
    args = argparse.Namespace()
    args.class_num = 15  # 假设有5个类别
    args.log_root = "./test_vis1"  # 测试输出目录

    # 生成模拟数据
    batch_size = 64
    feature_dim = 128

    # 随机生成特征和标签
    f_sh_hs = torch.randn(batch_size, feature_dim)
    f_sh_radar = torch.randn(batch_size, feature_dim)
    f_sp_hs = torch.randn(batch_size, feature_dim)
    f_sp_radar = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, args.class_num, (batch_size,))

    # 调用可视化函数
    visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels, 0, args)
    print("测试可视化完成，图片保存在:", os.path.abspath(args.log_root))


# 可以直接运行这个函数进行测试
if __name__ == "__main__":
    visualize_features_t()

