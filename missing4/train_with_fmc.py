import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 导入自定义模块
from models1 import Drfuse, IntraClassCompactnessLoss
from fmc_module import FMCModule
from fmc_train import FMCTrainer
from config import get_args

# 尝试导入数据加载器（如果可用）
try:
    from src.huston2013_dataloader import create_data_loader
    DATA_LOADER_AVAILABLE = True
except ImportError:
    print("警告: 无法导入数据加载器，将使用模拟数据")
    DATA_LOADER_AVAILABLE = False


def create_demo_dataloader(args):
    """创建演示数据加载器"""
    class DemoDataset:
        def __init__(self, num_samples=1000):
            self.num_samples = num_samples

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # 创建模拟的多模态数据
            hsi_data = torch.randn(args.hsi_channels, args.image_size, args.image_size)
            radar_data = torch.randn(args.radar_channels, args.image_size, args.image_size)

            # 拼接数据
            combined_data = torch.cat([hsi_data, radar_data], dim=0)

            # 随机标签
            label = torch.randint(0, args.class_num, (1,)).item()

            return combined_data, label

    from torch.utils.data import DataLoader
    dataset = DemoDataset(num_samples=2000)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataloader


def train_drfuse_with_fmc():
    """训练集成FMC的Drfuse模型"""

    # 获取参数
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print(f"使用设备: {device}")
    print(f"训练轮数: {args.train_epoch}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")

    # 创建数据加载器
    if DATA_LOADER_AVAILABLE:
        print("使用真实数据加载器...")
        train_loader = create_data_loader(args.data_root, args.pair_modalities,
                                        batch_size=args.batch_size, split='train')
        test_loader = create_data_loader(args.data_root, args.pair_modalities,
                                       batch_size=args.batch_size, split='test')
    else:
        print("使用演示数据...")
        train_loader = create_demo_dataloader(args)
        test_loader = create_demo_dataloader(args)

    # 步骤1: 训练基础Drfuse模型
    print("\n" + "="*50)
    print("步骤1: 训练基础Drfuse模型")
    print("="*50)

    drfuse_model = Drfuse(args, hsi_channels=args.hsi_channels, radar_channels=args.radar_channels)
    drfuse_model = drfuse_model.to(device)

    # 定义损失函数和优化器
    criterion_ce = nn.CrossEntropyLoss()
    criterion_intra = IntraClassCompactnessLoss()
    optimizer = optim.Adam(drfuse_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练Drfuse模型
    best_accuracy = 0.0
    best_model_path = os.path.join(args.model_root, 'drfuse_best_model.pth')

    for epoch in range(args.train_epoch):
        # 训练阶段
        drfuse_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(train_loader, desc=f'DrFuse训练 Epoch {epoch+1}/{args.train_epoch}')

        for batch_idx, (data, labels) in enumerate(progress_bar):
            data, labels = data.to(device), labels.to(device)

            # 分离模态数据
            hsi_data = data[:, :args.hsi_channels, :, :]
            radar_data = data[:, args.hsi_channels:, :, :]

            # 前向传播
            optimizer.zero_grad()
            outputs, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = drfuse_model(hsi_data, radar_data)

            # 计算损失
            ce_loss = criterion_ce(outputs, labels)
            intra_loss = criterion_intra([f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar], labels)
            total_loss = ce_loss + intra_loss

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 统计
            train_loss += total_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # 更新进度条
            accuracy = 100 * train_correct / train_total
            progress_bar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{accuracy:.2f}%'
            })

        # 验证阶段
        if (epoch + 1) % 10 == 0:
            drfuse_model.eval()
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for data, labels in test_loader:
                    data, labels = data.to(device), labels.to(device)

                    hsi_data = data[:, :args.hsi_channels, :, :]
                    radar_data = data[:, args.hsi_channels:, :, :]

                    outputs, _, _, _, _ = drfuse_model(hsi_data, radar_data)

                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

            test_accuracy = 100 * test_correct / test_total
            print(f"验证准确率: {test_accuracy:.2f}%")

            # 保存最佳模型
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                torch.save({
                    'epoch': epoch,
                    'model_state': drfuse_model.state_dict(),
                    'accuracy': test_accuracy,
                    'args': args
                }, best_model_path)
                print(f"保存最佳模型，准确率: {test_accuracy:.2f}%")

    print(f"Drfuse模型训练完成，最佳准确率: {best_accuracy:.2f}%")

    # 步骤2: 训练FMC模块
    if args.use_fmc:
        print("\n" + "="*50)
        print("步骤2: 训练FMC模块")
        print("="*50)

        # 创建FMC训练器
        fmc_trainer = FMCTrainer(args, drfuse_model, device)

        # 准备特征数据
        print("准备特征数据...")
        feature_dataset = fmc_trainer.prepare_feature_data(train_loader)

        # 训练FMC模块
        fmc_trainer.train_fmc_module(feature_dataset, num_epochs=args.fmc_epochs)

        # 保存FMC模型
        fmc_model_path = os.path.join(args.output_dir, 'fmc_final_model.pth')
        fmc_trainer.fmc_module.save_models(fmc_model_path)

        print(f"FMC模块训练完成，模型已保存到: {fmc_model_path}")

    # 步骤3: 评估模态缺失场景
    print("\n" + "="*50)
    print("步骤3: 评估模态缺失场景")
    print("="*50)

    # 创建集成模型
    from fmc_inference import DrfuseWithFMC

    if args.use_fmc:
        # 加载训练好的FMC模块
        fmc_module = FMCModule(feature_dim=args.feature_dim, device=device)
        fmc_module.load_models(fmc_model_path)

        # 创建集成模型
        integrated_model = DrfuseWithFMC(args, drfuse_model, fmc_module, device)

        # 评估不同模态缺失场景
        scenarios = ['both', 'hsi_only', 'radar_only']
        results = integrated_model.evaluate_modality_robustness(test_loader, scenarios)

        # 打印结果
        print("\n模态缺失评估结果:")
        print("-" * 40)
        for scenario, result in results.items():
            print(f"{scenario}: 准确率={result['accuracy']:.2f}%, 损失={result['loss']:.4f}")

        # 保存集成模型
        integrated_model_path = os.path.join(args.model_root, 'drfuse_with_fmc.pth')
        integrated_model.save_model(integrated_model_path)
        print(f"\n集成模型已保存: {integrated_model_path}")

    print("\n训练完成!")


if __name__ == '__main__':
    train_drfuse_with_fmc()