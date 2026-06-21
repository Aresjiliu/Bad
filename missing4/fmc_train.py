import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import datetime

from fmc_module import FMCModule
from models1 import Drfuse
from config import get_args


class FMCTrainer:
    """
    FMC对抗学习训练器
    用于训练特征映射和补全模块
    """

    def __init__(self, args, drfuse_model, device='cuda'):
        self.args = args
        self.device = device
        self.drfuse_model = drfuse_model

        # 初始化FMC模块
        self.fmc_module = FMCModule(feature_dim=args.feature_dim, device=device)

        # 训练历史
        self.train_history = {
            'd_losses': [],
            'g_losses': [],
            'recon_losses': [],
            'adv_losses': []
        }

    def prepare_feature_data(self, dataloader):
        """
        准备特征数据用于FMC训练
        Args:
            dataloader: 数据加载器
        Returns:
            feature_dataset: 特征数据集
        """
        print("准备特征数据...")
        feature_dataset = []

        self.drfuse_model.eval()
        with torch.no_grad():
            for batch_idx, (data, labels) in enumerate(tqdm(dataloader)):
                # 分离模态数据
                if self.args.pair_modalities == 'hsi+lidar':
                    hsi_data = data[:, :self.args.hsi_channels, :, :].to(self.device)
                    radar_data = data[:, self.args.hsi_channels:, :, :].to(self.device)
                else:
                    # 根据实际数据格式调整
                    hsi_data = data[0].to(self.device)
                    radar_data = data[1].to(self.device)

                # 提取特征
                output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.drfuse_model(hsi_data, radar_data)

                # 保存特征数据
                feature_data = {
                    'f_sh_hs': f_sh_hs.cpu(),
                    'f_sh_radar': f_sh_radar.cpu(),
                    'f_sp_hs': f_sp_hs.cpu(),
                    'f_sp_radar': f_sp_radar.cpu(),
                    'labels': labels
                }
                feature_dataset.append(feature_data)

                # 限制数据量
                if batch_idx >= 100:  # 限制为100个批次
                    break

        return feature_dataset

    def train_fmc_module(self, feature_dataset, num_epochs=100):
        """
        训练FMC模块
        Args:
            feature_dataset: 特征数据集
            num_epochs: 训练轮数
        """
        print(f"开始FMC模块训练，共{num_epochs}轮...")

        # 创建特征数据加载器
        batch_size = self.args.batch_size

        for epoch in range(num_epochs):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_adv_loss = 0.0

            # 随机打乱数据
            np.random.shuffle(feature_dataset)

            # 分批训练
            num_batches = len(feature_dataset) // batch_size
            progress_bar = tqdm(range(num_batches), desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx in progress_bar:
                # 获取批次数据
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(feature_dataset))

                batch_data = feature_dataset[start_idx:end_idx]

                # 准备批次特征
                real_features = {
                    'f_sh_hs': torch.cat([data['f_sh_hs'] for data in batch_data], dim=0).to(self.device),
                    'f_sh_radar': torch.cat([data['f_sh_radar'] for data in batch_data], dim=0).to(self.device),
                    'f_sp_hs': torch.cat([data['f_sp_hs'] for data in batch_data], dim=0).to(self.device),
                    'f_sp_radar': torch.cat([data['f_sp_radar'] for data in batch_data], dim=0).to(self.device)
                }

                # 执行训练步骤
                losses = self.fmc_module.train_step(real_features, len(batch_data))

                # 累计损失
                epoch_d_loss += losses['d_loss']
                epoch_g_loss += losses['g_loss']
                epoch_recon_loss += (losses['recon_hs'] + losses['recon_radar']) / 2
                epoch_adv_loss += (losses['g_hs'] + losses['g_radar']) / 2

                # 更新进度条
                progress_bar.set_postfix({
                    'D_loss': f"{losses['d_loss']:.4f}",
                    'G_loss': f"{losses['g_loss']:.4f}",
                    'Recon': f"{(losses['recon_hs'] + losses['recon_radar']) / 2:.4f}"
                })

            # 计算平均损失
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            avg_recon_loss = epoch_recon_loss / num_batches
            avg_adv_loss = epoch_adv_loss / num_batches

            # 保存历史
            self.train_history['d_losses'].append(avg_d_loss)
            self.train_history['g_losses'].append(avg_g_loss)
            self.train_history['recon_losses'].append(avg_recon_loss)
            self.train_history['adv_losses'].append(avg_adv_loss)

            # 打印训练信息
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                      f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
                      f"Recon: {avg_recon_loss:.4f}, Adv: {avg_adv_loss:.4f}")

            # 保存检查点
            if (epoch + 1) % 50 == 0:
                checkpoint_path = os.path.join(self.args.output_dir, f'fmc_checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_path, epoch + 1)

        print("FMC模块训练完成!")

    def test_modality_completion(self, test_features, missing_modality):
        """
        测试模态补全效果
        Args:
            test_features: 测试特征
            missing_modality: 缺失模态 ('hsi' 或 'radar')
        Returns:
            completed_features: 补全后的特征
        """
        self.fmc_module.eval()
        with torch.no_grad():
            if missing_modality == 'radar':
                # HSI模态可用，生成雷达模态
                available_shared = test_features['f_sh_hs'].to(self.device)
                available_specific = test_features['f_sp_hs'].to(self.device)
                generated_shared, generated_specific = self.fmc_module.generate_missing_modality_features(
                    'hsi', available_shared, available_specific, 'radar'
                )

                completed_features = {
                    'f_sh_hs': test_features['f_sh_hs'],
                    'f_sh_radar': generated_shared.cpu(),
                    'f_sp_hs': test_features['f_sp_hs'],
                    'f_sp_radar': generated_specific.cpu()
                }

            elif missing_modality == 'hsi':
                # 雷达模态可用，生成HSI模态
                available_shared = test_features['f_sh_radar'].to(self.device)
                available_specific = test_features['f_sp_radar'].to(self.device)
                generated_shared, generated_specific = self.fmc_module.generate_missing_modality_features(
                    'radar', available_shared, available_specific, 'hsi'
                )

                completed_features = {
                    'f_sh_hs': generated_shared.cpu(),
                    'f_sh_radar': test_features['f_sh_radar'],
                    'f_sp_hs': generated_specific.cpu(),
                    'f_sp_radar': test_features['f_sp_radar']
                }

            else:
                raise ValueError(f"不支持的缺失模态: {missing_modality}")

        return completed_features

    def evaluate_completion_quality(self, original_features, completed_features, missing_modality):
        """
        评估补全质量
        Args:
            original_features: 原始完整特征
            completed_features: 补全后的特征
            missing_modality: 缺失模态
        Returns:
            metrics: 评估指标
        """
        metrics = {}

        if missing_modality == 'radar':
            # 计算雷达特征的相似度
            original_radar_shared = original_features['f_sh_radar']
            original_radar_specific = original_features['f_sp_radar']
            completed_radar_shared = completed_features['f_sh_radar']
            completed_radar_specific = completed_features['f_sp_radar']

            # L2距离
            shared_diff = torch.norm(original_radar_shared - completed_radar_shared, p=2)
            specific_diff = torch.norm(original_radar_specific - completed_radar_specific, p=2)

            metrics['shared_l2_distance'] = shared_diff.item()
            metrics['specific_l2_distance'] = specific_diff.item()

            # 余弦相似度
            shared_cosine = F.cosine_similarity(
                original_radar_shared.flatten(1), completed_radar_shared.flatten(1), dim=1
            ).mean()
            specific_cosine = F.cosine_similarity(
                original_radar_specific.flatten(1), completed_radar_specific.flatten(1), dim=1
            ).mean()

            metrics['shared_cosine_similarity'] = shared_cosine.item()
            metrics['specific_cosine_similarity'] = specific_cosine.item()

        elif missing_modality == 'hsi':
            # 计算HSI特征的相似度
            original_hs_shared = original_features['f_sh_hs']
            original_hs_specific = original_features['f_sp_hs']
            completed_hs_shared = completed_features['f_sh_hs']
            completed_hs_specific = completed_features['f_sp_hs']

            # L2距离
            shared_diff = torch.norm(original_hs_shared - completed_hs_shared, p=2)
            specific_diff = torch.norm(original_hs_specific - completed_hs_specific, p=2)

            metrics['shared_l2_distance'] = shared_diff.item()
            metrics['specific_l2_distance'] = specific_diff.item()

            # 余弦相似度
            shared_cosine = F.cosine_similarity(
                original_hs_shared.flatten(1), completed_hs_shared.flatten(1), dim=1
            ).mean()
            specific_cosine = F.cosine_similarity(
                original_hs_specific.flatten(1), completed_hs_specific.flatten(1), dim=1
            ).mean()

            metrics['shared_cosine_similarity'] = shared_cosine.item()
            metrics['specific_cosine_similarity'] = specific_cosine.item()

        return metrics

    def save_checkpoint(self, path, epoch):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'fmc_module_state': self.fmc_module.state_dict(),
            'train_history': self.train_history,
            'args': self.args
        }
        torch.save(checkpoint, path)
        print(f"检查点已保存: {path}")

    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.fmc_module.load_state_dict(checkpoint['fmc_module_state'])
        self.train_history = checkpoint['train_history']
        print(f"检查点已加载: {path}")
        return checkpoint['epoch']


def main():
    """主函数"""
    # 获取参数
    args = get_args()

    # 设置设备
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载预训练的Drfuse模型
    print("加载预训练的Drfuse模型...")
    drfuse_model = Drfuse(args, hsi_channels=args.hsi_channels, radar_channels=args.radar_channels)

    # 加载模型权重
    model_path = os.path.join(args.model_dir, 'drfuse_best_model.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        drfuse_model.load_state_dict(checkpoint['model_state'])
        print(f"已加载Drfuse模型: {model_path}")
    else:
        print("警告: 未找到预训练模型，将使用随机初始化的模型")

    drfuse_model = drfuse_model.to(device)

    # 创建FMC训练器
    fmc_trainer = FMCTrainer(args, drfuse_model, device)

    # 这里需要添加数据加载器的实现
    # 由于数据加载器依赖外部模块，这里提供伪数据用于演示
    print("准备训练数据...")
    # feature_dataset = fmc_trainer.prepare_feature_data(dataloader)

    # 创建模拟数据进行演示
    feature_dataset = create_demo_feature_data(args, device, num_samples=1000)

    # 训练FMC模块
    fmc_trainer.train_fmc_module(feature_dataset, num_epochs=args.fmc_epochs)

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, 'fmc_final_model.pth')
    fmc_trainer.save_checkpoint(final_model_path, args.fmc_epochs)

    print("FMC模块训练完成!")


def create_demo_feature_data(args, device, num_samples=1000):
    """创建演示用的特征数据"""
    print("创建演示特征数据...")
    feature_dataset = []

    for i in range(num_samples // args.batch_size):
        # 创建模拟特征数据
        batch_data = {
            'f_sh_hs': torch.randn(args.batch_size, args.feature_dim, 7, 7),
            'f_sh_radar': torch.randn(args.batch_size, args.feature_dim, 7, 7),
            'f_sp_hs': torch.randn(args.batch_size, args.feature_dim, 7, 7),
            'f_sp_radar': torch.randn(args.batch_size, args.feature_dim, 7, 7),
            'labels': torch.randint(0, args.class_num, (args.batch_size,))
        }
        feature_dataset.append(batch_data)

    return feature_dataset


if __name__ == '__main__':
    main()