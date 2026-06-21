import torch
import torch.nn as nn
import numpy as np
from fmc_module import FMCModule
from models1 import Drfuse
import os


class DrfuseWithFMC(nn.Module):
    """
    集成FMC模块的Drfuse模型
    支持模态缺失情况下的推理
    """

    def __init__(self, args, drfuse_model, fmc_module, device='cuda'):
        super(DrfuseWithFMC, self).__init__()

        self.args = args
        self.device = device
        self.drfuse_model = drfuse_model
        self.fmc_module = fmc_module

        # 特征维度
        self.feature_dim = args.feature_dim

    def forward(self, x_hs=None, x_radar=None, modality='both'):
        """
        前向传播，支持不同模态组合
        Args:
            x_hs: HSI数据 [Batch, channels, H, W]
            x_radar: 雷达数据 [Batch, channels, H, W]
            modality: 模态类型 ('hsi', 'radar', 'both')
        Returns:
            output: 分类结果
            features: 特征字典
        """
        if modality == 'both':
            # 两个模态都可用，使用原始Drfuse模型
            output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.drfuse_model(x_hs, x_radar)
            features = {
                'f_sh_hs': f_sh_hs,
                'f_sh_radar': f_sh_radar,
                'f_sp_hs': f_sp_hs,
                'f_sp_radar': f_sp_radar
            }
            return output, features

        elif modality == 'hsi':
            # 只有HSI模态，需要生成雷达模态特征
            return self.forward_hsi_only(x_hs)

        elif modality == 'radar':
            # 只有雷达模态，需要生成HSI模态特征
            return self.forward_radar_only(x_radar)

        else:
            raise ValueError(f"不支持的模态类型: {modality}")

    def forward_hsi_only(self, x_hs):
        """
        只有HSI模态时的前向传播
        Args:
            x_hs: HSI数据
        Returns:
            output: 分类结果
            features: 特征字典
        """
        # 使用Drfuse的SFD模块提取HSI特征
        f_sh_hs, _, f_sp_hs, _ = self.drfuse_model.sfd(x_hs, torch.zeros_like(x_hs[:, :1, :, :]))

        # 使用FMC模块生成雷达模态特征
        with torch.no_grad():
            generated_sh_radar, generated_sp_radar = self.fmc_module.generate_missing_modality_features(
                'hsi', f_sh_hs, f_sp_hs, 'radar'
            )

        # 使用融合模块进行分类
        output = self.drfuse_model.fusion_module(
            f_sh_hs, generated_sh_radar, f_sp_hs, generated_sp_radar
        )

        features = {
            'f_sh_hs': f_sh_hs,
            'f_sh_radar': generated_sh_radar,
            'f_sp_hs': f_sp_hs,
            'f_sp_radar': generated_sp_radar
        }

        return output, features

    def forward_radar_only(self, x_radar):
        """
        只有雷达模态时的前向传播
        Args:
            x_radar: 雷达数据
        Returns:
            output: 分类结果
            features: 特征字典
        """
        # 使用Drfuse的SFD模块提取雷达特征
        _, f_sh_radar, _, f_sp_radar = self.drfuse_model.sfd(
            torch.zeros_like(x_radar).repeat(1, self.args.hsi_channels, 1, 1), x_radar
        )

        # 使用FMC模块生成HSI模态特征
        with torch.no_grad():
            generated_sh_hs, generated_sp_hs = self.fmc_module.generate_missing_modality_features(
                'radar', f_sh_radar, f_sp_radar, 'hsi'
            )

        # 使用融合模块进行分类
        output = self.drfuse_model.fusion_module(
            generated_sh_hs, f_sh_radar, generated_sp_hs, f_sp_radar
        )

        features = {
            'f_sh_hs': generated_sh_hs,
            'f_sh_radar': f_sh_radar,
            'f_sp_hs': generated_sp_hs,
            'f_sp_radar': f_sp_radar
        }

        return output, features

    def evaluate_modality_robustness(self, test_dataloader, missing_modality_scenarios):
        """
        评估模型在不同模态缺失场景下的鲁棒性
        Args:
            test_dataloader: 测试数据加载器
            missing_modality_scenarios: 缺失模态场景列表
        Returns:
            results: 评估结果
        """
        print("开始评估模态鲁棒性...")
        results = {}

        self.eval()
        with torch.no_grad():
            for scenario in missing_modality_scenarios:
                print(f"评估场景: {scenario}")
                correct = 0
                total = 0
                total_loss = 0.0

                # 创建进度条
                progress_bar = tqdm(test_dataloader, desc=f'Scenario: {scenario}')

                for batch_idx, (data, labels) in enumerate(progress_bar):
                    batch_size = labels.size(0)
                    labels = labels.to(self.device)

                    # 根据场景准备输入数据
                    if scenario == 'both':
                        # 两个模态都可用
                        hsi_data = data[:, :self.args.hsi_channels, :, :].to(self.device)
                        radar_data = data[:, self.args.hsi_channels:, :, :].to(self.device)
                        outputs, _ = self.forward(hsi_data, radar_data, modality='both')

                    elif scenario == 'hsi_only':
                        # 只有HSI模态
                        hsi_data = data[:, :self.args.hsi_channels, :, :].to(self.device)
                        outputs, _ = self.forward(hsi_data, None, modality='hsi')

                    elif scenario == 'radar_only':
                        # 只有雷达模态
                        radar_data = data[:, self.args.hsi_channels:, :, :].to(self.device)
                        outputs, _ = self.forward(None, radar_data, modality='radar')

                    else:
                        continue

                    # 计算损失和准确率
                    loss = nn.CrossEntropyLoss()(outputs, labels)
                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # 更新进度条
                    accuracy = 100 * correct / total if total > 0 else 0
                    progress_bar.set_postfix({
                        'Accuracy': f'{accuracy:.2f}%',
                        'Loss': f'{loss.item():.4f}'
                    })

                # 计算平均指标
                avg_accuracy = 100 * correct / total if total > 0 else 0
                avg_loss = total_loss / len(test_dataloader)

                results[scenario] = {
                    'accuracy': avg_accuracy,
                    'loss': avg_loss,
                    'correct': correct,
                    'total': total
                }

                print(f"场景 {scenario} - 准确率: {avg_accuracy:.2f}%, 平均损失: {avg_loss:.4f}")

        return results

    def save_model(self, path):
        """保存集成模型"""
        torch.save({
            'drfuse_state_dict': self.drfuse_model.state_dict(),
            'fmc_state_dict': self.fmc_module.state_dict(),
            'args': self.args
        }, path)
        print(f"集成模型已保存: {path}")

    def load_model(self, path):
        """加载集成模型"""
        checkpoint = torch.load(path)
        self.drfuse_model.load_state_dict(checkpoint['drfuse_state_dict'])
        self.fmc_module.load_state_dict(checkpoint['fmc_state_dict'])
        print(f"集成模型已加载: {path}")


def create_modality_missing_test_data(args, num_samples=100):
    """
    创建模态缺失测试数据
    Args:
        args: 参数
        num_samples: 样本数量
    Returns:
        test_scenarios: 测试场景数据
    """
    print("创建模态缺失测试数据...")
    test_scenarios = {}

    # 创建完整数据
    full_data = torch.randn(num_samples, args.hsi_channels + args.radar_channels, args.image_size, args.image_size)
    labels = torch.randint(0, args.class_num, (num_samples,))

    # 创建不同缺失场景
    for scenario in ['both', 'hsi_only', 'radar_only']:
        if scenario == 'both':
            test_scenarios[scenario] = (full_data, labels)

        elif scenario == 'hsi_only':
            # 只有HSI模态 - 将雷达模态置零
            hsi_only_data = full_data.clone()
            hsi_only_data[:, args.hsi_channels:, :, :] = 0
            test_scenarios[scenario] = (hsi_only_data, labels)

        elif scenario == 'radar_only':
            # 只有雷达模态 - 将HSI模态置零
            radar_only_data = full_data.clone()
            radar_only_data[:, :args.hsi_channels, :, :] = 0
            test_scenarios[scenario] = (radar_only_data, labels)

    return test_scenarios


def main():
    """主函数 - 演示FMC推理"""
    from config import get_args

    # 获取参数
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # 创建模拟的Drfuse模型
    drfuse_model = Drfuse(args, hsi_channels=args.hsi_channels, radar_channels=args.radar_channels)

    # 创建FMC模块
    fmc_module = FMCModule(feature_dim=args.feature_dim, device=device)

    # 创建集成模型
    integrated_model = DrfuseWithFMC(args, drfuse_model, fmc_module, device)

    # 创建测试数据
    test_scenarios = create_modality_missing_test_data(args, num_samples=100)

    # 评估不同场景
    print("开始评估模态缺失场景...")

    for scenario, (data, labels) in test_scenarios.items():
        print(f"\n评估场景: {scenario}")

        # 创建简单的数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # 评估当前场景
        results = integrated_model.evaluate_modality_robustness(dataloader, [scenario])

        if scenario in results:
            result = results[scenario]
            print(f"准确率: {result['accuracy']:.2f}%")
            print(f"平均损失: {result['loss']:.4f}")

    print("\n评估完成!")


if __name__ == '__main__':
    main()