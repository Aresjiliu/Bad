import torch
import torch.nn as nn
import torch.nn.functional as F
from models1 import SFDModuleMD, SpectralSqueeze, SS_DecoupledBlock, OrientedGradientBlock


class ModalitySpecificGenerator(nn.Module):
    """
    模态特定特征生成器
    基于可用模态的共享特征生成缺失模态的特定特征
    """

    def __init__(self, shared_dim=64, specific_dim=64, target_modality='hsi'):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.target_modality = target_modality

        # 特征转换网络 - 将共享特征转换为特定特征空间
        self.feature_transform = nn.Sequential(
            nn.Conv2d(shared_dim, shared_dim * 2, 3, padding=1),
            nn.BatchNorm2d(shared_dim * 2),
            nn.ReLU(),
            nn.Conv2d(shared_dim * 2, specific_dim, 3, padding=1),
            nn.BatchNorm2d(specific_dim),
            nn.ReLU()
        )

        # 模态特定生成网络
        if target_modality == 'hsi':
            # HSI特定生成器 - 处理高光谱特征
            self.modality_generator = nn.Sequential(
                # 多尺度特征生成
                nn.Conv2d(specific_dim, specific_dim, 3, padding=1, dilation=1),
                nn.BatchNorm2d(specific_dim),
                nn.ReLU(),
                nn.Conv2d(specific_dim, specific_dim, 3, padding=2, dilation=2),
                nn.BatchNorm2d(specific_dim),
                nn.ReLU(),
                nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
                nn.BatchNorm2d(specific_dim),
                nn.ReLU(),
                # 光谱注意力机制
                SpectralSqueeze(specific_dim, ratio=8),
                nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
                nn.Tanh()  # 使用Tanh确保输出范围合适
            )
        else:  # lidar
            # LiDAR特定生成器 - 处理空间几何特征
            self.modality_generator = nn.Sequential(
                # 空间特征生成
                OrientedGradientBlock(specific_dim, specific_dim // 2),
                nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
                nn.BatchNorm2d(specific_dim),
                nn.ReLU(),
                # 空间注意力
                nn.Conv2d(specific_dim, 1, 1),
                nn.Sigmoid(),
                nn.Conv2d(specific_dim, specific_dim, 3, padding=1),
                nn.Tanh()
            )

        # 残差连接增强
        self.residual_connection = nn.Sequential(
            nn.Conv2d(specific_dim, specific_dim, 1),
            nn.BatchNorm2d(specific_dim)
        )

    def forward(self, shared_features, available_modality_mask=None):
        """
        生成缺失模态的特定特征

        Args:
            shared_features: 可用模态的共享特征 [B, shared_dim, H, W]
            available_modality_mask: 可用模态标记 [B, 2]

        Returns:
            generated_specific: 生成的特定特征 [B, specific_dim, H, W]
            generation_features: 生成过程中的中间特征（用于正则化）
        """
        # 特征空间转换
        transformed_features = self.feature_transform(shared_features)

        # 模态特定生成
        generated_features = self.modality_generator(transformed_features)

        # 残差连接
        residual = self.residual_connection(transformed_features)
        generated_specific = generated_features + residual

        # 生成过程特征（用于辅助损失计算）
        generation_features = {
            'transformed': transformed_features,
            'generated': generated_features,
            'final': generated_specific
        }

        return generated_specific, generation_features


class MultiModalGenerator(nn.Module):
    """
    多模态特征生成器
    根据可用模态生成任意缺失模态的特征
    """

    def __init__(self, shared_dim=64, specific_dim=64, num_modalities=2):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.num_modalities = num_modalities

        # 为每个模态对创建生成器
        self.generators = nn.ModuleDict({
            'hsi_from_lidar': ModalitySpecificGenerator(shared_dim, specific_dim, 'hsi'),
            'lidar_from_hsi': ModalitySpecificGenerator(shared_dim, specific_dim, 'lidar'),
            'hsi_from_both': ModalitySpecificGenerator(shared_dim, specific_dim, 'hsi'),
            'lidar_from_both': ModalitySpecificGenerator(shared_dim, specific_dim, 'lidar')
        })

        # 模态融合网络 - 当多个模态可用时融合共享特征
        self.modality_fusion = nn.Sequential(
            nn.Conv2d(shared_dim * num_modalities, shared_dim, 1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU(),
            nn.Conv2d(shared_dim, shared_dim, 3, padding=1),
            nn.BatchNorm2d(shared_dim),
            nn.ReLU()
        )

        # 模态重要性评估
        self.modality_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(shared_dim * num_modalities, shared_dim),
            nn.ReLU(),
            nn.Linear(shared_dim, num_modalities),
            nn.Softmax(dim=1)
        )

    def forward(self, shared_features_dict, modality_mask):
        """
        为缺失模态生成特征

        Args:
            shared_features_dict: 可用模态的共享特征字典
                例如: {'hsi': f_sh_hs, 'lidar': f_sh_radar}
            modality_mask: 模态存在性标记 [B, num_modalities]

        Returns:
            generated_features: 生成的特征字典
            generation_info: 生成过程信息
        """
        B, C, H, W = list(shared_features_dict.values())[0].shape
        device = list(shared_features_dict.values())[0].device

        generated_features = {}
        generation_info = {}

        # 确定可用模态
        available_modalities = []
        available_features = []

        hsi_available = modality_mask[:, 0] > 0.5
        lidar_available = modality_mask[:, 1] > 0.5

        if hsi_available.any():
            available_modalities.append('hsi')
            available_features.append(shared_features_dict['hsi'])

        if lidar_available.any():
            available_modalities.append('lidar')
            available_features.append(shared_features_dict['lidar'])

        # 融合可用模态的共享特征（如果多个模态可用）
        if len(available_features) > 1:
            # 多个模态可用，融合它们的共享特征
            concat_features = torch.cat(available_features, dim=1)
            fused_shared = self.modality_fusion(concat_features)

            # 计算模态重要性权重
            importance_weights = self.modality_importance(concat_features)

            # 加权融合
            weighted_features = []
            for i, feat in enumerate(available_features):
                weight = importance_weights[:, i:i+1].view(-1, 1, 1, 1)
                weighted_features.append(feat * weight)

            source_features = sum(weighted_features)
            generation_info['fusion_weights'] = importance_weights
            generation_info['fusion_strategy'] = 'multi_modal'

        elif len(available_features) == 1:
            # 只有一个模态可用，直接使用其共享特征
            source_features = available_features[0]
            generation_info['fusion_strategy'] = 'single_modal'
        else:
            # 没有可用模态，创建零特征
            source_features = torch.zeros(B, self.shared_dim, H, W, device=device)
            generation_info['fusion_strategy'] = 'no_modal'

        generation_info['available_modalities'] = available_modalities
        generation_info['source_features'] = source_features

        # 为缺失模态生成特征
        if not hsi_available.any() and len(available_features) > 0:
            # HSI缺失，需要生成
            if 'lidar' in available_modalities:
                generator_key = 'hsi_from_lidar' if len(available_modalities) == 1 else 'hsi_from_both'
                generated_hsi, hsi_gen_info = self.generators[generator_key](source_features)
                generated_features['hsi'] = generated_hsi
                generation_info['hsi_generation'] = hsi_gen_info

        if not lidar_available.any() and len(available_features) > 0:
            # LiDAR缺失，需要生成
            if 'hsi' in available_modalities:
                generator_key = 'lidar_from_hsi' if len(available_modalities) == 1 else 'lidar_from_both'
                generated_lidar, lidar_gen_info = self.generators[generator_key](source_features)
                generated_features['lidar'] = generated_lidar
                generation_info['lidar_generation'] = lidar_gen_info

        return generated_features, generation_info


class ModalitySpecificDiscriminator(nn.Module):
    """
    模态特定判别器
    区分真实特征和生成特征
    """

    def __init__(self, feature_dim=64, modality='hsi'):
        super().__init__()
        self.feature_dim = feature_dim
        self.modality = modality

        # 特征提取网络
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim, feature_dim * 2, 3, padding=1, stride=2),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 3, padding=1, stride=2),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2)
        )

        # 判别器头部
        self.discriminator_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, 1)
        )

        # 模态特定特征提取（可选）
        if modality == 'hsi':
            self.modality_specific = nn.Sequential(
                SpectralSqueeze(feature_dim, ratio=8),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.LeakyReLU(0.2)
            )
        else:  # lidar
            self.modality_specific = nn.Sequential(
                OrientedGradientBlock(feature_dim, feature_dim // 2),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.LeakyReLU(0.2)
            )

    def forward(self, features, return_features=False):
        """
        判别特征的真实性

        Args:
            features: 输入特征 [B, feature_dim, H, W]
            return_features: 是否返回中间特征

        Returns:
            validity: 真实性分数 [B, 1]
            intermediate_features: 中间特征（如果return_features=True）
        """
        # 模态特定处理
        modality_features = self.modality_specific(features)

        # 特征提取
        extracted_features = self.feature_extractor(modality_features)

        # 判别
        validity = self.discriminator_head(extracted_features)

        if return_features:
            return validity, extracted_features
        else:
            return validity


class MultiModalDiscriminator(nn.Module):
    """
    多模态判别器
    管理多个模态的判别器
    """

    def __init__(self, feature_dim=64, num_modalities=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # 为每个模态创建判别器
        self.discriminators = nn.ModuleDict({
            'hsi': ModalitySpecificDiscriminator(feature_dim, 'hsi'),
            'lidar': ModalitySpecificDiscriminator(feature_dim, 'lidar')
        })

        # 多模态融合判别器（可选）
        self.fusion_discriminator = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim * 2, 3, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim, 1)
        )

    def forward(self, features_dict, modality_mask, return_individual=False):
        """
        判别多个模态的特征

        Args:
            features_dict: 特征字典 {'hsi': features, 'lidar': features}
            modality_mask: 模态存在性标记 [B, num_modalities]
            return_individual: 是否返回各个判别器的结果

        Returns:
            validity_scores: 真实性分数字典
            individual_scores: 各个判别器的结果（如果return_individual=True）
        """
        validity_scores = {}
        individual_scores = {}

        # 各个模态的判别
        for modality, features in features_dict.items():
            if features is not None:
                discriminator = self.discriminators[modality]
                validity = discriminator(features)
                validity_scores[f'{modality}_individual'] = validity

                if return_individual:
                    validity_with_features = discriminator(features, return_features=True)
                    individual_scores[modality] = validity_with_features

        # 融合判别（如果多个模态都存在）
        if len(features_dict) > 1 and all(f is not None for f in features_dict.values()):
            concat_features = torch.cat(list(features_dict.values()), dim=1)
            fusion_validity = self.fusion_discriminator(concat_features)
            validity_scores['fusion'] = fusion_validity

        if return_individual:
            return validity_scores, individual_scores
        else:
            return validity_scores


class AdversarialFeatureGeneration(nn.Module):
    """
    对抗特征生成模块
    整合生成器和判别器
    """

    def __init__(self, shared_dim=64, specific_dim=64, num_modalities=2, lambda_gp=10.0):
        super().__init__()
        self.shared_dim = shared_dim
        self.specific_dim = specific_dim
        self.num_modalities = num_modalities
        self.lambda_gp = lambda_gp  # 梯度惩罚系数

        # 生成器
        self.generator = MultiModalGenerator(shared_dim, specific_dim, num_modalities)

        # 判别器
        self.discriminator = MultiModalDiscriminator(specific_dim, num_modalities)

        # 特征一致性约束
        self.feature_consistency = nn.MSELoss()

    def generate_missing_features(self, shared_features_dict, modality_mask):
        """
        为缺失模态生成特征

        Args:
            shared_features_dict: 可用模态的共享特征
            modality_mask: 模态存在性标记

        Returns:
            generated_features: 生成的特征
            generation_info: 生成信息
        """
        return self.generator(shared_features_dict, modality_mask)

    def discriminate_features(self, features_dict, modality_mask, return_individual=False):
        """
        判别特征的真实性

        Args:
            features_dict: 特征字典
            modality_mask: 模态存在性标记
            return_individual: 是否返回各个判别器的结果

        Returns:
            validity_scores: 真实性分数
        """
        return self.discriminator(features_dict, modality_mask, return_individual)

    def compute_gradient_penalty(self, real_features, generated_features, modality):
        """
        计算梯度惩罚（用于WGAN-GP）

        Args:
            real_features: 真实特征
            generated_features: 生成特征
            modality: 模态类型

        Returns:
            gradient_penalty: 梯度惩罚值
        """
        batch_size = real_features.size(0)
        device = real_features.device

        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        interpolated = alpha * real_features + (1 - alpha) * generated_features
        interpolated.requires_grad_(True)

        # 判别器对插值特征的输出
        discriminator = self.discriminator.discriminators[modality]
        validity = discriminator(interpolated)

        # 计算梯度
        grad_outputs = torch.ones_like(validity)
        gradients = torch.autograd.grad(
            outputs=validity,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def forward(self, shared_features_dict, real_specific_features_dict, modality_mask, training_mode='generator'):
        """
        对抗训练的前向传播

        Args:
            shared_features_dict: 共享特征字典
            real_specific_features_dict: 真实特定特征字典
            modality_mask: 模态存在性标记
            training_mode: 训练模式 ('generator', 'discriminator', 'both')

        Returns:
            results: 训练结果字典
        """
        results = {}

        if training_mode in ['generator', 'both']:
            # 生成缺失模态的特征
            generated_features, generation_info = self.generate_missing_features(
                shared_features_dict, modality_mask
            )
            results['generated_features'] = generated_features
            results['generation_info'] = generation_info

        if training_mode in ['discriminator', 'both']:
            # 判别器训练
            # 组合真实特征和生成特征
            combined_features = {}
            for modality in ['hsi', 'lidar']:
                if modality in real_specific_features_dict and real_specific_features_dict[modality] is not None:
                    combined_features[modality] = real_specific_features_dict[modality]
                elif 'generated_features' in results and modality in results['generated_features']:
                    combined_features[modality] = results['generated_features'][modality]

            # 判别特征
            validity_scores = self.discriminate_features(combined_features, modality_mask)
            results['validity_scores'] = validity_scores

            # 计算梯度惩罚（如果使用WGAN-GP）
            if training_mode == 'discriminator' and self.lambda_gp > 0:
                gradient_penalties = {}
                for modality, generated_feat in results['generated_features'].items():
                    if modality in real_specific_features_dict and real_specific_features_dict[modality] is not None:
                        gp = self.compute_gradient_penalty(
                            real_specific_features_dict[modality],
                            generated_feat,
                            modality
                        )
                        gradient_penalties[modality] = gp
                results['gradient_penalties'] = gradient_penalties

        return results


class DrfuseAdversarial(nn.Module):
    """
    集成对抗学习的完整DrFuse模型
    """

    def __init__(self, args, hsi_channels=144, radar_channels=1, feature_dim=64, lambda_gp=10.0):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim
        self.lambda_gp = lambda_gp

        # 基础特征分解模块（使用原始SFDModuleMD）
        self.sfd = SFDModuleMD(
            hsi_channels=hsi_channels,
            radar_channels=radar_channels,
            feature_dim=feature_dim
        )

        # 对抗特征生成模块
        self.adversarial_generation = AdversarialFeatureGeneration(
            shared_dim=feature_dim,
            specific_dim=feature_dim,
            num_modalities=2,
            lambda_gp=lambda_gp
        )

        # 分类器（与原始模型保持一致）
        self.classifier = nn.Sequential(
            nn.Conv2d(2 * feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, args.class_num)
        )

        # 模态缺失时的备用分类器
        self.single_modality_classifiers = nn.ModuleDict({
            'hsi': nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, args.class_num)
            ),
            'lidar': nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, args.class_num)
            )
        })

    def forward(self, x_hs=None, x_radar=None, modality_mask=None, training_mode='normal'):
        """
        前向传播（支持对抗训练）

        Args:
            x_hs: HSI输入
            x_radar: LiDAR输入
            modality_mask: 模态存在性标记
            training_mode: 训练模式 ('normal', 'adversarial')

        Returns:
            results: 输出结果字典
        """
        B = None
        device = None

        # 确定批次大小和设备
        for x in [x_hs, x_radar]:
            if x is not None:
                B = x.shape[0]
                device = x.device
                break

        if B is None:
            raise ValueError("至少需要提供一个模态的输入")

        # 创建模态掩码（如果不提供）
        if modality_mask is None:
            modality_mask = torch.ones(B, 2, device=device)
            if x_hs is None:
                modality_mask[:, 0] = 0
            if x_radar is None:
                modality_mask[:, 1] = 0

        # 特征分解
        if x_hs is not None and x_radar is not None:
            # 两个模态都存在
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        elif x_hs is not None:
            # 只有HSI
            dummy_radar = torch.zeros(B, 1, x_hs.shape[2], x_hs.shape[3], device=device)
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, dummy_radar)
            if modality_mask is None:
                modality_mask = torch.ones(B, 2, device=device)
                modality_mask[:, 1] = 0  # LiDAR缺失
        else:
            # 只有LiDAR
            dummy_hsi = torch.zeros(B, 144, x_radar.shape[2], x_radar.shape[3], device=device)
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(dummy_hsi, x_radar)
            if modality_mask is None:
                modality_mask = torch.ones(B, 2, device=device)
                modality_mask[:, 0] = 0  # HSI缺失

        results = {
            'f_sh_hs': f_sh_hs,
            'f_sh_radar': f_sh_radar,
            'f_sp_hs': f_sp_hs,
            'f_sp_radar': f_sp_radar,
            'modality_mask': modality_mask
        }

        if training_mode == 'adversarial':
            # 对抗训练模式
            shared_features_dict = {}
            real_specific_features_dict = {}

            if modality_mask[:, 0].mean() > 0.5:  # HSI可用
                shared_features_dict['hsi'] = f_sh_hs
                real_specific_features_dict['hsi'] = f_sp_hs

            if modality_mask[:, 1].mean() > 0.5:  # LiDAR可用
                shared_features_dict['lidar'] = f_sh_radar
                real_specific_features_dict['lidar'] = f_sp_radar

            # 执行对抗生成
            adversarial_results = self.adversarial_generation(
                shared_features_dict, real_specific_features_dict, modality_mask, training_mode='both'
            )

            results['adversarial'] = adversarial_results

            # 使用生成的特征进行分类
            if 'generated_features' in adversarial_results:
                generated_features = adversarial_results['generated_features']

                # 合并真实特征和生成特征
                if 'hsi' in generated_features:
                    f_sp_hs = generated_features['hsi']
                if 'lidar' in generated_features:
                    f_sp_radar = generated_features['lidar']

        # 分类
        hsi_available = modality_mask[:, 0:1].view(-1, 1, 1, 1)
        lidar_available = modality_mask[:, 1:2].view(-1, 1, 1, 1)

        if hsi_available.mean() > 0.5 and lidar_available.mean() > 0.5:
            # 两个模态都存在
            fused_features = torch.cat([f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar], dim=1)
            output = self.classifier(fused_features)
        elif hsi_available.mean() > 0.5:
            # 只有HSI
            if training_mode == 'adversarial' and 'generated_features' in results.get('adversarial', {}):
                # 使用生成的LiDAR特征
                fused_features = torch.cat([f_sh_hs, f_sp_hs, results['adversarial']['generated_features']['lidar']], dim=1)
                output = self.classifier(fused_features)
            else:
                # 使用HSI专用分类器
                output = self.single_modality_classifiers['hsi'](f_sp_hs)
        elif lidar_available.mean() > 0.5:
            # 只有LiDAR
            if training_mode == 'adversarial' and 'generated_features' in results.get('adversarial', {}):
                # 使用生成的HSI特征
                fused_features = torch.cat([f_sh_radar, results['adversarial']['generated_features']['hsi'], f_sp_radar], dim=1)
                output = self.classifier(fused_features)
            else:
                # 使用LiDAR专用分类器
                output = self.single_modality_classifiers['lidar'](f_sp_radar)
        else:
            # 没有可用模态（异常情况）
            output = torch.zeros(B, self.args.class_num, device=device)

        results['output'] = output

        return results

    def set_lambda_gp(self, lambda_gp):
        """设置梯度惩罚系数"""
        self.lambda_gp = lambda_gp
        self.adversarial_generation.lambda_gp = lambda_gp


def test_adversarial_generation():
    """测试对抗生成模块"""
    print("测试对抗特征生成模块...")

    import argparse

    # 创建测试参数
    args = argparse.Namespace()
    args.class_num = 15

    # 创建模型
    model = DrfuseAdversarial(
        args=args,
        hsi_channels=144,
        radar_channels=1,
        feature_dim=64,
        lambda_gp=10.0
    )

    # 测试数据
    B = 4
    device = 'cpu'
    x_hs = torch.randn(B, 144, 7, 7, device=device)
    x_radar = torch.randn(B, 1, 7, 7, device=device)

    print(f"测试完整模态（对抗训练模式）...")
    results = model(x_hs, x_radar, training_mode='adversarial')
    print(f"输出形状: {results['output'].shape}")
    print(f"是否包含对抗结果: {'adversarial' in results}")

    print(f"\n测试HSI缺失...")
    modality_mask = torch.ones(B, 2, device=device)
    modality_mask[:, 0] = 0  # HSI缺失
    results_missing = model(None, x_radar, modality_mask, training_mode='normal')
    print(f"输出形状: {results_missing['output'].shape}")

    print(f"\n测试LiDAR缺失...")
    modality_mask2 = torch.ones(B, 2, device=device)
    modality_mask2[:, 1] = 0  # LiDAR缺失
    results_missing2 = model(x_hs, None, modality_mask2, training_mode='normal')
    print(f"输出形状: {results_missing2['output'].shape}")

    print("对抗生成模块测试通过!")


if __name__ == "__main__":
    test_adversarial_generation()