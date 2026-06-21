import torch
import torch.nn as nn
import torch.nn.functional as F
from models1 import (
    SFDModuleMD, CrossAttentionFusion, DynamicWeightFusion,
    SS_DecoupledBlock, OrientedGradientBlock, SpectralSqueeze,
    scaled_dot_product_attention
)


class LogitPoolingFusion(nn.Module):
    """Logit Pooling融合方法 - 当两个模态都存在时使用logit pooling"""

    def __init__(self, feature_dim, num_classes, pooling_type='mean'):
        super().__init__()
        self.pooling_type = pooling_type
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 分类器用于生成logits
        self.classifier_sh = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )

        self.classifier_sp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask=None):
        """
        使用logit pooling进行融合

        Args:
            f_sh_hs: HSI共享特征 [B, C, H, W]
            f_sh_radar: LiDAR共享特征 [B, C, H, W]
            f_sp_hs: HSI特定特征 [B, C, H, W]
            f_sp_radar: LiDAR特定特征 [B, C, H, W]
            modality_mask: 模态存在性标记 [B, 2] - 指示HSI和LiDAR是否存在

        Returns:
            fused_features: 融合后的特征
            logits: 分类logits
        """
        B, C, H, W = f_sh_hs.shape

        if modality_mask is not None:
            hsi_exist = modality_mask[:, 0:1]  # [B, 1]
            radar_exist = modality_mask[:, 1:2]  # [B, 1]
        else:
            # 默认两个模态都存在
            hsi_exist = torch.ones(B, 1, device=f_sh_hs.device)
            radar_exist = torch.ones(B, 1, device=f_sh_hs.device)

        # 生成各模态的logits
        logits_sh_hs = self.classifier_sh(f_sh_hs)  # [B, num_classes]
        logits_sh_radar = self.classifier_sh(f_sh_radar)  # [B, num_classes]
        logits_sp_hs = self.classifier_sp(f_sp_hs)  # [B, num_classes]
        logits_sp_radar = self.classifier_sp(f_sp_radar)  # [B, num_classes]

        # Logit Pooling融合
        if self.pooling_type == 'mean':
            # 平均池化
            pooled_logits_sh = (logits_sh_hs * hsi_exist + logits_sh_radar * radar_exist) / (hsi_exist + radar_exist + 1e-8)
            pooled_logits_sp = (logits_sp_hs * hsi_exist + logits_sp_radar * radar_exist) / (hsi_exist + radar_exist + 1e-8)
        elif self.pooling_type == 'max':
            # 最大池化
            logits_sh_stack = torch.stack([logits_sh_hs * hsi_exist, logits_sh_radar * radar_exist], dim=0)
            logits_sp_stack = torch.stack([logits_sp_hs * hsi_exist, logits_sp_radar * radar_exist], dim=0)
            pooled_logits_sh, _ = torch.max(logits_sh_stack, dim=0)
            pooled_logits_sp, _ = torch.max(logits_sp_stack, dim=0)
        else:
            # 加权平均（根据模态存在性）
            weight_sum = hsi_exist + radar_exist
            pooled_logits_sh = (logits_sh_hs * hsi_exist + logits_sh_radar * radar_exist) / (weight_sum + 1e-8)
            pooled_logits_sp = (logits_sp_hs * hsi_exist + logits_sp_radar * radar_exist) / (weight_sum + 1e-8)

        # 最终融合logits
        final_logits = pooled_logits_sh + pooled_logits_sp

        # 生成融合特征（用于后续处理）
        if modality_mask is not None:
            # 根据模态存在性选择特征
            if hsi_exist.sum() > radar_exist.sum():  # HSI主导
                fused_features = torch.cat([f_sh_hs, f_sp_hs], dim=1)
            elif radar_exist.sum() > hsi_exist.sum():  # LiDAR主导
                fused_features = torch.cat([f_sh_radar, f_sp_radar], dim=1)
            else:  # 平衡融合
                fused_sh = (f_sh_hs * hsi_exist.view(-1, 1, 1, 1) + f_sh_radar * radar_exist.view(-1, 1, 1, 1)) / (weight_sum.view(-1, 1, 1, 1) + 1e-8)
                fused_sp = (f_sp_hs * hsi_exist.view(-1, 1, 1, 1) + f_sp_radar * radar_exist.view(-1, 1, 1, 1)) / (weight_sum.view(-1, 1, 1, 1) + 1e-8)
                fused_features = torch.cat([fused_sh, fused_sp], dim=1)
        else:
            fused_features = torch.cat([
                (f_sh_hs + f_sh_radar) / 2,
                (f_sp_hs + f_sp_radar) / 2
            ], dim=1)

        return fused_features, final_logits


class MissingAwareFusionClassifier(nn.Module):
    """兼容模态缺失的特征融合分类器"""

    def __init__(self, feature_dim, num_classes, fusion_strategy='adaptive'):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy

        # 基础分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(2 * feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, num_classes)
        )

        # 模态权重生成器（用于自适应融合）
        self.modality_weight_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # HSI和LiDAR的权重
            nn.Softmax(dim=1)
        )

        # 缺失模态补偿网络
        self.missing_compensation = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

    def forward(self, f_sh_hs=None, f_sh_radar=None, f_sp_hs=None, f_sp_radar=None, modality_mask=None):
        """
        前向传播，支持模态缺失情况

        Args:
            f_sh_hs: HSI共享特征
            f_sh_radar: LiDAR共享特征
            f_sp_hs: HSI特定特征
            f_sp_radar: LiDAR特定特征
            modality_mask: 模态存在性标记 [B, 2]

        Returns:
            output: 分类输出
            fused_features: 融合特征
        """
        B = None
        device = None

        # 确定批次大小和设备
        for feat in [f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar]:
            if feat is not None:
                B = feat.shape[0]
                device = feat.device
                break

        if B is None:
            raise ValueError("至少需要提供一个特征")

        # 创建模态掩码（如果不提供）
        if modality_mask is None:
            modality_mask = torch.ones(B, 2, device=device)

        hsi_exist = modality_mask[:, 0:1]  # [B, 1]
        radar_exist = modality_mask[:, 1:2]  # [B, 1]

        # 处理缺失模态的情况
        if f_sh_hs is None:
            f_sh_hs = torch.zeros(B, self.feature_dim, 7, 7, device=device)
        if f_sh_radar is None:
            f_sh_radar = torch.zeros(B, self.feature_dim, 7, 7, device=device)
        if f_sp_hs is None:
            f_sp_hs = torch.zeros(B, self.feature_dim, 7, 7, device=device)
        if f_sp_radar is None:
            f_sp_radar = torch.zeros(B, self.feature_dim, 7, 7, device=device)

        # 根据融合策略进行特征融合
        if self.fusion_strategy == 'adaptive':
            # 自适应融合权重
            concat_features = torch.cat([f_sh_hs, f_sh_radar], dim=1)
            modality_weights = self.modality_weight_gen(concat_features)  # [B, 2]

            # 应用权重
            weight_hsi = modality_weights[:, 0:1].view(-1, 1, 1, 1)
            weight_radar = modality_weights[:, 1:2].view(-1, 1, 1, 1)

            # 加权融合
            fused_sh = weight_hsi * f_sh_hs + weight_radar * f_sh_radar
            fused_sp = weight_hsi * f_sp_hs + weight_radar * f_sp_radar

        elif self.fusion_strategy == 'mean':
            # 简单平均融合
            weight_sum = hsi_exist.view(-1, 1, 1, 1) + radar_exist.view(-1, 1, 1, 1)
            fused_sh = (f_sh_hs * hsi_exist.view(-1, 1, 1, 1) + f_sh_radar * radar_exist.view(-1, 1, 1, 1)) / (weight_sum + 1e-8)
            fused_sp = (f_sp_hs * hsi_exist.view(-1, 1, 1, 1) + f_sp_radar * radar_exist.view(-1, 1, 1, 1)) / (weight_sum + 1e-8)

        elif self.fusion_strategy == 'missing_aware':
            # 缺失感知融合：优先使用可用模态
            if hsi_exist.sum() > radar_exist.sum():
                fused_sh = f_sh_hs
                fused_sp = f_sp_hs
            elif radar_exist.sum() > hsi_exist.sum():
                fused_sh = f_sh_radar
                fused_sp = f_sp_radar
            else:
                # 两个模态都存在，使用平均融合
                fused_sh = (f_sh_hs + f_sh_radar) / 2
                fused_sp = (f_sp_hs + f_sp_radar) / 2
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # 拼接共享和特定特征
        fused_features = torch.cat([fused_sh, fused_sp], dim=1)

        # 分类
        output = self.classifier(fused_features)

        return output, fused_features


class SFDModuleMD_MissingAware(nn.Module):
    """支持模态缺失的特征分解模块"""

    def __init__(self, hsi_channels=144, radar_channels=1, feature_dim=64):
        super().__init__()

        # HSI分支
        self.hs_stream = nn.Sequential(
            SS_DecoupledBlock(hsi_channels, 64),
            SS_DecoupledBlock(64, feature_dim)
        )

        # LiDAR分支
        self.radar_stream = nn.Sequential(
            OrientedGradientBlock(radar_channels, 64),
            OrientedGradientBlock(64, feature_dim)
        )

        # 共享特征提取器
        self.E_sh_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])
        )

        self.E_sh_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])
        )

        # 特定特征提取器
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])
        )

        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])
        )

        # 模态缺失时的特征生成器
        self.missing_feature_generator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

    def forward(self, x_hs=None, x_radar=None, modality_mask=None):
        """
        支持模态缺失的前向传播

        Args:
            x_hs: HSI输入
            x_radar: LiDAR输入
            modality_mask: 模态存在性标记 [B, 2]

        Returns:
            tuple: (f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask)
        """
        B = None
        device = None

        # 确定批次大小和设备
        if x_hs is not None:
            B = x_hs.shape[0]
            device = x_hs.device
        elif x_radar is not None:
            B = x_radar.shape[0]
            device = x_radar.device
        else:
            raise ValueError("至少需要提供一个模态的输入")

        # 创建模态掩码（如果不提供）
        if modality_mask is None:
            modality_mask = torch.ones(B, 2, device=device)
            if x_hs is None:
                modality_mask[:, 0] = 0
            if x_radar is None:
                modality_mask[:, 1] = 0

        # 特征提取和分解
        if x_hs is not None:
            f_hs = self.hs_stream(x_hs)
            f_sh_hs = self.E_sh_hs(f_hs)
            f_sp_hs = self.E_sp_hs(f_hs)
        else:
            # HSI缺失，生成零特征或使用生成器
            f_sh_hs = torch.zeros(B, 64, 7, 7, device=device)
            f_sp_hs = torch.zeros(B, 64, 7, 7, device=device)

        if x_radar is not None:
            f_radar = self.radar_stream(x_radar)
            f_sh_radar = self.E_sh_radar(f_radar)
            f_sp_radar = self.E_sp_radar(f_radar)
        else:
            # LiDAR缺失，生成零特征或使用生成器
            f_sh_radar = torch.zeros(B, 64, 7, 7, device=device)
            f_sp_radar = torch.zeros(B, 64, 7, 7, device=device)

        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask


class Drfuse1(nn.Module):
    """兼容模态缺失的DrFuse模型"""

    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256, fusion_strategy='adaptive'):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy

        # 特征分解模块（支持模态缺失）
        self.sfd = SFDModuleMD_MissingAware(
            hsi_channels=hsi_channels,
            radar_channels=radar_channels,
            feature_dim=feature_dim
        )

        # Logit Pooling融合模块
        self.logit_pooling = LogitPoolingFusion(
            feature_dim=feature_dim,
            num_classes=args.class_num,
            pooling_type='mean'
        )

        # 传统特征融合模块（支持模态缺失）
        self.fusion_module = MissingAwareFusionClassifier(
            feature_dim=feature_dim,
            num_classes=args.class_num,
            fusion_strategy=fusion_strategy
        )

        # 融合方法选择
        self.use_logit_pooling = False  # 可以在配置中设置

    def forward(self, x_hs=None, x_radar=None, modality_mask=None, use_logit_pooling=None):
        """
        Drfuse1前向传播 - 支持模态缺失

        Args:
            x_hs: HSI输入
            x_radar: LiDAR输入
            modality_mask: 模态存在性标记 [B, 2]
            use_logit_pooling: 是否使用logit pooling（覆盖默认设置）

        Returns:
            tuple: (output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask)
        """
        # 特征分解
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask = self.sfd(
            x_hs, x_radar, modality_mask
        )

        # 确定使用哪种融合方法
        if use_logit_pooling is None:
            use_logit_pooling = self.use_logit_pooling

        if use_logit_pooling:
            # 使用Logit Pooling融合
            fused_features, output = self.logit_pooling(
                f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask
            )
        else:
            # 使用传统特征融合
            output, fused_features = self.fusion_module(
                f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask
            )

        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask

    def set_fusion_strategy(self, strategy):
        """设置融合策略"""
        self.fusion_strategy = strategy
        self.fusion_module.fusion_strategy = strategy

    def enable_logit_pooling(self, enable=True):
        """启用/禁用logit pooling"""
        self.use_logit_pooling = enable


def test_drfuse1():
    """测试Drfuse1模型"""
    import argparse

    # 创建测试参数
    args = argparse.Namespace()
    args.class_num = 15

    # 创建模型
    model = Drfuse1(args, hsi_channels=144, radar_channels=1, feature_dim=64)

    # 测试数据
    B = 4
    x_hs = torch.randn(B, 144, 7, 7)
    x_radar = torch.randn(B, 1, 7, 7)

    # 测试完整模态
    output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, mask = model(x_hs, x_radar)
    print(f"完整模态 - 输出形状: {output.shape}")

    # 测试HSI缺失
    output_missing, _, _, _, _, mask_missing = model(None, x_radar)
    print(f"HSI缺失 - 输出形状: {output_missing.shape}")

    # 测试LiDAR缺失
    output_missing2, _, _, _, _, mask_missing2 = model(x_hs, None)
    print(f"LiDAR缺失 - 输出形状: {output_missing2.shape}")

    # 测试logit pooling
    model.enable_logit_pooling(True)
    output_pooling, _, _, _, _, _ = model(x_hs, x_radar)
    print(f"Logit Pooling - 输出形状: {output_pooling.shape}")

    print("Drfuse1模型测试通过!")


if __name__ == "__main__":
    test_drfuse1()