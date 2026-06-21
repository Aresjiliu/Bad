import torch
import torch.nn as nn
import torch.nn.functional as F
from models1 import (
    SFDModuleMD, CrossAttentionFusion, DynamicWeightFusion,
    scaled_dot_product_attention
)


class AdaptiveCrossAttentionFusion(nn.Module):
    """自适应交叉注意力融合 - 支持模态缺失"""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"

        # 查询、键、值投影
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

        # 模态存在性编码
        self.modality_embedding = nn.Embedding(3, dim)  # 0:缺失, 1:HSI, 2:LiDAR

        # 层归一化
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, feat_a, feat_b, modality_mask_a=None, modality_mask_b=None):
        """
        自适应交叉注意力融合

        Args:
            feat_a: 模态A特征 [B, C, H, W]
            feat_b: 模态B特征 [B, C, H, W]
            modality_mask_a: 模态A存在性标记 [B]
            modality_mask_b: 模态B存在性标记 [B]

        Returns:
            fused_feat: 融合特征
            attention_weights: 注意力权重
        """
        B, C, H, W = feat_a.shape
        device = feat_a.device

        # 如果没有提供模态掩码，假设两个模态都存在
        if modality_mask_a is None:
            modality_mask_a = torch.ones(B, device=device, dtype=torch.long)
        if modality_mask_b is None:
            modality_mask_b = torch.ones(B, device=device, dtype=torch.long)

        # 添加模态存在性编码
        modality_emb_a = self.modality_embedding(modality_mask_a).view(B, C, 1, 1)
        modality_emb_b = self.modality_embedding(modality_mask_b).view(B, C, 1, 1)

        feat_a_emb = feat_a + modality_emb_a
        feat_b_emb = feat_b + modality_emb_b

        # 投影到查询、键、值
        q = self.q_proj(feat_a_emb)
        k = self.k_proj(feat_b_emb)
        v = self.v_proj(feat_b_emb)

        # 多头注意力
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        # 计算注意力
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v)

        # 重塑回原始形状
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        attn_output = self.out_proj(attn_output)

        # 残差连接和层归一化
        fused_feat = self.norm1(feat_a + attn_output)

        return fused_feat, attn_weights


class DynamicMissingAwareFusion(nn.Module):
    """动态缺失感知融合模块"""

    def __init__(self, feature_dim, num_modalities=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_modalities = num_modalities

        # 模态重要性评估网络
        self.modality_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, num_modalities),
            nn.Softmax(dim=1)
        )

        # 缺失模态补偿网络
        self.missing_compensation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU()
            ) for _ in range(num_modalities)
        ])

        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Conv2d(feature_dim * num_modalities, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

    def forward(self, features, modality_mask=None):
        """
        动态缺失感知融合

        Args:
            features: 特征列表 [feat1, feat2, ...]
            modality_mask: 模态存在性标记 [B, num_modalities]

        Returns:
            fused_feature: 融合特征
            modality_weights: 模态权重
        """
        B = features[0].shape[0]
        device = features[0].device
        num_modalities = len(features)

        # 如果没有提供模态掩码，假设所有模态都存在
        if modality_mask is None:
            modality_mask = torch.ones(B, num_modalities, device=device)

        # 处理缺失模态
        processed_features = []
        for i, feat in enumerate(features):
            if feat is None:
                # 如果特征缺失，使用补偿网络生成特征
                if i < len(self.missing_compensation):
                    # 使用其他可用特征作为输入来生成缺失特征
                    available_features = [f for f in features if f is not None]
                    if available_features:
                        # 使用第一个可用特征作为输入
                        base_feat = available_features[0]
                        comp_feat = self.missing_compensation[i](base_feat)
                        processed_features.append(comp_feat)
                    else:
                        # 如果所有特征都缺失，创建零特征
                        zero_feat = torch.zeros(B, self.feature_dim, 7, 7, device=device)
                        processed_features.append(zero_feat)
                else:
                    zero_feat = torch.zeros(B, self.feature_dim, 7, 7, device=device)
                    processed_features.append(zero_feat)
            else:
                processed_features.append(feat)

        # 计算模态重要性权重
        modality_weights = []
        for feat in processed_features:
            weight = self.modality_importance(feat)  # [B, num_modalities]
            modality_weights.append(weight)

        # 合并权重并应用模态掩码
        if modality_weights:
            combined_weights = torch.stack(modality_weights, dim=0).mean(dim=0)  # [B, num_modalities]
            combined_weights = combined_weights * modality_mask
            # 重新归一化权重
            weight_sum = combined_weights.sum(dim=1, keepdim=True) + 1e-8
            combined_weights = combined_weights / weight_sum
        else:
            combined_weights = modality_mask / (modality_mask.sum(dim=1, keepdim=True) + 1e-8)

        # 应用权重并融合特征
        weighted_features = []
        for i, feat in enumerate(processed_features):
            weight = combined_weights[:, i:i+1].view(-1, 1, 1, 1)
            weighted_feat = feat * weight
            weighted_features.append(weighted_feat)

        # 拼接所有加权特征
        concat_features = torch.cat(weighted_features, dim=1)

        # 通过融合网络
        fused_feature = self.fusion_network(concat_features)

        return fused_feature, combined_weights


class CozeFusionClassifierV2(nn.Module):
    """改进版Coze融合分类器 - 支持模态缺失"""

    def __init__(self, sh_dim, sp_dim, num_classes, fusion_strategy='adaptive'):
        super().__init__()
        self.sh_dim = sh_dim
        self.sp_dim = sp_dim
        self.num_classes = num_classes
        self.fusion_strategy = fusion_strategy

        # 共享特征融合（支持模态缺失）
        self.shared_fusion = AdaptiveCrossAttentionFusion(sh_dim)

        # 特定特征融合（支持模态缺失）
        self.specific_fusion = DynamicMissingAwareFusion(sp_dim, num_modalities=2)

        # 缺失感知分类器
        self.classifier = nn.Sequential(
            nn.Conv2d(2 * sh_dim, sh_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sh_dim, num_classes)
        )

        # 模态存在性检测器
        self.modality_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sh_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # HSI和LiDAR存在性
            nn.Sigmoid()
        )

    def forward(self, f_sh_hs=None, f_sh_radar=None, f_sp_hs=None, f_sp_radar=None, modality_mask=None):
        """
        V2版本前向传播 - 支持模态缺失

        Args:
            f_sh_hs: HSI共享特征
            f_sh_radar: LiDAR共享特征
            f_sp_hs: HSI特定特征
            f_sp_radar: LiDAR特定特征
            modality_mask: 模态存在性标记 [B, 2]

        Returns:
            output: 分类输出
            fused_features: 融合特征
            modality_info: 模态信息字典
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
            if f_sh_hs is None:
                modality_mask[:, 0] = 0
            if f_sh_radar is None:
                modality_mask[:, 1] = 0

        # 处理缺失模态的情况（用零特征填充）
        if f_sh_hs is None:
            f_sh_hs = torch.zeros(B, self.sh_dim, 7, 7, device=device)
        if f_sh_radar is None:
            f_sh_radar = torch.zeros(B, self.sh_dim, 7, 7, device=device)
        if f_sp_hs is None:
            f_sp_hs = torch.zeros(B, self.sp_dim, 7, 7, device=device)
        if f_sp_radar is None:
            f_sp_radar = torch.zeros(B, self.sp_dim, 7, 7, device=device)

        # 检测模态存在性
        concat_sh_features = torch.cat([f_sh_hs, f_sh_radar], dim=1)
        detected_modality = self.modality_detector(concat_sh_features)

        # 融合共享特征
        if self.fusion_strategy == 'adaptive':
            # 自适应交叉注意力融合
            hsi_exist_mask = (modality_mask[:, 0] > 0.5).long()
            radar_exist_mask = (modality_mask[:, 1] > 0.5).long()

            # 双向交叉注意力融合
            fused_sh_hs, attn_hs = self.shared_fusion(f_sh_hs, f_sh_radar, hsi_exist_mask, radar_exist_mask)
            fused_sh_radar, attn_radar = self.shared_fusion(f_sh_radar, f_sh_hs, radar_exist_mask, hsi_exist_mask)

            # 平均融合两个方向的注意力结果
            fused_sh = (fused_sh_hs + fused_sh_radar) / 2

        elif self.fusion_strategy == 'mean':
            # 简单平均融合
            hsi_weight = modality_mask[:, 0:1].view(-1, 1, 1, 1)
            radar_weight = modality_mask[:, 1:2].view(-1, 1, 1, 1)
            weight_sum = hsi_weight + radar_weight + 1e-8

            fused_sh = (f_sh_hs * hsi_weight + f_sh_radar * radar_weight) / weight_sum

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

        # 融合特定特征
        sp_features = [f_sp_hs, f_sp_radar]
        fused_sp, modality_weights = self.specific_fusion(sp_features, modality_mask)

        # 拼接融合后的特征
        fused_features = torch.cat([fused_sh, fused_sp], dim=1)

        # 分类
        output = self.classifier(fused_features)

        # 模态信息
        modality_info = {
            'detected_modality': detected_modality,
            'input_modality_mask': modality_mask,
            'modality_weights': modality_weights,
            'fusion_strategy': self.fusion_strategy
        }

        return output, fused_features, modality_info


class MdfuseV2(nn.Module):
    """改进版Mdfuse模型 - 支持模态缺失"""

    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256, fusion_strategy='adaptive'):
        super().__init__()
        self.feature_dim = feature_dim
        self.fusion_strategy = fusion_strategy

        # 特征分解模块（使用原始的SFDModuleMD）
        self.sfd = SFDModuleMD(
            hsi_channels=hsi_channels,
            radar_channels=radar_channels,
            feature_dim=feature_dim
        )

        # 改进版融合分类器（支持模态缺失）
        self.fusion_module = CozeFusionClassifierV2(
            sh_dim=feature_dim,
            sp_dim=feature_dim,
            num_classes=args.class_num,
            fusion_strategy=fusion_strategy
        )

        # 跨模态注意力机制（可选）
        self.cross_attention = CrossAttentionFusion(feature_dim)

    def forward(self, x_hs=None, x_radar=None, modality_mask=None):
        """
        V2版本前向传播 - 支持模态缺失

        Args:
            x_hs: HSI输入
            x_radar: LiDAR输入
            modality_mask: 模态存在性标记

        Returns:
            tuple: (output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_info)
        """
        # 处理缺失模态
        if x_hs is None and x_radar is None:
            raise ValueError("至少需要提供一个模态的输入")

        # 特征分解
        if x_hs is not None and x_radar is not None:
            # 两个模态都存在
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        elif x_hs is not None:
            # 只有HSI
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, torch.zeros_like(x_hs)[:, :1, :, :])
            if modality_mask is None:
                B = x_hs.shape[0]
                modality_mask = torch.ones(B, 2, device=x_hs.device)
                modality_mask[:, 1] = 0  # LiDAR缺失
        else:
            # 只有LiDAR
            dummy_hsi = torch.zeros(x_radar.shape[0], 144, x_radar.shape[2], x_radar.shape[3], device=x_radar.device)
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(dummy_hsi, x_radar)
            if modality_mask is None:
                B = x_radar.shape[0]
                modality_mask = torch.ones(B, 2, device=x_radar.device)
                modality_mask[:, 0] = 0  # HSI缺失

        # 应用跨模态注意力（如果两个模态都存在）
        if x_hs is not None and x_radar is not None:
            f_sh_hs_att = self.cross_attention(f_sh_hs, f_sh_radar)
            f_sh_radar_att = self.cross_attention(f_sh_radar, f_sh_hs)
            f_sh_hs, f_sh_radar = f_sh_hs_att, f_sh_radar_att

        # 融合和分类（支持模态缺失）
        output, fused_features, modality_info = self.fusion_module(
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_mask
        )

        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, modality_info

    def set_fusion_strategy(self, strategy):
        """设置融合策略"""
        self.fusion_strategy = strategy
        self.fusion_module.fusion_strategy = strategy


def test_fusion_v2():
    """测试融合模块V2"""
    import argparse

    # 创建测试参数
    args = argparse.Namespace()
    args.class_num = 15

    print("测试CozeFusionClassifierV2...")

    # 创建融合分类器
    fusion_classifier = CozeFusionClassifierV2(
        sh_dim=64, sp_dim=64, num_classes=15, fusion_strategy='adaptive'
    )

    # 测试数据
    B = 4
    f_sh_hs = torch.randn(B, 64, 7, 7)
    f_sh_radar = torch.randn(B, 64, 7, 7)
    f_sp_hs = torch.randn(B, 64, 7, 7)
    f_sp_radar = torch.randn(B, 64, 7, 7)

    # 测试完整模态
    output, fused_features, modality_info = fusion_classifier(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
    print(f"完整模态 - 输出形状: {output.shape}, 融合特征形状: {fused_features.shape}")

    # 测试HSI缺失
    output_missing, _, _ = fusion_classifier(None, f_sh_radar, None, f_sp_radar)
    print(f"HSI缺失 - 输出形状: {output_missing.shape}")

    # 测试MdfuseV2
    print("\n测试MdfuseV2...")
    model = MdfuseV2(args, hsi_channels=144, radar_channels=1, feature_dim=64, fusion_strategy='adaptive')

    # 测试完整模态
    x_hs = torch.randn(B, 144, 7, 7)
    x_radar = torch.randn(B, 1, 7, 7)
    output, _, _, _, _, info = model(x_hs, x_radar)
    print(f"完整模态 - 输出形状: {output.shape}")

    # 测试缺失模态
    output_missing, _, _, _, _, info = model(x_hs, None)
    print(f"LiDAR缺失 - 输出形状: {output_missing.shape}")

    print("融合模块V2测试通过!")


if __name__ == "__main__":
    test_fusion_v2()