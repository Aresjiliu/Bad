import torch
import torch.nn as nn
import torch.nn.functional as F

mdmb_seed = 7


class SpectralSqueeze(nn.Module):
    """光谱压缩模块（动态抑制冗余波段）"""

    def __init__(self, in_ch=144, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_ch // ratio, in_ch, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y  # 通道加权 (B,C,7,7)


class SS_DecoupledBlock(nn.Module):
    """光谱-空间解耦卷积"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 光谱压缩 → 空间卷积
        self.ss_block = nn.Sequential(
            SpectralSqueeze(in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=(1, 3), padding=(0, 1)),  # 光谱压缩
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=(3, 1), padding=(1, 0)),  # 空间卷积
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        return self.ss_block(x)


class OrientedGradientBlock(nn.Module):
    """方向梯度增强模块"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 多方向梯度卷积核
        self.conv_h = nn.Conv2d(in_ch, out_ch // 2, kernel_size=(1, 3), padding=(0, 1))  # 水平
        self.conv_v = nn.Conv2d(in_ch, out_ch // 2, kernel_size=(3, 1), padding=(1, 0))  # 垂直
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU()

    def forward(self, x):
        x_h = self.conv_h(x)  # 水平响应 (B,C/2,7,7)
        x_v = self.conv_v(x)  # 垂直响应 (B,C/2,7,7)
        x = torch.cat([x_h, x_v], dim=1)  # (B,C,7,7)
        return self.act(self.bn(x))


class SFDModule(nn.Module):
    def __init__(self, args, hsi_channels=144, radar_channels=1, feature_dim=64):
        super().__init__()

        # 高光谱分支 (144通道)
        self.hs_stream = nn.Sequential(
            SS_DecoupledBlock(hsi_channels, 64),
            SS_DecoupledBlock(64, feature_dim)
        )

        # 雷达分支 (1通道)
        self.radar_stream = nn.Sequential(
            OrientedGradientBlock(radar_channels, 64),
            OrientedGradientBlock(64, feature_dim))

        # 共享特征提取器 (参数共享)
        self.E_sh_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        # 共享特征提取器 (参数共享)
        self.E_sh_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )


    def forward(self, x_hs, x_radar):
        # 基础特征提·
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解
        f_sh_hs = self.E_sh_hs(f_hs)
        f_sh_radar = self.E_sh_radar(f_radar)

        f_sp_hs = self.E_sp_hs(f_hs)
        f_sp_radar = self.E_sp_radar(f_radar)

        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


class HierarchicalFusionClassifier(nn.Module):
    def __init__(self, sh_dim, sp_dim, num_classes):
        super().__init__()
        # 共享特征融合
        self.shared_fusion = nn.Sequential(
            nn.Linear(sh_dim * 2, 128),
            nn.ReLU()
        )

        # 特定特征融合
        self.specific_fusion = nn.Sequential(
            nn.Linear(sp_dim * 2, 128),
            nn.ReLU()
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar):
        # 融合共享特征
        fused_sh = self.shared_fusion(torch.cat([f_sh_hs, f_sh_radar], dim=1))

        # 融合特定特征
        fused_sp = self.specific_fusion(torch.cat([f_sp_hs, f_sp_radar], dim=1))

        # 联合分类
        fused = torch.cat([fused_sh, fused_sp], dim=1)
        output = self.classifier(fused)
        return output


def scaled_dot_product_attention(q, k, v, attn_mask=None):
    """
    手动实现scaled dot product attention，兼容PyTorch < 2.0版本

    参数:
        q: 查询张量，形状 (batch_size, num_heads, seq_len_q, d_k)
        k: 键张量，形状 (batch_size, num_heads, seq_len_k, d_k)
        v: 值张量，形状 (batch_size, num_heads, seq_len_v, d_v)
        attn_mask: 注意力掩码，形状 (batch_size, 1, seq_len_q, seq_len_k) 或广播兼容形状
                   掩码值为0的位置将被屏蔽

    返回:
        output: 注意力输出，形状 (batch_size, num_heads, seq_len_q, d_v)
        attn_weights: 注意力权重，形状 (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    d_k = q.size(-1)
    if d_k == 0:
        raise ValueError("查询向量维度d_k不能为0")

    # 计算注意力分数 (QK^T)/√d_k
    scores = torch.matmul(q, k.transpose(-2, -1))  # (batch, heads, seq_q, seq_k)
    scores = scores / torch.sqrt(torch.tensor(d_k, dtype=scores.dtype, device=scores.device))

    # 应用注意力掩码
    if attn_mask is not None:
        # 将掩码为0的位置设置为负无穷，确保softmax后权重为0
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

    # 计算注意力权重并应用到值向量
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)

    return output, attn_weights


class CrossAttentionFusion(nn.Module):
    """共享特征交叉注意力融合"""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, feat_a, feat_b):
        B, C, H, W = feat_a.shape

        # 计算交叉注意力
        q = feat_a.flatten(1).unsqueeze(1)  # [B, 1, C*H*W]
        k = v = feat_b.flatten(1).unsqueeze(1)
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v)

        # 残差融合
        fused = self.norm(feat_a + attn_output.view(B, C, H, W))
        return fused


class DynamicWeightFusion(nn.Module):
    """特定特征动态权重融合"""

    def __init__(self, dim):
        super().__init__()
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, feat_a, feat_b):
        w_a = self.uncertainty_estimator(feat_a)  # [B, 1, 7, 7]
        w_b = self.uncertainty_estimator(feat_b)

        # 加权融合
        fused = (w_a * feat_a + w_b * feat_b) / (w_a + w_b + 1e-8)
        return fused



class CozeFusionClassifier(nn.Module):
    def __init__(self, sh_dim, sp_dim, num_classes):
        super().__init__()
        # 共享特征融合
        self.shared_fusion = CrossAttentionFusion(sh_dim)
        self.specific_fusion = DynamicWeightFusion(sp_dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(2*sh_dim, sh_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sh_dim, num_classes)
        )

    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar):
        # 融合共享特征
        fused_sh = self.shared_fusion(f_sh_hs, f_sh_radar)

        # 融合特定特征
        fused_sp = self.specific_fusion(f_sp_hs, f_sp_radar)

        # 联合分类
        fused = torch.cat([fused_sh, fused_sp], dim=1)
        output = self.classifier(fused)
        return output

class Drfuse(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(args, hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.fusion_module = CozeFusionClassifier(feature_dim, feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        output = self.fusion_module(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar

class Drfuse1(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(args, hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.fusion_module = HierarchicalFusionClassifier(feature_dim, feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        output = self.fusion_module(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


class IntraClassCompactnessLoss(nn.Module):
    """类内紧凑性损失函数，减小同类样本特征间的距离"""

    def __init__(self, margin=1.0, weight=0.1):
        super(IntraClassCompactnessLoss, self).__init__()
        self.margin = margin  # 距离边际值
        self.weight = weight  # 损失权重

    def forward(self, features, labels):
        """
        计算类内紧凑性损失

        参数:
            features: 特征张量列表 [f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar]
                    每个特征形状为 [batch, channel, 7, 7]
            labels: 样本标签 [batch]

        返回:
            loss: 类内紧凑性损失值
        """
        total_loss = 0.0
        batch_size = labels.size(0)

        # 对每个特征计算类内紧凑性损失
        for feature in features:
            # 全局平均池化，减少到 [batch, channel]
            pooled_feature = F.adaptive_avg_pool2d(feature, (1, 1)).squeeze(-1).squeeze(-1)

            # 计算类内距离
            class_loss = 0.0
            unique_labels = torch.unique(labels)

            for label in unique_labels:
                # 获取当前类别的所有样本
                class_mask = (labels == label)
                class_features = pooled_feature[class_mask]

                if class_features.size(0) > 1:  # 至少需要两个样本才能计算距离
                    # 计算类内样本间的平均欧氏距离
                    dist_matrix = torch.cdist(class_features, class_features, p=2)
                    mean_dist = dist_matrix.mean()

                    # 使用边际损失函数，鼓励类内距离小于边际值
                    class_loss += F.relu(mean_dist - self.margin)

            # 平均每个类别的损失
            if len(unique_labels) > 0:
                class_loss /= len(unique_labels)
                total_loss += class_loss

        # 平均所有特征的损失
        total_loss /= len(features)

        return self.weight * total_loss