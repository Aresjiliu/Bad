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


class SFDModuleOrth(nn.Module):
    def __init__(self, hsi_channels=144, radar_channels=1, feature_dim=64):
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


class SFDModuleMD(nn.Module):
    def __init__(self, hsi_channels=144, radar_channels=1, feature_dim=64):
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

        # 共享特征提取器 (独立参数，模仿Drfuse设计)
        self.E_sh_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])  # 添加层归一化提升稳定性
        )
        self.E_sh_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])  # 添加层归一化提升稳定性
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])  # 添加层归一化提升稳定性
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.LayerNorm([feature_dim, 7, 7])  # 添加层归一化提升稳定性
        )

    def forward(self, x_hs, x_radar):
        # 基础特征提取
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解 - 使用独立的共享特征提取器
        f_sh_hs = self.E_sh_hs(f_hs)
        f_sh_radar = self.E_sh_radar(f_radar)

        f_sp_hs = self.E_sp_hs(f_hs)
        f_sp_radar = self.E_sp_radar(f_radar)
        
        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


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
        self.sfd = SFDModuleOrth(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.fusion_module = CozeFusionClassifier(feature_dim, feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        output = self.fusion_module(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


class Mdfuse(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModuleMD(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        
        # 添加跨模态注意力机制（模仿Drfuse设计）
        self.cross_attention = CrossAttentionFusion(feature_dim)
        
        # 增强的融合分类器
        self.fusion_module = CozeFusionClassifier(feature_dim, feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        
        # 应用跨模态注意力融合
        f_sh_hs_att = self.cross_attention(f_sh_hs, f_sh_radar)
        f_sh_radar_att = self.cross_attention(f_sh_radar, f_sh_hs)
        
        # 使用注意力增强的特征进行分类
        output = self.fusion_module(f_sh_hs_att, f_sh_radar_att, f_sp_hs, f_sp_radar)
        return output, f_sh_hs_att, f_sh_radar_att, f_sp_hs, f_sp_radar




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
        # 确保标签与特征在同一设备上
        device = features[0].device if features else torch.device('cpu')
        labels = labels.to(device)
        
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


class SpatialJSDLoss(nn.Module):
    """空间JSD对齐损失，用于对齐不同模态的共享特征"""
    
    def __init__(self, epsilon=1e-8):
        super(SpatialJSDLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, shared_A, shared_B):
        """
        计算空间JSD损失
        
        参数:
            shared_A: 模态A的共享特征 [B, C, H, W]
            shared_B: 模态B的共享特征 [B, C, H, W]
            
        返回:
            jsd_loss: JSD损失值
            pooled: 融合后的特征
        """
        # 转换为概率分布 (空间维度保持)
        P = torch.sigmoid(shared_A)
        Q = torch.sigmoid(shared_B)
        M = (P + Q) / 2

        # 计算KL散度 (添加epsilon防止log(0))
        kl_pm = F.kl_div(
            torch.log(P + self.epsilon),
            M + self.epsilon,
            reduction='none'
        )
        kl_qm = F.kl_div(
            torch.log(Q + self.epsilon),
            M + self.epsilon,
            reduction='none'
        )

        # 空间平均JSD损失
        jsd_loss = 0.5 * (kl_pm + kl_qm)
        jsd_loss = jsd_loss.mean(dim=[1, 2, 3]).mean()  # 批次平均

        # Logit Pooling融合
        pooled = torch.logit((P + Q) / 2 + self.epsilon)
        return jsd_loss, pooled


class SpatialOrthogonalityLoss(nn.Module):
    """空间正交约束损失，确保共享特征和特定特征的独立性"""
    
    def __init__(self):
        super(SpatialOrthogonalityLoss, self).__init__()
    
    def forward(self, shared, specific):
        """
        计算空间正交约束损失
        
        参数:
            shared: 共享特征 [B, C, H, W]
            specific: 特定特征 [B, D, H, W]
            
        返回:
            ortho_loss: 正交损失值
        """
        # 展平空间维度
        shared_flat = shared.flatten(2)  # [B, C, H*W]
        spec_flat = specific.flatten(2)  # [B, D, H*W]

        # 归一化
        norm_shared = F.normalize(shared_flat, dim=1)
        norm_spec = F.normalize(spec_flat, dim=1)

        similarity = torch.einsum('bch,bdh->bcd', norm_shared, norm_spec)

        # 空间位置平均
        return torch.mean(torch.abs(similarity))


class DrfuseLoss(nn.Module):
    """DrFuse模型总损失函数，包含JSD对齐损失和正交约束损失"""
    
    def __init__(self, jsdp_weight=1.0, ortho_weight=1.0):
        super(DrfuseLoss, self).__init__()
        self.jsd_loss = SpatialJSDLoss()
        self.ortho_loss = SpatialOrthogonalityLoss()
        self.jsdp_weight = jsdp_weight
        self.ortho_weight = ortho_weight
    
    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar):
        """
        计算DrFuse总损失
        
        参数:
            f_sh_hs: HSI共享特征 [B, C, H, W]
            f_sh_radar: LiDAR共享特征 [B, C, H, W]
            f_sp_hs: HSI特定特征 [B, C, H, W]
            f_sp_radar: LiDAR特定特征 [B, C, H, W]
            
        返回:
            total_loss: 总损失值
        """
        # JSD对齐损失
        aligned_shared, jsd_loss = self.jsd_loss(f_sh_hs, f_sh_radar)
        
        # 正交约束损失
        ortho_loss_A = self.ortho_loss(f_sh_hs, f_sp_hs)
        ortho_loss_B = self.ortho_loss(f_sh_radar, f_sp_radar)
        
        # 总损失
        total_loss = jsd_loss + ortho_loss_A + ortho_loss_B
        
        return total_loss


class IdentityClassificationLoss(nn.Module):
    """身份分类损失 (ID Loss)
    
    使用DrFuse模型输出的四个特征（f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar）
    分别通过四个独立的分类器进行分类，计算分类损失之和作为ID损失。
    这种损失函数可以增强特征的判别性，确保不同类型的特征都能有效区分不同类别。
    """
    
    def __init__(self, feature_dim=256, num_classes=15, loss_weight=1.0):
        """
        初始化身份分类损失
        
        参数:
            feature_dim: 特征维度，默认256
            num_classes: 类别数量，默认15
            loss_weight: 损失权重，默认1.0
        """
        super(IdentityClassificationLoss, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        
        # 四个独立的分类器，分别对应不同的特征类型
        self.classifier_sh_hs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(),  # [B, C]
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.classifier_sh_radar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.classifier_sp_hs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        self.classifier_sp_radar = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # 基础分类损失函数
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, target):
        """
        计算身份分类损失
        
        参数:
            f_sh_hs: HSI共享特征 [B, C, H, W]
            f_sh_radar: LiDAR共享特征 [B, C, H, W]  
            f_sp_hs: HSI特定特征 [B, C, H, W]
            f_sp_radar: LiDAR特定特征 [B, C, H, W]
            target: 目标标签 [B]
            
        返回:
            id_loss: 身份分类损失值
            cls_outputs: 四个分类器的输出字典，用于分析
        """
        # 确保目标标签与特征在同一设备上
        device = f_sh_hs.device
        target = target.to(device)
        
        # 确保整个模块在正确的设备上
        if next(self.classifier_sh_hs.parameters()).device != device:
            self.to(device)
        
        # 通过四个分类器分别进行预测
        output_sh_hs = self.classifier_sh_hs(f_sh_hs)
        output_sh_radar = self.classifier_sh_radar(f_sh_radar)
        output_sp_hs = self.classifier_sp_hs(f_sp_hs)
        output_sp_radar = self.classifier_sp_radar(f_sp_radar)
        
        # 计算四个分类损失
        loss_sh_hs = self.criterion(output_sh_hs, target)
        loss_sh_radar = self.criterion(output_sh_radar, target)
        loss_sp_hs = self.criterion(output_sp_hs, target)
        loss_sp_radar = self.criterion(output_sp_radar, target)
        
        # 总ID损失为四个分类损失之和
        id_loss = loss_sh_hs + loss_sh_radar + loss_sp_hs + loss_sp_radar
        
        # 存储分类输出，便于后续分析
        # cls_outputs = {
        #     'sh_hs': output_sh_hs,
        #     'sh_radar': output_sh_radar,
        #     'sp_hs': output_sp_hs,
        #     'sp_radar': output_sp_radar
        # }
        
        return self.loss_weight * id_loss
    
    def get_individual_losses(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, target):
        """
        获取各个分类器的单独损失，用于分析每个特征类型的贡献
        
        参数:
            同forward函数
            
        返回:
            losses: 包含各个分类损失的dict
        """
        with torch.no_grad():
            # 通过四个分类器分别进行预测
            output_sh_hs = self.classifier_sh_hs(f_sh_hs)
            output_sh_radar = self.classifier_sh_radar(f_sh_radar)
            output_sp_hs = self.classifier_sp_hs(f_sp_hs)
            output_sp_radar = self.classifier_sp_radar(f_sp_radar)
            
            # 计算四个分类损失
            loss_sh_hs = self.criterion(output_sh_hs, target)
            loss_sh_radar = self.criterion(output_sh_radar, target)
            loss_sp_hs = self.criterion(output_sp_hs, target)
            loss_sp_radar = self.criterion(output_sp_radar, target)
            
            losses = {
                'sh_hs': loss_sh_hs.item(),
                'sh_radar': loss_sh_radar.item(),
                'sp_hs': loss_sp_hs.item(),
                'sp_radar': loss_sp_radar.item(),
                'total': (loss_sh_hs + loss_sh_radar + loss_sp_hs + loss_sp_radar).item()
            }
            
        return losses



class SpatialMDLoss(nn.Module):
    def __init__(self, args, device='cuda', rho1=1.0, rho2=0.7, rho3=0.7, alpha=2.0, temperature=0.1, use_feature_losses=False):
        super(SpatialMDLoss, self).__init__()
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.alpha = alpha
        self.temperature = temperature
        self.lambda0 = args.lambda0
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lidar_lambda = args.lidar_lambda
        self.device = device
        self.use_feature_losses = use_feature_losses

        # 多尺度空间注意力机制
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        ).to(self.device)
        
        # 通道注意力机制
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(args.feature_dim if hasattr(args, 'feature_dim') else 128, (args.feature_dim if hasattr(args, 'feature_dim') else 128) // 16, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d((args.feature_dim if hasattr(args, 'feature_dim') else 128) // 16, args.feature_dim if hasattr(args, 'feature_dim') else 128, 1, bias=False),
            nn.Sigmoid()
        ).to(self.device)
        
        # 特征质量评估器
        self.feature_quality = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(args.feature_dim if hasattr(args, 'feature_dim') else 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        ).to(self.device)

    def compute_spatial_centers(self, features, labels):
        """增强版空间加权特征中心计算，包含特征质量评估和难样本挖掘"""
        unique_labels = torch.unique(labels)
        centers = []
        
        for l in unique_labels:
            mask = (labels == l)
            identity_features = features[mask]
            
            if identity_features.size(0) == 0:
                continue
            
            B, C, H, W = identity_features.shape
            
            # 1. 计算空间重要性（多尺度）
            spatial_importance = torch.norm(identity_features, dim=1, keepdim=True)  # [B, 1, H, W]
            
            # 2. 计算通道注意力
            channel_weights = self.channel_attention(identity_features)  # [B, C, 1, 1]
            
            # 3. 计算特征质量得分
            quality_scores = self.feature_quality(identity_features)  # [B, 1]
            
            # 4. 计算空间权重（多尺度注意力）
            spatial_weights = self.spatial_attention(spatial_importance)  # [B, 1, H, W]
            
            # 5. 难样本挖掘：降低低质量样本的权重
            hard_mining_weights = torch.where(quality_scores > 0.3, 
                                            quality_scores, 
                                            torch.zeros_like(quality_scores) * 0.1)
            
            # 6. 综合权重计算
            combined_weights = spatial_weights * hard_mining_weights.view(B, 1, 1, 1)
            
            # 7. 应用通道注意力
            weighted_features = identity_features * channel_weights * combined_weights
            
            # 8. 计算特征中心（归一化权重）
            weight_sum = combined_weights.sum() + 1e-8
            center = weighted_features.sum(dim=0) / weight_sum
            centers.append(center)
        
        return torch.stack(centers) if centers else torch.empty(0, *features.shape[1:], device=features.device)

    def contrastive_center_loss(self, centers_sp, centers_sh, labels):
        """对比学习损失：拉近同类共享特征中心，推远异类共享特征中心"""
        if centers_sp.size(0) == 0 or centers_sh.size(0) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 归一化特征中心
        centers_sp_norm = F.normalize(centers_sp.view(centers_sp.size(0), -1), dim=1)
        centers_sh_norm = F.normalize(centers_sh.view(centers_sh.size(0), -1), dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.mm(centers_sp_norm, centers_sh_norm.t()) / self.temperature
        
        # 对比损失：同类应该相似，异类应该不相似
        batch_size = centers_sp.size(0)
        labels_expand = labels.unsqueeze(1).expand(batch_size, batch_size)
        mask = torch.eq(labels_expand, labels_expand.t()).float()
        
        # 正样本对（同类）
        pos_pairs = similarity_matrix * mask
        # 负样本对（异类）
        neg_pairs = similarity_matrix * (1 - mask)
        
        # InfoNCE损失
        numerator = torch.exp(pos_pairs).sum(dim=1)
        denominator = torch.exp(similarity_matrix).sum(dim=1)
        
        contrastive_loss = -torch.log(numerator / (denominator + 1e-8) + 1e-8)
        return contrastive_loss.mean()

    def hard_negative_mining(self, centers, labels, num_negatives=5):
        """难负样本挖掘：选择最难的负样本进行训练"""
        if centers.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # 计算距离矩阵
        centers_flat = centers.view(centers.size(0), -1)
        distance_matrix = torch.cdist(centers_flat, centers_flat, p=2)
        
        # 为每个样本找到最难的负样本
        batch_size = centers.size(0)
        hard_negative_loss = 0.0
        
        for i in range(batch_size):
            # 找到同类样本
            same_class_mask = (labels == labels[i])
            # 找到异类样本
            diff_class_mask = ~same_class_mask
            
            if diff_class_mask.sum() == 0:
                continue
            
            # 选择距离最近的异类样本（最难负样本）
            neg_distances = distance_matrix[i][diff_class_mask]
            hardest_neg_idx = torch.argmin(neg_distances)
            hardest_neg_distance = neg_distances[hardest_neg_idx]
            
            # 选择同类样本中距离最远的（最难正样本）
            pos_distances = distance_matrix[i][same_class_mask]
            if pos_distances.size(0) > 1:  # 至少有2个同类样本
                hardest_pos_distance = pos_distances.topk(2, largest=True)[0][1]  # 选择第二远的（排除自己）
            else:
                continue
            
            # Triplet损失：正样本距离 + margin < 负样本距离
            triplet_loss = F.relu(hardest_pos_distance - hardest_neg_distance + self.rho1)
            hard_negative_loss += triplet_loss
        
        return hard_negative_loss / max(batch_size, 1)

    def pairwise_distance(self, x, y):
        """计算两组特征之间的欧氏距离矩阵，避免原地操作"""
        # 使用稳定的距离计算方式
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist = torch.norm(diff, p=2, dim=-1)
        return dist

    def feature_contrastive_loss(self, f_sh, f_sp, temperature=0.1):
        """特征级对比学习损失：在特征空间拉近共享特征，推远特定特征"""
        B, C, H, W = f_sh.shape
        
        # 展平并归一化
        f_sh_flat = F.normalize(f_sh.view(B, C, -1), dim=1)
        f_sp_flat = F.normalize(f_sp.view(B, C, -1), dim=1)
        
        # 计算相似度矩阵
        similarity = torch.bmm(f_sh_flat.transpose(1, 2), f_sp_flat) / temperature
        
        # 对比损失：最大化共享特征与特定特征的相似度
        # 使用稳定的损失函数：-log(sigmoid(x))，但确保不溢出
        positive_pairs = torch.diagonal(similarity, dim1=1, dim2=2)
        
        # 使用log-sigmoid损失，但添加数值稳定性保护
        log_sigmoid = torch.log(torch.sigmoid(positive_pairs) + 1e-8)
        contrastive_loss = -torch.mean(log_sigmoid)
        
        # 确保损失非负
        return torch.clamp(contrastive_loss, min=0.0)

    def feature_orthogonal_loss(self, f_sh, f_sp):
        """特征级正交约束损失：确保共享和特定特征正交"""
        B, C, H, W = f_sh.shape
        
        # 展平并归一化
        f_sh_norm = F.normalize(f_sh.view(B, C, -1), dim=2)
        f_sp_norm = F.normalize(f_sp.view(B, C, -1), dim=2)
        
        # 计算正交性损失
        ortho_loss = torch.mean(torch.abs(torch.bmm(f_sh_norm, f_sp_norm.transpose(1, 2))))
        
        return ortho_loss

    def forward(self, F_sp_V, F_sh_V, F_sp_I, F_sh_I, Label, return_aux_losses=False):
        """
        改进的Md_loss：返回单一损失值，保持代码整洁
        """
        # 数据设备转移
        F_sp_V = F_sp_V.to(self.device)
        F_sh_V = F_sh_V.to(self.device)
        F_sp_I = F_sp_I.to(self.device)
        F_sh_I = F_sh_I.to(self.device)
        Label = Label.to(self.device)

        # 计算各模态的空间加权特征中心
        unique_labels = torch.unique(Label)
        C_sp_V = self.compute_spatial_centers(F_sp_V, Label)
        C_sh_V = self.compute_spatial_centers(F_sh_V, Label)
        C_sp_I = self.compute_spatial_centers(F_sp_I, Label)
        C_sh_I = self.compute_spatial_centers(F_sh_I, Label)

        # 如果没有足够的身份，返回0损失
        if len(C_sp_V) == 0 or len(C_sh_V) == 0 or len(C_sp_I) == 0 or len(C_sh_I) == 0:
            return torch.tensor(0.0, device=self.device)

        P = C_sp_V.size(0)  # 身份数量

        # === 核心Md损失计算 ===
        total_loss = 0.0
        
        # 1. 跨模态共享特征对齐损失
        cross_modal_loss = 0.0
        for p in range(P):
            cross_modal_dist = torch.norm(C_sh_V[p] - C_sh_I[p], p=2)
            cross_modal_loss += self.alpha * cross_modal_dist
        total_loss += self.lambda0 * cross_modal_loss

        # 2. 特定-共享特征距离约束
        sp_sh_loss = 0.0
        for p in range(P):
            dist_sp_sh_V = torch.norm(C_sp_V[p] - C_sh_V[p], p=2)
            dist_sp_sh_I = torch.norm(C_sp_I[p] - C_sh_I[p], p=2)
            sp_sh_loss += (dist_sp_sh_V + dist_sp_sh_I) * 0.5
        total_loss += self.lambda1 * sp_sh_loss

        # 3. 对比学习损失（仅当有足够类别时）
        if P > 1:
            # 中心级对比损失
            centers_sp_norm = F.normalize(C_sp_V.view(P, -1), dim=1)
            centers_sh_norm = F.normalize(C_sh_V.view(P, -1), dim=1)
            
            similarity_matrix = torch.mm(centers_sp_norm, centers_sh_norm.t()) / self.temperature
            positive_pairs = torch.diag(similarity_matrix)
            log_sum_exp = torch.log(torch.sum(torch.exp(similarity_matrix), dim=1) + 1e-8)
            contrastive_loss = torch.mean(log_sum_exp - positive_pairs)
            
            total_loss += 0.2 * contrastive_loss  # 对比损失权重

        # === 返回单一损失值 ===
        # 为了保持与train_model的兼容性，忽略return_aux_losses参数
        # 始终返回单一损失值，保持代码整洁
        return total_loss




# class HierarchicalFusionClassifier(nn.Module):
#     def __init__(self, sh_dim, sp_dim, num_classes):
#         super().__init__()
#         # 共享特征融合
#         self.shared_fusion = nn.Sequential(
#             nn.Linear(sh_dim * 2, 128),
#             nn.ReLU()
#         )
#
#         # 特定特征融合
#         self.specific_fusion = nn.Sequential(
#             nn.Linear(sp_dim * 2, 128),
#             nn.ReLU()
#         )
#
#         # 最终分类器
#         self.classifier = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes)
#         )
#
#     def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar):
#         # 融合共享特征
#         fused_sh = self.shared_fusion(torch.cat([f_sh_hs, f_sh_radar], dim=1))
#
#         # 融合特定特征
#         fused_sp = self.specific_fusion(torch.cat([f_sp_hs, f_sp_radar], dim=1))
#
#         # 联合分类
#         fused = torch.cat([fused_sh, fused_sp], dim=1)
#         output = self.classifier(fused)
#         return output

# class Drfuse1(nn.Module):
#     def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
#         super().__init__()
#         self.sfd = SFDModule(args, hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
#         self.fusion_module = HierarchicalFusionClassifier(feature_dim, feature_dim, args.class_num)
#
#     def forward(self, x_hs, x_radar):
#         f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
#         output = self.fusion_module(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
#         return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar