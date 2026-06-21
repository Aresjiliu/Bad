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


def spatial_jsd_pooling(self, shared_A, shared_B):
    # 空间维度计算概率分布
    P = torch.sigmoid(shared_A)  # [B, S, H, W]
    Q = torch.sigmoid(shared_B)
    M = (P + Q) / 2

    # 空间JSD损失
    kl_pm = F.kl_div(P.log(), M, reduction='none').sum(dim=1)
    kl_qm = F.kl_div(Q.log(), M, reduction='none').sum(dim=1)
    jsd = (kl_pm + kl_qm) / 2  # [B, H, W]

    # Logit Pooling (空间维度)
    pooled = torch.logit((P + Q) / 2)  # [B, S, H, W]
    return pooled, jsd


def spatial_ortho_loss(self, shared, specific):
    # 计算每个空间位置的正交性
    shared_flat = shared.flatten(2)  # [B, S, H*W]
    spec_flat = specific.flatten(2)  # [B, D, H*W]

    # 批处理矩阵乘法计算相似度
    norm_shared = F.normalize(shared_flat, dim=1)
    norm_spec = F.normalize(spec_flat, dim=1)
    similarity = torch.bmm(norm_shared.transpose(1, 2), norm_spec)  # [B, H*W, D]

    # 空间位置平均
    return torch.mean(torch.abs(similarity))



class SFDModule(nn.Module):
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
        self.E_sh = nn.Sequential(
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
        f_sh_hs = self.E_sh(f_hs)
        f_sh_radar = self.E_sh(f_radar)

        f_sp_hs = self.E_sp_hs(f_hs)
        f_sp_radar = self.E_sp_radar(f_radar)
        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


class SingleModalityFusionClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, reduction_ratio=8):
        """
        单模态特征融合与分类模块

        参数:
            in_channels: 输入特征通道数 (dim)
            num_classes: 分类任务类别数
            reduction_ratio: 注意力机制的通道缩减比例
        """
        super(SingleModalityFusionClassifier, self).__init__()

        # 1. 空间特征融合模块
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),

        )

        # 2. 通道注意力模块 (用于融合特征优化)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid())

        # 3. 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid())

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes))

    def forward(self, F_sh, F_sp):
        # 1. 初步融合 (通道拼接)
        fused = self.spatial_fusion(torch.cat([F_sh, F_sp], dim=1))

        # 2. 应用通道注意力
        channel_weights = self.channel_attention(fused)
        fused_ch = fused * channel_weights

        # 3. 应用空间注意力
        spatial_weights = self.spatial_attention(
            torch.cat([fused_ch.mean(dim=1, keepdim=True),
                       fused_ch.max(dim=1, keepdim=True)[0]], dim=1))
        fused_sp = fused_ch * spatial_weights

        # 4. 残差连接
        fused_feature = fused + fused_sp

        # 5. 分类
        logits = self.classifier(fused_feature)

        return  logits


class SingleModalityFusionClassifier1(nn.Module):
    def __init__(self, in_channels, num_classes, reduction_ratio=8):
        """
        单模态特征融合与分类模块

        参数:
            in_channels: 输入特征通道数 (dim)
            num_classes: 分类任务类别数
            reduction_ratio: 注意力机制的通道缩减比例
        """
        super(SingleModalityFusionClassifier1, self).__init__()

        # 1. 空间特征融合模块
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 通道注意力模块 (用于融合特征优化)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid())

        # 3. 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid())

        # 4. 分类头
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes))

    def forward(self, F_sh, F_sp):
        # 1. 初步融合 (通道拼接)
        fused = self.spatial_fusion(torch.cat([F_sh, F_sp], dim=1))

        # 2. 应用通道注意力
        channel_weights = self.channel_attention(fused)
        fused_ch = fused * channel_weights

        # 3. 应用空间注意力
        spatial_weights = self.spatial_attention(
            torch.cat([fused_ch.mean(dim=1, keepdim=True),
                       fused_ch.max(dim=1, keepdim=True)[0]], dim=1))
        fused_sp = fused_ch * spatial_weights

        # 4. 残差连接
        fused_feature = fused + fused_sp

        # 5. 分类
        logits = self.classifier(fused_feature)

        return  logits



class SFD_CKD(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.hs_cls = SingleModalityFusionClassifier(feature_dim, args.class_num)
        self.radar_cls = SingleModalityFusionClassifier(feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        hs_output = self.hs_cls(f_sh_hs, f_sp_hs)
        radar_output = self.radar_cls(f_sh_radar, f_sp_radar)
        return hs_output,radar_output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar

class ModalFeatureDecomposition(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=64):
        super().__init__()
        self.sfd = SFDModule(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        return  f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar

class SpatialMDLoss(nn.Module):
    def __init__(self, args, device='cuda', rho1=1.0, rho2=0.7, rho3=0.7, alpha=2.0):
        super(SpatialMDLoss, self).__init__()
        self.rho1 = rho1
        self.rho2 = rho2
        self.rho3 = rho3
        self.alpha = alpha
        self.lambda0 = args.lambda0
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lidar_lambda = args.lidar_lambda
        self.device = device

        # 空间注意力机制 - 确保在正确设备上
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        ).to(self.device)

    def compute_spatial_centers(self, features, labels):
        """计算每个身份的空间加权特征中心"""
        unique_labels = torch.unique(labels)
        centers = []

        for l in unique_labels:
            mask = (labels == l)
            identity_features = features[mask]

            if identity_features.size(0) == 0:
                continue

            spatial_importance = torch.norm(identity_features, dim=1, keepdim=True)
            # 计算平均空间重要性
            mean_spatial_importance = spatial_importance.mean(dim=0, keepdim=True)


            # 计算空间权重 - 需要梯度
            spatial_weights = self.spatial_attention(mean_spatial_importance)

            # 计算加权特征中心 - 使用detach()避免梯度问题
            weighted_features = identity_features * spatial_weights
            center = weighted_features.sum(dim=0) / (mask.sum() * spatial_weights)
            centers.append(center)

        return torch.stack(centers) if centers else torch.empty(0, *features.shape[1:], device=features.device)

    def pairwise_distance(self, x, y):
        """计算两组特征之间的欧氏距离矩阵，避免原地操作"""
        # 使用稳定的距离计算方式
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        dist = torch.norm(diff, p=2, dim=-1)
        return dist

    def forward(self, F_sp_V, F_sh_V, F_sp_I, F_sh_I, Label):
        # 确保所有输入在同一设备上
        F_sp_V = F_sp_V.to(self.device)
        F_sh_V = F_sh_V.to(self.device)
        F_sp_I = F_sp_I.to(self.device)
        F_sh_I = F_sh_I.to(self.device)
        Label = Label.to(self.device)


        # 计算各模态的空间加权特征中心

        C_sp_V = self.compute_spatial_centers(F_sp_V, Label)
        C_sh_V = self.compute_spatial_centers(F_sh_V, Label)
        C_sp_I = self.compute_spatial_centers(F_sp_I, Label)
        C_sh_I = self.compute_spatial_centers(F_sh_I, Label)

        # 如果没有足够的身份，返回0损失
        if len(C_sp_V) == 0 or len(C_sh_V) == 0 or len(C_sp_I) == 0 or len(C_sh_I) == 0:
            return torch.tensor(0.0, device=self.device)

        P = C_sp_V.size(0)  # 身份数量

        L_dc = 0.0
        for p in range(P):
            # 可见模态
            dist_sp_V = self.pairwise_distance(C_sp_V[p].unsqueeze(0), C_sp_V)
            max_sp_V = dist_sp_V.max()
            dist_sh_V = self.pairwise_distance(C_sp_V[p].unsqueeze(0), C_sh_V)
            min_sh_V = dist_sh_V.min()
            term_V = torch.clamp(max_sp_V - min_sh_V + self.rho1, min=0)

            # 红外模态
            dist_sp_I = self.pairwise_distance(C_sp_I[p].unsqueeze(0), C_sp_I)
            max_sp_I = dist_sp_I.max()
            dist_sh_I = self.pairwise_distance(C_sp_I[p].unsqueeze(0), C_sh_I)
            min_sh_I = dist_sh_I.min()
            term_I = torch.clamp(max_sp_I - min_sh_I + self.rho1, min=0)

            L_dc += term_V + term_I*self.lidar_lambda

        # 计算L_sps
        L_sps = 0.0
        for p in range(P):
            # 可见模态
            dist_V = self.pairwise_distance(C_sp_V[p].unsqueeze(0), C_sp_V)
            dist_V = dist_V.clone()
            dist_V[:, p] = float('inf')  # 排除自身
            min_dist_V = dist_V.min()
            term_V = torch.clamp(self.rho2 - min_dist_V, min=0)

            # 红外模态
            dist_I = self.pairwise_distance(C_sp_I[p].unsqueeze(0), C_sp_I)
            dist_I = dist_I.clone()
            dist_I[:, p] = float('inf')
            min_dist_I = dist_I.min()
            term_I = torch.clamp(self.rho2 - min_dist_I, min=0)

            L_sps += term_V + term_I

        # 计算L_shs
        L_shs = 0.0
        # 合并所有共享特征中心
        all_C_sh = torch.cat([C_sh_V, C_sh_I], dim=0)

        for p in range(P):
            # 同一身份的跨模态共享中心距离
            cross_modal_dist = torch.norm(C_sh_V[p] - C_sh_I[p], p=2)
            L_shs += self.alpha * cross_modal_dist

            # 可见共享中心与其他所有共享中心的最小距离
            dist_V_all = self.pairwise_distance(C_sh_V[p].unsqueeze(0), all_C_sh)
            dist_V_all = dist_V_all.clone()
            dist_V_all[:, p] = float('inf')  # 排除自身
            min_dist_V = dist_V_all.min()
            term_V = torch.clamp(self.rho3 - min_dist_V, min=0)

            # 红外共享中心与其他所有共享中心的最小距离
            dist_I_all = self.pairwise_distance(C_sh_I[p].unsqueeze(0), all_C_sh)
            dist_I_all =dist_I_all.clone()
            dist_I_all[:, P + p] = float('inf')  # 排除自身
            min_dist_I = dist_I_all.min()
            term_I = torch.clamp(self.rho3 - min_dist_I, min=0)

            L_shs += term_V + term_I

        total_loss = self.lambda0*L_shs + self.lambda1 * L_dc + self.lambda2 * L_sps
        return total_loss


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