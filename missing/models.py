import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, num_classes=10, feature_dim=128):
        super().__init__()

        # 高光谱分支 (144通道)
        self.hs_stream = nn.Sequential(
            SS_DecoupledBlock(144, 64),
            SS_DecoupledBlock(64, 128)
        )

        # 雷达分支 (1通道)
        self.radar_stream = nn.Sequential(
            OrientedGradientBlock(1, 64),
            OrientedGradientBlock(64, 128))

        # 共享特征提取器 (参数共享)
        self.E_sh = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier_sp_hs = nn.Linear(256, num_classes)
        self.classifier_sp_radar = nn.Linear(256, num_classes)
        self.classifier_sh = nn.Linear(256, num_classes)

        # MD损失的超参数
        self.rho1 = 1.0  # Decorrelation Loss的margin
        self.rho2 = 0.7  # Modality-specific Separation的margin
        self.rho3 = 0.7  # Modality-shared Separation的margin
        self.alpha = 2.0  # Modality-shared Separation的权重
        self.lambda1 = 0.5  # L_dc的权重
        self.lambda2 = 0.5  # L_sps的权重

    def forward(self, x_hs, x_radar, labels):
        # 基础特征提取
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解
        f_sh_hs = self.E_sh(f_hs).flatten(1)  # (B,256)
        f_sh_radar = self.E_sh(f_radar).flatten(1)  # 共享参数

        f_sp_hs = self.E_sp_hs(f_hs).flatten(1)  # (B,256)
        f_sp_radar = self.E_sp_radar(f_radar).flatten(1)
        # 按身份分组计算特征中心
        unique_labels = torch.unique(labels)
        centers_sp_hs = []
        centers_sp_radar = []
        centers_sh_hs = []
        centers_sh_radar = []

        for label in unique_labels:
            mask = (labels == label)
            # 高光谱模态
            centers_sp_hs.append(f_sp_hs[mask].mean(dim=0))
            centers_sh_hs.append(f_sh_hs[mask].mean(dim=0))
            # 雷达模态
            centers_sp_radar.append(f_sp_radar[mask].mean(dim=0))
            centers_sh_radar.append(f_sh_radar[mask].mean(dim=0))

        centers_sp_hs = torch.stack(centers_sp_hs)  # (P,256)
        centers_sp_radar = torch.stack(centers_sp_radar)
        centers_sh_hs = torch.stack(centers_sh_hs)
        centers_sh_radar = torch.stack(centers_sh_radar)


        L_dc = 0.0
        for i in range(len(unique_labels)):
            # 高光谱模态
            max_sp_dist = torch.max(torch.cdist(centers_sp_hs[i].unsqueeze(0), centers_sp_hs))
            min_sh_dist = torch.min(torch.cdist(centers_sp_hs[i].unsqueeze(0), centers_sh_hs))
            L_dc += torch.relu(max_sp_dist - min_sh_dist + self.rho1)

            # 雷达模态
            max_sp_dist = torch.max(torch.cdist(centers_sp_radar[i].unsqueeze(0), centers_sp_radar))
            min_sh_dist = torch.min(torch.cdist(centers_sp_radar[i].unsqueeze(0), centers_sh_radar))
            L_dc += torch.relu(max_sp_dist - min_sh_dist + self.rho1)

        # ===========================================
        # 2. Modality-specific Separation Loss (L_sps)
        # ===========================================
        L_sps = 0.0
        for i in range(len(unique_labels)):
            # 高光谱模态
            dists = torch.norm(centers_sp_hs[i] - centers_sp_hs, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho2, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho2 - min_dist)

            # 雷达模态
            dists = torch.norm(centers_sp_radar[i] - centers_sp_radar, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho2, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho2 - min_dist)

        # ===========================================
        # 3. Modality-shared Separation Loss (L_shs)
        # ===========================================
        L_shs = 0.0
        for i in range(len(unique_labels)):
            # 同一身份的跨模态对齐
            L_shs += self.alpha * torch.norm(centers_sh_hs[i] - centers_sh_radar[i], p=2)

            # 不同身份的高光谱模
            dists = torch.norm(centers_sh_hs[i] - centers_sh_hs, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho3, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho3 - min_dist)
            # 不同身份的雷达模态
            dists = torch.norm(centers_sh_radar[i] - centers_sh_radar, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho3, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho3 - min_dist)

        # ===========================================
        # 4. ID Loss
        # ===========================================
        id_loss = 0.0
        id_loss += F.cross_entropy(self.classifier_sp_hs(f_sp_hs), labels)
        id_loss += F.cross_entropy(self.classifier_sp_radar(f_sp_radar), labels)
        id_loss += F.cross_entropy(self.classifier_sh(f_sh_hs), labels)
        id_loss += F.cross_entropy(self.classifier_sh(f_sh_radar), labels)

        # ===========================================
        # 总MD损失
        # ===========================================
        MD_loss = L_shs + self.lambda1 * L_dc + self.lambda2 * L_sps
        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, MD_loss, id_loss





class SFDModule1(nn.Module):
    def __init__(self, num_classes=10, feature_dim=128):
        super().__init__()

        # 高光谱分支 (144通道)
        self.hs_stream = nn.Sequential(
            SS_DecoupledBlock(144, 64),
            SS_DecoupledBlock(64, 128)
        )

        # 雷达分支 (1通道)
        self.radar_stream = nn.Sequential(
            OrientedGradientBlock(1, 64),
            OrientedGradientBlock(64, 128))

        # 共享特征提取器 (参数共享)
        self.E_sh = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )


    def forward(self, x_hs, x_radar, labels):
        # 基础特征提取
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解
        f_sh_hs = self.E_sh(f_hs).flatten(1)  # (B,256)
        f_sh_radar = self.E_sh(f_radar).flatten(1)  # 共享参数

        f_sp_hs = self.E_sp_hs(f_hs).flatten(1)  # (B,256)
        f_sp_radar = self.E_sp_radar(f_radar).flatten(1)

        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar


class SFDModule2(nn.Module):
    def __init__(self, num_classes=10, feature_dim=128):
        super().__init__()

        # 高光谱分支 (144通道)
        self.hs_stream = nn.Sequential(
            SS_DecoupledBlock(144, 64),
            SS_DecoupledBlock(64, 128)
        )

        # 雷达分支 (1通道)
        self.radar_stream = nn.Sequential(
            OrientedGradientBlock(1, 64),
            OrientedGradientBlock(64, 128))

        # 共享特征提取器 (参数共享)
        self.E_sh = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.rho1 = 1.0  # Decorrelation Loss的margin
        self.rho2 = 0.7  # Modality-specific Separation的margin
        self.rho3 = 0.7  # Modality-shared Separation的margin
        self.alpha = 2.0  # Modality-shared Separation的权重
        self.lambda1 = 0.5  # L_dc的权重
        self.lambda2 = 0.5  # L_sps的权重

    def forward(self, x_hs, x_radar, labels):
        # 基础特征提取
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解
        f_sh_hs = self.E_sh(f_hs).flatten(1)  # (B,256)
        f_sh_radar = self.E_sh(f_radar).flatten(1)  # 共享参数

        f_sp_hs = self.E_sp_hs(f_hs).flatten(1)  # (B,256)
        f_sp_radar = self.E_sp_radar(f_radar).flatten(1)

        unique_labels = torch.unique(labels)
        centers_sp_hs = []
        centers_sp_radar = []
        centers_sh_hs = []
        centers_sh_radar = []

        for label in unique_labels:
            mask = (labels == label)
            # 高光谱模态
            centers_sp_hs.append(f_sp_hs[mask].mean(dim=0))
            centers_sh_hs.append(f_sh_hs[mask].mean(dim=0))
            # 雷达模态
            centers_sp_radar.append(f_sp_radar[mask].mean(dim=0))
            centers_sh_radar.append(f_sh_radar[mask].mean(dim=0))

        centers_sp_hs = torch.stack(centers_sp_hs)  # (P,256)
        centers_sp_radar = torch.stack(centers_sp_radar)
        centers_sh_hs = torch.stack(centers_sh_hs)
        centers_sh_radar = torch.stack(centers_sh_radar)

        # ===========================================
        # 1. Decorrelation Loss (L_dc)
        # ===========================================
        L_dc = 0.0
        for i in range(len(unique_labels)):
            # 高光谱模态
            max_sp_dist = torch.max(torch.cdist(centers_sp_hs[i].unsqueeze(0), centers_sp_hs))
            min_sh_dist = torch.min(torch.cdist(centers_sp_hs[i].unsqueeze(0), centers_sh_hs))
            L_dc += torch.relu(max_sp_dist - min_sh_dist + self.rho1)

            # 雷达模态
            max_sp_dist = torch.max(torch.cdist(centers_sp_radar[i].unsqueeze(0), centers_sp_radar))
            min_sh_dist = torch.min(torch.cdist(centers_sp_radar[i].unsqueeze(0), centers_sh_radar))
            L_dc += torch.relu(max_sp_dist - min_sh_dist + self.rho1)

        # ===========================================
        # 2. Modality-specific Separation Loss (L_sps)
        # ===========================================
        L_sps = 0.0
        for i in range(len(unique_labels)):
            # 高光谱模态
            dists = torch.norm(centers_sp_hs[i] - centers_sp_hs, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho2, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho2 - min_dist)

            # 雷达模态
            dists = torch.norm(centers_sp_radar[i] - centers_sp_radar, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho2, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho2 - min_dist)

        # ===========================================
        # 3. Modality-shared Separation Loss (L_shs)
        # ===========================================
        L_shs = 0.0
        for i in range(len(unique_labels)):
            # 同一身份的跨模态对齐
            L_shs += self.alpha * torch.norm(centers_sh_hs[i] - centers_sh_radar[i], p=2)

            # 不同身份的高光谱模
            dists = torch.norm(centers_sh_hs[i] - centers_sh_hs, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho3, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho3 - min_dist)
            # 不同身份的雷达模态
            dists = torch.norm(centers_sh_radar[i] - centers_sh_radar, dim=1)
            filtered_dists = dists[dists > 0]  # 排除自身

            if filtered_dists.numel() == 0:  # 检查是否为空
                min_dist = torch.tensor(self.rho3, device=dists.device)  # 默认值为rho2
            else:
                min_dist = torch.min(filtered_dists)

            L_sps += torch.relu(self.rho3 - min_dist)
        return f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar




class FMCModule(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(FMCModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Generator: 从可见光共享特征生成红外模态特定特征
        self.generator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        # Discriminator: 区分生成的红外模态特定特征和真实的红外模态特定特征
        self.discriminator = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # 身份分类器
        self.identity_classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, F_sh_V, F_sp_I, labels):
        # 生成缺失的红外模态特定特征
        F_sp_I_generated = self.generator(F_sh_V)

        # 判别器输出
        real_score = self.discriminator(F_sp_I)
        fake_score = self.discriminator(F_sp_I_generated)

        # 计算对抗损失
        adversarial_loss = torch.mean(torch.log(1 - fake_score)) - torch.mean(torch.log(real_score))

        # 计算特征一致性损失
        centers = torch.stack([F_sp_I[labels == i].mean(dim=0) for i in range(self.num_classes)])
        feature_consistency_loss = F.l1_loss(F_sp_I_generated, centers[labels])

        # 计算身份一致性损失
        identity_scores = self.identity_classifier(F_sp_I_generated)
        identity_consistency_loss = F.cross_entropy(identity_scores, labels)

        # 总损失
        total_loss = adversarial_loss + feature_consistency_loss + identity_consistency_loss

        return F_sp_I_generated, total_loss


class SFFModule(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(SFFModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # 可学习的权重参数
        self.omega1_V = nn.Parameter(torch.tensor(0.5))  # 可见光模态特定特征的权重
        self.omega2_V = nn.Parameter(torch.tensor(0.5))  # 生成的红外模态特定特征的权重
        self.omega1_I = nn.Parameter(torch.tensor(0.5))  # 红外模态特定特征的权重
        self.omega2_I = nn.Parameter(torch.tensor(0.5))  # 生成的可见光模态特定特征的权重

        # 共享的分类器
        self.shared_classifier1 = nn.Linear(2 * feature_dim, num_classes)
        self.shared_classifier2 = nn.Linear(2 * feature_dim, num_classes)

        # 跨模态中心损失的参数
        self.beta = 2.0
        self.rho4 = 0.7
        self.eps = 1e-6  # 数值稳定常数
        self.rho4_tensor = torch.tensor(self.rho4)

    def forward(self, F_sh_V, F_sp_V, F_sp_I_generated, F_sh_I, F_sp_I, F_sp_V_generated, labels):
        # 融合可见光模态的模态特定特征和生成的红外模态特定特征
        F_fu_V = self.omega1_V * F_sp_V + self.omega2_V * F_sp_I_generated

        # 拼接模态共享特征和融合后的模态特定特征
        F_fp_V = torch.cat((F_sh_V, F_fu_V), dim=1)

        # 融合红外模态的模态特定特征和生成的可见光模态特定特征
        F_fu_I = self.omega1_I * F_sp_I + self.omega2_I * F_sp_V_generated

        # 拼接模态共享特征和融合后的模态特定特征
        F_fp_I = torch.cat((F_sh_I, F_fu_I), dim=1)
        # F_fp_V = torch.cat((F_sh_V, F_sp_V), dim=1)
        # F_fp_I = torch.cat((F_sh_I, F_sp_I), dim=1)
        # 计算分类分数
        S_fp_V = self.shared_classifier1(F_fp_V)
        S_fp_I = self.shared_classifier2(F_fp_I)

        # 计算身份分类损失
        identity_loss_V = F.cross_entropy(S_fp_V, labels)
        identity_loss_I = F.cross_entropy(S_fp_I, labels)
        identity_loss = identity_loss_V + identity_loss_I

        # 计算跨模态中心损失
        centers_V = torch.stack([F_fp_V[labels == i].mean(dim=0) for i in range(self.num_classes)])
        centers_I = torch.stack([F_fp_I[labels == i].mean(dim=0) for i in range(self.num_classes)])

        center_loss = 0
        # 总损失
        zero_tensor = torch.tensor(0.0,
                                   dtype=centers_V.dtype,
                                   device=centers_V.device)

        for p in range(self.num_classes):
            # ================= 模态间差异 =================
            # 检查当前类别是否存在样本
            valid_p = (labels == p).sum().item() > 0
            if not valid_p:
                continue

            # 计算跨模态中心差异 (V模态和I模态)
            diff_inter = centers_V[p] - centers_I[p]
            norm_inter = torch.norm(diff_inter, p=2) + self.eps  # 防止零梯度
            center_loss += self.beta * norm_inter

            # ================= 模态内差异 =================
            for j in range(self.num_classes):
                if j == p:
                    continue

                # 检查对比类别是否存在样本
                valid_j = (labels == j).sum().item() > 0
                if not valid_j:
                    continue

                # V模态内差异
                diff_intra_V = centers_V[p] - centers_V[j]
                norm_intra_V = torch.norm(diff_intra_V, p=2) + self.eps
                term_V = torch.max(self.rho4_tensor - norm_intra_V, zero_tensor)
                center_loss += term_V

                # I模态内差异
                diff_intra_I = centers_I[p] - centers_I[j]
                norm_intra_I = torch.norm(diff_intra_I, p=2) + self.eps
                term_I = torch.max(self.rho4_tensor - norm_intra_I, zero_tensor)
                center_loss += term_I

        total_loss = identity_loss + center_loss
        # print("identity_loss={} ,center_loss={}".format(identity_loss, center_loss))
        return S_fp_V, S_fp_I, identity_loss, center_loss


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

    def forward(self, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, label):
        # 融合共享特征
        fused_sh = self.shared_fusion(torch.cat([f_sh_hs, f_sh_radar], dim=1))

        # 融合特定特征
        fused_sp = self.specific_fusion(torch.cat([f_sp_hs, f_sp_radar], dim=1))

        # 联合分类
        fused = torch.cat([fused_sh, fused_sp], dim=1)
        output = self.classifier(fused)
        loss = F.cross_entropy(output, label)
        return output, loss


class MultiLayerDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Linear(128, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, num_neg=10):
        super().__init__()
        self.margin = margin
        self.num_neg = num_neg  # 每个样本采样固定数量的负样本

    def forward(self, F_gen, F_real, labels):
        batch_size = F_gen.size(0)
        pos_dist = torch.norm(F_gen - F_real, dim=1)  # (B,)

        # 计算所有样本对的距离矩阵 (B, B)
        neg_dist = torch.norm(F_gen.unsqueeze(1) - F_real.unsqueeze(0), dim=2)  # (B, B)

        loss = 0.0
        for i in range(batch_size):
            # 对每个样本 i，选择其负样本（标签不同的样本）
            neg_mask = (labels != labels[i])  # (B,)
            if neg_mask.sum() == 0:
                continue  # 如果没有负样本，跳过

            # 随机采样固定数量的负样本
            neg_indices = torch.where(neg_mask)[0]
            if len(neg_indices) > self.num_neg:
                neg_indices = neg_indices[torch.randperm(len(neg_indices))[:self.num_neg]]

            # 计算对比损失
            sampled_neg_dist = neg_dist[i, neg_indices]  # (num_neg,)
            loss += torch.relu(pos_dist[i] - sampled_neg_dist + self.margin).mean()

        return loss / batch_size  # 取平均


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.dim = dim

    def forward(self, source, target):
        Q = self.query(source)
        K = self.key(target)
        V = self.value(target)
        attn = self.softmax(Q @ K.transpose(-2, -1) / (self.dim ** 0.5))
        return attn @ V


class EnhancedGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.attn = CrossModalAttention(512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()

    def forward(self, F_sh):
        x = self.relu(self.fc1(F_sh))
        x = self.attn(x, x)  # 自注意力增强特征交互
        return self.fc2(x)


class EnhancedFMCModule(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.generator = EnhancedGenerator(feature_dim, feature_dim)
        self.discriminator = MultiLayerDiscriminator(feature_dim)
        self.identity_classifier = nn.Linear(feature_dim, num_classes)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)

    def forward(self, F_sh_source, F_sp_target, labels):
        F_sp_gen = self.generator(F_sh_source)

        # 对抗损失
        real_score = self.discriminator(F_sp_target)
        fake_score = self.discriminator(F_sp_gen)
        adversarial_loss = -torch.mean(torch.log(real_score + 1e-8)) - torch.mean(torch.log(1 - fake_score + 1e-8))

        # 特征一致性损失
        feature_consistency_loss = F.l1_loss(F_sp_gen, F_sp_target)

        # 身份一致性损失
        identity_scores = self.identity_classifier(F_sp_gen)
        identity_consistency_loss = F.cross_entropy(identity_scores, labels)

        # 对比学习损失
        contrastive_loss = self.contrastive_loss(F_sp_gen, F_sp_target, labels)

        # 动态权重调整
        total_loss = (
                adversarial_loss +
                0.5 * feature_consistency_loss +
                0.5 * identity_consistency_loss +
                0.3 * contrastive_loss
        )

        return F_sp_gen, total_loss



class UnifiedFMCNet(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(num_classes=num_classes, feature_dim=feature_dim)
        self.fmc_hs2radar = FMCModule(feature_dim, num_classes)
        self.fmc_radar2hs = FMCModule(feature_dim, num_classes)
        self.sff = SFFModule(feature_dim, num_classes)
        self.full = HierarchicalFusionClassifier(feature_dim, feature_dim, num_classes)

    def forward(self, x_hs, x_radar, labels):
        # ------------------
        # Step 1: 特征分解
        # ------------------
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, md_loss, id_loss = self.sfd(x_hs, x_radar, labels)

        f_sp_radar_gen, fmc_loss_hs2radar = self.fmc_hs2radar(f_sh_hs, f_sp_radar, labels)
        # 雷达共享特征 -> 高光谱模态特定特征
        f_sp_hs_gen, fmc_loss_radar2hs = self.fmc_radar2hs(f_sh_radar, f_sp_hs, labels)
        f_fp_hs, f_fp_radar, identity_loss, center_loss = self.sff(
            f_sh_hs, f_sp_hs, f_sp_radar_gen,
            f_sh_radar, f_sp_radar, f_sp_hs_gen,
            labels
        )
        full, full_loss = self.full(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels)

        total_loss = fmc_loss_hs2radar + fmc_loss_radar2hs + identity_loss + center_loss + md_loss + id_loss+full_loss
        losss = [md_loss, id_loss, fmc_loss_hs2radar, fmc_loss_radar2hs, identity_loss, center_loss, full_loss]
        return f_fp_hs, f_fp_radar,full,  total_loss, losss


