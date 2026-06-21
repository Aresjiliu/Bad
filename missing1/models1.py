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
    def __init__(self, hsi_channels=144, radar_channels=1, feature_dim=128):
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
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )

        # 模态特定特征提取器 (独立参数)
        self.E_sp_hs = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 输出 (B,256,1,1)
        )
        self.E_sp_radar = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x_hs, x_radar):
        # 基础特征提取
        f_hs = self.hs_stream(x_hs)  # (B,128,7,7)
        f_radar = self.radar_stream(x_radar)  # (B,128,7,7)

        # 特征分解
        f_sh_hs = self.E_sh(f_hs).flatten(1)  # (B,256)
        f_sh_radar = self.E_sh(f_radar).flatten(1)  # 共享参数

        f_sp_hs = self.E_sp_hs(f_hs).flatten(1)  # (B,256)
        f_sp_radar = self.E_sp_radar(f_radar).flatten(1)
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



class SingleClassifier(nn.Module):
    def __init__(self, sp_dim, num_classes):
        super().__init__()
        # 共享特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(sp_dim * 2, 128),
            nn.ReLU()
        )

        # 最终分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, f_sh, f_sp):
        # 融合特定特征
        fused = self.feature_fusion(torch.cat([f_sh, f_sp], dim=1))

        output = self.classifier(fused)
        return output

class SFD_Single(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.full = HierarchicalFusionClassifier(feature_dim, feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        output = self.full(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)
        return output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar



class SFD_CKD(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(hsi_channels=hsi_channels, radar_channels=radar_channels, feature_dim=feature_dim)
        self.hs_cls = SingleClassifier( feature_dim, args.class_num)
        self.radar_cls = SingleClassifier(feature_dim, args.class_num)

    def forward(self, x_hs, x_radar):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = self.sfd(x_hs, x_radar)
        hs_output = self.hs_cls(f_sh_hs, f_sp_hs)
        radar_output = self.radar_cls(f_sh_radar, f_sp_radar)
        return hs_output,radar_output, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar



class MDMB_fusion_share(nn.Module):
    def __init__(self, input_channel, args):
        super(MDMB_fusion_share, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)
                                            )
        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.args = args
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Base(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels, feature_dim=256):
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

        self.classfier = MDMB_fusion_share(feature_dim * 2, args)

    def forward(self, x_hs, x_radar):
        f_hsi = self.hs_stream(x_hs)
        f_radar = self.radar_stream(x_radar)
        output = self.classfier(torch.cat([f_hsi, f_radar], dim=1))
        return output


import torch
import torch.nn as nn
import torch.nn.functional as F

