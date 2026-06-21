import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os


# 假设 SPP 模块在此定义
class SPP(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SPP, self).__init__()
        self.pool_sizes = pool_sizes
        self.poolings = nn.ModuleList([nn.AdaptiveMaxPool2d(output_size=pool_size) for pool_size in pool_sizes])

    def forward(self, x):
        features = [x]
        for pool in self.poolings:
            pooled = pool(x)
            # 将池化后的特征图上采样回原始空间尺寸
            pooled = F.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=True)
            features.append(pooled)
        return torch.cat(features, dim=1)  # 在通道维度拼接


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


mdmb_seed = 7
couple_seed = 7
set_seed(mdmb_seed)


# 通道注意力模块（SE块）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channels, 1, 1)
        return x * y


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B,1,H,W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B,1,H,W]
        x = torch.cat([avg_out, max_out], dim=1)  # [B,2,H,W]
        x = self.conv(x)  # [B,1,H,W]
        return self.sigmoid(x)  # [B,1,H,W]


# CBAM模块（结合通道和空间注意力）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # [B, C, H, W]
        attention = self.spatial_attention(x)  # [B,1,H,W]
        x = x * attention  # 保持 [B, C, H, W]
        return x


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=True):
        super(ResidualBlock, self).__init__()
        self.use_cbam = use_cbam
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if self.use_cbam:
            self.cbam = CBAM(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)  # [B, out_channels, H, W]
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # [B, out_channels, H, W]
        out = self.bn2(out)

        if self.use_cbam:
            out = self.cbam(out)  # [B, out_channels, H, W]

        if self.downsample is not None:
            identity = self.downsample(x)  # [B, out_channels, H, W]

        out += identity  # [B, out_channels, H, W]
        out = self.relu(out)
        return out


# 修改后的Couple_CNN模块，减少下采样，适应小输入尺寸
class Couple_CNN(nn.Module):
    def __init__(self, input_channel, pool_sizes=[1, 2, 4]):
        super(Couple_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer2 = ResidualBlock(64, 128, stride=1, downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(128),
        ))
        self.layer3 = ResidualBlock(128, 256, stride=1, downsample=nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        ))
        self.spp = SPP(pool_sizes=pool_sizes)  # SPP 模块
        # SPP 输出通道数 = 256 * (1 + len(pool_sizes)) = 256 * 4 = 1024
        self.attention = CBAM(256 * (1 + len(pool_sizes)))  # CBAM 的通道数应为 1024

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.layer1(x)  # [B, 64, 7, 7]
        x = self.layer2(x)  # [B, 128, 7, 7]
        x = self.layer3(x)  # [B, 256, 7, 7]
        x = self.spp(x)  # [B, 1024, 7, 7]
        x = self.attention(x)  # 添加CBAM，保持 [B, 1024, 7, 7]
        return x


# 模态注意力融合模块
class ModalityAttentionFusion(nn.Module):
    def __init__(self, hsi_channels, lidar_channels):
        super(ModalityAttentionFusion, self).__init__()
        # 1x1卷积调整通道数
        self.hsi_conv = nn.Conv2d(hsi_channels, 256, kernel_size=1, bias=False)
        self.lidar_conv = nn.Conv2d(lidar_channels, 256, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        # 注意力权重
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 2, 256, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, hsi, lidar):
        hsi = self.hsi_conv(hsi)  # [B, 256, 7, 7]
        lidar = self.lidar_conv(lidar)  # [B, 256, 7, 7]
        combined = torch.cat([hsi, lidar], dim=1)  # [B, 512, 7, 7]
        weights = self.attention(combined)  # [B, 2, 1, 1]
        hsi_weight, lidar_weight = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        fused = hsi * hsi_weight + lidar * lidar_weight  # 加权融合 [B, 256, 7, 7]
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused


# MDMB_fusion_share模块
class MDMB_fusion_share(nn.Module):
    def __init__(self, input_channel, args):
        super().__init__()

        self.block_5_1 = nn.Sequential(
            nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.block_5_2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, args.class_num, bias=True)
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5_1(x)
        x = self.block_5_2(x)
        x = self.pooling(x)
        feature = x
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, feature


# MCL_Fusion模块
class MCL_Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cbam = CBAM(in_channels * 2)  # 输入通道数为 2 * in_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def forward(self, teacher_feature, student_feature):
        assert teacher_feature.size() == student_feature.size(), "Teacher and student features must have the same size"
        combined_feature = torch.cat((teacher_feature, student_feature), dim=1)  # [B, 512, 7, 7]
        combined_feature = self.cbam(combined_feature)  # [B, 512, 7, 7]
        mapped_feature = self.block(combined_feature)  # [B, 256, 7, 7]
        compositional_feature = student_feature + mapped_feature  # [B, 256, 7, 7]
        return compositional_feature


# 最终的MDMB_MCL模型
class MDMB_MCL(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel, pool_sizes=[1, 2, 4]):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel, pool_sizes=pool_sizes)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel, pool_sizes=pool_sizes)
        self.modality_fusion = ModalityAttentionFusion(
            hsi_channels=256 * (1 + len(pool_sizes)),
            lidar_channels=256 * (1 + len(pool_sizes))
        )  # [B, 256, 7, 7]
        self.fusion = MCL_Fusion(256)  # 输入通道数为256，输出为256
        self.share_bone = MDMB_fusion_share(256, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)  # [B, 1024, 7, 7]
        x_lidar = self.special_bone_lidar(lidar)  # [B, 1024, 7, 7]
        fused_modality = self.modality_fusion(x_hsi, x_lidar)  # [B, 256, 7, 7]
        fused_feature = self.fusion(fused_modality, fused_modality)  # [B, 256, 7, 7]
        x, feature = self.share_bone(fused_feature)  # [B, class_num], [B, 128, 1, 1]
        return x


