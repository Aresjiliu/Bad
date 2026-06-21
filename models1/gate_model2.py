import torch.nn as nn
import torch
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


class DynamicGatedFusion(nn.Module):
    """动态门控融合（跨模态注意力机制）"""

    def __init__(self, hsi_ch, sar_ch):
        super().__init__()
        self.query = nn.Conv2d(hsi_ch, hsi_ch // 4, kernel_size=1)  # Query生成
        self.key = nn.Conv2d(sar_ch, hsi_ch // 4, kernel_size=1)  # Key生成
        self.value = nn.Conv2d(sar_ch, hsi_ch, kernel_size=1)  # Value生成
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hsi_feat, sar_feat):
        B, C, H, W = hsi_feat.size()
        # 生成Q/K/V
        Q = self.query(hsi_feat).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//4)
        K = self.key(sar_feat).view(B, -1, H * W)  # (B, C//4, HW)
        V = self.value(sar_feat).view(B, -1, H * W)  # (B, C, HW)

        # 注意力权重计算
        attn = torch.bmm(Q, K) / (C ** 0.5)  # (B, HW, HW)
        attn = self.softmax(attn)

        # 加权融合
        fused = torch.bmm(V, attn.permute(0, 2, 1))  # (B, C, HW)
        fused = fused.view(B, -1, H, W)  # (B, C, H, W)
        return hsi_feat + fused  # 残差连接


class MDMB(nn.Module):
    def __init__(self, args, hsi_channels=144, radar_channels=4):
        super().__init__()
        # 高光谱分支：光谱-空间解耦
        self.hsi_block1 = SS_DecoupledBlock(hsi_channels, 16)
        self.hsi_block2 = SS_DecoupledBlock(16, 32)
        self.hsi_block3 = SS_DecoupledBlock(32, 64)

        # 雷达分支：方向梯度增强
        self.radar_block1 = OrientedGradientBlock(radar_channels, 16)
        self.radar_block2 = OrientedGradientBlock(16, 32)
        self.radar_block3 = OrientedGradientBlock(32, 64)

        # 动态门控融合
        self.fusion1 = DynamicGatedFusion(16, 16)
        self.fusion2 = DynamicGatedFusion(32, 32)
        self.fusion3 = DynamicGatedFusion(64, 64)

        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, args.class_num)
        )

    def forward(self, hsi, radar):
        # Stage 1
        hsi_f1 = self.hsi_block1(hsi)  # (B,16,7,7)
        radar_f1 = self.radar_block1(radar)  # (B,16,7,7)
        fused_f1 = self.fusion1(hsi_f1, radar_f1)

        # Stage 2
        hsi_f2 = self.hsi_block2(fused_f1)  # (B,32,7,7)
        radar_f2 = self.radar_block2(radar_f1)
        fused_f2 = self.fusion2(hsi_f2, radar_f2)

        # Stage 3
        hsi_f3 = self.hsi_block3(fused_f2)  # (B,64,7,7)
        radar_f3 = self.radar_block3(radar_f2)
        fused_f3 = self.fusion3(hsi_f3, radar_f3)

        # Classification
        output = self.classifier(fused_f3)
        return output