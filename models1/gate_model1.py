import torch
import torch.nn as nn

class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x  # 输出深层特征


class GatedFusion(nn.Module):
    def __init__(self, hsi_channel):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(hsi_channel, hsi_channel, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, hsi_feat, radar_feat):
        gate = self.gate(hsi_feat)
        fused_feat = hsi_feat + gate * radar_feat
        return fused_feat

class MDMB(nn.Module):
    def __init__(self, args, hsi_channels, radar_channels):
        super().__init__()
        self.hsi_net = Couple_CNN(hsi_channels)
        self.radar_net = Couple_CNN(radar_channels)
        self.fusion1 = GatedFusion(16)
        self.fusion2 = GatedFusion(32)
        self.fusion3 = GatedFusion(64)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, args.class_num),
        )

    def forward(self, hsi, radar):
        hsi_f1 = self.hsi_net.block1(hsi)
        radar_f1 = self.radar_net.block1(radar)
        fused_f1 = self.fusion1(hsi_f1, radar_f1)

        hsi_f2 = self.hsi_net.block2(fused_f1)
        radar_f2 = self.radar_net.block2(radar_f1)
        fused_f2 = self.fusion2(hsi_f2, radar_f2)

        hsi_f3 = self.hsi_net.block3(fused_f2)
        radar_f3 = self.radar_net.block3(radar_f2)
        fused_f3 = self.fusion3(hsi_f3, radar_f3)

        output = self.classifier(fused_f3)
        return output