import torch.nn as nn
import torch
from models1.base_model import *


class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    )

        for m in self.modules():
            torch.manual_seed(couple_seed)
            torch.cuda.manual_seed(couple_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x



class MDMB_fusion_share(nn.Module):
    def __init__(self, input_channel, args):
        super().__init__()

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
        feature = x
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, feature


class MDMB(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)
        return x




class MDMB_DGD(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_bone = MDMB_fusion_share(256, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)

        hsi_save = x_hsi
        lidar_save = x_lidar

        x = torch.cat((x_hsi, x_lidar), dim=1)
        x, feature = self.share_bone(x)

        x_hsi_out = torch.cat((hsi_save, torch.zeros_like(lidar_save)), dim=1)
        x_hsi_out, _ = self.share_bone(x_hsi_out)

        x_lidar_out = torch.cat((torch.zeros_like(hsi_save), lidar_save), dim=1)
        x_lidar_out, _ = self.share_bone(x_lidar_out)

        return x, feature,  x_hsi_out, x_lidar_out



class MDMB_MCL(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.fusion = MCL_Fusion(128)
        self.share_bone = MDMB_fusion_share(128, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        fused_feature = self.fusion(x_hsi, x_lidar)
        x, feature = self.share_bone(fused_feature)
        return x

class MDMB_MCL_SE(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_SE_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_SE_CNN(input_channel=modality_2_channel)
        self.fusion = MCL_Fusion(128)
        self.share_bone = MDMB_fusion_share(128, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        fused_feature = self.fusion(x_hsi, x_lidar)
        x, feature = self.share_bone(fused_feature)
        return x


class MDMB_MCL_DROP(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_SE_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_SE_CNN(input_channel=modality_2_channel)
        self.fusion = MCL_Fusion(128)
        self.share_bone = MDMB_fusion_share_drop(128, args)
        self.args = args

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        fused_feature = self.fusion(x_hsi, x_lidar)
        x, feature = self.share_bone(fused_feature)
        return x