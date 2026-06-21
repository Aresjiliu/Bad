import torch
import torch.nn as nn

from lib.model_arch_utils import SPP
import torch.nn.functional as F
import random
import os

mdmb_seed = 7
couple_seed = 7


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


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




class Couple_SE_CNN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SEBlock(32),
        )

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    SEBlock(64),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    SEBlock(128)
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


class MDMB_fusion_share_drop(nn.Module):
    def __init__(self, input_channel, args):
        super().__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
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


class MCL_Fusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels*2, in_channels, kernel_size=1, stride=1, padding=0),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(),
                                   )

    def forward(self, teacher_feature, student_feature):
        assert teacher_feature.size() == student_feature.size(), "Teacher and student features must have the same size"
        combined_feature = torch.cat((teacher_feature, student_feature), dim=1)
        mapped_feature = self.block(combined_feature)
        compositional_feature = student_feature + mapped_feature
        return compositional_feature



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