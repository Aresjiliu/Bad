import torch
import torch.nn as nn

from lib.model_arch_utils import SPP
import torch.nn.functional as F
import random
import os

from models import *


mdmb_seed = 7
couple_seed = 7




class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
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
        return x


class Classfier(nn.Module):
    def __init__(self, input_channel, args):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                             bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((1, 1),))
        self.fc = nn.Linear(128, args.class_num, bias=True)
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
        x = self.block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class Base(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel)
        self.share_c = Classfier(256, args)
        self.hsi_c = Classfier(128, args)
        self.lidar_c = Classfier(128, args)
        self.args = args

    def forward(self, hsi, lidar, labels):
        x_hsi = self.special_bone_hsi(hsi)
        x_lidar = self.special_bone_lidar(lidar)
        share_output = self.share_c(torch.cat((x_hsi, x_lidar), dim=1))
        hsi_output = self.hsi_c(x_hsi)
        lidar_output = self.lidar_c(x_lidar)
        share_loss = F.cross_entropy(share_output, labels)
        hsi_loss = F.cross_entropy(hsi_output, labels)
        lidar_loss = F.cross_entropy(lidar_output, labels)
        full_loss = share_loss+hsi_loss+lidar_loss
        losses = [share_loss, hsi_loss, lidar_loss]
        return share_output, hsi_output, lidar_output, full_loss, losses

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



class SingleFusionClassifier(nn.Module):
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
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, f_sh, f_sp,label):
        # 融合共享特征
        fused_single = self.shared_fusion(torch.cat([f_sh, f_sp], dim=1))
        output = self.classifier(fused_single)
        loss = F.cross_entropy(output, label)
        return output, loss



class Base_SFD(nn.Module):
    def __init__(self, num_classes=10, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(num_classes=num_classes, feature_dim=feature_dim)
        self.shared_classifier1 = nn.Linear(4 * feature_dim, num_classes)
        self.shared_classifier2 = nn.Linear(2 * feature_dim, num_classes)
        self.shared_classifier3 = nn.Linear(2 * feature_dim, num_classes)

    def forward(self, x_hs, x_radar, labels):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, md_loss, id_loss = self.sfd(x_hs, x_radar, labels)
        share_output = self.shared_classifier1(torch.cat((f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar), dim=1))
        hsi_output = self.shared_classifier2(torch.cat((f_sh_hs, f_sp_hs), dim=1))
        lidar_output = self.shared_classifier3(torch.cat((f_sh_radar, f_sp_radar), dim=1))
        share_loss = F.cross_entropy(share_output, labels)
        hsi_loss = F.cross_entropy(hsi_output, labels)
        lidar_loss = F.cross_entropy(lidar_output, labels)
        full_loss = share_loss+hsi_loss+lidar_loss+md_loss
        losses = [share_loss, hsi_loss, lidar_loss, md_loss]
        return share_output, hsi_output, lidar_output, full_loss, losses



class Base_SFD1(nn.Module):
    def __init__(self, num_classes=15, feature_dim=256):
        super().__init__()
        self.sfd = SFDModule(num_classes=num_classes, feature_dim=feature_dim)
        self.shared_classifier1 = HierarchicalFusionClassifier(feature_dim, feature_dim, num_classes)

    def forward(self, x_hs, x_radar, labels):
        f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, md_loss, id_loss = self.sfd(x_hs, x_radar, labels)
        full, share_loss = self.shared_classifier1(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels)
        full_loss = share_loss+md_loss+id_loss
        losses = [md_loss, id_loss, share_loss]
        return full, full, full, full_loss, losses

