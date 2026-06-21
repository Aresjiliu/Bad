import torch
import torch.nn as nn

from lib.model_arch_utils import SPP
import torch.nn.functional as F
from models.resnet import resnet18
import random
import os

mdmb_seed = 7
couple_seed = 7
def custom_bernoulli(probs):
    probs = torch.clamp(probs, min=0, max=1)
    unifs = torch.rand_like(probs)
    return (unifs < probs).float()


class Conv2d_Prune(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 osc=0.85):
        super(Conv2d_Prune, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)

        self.sc = nn.Parameter(torch.tensor([osc] * out_channels), requires_grad=True)
        self.temp = nn.Parameter(torch.tensor([1.0] * out_channels), requires_grad=False)
        self.masked_weight = torch.zeros_like(self.weight)
        self.saved_weight = nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.qc = torch.tensor([0.0] * out_channels)
        self.generate_qc()

    def set_temp(self, temp):
        self.temp.data = temp

    def generate_qc(self):
        mask_c = torch.sigmoid(self.temp * self.sc)
        self.qc = custom_bernoulli(mask_c).cuda()

    def forward(self, x):
        if self.training:
            mask_c = torch.sigmoid(self.temp * self.sc)
            self.masked_weight = self.weight * self.qc.view(-1, 1, 1, 1) * mask_c.view(-1, 1, 1, 1)
            out = F.conv2d(x, self.masked_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return out
        else:
            return F.conv2d(x, self.saved_weight.cuda(), self.bias, self.stride, self.padding, self.dilation, self.groups)


class Couple_CNN_Prune(nn.Module):
    def __init__(self, input_channel, osc):
        super(Couple_CNN_Prune, self).__init__()
        self.block1 = nn.Sequential(
            Conv2d_Prune(input_channel, 32, kernel_size=3, stride=1, padding=1,
                         bias=False, osc=osc),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(Conv2d_Prune(32, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False, osc=osc),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(Conv2d_Prune(64, 128, kernel_size=3, stride=2, padding=1,
                                                 bias=False, osc=osc),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    # nn.MaxPool2d(kernel_size=2)
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



class MDMB_fusion_prune(nn.Module):
    def __init__(self, input_channel, args):
        super(MDMB_fusion_prune, self).__init__()

        self.block_5_1 = nn.Sequential(Conv2d_Prune(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                    bias=False, osc=args.osc),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(Conv2d_Prune(128, 64, kernel_size=3, stride=1, padding=1,
                                                    bias=False, osc=args.osc),
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



class HSI_Lidar_Couple_Prune(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()

        self.special_bone_hsi = Couple_CNN_Prune(modality_1_channel, args.osc)
        self.special_bone_lidar = Couple_CNN_Prune(modality_2_channel, args.osc)
        self.share_bone = MDMB_fusion_prune(256, args)
        self.p = args.p
        self.gama = args.gama
        self.t_max = args.t_max
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

        return x, x_hsi_out, x_lidar_out

    def l1_loss(self, epoch):
        total_l1_loss = 0
        for module in self.modules():
            if isinstance(module, Conv2d_Prune):
                maskc = torch.sigmoid(module.temp * module.sc)
                total_l1_loss += torch.norm(maskc, 1)
        return total_l1_loss

    def update_temp(self):
        for module in self.modules():
            if isinstance(module, Conv2d_Prune):
                module.generate_qc()
                maskc = torch.sigmoid(module.temp * module.sc)
                prob = custom_bernoulli(maskc)
                temp = module.temp.data*torch.pow(self.gama, prob)
                module.set_temp(torch.clamp(temp, max=self.t_max))