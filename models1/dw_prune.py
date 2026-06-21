import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os


# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


mdmb_seed = 7
couple_seed = 7
set_seed(mdmb_seed)

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



# 定义深度可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, osc=0.85):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = Conv2d_Prune(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels, bias=False, osc=osc)
        self.pointwise = Conv2d_Prune(in_channels, out_channels, kernel_size=1, bias=False, osc=osc)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)  # [B, C, H, W]
        x = self.pointwise(x)  # [B, C', H, W]
        x = self.bn(x)
        x = self.relu(x)
        return x


# 定义优化后的 Couple_CNN 模块
class Couple_CNN(nn.Module):
    def __init__(self, input_channel,osc):
        super(Couple_CNN, self).__init__()
        self.block1 = DepthwiseSeparableConv(input_channel, 32, kernel_size=3, stride=1, padding=1, osc=osc)
        self.block2 = DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1, osc=osc)
        self.block3 = DepthwiseSeparableConv(64, 128, kernel_size=3, stride=2, padding=1, osc=osc)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)  # [B,32,7,7]
        x = self.block2(x)  # [B,64,7,7]
        x = self.block3(x)  # [B,128,4,4]
        return x


# 定义 MCL_Fusion 模块，使用深度可分离卷积和 SEBlock
class MCL_Fusion(nn.Module):
    def __init__(self, in_channels, osc):
        super().__init__()
        self.block = DepthwiseSeparableConv(in_channels * 2, in_channels, kernel_size=1, stride=1, padding=0, osc=osc)

    def forward(self, teacher_feature, student_feature):
        assert teacher_feature.size() == student_feature.size(), "Teacher and student features must have the same size"
        combined_feature = torch.cat((teacher_feature, student_feature), dim=1)  # [B, 256, 4, 4]
        mapped_feature = self.block(combined_feature)  # [B,128,4,4]
        compositional_feature = student_feature + mapped_feature  # [B,128,4,4]
        return compositional_feature


# 定义优化后的 MDMB_fusion_share 模块
class MDMB_fusion_share(nn.Module):
    def __init__(self, input_channel, args):
        super().__init__()
        self.block_5_1 = DepthwiseSeparableConv(input_channel, 128, kernel_size=3, stride=1, padding=1, osc=args.osc)

        self.block_5_2 = DepthwiseSeparableConv(128, 64, kernel_size=3, stride=1, padding=1, osc=args.osc)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.args = args

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5_1(x)  # [B,64,4,4]
        x = self.block_5_2(x)  # [B,32,4,4]
        x = self.pooling(x)  # [B,32,1,1]
        feature = x
        x = x.view(x.shape[0], -1)  # [B,32]
        x = self.fc(x)  # [B,class_num]
        return x, feature


# 定义优化后的 MDMB_MCL 模型
class MDMB_MCL(nn.Module):
    def __init__(self, args, modality_1_channel, modality_2_channel):
        super().__init__()
        self.special_bone_hsi = Couple_CNN(input_channel=modality_1_channel, osc=args.osc)
        self.special_bone_lidar = Couple_CNN(input_channel=modality_2_channel, osc=args.osc)
        self.fusion = MCL_Fusion(128, args.osc)  # 输入通道数为128，输出通道数保持128
        self.share_bone = MDMB_fusion_share(128, args)
        self.args = args
        self.t_max = args.t_max
        self.gama = args.gama

    def forward(self, hsi, lidar):
        x_hsi = self.special_bone_hsi(hsi)  # [B,128,4,4]
        x_lidar = self.special_bone_lidar(lidar)  # [B,128,4,4]
        fused_feature = self.fusion(x_hsi, x_lidar)  # [B,128,4,4]
        x, feature = self.share_bone(fused_feature)  # [B,class_num], [B,32,1,1]
        return x

    def l1_loss(self):
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
