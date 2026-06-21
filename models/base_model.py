import torch
import torch.nn as nn

from lib.model_arch_utils import SPP
import torch.nn.functional as F
from models.resnet import resnet18
import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

mdmb_seed = 7
couple_seed = 7


def generate_unique_random_array_as_float(fan_out, seed):
    # 设置随机数种子
    random.seed(seed)

    # 初始化数组
    random_array = []

    # 定义两个区间的值范围
    first_column_min, first_column_max = 1.0, 2.0  # 第一列的值在 [1.0, 2.0) 内
    second_column_min, second_column_max = 1.0, 5.0  # 第二列的值在 [1.0, 6.0) 内

    # 用于检查值是否重复的集合
    used_values = set()

    # 生成数组，直到数组长度达到 fan_out*2
    while len(random_array) < fan_out:
        # 生成随机的第一列值和第二列值
        first_column_value = random.uniform(first_column_min, first_column_max)
        second_column_value = random.uniform(second_column_min, second_column_max)

        # 检查这两个值是否已经存在于数组中
        if (first_column_value, second_column_value) not in used_values:
            # 如果值尚未使用，则添加到数组和已使用值集合中
            random_array.append([first_column_value, second_column_value])
            used_values.add((first_column_value, second_column_value))
        else:
            # 如果值已经存在，跳过并重新生成
            continue

    return torch.Tensor(random_array).cuda()


class SineFMConv2d(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, fan_out, block_id, c_s=16, stride=1, padding=1, bias=False):
        super(SineFMConv2d, self).__init__()
        self.C_in = c_in
        self.C_s = c_s
        self.C_out = c_out
        self.kernel_size = kernel_size
        self.fan_out = fan_out
        self.block_id = block_id
        self.stride = stride
        self.sin = generate_unique_random_array_as_float(self.fan_out, self.block_id)
        # 定义种子滤波器
        self.seed_conv = nn.Conv2d(c_in, c_s, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        # 定义线性组合层，用于结合种子特征图和生成的特征图
        # 输入通道数为 C_s * fan_out，输出通道数为 C_out
        self.linear_combination = nn.Conv2d(self.C_s * self.fan_out, self.C_out, kernel_size=1, bias=False)

    def sinusoidal_func(self, x, freq_phase):
        # 假设 x 是四维张量，形状为 (batch_size, C_in, H, W)
        B, C_in, H, W = x.size()
        fan_out = freq_phase.size(0)
        C_out = C_in * fan_out

        # 初始化输出张量，这里需要考虑批次大小 B
        output = torch.zeros(B, C_out, H, W).cuda()

        # 对每个输入通道应用正弦变换
        for b in range(B):
            for i in range(fan_out):
                freq, phase = freq_phase[i]
                sine_wave = torch.sin(freq * x[b, i, :, :] + phase)
                output[b, i * C_in:(i + 1) * C_in, :, :] = sine_wave

        return output

    def forward(self, x):
        # 生成种子特征图
        y_s = self.seed_conv(x)
        # 应用非线性变换函数
        y_g = self.sinusoidal_func(y_s, self.sin)
        # 线性组合所有生成的特征图
        y = self.linear_combination(y_g)
        return y


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


class WCRN(nn.Module):
    def __init__(self, num_classes=9):
        super(WCRN, self).__init__()

        self.conv1a = nn.Conv2d(103, 64, kernel_size=3, stride=1, padding=0)
        self.conv1b = nn.Conv2d(103, 64, kernel_size=1, stride=1, padding=0)
        self.maxp1 = nn.MaxPool2d(kernel_size=3)
        self.maxp2 = nn.MaxPool2d(kernel_size=5)

        self.bn1 = nn.BatchNorm2d(128)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv1a(x)
        out1 = self.conv1b(x)
        out = self.maxp1(out)
        out1 = self.maxp2(out1)

        out = torch.cat((out, out1), 1)

        out1 = self.bn1(out)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2a(out1)
        out1 = nn.ReLU()(out1)
        out1 = self.conv2b(out1)

        out = torch.add(out, out1)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)

        return out


class MDMB_extract(nn.Module):
    '''
    More Diverse Means Better: Multimodal Deep Learning Meets Remote Sensing Imagery Classificatio
    '''

    def __init__(self, input_channel):
        super(MDMB_extract, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(input_channel, 16, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(16),
                                    nn.ReLU(),
                                    )
        self.block2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,
                                              bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
                                    )

        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x


class MDMB_fusion(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )

        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x = self.block_6(x)
        x_feature = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = self.block_7(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x_dropout = x
        return x_dropout, x_feature


class MDMB_fusion_dropout(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_dropout, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )

        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x = self.block_6(x)
        x_feature = x
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # x = self.block_7(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x_dropout = self.dropout(x)
        return x_dropout, x_feature


class MDMB_fusion_baseline(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_baseline, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )

        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x = self.block_6(x)
        # x = self.block_7(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        # x = self.dropout(x)
        return x


class MDMB_fusion_spp(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_spp, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )
        self.spp = SPP(merge='max')
        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.class_num = class_num
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x_feature = self.block_6(x)
        x = self.avgpooling(x_feature)
        # x=self.block_7(x)
        # x_whole= x.view(x.shape[0], -1)
        x = x.view(x.shape[0], -1)
        x_whole = self.fc(x)
        # x_whole = self.dropout(x_whole)

        x_spp = self.spp(x_feature)
        feature_num = x_spp.shape[-1]
        patch_score = torch.zeros(x_spp.shape[0], self.class_num, feature_num)
        patch_strength = torch.zeros(x_spp.shape[0], feature_num)

        for i in range(feature_num):
            patch_feature = x_spp[:, :, i]
            patch_strength[:, i] = torch.mean(patch_feature, dim=1)
            # patch_feature = torch.unsqueeze(patch_feature, 2)
            # patch_feature = torch.unsqueeze(patch_feature, 3)
            # patch_logits = self.block_7(patch_feature)
            # patch_logits = patch_logits.view(patch_logits.shape[0], -1)
            patch_logits = self.fc(patch_feature)
            patch_score[:, :, i] = patch_logits

        return x_whole, patch_score, patch_strength


class MDMB_fusion_dad(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_dad, self).__init__()

        self.block_5 = nn.Sequential(
            nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )
        self.spp = SPP()
        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x_feature = self.block_6(x)
        x = self.avgpooling(x_feature)
        # x_whole = self.block_7(x)
        # x_whole = torch.flatten(x_whole, 1)
        x = x.view(x.shape[0], -1)
        x_whole = self.fc(x)
        # x_whole = self.dropout(x_whole)

        return x_whole, x_feature


class MDMB_fusion_middle(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_middle, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(input_channel, 64, kernel_size=1, stride=1, padding=0,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

        self.fc = nn.Linear(128, class_num, bias=True)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_hsi, x_lidar):
        x_hsi = self.block_5_1(x_hsi)
        x_lidar = self.block_5_2(x_lidar)
        x = torch.cat((x_hsi, x_lidar), dim=1)
        feature = x

        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x, feature


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
        self.dropout = nn.Dropout(0.5)
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

        #
        if self.args.location == 'before':
            feature = x
            x = self.pooling(x)
        elif self.args.location == 'after':
            x = self.pooling(x)
            feature = x
        else:
            raise Exception('error location')

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x, feature

class MDMB_fusion_trans(nn.Module):
    def __init__(self, input_channel, args):
        super(MDMB_fusion_trans, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 84, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(84),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(84, 45, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(45),
                                       nn.ReLU(),
                                       )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)
                                            )
        self.fc = nn.Linear(45, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
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

        #
        if self.args.location == 'before':
            feature = x
            x = self.pooling(x)
        elif self.args.location == 'after':
            x = self.pooling(x)
            feature = x
        else:
            raise Exception('error location')

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x, feature

class MDMB_fusion_Dymic(nn.Module):
    def __init__(self, input_channel, dymic_channel, args):
        super(MDMB_fusion_Dymic, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, dymic_channel[0], kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(dymic_channel[0]),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(dymic_channel[0], dymic_channel[1], kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(dymic_channel[1]),
                                       nn.ReLU(),
                                       )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)
                                            )
        self.fc = nn.Linear(dymic_channel[1], args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
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

        #
        if self.args.location == 'before':
            feature = x
            x = self.pooling(x)
        elif self.args.location == 'after':
            x = self.pooling(x)
            feature = x
        else:
            raise Exception('error location')

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x, feature

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

        #
        if self.args.location == 'before':
            feature = x
            x = self.pooling(x)
        elif self.args.location == 'after':
            x = self.pooling(x)
            feature = x
        else:
            raise Exception('error location')
        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        return x, feature


class MDMB_fusion_share_FM(nn.Module):
    def __init__(self, input_channel, args):
        super(MDMB_fusion_share_FM, self).__init__()

        self.block_5_1 = nn.Sequential(SineFMConv2d(input_channel, 128, kernel_size=3, fan_out=5, block_id=1),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(SineFMConv2d(128, 64, kernel_size=3, fan_out=5, block_id=2),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       )

        self.block_6 = nn.Sequential(
            SineFMConv2d(64, args.class_num, kernel_size=1, fan_out=5, block_id=3, padding=0, bias=True),
        )

        self.pooling = nn.AdaptiveAvgPool2d((1, 1)
                                            )
        self.fc = nn.Linear(64, args.class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
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

        #
        if self.args.location == 'before':
            feature = x
            x = self.pooling(x)
        elif self.args.location == 'after':
            x = self.pooling(x)
            feature = x
        else:
            raise Exception('error location')

        # feature = x
        # x = self.pooling(x)
        # # feature = x

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        # x = self.fc(x)
        # x = self.dropout(x)
        return x, feature


class MDMB_fusion_share_tri(nn.Module):
    def __init__(self, input_channel, class_num):
        super(MDMB_fusion_share_tri, self).__init__()

        self.block_5_1 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       )

        self.block_5_2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1,
                                                 bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       nn.AdaptiveAvgPool2d((1, 1))
                                       )

        self.block_6 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True),
                                     )
        self.fc = nn.Linear(64, class_num, bias=True)
        self.dropout = nn.Dropout(0.5)
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

        feature = x

        x = x.view(x.shape[0], -1)

        x = self.fc(x)
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)

        # x = self.fc(x)
        # x = self.dropout(x)
        return x, feature


class CCR(nn.Module):
    '''
    Convolutional Neural Networks for Multimodal Remote Sensing Data Classification
    '''

    def __init__(self, input_channel, class_num):
        super(CCR, self).__init__()
        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     )

        self.block_8 = nn.Sequential(nn.Conv2d(128, input_channel, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(input_channel),
                                     )

        self.fc = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                          bias=True), )
        self.dropout = nn.Dropout(0.5)
        for m in self.modules():
            torch.manual_seed(mdmb_seed)
            torch.cuda.manual_seed(mdmb_seed)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block_5(x)
        x_feature = self.block_6(x)
        x = self.fc(x_feature)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)

        x_rec = self.block_7(x_feature)
        x_rec = self.block_8(x_rec)
        return x, x_rec


class En_De(nn.Module):
    def __init__(self, input_channel, class_num):
        super(En_De, self).__init__()
        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )
        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     )
        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0,
                                               bias=True),
                                     )


class Cross_Fusion(nn.Module):
    def __init__(self, input_channel, class_num):
        super(Cross_Fusion, self).__init__()

        self.block_5 = nn.Sequential(nn.Conv2d(input_channel, 128, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     )

        self.block_6 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0,
                                               bias=False),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((1, 1))
                                     )

        self.block_7 = nn.Sequential(nn.Conv2d(64, class_num, kernel_size=1, stride=1, padding=0),
                                     )


class Couple_CNN(nn.Module):
    def __init__(self, input_channel):
        super(Couple_CNN, self).__init__()
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





class Couple_CNN_Dymic(nn.Module):
    def __init__(self, input_channel, dymic_channel):
        super(Couple_CNN_Dymic, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, dymic_channel[0], kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(dymic_channel[0]),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(nn.Conv2d(dymic_channel[0], dymic_channel[1], kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(dymic_channel[1]),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(nn.Conv2d(dymic_channel[1], dymic_channel[2], kernel_size=3, stride=2, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(dymic_channel[2]),
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

class Couple_CNN_FM(nn.Module):
    def __init__(self, input_channel):
        super(Couple_CNN_FM, self).__init__()
        self.block1 = nn.Sequential(
            SineFMConv2d(input_channel, 32, kernel_size=3, fan_out=5, block_id=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(SineFMConv2d(32, 64, kernel_size=3, fan_out=5, block_id=2),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    )
        self.block3 = nn.Sequential(SineFMConv2d(64, 128, kernel_size=3, fan_out=5, block_id=1, stride=3),
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


class CCR_CNN(nn.Module):
    def __init__(self, input_channel):
        super(CCR_CNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,
                                              bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2)
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


class AlexNet(nn.Module):
    def __init__(self, input_channel):
        super(AlexNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                                    nn.Conv2d(192, 384, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    )
        self.block3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, input_channel):
        super(ResNet18, self).__init__()
        self.model = resnet18(input_channel=input_channel)
        self.model.layer2[0].conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.layer2[0].downsample[0] = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                       bias=False)

        self.block_1 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu)
        self.block_2 = self.model.layer1
        self.block_3 = self.model.layer2

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        # print(x.shape)
        x = self.block_3(x)
        # print(x.shape)

        return x
