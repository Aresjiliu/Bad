import sys

sys.path.append('..')
from models1.gate_model2 import *
from models1.config import args
import torch
import torch.nn as nn
from models1.train_model import *
import torch.optim as optim

import numpy as np
import random
import os
from thop import profile, clever_format

class CombinedInputModel(nn.Module):
    def __init__(self, model):
        super(CombinedInputModel, self).__init__()
        self.model = model

    def forward(self, x):
        # 假设 x 包含两个模态，沿通道维度合并
        hsi = x[:, :144, :, :]   # [B, 144, 7, 7]
        lidar = x[:, 144:, :, :] # [B, 1, 7, 7]
        return self.model(hsi, lidar)


# 定义模型和输入
modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
modality_1_channel = modality_to_channel[args.pair_modalities[0]]
modality_2_channel = modality_to_channel[args.pair_modalities[1]]
original_model = MDMB(args, modality_1_channel, modality_2_channel)
model = CombinedInputModel(original_model).to('cpu')
model.eval()

# 创建一个虚拟输入
input_tensor = torch.randn(1, 145, 7, 7)  # 根据你的模型输入调整形状

# 使用 thop 统计 FLOPs 和参数量
flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
flops, params = clever_format([flops, params], "%.3f")  # 格式化输出

print(f"Total parameters: {params}")
print(f"Total FLOPs: {flops}")