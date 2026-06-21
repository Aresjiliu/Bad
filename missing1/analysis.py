import sys

sys.path.append('..')
from models1 import *
from config import args
import torch
from thop import profile


def analyze_sfd_module():
    # 创建SFD模块的包装类
    class SFDWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            hsi = x[:, :144, :, :]
            lidar = x[:, 144:, :, :]
            return self.model(hsi, lidar)

    # 初始化模型
    sfd = SFDModule(hsi_channels=144, radar_channels=1, feature_dim=128)
    model = SFDWrapper(sfd)

    # 创建输入
    input_tensor = torch.randn(1, 145, 7, 7)

    # 计算FLOPs和参数
    flops, params = profile(model, inputs=(input_tensor,))
    print("=== SFDModule 分析 ===")
    print(f"FLOPs: {flops / 1e6:.2f}M")
    print(f"Parameters: {params / 1e3:.2f}K")


def analyze_fusion_classifier():
    # 创建分类器的包装类
    class ClassifierWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # 将输入拆分为4个向量
            split_size = x.shape[1] // 4
            f_sh_hs = x[:, :split_size]
            f_sh_radar = x[:, split_size:split_size * 2]
            f_sp_hs = x[:, split_size * 2:split_size * 3]
            f_sp_radar = x[:, split_size * 3:]
            return self.model(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar)

    # 初始化模型 (假设feature_dim=256)
    classifier = HierarchicalFusionClassifier(256, 256, args.class_num)
    model = ClassifierWrapper(classifier)

    # 创建输入 (4个256维向量拼接)
    input_tensor = torch.randn(1, 1024)

    # 计算FLOPs和参数
    flops, params = profile(model, inputs=(input_tensor,))
    print("\n=== HierarchicalFusionClassifier 分析 ===")
    print(f"FLOPs: {flops / 1e6:.2f}M")
    print(f"Parameters: {params / 1e3:.2f}K")


if __name__ == '__main__':
    analyze_sfd_module()
    analyze_fusion_classifier()