import sys
sys.path.append('..')
from models1 import *
from config import args
import torch
from thop import profile  # 替换为 thop 库
from models.resnet_ensemble import HSI_Lidar_Couple_Share
class CombinedInputModel(nn.Module):
    def __init__(self, model):
        super(CombinedInputModel, self).__init__()
        self.model = model

    def forward(self, x):
        hsi = x[:, :144, :, :]   # [B, 144, 7, 7]
        lidar = x[:, 144:, :, :] # [B, 1, 7, 7]
        return self.model(hsi, lidar)

def deeppix_main(args):
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    original_model = SFD_Single(args, modality_1_channel, modality_2_channel, feature_dim=128)
    # original_model = HSI_Lidar_Couple_Share(args, modality_1_channel, modality_2_channel)
    model = CombinedInputModel(original_model).to('cpu')
    model.eval()

    # 使用 thop 计算 FLOPs 和参数
    input_tensor = torch.randn(1, 145, 7, 7)  # 创建虚拟输入
    flops, params = profile(model, inputs=(input_tensor, ))
    print(f"FLOPs: {flops/1e6:.2f} M")
    print(f"Params: {params/1e3:.2f} K")

if __name__ == '__main__':
    deeppix_main(args=args)