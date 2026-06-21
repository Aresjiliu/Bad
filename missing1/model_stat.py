import sys
sys.path.append('..')
from models1 import *
from config import args
from torchstat import stat


class CombinedInputModel(nn.Module):
    def __init__(self, model):
        super(CombinedInputModel, self).__init__()
        self.model = model

    def forward(self, x):
        # 假设 x 包含两个模态，沿通道维度合并
        hsi = x[:, :144, :, :]   # [B, 144, 7, 7]
        lidar = x[:, 144:, :, :] # [B, 1, 7, 7]
        return self.model(hsi, lidar)


def deeppix_main(args):
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    original_model = Base(args, modality_1_channel, modality_2_channel, feature_dim=128)
    model = CombinedInputModel(original_model).to('cpu')
    model.eval()

    stat(model, (145, 7, 7))

if __name__ == '__main__':
    deeppix_main(args=args)

