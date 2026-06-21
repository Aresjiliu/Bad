'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Couple_Share, HSI_Lidar_Couple_Dymic, HSI_Lidar_Couple_Dymic_Sinput
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.multimdal_fusion_config import args
import torch

from lib.model_develop import calc_accuracy_multi, calc_accuracy_multi_first, calc_accuracy_multi_second
import time
from torchstat import stat
from thop import profile
from tqdm import tqdm

def get_acc(model, test_loader, args):
    for param in model.parameters():
        param.requires_grad = False
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")
    print(calc_accuracy_multi(model, test_loader, args, hter=False, verbose=True))


def get_three_acc(model, test_loader, args):
    for param in model.parameters():
        param.requires_grad = False
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")
    print(calc_accuracy_multi(model, test_loader, args, hter=False, verbose=True))
    print(calc_accuracy_multi_first(model, test_loader, args, hter=False, verbose=True))
    print(calc_accuracy_multi_second(model, test_loader, args, hter=False, verbose=True))


def get_fps(model, batch_size, num_batches):
    # 确保模型在评估模式
    model.eval()

    # 生成随机输入数据并移动到CUDA
    input1 = torch.randn(batch_size, 244, 7, 7).cuda()  # 第一种模态数据
    input2 = torch.randn(batch_size, 4, 7, 7).cuda()  # 第二种模态数据

    # 测量推理时间
    start_time = time.time()

    with torch.no_grad():  # 不计算梯度
        for _ in tqdm(range(num_batches), desc='Testing FPS', unit='batch'):
            model(input1, input2)  # 批量推理

    end_time = time.time()

    # 计算总时间和FPS
    total_time = end_time - start_time
    total_samples = batch_size * num_batches
    fps = total_samples / total_time

    print(f"Processed {total_samples} samples in {total_time:.2f} seconds.")
    print(f"FPS: {fps:.2f}")

def get_channel(weight):
    channel = []
    for name, value in weight.items():
        if name.endswith("0.weight"):
            channel.append(value.shape[0])
    return channel


def conpact_load(args, path):
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    saved_model = torch.load(path)
    channels = get_channel(saved_model)
    model = HSI_Lidar_Couple_Dymic(args, modality_1_channel, modality_2_channel, channels)
    model.load_state_dict(saved_model)
    get_three_acc(model, test_loader, args)


def common_load(args, path):
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    saved_model = torch.load(path)
    model = HSI_Lidar_Couple_Share(args, modality_1_channel, modality_2_channel)
    model.load_state_dict(saved_model)
    get_three_acc(model, test_loader, args)

def get_stat(args, path):
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    saved_model = torch.load(path)
    channels = get_channel(saved_model)

    test_loader = huston2013_multi_dataloader(train=False, args=args)
    model1 = HSI_Lidar_Couple_Dymic(args, modality_1_channel, modality_2_channel, channels)
    model1.load_state_dict(saved_model)
    get_three_acc(model1, test_loader, args)
    get_fps(model1, 32, 30000)

    model = HSI_Lidar_Couple_Dymic_Sinput(args, modality_1_channel, modality_2_channel, channels)
    stat(model, (248, 7, 7))





def get_profile(args, path):
    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    saved_model = torch.load(path)
    channels = get_channel(saved_model)
    model = HSI_Lidar_Couple_Dymic_Sinput(args, modality_1_channel, modality_2_channel, channels)
    for param in model.parameters():
        param.requires_grad = False
    input = torch.randn(1, 145, 7, 7)
    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops}, Params: {params}")


if __name__ == '__main__':
    # conpact_load(args=args, path="../output/models/transfer/0.7fin.pth")
    get_stat(args=args, path="../output/models/transfer_berlin/1.0fin.pth")

