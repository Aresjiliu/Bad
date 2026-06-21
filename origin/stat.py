import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Couple_sinput
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.multimdal_fusion_config import args
import torch
from torchstat import stat
from lib.model_develop import calc_accuracy_multi

import time

def get_time(model, test_loader, args):
    for param in model.parameters():
        param.requires_grad = False
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")
    start = time.time()
    print(calc_accuracy_multi(model, test_loader, args, hter=False, verbose=True))
    end = time.time()
    print("the programe spent "+str(end-start)+" s")

def deeppix_main(args):
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    args.log_name = args.name
    args.model_name = args.name

    if isinstance(args.p, str):
        args.p = eval(args.p)

    args.model_root = args.model_root + "/prune_huston"
    args.log_root = args.log_root + "/prune_huston"


    modality_to_channel = {'hsi': 144, 'ms': 8, 'lidar': 1}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = HSI_Lidar_Couple_sinput(args, modality_1_channel, modality_2_channel)
    # path1 = "../output/models/share_unimodel_center_drop/check_dad.pth"
    # path1 = "../output/models/transfer/0.5fin.pth"
    # saved_model = torch.load(path1)
    # model.load_state_dict(saved_model)
    for param in model.parameters():
        param.requires_grad = False
    # 如果有GPU
    # if torch.cuda.is_available():
    #     model.cuda()  # 将所有的模型参数移动到GPU上
    # test_time(model, test_loader, args)
    #
    # stat(model, (145, 7, 7))




if __name__ == '__main__':
    deeppix_main(args=args)
