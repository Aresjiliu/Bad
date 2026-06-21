'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models.resnet_ensemble import HSI_Lidar_Couple_Prune, HSI_Lidar_Couple_Share
from src.huston2013_dataloader import huston2013_multi_dataloader
from configuration.prune_config import args
from lib.model_develop import calc_accuracy_multi
import torch
import torch.nn as nn
from lib.model_develop import train_base_multi_share_unimodal_center_prune
from lib.processing_utils import get_file_list
import torch.optim as optim

import numpy as np
import datetime
import random
import os
import time



def deeppix_main(args):
    train_loader = huston2013_multi_dataloader(train=True, args=args)
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

    model = HSI_Lidar_Couple_Prune(args, modality_1_channel, modality_2_channel)
    saved_model = torch.load("../output/models/prune_huston_succ/hsi_lidar_osc_0.5_l1_0.001_gama_1.01.pth")
    model.load_state_dict(saved_model)
    model.eval()
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


if __name__ == '__main__':
    deeppix_main(args=args)
