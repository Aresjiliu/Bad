'''
a template for train, you need to fix your own main function
'''

import sys

sys.path.append('..')
from models1.resnet_ensemble import *
from src.huston2013_dataloader import huston2013_multi_dataloader
from models1.config import args
import torch
import torch.nn as nn
from models1.train_model import *
import torch.optim as optim

import numpy as np
import random
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def deeppix_main(args):
    train_loader = huston2013_multi_dataloader(train=True, args=args)
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    args.log_name = args.name
    args.model_name = args.name

    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    model = MDMB_MCL_SE(args, modality_1_channel, modality_2_channel)
    # 如果有GPU
    if torch.cuda.is_available():
        model.cuda()  # 将所有的模型参数移动到GPU上
        print("GPU is using")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(filter(lambda param: param.requires_grad, model.parameters()), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)



    args.retrain = False
    train_mcl(model=model, cost=criterion, optimizer=optimizer,
                                                 train_loader=train_loader,
                                                 test_loader=test_loader,
                                                 args=args)


if __name__ == '__main__':
    deeppix_main(args=args)
