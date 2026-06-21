'''模型训练相关的函数'''

import sys

sys.path.append('..')
from models1.gate_model import *
from src.huston2013_dataloader import huston2013_multi_dataloader
from models1.config import args
import torch
import torch.nn as nn
from models1.train_model import *
import torch.optim as optim

import numpy as np
import random
import os




def calc_accuracy_multi(model, loader, args, verbose=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    num_classes = args.class_num
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            outputs_batch = model(img_m1, img_m2)
            if isinstance(outputs_batch, tuple):
                output_batch = outputs_batch[0]
            else:
                output_batch = outputs_batch

            outputs_full.append(output_batch)
            labels_full.append(target)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)

    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)
    print(f"整体准确率：{accuracy}")

    correct_counts = [0] * num_classes
    total_counts = [0] * num_classes

    # 遍历所有预测结果和标签
    for predicted, label in zip(labels_predicted, labels_full):
        # 更新总数
        total_counts[label.item()] += 1
        # 如果预测正确，更新正确预测数
        if predicted.item() == label.item():
            correct_counts[label.item()] += 1

    # 计算每个类的分类精度
    class_accuracies = [correct_counts[i] / total_counts[i] if total_counts[i] != 0 else 0 for i in range(num_classes)]
    print(class_accuracies)
    predict_arr = np.array(labels_predicted.cpu())
    label_arr = np.array(labels_full.cpu())
    aa_acc = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)
        diff = label_position - prediction_position
        wrong_num = np.sum(diff == 1)
        all_num = np.sum(label_position == 1)
        aa_acc = aa_acc + ((all_num - wrong_num) / all_num)
    aa_acc = aa_acc / args.class_num

    Pe = 0
    for i in range(args.class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)

        Pe = Pe + np.sum(label_position) * np.sum(prediction_position)

    Pe = Pe / (len(label_arr) * len(label_arr))
    ka_acc = (accuracy - Pe) / (1 - Pe)
    return [accuracy, aa_acc, ka_acc]



def common_load(args):
    test_loader = huston2013_multi_dataloader(train=False, args=args)

    modality_to_channel = {'hsi': 144, 'hsi1': 244, 'ms': 8, 'lidar': 1, 'sar': 4}
    modality_1_channel = modality_to_channel[args.pair_modalities[0]]
    modality_2_channel = modality_to_channel[args.pair_modalities[1]]

    saved_model = torch.load(args.model_root)
    model = MDMB(args, modality_1_channel, modality_2_channel)
    model.load_state_dict(saved_model)
    calc_accuracy_multi(model, test_loader, args)


if __name__ == '__main__':
    common_load(args=args)