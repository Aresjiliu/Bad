import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

import csv
import os
import time

import os
import torch.nn as nn
import torch.nn.functional as F
from lib.model_develop_utils import GradualWarmupScheduler
from scipy.linalg import orthogonal_procrustes
from models1 import *
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

sns.set_style('whitegrid')
ref_features_2d = None



def compute_krc(logits_A, logits_B):
    """
    计算每个样本的Kendall Rank Correlation (KRC)
    输入:
        logits_A: (batch_size, num_classes)
        logits_B: (batch_size, num_classes)
    输出:
        krc_per_sample: (batch_size,)
    """
    batch_size, num_classes = logits_A.shape

    # 生成所有类别对 (j, k) 的索引（j < k）
    j, k = torch.triu_indices(num_classes, num_classes, offset=1)

    # 计算排名矩阵（argsort两次得到排名）
    ranks_A = logits_A.argsort(dim=1).argsort(dim=1)  # 形状: (batch_size, num_classes)
    ranks_B = logits_B.argsort(dim=1).argsort(dim=1)  # 形状: (batch_size, num_classes)

    # 提取所有类别对的排名差异符号
    sign_A = torch.sign(ranks_A[:, j] - ranks_A[:, k])  # 形状: (batch_size, num_pairs)
    sign_B = torch.sign(ranks_B[:, j] - ranks_B[:, k])  # 形状: (batch_size, num_pairs)

    # 计算一致对数量（符号相同则为一致）
    concordant = (sign_A * sign_B > 0).sum(dim=1)  # 形状: (batch_size,)
    total_pairs = num_classes * (num_classes - 1) // 2

    # 计算KRC
    krc_per_sample = (concordant - (total_pairs - concordant)) / total_pairs
    return krc_per_sample


def ckd_loss(f_hs, f_radar, threshold):
    probs_A = F.softmax(f_hs, dim=1)
    probs_B = F.softmax(f_radar, dim=1)
    krc = compute_krc(f_hs, f_radar)
    mask = (krc > threshold).float()
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    loss_kd = kl_loss(torch.log(probs_B + 1e-8), probs_A) * mask.mean()
    return loss_kd, krc


def visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels, epoch, args, num_sample=10):
    """可视化特征中心及随机采样点"""
    # 计算每个类别的四种特征中心
    global ref_features_2d
    unique_labels = torch.unique(labels)
    centers_sh_hs = []
    centers_sh_radar = []
    centers_sp_hs = []
    centers_sp_radar = []

    # 随机采样
    sampled_features_sh_hs = []
    sampled_features_sh_radar = []
    sampled_features_sp_hs = []
    sampled_features_sp_radar = []
    sampled_labels = []

    for label in unique_labels:
        mask = (labels == label)
        # 计算每个类别的四种特征中心
        centers_sh_hs.append(f_sh_hs[mask].mean(dim=0))
        centers_sh_radar.append(f_sh_radar[mask].mean(dim=0))
        centers_sp_hs.append(f_sp_hs[mask].mean(dim=0))
        centers_sp_radar.append(f_sp_radar[mask].mean(dim=0))

        # 随机采样
        sample_indices = torch.randperm(mask.sum())[:num_sample]
        sampled_features_sh_hs.append(f_sh_hs[mask][sample_indices])
        sampled_features_sh_radar.append(f_sh_radar[mask][sample_indices])
        sampled_features_sp_hs.append(f_sp_hs[mask][sample_indices])
        sampled_features_sp_radar.append(f_sp_radar[mask][sample_indices])
        sampled_labels.extend([label.item()] * (4 * num_sample))

    # 堆叠所有中心点
    centers_sh_hs = torch.stack(centers_sh_hs)  # (class_num, feature_dim)
    centers_sh_radar = torch.stack(centers_sh_radar)
    centers_sp_hs = torch.stack(centers_sp_hs)
    centers_sp_radar = torch.stack(centers_sp_radar)

    # 堆叠所有采样点
    sampled_features_sh_hs = torch.cat(sampled_features_sh_hs, dim=0)
    sampled_features_sh_radar = torch.cat(sampled_features_sh_radar, dim=0)
    sampled_features_sp_hs = torch.cat(sampled_features_sp_hs, dim=0)
    sampled_features_sp_radar = torch.cat(sampled_features_sp_radar, dim=0)

    # 合并所有特征中心和采样点
    features = torch.cat([centers_sh_hs, centers_sh_radar, centers_sp_hs, centers_sp_radar,
                          sampled_features_sh_hs, sampled_features_sh_radar,
                          sampled_features_sp_hs, sampled_features_sp_radar], dim=0)
    labels = torch.cat([unique_labels.repeat(4).cpu(), torch.tensor(sampled_labels)])

    # 创建特征类型标记
    feature_types = []
    feature_types.extend(['sh_hs_center'] * len(unique_labels))
    feature_types.extend(['sh_radar_center'] * len(unique_labels))
    feature_types.extend(['sp_hs_center'] * len(unique_labels))
    feature_types.extend(['sp_radar_center'] * len(unique_labels))
    feature_types.extend(['sh_hs_sample'] * (len(unique_labels) * num_sample))
    feature_types.extend(['sh_radar_sample'] * (len(unique_labels) * num_sample))
    feature_types.extend(['sp_hs_sample'] * (len(unique_labels) * num_sample))
    feature_types.extend(['sp_radar_sample'] * (len(unique_labels) * num_sample))

    # TSNE降维 - 显式指定参数避免警告
    tsne = TSNE(
        n_components=2,
        random_state=42,
        init='pca',  # 显式指定初始化方式
        learning_rate='auto',  # 显式指定学习率
        perplexity=30
    )
    if epoch == (args.vis_interval -1):  # 初始epoch作为参考
        ref_features_2d = tsne.fit_transform(features.cpu().numpy())
        features_2d = ref_features_2d
    else:  # 后续epoch对齐到参考坐标系
        current_features_2d = tsne.fit_transform(features.cpu().numpy())
        # Procrustes变换（平移、旋转、缩放）
        R, _ = orthogonal_procrustes(current_features_2d, ref_features_2d)
        features_2d = np.dot(current_features_2d, R)  # 仅旋转，保持尺度一致

    # 绘制散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=labels,
        style=feature_types,
        palette=sns.color_palette("hls", args.class_num),
        s=100,
        alpha=0.7
    )

    plt.title(f'Feature Centers and Samples at Epoch {epoch}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # 保存图片
    os.makedirs(args.figure_root, exist_ok=True)
    save_path = os.path.join(args.figure_root, f'epoch_{epoch}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("saved success")



def calc_accuracy_double(model, loader, args,epoch,  verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full0 = []
    outputs_full1 = []
    labels_full = []
    all_f_sh_hs, all_f_sh_radar = [], []
    all_f_sp_hs, all_f_sp_radar = [], []
    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            output_batch0,output_batch1,f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = model(img_m1, img_m2)
            outputs_full0.append(output_batch0)
            outputs_full1.append(output_batch1)
            labels_full.append(target)
            # all_f_sh_hs.append(f_sh_hs)
            # all_f_sh_radar.append(f_sh_radar)
            # all_f_sp_hs.append(f_sp_hs)
            # all_f_sp_radar.append(f_sp_radar)

    model.train(mode_saved)
    outputs_full0 = torch.cat(outputs_full0, dim=0)
    outputs_full1 = torch.cat(outputs_full1, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    acc0 = cal_standard(outputs_full0, labels_full, args.class_num)[0]
    acc1 = cal_standard(outputs_full1, labels_full, args.class_num)[0]
    # if (epoch + 1) % args.vis_interval == 0:
    #     f_sh_hs = torch.mean(torch.cat(all_f_sh_hs, dim=0), dim=(2,3))
    #     f_sh_radar = torch.mean(torch.cat(all_f_sh_radar, dim=0), dim=(2,3))
    #     f_sp_hs = torch.mean(torch.cat(all_f_sp_hs, dim=0), dim=(2,3))
    #     f_sp_radar = torch.mean(torch.cat(all_f_sp_radar, dim=0), dim=(2,3))
    #     visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels_full, epoch, args)
    return acc0, acc1

def calc_accuracy_double(model, loader, args,epoch,  verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    labels_full = []
    all_f_sh_hs, all_f_sh_radar = [], []
    all_f_sp_hs, all_f_sp_radar = [], []
    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = model(img_m1, img_m2)

            labels_full.append(target)
            all_f_sh_hs.append(f_sh_hs)
            all_f_sh_radar.append(f_sh_radar)
            all_f_sp_hs.append(f_sp_hs)
            all_f_sp_radar.append(f_sp_radar)

    labels_full = torch.cat(labels_full, dim=0)
    return



def calc_accuracy_double_vis(model, loader, args, epoch, verbose=False):
    """
    功能：测试集前向 + 特征分解 + 保存 5 个 npy + 每类 1 张 png
    存放路径：args.figure_root / f"{args.identity}{epoch}"
    """
    device = next(model.parameters()).device
    out_dir = Path(args.figure_root) / f"{args.identity}{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 前向推理 ----------
    f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels_all = [], [], [], [], []
    for batch in tqdm(loader, desc="Test forward", disable=not verbose):
        m1, m2, y = batch['m_1'].to(device), batch['m_2'].to(device), batch['label'].to(device)
        f1, f2, f3, f4 = model(m1, m2)          # [B, C, 7, 7]
        f_sh_hs.append(f1.cpu())
        f_sh_radar.append(f2.cpu())
        f_sp_hs.append(f3.cpu())
        f_sp_radar.append(f4.cpu())
        labels_all.append(y.cpu())

    # 拼接 & 保存 5 个 npy
    feats_np = {
        'hsi_share.npy': torch.cat(f_sh_hs, 0).detach().cpu().numpy(),
        'lidar_share.npy': torch.cat(f_sh_radar, 0).detach().cpu().numpy(),
        'hsi_spec.npy': torch.cat(f_sp_hs, 0).detach().cpu().numpy(),
        'lidar_spec.npy': torch.cat(f_sp_radar, 0).detach().cpu().numpy(),
        'label.npy': torch.cat(labels_all, 0).cpu().numpy().astype(np.int64)
    }
    # for name, arr in feats_np.items():
    #     np.save(out_dir / name, arr)

    # ---------- 2. 每类可视化 ----------
    label = feats_np['label.npy']
    feats = {
        'HSI-share':   feats_np['hsi_share.npy'],
        'LiDAR-share': feats_np['lidar_share.npy'],
        'HSI-spec':    feats_np['hsi_spec.npy'],
        'LiDAR-spec':  feats_np['lidar_spec.npy']
    }
    names = list(feats.keys())
    colors = sns.color_palette('tab10', 4)

    for k in range(args.class_num):
        idx = np.where(label == k)[0]
        if len(idx) == 0:
            continue
        X_list = [feats[n][idx].reshape(len(idx), -1) for n in names]
        X_all  = np.vstack(X_list)
        emb  = TSNE(n_components=2, init='pca', learning_rate='auto',random_state=42, perplexity=30).fit_transform(X_all)
        splits = np.cumsum([len(x) for x in X_list])
        emb_list = np.split(emb, splits[:-1])

        plt.figure(figsize=(5, 5))
        for emb_i, color, mod_name in zip(emb_list, colors, names):
            plt.scatter(emb_i[:, 0], emb_i[:, 1], s=8, color=color, label=mod_name)
        plt.title(f'Class {k}')
        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(out_dir / f'class{k}.png', dpi=300)
        plt.close()

    print(f"已保存到 {out_dir.resolve()}")


def calc_accuracy_double_vis_berlin(model, loader, args, epoch, verbose=False):
    """
    功能：测试集前向 + 特征分解 + 保存 5 个 npy + 每类 1 张 png
    存放路径：args.figure_root / f"{args.identity}{epoch}"
    """
    device = next(model.parameters()).device
    out_dir = Path(args.figure_root) / f"{args.identity}{epoch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    MAX_SAMPLES = 500  # 新增：上限
    count = 0
    # ---------- 1. 前向推理 ----------
    f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels_all = [], [], [], [], []
    for batch in tqdm(loader, desc="Test forward", disable=not verbose):
        if count >= MAX_SAMPLES:          # ← 新增：提前终止
            break
        m1, m2, y = batch['m_1'].to(device), batch['m_2'].to(device), batch['label'].to(device)
        f1, f2, f3, f4 = model(m1, m2)          # [B, C, 7, 7]
        f_sh_hs.append(f1.cpu())
        f_sh_radar.append(f2.cpu())
        f_sp_hs.append(f3.cpu())
        f_sp_radar.append(f4.cpu())
        labels_all.append(y.cpu())
        count+=1

    # 拼接 & 保存 5 个 npy
    feats_np = {
        'hsi_share.npy': torch.cat(f_sh_hs, 0).detach().cpu().numpy(),
        'lidar_share.npy': torch.cat(f_sh_radar, 0).detach().cpu().numpy(),
        'hsi_spec.npy': torch.cat(f_sp_hs, 0).detach().cpu().numpy(),
        'lidar_spec.npy': torch.cat(f_sp_radar, 0).detach().cpu().numpy(),
        'label.npy': torch.cat(labels_all, 0).cpu().numpy().astype(np.int64)
    }
    for name, arr in feats_np.items():
        np.save(out_dir / name, arr)

    # ---------- 2. 每类可视化 ----------
    label = feats_np['label.npy']
    feats = {
        'HSI-share':   feats_np['hsi_share.npy'],
        'LiDAR-share': feats_np['lidar_share.npy'],
        'HSI-spec':    feats_np['hsi_spec.npy'],
        'LiDAR-spec':  feats_np['lidar_spec.npy']
    }
    names = list(feats.keys())
    colors = sns.color_palette('tab10', 4)

    for k in range(args.class_num):
        idx = np.where(label == k)[0]
        if len(idx) == 0:
            continue
        X_list = [feats[n][idx].reshape(len(idx), -1) for n in names]
        X_all  = np.vstack(X_list)
        emb  = TSNE(n_components=2, init='pca', learning_rate='auto',random_state=42, perplexity=30).fit_transform(X_all)
        splits = np.cumsum([len(x) for x in X_list])
        emb_list = np.split(emb, splits[:-1])

        plt.figure(figsize=(5, 5))
        for emb_i, color, mod_name in zip(emb_list, colors, names):
            plt.scatter(emb_i[:, 0], emb_i[:, 1], s=8, color=color, label=mod_name)
        plt.title(f'Class {k}')
        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(out_dir / f'class{k}.png', dpi=300)
        plt.close()
    print(f"已保存到 {out_dir.resolve()}")



def record_center(model, loader, args,epoch,  verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full0 = []
    outputs_full1 = []
    labels_full = []
    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            output_batch0,output_batch1,f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = model(img_m1, img_m2)
            outputs_full0.append(output_batch0)
            outputs_full1.append(output_batch1)
            labels_full.append(target)

    model.train(mode_saved)
    outputs_full0 = torch.cat(outputs_full0, dim=0)
    outputs_full1 = torch.cat(outputs_full1, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    acc0 = cal_standard(outputs_full0, labels_full, args.class_num)[0]
    acc1 = cal_standard(outputs_full1, labels_full, args.class_num)[0]
    return acc0, acc1




def calc_accuracy_multi(model, loader, args, epoch, verbose=False, hter=False):
    """
    :param model: model network
    :param loader: torch.utils.data.DataLoader
    :param verbose: show progress bar, bool
    :return accuracy, float
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    all_f_sh_hs, all_f_sh_radar = [], []
    all_f_sp_hs, all_f_sp_radar = [], []

    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):

        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            output_batch, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = model(img_m1, img_m2)
            outputs_full.append(output_batch)
            labels_full.append(target)
            all_f_sh_hs.append(f_sh_hs)
            all_f_sh_radar.append(f_sh_radar)
            all_f_sp_hs.append(f_sp_hs)
            all_f_sp_radar.append(f_sp_radar)

    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)
    if (epoch + 1) % args.vis_interval == 0:
        f_sh_hs = torch.cat(all_f_sh_hs, dim=0)
        f_sh_radar = torch.cat(all_f_sh_radar, dim=0)
        f_sp_hs = torch.cat(all_f_sp_hs, dim=0)
        f_sp_radar = torch.cat(all_f_sp_radar, dim=0)
        visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels_full, epoch, args)

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

    if hter:
        predict_arr = np.array(labels_predicted.cpu())
        label_arr = np.array(labels_full.cpu())

        living_wrong = 0  # living -- spoofing
        living_right = 0
        spoofing_wrong = 0  # spoofing ---living
        spoofing_right = 0

        for i in range(len(predict_arr)):
            if predict_arr[i] == label_arr[i]:
                if label_arr[i] == 1:
                    living_right += 1
                else:
                    spoofing_right += 1
            else:
                # 错误
                if label_arr[i] == 1:
                    living_wrong += 1
                else:
                    spoofing_wrong += 1
        try:
            FRR = living_wrong / (living_wrong + living_right)
            APCER = living_wrong / (spoofing_right + living_wrong)
            NPCER = spoofing_wrong / (spoofing_wrong + living_right)
            FAR = spoofing_wrong / (spoofing_wrong + spoofing_right)
            HTER = (FAR + FRR) / 2

            FAR = float("%.6f" % FAR)
            FRR = float("%.6f" % FRR)
            HTER = float("%.6f" % HTER)
            APCER = float("%.6f" % APCER)
            NPCER = float("%.6f" % NPCER)
            accuracy = float("%.6f" % accuracy)
        except Exception as e:
            print(e)
            return [accuracy, 1, 1, 1, 1, 1]

        return [accuracy, FAR, FRR, HTER, APCER, NPCER]
    else:
        return [accuracy, aa_acc, ka_acc]



def cal_standard(outputs_full, labels_full, class_num):
    _, labels_predicted = torch.max(outputs_full.data, dim=1)
    accuracy = torch.sum(labels_full == labels_predicted).item() / float(len(labels_full))
    accuracy = float("%.6f" % accuracy)

    predict_arr = np.array(labels_predicted.cpu())
    label_arr = np.array(labels_full.cpu())
    aa_acc = 0
    for i in range(class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)
        diff = label_position - prediction_position
        wrong_num = np.sum(diff == 1)
        all_num = np.sum(label_position == 1)
        aa_acc = aa_acc + ((all_num - wrong_num) / all_num)
    aa_acc = aa_acc / class_num

    Pe = 0
    for i in range(class_num):
        label_position = np.where(label_arr == i, 1, 0)
        prediction_position = np.where(predict_arr == i, 1, 0)

        Pe = Pe + np.sum(label_position) * np.sum(prediction_position)

    Pe = Pe / (len(label_arr) * len(label_arr))
    ka_acc = (accuracy - Pe) / (1 - Pe)
    return [accuracy, aa_acc, ka_acc]



def train_single_ckd(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    # Initialize and open timer
    start = time.time()
    md_loss = SpatialMDLoss(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    md_sum = 0
    cls_sum = 0
    ckd_sum = 0
    krc_cal = []
    md_adjust = 0.001

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)
            hs_output = outputs[0]
            radar_output = outputs[1]
            md = md_loss(*outputs[2:], target)*md_adjust
            md_sum += md.item()
            cls_loss = cost(hs_output, target) + cost(radar_output, target)
            cls_sum += cls_loss.item()
            # ckd, krc = ckd_loss(hs_output, radar_output, args.threshold)
            # ckd_sum += ckd
            # krc_cal.extend(krc.cpu().tolist())
            train_loss = md + cls_loss
            train_loss.backward()
            optimizer.step()

        acc0, acc1 = calc_accuracy_double(model, args=args, loader=test_loader, epoch=epoch, hter=False, verbose=True)
        accuracy_test = (acc0 +acc1)/2
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(cls_sum / len(train_loader))
        log_list.append(md_sum / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(acc0)
        log_list.append(acc1)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, acc1={:.5f},acc2={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            acc0,
                                                                                                            acc1,
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print("Epoch {},md_loss={:.5f},  cls_loss={:.5f}, ckd_loss={:.5f}, krc_avg={:.5f}".format(epoch, md_sum / len(train_loader),
                                                                 cls_sum / len(train_loader), ckd_sum / len(train_loader), 0))

        md_sum = 0
        cls_sum = 0
        ckd_sum = 0
        krc_cal = []

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])

        # save model and para
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)


def train_depose(model, cost, optimizer, train_loader, test_loader, args):
    '''
    适用于多模态分类的基础训练函数
    :param model:
    :param cost:
    :param optimizer:
    :param train_loader:
    :param test_loader:
    :param args:
    :return:
    '''
    print(args)

    start = time.time()
    md_loss = SpatialMDLoss(args)
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # save args
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    #  learning rate decay
    if args.lr_decrease == 'cos':
        print("lrcos is using")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)

        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("multi_step is using")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.train_epoch * 1 / 6),
                                                                              int(args.train_epoch * 2 / 6),
                                                                              int(args.train_epoch * 3 / 6)])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1,
                                                      after_scheduler=cos_scheduler)

    # Training initialization
    epoch_num = args.train_epoch
    log_interval = args.log_interval
    save_interval = args.save_interval
    batch_num = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    md_sum = 0
    cls_sum = 0
    ckd_sum = 0
    krc_cal = []
    md_adjust = 0.001

    if args.retrain:
        if not os.path.exists(models_dir):
            print("no trained model")
        else:
            state_read = torch.load(models_dir)
            model.load_state_dict(state_read['model_state'])
            optimizer.load_state_dict(state_read['optim_state'])
            epoch = state_read['Epoch']
            print("retaining")

    # Train
    while epoch < epoch_num:
        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epoch_num))):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], \
                batch_sample['label']
            if epoch == 0:
                continue
            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            outputs = model(img_hsi, img_lidar)
            md = md_loss(*outputs,target)*md_adjust
            md_sum += md.item()
            md.backward()
            optimizer.step()
        if (epoch+1)%50 ==0:
            calc_accuracy_double_vis(model, loader=test_loader, args=args, epoch=epoch, verbose=False)
        md_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()
        if epoch < 20:
            print(epoch, optimizer.param_groups[0]['lr'])


        #  save log
        with open(log_dir, 'a+', newline='') as f:
            # 训练结果
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []
        epoch = epoch + 1
    train_duration_sec = int(time.time() - start)
    print("training is end", train_duration_sec)



