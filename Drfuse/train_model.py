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
        output, f1, f2, f3, f4 = model(m1, m2)  # [B, C, 7, 7]
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
        'HSI-share': feats_np['hsi_share.npy'],
        'LiDAR-share': feats_np['lidar_share.npy'],
        'HSI-spec': feats_np['hsi_spec.npy'],
        'LiDAR-spec': feats_np['lidar_spec.npy']
    }
    names = list(feats.keys())
    colors = sns.color_palette('tab10', 4)

    for k in range(args.class_num):
        idx = np.where(label == k)[0]
        if len(idx) == 0:
            continue
        X_list = [feats[n][idx].reshape(len(idx), -1) for n in names]
        X_all = np.vstack(X_list)
        emb = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30).fit_transform(
            X_all)
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


def calc_accuracy_single(model, loader, args, epoch, verbose=False, hter=False):
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
    labels_full = []
    # all_f_sh_hs, all_f_sh_radar = [], []
    # all_f_sp_hs, all_f_sp_radar = [], []
    for batch_sample in tqdm(iter(loader), desc="Full forward pass", total=len(loader), disable=not verbose):
        img_m1, img_m2, target = batch_sample['m_1'], batch_sample['m_2'], \
            batch_sample['label']
        if torch.cuda.is_available():
            img_m1 = img_m1.cuda()
            img_m2 = img_m2.cuda()
            target = target.cuda()

        with torch.no_grad():
            output_batch0, f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar = model(img_m1, img_m2)
            outputs_full0.append(output_batch0)
            # outputs_full1.append(output_batch1)
            labels_full.append(target)
            # all_f_sh_hs.append(f_sh_hs)
            # all_f_sh_radar.append(f_sh_radar)
            # all_f_sp_hs.append(f_sp_hs)
            # all_f_sp_radar.append(f_sp_radar)

    model.train(mode_saved)
    outputs_full0 = torch.cat(outputs_full0, dim=0)
    # outputs_full1 = torch.cat(outputs_full1, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    acc0 = cal_standard(outputs_full0, labels_full, args.class_num)[0]
    # acc1 = cal_standard(outputs_full1, labels_full, args.class_num)[0]
    # if (epoch + 1) % args.vis_interval == 0:
    #     f_sh_hs = torch.cat(all_f_sh_hs, dim=0)
    #     f_sh_radar = torch.cat(all_f_sh_radar, dim=0)
    #     f_sp_hs = torch.cat(all_f_sp_hs, dim=0)
    #     f_sp_radar = torch.cat(all_f_sp_radar, dim=0)
    #     visualize_features(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar, labels_full, epoch, args)
    return acc0


def spatial_jsd_pooling(shared_A, shared_B):
    """空间JSD对齐与损失计算"""
    # 转换为概率分布 (空间维度保持)
    P = torch.sigmoid(shared_A)
    Q = torch.sigmoid(shared_B)
    M = (P + Q) / 2

    # 计算KL散度 (添加epsilon防止log(0))
    epsilon = 1e-8
    kl_pm = F.kl_div(
        torch.log(P + epsilon),
        M + epsilon,
        reduction='none'
    )
    kl_qm = F.kl_div(
        torch.log(Q + epsilon),
        M + epsilon,
        reduction='none'
    )

    # 空间平均JSD损失
    jsd_loss = 0.5 * (kl_pm + kl_qm)
    jsd_loss = jsd_loss.mean(dim=[1, 2, 3]).mean()  # 批次平均

    # Logit Pooling融合
    pooled = torch.logit((P + Q) / 2 + epsilon)
    return pooled, jsd_loss


def spatial_ortho_loss(shared, specific):
    """空间正交约束损失"""
    # 展平空间维度
    shared_flat = shared.flatten(2)  # [B, C, H*W]
    spec_flat = specific.flatten(2)  # [B, D, H*W]

    # 归一化
    norm_shared = F.normalize(shared_flat, dim=1)
    norm_spec = F.normalize(spec_flat, dim=1)

    similarity = torch.einsum('bch,bdh->bcd', norm_shared, norm_spec)

    # 空间位置平均
    return torch.mean(torch.abs(similarity))


def defuse_loss(f_sh_hs, f_sh_radar, f_sp_hs, f_sp_radar):
    aligned_shared, jsd_loss = spatial_jsd_pooling(f_sh_hs, f_sh_radar)
    ortho_loss_A = spatial_ortho_loss(f_sh_hs, f_sp_hs)
    ortho_loss_B = spatial_ortho_loss(f_sh_radar, f_sp_radar)
    return jsd_loss + ortho_loss_A + ortho_loss_B


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
    criterion_intra = IntraClassCompactnessLoss(margin=0.5, weight=0.1)
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
    intra_sum = 0

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
            md = defuse_loss(*outputs)
            md_sum += md.item()
            intra = criterion_intra(outputs, target)
            intra_sum += intra.item()
            (md+intra).backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            calc_accuracy_double_vis(model, loader=test_loader, args=args, epoch=epoch, verbose=False)

        log_list.append(md_sum / len(train_loader))
        md_sum = 0

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
    criterion_intra = IntraClassCompactnessLoss(margin=0.5, weight=0.1)
    start = time.time()
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
    intra_sum = 0
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
            output = outputs[0]
            md = defuse_loss(*outputs[1:])
            md_sum += md.item()
            cls_loss = cost(output, target)
            cls_sum += cls_loss.item()
            intra = criterion_intra(outputs[1:], target)
            intra_sum += intra.item()
            train_loss = md + cls_loss + intra
            train_loss.backward()
            optimizer.step()

        accuracy_test = calc_accuracy_single(model, args=args, loader=test_loader, epoch=epoch, hter=False,
                                             verbose=True)
        if (epoch + 1) % 150 == 0:
            calc_accuracy_double_vis(model, loader=test_loader, args=args, epoch=epoch, verbose=False)
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(cls_sum / len(train_loader))
        log_list.append(md_sum / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                           accuracy_test,
                                                                           accuracy_best))

        md_sum = 0
        cls_sum = 0
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
