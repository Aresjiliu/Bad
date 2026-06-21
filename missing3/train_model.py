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
from models1 import SpatialMDLoss, IdentityClassificationLoss, IntraClassCompactnessLoss
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
    batch_num = 0
    epoch = 0
    log_list = []  # log need to save
    defuse_sum = 0
    intra_sum = 0
    md_sum = 0
    id_sum = 0
    
    # 初始化损失函数（根据配置）
    criterion_drfuse = DrfuseLoss() if args.use_drfuse_loss else None  # DrFuse损失函数
    criterion_id = IdentityClassificationLoss(feature_dim=128, num_classes=args.class_num) if args.use_id_loss else None  # ID损失函数
    criterion_intra = IntraClassCompactnessLoss(margin=0.5, weight=0.1) if args.use_intra_loss else None  # 类内紧凑性损失
    criterion_md = SpatialMDLoss(args) if args.use_md_loss else None  # MD损失函数

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

            # 前向传播
            outputs = model(img_hsi, img_lidar)
            
            # 计算损失
            total_loss = 0
            
            if criterion_drfuse is not None:
                defuse = criterion_drfuse(*outputs[1:])  # 使用DrFuse损失函数
                defuse_scalar = defuse.mean() if defuse.numel() > 1 else defuse
                defuse_sum += defuse_scalar.item()
                total_loss += defuse_scalar
            else:
                defuse_sum += 0
            
            if criterion_id is not None:
                id_loss = criterion_id(*outputs[1:], target)  # ID损失函数
                id_scalar = id_loss.mean() if id_loss.numel() > 1 else id_loss
                id_sum += id_scalar.item()
                total_loss += id_scalar
            else:
                id_sum += 0
            
            if criterion_intra is not None:
                intra = criterion_intra(outputs[1:], target)  # 类内紧凑性损失
                intra_scalar = intra.mean() if intra.numel() > 1 else intra
                intra_sum += intra_scalar.item()
                total_loss += intra_scalar
            else:
                intra_sum += 0
            
            if criterion_md is not None:
                if hasattr(criterion_md, 'forward') and 'return_aux_losses' in criterion_md.forward.__code__.co_varnames:
                    # 增强版SpatialMDLoss支持返回辅助损失
                    md_loss, md_aux = criterion_md(*outputs[1:], target, return_aux_losses=True)
                    md_scalar = md_loss.mean() if md_loss.numel() > 1 else md_loss
                    md_sum += md_scalar.item()
                    total_loss += md_scalar
                    
                    # 记录MD损失的详细组件（可选，用于调试）
                    # if epoch % 50 == 0:  # 每50个epoch记录一次
                    #     print(f"MD Loss Components - L_shs: {md_aux['L_shs']:.4f}, L_dc: {md_aux['L_dc']:.4f}, "
                    #           f"L_sps: {md_aux['L_sps']:.4f}, contrastive: {md_aux['contrastive']:.4f}, "
                    #           f"hard_mining: {md_aux['hard_mining']:.4f}")
                else:
                    # 传统MD损失
                    md_loss = criterion_md(*outputs[1:], target)  # MD损失函数
                    md_scalar = md_loss.mean() if md_loss.numel() > 1 else md_loss
                    md_sum += md_scalar.item()
                    total_loss += md_scalar
            else:
                md_sum += 0
            total_loss.backward()
            optimizer.step()
        if (epoch + 1) % 50 == 0:
            calc_accuracy_double_vis(model, loader=test_loader, args=args, epoch=epoch, verbose=False)

        log_list.append(defuse_sum / len(train_loader))
        log_list.append(id_sum / len(train_loader))
        log_list.append(intra_sum / len(train_loader))
        log_list.append(md_sum / len(train_loader))
        defuse_sum = 0
        id_sum = 0
        intra_sum = 0
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
    训练函数 - 添加调试输出
    '''
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
    
    # 初始化损失函数（根据配置）
    criterion_drfuse = DrfuseLoss() if args.use_drfuse_loss else None  # DrFuse损失函数
    criterion_id = IdentityClassificationLoss(feature_dim=128, num_classes=args.class_num) if args.use_id_loss else None  # ID损失函数
    criterion_intra = IntraClassCompactnessLoss(margin=0.5, weight=0.1) if args.use_intra_loss else None  # 类内紧凑性损失
    criterion_md = SpatialMDLoss(args) if args.use_md_loss else None  # MD损失函数
    criterion_cls = cost  # 分类损失函数
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
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []  # log need to save
    defuse_sum = 0
    cls_sum = 0
    id_sum = 0
    intra_sum = 0
    md_sum = 0
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
            
            # 前向传播
            outputs = model(img_hsi, img_lidar)
            output = outputs[0]
            
            # 计算各种损失
            total_loss = 0
            
            if criterion_drfuse is not None:
                defuse = criterion_drfuse(*outputs[1:])  # DrFuse损失
                defuse_scalar = defuse.mean() if defuse.numel() > 1 else defuse
                defuse_sum += defuse_scalar.item()
                total_loss += defuse_scalar

            
            if criterion_cls is not None:
                cls_loss = criterion_cls(output, target)  # 分类损失
                cls_scalar = cls_loss.mean() if cls_loss.numel() > 1 else cls_loss
                cls_sum += cls_scalar.item()
                total_loss += cls_scalar

            
            if criterion_id is not None:
                id_loss = criterion_id(*outputs[1:], target)  # ID损失函数
                id_scalar = id_loss.mean() if id_loss.numel() > 1 else id_loss
                id_sum += id_scalar.item()
                total_loss += id_scalar

            
            if criterion_intra is not None:
                intra = criterion_intra(outputs[1:], target)  # 类内紧凑性损失
                intra_scalar = intra.mean() if intra.numel() > 1 else intra
                intra_sum += intra_scalar.item()
                total_loss += intra_scalar

            
            if criterion_md is not None:
                md_loss = criterion_md(*outputs[1:], target)  # MD损失函数
                md_scalar = md_loss.mean() if md_loss.numel() > 1 else md_loss
                md_sum += md_scalar.item()
                total_loss += md_scalar
            
            train_loss = total_loss
            train_loss.backward()
            optimizer.step()

        accuracy_test = calc_accuracy_single(model, args=args, loader=test_loader, epoch=epoch, hter=False,
                                             verbose=True)

        # if (epoch + 1) % 150 == 0:
        #     calc_accuracy_double_vis(model, loader=test_loader, args=args, epoch=epoch, verbose=False)
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(cls_sum / len(train_loader))
        log_list.append(defuse_sum / len(train_loader))
        log_list.append(id_sum / len(train_loader))
        log_list.append(intra_sum / len(train_loader))
        log_list.append(md_sum / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                           accuracy_test,
                                                                           accuracy_best))

        defuse_sum = 0
        cls_sum = 0
        id_sum = 0
        intra_sum = 0
        md_sum = 0
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
