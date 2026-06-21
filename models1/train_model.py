import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
from tqdm import tqdm
import time
import csv
import os
import time

import os
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F

from lib.model_develop_utils import GradualWarmupScheduler
from loss.mmd_loss import MMD_loss
from models1.dw_prune import Conv2d_Prune


def calc_accuracy_multi(model, loader, args, verbose=False, hter=False):
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


def train_mcl(model, cost, optimizer, train_loader, test_loader, args):
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

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

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
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0


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

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs


            cls_loss = cost(output, target)
            cls_sum += cls_loss.item()
            cls_loss.backward()
            optimizer.step()

        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(cls_sum / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(result_test[1])
        log_list.append(result_test[2])
        log_list.append(accuracy_best)
        print(
            "Epoch {},accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,accuracy_test,accuracy_best))

        print(train_loss / len(train_loader), unimodal_loss_sum / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        unimodal_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
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


def train_mcl_prune(model, cost, optimizer, train_loader, test_loader, args):
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

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

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
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
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
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    l1_loss_sum = 0


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

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            cls_loss = cost(output, target)
            l1_loss = model.l1_loss()
            cls_sum += cls_loss.item()
            l1_loss_sum += l1_loss.item()
            total_loss = cls_loss+l1_loss*args.l1_loss
            total_loss.backward()
            optimizer.step()
        for module in model.modules():
            if isinstance(module, Conv2d_Prune):
                module.saved_weight = nn.Parameter(module.masked_weight, requires_grad=False)
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        model.update_temp()
        log_list.append(cls_sum / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(result_test[1])
        log_list.append(result_test[2])
        log_list.append(accuracy_best)
        print(
            "Epoch {},accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,accuracy_test,accuracy_best))

        print(train_loss / len(train_loader), unimodal_loss_sum / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        unimodal_loss_sum = 0
        cls_sum = 0
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
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



def train_dgd(model, cost, optimizer, train_loader, test_loader, args):
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

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

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
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
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
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0


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

            if isinstance(outputs, tuple):
                output = outputs[0]
            else:
                output = outputs

            hsi_out = outputs[-2]
            lidar_out = outputs[-1]
            cls_loss1 = cost(output, target)
            cls_loss2 = cost(hsi_out, target)
            cls_loss3 = cost(lidar_out, target)
            cls_loss = cls_loss1
            unimodal_loss = mse_func(hsi_out, lidar_out) + cls_loss2 + cls_loss3

            total_loss = cls_loss + unimodal_loss * args.labma_unimodal

            train_loss += total_loss.item()
            unimodal_loss_sum += unimodal_loss.item()
            cls_sum += cls_loss1.item()
            total_loss.backward()
            optimizer.step()

        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(result_test[1])
        log_list.append(result_test[2])
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), unimodal_loss_sum / len(train_loader), cls_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        unimodal_loss_sum = 0
        cls_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
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


def train_base_multi_shaspec(model, cost, optimizer, train_loader, test_loader, args):
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

    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'
    mse_func = nn.MSELoss()

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
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[np.int(args.train_epoch * 1 / 6),
                                                                              np.int(args.train_epoch * 2 / 6),
                                                                              np.int(args.train_epoch * 3 / 6)])
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
    rec_loss_sum = 0
    unimodal_loss_sum = 0
    cls_sum = 0
    dco_loss_sum = 0
    dao_loss_sum = 0
    is_rec = False

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

            torch.autograd.set_detect_anomaly(True)

            target_predict, dco_predict, specific_feature_label, m1_feature_share_cache, m2_feature_share_cache, m1_predict, m2_predict = model(
                img_hsi, img_lidar)

            task_loss = cost(target_predict, target)
            cls_loss2 = cost(m1_predict, target)
            cls_loss3 = cost(m2_predict, target)
            unimodal_loss = mse_func(m1_predict, m2_predict) + cls_loss2 + cls_loss3

            dao_loss = mse_func(m1_feature_share_cache, m2_feature_share_cache)

            dco_loss = cost(dco_predict, specific_feature_label)

            cls_loss = task_loss + 1.0 * dao_loss + 0.02 * dco_loss + unimodal_loss * 0

            total_loss = cls_loss

            train_loss += total_loss.item()
            dao_loss_sum += dao_loss.item()
            dco_loss_sum += dco_loss.item()
            cls_sum += cls_loss2.item() + cls_loss3.item()
            total_loss.backward()
            optimizer.step()

        # testing
        result_test = calc_accuracy_multi(model, args=args, loader=test_loader, hter=False, verbose=True)
        accuracy_test = result_test[0]
        if accuracy_test > accuracy_best and epoch > 5:
            accuracy_best = accuracy_test
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
        log_list.append(train_loss / len(train_loader))
        log_list.append(accuracy_test)
        log_list.append(accuracy_best)
        print(
            "Epoch {}, cls_loss={:.5f},rec_loss={:.5f}, accuracy_test={:.5f},  accuracy_best={:.5f}".format(epoch,
                                                                                                            train_loss / len(
                                                                                                                train_loader),
                                                                                                            rec_loss_sum / len(
                                                                                                                train_loader),
                                                                                                            accuracy_test,
                                                                                                            accuracy_best))

        print(train_loss / len(train_loader), dco_loss_sum / len(train_loader), dao_loss_sum / len(train_loader))
        train_loss = 0
        rec_loss_sum = 0
        cls_sum = 0
        dao_loss_sum = 0
        dco_loss_sum = 0

        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step(epoch=epoch)
            else:
                cos_scheduler.step(epoch=epoch)
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