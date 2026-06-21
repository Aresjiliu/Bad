import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import csv
import os
import time
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

from lib.model_develop_utils import GradualWarmupScheduler
from models1 import *
from models1_drfuse1 import Drfuse1
from models1_fusion_v2 import MdfuseV2
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')


def create_modality_mask(batch_size, missing_rate=0.0, device='cuda'):
    """
    创建模态缺失掩码

    Args:
        batch_size: 批次大小
        missing_rate: 模态缺失率 (0.0-1.0)
        device: 设备

    Returns:
        modality_mask: [B, 2] 模态存在性标记
    """
    modality_mask = torch.ones(batch_size, 2, device=device)

    if missing_rate > 0:
        # 随机选择缺失的样本
        num_missing = int(batch_size * missing_rate)
        missing_indices = random.sample(range(batch_size), num_missing)

        for idx in missing_indices:
            # 随机选择缺失的模态（HSI或LiDAR）
            if random.random() < 0.5:
                modality_mask[idx, 0] = 0  # HSI缺失
            else:
                modality_mask[idx, 1] = 0  # LiDAR缺失

    return modality_mask


def apply_modality_mask(data_hsi, data_lidar, modality_mask):
    """
    应用模态缺失掩码到数据

    Args:
        data_hsi: HSI数据
        data_lidar: LiDAR数据
        modality_mask: 模态存在性标记 [B, 2]

    Returns:
        masked_hsi: 掩码后的HSI数据
        masked_lidar: 掩码后的LiDAR数据
    """
    B = modality_mask.shape[0]
    device = modality_mask.device

    # 创建掩码后的数据
    if data_hsi is not None:
        masked_hsi = data_hsi.clone()
        for i in range(B):
            if modality_mask[i, 0] == 0:  # HSI缺失
                masked_hsi[i] = torch.zeros_like(masked_hsi[i])
    else:
        masked_hsi = None

    if data_lidar is not None:
        masked_lidar = data_lidar.clone()
        for i in range(B):
            if modality_mask[i, 1] == 0:  # LiDAR缺失
                masked_lidar[i] = torch.zeros_like(masked_lidar[i])
    else:
        masked_lidar = None

    return masked_hsi, masked_lidar


def calc_accuracy_missing_aware(model, loader, args, epoch, verbose=False, missing_rate=0.0):
    """
    支持模态缺失的精度计算函数

    Args:
        model: 模型
        loader: 数据加载器
        args: 参数
        epoch: 当前epoch
        verbose: 是否显示进度条
        missing_rate: 模态缺失率

    Returns:
        accuracy: 分类精度
        missing_accuracy: 模态缺失情况下的精度
    """
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()

    outputs_full = []
    labels_full = []
    modality_masks = []

    for batch_sample in tqdm(iter(loader), desc="Missing-aware evaluation", total=len(loader), disable=not verbose):
        img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['label']

        if torch.cuda.is_available():
            img_hsi = img_hsi.cuda()
            img_lidar = img_lidar.cuda()
            target = target.cuda()

        # 创建模态缺失掩码
        batch_size = img_hsi.shape[0]
        modality_mask = create_modality_mask(batch_size, missing_rate, img_hsi.device)

        # 应用模态缺失掩码
        masked_hsi, masked_lidar = apply_modality_mask(img_hsi, img_lidar, modality_mask)

        with torch.no_grad():
            # 支持模态缺失的前向传播
            if hasattr(model, 'forward'):
                # Drfuse1或MdfuseV2模型
                outputs = model(masked_hsi, masked_lidar, modality_mask)
                output_batch = outputs[0]
            else:
                # 原始模型（不支持模态缺失）
                output_batch, _, _, _, _ = model(masked_hsi, masked_lidar)

            outputs_full.append(output_batch)
            labels_full.append(target)
            modality_masks.append(modality_mask)

    model.train(mode_saved)

    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    modality_masks = torch.cat(modality_masks, dim=0)

    # 计算总体精度
    overall_accuracy = cal_standard(outputs_full, labels_full, args.class_num)[0]

    # 计算不同模态缺失情况下的精度
    hsi_missing_mask = (modality_masks[:, 0] == 0)
    lidar_missing_mask = (modality_masks[:, 1] == 0)
    both_exist_mask = (modality_masks[:, 0] == 1) & (modality_masks[:, 1] == 1)

    # HSI缺失精度
    if hsi_missing_mask.sum() > 0:
        hsi_missing_accuracy = cal_standard(
            outputs_full[hsi_missing_mask], labels_full[hsi_missing_mask], args.class_num
        )[0]
    else:
        hsi_missing_accuracy = 0.0

    # LiDAR缺失精度
    if lidar_missing_mask.sum() > 0:
        lidar_missing_accuracy = cal_standard(
            outputs_full[lidar_missing_mask], labels_full[lidar_missing_mask], args.class_num
        )[0]
    else:
        lidar_missing_accuracy = 0.0

    # 双模态存在精度
    if both_exist_mask.sum() > 0:
        both_exist_accuracy = cal_standard(
            outputs_full[both_exist_mask], labels_full[both_exist_mask], args.class_num
        )[0]
    else:
        both_exist_accuracy = overall_accuracy

    return overall_accuracy, {
        'hsi_missing': hsi_missing_accuracy,
        'lidar_missing': lidar_missing_accuracy,
        'both_exist': both_exist_accuracy,
        'missing_rate': missing_rate
    }


def train_single_missing_aware(model, cost, optimizer, train_loader, test_loader, args):
    """
    支持模态缺失的训练函数

    Args:
        model: 支持模态缺失的模型（Drfuse1或MdfuseV2）
        cost: 损失函数
        optimizer: 优化器
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        args: 训练参数
    """
    print("使用支持模态缺失的训练函数")
    print(args)

    # 初始化计时
    start = time.time()

    # 创建必要的目录
    if not os.path.exists(args.model_root):
        os.makedirs(args.model_root)
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    models_dir = args.model_root + '/' + args.name + '.pt'
    log_dir = args.log_root + '/' + args.name + '.csv'

    # 保存参数
    with open(log_dir, 'a+', newline='') as f:
        my_writer = csv.writer(f)
        args_dict = vars(args)
        for key, value in args_dict.items():
            my_writer.writerow([key, value])
        f.close()

    # 初始化损失函数
    criterion_cls = cost  # 分类损失

    # 模态缺失训练参数
    missing_rate_start = getattr(args, 'missing_rate_start', 0.0)
    missing_rate_end = getattr(args, 'missing_rate_end', 0.3)
    missing_rate_schedule = getattr(args, 'missing_rate_schedule', 'linear')

    print(f"模态缺失训练配置: 起始缺失率={missing_rate_start}, 结束缺失率={missing_rate_end}")

    # 学习率调度
    if args.lr_decrease == 'cos':
        print("使用余弦退火学习率调度")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1, after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("使用多步学习率调度")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
            int(args.train_epoch * 1 / 6),
            int(args.train_epoch * 2 / 6),
            int(args.train_epoch * 3 / 6)
        ])
        if args.lr_warmup:
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1, after_scheduler=cos_scheduler)

    # 训练初始化
    epoch_num = args.train_epoch
    save_interval = args.save_interval
    batch_num = 0
    train_loss = 0
    epoch = 0
    accuracy_best = 0
    log_list = []
    cls_sum = 0

    # 加载预训练模型（如果存在）
    if args.retrain and os.path.exists(models_dir):
        print("加载预训练模型")
        state_read = torch.load(models_dir)
        model.load_state_dict(state_read['model_state'])
        optimizer.load_state_dict(state_read['optim_state'])
        epoch = state_read['Epoch']

    # 训练循环
    while epoch < epoch_num:
        # 计算当前epoch的模态缺失率
        if missing_rate_schedule == 'linear':
            current_missing_rate = missing_rate_start + (missing_rate_end - missing_rate_start) * epoch / epoch_num
        elif missing_rate_schedule == 'exponential':
            current_missing_rate = missing_rate_start * (missing_rate_end / missing_rate_start) ** (epoch / epoch_num)
        else:
            current_missing_rate = missing_rate_start

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_num}, Missing Rate: {current_missing_rate:.3f}")):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['label']

            if epoch == 0:
                continue

            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            # 创建模态缺失掩码
            batch_size = img_hsi.shape[0]
            modality_mask = create_modality_mask(batch_size, current_missing_rate, img_hsi.device)

            # 应用模态缺失掩码
            masked_hsi, masked_lidar = apply_modality_mask(img_hsi, img_lidar, modality_mask)

            # 前向传播（支持模态缺失）
            if hasattr(model, 'forward'):
                # Drfuse1或MdfuseV2模型
                outputs = model(masked_hsi, masked_lidar, modality_mask)
                output = outputs[0]
            else:
                # 原始模型（不支持模态缺失）
                output, _, _, _, _ = model(masked_hsi, masked_lidar)

            # 计算损失
            total_loss = 0

            if criterion_cls is not None:
                cls_loss = criterion_cls(output, target)
                cls_scalar = cls_loss.mean() if cls_loss.numel() > 1 else cls_loss
                cls_sum += cls_scalar.item()
                total_loss += cls_scalar

            # 可以在这里添加其他损失函数（如模态一致性损失）

            total_loss.backward()
            optimizer.step()

        # 评估模型（使用不同的模态缺失率）
        test_missing_rates = [0.0, 0.1, 0.2, 0.3]
        eval_results = {}

        for test_missing_rate in test_missing_rates:
            accuracy, missing_info = calc_accuracy_missing_aware(
                model, test_loader, args, epoch, verbose=False, missing_rate=test_missing_rate
            )
            eval_results[f'missing_rate_{test_missing_rate}'] = {
                'accuracy': accuracy,
                'missing_info': missing_info
            }

        # 记录最佳精度（基于无缺失情况的性能）
        current_accuracy = eval_results['missing_rate_0.0']['accuracy']
        if current_accuracy > accuracy_best and epoch > 5:
            accuracy_best = current_accuracy
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，精度: {accuracy_best:.4f}")

        # 打印评估结果
        print(f"Epoch {epoch}, Current Accuracy: {current_accuracy:.4f}, Best Accuracy: {accuracy_best:.4f}")
        for missing_rate_key, result in eval_results.items():
            acc = result['accuracy']
            print(f"  {missing_rate_key}: {acc:.4f}")

        # 记录日志
        log_list.append(cls_sum / len(train_loader))
        log_list.append(current_accuracy)
        log_list.append(accuracy_best)

        # 添加不同缺失率下的性能
        for missing_rate_key, result in eval_results.items():
            log_list.append(result['accuracy'])

        cls_sum = 0

        # 学习率调度
        if args.lr_decrease:
            if args.lr_warmup:
                scheduler_warmup.step()
            else:
                cos_scheduler.step()

        if epoch < 20:
            print(f"Epoch {epoch}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存模型和参数
        if epoch % save_interval == 0:
            train_state = {
                "Epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": args,
                "current_missing_rate": current_missing_rate,
                "eval_results": eval_results
            }
            models_dir = args.model_root + '/' + args.name + '.pt'
            if torch.__version__ > '1.6.0':
                torch.save(train_state, models_dir, _use_new_zipfile_serialization=False)
            else:
                torch.save(train_state, models_dir)

        # 保存日志
        with open(log_dir, 'a+', newline='') as f:
            my_writer = csv.writer(f)
            my_writer.writerow(log_list)
            log_list = []

        epoch += 1

    train_duration_sec = int(time.time() - start)
    print(f"训练完成，总耗时: {train_duration_sec}秒")
    print(f"最佳精度: {accuracy_best:.4f}")


def evaluate_missing_awareness(model, test_loader, args, missing_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    全面评估模型的模态缺失鲁棒性

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        args: 参数
        missing_rates: 要测试的缺失率列表

    Returns:
        evaluation_report: 评估报告
    """
    print("进行模态缺失鲁棒性评估...")

    model.eval()
    evaluation_results = {}

    for missing_rate in missing_rates:
        print(f"测试缺失率: {missing_rate}")
        accuracy, missing_info = calc_accuracy_missing_aware(
            model, test_loader, args, epoch=0, verbose=False, missing_rate=missing_rate
        )

        evaluation_results[f'missing_rate_{missing_rate}'] = {
            'overall_accuracy': accuracy,
            'hsi_missing_accuracy': missing_info['hsi_missing'],
            'lidar_missing_accuracy': missing_info['lidar_missing'],
            'both_exist_accuracy': missing_info['both_exist']
        }

    # 计算鲁棒性指标
    baseline_accuracy = evaluation_results['missing_rate_0.0']['overall_accuracy']
    robustness_scores = {}

    for missing_rate in missing_rates[1:]:
        key = f'missing_rate_{missing_rate}'
        current_accuracy = evaluation_results[key]['overall_accuracy']
        robustness_score = 1.0 - (baseline_accuracy - current_accuracy) / baseline_accuracy
        robustness_scores[f'robustness_{missing_rate}'] = robustness_score

    # 生成评估报告
    evaluation_report = {
        'baseline_accuracy': baseline_accuracy,
        'detailed_results': evaluation_results,
        'robustness_scores': robustness_scores,
        'average_robustness': sum(robustness_scores.values()) / len(robustness_scores)
    }

    print("评估报告:")
    print(f"基线精度: {baseline_accuracy:.4f}")
    print(f"平均鲁棒性得分: {evaluation_report['average_robustness']:.4f}")

    for missing_rate, robustness in robustness_scores.items():
        print(f"{missing_rate}: {robustness:.4f}")

    return evaluation_report


def visualize_missing_impact(model, test_loader, args, num_samples=100):
    """
    可视化模态缺失对分类结果的影响

    Args:
        model: 模型
        test_loader: 测试数据加载器
        args: 参数
        num_samples: 要可视化的样本数量
    """
    print("可视化模态缺失影响...")

    model.eval()
    device = next(model.parameters()).device

    # 收集样本
    samples_collected = 0
    results = []

    with torch.no_grad():
        for batch_sample in test_loader:
            if samples_collected >= num_samples:
                break

            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['label']
            batch_size = img_hsi.shape[0]

            if samples_collected + batch_size > num_samples:
                # 只取需要的样本数量
                needed_samples = num_samples - samples_collected
                img_hsi = img_hsi[:needed_samples]
                img_lidar = img_lidar[:needed_samples]
                target = target[:needed_samples]

            if torch.cuda.is_available():
                img_hsi = img_hsi.to(device)
                img_lidar = img_lidar.to(device)
                target = target.to(device)

            # 测试不同模态组合
            # 1. 完整模态
            if hasattr(model, 'forward'):
                outputs_complete = model(img_hsi, img_lidar)
                probs_complete = F.softmax(outputs_complete[0], dim=1)
            else:
                outputs_complete = model(img_hsi, img_lidar)
                probs_complete = F.softmax(outputs_complete[0], dim=1)

            # 2. HSI缺失
            modality_mask_hsi_missing = torch.ones(batch_size, 2, device=device)
            modality_mask_hsi_missing[:, 0] = 0
            if hasattr(model, 'forward'):
                outputs_hsi_missing = model(None, img_lidar, modality_mask_hsi_missing)
                probs_hsi_missing = F.softmax(outputs_hsi_missing[0], dim=1)
            else:
                dummy_hsi = torch.zeros_like(img_hsi)
                outputs_hsi_missing = model(dummy_hsi, img_lidar)
                probs_hsi_missing = F.softmax(outputs_hsi_missing[0], dim=1)

            # 3. LiDAR缺失
            modality_mask_lidar_missing = torch.ones(batch_size, 2, device=device)
            modality_mask_lidar_missing[:, 1] = 0
            if hasattr(model, 'forward'):
                outputs_lidar_missing = model(img_hsi, None, modality_mask_lidar_missing)
                probs_lidar_missing = F.softmax(outputs_lidar_missing[0], dim=1)
            else:
                dummy_lidar = torch.zeros_like(img_lidar)
                outputs_lidar_missing = model(img_hsi, dummy_lidar)
                probs_lidar_missing = F.softmax(outputs_lidar_missing[0], dim=1)

            # 记录结果
            for i in range(batch_size):
                results.append({
                    'target': target[i].item(),
                    'prob_complete': probs_complete[i].max().item(),
                    'prob_hsi_missing': probs_hsi_missing[i].max().item(),
                    'prob_lidar_missing': probs_lidar_missing[i].max().item(),
                    'pred_complete': probs_complete[i].argmax().item(),
                    'pred_hsi_missing': probs_hsi_missing[i].argmax().item(),
                    'pred_lidar_missing': probs_lidar_missing[i].argmax().item()
                })

            samples_collected += batch_size

    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 置信度分布对比
    complete_probs = [r['prob_complete'] for r in results]
    hsi_missing_probs = [r['prob_hsi_missing'] for r in results]
    lidar_missing_probs = [r['prob_lidar_missing'] for r in results]

    axes[0, 0].hist([complete_probs, hsi_missing_probs, lidar_missing_probs],
                    bins=20, alpha=0.7, label=['Complete', 'HSI Missing', 'LiDAR Missing'])
    axes[0, 0].set_xlabel('Max Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Confidence Distribution Comparison')
    axes[0, 0].legend()

    # 2. 精度对比
    complete_correct = sum(1 for r in results if r['pred_complete'] == r['target'])
    hsi_missing_correct = sum(1 for r in results if r['pred_hsi_missing'] == r['target'])
    lidar_missing_correct = sum(1 for r in results if r['pred_lidar_missing'] == r['target'])

    total_samples = len(results)
    accuracies = [
        complete_correct / total_samples,
        hsi_missing_correct / total_samples,
        lidar_missing_correct / total_samples
    ]

    axes[0, 1].bar(['Complete', 'HSI Missing', 'LiDAR Missing'], accuracies)
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Comparison')
    axes[0, 1].set_ylim(0, 1)

    # 3. 置信度下降分析
    confidence_drop_hsi = [complete_probs[i] - hsi_missing_probs[i] for i in range(len(results))]
    confidence_drop_lidar = [complete_probs[i] - lidar_missing_probs[i] for i in range(len(results))]

    axes[1, 0].boxplot([confidence_drop_hsi, confidence_drop_lidar], labels=['HSI Missing', 'LiDAR Missing'])
    axes[1, 0].set_ylabel('Confidence Drop')
    axes[1, 0].set_title('Confidence Drop Analysis')

    # 4. 错误类型分析
    error_types = []
    for r in results:
        if r['pred_complete'] == r['target']:
            if r['pred_hsi_missing'] != r['target'] and r['pred_lidar_missing'] != r['target']:
                error_types.append('Both Missing Wrong')
            elif r['pred_hsi_missing'] != r['target']:
                error_types.append('HSI Missing Wrong')
            elif r['pred_lidar_missing'] != r['target']:
                error_types.append('LiDAR Missing Wrong')
            else:
                error_types.append('All Correct')
        else:
            error_types.append('Complete Wrong')

    error_counts = {}
    for error_type in error_types:
        error_counts[error_type] = error_counts.get(error_type, 0) + 1

    axes[1, 1].pie(error_counts.values(), labels=error_counts.keys(), autopct='%1.1f%%')
    axes[1, 1].set_title('Error Type Distribution')

    plt.tight_layout()

    # 保存可视化结果
    vis_path = os.path.join(args.figure_root, f"missing_impact_{args.name}.png")
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"模态缺失影响可视化已保存至: {vis_path}")


# 辅助函数（与原始代码保持一致）
def cal_standard(outputs_full, labels_full, class_num):
    """计算标准分类指标"""
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


def test_functions():
    """测试新功能"""
    print("测试模态缺失相关函数...")

    # 测试模态掩码创建
    B = 8
    device = 'cpu'
    missing_rate = 0.3

    modality_mask = create_modality_mask(B, missing_rate, device)
    print(f"模态掩码形状: {modality_mask.shape}")
    print(f"缺失样本数量: {(modality_mask.sum(dim=1) < 2).sum()}")

    # 测试数据掩码应用
    img_hsi = torch.randn(B, 144, 7, 7)
    img_lidar = torch.randn(B, 1, 7, 7)

    masked_hsi, masked_lidar = apply_modality_mask(img_hsi, img_lidar, modality_mask)
    print(f"掩码后HSI形状: {masked_hsi.shape}")
    print(f"掩码后LiDAR形状: {masked_lidar.shape}")

    print("测试完成!")


if __name__ == "__main__":
    test_functions()