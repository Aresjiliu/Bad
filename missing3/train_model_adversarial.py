import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import os
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

from models1_adversarial_generation import DrfuseAdversarial
from models1_adversarial_losses import AdversarialFeatureLoss, FeatureConsistencyLoss, ModalityAlignmentLoss
from train_model_missing import calc_accuracy_missing_aware, create_modality_mask, apply_modality_mask
from train_model import cal_standard


class AdversarialTrainer:
    """
    对抗训练管理器
    管理生成器和判别器的交替训练
    """

    def __init__(self, model, generator_optimizer, discriminator_optimizer, adversarial_loss_fn,
                 generator_update_freq=1, discriminator_update_freq=1, adversarial_start_epoch=10):
        self.model = model
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.adversarial_loss_fn = adversarial_loss_fn
        self.generator_update_freq = generator_update_freq
        self.discriminator_update_freq = discriminator_update_freq
        self.adversarial_start_epoch = adversarial_start_epoch
        self.current_epoch = 0

        # 训练统计
        self.gen_loss_history = []
        self.dis_loss_history = []
        self.adversarial_training_active = False

    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.current_epoch = epoch
        self.adversarial_training_active = epoch >= self.adversarial_start_epoch

    def train_generator(self, shared_features_dict, real_specific_features_dict, modality_mask):
        """
        训练生成器

        Args:
            shared_features_dict: 共享特征字典
            real_specific_features_dict: 真实特定特征字典
            modality_mask: 模态存在性标记

        Returns:
            gen_loss: 生成器损失
            gen_loss_dict: 生成器损失分量
        """
        # 前向传播生成特征
        adversarial_results = self.model.adversarial_generation(
            shared_features_dict, real_specific_features_dict, modality_mask, training_mode='generator'
        )

        generated_features = adversarial_results['generated_features']

        # 判别生成的特征
        combined_features = {}
        for modality in ['hsi', 'lidar']:
            if modality in real_specific_features_dict and real_specific_features_dict[modality] is not None:
                combined_features[modality] = generated_features.get(modality, real_specific_features_dict[modality])
            elif modality in generated_features:
                combined_features[modality] = generated_features[modality]

        validity_scores = self.model.adversarial_generation.discriminate_features(
            combined_features, modality_mask
        )

        # 计算生成器损失
        gen_loss, gen_loss_dict = self.adversarial_loss_fn.generator_loss(
            generated_features, real_specific_features_dict, shared_features_dict,
            validity_scores, modality_mask
        )

        # 反向传播
        self.generator_optimizer.zero_grad()
        gen_loss.backward()
        self.generator_optimizer.step()

        return gen_loss.item(), gen_loss_dict

    def train_discriminator(self, shared_features_dict, real_specific_features_dict, modality_mask):
        """
        训练判别器

        Args:
            shared_features_dict: 共享特征字典
            real_specific_features_dict: 真实特定特征字典
            modality_mask: 模态存在性标记

        Returns:
            dis_loss: 判别器损失
            dis_loss_dict: 判别器损失分量
        """
        # 生成假特征
        with torch.no_grad():
            adversarial_results = self.model.adversarial_generation(
                shared_features_dict, real_specific_features_dict, modality_mask, training_mode='generator'
            )

        generated_features = adversarial_results['generated_features']

        # 组合真实特征和生成特征
        combined_features = {}
        real_validity_dict = {}
        fake_validity_dict = {}

        for modality in ['hsi', 'lidar']:
            if modality in real_specific_features_dict and real_specific_features_dict[modality] is not None:
                # 真实特征
                combined_features[modality] = real_specific_features_dict[modality]
                real_validity = self.model.adversarial_generation.discriminate_features(
                    {modality: real_specific_features_dict[modality]}, modality_mask
                )
                for key, value in real_validity.items():
                    real_validity_dict[f"{modality}_{key}"] = value

            if modality in generated_features:
                # 生成特征
                combined_features[modality] = generated_features[modality]
                fake_validity = self.model.adversarial_generation.discriminate_features(
                    {modality: generated_features[modality]}, modality_mask
                )
                for key, value in fake_validity.items():
                    fake_validity_dict[f"{modality}_{key}"] = value

        # 计算梯度惩罚（如果使用WGAN-GP）
        gradient_penalties = {}
        if self.adversarial_loss_fn.lambda_gp > 0:
            for modality, generated_feat in generated_features.items():
                if modality in real_specific_features_dict and real_specific_features_dict[modality] is not None:
                    gp = self.model.adversarial_generation.compute_gradient_penalty(
                        real_specific_features_dict[modality],
                        generated_feat,
                        modality
                    )
                    gradient_penalties[modality] = gp

        # 计算判别器损失
        dis_loss, dis_loss_dict = self.adversarial_loss_fn.discriminator_loss(
            real_validity_dict, fake_validity_dict, gradient_penalties
        )

        # 反向传播
        self.discriminator_optimizer.zero_grad()
        dis_loss.backward()
        self.discriminator_optimizer.step()

        return dis_loss.item(), dis_loss_dict

    def train_step(self, shared_features_dict, real_specific_features_dict, modality_mask, batch_idx):
        """
        执行一个训练步骤

        Args:
            shared_features_dict: 共享特征字典
            real_specific_features_dict: 真实特定特征字典
            modality_mask: 模态存在性标记
            batch_idx: 批次索引

        Returns:
            losses: 损失字典
        """
        losses = {}

        if not self.adversarial_training_active:
            return losses

        # 训练判别器
        if batch_idx % self.discriminator_update_freq == 0:
            dis_loss, dis_loss_dict = self.train_discriminator(
                shared_features_dict, real_specific_features_dict, modality_mask
            )
            losses['discriminator'] = dis_loss
            losses.update({f'dis_{k}': v for k, v in dis_loss_dict.items()})

        # 训练生成器
        if batch_idx % self.generator_update_freq == 0:
            gen_loss, gen_loss_dict = self.train_generator(
                shared_features_dict, real_specific_features_dict, modality_mask
            )
            losses['generator'] = gen_loss
            losses.update({f'gen_{k}': v for k, v in gen_loss_dict.items()})

        return losses


def train_single_adversarial(model, cost, optimizer, train_loader, test_loader, args):
    """
    支持对抗学习的训练函数

    Args:
        model: DrfuseAdversarial模型
        cost: 分类损失函数
        optimizer: 主优化器（用于分类器）
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        args: 训练参数
    """
    print("使用支持对抗学习的训练函数")
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

    # 创建对抗损失函数
    adversarial_loss_fn = AdversarialFeatureLoss(
        adversarial_type=getattr(args, 'adversarial_type', 'wgan-gp'),
        consistency_type=getattr(args, 'consistency_type', 'mse'),
        alignment_type=getattr(args, 'alignment_type', 'mmd'),
        lambda_adv=getattr(args, 'lambda_adv', 1.0),
        lambda_consistency=getattr(args, 'lambda_consistency', 1.0),
        lambda_alignment=getattr(args, 'lambda_alignment', 1.0),
        lambda_gp=getattr(args, 'lambda_gp', 10.0)
    )

    # 创建生成器和判别器优化器
    generator_params = list(model.adversarial_generation.generator.parameters())
    discriminator_params = list(model.adversarial_generation.discriminator.parameters())

    generator_optimizer = optim.Adam(
        generator_params,
        lr=getattr(args, 'generator_lr', args.lr * 0.1),
        betas=(0.5, 0.999),
        weight_decay=getattr(args, 'generator_weight_decay', 1e-4)
    )

    discriminator_optimizer = optim.Adam(
        discriminator_params,
        lr=getattr(args, 'discriminator_lr', args.lr * 0.1),
        betas=(0.5, 0.999),
        weight_decay=getattr(args, 'discriminator_weight_decay', 1e-4)
    )

    # 创建对抗训练管理器
    adversarial_trainer = AdversarialTrainer(
        model=model,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        adversarial_loss_fn=adversarial_loss_fn,
        generator_update_freq=getattr(args, 'generator_update_freq', 1),
        discriminator_update_freq=getattr(args, 'discriminator_update_freq', 1),
        adversarial_start_epoch=getattr(args, 'adversarial_start_epoch', 10)
    )

    # 对抗训练参数
    adversarial_training_enabled = getattr(args, 'use_adversarial_training', True)
    missing_rate_adversarial = getattr(args, 'missing_rate_adversarial', 0.3)
    adversarial_schedule = getattr(args, 'adversarial_schedule', 'gradual')

    print(f"对抗训练配置:")
    print(f"  启用对抗训练: {adversarial_training_enabled}")
    print(f"  对抗损失类型: {adversarial_loss_fn.adversarial_loss.loss_type}")
    print(f"  特征一致性类型: {adversarial_loss_fn.consistency_loss.consistency_type}")
    print(f"  模态对齐类型: {adversarial_loss_fn.alignment_loss.alignment_type}")
    print(f"  对抗训练起始epoch: {adversarial_trainer.adversarial_start_epoch}")
    print(f"  模态缺失率: {missing_rate_adversarial}")

    # 学习率调度
    if args.lr_decrease == 'cos':
        print("使用余弦退火学习率调度")
        cos_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch + 20, eta_min=1e-8)
        if args.lr_warmup:
            from lib.model_develop_utils import GradualWarmupScheduler
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1, after_scheduler=cos_scheduler)
    elif args.lr_decrease == 'multi_step':
        print("使用多步学习率调度")
        cos_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[
            int(args.train_epoch * 1 / 6),
            int(args.train_epoch * 2 / 6),
            int(args.train_epoch * 3 / 6)
        ])
        if args.lr_warmup:
            from lib.model_develop_utils import GradualWarmupScheduler
            scheduler_warmup = GradualWarmupScheduler(args, optimizer, multiplier=1, after_scheduler=cos_scheduler)

    # 训练初始化
    epoch_num = args.train_epoch
    save_interval = args.save_interval
    batch_num = 0
    epoch = 0
    accuracy_best = 0
    log_list = []
    cls_sum = 0
    adversarial_gen_sum = 0
    adversarial_dis_sum = 0

    # 加载预训练模型（如果存在）
    if args.retrain and os.path.exists(models_dir):
        print("加载预训练模型")
        state_read = torch.load(models_dir)
        model.load_state_dict(state_read['model_state'])
        optimizer.load_state_dict(state_read['optim_state'])
        epoch = state_read['Epoch']

    # 训练循环
    while epoch < epoch_num:
        adversarial_trainer.set_epoch(epoch)

        # 计算当前epoch的模态缺失率（用于对抗训练）
        if adversarial_schedule == 'gradual':
            current_missing_rate = missing_rate_adversarial * min(epoch / adversarial_trainer.adversarial_start_epoch, 1.0)
        else:
            current_missing_rate = missing_rate_adversarial if epoch >= adversarial_trainer.adversarial_start_epoch else 0.0

        for batch_idx, batch_sample in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch}/{epoch_num}, Adv: {adversarial_trainer.adversarial_training_active}")):

            batch_num += 1
            img_hsi, img_lidar, target = batch_sample['m_1'], batch_sample['m_2'], batch_sample['label']

            if epoch == 0:
                continue

            if torch.cuda.is_available():
                img_hsi = img_hsi.cuda()
                img_lidar = img_lidar.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            # 创建模态缺失掩码（用于对抗训练）
            batch_size = img_hsi.shape[0]
            if adversarial_training_enabled and epoch >= adversarial_trainer.adversarial_start_epoch:
                modality_mask = create_modality_mask(batch_size, current_missing_rate, img_hsi.device)
            else:
                modality_mask = torch.ones(batch_size, 2, device=img_hsi.device)

            # 应用模态缺失掩码
            masked_hsi, masked_lidar = apply_modality_mask(img_hsi, img_lidar, modality_mask)

            # 前向传播
            training_mode = 'adversarial' if adversarial_trainer.adversarial_training_active else 'normal'
            results = model(masked_hsi, masked_lidar, modality_mask, training_mode)

            output = results['output']

            # 计算分类损失
            cls_loss = cost(output, target)
            cls_scalar = cls_loss.mean() if cls_loss.numel() > 1 else cls_loss
            cls_sum += cls_scalar.item()
            total_loss = cls_scalar

            # 对抗训练
            if adversarial_trainer.adversarial_training_active and 'adversarial' in results:
                adversarial_results = results['adversarial']

                # 准备对抗训练数据
                shared_features_dict = {}
                real_specific_features_dict = {}

                if modality_mask[:, 0].mean() > 0.5:  # HSI可用
                    shared_features_dict['hsi'] = results['f_sh_hs']
                    real_specific_features_dict['hsi'] = results['f_sp_hs']

                if modality_mask[:, 1].mean() > 0.5:  # LiDAR可用
                    shared_features_dict['lidar'] = results['f_sh_radar']
                    real_specific_features_dict['lidar'] = results['f_sp_radar']

                # 执行对抗训练步骤
                if shared_features_dict and real_specific_features_dict:
                    adv_losses = adversarial_trainer.train_step(
                        shared_features_dict, real_specific_features_dict, modality_mask, batch_idx
                    )

                    if adv_losses:
                        adversarial_gen_sum += adv_losses.get('generator', 0)
                        adversarial_dis_sum += adv_losses.get('discriminator', 0)

            total_loss.backward()
            optimizer.step()

        # 评估模型
        print(f"评估Epoch {epoch}...")
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

        # 记录最佳精度
        current_accuracy = eval_results['missing_rate_0.0']['accuracy']
        if current_accuracy > accuracy_best and epoch > 5:
            accuracy_best = current_accuracy
            save_path = os.path.join(args.model_root, args.name + '.pth')
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型，精度: {accuracy_best:.4f}")

        # 打印结果
        print(f"Epoch {epoch}, Current Accuracy: {current_accuracy:.4f}, Best Accuracy: {accuracy_best:.4f}")
        if adversarial_trainer.adversarial_training_active:
            print(f"  生成器损失: {adversarial_gen_sum / len(train_loader):.4f}")
            print(f"  判别器损失: {adversarial_dis_sum / len(train_loader):.4f}")
        for missing_rate_key, result in eval_results.items():
            acc = result['accuracy']
            print(f"  {missing_rate_key}: {acc:.4f}")

        # 记录日志
        log_list.append(cls_sum / len(train_loader))
        if adversarial_trainer.adversarial_training_active:
            log_list.append(adversarial_gen_sum / len(train_loader))
            log_list.append(adversarial_dis_sum / len(train_loader))
        else:
            log_list.append(0)
            log_list.append(0)
        log_list.append(current_accuracy)
        log_list.append(accuracy_best)

        # 添加不同缺失率下的性能
        for missing_rate_key, result in eval_results.items():
            log_list.append(result['accuracy'])

        cls_sum = 0
        adversarial_gen_sum = 0
        adversarial_dis_sum = 0

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
                "eval_results": eval_results,
                "adversarial_training_active": adversarial_trainer.adversarial_training_active
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


def evaluate_adversarial_robustness(model, test_loader, args, missing_rates=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """
    评估对抗训练的鲁棒性

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        args: 参数
        missing_rates: 要测试的缺失率列表

    Returns:
        evaluation_report: 评估报告
    """
    print("进行对抗训练鲁棒性评估...")

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
        'average_robustness': sum(robustness_scores.values()) / len(robustness_scores) if robustness_scores else 0.0
    }

    print("评估报告:")
    print(f"基线精度: {baseline_accuracy:.4f}")
    print(f"平均鲁棒性得分: {evaluation_report['average_robustness']:.4f}")

    for missing_rate, robustness in robustness_scores.items():
        print(f"{missing_rate}: {robustness:.4f}")

    return evaluation_report


def compare_training_methods(original_results, adversarial_results):
    """
    对比不同训练方法的效果

    Args:
        original_results: 原始方法的结果
        adversarial_results: 对抗训练方法的结果

    Returns:
        comparison_report: 对比报告
    """
    print("对比不同训练方法的效果...")

    comparison_report = {
        'baseline_improvement': adversarial_results['baseline_accuracy'] - original_results['baseline_accuracy'],
        'robustness_improvement': adversarial_results['average_robustness'] - original_results['average_robustness'],
        'detailed_comparison': {}
    }

    for missing_rate in ['0.1', '0.2', '0.3', '0.4', '0.5']:
        original_acc = original_results['detailed_results'][f'missing_rate_{missing_rate}']['overall_accuracy']
        adversarial_acc = adversarial_results['detailed_results'][f'missing_rate_{missing_rate}']['overall_accuracy']

        comparison_report['detailed_comparison'][f'missing_rate_{missing_rate}'] = {
            'original': original_acc,
            'adversarial': adversarial_acc,
            'improvement': adversarial_acc - original_acc
        }

    print("对比结果:")
    print(f"基线精度提升: {comparison_report['baseline_improvement']:.4f}")
    print(f"鲁棒性提升: {comparison_report['robustness_improvement']:.4f}")

    for missing_rate, comparison in comparison_report['detailed_comparison'].items():
        print(f"{missing_rate}: 原始={comparison['original']:.4f}, 对抗={comparison['adversarial']:.4f}, 提升={comparison['improvement']:.4f}")

    return comparison_report


def test_adversarial_training():
    """测试对抗训练功能"""
    print("测试对抗训练功能...")

    import argparse

    # 创建测试参数
    args = argparse.Namespace()
    args.class_num = 15
    args.lr = 0.001
    args.train_epoch = 5
    args.save_interval = 2
    args.lr_decrease = 'cos'
    args.lr_warmup = False
    args.retrain = False
    args.model_root = '../output/models'
    args.log_root = '../output/logs'
    args.name = 'test_adversarial'
    args.figure_root = '../output/figures'

    # 对抗训练参数
    args.use_adversarial_training = True
    args.adversarial_type = 'wgan-gp'
    args.lambda_adv = 1.0
    args.lambda_consistency = 1.0
    args.lambda_alignment = 1.0
    args.lambda_gp = 10.0
    args.adversarial_start_epoch = 2
    args.missing_rate_adversarial = 0.3

    # 创建模型
    model = DrfuseAdversarial(
        args=args,
        hsi_channels=144,
        radar_channels=1,
        feature_dim=64,
        lambda_gp=args.lambda_gp
    )

    print(f"创建对抗训练模型成功")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 创建优化器
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

    # 创建损失函数
    cost = nn.CrossEntropyLoss()

    # 创建对抗损失函数
    adversarial_loss_fn = AdversarialFeatureLoss(
        adversarial_type=args.adversarial_type,
        lambda_adv=args.lambda_adv,
        lambda_consistency=args.lambda_consistency,
        lambda_alignment=args.lambda_alignment,
        lambda_gp=args.lambda_gp
    )

    # 创建对抗训练管理器
    adversarial_trainer = AdversarialTrainer(
        model=model,
        generator_optimizer=optim.Adam(model.adversarial_generation.generator.parameters(), lr=args.lr * 0.1),
        discriminator_optimizer=optim.Adam(model.adversarial_generation.discriminator.parameters(), lr=args.lr * 0.1),
        adversarial_loss_fn=adversarial_loss_fn,
        adversarial_start_epoch=args.adversarial_start_epoch
    )

    print("对抗训练配置测试通过!")
    return model, optimizer, cost, adversarial_trainer


if __name__ == "__main__":
    test_adversarial_training()