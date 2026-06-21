import os
import numpy as np
from argparse import ArgumentParser

# Enhanced configuration for adversarial training with FMC compensation

parser = ArgumentParser()

# Basic training parameters (inherited from original config)
parser.add_argument('--train_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_warmup', type=bool, default=True)
parser.add_argument('--total_epoch', type=int, default=5)
parser.add_argument('--lr_decrease', type=str, default='cos', help='the methods of learning rate decay')
parser.add_argument('--mixup', type=bool, default=False, help='using mixup or not')
parser.add_argument('--mixup_alpha', type=float, default=0.2)
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.90)
parser.add_argument('--class_num', type=int, default=15)
parser.add_argument('--retrain', type=bool, default=False, help='Separate training for the same training process')
parser.add_argument('--log_interval', type=int, default=10, help='How many batches to print the output once')
parser.add_argument('--save_interval', type=int, default=50, help='How many batches to save the model once')
parser.add_argument('--model_root', type=str, default='../output/models/')
parser.add_argument('--log_root', type=str, default='../output/logs/')
parser.add_argument('--modal', type=str, default='multi')
parser.add_argument('--model_name', type=str, default='mnist_cnn_best')
parser.add_argument('--log_name', type=str, default='.csv')
parser.add_argument('--backbone', type=str, default='couple_cross_fc')
parser.add_argument('--patch_size', type=int, default=7)
parser.add_argument('--p', default=[0, 0], help='para for modality dropout')
parser.add_argument('--location', type=str, default='after')

parser.add_argument('--labma_unimodal', type=float, default=1.0)
parser.add_argument('--data_root', type=str, default='Huston2013')
parser.add_argument('--pair_modalities', type=str, default='hsi+lidar')
parser.add_argument('--l1_loss', type=float, default=0.003)
parser.add_argument('--gama', type=float, default=1.003)
parser.add_argument('--osc', type=float, default=0.8)
parser.add_argument('--t_max', type=float, default=2.0)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--loss_num', type=int, default=3)
parser.add_argument('--figure_root', type=str, default='../output/depose/')
parser.add_argument('--vis_interval', type=int, default=50)
parser.add_argument('--identity', type=str, default='mcl')
parser.add_argument('--threshold', type=float, default=0.0)
parser.add_argument('--lambda0', type=float, default=0.8)
parser.add_argument('--lambda1', type=float, default=0.3)
parser.add_argument('--lambda2', type=float, default=0.2)
parser.add_argument('--lidar_lambda', type=float, default=0.8)

# Loss function switches
parser.add_argument('--use_md_loss', type=bool, default=False, help='是否使用MD损失函数')
parser.add_argument('--use_id_loss', type=bool, default=False, help='是否使用ID损失函数（IdentityClassificationLoss）')
parser.add_argument('--use_intra_loss', type=bool, default=False, help='是否使用类内紧凑性损失（IntraClassCompactnessLoss）')
parser.add_argument('--use_drfuse_loss', type=bool, default=False, help='是否使用DrFuse损失函数')

# Enhanced adversarial training parameters
parser.add_argument('--use_fmc_enhancement', type=bool, default=True, help='是否使用FMC增强')
parser.add_argument('--use_adversarial_training', type=bool, default=True, help='是否使用对抗训练')
parser.add_argument('--feature_dim', type=int, default=128, help='特征维度')
parser.add_argument('--shared_dim', type=int, default=64, help='共享特征维度')
parser.add_argument('--specific_dim', type=int, default=64, help='特定特征维度')
parser.add_argument('--compensation_mode', type=str, default='adaptive', help='补偿模式')

# Adversarial training loss weights
parser.add_argument('--lambda_adv', type=float, default=1.0, help='对抗损失权重')
parser.add_argument('--lambda_rec', type=float, default=0.5, help='重构损失权重')
parser.add_argument('--lambda_cls', type=float, default=0.3, help='分类损失权重')
parser.add_argument('--lambda_consistency', type=float, default=0.2, help='一致性损失权重')
parser.add_argument('--lambda_feature', type=float, default=0.1, help='特征匹配损失权重')
parser.add_argument('--lambda_fmc', type=float, default=0.3, help='FMC损失权重')

# Modality missing simulation
parser.add_argument('--missing_modality_rate', type=float, default=0.3, help='模态缺失率')
parser.add_argument('--curriculum_learning', type=bool, default=True, help='是否使用课程学习')
parser.add_argument('--missing_rate_schedule', type=str, default='linear', help='缺失率调度策略')
parser.add_argument('--initial_missing_rate', type=float, default=0.1, help='初始缺失率')
parser.add_argument('--final_missing_rate', type=float, default=0.5, help='最终缺失率')

# Generator and discriminator specific parameters
parser.add_argument('--generator_lr', type=float, default=0.0002, help='生成器学习率')
parser.add_argument('--discriminator_lr', type=float, default=0.0002, help='判别器学习率')
parser.add_argument('--generator_beta1', type=float, default=0.5, help='生成器Adam beta1')
parser.add_argument('--generator_beta2', type=float, default=0.999, help='生成器Adam beta2')
parser.add_argument('--discriminator_beta1', type=float, default=0.5, help='判别器Adam beta1')
parser.add_argument('--discriminator_beta2', type=float, default=0.999, help='判别器Adam beta2')

# FMC module specific parameters
parser.add_argument('--fmc_spatial_attention_dim', type=int, default=8, help='FMC空间注意力维度')
parser.add_argument('--fmc_channel_attention_ratio', type=int, default=16, help='FMC通道注意力压缩比')
parser.add_argument('--fmc_residual_scale', type=float, default=1.0, help='FMC残差缩放因子')
parser.add_argument('--fmc_cross_modal_layers', type=int, default=3, help='FMC跨模态转换层数')

# Multi-scale discriminator parameters
parser.add_argument('--discriminator_scales', type=list, default=['original', 'downsampled_2', 'downsampled_4'],
                    help='判别器多尺度')
parser.add_argument('--discriminator_feature_dim', type=int, default=128, help='判别器特征维度')
parser.add_argument('--discriminator_leaky_relu_slope', type=float, default=0.2, help='判别器LeakyReLU斜率')

# Training stability parameters
parser.add_argument('--gradient_penalty', type=bool, default=False, help='是否使用梯度惩罚')
parser.add_argument('--gradient_penalty_lambda', type=float, default=10.0, help='梯度惩罚系数')
parser.add_argument('--discriminator_train_interval', type=int, default=1, help='判别器训练间隔')
parser.add_argument('--generator_train_interval', type=int, default=1, help='生成器训练间隔')
parser.add_argument('--feature_matching', type=bool, default=True, help='是否使用特征匹配')
parser.add_argument('--spectral_normalization', type=bool, default=False, help='是否使用谱归一化')

# Evaluation and testing parameters
parser.add_argument('--test_missing_scenarios', type=bool, default=True, help='是否测试缺失场景')
parser.add_argument('--test_scenarios', type=list,
                    default=['complete', 'hsi_missing', 'lidar_missing', 'random_missing'],
                    help='测试场景')
parser.add_argument('--save_generated_features', type=bool, default=True, help='是否保存生成的特征')
parser.add_argument('--visualize_attention_maps', type=bool, default=True, help='是否可视化注意力图')
parser.add_argument('--feature_analysis_interval', type=int, default=10, help='特征分析间隔')

# Advanced training strategies
parser.add_argument('--progressive_training', type=bool, default=False, help='是否使用渐进式训练')
parser.add_argument('--progressive_epochs', type=int, default=50, help='渐进式训练周期')
parser.add_argument('--self_supervised_pretraining', type=bool, default=False, help='是否使用自监督预训练')
parser.add_argument('--self_supervised_epochs', type=int, default=100, help='自监督预训练周期')

# Data augmentation for robustness
parser.add_argument('--feature_dropout', type=float, default=0.1, help='特征dropout率')
parser.add_argument('--feature_noise', type=float, default=0.01, help='特征噪声强度')
parser.add_argument('--modality_corruption', type=float, default=0.2, help='模态损坏率')

# Memory and computational efficiency
parser.add_argument('--mixed_precision', type=bool, default=True, help='是否使用混合精度训练')
parser.add_argument('--gradient_accumulation', type=int, default=1, help='梯度累积步数')
parser.add_argument('--memory_efficient', type=bool, default=False, help='是否使用内存高效模式')

# Reproducibility
parser.add_argument('--seed', type=int, default=42, help='随机种子')
parser.add_argument('--deterministic', type=bool, default=True, help='是否使用确定性算法')

# Logging and monitoring
parser.add_argument('--wandb_logging', type=bool, default=False, help='是否使用WandB日志')
parser.add_argument('--tensorboard_logging', type=bool, default=True, help='是否使用TensorBoard日志')
parser.add_argument('--log_feature_statistics', type=bool, default=True, help='是否记录特征统计信息')
parser.add_argument('--log_generation_quality', type=bool, default=True, help='是否记录生成质量')

# Model saving and checkpointing
parser.add_argument('--save_best_only', type=bool, default=True, help='是否只保存最佳模型')
parser.add_argument('--save_frequency', type=int, default=10, help='模型保存频率')
parser.add_argument('--checkpoint_keep_num', type=int, default=3, help='保留的检查点数量')

args = parser.parse_args()
args.pair_modalities = args.pair_modalities.split('+')
args.data_root = '../data/' + args.data_root
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
args.name = args.pair_modalities[0] + "_" + args.pair_modalities[1] + str(args.identity)
args.model_root = args.model_root + str(args.identity)
args.log_root = args.log_root + str(args.identity)
args.figure_root = args.figure_root + str(args.identity)

# Create directories if they don't exist
os.makedirs(args.model_root, exist_ok=True)
os.makedirs(args.log_root, exist_ok=True)
os.makedirs(args.figure_root, exist_ok=True)

print("Enhanced configuration loaded with FMC and adversarial training parameters")