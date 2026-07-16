# Trento CUDA Smoke 实验记录

日期：2026-07-16  
命令环境：`conda run -n hslinets`，CUDA  
数据集：`D:\Academic\data\Trento-main`  
Split：`output/splits/trento_seed0.npz`  
协议：每类训练样本 129/125/105/154/184/122，共 819 train、29,395 test  
辅助模态：`aux_channel_mode=first`

## 目的

本次实验不是正式结果，只用于验证 Trento 已经接入 BRM-Net 的完整训练链路：

1. 原始 `.mat` 数据读取。
2. 固定 split 加载。
3. patch DataLoader。
4. BRM-Net 训练 1 epoch。
5. full/missing/degraded modality 评估。
6. compact export 和资源统计。

## 关键结果

| 状态 | OA | AA | Kappa |
| --- | ---: | ---: | ---: |
| full | 0.9042 | 0.7442 | 0.8719 |
| main_only | 0.8600 | 0.6665 | 0.8122 |
| aux_only | 0.5881 | 0.3917 | 0.4613 |
| aux_noise_high | 0.9061 | 0.7470 | 0.8745 |
| aux_downsample_4 | 0.8851 | 0.7385 | 0.8470 |
| aux_occlusion_50 | 0.6050 | 0.5466 | 0.4991 |

资源统计：

| 项 | Params | MACs | Params ratio | MACs ratio |
| --- | ---: | ---: | ---: | ---: |
| baseline | 442,248 | 8,624,640 | 1.0000 | 1.0000 |
| expected | 442,247.19 | 8,624,624 | 1.0000 | 1.0000 |
| compact | 442,248 | 8,624,640 | 1.0000 | 1.0000 |

## 解释

这个 1 epoch smoke 已经说明 Trento 接入链路可用，但不能和文献 99% OA 结果直接比较。当前 full OA 约 90.4%，明显低于 Trento 近年论文的饱和水平，主要原因是训练轮数极少、没有做正式调参，也未启用质量退化监督。

从模态状态看，HSI 主模态已经有较强可用性，LiDAR 单模态较弱；50% 遮挡导致 OA 大幅下降，说明 Trento 可以用于验证退化鲁棒性，而不仅仅是 full-modality 精度。

## 下一步

1. 跑 Trento seed0 的 20 epoch baseline，预算 100%。
2. 若 20 epoch full OA 明显上升，再补 80% profile。
3. 正式实验前保留 `aux_channel_mode=first`，后续再比较 `both` 或 `mean`。
4. 不提交 smoke checkpoint，只保留指标和说明。
