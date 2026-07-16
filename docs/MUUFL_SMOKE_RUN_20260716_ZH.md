# MUUFL Gulfport CUDA Smoke 实验记录

日期：2026-07-16  
数据集：`D:\Academic\data\MUUFLGulfport-master`  
Split：`output\splits\muufl_seed0.npz`  
输出目录：`output\muufl_smoke\muufl_hsi-lidar_random_splitseed42_trainseed0_gatehard_concrete_budget100_metricmacs`

## 目的

本次实验只验证 MUUFL 是否已经接入 BRM-Net 的完整实验链路，不作为正式论文结果。检查范围包括：真实 `.mat` 读取、固定坐标 split、patch dataset、CUDA 训练、缺失/退化评估、compact export、资源统计和指标落盘。

## 数据协议

- 使用 scene-label 文件：`MUUFLGulfportSceneLabels\muufl_gulfport_campus_1_hsi_220_label.mat`
- HSI：`325 x 220 x 64`
- LiDAR：`325 x 220 x 2`
- 类别数：11
- 原始未标注标签 `-1` 在加载阶段映射为背景 `0`
- 训练协议：每类固定小样本抽样，1-9 类各 150，10-11 类各 100
- seed0 split：1550 train / 52137 test

## 运行命令

```powershell
conda run -n hslinets python scripts/run_brmnet_houston.py `
  --dataset muufl `
  --data-format raw `
  --data-root 'D:\Academic\data\MUUFLGulfport-master' `
  --split-protocol random `
  --split-file 'output\splits\muufl_seed0.npz' `
  --class-num 11 `
  --aux-channel-mode both `
  --epochs 1 `
  --batch-size 64 `
  --patch-size 7 `
  --num-workers 0 `
  --compact-finetune-epochs 0 `
  --output-dir 'output\muufl_smoke' `
  --device cuda `
  --target-budget 1.0 `
  --lambda-budget 0.0 `
  --latency-warmup 0 `
  --latency-iterations 1
```

## Smoke 指标

| 状态 | OA | AA | Kappa |
| --- | ---: | ---: | ---: |
| full | 80.80 | 69.73 | 74.98 |
| main_only | 65.40 | 49.58 | 54.53 |
| aux_only | 33.27 | 35.84 | 24.87 |
| aux_downsample_4 | 79.12 | 68.44 | 72.91 |
| aux_occlusion_50 | 73.86 | 56.12 | 65.79 |

## 观察

MUUFL 明显比 Trento 更难。1 epoch smoke 已经达到 full OA 80.8%，说明数据链路可用，但 AA 只有 69.7%，且第 9、10 类在该短训练下几乎没有学到。这与前期数据审计一致：MUUFL 类别极不均衡，不能只报告 OA，正式实验必须报告 AA、Kappa、per-class accuracy 和混淆矩阵。

模态诊断也有价值。HSI-only 仍有 65.4% OA，LiDAR-only 只有 33.3% OA，说明 MUUFL 的两通道 LiDAR 单独分类能力弱，但与 HSI 融合后明显提升。50% 遮挡把 OA 从 80.8% 拉到 73.9%，适合作为论文中的退化鲁棒性压力测试。

## 下一步

1. 跑 MUUFL seed0 的 20 epoch 100% baseline，得到正式可比较的起点。
2. 跑 MUUFL seed0 的 80% p=0.25 multi-degradation 设置，观察是否像 Trento 一样保持 full OA 并改善退化状态。
3. 若 seed0 可用，再补 seed1/seed2；若小类仍为 0，应优先增加 class-balanced loss 或 weighted sampler 的消融，而不是盲目拉长训练。
