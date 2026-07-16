# MUUFL 三随机种子正式实验结果

## 实验设置

- 数据集：MUUFL Gulfport 64-band scene-label 协议。
- 输入模态：HSI 64 通道 + LiDAR `z` 立方体双通道。
- 划分方式：固定随机划分 `output/splits/muufl_seed{0,1,2}.npz`。
- 训练样本：每个 seed 1550 个训练像素，其中类别 1-9 每类 150 个，类别 10-11 每类 100 个。
- 测试样本：每个 seed 52137 个测试像素。
- 对比设置：
  - `muufl_100_baseline`：100% MACs 目标，无模态 dropout，无辅助模态质量退化增强。
  - `muufl_80_p025_multideg`：80% MACs 目标，模态 dropout 概率 0.25，辅助模态质量退化概率 0.25，退化类型为噪声、下采样、遮挡。

## 三种子汇总

| Variant | Seeds | Full OA | Main-only OA | Aux-only OA | Noise-high OA | Downsample4 OA | Occlusion50 OA | MACs | Params | Compact latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| muufl_100_baseline | 3 | 88.51 +/- 1.19 | 71.09 +/- 4.37 | 26.24 +/- 1.18 | 88.52 +/- 1.15 | 84.12 +/- 1.31 | 80.08 +/- 2.32 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.90 +/- 0.29 |
| muufl_80_p025_multideg | 3 | 86.87 +/- 1.79 | 83.03 +/- 2.14 | 57.20 +/- 2.14 | 86.74 +/- 1.78 | 86.66 +/- 2.03 | 85.99 +/- 2.08 | 79.97 +/- 0.07 | 82.25 +/- 0.33 | 1.96 +/- 0.10 |

对应机器可读结果：

- `docs/generated/muufl_formal_seed0_seed1_seed2_runs.csv`
- `docs/generated/muufl_formal_seed0_seed1_seed2_summary.csv`
- `docs/generated/muufl_formal_seed0_seed1_seed2_summary.md`

## 结果解读

MUUFL 是当前三个数据集中最能暴露问题的数据集。它的类别极不均衡，且 LiDAR 辅助模态单独使用时判别能力较弱。因此，100% baseline 在 full OA 上更高是合理的：88.51% 对 86.87%。这说明 80% 多退化设置不能被包装成 clean full-modality accuracy 的全面提升。

但 80% 多退化设置显著改善了缺失和退化模态状态：

- main-only OA 从 71.09% 提升到 83.03%，说明模型在辅助模态不可用时更少依赖 LiDAR 分支。
- aux-only OA 从 26.24% 提升到 57.20%，说明辅助分支在训练过程中获得了更强的独立判别能力。
- downsample4 OA 从 84.12% 提升到 86.66%，occlusion50 OA 从 80.08% 提升到 85.99%，说明多退化增强确实改善了质量下降场景。
- MACs 从 100.00% 降到 79.97%，Params 从 100.00% 降到 82.25%，给出了明确的轻量化收益。

因此，MUUFL 对论文最有价值的结论是：当前方法不是为了在所有数据集上追求最高 clean OA，而是通过资源约束、模态 dropout 和质量退化监督，让模型在计算受限和模态质量不可靠时保持更稳健的表现。

## Per-class 观察

已生成 per-class full-modality accuracy 表：

- `docs/generated/muufl_formal_per_class_seed0_seed1_seed2_runs.csv`
- `docs/generated/muufl_formal_per_class_seed0_seed1_seed2_summary.csv`
- `docs/generated/muufl_formal_per_class_seed0_seed1_seed2_summary.md`

当前 per-class 结果显示，80% 多退化设置的 clean full OA 下降主要来自类别 3、9 和 1，而不是稀有类彻底失败。类别 10 在 80% 设置下略有提升，类别 11 基本持平；类别 4、5、7 也有提升。这说明 MUUFL 的问题更像是压缩和鲁棒训练带来的类别间取舍，而不是简单的 rare-class collapse。

因此，下一步是否需要 weighted loss 或 class-balanced sampler 应谨慎处理。它可以作为附加消融，但不应立刻替代当前主线。更稳妥的做法是先把 per-class table 放入论文，说明该方法在整体鲁棒性提升的同时，对部分高混淆类别存在 clean accuracy tradeoff。

## 论文写法建议

在论文中，MUUFL 不应只放入主表的 OA 指标。建议报告 OA、AA、Kappa、MACs、Params，并额外给出 missing/degraded modality diagnostic table。原因是 MUUFL 类别不均衡严重，单独 OA 容易掩盖稀有类问题。

推荐表述：

> On MUUFL, the compact degradation-aware setting sacrifices 1.64 percentage points of full-modality OA, but improves main-only OA by 11.94 points, auxiliary-only OA by 30.96 points, and occlusion robustness by 5.91 points while reducing MACs by approximately 20%. This result supports the proposed method as a robustness-efficiency tradeoff rather than a clean-accuracy-only optimization.

## 下一步

1. 将 Houston2013、Trento、MUUFL 结果整理到一个统一的 multi-dataset table。
2. 对 MUUFL 补充 per-class accuracy 和 AA/Kappa 的论文表格，避免只依赖 OA。
3. 如果时间允许，对 MUUFL 增加 class-balanced sampler 或 weighted CE ablation，专门处理稀有类别。
4. 将多数据集结论写入 LaTeX 实验章节：Houston 支撑方法主线，Trento 支撑近饱和精度下的压缩，MUUFL 支撑困难数据集上的鲁棒性收益。
