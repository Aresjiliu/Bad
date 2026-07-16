# Trento seed0 正式实验结果记录

日期：2026-07-16

## 实验设置

- 数据集：`D:\Academic\data\Trento-main`
- 固定划分：`output\splits\trento_seed0.npz`
- 协议：每类训练样本数为 129/125/105/154/184/122，共 819 个训练像素；测试集 29,395 个像素。
- 模态：HSI 63 bands + auxiliary `first` channel。
- 训练：20 epochs；CUDA；batch size 64。
- 对比：
  - `trento_100`：100% budget，不加质量退化监督。
  - `trento_p025_80`：80% MAC budget，`lambda_budget=1.0`，`lambda_quality=1.0`，modality dropout 0.25，多类型退化监督概率 0.25，退化类型为 noise/downsample_4/occlusion_50。

## 核心结果

| Variant | Source full OA | Compact full OA | Compact AA | Kappa | Compact MACs | Params | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% baseline | 99.47 | 99.24 | 99.05 | 98.98 | 100.00 | 100.00 | 1.894 ms |
| 80% p=0.25 multi-deg. | 99.14 | 99.52 | 99.26 | 99.36 | 80.01 | 80.37 | 1.964 ms |

## 鲁棒性与缺失模态结果

| Variant | Main-only OA | Aux-only OA | Aux noise high OA | Aux downsample 4 OA | Aux occlusion 50 OA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% baseline | 87.85 | 71.20 | 98.40 | 98.20 | 97.12 |
| 80% p=0.25 multi-deg. | 94.82 | 52.06 | 99.57 | 99.22 | 99.38 |

## 初步判断

这组结果比一轮 smoke 更接近论文可用证据。Trento seed0 上，当前 80% 预算方案在完整模态、强噪声、下采样和遮挡场景均优于 100% baseline，同时实际 MACs 降到约 80%。这对论文非常有利，因为它能支持“精度-效率-鲁棒性联合优化”的主线，而不是只证明轻量化。

需要谨慎表述的点是 aux-only：80% p=0.25 方案的 aux-only OA 从 71.20% 降到 52.06%。这说明该方法更偏向以 HSI 主模态为核心、对退化辅助模态进行抑制；它不是一个增强纯辅助模态单独分类能力的方法。论文中应把 aux-only 写成极端诊断场景，而不是主要目标场景。

## 下一步

1. 跑 Trento seed1/seed2 的 `100% baseline` 与 `80% p=0.25 multi-deg.`，确认 seed0 的强结果是否稳定。
2. 生成 Trento 3-seed summary，并与 Houston 表格并列。
3. 在 MUUFL loader 完成后，用 MUUFL 检验该方法是否仍能在更不平衡的数据集上保持优势。

## 3-seed 更新

seed1/seed2 已完成后，seed0 的“80% 方案全面优于 100% baseline”不能直接扩大为最终结论。3-seed 平均结果显示，80% p=0.25 multi-degradation 方案以约 79.90% MACs 达到 98.94% full OA，而 100% baseline 为 99.10% full OA。也就是说，该方案的主要价值是显著降低 MACs，同时维持接近饱和的完整模态精度。

更有价值的是鲁棒性方向：80% 方案在 downsample4 和 occlusion50 上的 3-seed 平均 OA 分别为 98.51% 和 98.16%，高于 100% baseline 的 97.95% 和 96.46%。这支持把 Trento 写成“轻量预算下仍保持强退化鲁棒性”的多数据集证据。

当前不足也很明确：80% 方案的 compact latency 均值为 2.16 ms，100% baseline 为 2.02 ms，未体现 MAC 下降带来的实际延迟收益。这可能来自 CUDA 小模型计时噪声、结构化通道减少不够大、或 compact 模型内存访问/卷积核形状不占优。论文里应优先报告 MACs/Params，latency 需要后续 ONNX Runtime 或更稳定的批量计时验证。
