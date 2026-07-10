# BRM-Net 退化可靠性诊断分析

本分析基于 `docs/generated/brmnet_priority_summary.csv` 中的三种退化状态：噪声、分辨率损失和遮挡。相关性只用于诊断可靠性分支是否随退化强度发生合理变化；由于每个退化族的点数很少，不能作为严格统计显著性结论。

## 关键结论

- 当前主方案 `quality_multi_degradation_p025` 的完整模态 OA 为 86.90%，三个强退化状态平均 OA 为 84.76%。
- 从结果形态看，单纯的可靠性加权不足以稳定解决退化但可用的辅助模态；必须在训练中显式制造退化并提供质量监督。
- 噪声-only 监督主要修复高噪声状态，但对遮挡状态仍然不足，说明退化类型覆盖本身就是方法贡献的一部分。
- p=0.25 的轻量多退化 schedule 比 p=0.50 更适合作为当前论文主结果，因为它在完整模态精度和强退化鲁棒性之间更平衡。
- 需要注意，fusion weight 并不在所有退化族上都单调下降，尤其噪声场景下仍存在权重升高现象。因此论文中更稳妥的说法是“退化监督改善了鲁棒表示与融合行为”，而不是“可靠性权重总能自动抑制退化模态”。

## 退化状态明细

| Method | Mode | Severity | OA (%) | q_aux | Aux fusion | Full-OA drop (%) |
|---|---|---:|---:|---:|---:|---:|
| Full baseline | aux_downsample_4 | 0.75 | 78.37 | 0.181 | 0.406 | 8.60 |
| Full baseline | aux_noise_high | 0.50 | 72.08 | 0.538 | 0.492 | 14.90 |
| Full baseline | aux_occlusion_50 | 0.50 | 74.21 | 0.336 | 0.442 | 12.76 |
| Full baseline | full | 0.00 | 86.97 | 0.256 | 0.424 | 0.00 |
| Noise quality | aux_downsample_4 | 0.75 | 80.14 | 0.936 | 0.485 | 7.31 |
| Noise quality | aux_noise_high | 0.50 | 85.15 | 0.579 | 0.397 | 2.30 |
| Noise quality | aux_occlusion_50 | 0.50 | 73.32 | 0.797 | 0.450 | 14.13 |
| Noise quality | full | 0.00 | 87.45 | 0.895 | 0.474 | 0.00 |
| Multi-deg. p=0.25 | aux_downsample_4 | 0.75 | 84.15 | 0.826 | 0.457 | 2.75 |
| Multi-deg. p=0.25 | aux_noise_high | 0.50 | 84.55 | 0.909 | 0.477 | 2.34 |
| Multi-deg. p=0.25 | aux_occlusion_50 | 0.50 | 85.57 | 0.528 | 0.384 | 1.33 |
| Multi-deg. p=0.25 | full | 0.00 | 86.90 | 0.882 | 0.471 | 0.00 |
| Multi-deg. p=0.50 | aux_downsample_4 | 0.75 | 84.38 | 0.662 | 0.417 | 1.60 |
| Multi-deg. p=0.50 | aux_noise_high | 0.50 | 85.11 | 0.822 | 0.456 | 0.88 |
| Multi-deg. p=0.50 | aux_occlusion_50 | 0.50 | 84.66 | 0.548 | 0.389 | 1.33 |
| Multi-deg. p=0.50 | full | 0.00 | 85.99 | 0.759 | 0.440 | 0.00 |
| Uniform fusion | aux_downsample_4 | 0.75 | 78.35 | 0.513 | 0.500 | 8.37 |
| Uniform fusion | aux_noise_high | 0.50 | 77.01 | 0.513 | 0.500 | 9.71 |
| Uniform fusion | aux_occlusion_50 | 0.50 | 73.87 | 0.520 | 0.500 | 12.84 |
| Uniform fusion | full | 0.00 | 86.72 | 0.515 | 0.500 | 0.00 |

## 相关性诊断

| Method | Family | Metric | n | Pearson r | Spearman r |
|---|---|---|---:|---:|---:|
| Full baseline | noise | OA | 3 | -0.978 | -1.000 |
| Full baseline | noise | Aux fusion weight | 3 | 0.984 | 1.000 |
| Full baseline | resolution | OA | 2 | -1.000 | -1.000 |
| Full baseline | resolution | Aux fusion weight | 2 | -1.000 | -1.000 |
| Full baseline | occlusion | OA | 2 | -1.000 | -1.000 |
| Full baseline | occlusion | Aux fusion weight | 2 | 1.000 | 1.000 |
| Uniform fusion | noise | OA | 3 | -0.993 | -1.000 |
| Uniform fusion | noise | Aux fusion weight | 3 | 0.000 | 0.000 |
| Uniform fusion | resolution | OA | 2 | -1.000 | -1.000 |
| Uniform fusion | resolution | Aux fusion weight | 2 | 0.000 | 0.000 |
| Uniform fusion | occlusion | OA | 2 | -1.000 | -1.000 |
| Uniform fusion | occlusion | Aux fusion weight | 2 | 0.000 | 0.000 |
| Noise quality | noise | OA | 3 | -0.998 | -1.000 |
| Noise quality | noise | Aux fusion weight | 3 | -0.871 | -1.000 |
| Noise quality | resolution | OA | 2 | -1.000 | -1.000 |
| Noise quality | resolution | Aux fusion weight | 2 | 1.000 | 1.000 |
| Noise quality | occlusion | OA | 2 | -1.000 | -1.000 |
| Noise quality | occlusion | Aux fusion weight | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.50 | noise | OA | 3 | -0.955 | -1.000 |
| Multi-deg. p=0.50 | noise | Aux fusion weight | 3 | 0.994 | 1.000 |
| Multi-deg. p=0.50 | resolution | OA | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.50 | resolution | Aux fusion weight | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.50 | occlusion | OA | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.50 | occlusion | Aux fusion weight | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.25 | noise | OA | 3 | -0.939 | -1.000 |
| Multi-deg. p=0.25 | noise | Aux fusion weight | 3 | 0.998 | 1.000 |
| Multi-deg. p=0.25 | resolution | OA | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.25 | resolution | Aux fusion weight | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.25 | occlusion | OA | 2 | -1.000 | -1.000 |
| Multi-deg. p=0.25 | occlusion | Aux fusion weight | 2 | -1.000 | -1.000 |

## 下一步写作使用方式

论文中应把该分析作为机制诊断，而不是主要结果表。主文可以引用退化鲁棒性表说明 p=0.25 多退化监督的收益；补充材料或汇报中再展示相关性图，解释可靠性分支为什么需要退化目标监督。写作时应避免把 q_aux 或 fusion weight 解释为完全校准的物理质量分数，当前证据更适合支持“训练目标使模型在退化状态下保持更稳定的分类性能”。
