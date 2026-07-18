# BRM-Net 投稿版导师汇报（2026-07-19）

## 1. 当前定位

当前建议先走投稿轨道，论文主线固定为：

> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

投稿版只证明一个集中问题：在多模态遥感分类中，能否把预算可控的结构化 compact export 与退化监督可靠性融合结合起来，在真实导出的轻量模型上保持完整、缺失和退化模态下的稳定表现。

暂不把 dynamic routing、weighted CE、patch-level router、demo system 作为投稿贡献。这些内容保留到毕业论文扩展轨道。

## 2. 当前稿件状态

- Readiness scanner 状态：**PASS**。
- 当前 PDF：`D:\Academic\paper_submission\brmnet_pricai2026\paper.pdf`，大小 396.5 KB。
- 当前主文已聚焦 compact export、fusion compatibility、availability mask、degradation-supervised reliability 和多数据集 trade-off。

## 3. 已完成的关键证据

| 证据问题 | 当前结果 | 解释 |
|---|---:|---|
| 预算 65% | 64.96 ± 0.12 actual MAC ratio | 证明目标预算与实际导出资源可对齐。 |
| 预算 80% | 80.01 ± 0.10 actual MAC ratio | 当前主实验默认预算。 |
| 预算 90% | 90.07 ± 0.08 actual MAC ratio | 形成预算 sweep，不是单点结果。 |
| Source vs compact | 86.90 ± 1.65 compact full OA | 证明结果来自物理导出模型，不是只看软门控源模型。 |
| Uniform-width 对照 | 86.29 ± 0.77 full OA | 比 learned nonuniform export 低 0.61 pp；说明统一缩宽不是完全等价替代。 |
| w/o fusion mask | 76.72 ± 2.97 HSI-only OA | 比主方案低 3.32 pp；说明缺失模态需要 fusion availability mask。 |
| Downsample-4 robustness | 84.15 ± 1.46 OA | 比 compact baseline 高 5.78 pp；说明退化监督对低分辨率辅助模态有效。 |

## 4. 多数据集结果

| 数据集 | 投稿版使用方式 | Full OA | 说明 |
|---|---|---:|---|
| Houston2013 | 主实验与消融 | 86.90 ± 1.65 | 官方 split，证据最完整。 |
| Trento | 外部近饱和场景 | 98.94 ± 0.94 | 约 80% MAC 下保持接近饱和准确率。 |
| MUUFL | 困难 stress test | 86.87 ± 1.79 | clean full OA 非绝对最优，但 missing/degraded modality trade-off 更有信息量。 |

## 5. 可靠性曲线结论

- Under the selected multi-degradation p=0.25 setting, aux reliability changes by +0.023 from low to high noise and by -0.296 from 25% to 50% occlusion.
- Downsample-4 compact OA improves from 78.37% for the compact baseline to 84.15% with multi-degradation supervision.
- The fusion weight curves are diagnostic rather than a calibration guarantee: they expose whether the reliability branch changes behavior under controlled corruption, but they should not be interpreted as physical sensor-quality measurements.

这部分在论文中应作为机制诊断，而不是写成完全校准的物理质量估计。当前稿件已经采用保守表述。

## 6. 当前风险

- 创新性风险：模型本身仍偏轻量，投稿叙事必须强调“真实 compact export + 退化监督可靠性 + 多状态评估协议”的组合贡献，而不是只说轻量化。
- 可靠性解释风险：q_aux 和 fusion weight 不是所有退化族都严格单调，不能写成完全校准分数。
- 多数据集风险：MUUFL 是 trade-off 证据，不是 clean full OA 全面胜利，需要继续诚实表述。
- 篇幅风险：routing、weighted CE、demo 等毕业论文内容不能重新进入投稿主文。

## 7. 建议请导师决策的问题

1. 当前投稿是否以 Houston2013 为主、Trento/MUUFL 为外部验证，而不是继续新增第四个数据集？
2. MUUFL 是否保留在主文多数据集表中，还是放到补充材料以降低解释成本？
3. 是否接受当前题目中的 `Fusion-Compatible Budgeted Compact Export` 表述？
4. 下一步是否先进入语言润色和 related work 补强，而不是继续加实验？

## 8. 下一步执行建议

- 短期：把该汇报交给导师，确认投稿主线和 MUUFL 展示方式。
- 中期：做 related work 引用密度审查，补 compact export、missing modality、degradation-aware fusion 三类引用。
- 长期：投稿版冻结后，转入毕业论文扩展轨道，整理 routing、weighted CE、演示系统和失败分析章节。

