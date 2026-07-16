# 多数据集正式证据汇总

## 统一结论

当前实验已经从单一 Houston2013 扩展到 Houston2013、Trento、MUUFL 三个 HS-LiDAR/多模态遥感数据集。三者承担的论文作用不同：

- Houston2013：支撑质量感知预算路由和 Pareto 标签修正，证明方法不仅是静态轻量化。
- Trento：支撑近饱和精度下的约 20% MACs 压缩，说明方法不会只在困难数据集上产生偶然收益。
- MUUFL：支撑困难类别不均衡场景下的缺失/退化模态鲁棒性，说明方法确实改善模态质量不可靠问题。

## 主结果表

| Dataset | Main comparison | Clean / target OA | Robustness evidence | MACs | Thesis role |
| --- | --- | ---: | --- | ---: | --- |
| Houston2013 | Pareto routing (`delta=0.01`) vs static 100% | 75.37 +/- 1.64 mean state OA | Mean regret 0.10 points from per-state best profile | 88.10 +/- 4.03 | Quality-conditioned budget routing |
| Trento | 80% p=0.25 multi-degradation vs 100% baseline | 98.94 +/- 0.94 full OA | Occlusion50 improves 96.46 -> 98.16; downsample4 improves 97.95 -> 98.51 | 79.90 +/- 0.10 | Near-saturated accuracy with compression |
| MUUFL | 80% p=0.25 multi-degradation vs 100% baseline | 86.87 +/- 1.79 full OA | Main-only improves 71.09 -> 83.03; aux-only improves 26.24 -> 57.20; occlusion50 improves 80.08 -> 85.99 | 79.97 +/- 0.07 | Hard-data missing/degraded modality robustness |

## 关键取舍

多数据集结果不应被写成“所有指标都提升”。更准确的论文叙事是：

1. 在 Houston2013 上，Pareto 标签修正后的路由策略以约 88.10% MACs 达到 75.37% mean state OA，并略高于静态 100% 的 74.83% mean state OA。
2. 在 Trento 上，80% 多退化设置保持接近饱和的 full OA，略低于 100% baseline 约 0.16 个百分点，但带来稳定的约 20% MACs 压缩和退化场景鲁棒性提升。
3. 在 MUUFL 上，80% 多退化设置牺牲 1.64 个百分点 full OA，但显著提升 main-only、aux-only 和 occlusion50 等不可靠模态场景，且保持约 20% MACs 压缩。

这套叙事比单纯堆叠轻量化结果更适合作为硕士毕业论文主线，因为它包含方法设计、资源约束、模态缺失、质量退化、多数据集验证和负面结果解释。

## 建议放入论文的表格结构

建议实验章节至少包含三张表：

1. 主表：Houston2013/Trento/MUUFL 的 full OA、AA、Kappa、MACs、Params。
2. 鲁棒性诊断表：full、main-only、aux-only、noise-high、downsample4、occlusion50。
3. 路由分析表：Houston2013 上 static 65/80/100、hand oracle、Pareto oracle、LOO learned routing 的 OA/MAC/regret。

其中 MUUFL 必须报告 AA、Kappa 和 per-class accuracy。MUUFL 类别不均衡严重，单独 OA 会削弱论文可信度。

## 下一步优先级

1. 生成统一 LaTeX 表格片段，把 Trento 和 MUUFL 的正式结果写入 `paper/tables` 或论文主 `.tex`。
2. 为 MUUFL 生成 per-class accuracy 表和图，检查稀有类别是否仍然失败。
3. 如果 MUUFL 稀有类表现太弱，优先做 class-balanced sampler 或 weighted CE 小消融。
4. 将当前三数据集结果转化为实验章节文字：每个数据集明确回答一个论文问题，避免把所有数据集写成同一种对比。
