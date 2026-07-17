# 进一步调研、现状分析与推进计划（2026-07-17）

## 1. 当前定位

当前项目不适合继续包装成“单纯轻量化”或“单纯模态缺失”的工作。更稳妥的论文定位是：

> 面向资源受限多模态遥感分类的预算约束紧凑融合框架，在模型压缩、模态缺失、辅助模态退化和可导出结构之间建立可复现实验链。

这个定位的优点是能容纳现有结果中的正负现象：

- Houston2013：支撑预算 profile、Pareto 标签、质量/退化感知路由等方法线。
- Trento：支撑近饱和精度下约 20% MACs 节省，说明方法不是只在一个数据集上有效。
- MUUFL：支撑困难不均衡数据上的鲁棒性收益，同时暴露 clean full OA 与类别准确率取舍。

## 2. 最新调研判断

结合本地调研材料和本轮检索，最新多模态遥感/模态缺失方向的趋势主要有四类：

1. 从“缺失模态补全”转向“缺失/退化条件下的可靠推理”。也就是说，论文不只问能不能重建缺失模态，而是问缺失、噪声、分辨率下降、遮挡时模型是否仍能稳定分类。
2. 从固定融合转向动态融合、动态路由或条件计算。模型需要根据输入质量、模态可用性或预算切换不同计算路径。
3. 从单一精度指标转向 accuracy-efficiency-robustness trade-off。尤其在星载、机载、边缘部署场景下，MACs、参数、延迟和导出结构必须一起报告。
4. 从只报 OA 转向更细粒度的 AA、Kappa、per-class、confusion matrix 和失败案例分析。MUUFL 这类强不均衡数据集尤其不能只看 OA。

本轮网络检索入口：

- [arXiv: missing modality multimodal remote sensing classification](https://arxiv.org/search/?query=missing+modality+multimodal+remote+sensing+classification&searchtype=all)
- [arXiv: multimodal remote sensing classification missing modality](https://arxiv.org/search/?query=multimodal+remote+sensing+classification+missing+modality&searchtype=all)
- [IEEE Xplore: multimodal remote sensing missing modality classification](https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=multimodal%20remote%20sensing%20missing%20modality%20classification)

## 3. 当前实验现状

### 已具备的论文工作量

- 三数据集：Houston2013、Trento、MUUFL。
- 三 seed：主要 formal 实验已完成 seed0/1/2。
- 多状态：full、main-only、aux-only、noise-high、downsample4、occlusion50。
- 多指标：OA、AA、Kappa、MACs、Params、Latency、per-class、confusion matrix。
- 多方法层次：预算门控、compact export、退化感知可靠性监督、profile/Pareto 路由、weighted CE 补充消融。

### 本轮新增结果

新增 MUUFL weighted CE 三 seed 消融：

| Variant | Full OA | Full AA | Main-only OA | Aux-only OA | Occlusion50 OA | MACs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 80% multi-deg. | 86.87 +/- 1.79 | 89.31 +/- 1.41 | 83.03 +/- 2.14 | 57.20 +/- 2.14 | 85.99 +/- 2.08 | 79.97 +/- 0.07 |
| 80% multi-deg. + weighted CE | 87.28 +/- 1.43 | 89.75 +/- 1.04 | 82.64 +/- 2.20 | 56.41 +/- 5.09 | 86.99 +/- 2.18 | 79.96 +/- 0.07 |

结论：weighted CE 可以略微提高 full OA、AA 和 occlusion50，但会降低 main-only、aux-only，并且第 2 类明显下降。因此它不能作为主创新点，只能作为“类别不均衡补救尝试”的补充消融。

新增 Houston2013 validation-derived Pareto routing 初版：

| Router | Samples | Mean OA | Mean MACs | Routing acc. | Mean regret | Mean saving |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| State-level LOO Pareto router | 11/seed | 0.7464 | 0.8929 | 0.6667 | 0.0082 | 0.1071 |
| Validation-state-derived router | 33 total | 0.7459 | 0.8622 | 0.3333 | 0.0243 | 0.0799 |

解释：validation-state-derived router 以更低 MACs 达到接近旧 LOO 的 mean OA，但 routing accuracy 与 regret 更差。进一步检查发现，Pareto target 跨 seed 不稳定：除 aux-only 外，几乎所有状态在 seed0/1/2 中都出现不同 budget label。因此当前瓶颈不是“router 没训练好”这么简单，而是 profile label 本身随 seed 波动。后续如果继续强化路由，应优先保存 patch-wise quality/confidence 特征，或者改用更稳定的 constrained label。

## 4. 现阶段最实在的取舍

### 不建议继续扩大的方向

- 不建议把 weighted CE 升级成核心方法。收益太小，副作用明确。
- 不建议现在启动第四、第五个数据集。当前论文更缺的是深度分析，而不是再扩数据集数量。
- 不建议把 learned router 写成成熟贡献。当前 leave-one-state-out 结果仍说明 state-level 样本太少。

### 建议保留为主线的方向

1. 预算约束结构导出：能真实导出 80% MACs compact model。
2. 退化感知可靠性监督：能解释 degraded auxiliary 下的鲁棒性提升。
3. 多数据集证据链：Houston2013 方法完整，Trento 证明轻量化可迁移，MUUFL 证明困难数据鲁棒性。
4. 失败模式分析：MUUFL confusion matrix、per-class delta、weighted CE 负面/混合结果。

## 5. 下一步计划

### P0：论文内容收束

- 把 weighted CE 表作为补充消融加入实验节。
- 在 discussion 中强调它说明“静态类别权重不足以解决鲁棒训练的类别取舍”。
- 避免声称所有数据集 clean OA 均提升，改为强调 trade-off。

### P1：路由创新加深

- 已完成 validation-state-derived 初版：每个 seed/state 作为一个监督样本，输出 summary、sample table 和 label stability table。
- 下一步应推进真正 patch-level：在验证集 batch 上保存质量探针、预测置信度、正确/错误、模态状态，并由 profile bank 生成更细的路由标签。
- 指标继续使用 achieved OA、MACs、regret、saving 和 routing accuracy，但必须增加 label stability 或 label entropy，否则 learned router 的负面结果无法解释。

### P2：图表与答辩材料

- 保留 Figure 1 架构图、Figure 2 结果总览、MUUFL error analysis。
- 为答辩准备一页“为什么不是简单轻量化”的图：三数据集 x 多状态 x 多指标。
- 增加一页“负面结果如何解释”：MUUFL clean OA 下降、weighted CE 副作用、Trento latency 不随 MACs 下降。

### P3：演示系统

- MVP 不需要重训模型，优先读取现有 compact metrics 和 figures。
- 页面包括数据集选择、模态状态选择、预算 profile、结果对比、资源统计和 failure-case 解释。
- 后端可以先是静态 JSON + FastAPI，前端展示论文中的核心实验表和图。

## 6. 当前判断

当前工作已经具备硕士论文的基本工作量，但创新性需要靠“系统化问题定义 + 多维实验证据 + 可解释失败分析”来支撑，而不是靠某一个单点技巧。
后续最有价值的代码推进不是继续堆小消融，而是把 patch-level / validation-derived routing 做出来；最有价值的论文推进是把现有正负结果组织成清晰、诚实、有深度的 trade-off 叙事。
