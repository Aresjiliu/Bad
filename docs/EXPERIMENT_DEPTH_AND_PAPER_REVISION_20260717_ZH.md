# 论文内容与实验深度推进记录

## 当前论文主线

当前论文不应再写成单纯的“轻量化模型”或“模态缺失模型”。更准确的主线是：在多模态遥感分类中，同时处理资源预算、模态缺失、辅助模态退化和可部署紧凑导出。这个主线比单一剪枝或单一缺失模态更适合作为硕士论文，因为它要求同时回答三个问题。

第一，模型是否能按目标资源预算导出真实紧凑结构，而不是只在训练时施加稀疏正则。Houston2013 的 65/80/90/100 profile 和 hard-concrete compact export 回答这个问题。

第二，模型是否能在模态缺失或辅助模态质量下降时保持可用。Houston2013 的 robustness ablation 回答这个问题，尤其是 multi-degradation p=0.25 对 noise、downsample、occlusion 的提升。

第三，结论是否只依赖一个数据集。Trento 和 MUUFL 回答这个问题：Trento 证明近饱和数据集上可以保留高精度并节省约 20% MACs；MUUFL 证明困难不均衡数据集上，方法更像鲁棒性收益而不是 clean OA 收益。

## 已推进的论文修改

已修改 `D:\Academic\paper_submission\brmnet_pricai2026` 中的论文内容：

- `sections/00_abstract.tex`：加入 Trento 和 MUUFL 多数据集结论，并明确 MUUFL clean OA 有损失。
- `sections/01_intro.tex`：贡献点加入 multi-dataset validation。
- `sections/04_experiments.tex`：扩展数据集协议描述，新增 Multi-Dataset Validation 小节，加入 Trento/MUUFL 解释和 MUUFL per-class 分析。
- `sections/05_discussion.tex`：把讨论从 Houston 单数据集扩展为三数据集解释，并明确 learned router、HS-MS/HS-SAR、连续退化曲线仍是限制。
- `sections/06_conclusion.tex`：将结论改成 accuracy-efficiency-robustness trade-off，而不是单纯 clean accuracy 提升。
- `tables/multidataset_formal_evidence.tex`：新增三数据集证据表。
- `tables/muufl_per_class_accuracy.tex`：新增 MUUFL per-class 表。

## 当前实验深度评估

### 已具备的深度

1. 多 seed：Houston、Trento、MUUFL 的关键结果都已经三随机种子汇总。
2. 多状态：不仅报告 full OA，还报告 main-only、aux-only、noise-high、downsample4、occlusion50。
3. 多目标：同时报告 OA、AA、Kappa、MACs、Params、latency、per-class accuracy。
4. 多数据集：Houston2013、Trento、MUUFL 承担不同论文角色。
5. 路由分析：Houston 上已经有 static profile、hand oracle、Pareto/utility label、leave-one-state-out learned routing。
6. 负面结果可解释：MUUFL 80% 设置 clean full OA 下降，但鲁棒性显著提升；learned LOO router 尚不成熟，但 Pareto label 本身有效。

### 仍缺的深度

1. 缺少统一的主实验 LaTeX 表，将 Houston、Trento、MUUFL 的 OA/AA/Kappa/MACs/Params 放在同一格式下。
2. Trento 和 MUUFL 目前主要比较 100% baseline 与 80% p=0.25 multi-degradation，缺少更多预算点或 p 值消融。
3. MUUFL per-class 表已经生成，但还没有混淆矩阵或类别间错误模式可视化。
4. learned router 的训练样本仍是 state-level，泛化能力不强；如果作为论文创新点，需要进一步发展 patch-level 或 validation-derived labels。
5. 目前延迟结果和 MACs 不完全一致，尤其 Trento 上 80% compact latency 没有同步下降，需要在论文中作为限制解释。

## 下一步优先级

### P0：立即增强论文可信度

生成统一多数据集主表，字段包括 Dataset、Method、Full OA、AA、Kappa、Main-only OA、Aux-only OA、Occlusion50 OA、MACs、Params。该表比当前 summary table 更适合论文主实验，因为它同时体现准确率、鲁棒性和轻量化。

已完成：生成 `docs/generated/multidataset_detailed_proposed_summary.md`、`docs/generated/multidataset_detailed_proposed_summary.csv`、`docs/generated/multidataset_detailed_proposed_table.tex`，并同步到论文目录 `tables/multidataset_detailed_proposed.tex`。

### P1：增强 MUUFL 深度

生成 MUUFL 混淆矩阵和 class-wise delta figure。目标不是证明所有类别都提升，而是解释哪些类别受益、哪些类别受损。该图能支撑“robustness-efficiency trade-off”叙事。

已完成：新增 `scripts/plot_muufl_error_analysis.py` 和 `tests/test_plot_muufl_error_analysis.py`，生成 `docs/generated/muufl_error_analysis.{md,csv,pdf,png}`，并同步论文图 `D:\Academic\paper_submission\brmnet_pricai2026\figures\generated\fig_muufl_error_analysis.pdf`。论文实验节已加入 Figure `fig:muufl_error_analysis`。

### P2：补充小规模消融

优先级最高的小消融不是大规模重新训练，而是 MUUFL weighted CE / class-balanced sampler 的 seed0 试验。它回答一个明确问题：类别 3 和 9 的 clean OA 损失是否可以通过类别均衡缓解。如果 seed0 有明显提升，再扩展到 seed1/seed2；如果没有，就作为负面结果说明鲁棒训练与类别混淆之间仍有取舍。

已完成：新增 `--class-weighting inverse_frequency`，完成 MUUFL seed0/seed1/seed2 weighted CE 消融，并生成 `docs/generated/muufl_weighted_ce_seed0_seed1_seed2_summary.md`。三 seed 结果显示 weighted CE 仅小幅提升 full OA/AA，并带来 aux-only 和部分类别副作用。因此该消融应写作“类别不均衡诊断”，而不是新的主方法组件。外部 LaTeX 论文已加入 `tables/muufl_weighted_ce_ablation.tex` 和相应解释段落。

### P3：路由创新加深

将 Pareto label 从 11 个状态级样本扩展到 validation patch 级样本。当前 LOO learned routing 的问题不是方向错误，而是训练样本太少。下一步应从验证集 patch 的质量探针输出、模态状态、预测置信度中生成更多训练样本，再训练轻量 router。

## 论文写作取舍

不要把当前工作写成“提出了复杂大模型并全面超过 SOTA”。更稳的写法是：“提出了一个可部署的紧凑多模态框架，并通过预算导出、模态缺失、质量退化和多数据集实验系统地分析 accuracy-efficiency-robustness trade-off。”这个表述能容纳 MUUFL clean OA 下降、Trento latency 不下降、learned router 不成熟等现象，同时仍然体现工作量和创新性。
