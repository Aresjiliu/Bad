# BRM-Net 投稿优先分轨计划（2026-07-17）

## 1. 分轨决策

参考 `D:\Download\BRM-Net_毕业设计与学术投稿双目标分轨规划.md` 后，当前项目从即刻起采用“双目标分轨”：

- **当前优先目标**：先完成一篇可投稿论文。
- **毕业论文目标**：保留为第二阶段，在投稿范围冻结后再展开。
- **共享底座**：代码、数据划分、实验日志、结果 CSV、图表脚本、BibTeX、核心方法事实必须共用。
- **严格隔离**：投稿正文不再承载毕业设计的完整工作量证明。

投稿论文的唯一主线冻结为：

> **Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification**

投稿论文只回答三个问题：

1. 能否在指定 MAC 预算下导出真实紧凑的多模态模型？
2. 多模态分支被结构化压缩后，如何保持融合接口有效？
3. 不使用重建网络时，退化监督的可靠性融合能否提升缺失/低质量辅助模态下的鲁棒性？

## 2. 当前混淆问题

当前稿件 `D:\Academic\paper_submission\brmnet_pricai2026` 已经包含下列投稿范围外内容：

- Houston budget-profile routing comparison；
- state-level LOO router、validation-derived router、mean-profile stable-label router；
- MUUFL per-class 长表；
- MUUFL confusion/error analysis 主图；
- MUUFL weighted CE 消融；
- 三数据集完整叙事和较长的失败分析；
- 讨论中关于 patch-level routing、class-wise maps、future router 的较多展开。

这些内容本身有价值，但会让投稿主线发散。新的处理规则是：

| 内容 | 投稿正文 | 投稿附录 | 毕业论文 |
|---|---:|---:|---:|
| static compact export | 保留 |  | 保留 |
| fusion-compatible constraint | 保留 |  | 保留 |
| degradation-supervised reliability | 保留 |  | 保留 |
| availability mask / modality dropout | 保留 |  | 保留 |
| soft-mask-only vs compact export | 保留 |  | 保留 |
| uniform scaling / lightweight baseline | 保留 |  | 保留 |
| continuous corruption curve | 保留 |  | 保留 |
| actual MACs / Params / latency | 保留 |  | 保留 |
| Trento 或 MUUFL 外部验证 | 保留一个主外部场景，另一个压缩 | 可放完整表 | 保留完整 |
| routing / Pareto profile / stable-label router | 删除或一句话 future | 可选补充 | 保留一章 |
| weighted CE | 删除 | 可选补充 | 保留为失败/诊断 |
| MUUFL per-class 长表 | 不放正文 | 可放补充 | 保留 |
| 演示系统 | 不做投稿任务 |  | 毕业阶段做 |

## 3. 投稿 Scope Freeze

### 3.1 标题

首选：

> Fusion-Compatible Budgeted Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

备选：

> Budgeted Compact Fusion with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

标题中不再出现 routing、profile、class imbalance、demo system。

### 3.2 贡献点

投稿只保留四个贡献：

1. Formulate compact multimodal remote sensing classification under joint resource-budget, modality-availability, and degraded-modality conditions.
2. Propose a fusion-compatible structured export mechanism that physically removes channels while preserving valid multimodal fusion interfaces under explicit MAC budgets.
3. Introduce degradation-supervised reliability fusion to down-weight corrupted but available auxiliary modalities without reconstructing missing inputs.
4. Evaluate exported models through an accuracy-efficiency-robustness protocol covering MACs, parameters, latency, missing modalities, controlled corruption, and external-scene validation.

### 3.3 研究问题

投稿实验只围绕五个 RQ：

- **RQ1**：target MAC 与 actual MAC 是否匹配，是否能真实导出 compact model？
- **RQ2**：learned structural export 是否优于 soft-mask-only、uniform width scaling 和普通轻量 baseline？
- **RQ3**：fusion-compatible constraint 是否必要？
- **RQ4**：degradation-supervised reliability 是否优于 availability mask、modality dropout 或 reliability-only？
- **RQ5**：方法是否能在一个外部数据场景保持合理 accuracy-efficiency-robustness trade-off？

### 3.4 投稿正文图表上限

正文控制在 4 张主图 + 4 张主表以内：

1. **Figure 1**：方法图。Training / Export / Inference 三段式。
2. **Table 1**：Houston2013 主结果，对比 baseline、BRM-Net、资源指标。
3. **Figure 2**：target-actual budget + OA-MAC/Latency trade-off。
4. **Table 2**：关键消融：soft-mask-only、uniform width、w/o availability、w/o degradation reliability、w/o fusion-compatible constraint。
5. **Figure 3**：continuous corruption curves + predicted reliability calibration。
6. **Table 3**：外部场景验证。优先 Trento；MUUFL 作为补充/附录，除非目标 venue 篇幅允许。
7. **Table 4**：latency / Params / MAC protocol summary。
8. **Figure 4**：可选 reliability diagnostic，不再放 MUUFL 大型 error-analysis 主图。

## 4. 立即从投稿正文移出的内容

### 4.1 Routing 全部移出正文

当前 routing 结果适合毕业论文第 7 章，不适合当前投稿主线。原因：

- 需要解释 profile bank、oracle、Pareto label、label stability、state-level router、validation-derived router，篇幅成本过高；
- learned router 尚未达到 patch-level 或真实延迟收益；
- 会把论文从 compact export + reliability fusion 拉向 dynamic routing，导致贡献点分裂；
- 分轨文件明确建议 routing 成熟后作为未来第二篇论文。

投稿处理：

- 删除 `routing_policy_comparison` 正文表。
- 实验节只保留一句：dynamic profile routing is left as future work because this paper focuses on static exported compact models.
- 如确实需要，可以在补充材料放 oracle profile 结果，但不作为贡献。

### 4.2 Weighted CE 移出正文

MUUFL weighted CE 是有价值的失败/诊断结果，但不是投稿主贡献。原因：

- 收益很小且副作用明确；
- 与 compact export / reliability fusion 主线弱相关；
- 会引入 class imbalance 这个新问题。

投稿处理：

- 删除 `muufl_weighted_ce_ablation` 正文表。
- 删除正文中较长的 class-weighted CE 解释。
- 保留到毕业论文“类别不平衡诊断与负面结果”。

### 4.3 MUUFL 深度分析降级

MUUFL 可作为硬数据集补充，但不建议在投稿正文放长篇 per-class 和 confusion matrix。原因：

- 会让论文从方法投稿变成数据集诊断；
- MUUFL clean full OA 下降，需要较长解释；
- 投稿中只需证明外部场景或难例下 trade-off 合理。

投稿处理：

- 若保留 MUUFL：只放在 Table 3 的一行，报告 Full OA / AA / Kappa / main-only / occlusion / MACs。
- per-class 表、confusion matrix、weighted CE 全部转附录或毕业论文。
- 如果篇幅紧，投稿主外部验证只保留 Trento，MUUFL 转 supplement。

## 5. 投稿优先执行计划

### P0：投稿范围冻结与稿件瘦身（当天完成）

1. 在稿件中删除/注释 routing 正文表和对应解释段。
2. 删除/注释 MUUFL weighted CE 正文表和对应解释段。
3. 缩短 MUUFL per-class/error-analysis 叙事，转为 supplement/thesis-only。
4. Introduction 贡献点改成四条投稿贡献，不再提 routing/profile。
5. Discussion 只讨论 compact export、reliability、latency 与外部验证限制。

完成标准：

- 稿件主线读起来只剩 compact export + degradation-supervised reliability。
- 任意一张图表都能回答 RQ1-RQ5 之一。
- 正文不再需要解释 dynamic profile router。

### P1：建立唯一投稿结果源（1 天）

1. 创建 `docs/PAPER_CLAIMS_EVIDENCE_MATRIX_20260717_ZH.md`。
2. 创建或生成 `docs/generated/paper_canonical_results.csv`。
3. 每个论文主张绑定：
   - result file；
   - run dir；
   - seed；
   - config；
   - git commit；
   - table/figure 文件。
4. 标记每条证据状态：
   - `ready`：可进正文；
   - `needs_ablation`：缺关键消融；
   - `supplement_only`：只进附录；
   - `thesis_only`：不进投稿。

完成标准：

- 投稿正文表格不再手工选数，而是从 canonical results 导出。
- 毕业论文和投稿可以共享同一事实来源。

### P2：补投稿必需关键实验（2-5 天）

投稿最低需要补齐或整理：

1. `soft-mask-only vs compact export`
   - 现有 summary 已包含 source 与 compact 指标，优先整理成表，不一定重新训练。
2. `uniform width scaling`
   - 如果已有 `fusion-mode uniform` 仅代表 uniform fusion，不等价于 uniform width。投稿中需明确命名，避免误用。
   - 若没有真正 uniform width baseline，则实现一个 80% channel-width baseline 或降级为 “uniform fusion baseline”。
3. `w/o availability mask`
   - 若实现成本低，补 seed0/1/2；若时间紧，先 Houston seed0，正文写为 ablation diagnostic。
4. `w/o degradation reliability`
   - 可利用 baseline / noise-only / multi-deg existing variants 整理。
5. `w/o fusion-compatible constraint`
   - 若代码不支持，不要硬写。可以用 `soft-mask-only` 作为弱替代，或把该消融移到 future/limitation。

完成标准：

- Table 2 可以闭合 RQ2-RQ4。
- 任何缺失消融都在论文中降低相应声明强度。

### P3：连续退化与可靠性图（1-3 天）

1. 用现有 degradation modes 先生成离散曲线：
   - clean；
   - noise low/mid/high；
   - downsample2/4；
   - occlusion25/50。
2. 图中同时显示：
   - OA；
   - predicted `q_aux`；
   - fusion weight for auxiliary branch。
3. 若连续 severity 数据不足，先命名为 “controlled corruption levels”，不要写成完整连续校准。

完成标准：

- Figure 3 能证明退化监督不是只在单点有效。
- 文中不把 heuristic quality target 夸大为 calibrated probability。

### P4：重新编译并做投稿质量门检查（1 天）

1. 使用 TeX Live 2026 编译。
2. 检查：
   - undefined references；
   - overfull boxes；
   - 表图是否过多；
   - 内部过程词；
   - routing/profile 是否残留在正文贡献中；
   - 数字是否来自 canonical results。
3. 输出 `docs/PAPER_SUBMISSION_READINESS_CHECK_20260717_ZH.md`。

完成标准：

- 稿件达到“可给导师内审”的投稿版本。
- 后续不再随意加入毕业扩展内容。

## 6. 暂停清单

在投稿版本完成前，以下任务暂停：

- patch-level routing；
- learned profile router；
- demo MVP；
- weighted CE 深挖；
- MUUFL 复杂类别改进；
- private/shared gate；
- ONNX 或前端系统；
- 第四/第五数据集；
- 大规模新 related work 扩展。

这些不是删除，而是转入毕业论文或未来论文储备。

## 7. 与前一版计划的关系

此前 `docs/REVIEW_GUIDANCE_CONCRETE_ACTION_PLAN_20260717_ZH.md` 中的很多建议仍然正确，但需要重新分流：

- 投稿继续做：manifest、内部词扫描、soft-mask-only、uniform baseline、w/o availability、w/o degradation、continuous corruption、latency protocol。
- 投稿暂不做：patch-level routing、demo MVP、routing policy comparison 主文展示、weighted CE 主文展示。
- 毕业阶段再做：routing 章节、演示系统、MUUFL class imbalance 深度分析、失败案例库。

## 8. 下一步最小行动

下一次实际推进应从稿件瘦身开始，而不是继续跑新实验：

1. 修改 `D:\Academic\paper_submission\brmnet_pricai2026\sections\01_intro.tex`，冻结四条贡献。
2. 修改 `sections\04_experiments.tex`，删除 routing 和 weighted CE 正文段落。
3. 修改 `sections\05_discussion.tex`，把 routing 和 weighted CE 移到 limitation/future 或删除。
4. 修改 `paper.tex` 或实验节输入，移除投稿正文中的 routing/weighted CE 表。
5. 编译 PDF，确认页数、图表数量和主线集中度。

完成这一步后，再补实验缺口。否则继续加实验只会进一步放大目标混淆。
