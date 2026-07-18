# BRM-Net 长期双轨改进计划（2026-07-19）

## 总目标

当前项目继续采用“双轨制”：

- **A 轨：论文投稿优先**。目标是在短期内形成一篇主线集中、证据闭环、可给导师审阅并继续投稿打磨的英文稿件。
- **B 轨：硕士毕业论文扩展**。目标是在投稿版稳定后，把已有 routing、多数据集、类别不均衡、演示系统等工作扩展成能体现工作量和系统性的毕业论文。

原则是：**同一个代码与实验事实库，两套叙事边界**。投稿版不承担全部毕业工作量证明；毕业论文不被投稿篇幅限制压缩。

## A 轨：投稿优先路线

### A1. 投稿主线冻结

投稿主线固定为：

> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

投稿只证明五件事：

1. 目标 MAC 预算与实际 compact export 资源匹配。
2. compact export 是物理结构导出，而不是软 mask 伪压缩。
3. learned nonuniform export 相比 target-matched uniform-width export 更合理。
4. availability mask 与 degradation-supervised reliability 分别处理“缺失模态”和“退化但可用模态”。
5. Houston2013 主实验结论能在 Trento/MUUFL 形成外部场景 trade-off 证据。

### A2. 已完成基础

- 三 seed Houston2013 budgeted compact export。
- source gated model vs physically exported compact model。
- strict uniform-width export 消融。
- w/o fusion availability mask 消融。
- degradation-supervised reliability 表格。
- controlled-corruption reliability curves。
- Trento/MUUFL 多数据集 evidence。
- TeX Live 2026 编译路径。

### A3. 当前最高优先级

1. **投稿 readiness scanner**：自动检查稿件是否残留 routing/weighted CE/prototype/preliminary 等不适合投稿主线的词或表述。
2. **claim-evidence consistency check**：检查 `docs/PAPER_CLAIMS_EVIDENCE_MATRIX_20260717_ZH.md` 与正文图表是否一致。
3. **paper polish pass**：根据 scanner 报告修订 abstract、intro、experiments、discussion 中不稳妥的表达。
4. **advisor report note**：生成一份中文汇报文档，说明当前投稿版主线、完成证据、剩余风险和下一步。

### A4. 投稿停止条件

满足以下条件后，不再继续加新实验，转入导师反馈：

- `latexmk` 编译通过。
- log 中无 undefined reference、LaTeX Error、Fatal、Overfull。
- readiness scanner 无 P0/P1 问题。
- 所有主表/主图都能对应到一个 claim。
- 正文不再把 routing、weighted CE、demo、patch-level router 写成投稿贡献。

## B 轨：毕业论文扩展路线

投稿版稳定后，毕业论文按章节扩展：

1. **系统综述与问题定义**：多模态遥感、模态缺失、轻量化、动态推理。
2. **BRM-Net 投稿版核心方法**：预算门控、compact export、availability mask、degradation reliability。
3. **多数据集实验**：Houston2013、Trento、MUUFL 的完整结果与失败分析。
4. **质量条件预算路由**：65/80/100 profile bank、Pareto labels、stable-label router、label instability。
5. **负面结果与诊断**：MUUFL weighted CE、副作用、类别不均衡、Trento latency/MAC 不一致。
6. **演示系统**：JSON-backed dashboard，展示数据集、模态状态、profile、AER 表、可靠性曲线、失败案例。

## C 轨：工程资产整理

长期维护三类工程资产：

1. **可复现实验资产**：runner、matrix、summary、generated CSV/MD。
2. **论文生成资产**：LaTeX tables、figures、readiness report。
3. **毕业展示资产**：pipeline note、demo JSON、dashboard、答辩图。

## 未来 10 个连续推进任务

1. 实现投稿 readiness scanner，生成 `docs/PAPER_SUBMISSION_READINESS_CHECK_YYYYMMDD_ZH.md`。
2. 根据 scanner 修订投稿稿件中的范围外词和弱表述。
3. 将 claim-evidence matrix 与正文图表编号对齐。
4. 生成导师汇报版中文 md：当前主线、证据、风险、下一步。
5. 对 Figure 1/2/3 做一次最终视觉审查，记录是否需要重画。
6. 做一次 related work 引用密度审查，补足与 compact export / degradation reliability 直接相关的引用。
7. 冻结投稿版实验 manifest：git commit、summary hash、paper pdf hash、生成脚本命令。
8. 进入毕业扩展：整理 routing 章节的核心结果，不进入投稿主文。
9. 设计 demo system 的 JSON 数据接口和最小页面结构。
10. 实现 demo MVP，只读取已有 generated 文件，不依赖重新训练。

## 本轮执行项

本轮从任务 1 开始：实现投稿 readiness scanner。它是后续长期推进的质量门禁，可以避免每次修论文时重新混入毕业论文范围外内容。
