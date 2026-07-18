# BRM-Net 投稿稿写作修订记录（2026-07-19）

## 修订目标

本轮修订服务于“投稿优先”轨道，重点不是继续扩大方法范围，而是让题目、摘要和引言更准确地表达当前已经有证据支撑的主线：

> fusion-compatible budgeted compact export + degradation-supervised reliability + multi-state evaluation

因此，本轮避免把 thesis-only 的 dynamic routing、patch-level router、weighted CE 和 demo system 重新写入投稿主线。

## 已修改的外部投稿文件

- `D:\Academic\paper_submission\brmnet_pricai2026\paper.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\00_abstract.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\01_intro.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\02_related_work.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\03_method.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\04_experiments.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\05_discussion.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\06_conclusion.tex`

## 核心修改

### 1. 题目收紧

原题目：

> Budgeted Compact Fusion with Degradation-Calibrated Reliability for Multimodal Remote Sensing

修订后：

> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

修改理由：原题目容易让审稿人认为贡献是普通“轻量融合模块”。新题目把真正有证据支撑的创新点提前：物理 compact export、fusion compatibility、degradation-supervised reliability。

### 2. 摘要增强证据链

摘要新增或强化了三点：

- final model must be physically compact rather than only softly gated during training；
- export procedure preserves terminal feature dimensions required by multimodal fusion；
- reliability estimates are diagnostic fusion signals, not overclaimed physical sensor-quality calibration.

这样可以把“模型结构不复杂”的风险转化为“可导出、可验证、可部署”的优势。

### 3. 引言补足技术 gap

引言新增了两个关键 gap：

- 软门控压缩不等于真实可部署网络，必须报告实际 MAC/parameter reduction；
- 多模态结构压缩不同于单主干剪枝，因为分支内通道删除不能破坏融合层所需的终端表示。

同时加入了保守定位：

> BRM-Net is not intended to replace large foundation or expert-routing systems, but to test whether a lightweight multimodal classifier can be exported at a requested budget and remain reliable under complete, missing, and degraded input states.

这句话用于防止论文被要求与大型 foundation/expert routing 系统正面对打。

### 4. 全文术语一致性清理

进一步通读 Method、Experiments、Discussion 和 Conclusion 后，统一了以下表述：

- 将缺失模态处理从容易混淆的 `routing` 改为 `availability-aware fusion`；
- 将 `degradation-calibrated reliability` 改为更稳妥的 `degradation-supervised reliability`；
- 将实验中的 `useful routing` 改为 `useful fusion weights`；
- 将 related-work 定位表中的 BRM-Net 主区别改为 `Joint compact export and degradation-supervised reliability`；
- 保留“not fully calibrated physical quality measurements”等限制性表达，因为这是对可靠性曲线非严格单调现象的必要保守说明。

## 验证结果

- TeX Live 2026 编译成功：`D:\Academic\paper_submission\brmnet_pricai2026\paper.pdf`
- PDF 页数：10 pages
- PDF 大小：407697 bytes
- readiness scanner：PASS，0 findings
- LaTeX 日志硬错误扫描：未发现 `Undefined`、`LaTeX Error`、`Fatal`、`Overfull`、`Warning: Citation`

## 下一步建议

下一步不应急于加新实验。更实在的推进顺序是：

1. 准备一版给导师的邮件或汇报说明，明确请导师决定题目、MUUFL 是否留主文、是否进入最终语言润色。
2. 若导师确认投稿主线，下一步进入逐段语言润色和图表 caption 压缩。
3. 若导师认为创新性仍不足，再从 thesis track 中选择 patch-level routing 或 larger-backbone transfer 作为补充，而不是继续在当前投稿主线里临时堆模块。
