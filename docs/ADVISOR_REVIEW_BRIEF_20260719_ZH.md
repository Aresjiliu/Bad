# BRM-Net 投稿稿导师审阅 brief（2026-07-19）

老师您好，我目前已将 BRM-Net 项目先收束到“投稿优先”轨道，暂时不把毕业论文扩展内容全部塞进投稿稿件。当前稿件主线是：

> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

核心想证明的问题不是提出一个很大的新架构，而是证明一个轻量多模态遥感分类模型可以在指定资源预算下真实导出为 compact model，并且在完整模态、缺失模态、退化但仍可用模态下保持可解释的鲁棒性表现。

## 1. 当前稿件已经完成的主要证据

### 1.1 真实 compact export，而不是只看软门控

当前 Houston2013 三种预算实验已经完成：

| Target MAC budget | Actual MAC ratio | 说明 |
|---:|---:|---|
| 65% | 64.96 +/- 0.12 | 目标预算和实际导出 MAC 对齐 |
| 80% | 80.01 +/- 0.10 | 当前主实验默认预算 |
| 90% | 90.07 +/- 0.08 | 形成预算 sweep |

这部分用于说明：模型不是只在原网络里加 soft mask，而是可以实际删除通道并形成较小网络。

### 1.2 fusion-compatible export

已加入 source gated model 与 exported compact model 的对照，并加入 target-matched uniform-width export baseline。当前主要结论是：

- exported compact model 的 full OA 为 86.90 +/- 1.65；
- uniform-width export 的 full OA 为 86.29 +/- 0.77；
- learned nonuniform export 在 main-only、downsample-4、occlusion-50 等状态下更有优势。

这部分用于说明：本文不是泛泛地做剪枝，而是在多模态分支和融合层之间处理结构兼容问题。

### 1.3 缺失模态和 availability mask

已完成 w/o fusion availability mask 消融。结果显示 disabling fusion mask 后，HSI-only OA 从主方法的 80.04 +/- 1.93 降到 76.72 +/- 2.97。该结果支持一个比较明确的说法：availability mask 最重要的作用是避免缺失模态仍参与 fusion softmax。

### 1.4 退化监督可靠性

当前主方法使用 light multi-degradation schedule，退化概率 p=0.25。Houston2013 上，在 80% MAC budget 下：

- compact full baseline 的 adverse-state average OA：74.89%；
- degradation-supervised compact model 的 adverse-state average OA：84.76%；
- full-modality OA 保持在 86.90%。

可靠性曲线已经改成 controlled-corruption curves，展示 OA、aux reliability score 和 aux fusion weight 随 noise、downsample、occlusion severity 的变化。稿件中已经保守表述：这些 reliability/fusion-weight 曲线是机制诊断，不宣称是完全校准的物理传感器质量分数。

### 1.5 多数据集证据

当前主文保留 Houston2013、Trento、MUUFL 三个数据集：

| Dataset | 当前作用 | Full OA | 解释 |
|---|---|---:|---|
| Houston2013 | 主实验与消融 | 86.90 +/- 1.65 | 官方 split，证据最完整 |
| Trento | 外部 HSI-LiDAR 验证 | 98.94 +/- 0.94 | 约 80% MAC 下保持接近饱和精度 |
| MUUFL | 困难 stress test | 86.87 +/- 1.79 | clean full OA 有下降，但 missing/degraded robustness 明显改善 |

MUUFL 的负面结果没有隐藏，当前稿件把它写成 trade-off evidence，而不是 SOTA claim。

## 2. 当前稿件的保守边界

为了避免主题发散，我已经把以下内容移出投稿主线，保留给毕业论文或后续扩展：

- dynamic budget-profile routing；
- patch-level router；
- weighted CE / class-balanced loss；
- MUUFL class-wise 深度诊断；
- demo system；
- 更大 backbone 或 foundation/expert routing 对比。

当前投稿稿件只围绕 compact export、fusion compatibility、availability-aware fusion、degradation-supervised reliability 展开。

## 3. 需要老师帮忙判断的问题

### 问题 1：题目是否接受当前版本？

当前题目是：

> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification

优点是准确，能直接点出 compact export 和 reliability；缺点是偏长。

可选短题目：

> Fusion-Compatible Compact Export for Reliable Multimodal Remote Sensing Classification

### 问题 2：MUUFL 是否留在主文？

保留 MUUFL 的优点是能体现多数据集和困难场景；缺点是 clean full OA 不是全面胜利，需要解释 trade-off。我的倾向是保留在主文，但只作为 stress test，不把它写成主贡献。

### 问题 3：当前投稿是否先停止加实验，进入语言润色？

我的判断是当前结构已经具备 advisor review 的基本完整性。继续临时加实验可能会让主线再次发散。更稳妥的是先润色全文逻辑和表达，再根据老师意见决定是否补一个更强 baseline 或更大 backbone。

## 4. 我接下来的执行建议

如果老师认可当前投稿主线，我下一步会做三件事：

1. 全文语言润色，重点压缩 Method 和 Experiments 里重复解释的句子。
2. 检查每个表格和图注是否自解释，尤其是 Figure 1 架构图和 Figure 2/3 结果图。
3. 准备最终投稿包，包括 PDF、LaTeX 源码、figures、tables、README、代码复现说明和匿名性检查。

如果老师认为创新性仍不足，我建议不要推翻当前稿件，而是优先补一个“larger-backbone transfer”或“stronger lightweight baseline”实验，因为这两类补充最容易增强说服力，又不破坏当前投稿主线。
