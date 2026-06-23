# BRM-Net 改进方向调研报告

> 日期：2026-06-23  
> 目的：结合当前代码问题和最新论文趋势，确定下一步最值得推进的改进方向。

## 1. 当前代码暴露出的研究问题

当前服务器代码已经说明一个事实：复杂缺失模态架构会很快失控。

已有 `missing*`、`Drfuse`、`Fmc`、`claude` 等分支都尝试过共享/特异特征分解、动态融合、类内紧凑、FMC 等机制，但这些方向有三个风险：

1. 模块越来越重，和轻量化目标冲突；
2. 论文叙事变成“堆模块”，创新边界不清；
3. 代码难维护，实验难复现。

因此改进方向应从“复杂恢复/分解”转向：

> 轻量条件计算、预算感知结构选择、模态可靠性估计、可解释部署评估。

## 2. 文献趋势

### 2.1 资源受限推理已经成为遥感 AI 的真实问题

Diana 等人在 2024 年 Remote Sensing 综述中讨论了星上神经网络推理所需的硬件加速和降低计算需求的软件技术，说明资源受限不是论文包装，而是遥感智能处理的真实约束。  
来源：[Review on Hardware Devices and Software Techniques Enabling Neural Network Inference Onboard Satellites](https://www.mdpi.com/2072-4292/16/21/3957)

对当前工作的启发：

- 论文不能只报 OA；
- 必须报告 Params、FLOPs、Latency、Model Size；
- 最好补 batch=1 CPU/ONNX 延时；
- 不能声称真实上星，只能写 resource-constrained / onboard-inspired。

### 2.2 高光谱压缩 benchmark 强调精度-效率权衡

2026 年高光谱分类压缩 benchmark 系统比较了 pruning、quantization、knowledge distillation，强调分类精度、内存消耗和推理效率三者的权衡。  
来源：[A Benchmark Study of Neural Network Compression Methods for Hyperspectral Image Classification](https://arxiv.org/abs/2603.04720)

对当前工作的启发：

- 当前预算门控结果必须做成 Pareto 曲线；
- 仅给 65% 单点不够；
- 50/65/80/90 多预算结果要成为主实验；
- 可以补一个量化或模型大小实验，但不要让它喧宾夺主。

### 2.3 缺失模态学习正在转向条件计算和专家选择

MaMOL 将遥感缺失模态分类重新表述为缺失模式下的条件专家选择问题，通过动态路由和轻量 LoRA 专家处理不同缺失模式。  
来源：[Rethinking Efficient Mixture-of-Experts for Remote Sensing Modality-Missing Classification](https://arxiv.org/abs/2511.11460)

对当前工作的启发：

- 不要为每个缺失组合训练独立网络；
- 可以把缺失/退化模态写成“条件计算/可靠性路由”；
- 当前硕士论文不适合做完整 MoE，但可以做轻量 MQE + RGF。

### 2.4 模态专家和专家剪枝是前沿趋势，但不宜照搬大模型

RingMoE 使用模态专家、协同专家、共享专家和动态专家剪枝，面向多模态遥感基础模型。  
来源：[RingMoE](https://arxiv.org/abs/2504.03166)

MAPEX 使用 modality-conditioned token routing 和 modality-aware expert pruning，仅保留与任务模态相关的专家。  
来源：[MAPEX](https://arxiv.org/abs/2507.07527)

对当前工作的启发：

- 可以借鉴“模态专家 + 共享融合 + 动态剪枝”的语言；
- 不能照搬大模型 MoE；
- 当前代码最现实的是把每个模态分支看成轻量专家，把预算门控看成专家/通道级选择，把 RGF 看成可靠性路由。

### 2.5 Missing modality survey 支持“模态不完整”作为独立问题

2024 年缺失模态多模态学习综述指出，多模态训练和推理中模态缺失可能来自传感器限制、成本、隐私、数据丢失等因素，会影响性能，需要专门的鲁棒学习方法。  
来源：[Deep Multimodal Learning with Missing Modality: A Survey](https://arxiv.org/abs/2409.07825)

对当前工作的启发：

- 第四章/投稿论文中可以正当化“模态可靠性”；
- 但方法应保持轻量，不走复杂生成式补全路线；
- 实验必须至少包含 complete/main-only/aux-only/noise。

### 2.6 OFA/Slimmable 思路说明多预算模型族有合理性

Once-for-All 通过一次训练支持多种结构设置，并为不同设备约束选择子网。  
来源：[Once-for-All](https://arxiv.org/abs/1908.09791)

Network Slimming 通过 BN 缩放因子 L1 稀疏化并裁剪通道，是结构化通道剪枝的经典工作。  
来源：[Network Slimming](https://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

对当前工作的启发：

- 你的预算门控不能只说“我做了剪枝”；
- 必须强调它在多模态分支和融合层中的预算分配；
- 后续可尝试“一次训练，多预算导出”，但这不是 P0。

## 3. 推荐改进方向排序

### 方向 A：预算门控 + 可靠性融合的最小统一框架

优先级：最高。

内容：

- 抽取 `BudgetGatedConv2d`；
- 建立双分支 `BudgetGatedEncoder`；
- 加 MQE；
- 加 RGF；
- 统一 loss：`L_cls + lambda_b L_budget + lambda_q L_quality`。

优点：

- 和现有代码最兼容；
- 能体现新意；
- 实验量可控；
- 适合硕士论文和 CCF-C 投稿。

风险：

- 需要补最小退化实验；
- 需要把旧日志整理成可信表格。

### 方向 B：多预算模型族

优先级：中高。

内容：

- 固定一个训练流程；
- 通过不同门控阈值或预算目标导出 50/65/80/90；
- 画 Pareto 曲线。

优点：

- 工作量明显；
- 贴近 OFA/Slimmable 方向；
- 和已有多预算结果相匹配。

风险：

- 如果重新训练多预算会耗时；
- 当前代码还没有干净 export pipeline。

### 方向 C：轻量模态专家/路由

优先级：中。

内容：

- 把 HSI 分支、LiDAR/SAR/MS 分支视作 modality experts；
- RGF 权重作为轻量 routing；
- 可视化不同退化条件下的权重。

优点：

- 可以借鉴 MaMOL/RingMoE/MAPEX 的前沿语言；
- 不需要真的实现大 MoE。

风险：

- 容易被质疑“只是加权融合”，必须靠退化曲线和权重可视化支撑。

### 方向 D：继续 DrFuse/FMC 复杂分解

优先级：低。

原因：

- 当前代码已证明这条路容易复杂化；
- 和轻量化冲突；
- 很难在短时间内做出稳定、漂亮、可解释的结果。

## 4. 下一步建议

最推荐的推进顺序：

1. 使用当前分支 `brmnet-core-extraction` 作为新基础；
2. 只保留 `brmnet_core/` 这条干净主线；
3. 从旧代码复用 dataloader 和 evaluator；
4. 新建统一训练入口，不再修改 `missing*`；
5. 先跑 Houston2013-HS-LiDAR：
   - full；
   - HSI only；
   - aux only；
   - aux noise；
   - budget 65；
6. 再扩展到 Berlin-HS-SAR 和 HS-MS；
7. 最后回填投稿工程。

## 5. 结论

当前最有价值的改进不是继续堆复杂架构，而是把已有实验中真正有效的部分抽象清楚：

> 预算门控解决“资源受限时保留哪些通道”，可靠性融合解决“模态不稳定时相信哪个模态”，多预算评估解决“部署时如何选精度-效率折中”。

这条线比 DrFuse/FMC 更轻，也更容易答辩和投稿。

