# 当前实验诊断与下一步改进计划

日期：2026-07-16

## 1. 当前实验结论

当前质量条件预算路由已经完成了 65/80/100 三个可部署 profile 的 seed0 和 seed1 正式实验。结果表明，代码链路已经成立，但当前路由策略还不能作为最终论文主结果。

| 策略 | 2-seed mean OA | 2-seed mean MACs | 结论 |
|---|---:|---:|---|
| static 65% | 0.7315 | 0.6124 | 最省算力，但精度不足。 |
| static 80% | 0.7503 | 0.7554 | 当前最强固定 profile 基线。 |
| static 100% | 0.7374 | 0.9420 | 算力最高但不稳定，不应默认作为最强模型。 |
| hand oracle / learned oracle | 0.7391 | 0.8915 | 手工规则过于保守，倾向选择 100%。 |
| utility label | 0.7425 | 0.6441 | 已证明节省算力潜力，但惩罚系数需要系统调参。 |
| learned utility | 0.7346 | 0.6166 | 11 个状态级样本太少，学习路由泛化证据不足。 |

最关键判断：当前问题不是“再训练一个更大的模型”，而是“路由标签和评价目标尚未定义好”。如果 static 80% 同时比 100% 和 hand oracle 更稳，说明固定 profile 本身已经形成较强 baseline，论文必须证明动态策略在精度、算力、鲁棒性三目标上有可解释收益。

## 2. 文献调研后的定位

### 2.1 动态网络与条件计算

Dynamic Neural Networks 综述将动态网络分为 instance-wise、spatial-wise、temporal-wise 等类别，核心是根据输入自适应调整结构或参数，以获得精度和计算效率优势：
https://arxiv.org/abs/2102.04906

2024 年 conditional computation 综述进一步强调，条件计算的典型实现包括 MoE、token selection、early exit，以及按输入动态激活或关闭计算图的一部分：
https://arxiv.org/abs/2403.07965

对本项目的启发：论文不能只说“我有动态路由”，因为这是成熟概念。更合理的表述是：面向遥感多模态缺失/退化场景，使用低成本质量探针选择有限个可部署预算 profile，并用 AER 指标评价精度-效率-鲁棒性。

### 2.2 Slimmable / Once-for-All 与多预算部署

Slimmable Neural Networks 证明单个网络可以在不同宽度下运行，并通过 switchable BN 支持运行时精度-效率折中：
https://arxiv.org/abs/1812.08928

Once-for-All 提出一次训练、按设备/约束选择子网，强调在多硬件和多资源约束下快速得到专用子网：
https://arxiv.org/abs/1908.09791

对本项目的启发：当前 65/80/100 profile 是论文可以站住的工程基础，但不必立刻实现完整 OFA。短期更适合做“有限 profile bank + 质量条件选择”，中期再考虑共享权重 slimmable supernet。

### 2.3 缺失模态遥感与动态融合

Incomplete Multimodal Learning for Remote Sensing Data Fusion 指出传统多模态遥感方法通常假设训练和测试阶段模态完整，模态不完整会导致明显退化，并通过随机模态组合、重建、对比损失提升不完整输入下的表现：
https://arxiv.org/abs/2304.11381

Dynamic Cross-Modal Feature Interaction Network for HSI and LiDAR Data Classification 已经在 HSI-LiDAR 分类中使用动态路由与跨模态交互块，强调数据依赖的计算路径：
https://arxiv.org/html/2503.06945v1

对本项目的风险：如果论文只宣称“HSI-LiDAR 动态路由”，创新性会被最新 DCMNet 类工作压住。更稳的差异点应是：动态路径不是为追求更复杂的融合，而是为了在模态缺失/退化下进行资源受限 profile 选择，并显式报告 MACs/latency/预算命中。

### 2.4 多目标与 Pareto 选择

Multi-Objective NAS by Learning Search Space Partitions 强调部署模型不能只看精度，还需要同时考虑模型大小、延迟、FLOPs 等指标，并围绕 Pareto frontier 提高多目标搜索效率：
https://arxiv.org/abs/2406.00291

对本项目的直接启发：下一步不应继续依赖手工 oracle，而应把标签定义为 Pareto/约束标签。例如：在每个退化状态下，先找该状态的最佳 OA，再选择 OA 损失不超过阈值的最小 MAC profile。

## 3. 下一步技术方向

### 3.1 主方向：Pareto-tolerance budget routing

目标：替换当前过于保守的 hand oracle。

定义每个状态 `s` 下三种 profile 的结果为 `(OA_s,b, MAC_s,b)`。令 `OA_best(s)` 为该状态的最高 OA。给定容忍阈值 `delta`：

```text
candidate(s, delta) = {b | OA_s,b >= OA_best(s) - delta}
label(s, delta) = argmin_b MAC_s,b, b in candidate(s, delta)
```

建议扫：

```text
delta = [0.000, 0.005, 0.010, 0.020, 0.030]
```

汇报指标：

- mean OA
- mean MACs ratio
- OA regret vs per-state best
- MAC saving vs static 100%
- profile selection distribution
- 每个模态/退化状态的选择热图

预期贡献：把“动态路由”从启发式规则变成可解释的精度-资源约束决策。

### 3.2 次方向：utility penalty sweep

当前 `lambda=0.2` 的 utility label 已经把 MACs 从 static 80% 的 0.7554 降到 0.6441，但 OA 降到 0.7425。因此需要系统扫惩罚系数：

```text
lambda = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
utility = OA - lambda * MACs
```

该实验用于回答：动态预算是否存在稳定优于固定 65/80/100 的折中点。

### 3.3 路由器泛化验证

当前 learned router 主要是状态级 in-sample 拟合，训练样本只有 11 个状态，不足以支撑泛化结论。

优先实现两个层级：

1. Leave-one-state-out：每次留出一个模态/退化状态，训练 10 个状态，测试 1 个状态，报告平均路由准确率和最终 OA/MAC。
2. Patch-level router：从验证 batch 采样质量探针输出，使用状态级 Pareto label 或 batch-level label 扩充训练样本，使路由器从“状态表拟合”变成“输入质量特征驱动”。

### 3.4 质量探针校准

当前论文要强调 quality-aware，就必须证明质量探针不是装饰模块。

建议补充：

- `pre_q_main/pre_q_aux/pre_u_main/pre_u_aux` 与退化强度的曲线。
- 质量分数与 OA drop 的相关性。
- 质量分数与 profile label 的对应关系。
- 如果相关性弱，增加 pairwise/ranking loss：同一样本 clean 的质量应高于 noise/downsample/occlusion 后的质量。

## 4. 实施计划

### P0：先闭合证据链

1. 完成 seed2 的 65/80/100 正式 profile 实验。
2. 生成 3-seed static 65/80/100、hand oracle、utility、learned utility 总表。
3. 实现 Pareto-tolerance label，并输出 delta sweep 表。
4. 实现 utility lambda sweep，并输出 Pareto 曲线。
5. 更新论文结果表：不再把 hand oracle 作为核心主张，而是作为失败的启发式对照。

完成标准：找到一个动态策略在 3-seed 上满足以下至少一条：

- OA 接近或超过 static 80%，MACs 明显更低。
- OA 高于 static 65%，MACs 接近 static 65%。
- 在缺失/退化状态子集上明显优于固定 profile。

### P1：把方法从“策略表”推进到“学习路由”

1. 实现 leave-one-state-out router evaluation。
2. 实现 patch-level quality feature collection。
3. 使用 Pareto label 训练小型 MLP / logistic router。
4. 报告路由准确率、真实 OA/MAC、混淆矩阵。
5. 绘制质量特征到预算选择的可解释图。

完成标准：学习路由不能只报告 accuracy，必须报告其选择后实际得到的 OA/MAC。

### P2：论文与图表增强

1. 画 AER Pareto 曲线：x=MACs，y=OA，颜色/形状表示鲁棒性或策略。
2. 画 per-state profile selection heatmap。
3. 画 quality score vs degradation severity 曲线。
4. 画 seed mean±std 主结果表。
5. 方法章节按以下结构重写：
   - 多预算可部署 profile 构建
   - 低成本模态质量探针
   - Pareto 约束预算标签
   - 学习式质量条件预算路由
   - AER 评价协议

## 5. 论文表述建议

当前最稳的论文题眼应从：

```text
面向模态缺失的轻量化多模态融合网络
```

调整为：

```text
面向模态缺失与退化的质量感知预算自适应遥感多模态分类方法
```

核心创新点建议写成三条：

1. 构建面向 HSI-LiDAR 缺失/退化输入的可部署多预算 profile bank，在同一实验协议下报告精度、MACs 和鲁棒性。
2. 提出低成本质量探针驱动的预算选择机制，将模态可用性、质量和不确定性转化为 profile 选择依据。
3. 设计 Pareto/约束效用标签，使动态路由目标从手工规则转为可解释的精度-资源折中，并通过多 seed 和多退化状态验证。

## 6. 立即执行顺序

最实在的下一步不是继续改大模型，而是：

1. 先跑完 seed2，避免当前结论停留在 2-seed。
2. 马上实现 Pareto-tolerance label sweep，因为它直接解决 hand oracle 失败的问题。
3. 同时实现 utility lambda sweep，因为它能快速找到可汇报的精度-算力折中点。
4. 找到最优策略后，再做 learned router；否则学习器只是在学习错误标签。
5. 最后把论文实验主线改为“static profile baseline -> hand oracle failure -> Pareto label correction -> learned quality router”。

这个叙事能体现工作量，也能把当前不理想结果转化为合理的科研推进过程。
