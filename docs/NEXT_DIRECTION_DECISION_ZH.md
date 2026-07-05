# 下一步方向分析与调研决策

日期：2026-07-05  
分支：`brmnet-core-extraction`  
最新代码提交：`dd08705 Add class-balanced validation checkpointing`

## 1. 直接结论

当前不建议重新开始，也不建议继续在小 BRM-Net 上只堆轻量化实验。

下一步应采用“双线收敛”：

1. **短线保底**：继续用当前 `brmnet_core` 小模型完成多 seed、退化评估、真实 compact 导出和论文机制消融，保证有可复现实验闭环。
2. **主线升级**：把当前已经验证的 Hard-Concrete 预算控制、compact export、MQE/RGF 思想迁移到旧代码/服务器代码中的 ResNet 类双分支主干，用更大模型体现硕士论文工作量和真实压缩收益。

论文主线建议固定为：

> **预算-可靠性联合感知的多模态遥感轻量化分类框架。**

不要再写成“一个剪枝方法”。更稳的表述是：

> 在资源预算受限和模态质量不稳定条件下，通过预算感知结构选择与模态可靠性门控融合，实现多模态遥感分类的精度-效率-鲁棒性折中。

## 2. 当前代码与实验说明了什么

### 2.1 已经成立的部分

当前 `brmnet_core` 已经形成完整的机制闭环：

- Houston2013 official/raw 数据协议；
- Hard-Concrete 通道门控；
- 面向 Params/MACs 的真实资源预算；
- 自动阈值投影；
- compact model 导出；
- source/compact validation best checkpoint；
- full / main-only / aux-only / aux-noise 退化矩阵评估。

最新 seed0 结果：

| 目标 MACs | Best source epoch | Source OA | Compact OA | 实际 MACs | 实际 Params |
|---:|---:|---:|---:|---:|---:|
| 65% | 9 | 87.22% | 87.60% | 65.09% | 64.66% |
| 80% | 9 | 86.99% | 86.90% | 80.12% | 79.53% |
| 90% | 9 | 86.67% | 86.23% | 90.08% | 90.74% |

这说明：

- 资源预算命中是可信的；
- compact 导出是可信的；
- 结构宽度随预算增加是单调的；
- 单 seed OA 非单调，不能直接当最终论文结论。

### 2.2 仍然不足的部分

当前问题不在“代码完全不能用”，而在证据层次不足：

1. 小 BRM-Net 只有约 0.467M 参数，继续裁剪的绝对收益有限。
2. MQE/RGF 已在代码中存在，但缺少训练阶段的 availability mask、modality dropout 和缺失模态 masked fusion，因此还不能证明真正的缺失模态鲁棒性。
3. validation OA 很快达到 100%，说明训练集切出的 10% validation 太小，只能用于避免 test-set selection，不能作为强泛化证据。
4. 论文工作量不能只靠 65/80/90 三档预算表支撑，需要迁移到更大主干并补实验矩阵。

## 3. 外部调研对方向的约束

### 3.1 模态缺失方向是成立的，但不应做重型补全

`Deep Multimodal Learning with Missing Modality: A Survey` 指出，多模态训练/测试中模态缺失可能来自传感器限制、成本、隐私和数据丢失，并会损害性能；这支持论文把“模态质量不稳定”作为第四章问题来源。  
来源：https://arxiv.org/abs/2409.07825

但当前时间和代码条件下，不建议做生成式补全、DrFuse/FMC 全量分解或复杂重建。那些路线会显著增加参数和训练复杂度，和轻量化主线冲突。

### 3.2 前沿趋势是模态专家和条件路由，但应做轻量化版本

RingMoE 使用模态专家、协同专家、共享专家和动态专家剪枝，把多模态遥感基础模型从 14.7B 参数压缩到 1B 级别。  
来源：https://arxiv.org/abs/2504.03166

MAPEX 使用 modality-conditioned token routing，并通过 modality-aware expert pruning 只保留任务相关专家。  
来源：https://arxiv.org/abs/2507.07527

MaMOL 将遥感缺失模态分类从条件计算角度重构，用轻量 LoRA/MoE 思想处理多种缺失组合。  
来源：https://arxiv.org/abs/2511.11460

这些论文说明“模态可靠性/可用性驱动的动态路由”是合理叙事。但本项目不应照搬大 MoE，而应将：

- HSI 分支；
- LiDAR/SAR/MS 分支；
- 共享融合层；
- MQE/RGF 权重；
- Hard-Concrete 通道门控；

解释为一个轻量级的“模态专家化 + 预算剪枝”版本。

### 3.3 压缩实验必须报告真实效率指标

2026 年高光谱分类压缩 benchmark 明确从分类精度、内存消耗和推理效率评价 pruning、quantization、knowledge distillation。  
来源：https://arxiv.org/abs/2603.04720

星上推理综述也说明神经网络上星/边缘推理需要考虑硬件和软件层面的资源约束。  
来源：https://www.mdpi.com/2072-4292/16/21/3957

因此论文主表不能只写 OA。必须至少包含：

- OA / AA / Kappa；
- Params；
- MACs 或 FLOPs；
- model size；
- batch=1 latency；
- 最好补 GPU 与 CPU 各一组。

## 4. 候选路线评估

### 方案 A：继续小 BRM-Net 多 seed

优点：

- 最快；
- 风险最低；
- 可复现性最好；
- 当前代码已经能跑。

缺点：

- 工作量不足；
- 主干太小，压缩收益不显著；
- 不能充分支撑硕士论文“设计与重构”的厚度。

结论：必须做，但只能作为保底和消融，不应作为最终唯一主线。

### 方案 B：迁移到 ResNet 类双分支大主干

优点：

- 能体现真实压缩收益；
- 能复用旧 `models/resnet_ensemble.py` 和服务器代码资产；
- 能把当前机制升级成论文主结果；
- 可解释为“方法从原型到主干的迁移验证”。

缺点：

- exporter 更复杂；
- 需要处理 ResNet block 内残差依赖；
- 实验时间更长。

结论：这是当前最值得投入的主线。

### 方案 C：实现完整缺失模态复杂网络

优点：

- 看起来工作量大；
- 可写成独立第四章。

缺点：

- 与轻量化冲突；
- 易变成堆模块；
- 代码已有历史分支证明维护风险高；
- 时间紧时很难稳定。

结论：不推荐。只保留轻量 availability mask、modality dropout、masked RGF。

### 方案 D：只做系统/PPT/论文包装

优点：

- 快；
- 有助于汇报。

缺点：

- 不能解决导师最可能追问的“工作量和实验主结果”。

结论：可并行做，但不能替代主实验。

## 5. 最终路线

推荐采用：

> **A 保底 + B 主攻 + C 的轻量子集 + D 收口。**

具体展开：

1. 当前小 BRM-Net 保留为“机制验证模型”。
2. 旧/服务器 ResNet 类双分支主干作为“论文主模型”。
3. 缺失模态只做轻量 masked RGF + modality dropout，不做重型生成式补偿。
4. 实验围绕 accuracy-efficiency-robustness 三维展开。

## 6. 下一步三阶段计划

### 阶段 1：两天内完成保底统计

目标：让当前小模型结果可以进入汇报，并回答“65% 为什么比 90% 高”。

任务：

1. 跑 seed 1、seed 2 的 65/80/90。
2. 汇总 mean ± std。
3. 生成 OA-MACs、OA-Params 表。
4. 汇报口径改成“目标预算可控，精度在轻量区间内稳定”，不声称预算越高精度越高。

验收：

- 65/80/90 每档至少 3 seeds；
- 真实 MACs 命中误差 < 1 pp；
- compact 结果可自动汇总。

### 阶段 2：实现可靠性训练闭环

目标：把 MQE/RGF 从“有输出”变成“有训练约束、有缺失模态效果”。

最小实现：

1. batch 增加 availability mask；
2. 训练时 modality dropout，禁止双模态同时缺失；
3. RGF 改成 masked softmax，缺失模态权重严格为 0；
4. quality target 由退化强度生成；
5. 输出不同模态状态下的 fusion weights。

验收：

- HSI-only / aux-only 至少有一项明显提升；
- 缺失模态平均权重 < 0.05；
- 完整模态 OA 下降不超过 1 pp；
- 能画“噪声强度-模态权重”曲线。

### 阶段 3：迁移到 ResNet 类双分支主干

目标：形成论文级工作量和主结果。

建议先做非残差复杂剪枝版本，降低 exporter 风险：

1. 从 `models/resnet_ensemble.py` 抽取双分支主干结构和输入适配；
2. 先用固定宽度/统一缩宽跑 baseline；
3. 再接入 Hard-Concrete gate；
4. 对残差 block 内部使用 block-level 或 stage-level gate，避免一开始做任意通道依赖图；
5. compact 导出优先支持 stage-level width，而不是每个卷积任意裁剪。

验收：

- 参数量明显大于当前 0.467M；
- 65%/80%/90% 有真实 Params/MACs 减少；
- 至少一个预算点相对原始模型 OA 下降 ≤ 1 pp；
- Ours 优于同 MACs 的统一缩宽 baseline。

## 7. 汇报时应怎么讲

建议对导师这样讲：

> 当前我没有继续堆复杂缺失模态架构，而是先把预算门控做成了真实可导出的 compact 结构，并补齐了验证集 best checkpoint。现有小模型已证明资源预算控制、结构导出和退化评估闭环可行，但它本身太小，不适合作为最终主模型。下一步我准备把这套机制迁移到 ResNet 类双分支主干，同时只保留轻量模态可靠性门控，不做重型生成式补全。这样论文可以从单一剪枝方法升级为预算-可靠性联合感知框架，实验上也能体现多 seed、多预算、模态缺失和部署效率四类工作量。

## 8. 立即执行清单

按优先级：

1. 跑 `output/structured_pruning_validation` 的 seed 1、seed 2。
2. 给 `scripts/summarize_brmnet_experiments.py` 增加或确认 validation 新目录汇总。
3. 实现 masked RGF 和 modality dropout 的测试用例。
4. 抽取 ResNet 主干迁移设计文档。
5. 只在以上四项完成后，再考虑 ONNX/INT8 或系统界面。

## 9. 停止继续发散的边界

以下事情暂时不做：

- 不继续扩展 `missing4/Drfuse/FMC`；
- 不做生成式模态补全；
- 不做大规模 MoE；
- 不在小 BRM-Net 上追求单 seed 最高 OA；
- 不在没有大主干结果前过度打磨系统界面；
- 不把随机划分高精度当主结果。

## 10. 对投稿时间的现实判断

截至 2026-07-05：

- PRICAI 2026 官方页面显示截稿为 2026-06-27，已经错过；
- ACCV 2026 截稿为 2026-07-05，当前不适合仓促投；
- ICTAI 2026 EasyChair CFP 显示截稿为 2026-07-21，仍有窗口，但只有在两周内补齐可靠性闭环和主表时才值得冲；
- IJCNN 2027 regular paper 截稿为 2027-01-31，是中期稳妥备选。

因此短期目标应从“马上投稿”调整为：

> 先用 7-10 天补出导师能认可的实验主线和论文结构，再决定是否压缩成 ICTAI 工具/应用型版本。
