# BRM-Net 当前实验、理想目标与下一步计划

日期：2026-06-25

## 1. 当前结论

当前工作已经完成了“可信实验基础设施”和“软预算控制原型”，但尚未完成论文题目所要求的两个核心证明：

1. 预算变化能够产生真实可部署的紧凑结构；
2. 模态缺失或严重退化时，可靠性模块能够明显降低性能损失。

因此，当前状态不是“推翻重做”，而是：

> 保留数据协议、训练引擎、评价指标和结果管理；重做门控离散化、结构导出和缺失模态训练机制。

## 2. 已完成的有效成果

### 2.1 实验协议

- Houston 2013 official/random 两套划分已经统一。
- 两套协议均为 2,832 个训练像素、12,197 个测试像素。
- 已记录坐标指纹、归一化统计、配置、训练历史和混淆矩阵。
- 已验证随机像素划分会显著高估结果：确定性门控三 seed 下，random OA 为 `0.9740 ± 0.0058`，official OA 为 `0.8601 ± 0.0104`。

该部分已经达到论文级可复现要求，后续不应重写。

### 2.2 分类与预算实验

官方划分 20 epochs、单 seed 的软预算结果：

| 目标软预算 | Full OA | AA | Kappa | 软保留率 | 硬保留率 |
|---:|---:|---:|---:|---:|---:|
| 0.65 | 0.8668 | 0.8868 | 0.8560 | 0.650005 | 1.0000 |
| 0.80 | 0.8837 | 0.9034 | 0.8743 | 0.800001 | 1.0000 |
| 0.90 | 0.8849 | 0.9023 | 0.8755 | 0.900001 | 1.0000 |

可得出：

- 目标软保留率能够被精确维持；
- 80% 与 90% 的分类性能接近；
- 65% 比 90% 低约 1.81 个 OA 百分点；
- 所有预算下 640 个门控通道仍全部激活，没有形成结构差异。

### 2.3 模态退化实验

| 预算 | Full OA | HSI-only OA | LiDAR-only OA | LiDAR noise OA |
|---:|---:|---:|---:|---:|
| 0.65 | 0.8668 | 0.4331 | 0.2373 | 0.8592 |
| 0.80 | 0.8837 | 0.4026 | 0.2216 | 0.8757 |
| 0.90 | 0.8849 | 0.4507 | 0.1867 | 0.8801 |

模型对轻度 LiDAR 噪声较稳定，但对完整模态置零非常脆弱。当前 MQE/RGF 还不能称为有效的缺失模态方法。

## 3. 与理想目标的比较

现有论文材料中的 Houston-HS-LiDAR 参考目标为：

- Original OA：89.41%；
- 65% 模式 OA：89.16%；
- Params：15.06M -> 10.03M，减少约 33.4%；
- FLOPs：4.11G -> 2.71G，减少约 34.1%；
- Latency：8.06ms -> 5.74ms，减少约 28.8%；
- HSI-only OA：84.81%；
- LiDAR-only OA：60.75%。

这些历史结果的网络和协议尚未完全核对，因此只能作为论文目标参考，不能与当前最小 BRM-Net 直接合并。

| 维度 | 当前状态 | 理想目标 | 判断 |
|---|---|---|---|
| Official full OA | 单 seed 最好 88.49%；三 seed 基线 86.01% | 稳定接近 89%，三 seed 方差可控 | 接近但未稳定达到 |
| 65% OA | 86.68% | 约 89.16%，相对完整模型下降不超过 0.5-1.0 pp | 差距明显 |
| 预算语义 | 输出通道概率均值 | 实际 Params/FLOPs/Latency 预算 | 尚未达到 |
| 硬通道差异 | 65/80/90 均为 640/640 | 不同预算产生单调紧凑结构 | 完全未达到 |
| 真实压缩 | Params/FLOPs 不变 | 65% 档约降低 30%-35% | 完全未达到 |
| HSI-only | 40%-48% | 历史参考约 84.81% | 严重不足 |
| LiDAR-only | 18%-24% | 历史参考约 60.75% | 严重不足 |
| 多预算稳定性 | 单 seed | 3 seeds，均值和标准差 | 未完成 |
| 多数据集 | 仅 Houston HSI-LiDAR 新流程 | HS-MS、HS-LiDAR、HS-SAR | 未完成 |
| 部署证据 | 无紧凑模型实测 | Params/FLOPs/模型大小/延迟/显存 | 未完成 |

总体完成度判断：

- 实验基础设施：约 85%；
- 分类基线：约 65%；
- 真实轻量化：约 25%；
- 缺失模态鲁棒性：约 20%；
- 论文完整证据链：约 45%。

## 4. 当前技术方案的根因问题

### 4.1 软门控没有形成结构学习

当前门控位于：

```text
BudgetGatedConv2d -> BatchNorm -> ReLU
```

门控只缩放卷积输出，而随后 BN 会重新标准化通道尺度。只要门控概率不为零，连续幅值差异很容易被 BN 和后续参数吸收。因此：

- 目标初始化可以精确维持概率均值；
- 但各通道不会自然分裂为“保留”和“删除”两组；
- 65/80/90 更像不同幅值初始化，不是真实子网搜索。

### 4.2 当前预算不是计算资源预算

输出通道概率均值没有考虑：

- 前后层输入/输出通道的乘积；
- 卷积核大小；
- 特征图空间尺寸；
- 两个模态分支与融合接口的结构依赖。

因此 65% 软通道概率不等于 65% Params 或 65% FLOPs。

### 4.3 融合接口存在结构耦合

两个编码器输出通过加权相加融合，末层通道必须对齐。独立裁剪两个分支会造成形状不一致。

需要将以下通道作为一个依赖组处理：

```text
main encoder output
aux encoder output
fusion input/output
classifier first-layer input
```

这可以转化为论文中的“多模态融合接口共享结构约束”，比普通单分支剪枝更有专门性。

### 4.4 可靠性融合无法真正屏蔽缺失模态

当前质量分数经过 Sigmoid，被限制在 `[0,1]`，再以温度 1 做 softmax。即使两个分数分别为 1 和 0，权重也约为：

```text
0.731 / 0.269
```

缺失模态仍会获得约 27% 权重。零输入经过带 BN 的编码器后也不保证得到零特征，所以仅把输入置零不足以表达“该模态不可用”。

## 5. 调研后的技术取舍

### 5.1 离散门控

L0/Hard-Concrete 方法能够通过可微随机门控学习精确零值，适合替换当前永不归零的 Sigmoid 门控。它比直接使用固定阈值更有理论依据。

参考：

- Louizos et al., ICLR 2018, Learning Sparse Neural Networks through L0 Regularization  
  https://openreview.net/forum?id=H1Y8hhg0b

### 5.2 结构化导出

DepGraph 显式建模层间结构依赖，用于一致地删除前后层相关通道。当前 BRM-Net 结构较小，可先手工实现导出并用 DepGraph/Torch-Pruning 作为校验或后续扩展工具。

参考：

- Fang et al., CVPR 2023, DepGraph: Towards Any Structural Pruning  
  https://openaccess.thecvf.com/content/CVPR2023/html/Fang_DepGraph_Towards_Any_Structural_Pruning_CVPR_2023_paper.html

### 5.3 真实资源约束

MorphNet 和 DMCP 均说明，结构学习应直接面向 FLOPs 等资源约束，而不是只约束通道数量。下一版预算损失应升级为预期 FLOPs/Params 比例。

参考：

- Gordon et al., CVPR 2018, MorphNet  
  https://openaccess.thecvf.com/content_cvpr_2018/html/Gordon_MorphNet_Fast__CVPR_2018_paper.html
- Guo et al., CVPR 2020, DMCP  
  https://openaccess.thecvf.com/content_CVPR_2020/html/Guo_DMCP_Differentiable_Markov_Channel_Pruning_for_Neural_Networks_CVPR_2020_paper.html

### 5.4 多预算网络

OFA 通过渐进式收缩训练一次支持多种子网，但当前项目还没有可靠的单预算结构导出。现在直接做 OFA 会扩大风险，应放在三个独立预算均能导出之后。

参考：

- Cai et al., ICLR 2020, Once-for-All  
  https://openreview.net/forum?id=HylxE1HKwS

### 5.5 缺失模态路线

2024 年 missing-modality survey 将该问题视为独立于普通多模态学习的设置。DPMamba 和 MaMOL 强调用统一模型处理多种缺失组合，但其 Mamba、蒸馏或 MoE 路线对当前项目过重。

当前最实在的简化路线是：

- 显式 modality availability mask；
- 训练时 modality dropout；
- 噪声、遮挡、降采样退化采样；
- 轻量质量监督；
- masked softmax 让缺失模态权重严格为零。

参考：

- Wu et al., 2024, Deep Multimodal Learning with Missing Modality: A Survey  
  https://arxiv.org/abs/2409.07825
- Yang et al., IJCAI 2025, DPMamba  
  https://www.ijcai.org/proceedings/2025/248
- Gao et al., 2025, MaMOL  
  https://arxiv.org/abs/2511.11460

## 6. 清晰的下一步计划

### P0：真实结构化轻量化闭环

预计：2-4 天。

1. 将门控移到 BN 之后，改为 `Conv -> BN -> HardConcreteGate -> ReLU`。
2. 为内部层使用独立门控，为两个模态的最终接口使用共享通道门控。
3. 将预算损失从通道概率均值升级为预期 Params/FLOPs 比例。
4. 实现 compact exporter，实际删除 Conv/BN/Linear 的相关通道。
5. 验证导出前硬掩码模型与导出后紧凑模型输出一致。
6. 输出 Params、FLOPs、模型大小、batch=1 GPU 延迟和峰值显存。

验收标准：

- 65/80/90 实际 FLOPs 比例误差不超过 3%-5%；
- Params 和 FLOPs 随预算单调变化；
- 导出前后 logits 最大误差小于 `1e-5`；
- 65% 档 OA 相对 90% 档下降不超过 1.5 pp；
- 至少一个预算产生可测的延迟下降。

### P1：缺失模态最低可用闭环

预计：2-3 天。

1. Batch 增加 `availability_mask` 和退化强度标签。
2. RGF 改为 masked softmax，缺失模态权重严格为零。
3. 训练时随机执行 main/aux modality dropout，禁止两模态同时缺失。
4. 增加轻、中、重噪声以及 25%/50% 遮挡。
5. 用退化强度代理监督 MQE，并记录质量分数和融合权重。

验收标准：

- 完整模态 OA 相对不使用 dropout 的模型下降不超过 1 pp；
- HSI-only OA 至少比当前提高 20 pp，第一阶段目标达到 70%；
- 缺失模态平均融合权重小于 0.05；
- 随噪声强度增加，对应模态权重整体单调下降。

### P2：迁移到有论文工作量的主干网络

预计：3-5 天。

当前 BRM-Net 只有约 0.467M 参数，适合验证机制，但结构过小，继续裁剪的绝对收益和论文工作量有限。

1. 保留当前模型作为轻量原型与消融网络。
2. 从服务器代码复用 ResNet-18 双分支或 `HSI_Lidar_Couple_Prune` 主干。
3. 通过 adapter 接入新门控、共享融合接口和 exporter，不重写历史训练代码。
4. 在同一 official split 下对比：
   - 原始主干；
   - 统一缩宽；
   - 旧剪枝方法；
   - 新预算-可靠性方法。

验收标准：

- 65% 档真实 Params/FLOPs 减少约 25%-35%；
- OA 下降不超过 1 pp；
- 新方法优于相同 FLOPs 的统一缩宽基线；
- 方法在小模型和主干模型上均可工作。

### P3：论文主实验

预计：3-6 天。

1. Houston official 65/80/90 × seeds 0/1/2。
2. 补 50% 档作为极低资源压力测试。
3. 扩展 Houston HS-MS 和 Berlin HS-SAR。
4. 生成：
   - OA-Params；
   - OA-FLOPs；
   - OA-Latency；
   - 预算-缺失模态二维表；
   - 分支/融合层保留率图；
   - 可靠性权重随退化强度曲线。

### P4：论文与系统收口

1. 当前软预算结果改名为“连续门控诊断实验”，不作为压缩主结果。
2. 只有 compact model 实测结果才能写入 Params/FLOPs/Latency 主表。
3. 更新 LaTeX 表格、实验章节和贡献点。
4. 原型系统只实现三项核心操作：
   - 预算模式切换；
   - 模态状态切换；
   - 精度-效率-可靠性结果展示。

## 7. 停止条件

- 若 P0 的 65% 真实结构在两轮超参数配置后 OA 下降仍超过 2 pp，先保留 80%/90% 主结果，65% 作为压力测试。
- 若 modality dropout 两轮配置后 HSI-only 提升不足 10 pp，则将论文表述降级为“退化模态鲁棒融合”，不宣称解决任意缺失模态。
- 在三个独立预算尚未稳定导出前，不做 OFA。
- 在 compact model 尚未生成前，不做 INT8/ONNX 优化。
- Houston official 主实验尚未闭环前，不扩展第三个数据集。

## 8. 最终判断

当前不应该重新开始，也不应该继续在现有软门控上追加更多实验。

最优取舍是：

> 用当前干净分支保留已经完成的数据和实验工程，将门控升级为可归零、可导出、按真实资源计费的结构学习；同时用显式模态掩码和 modality dropout 将 MQE/RGF 从展示模块变成有效的鲁棒融合模块。当前小网络用于验证机制，服务器 ResNet 类主干用于体现论文级工作量和真实压缩收益。
