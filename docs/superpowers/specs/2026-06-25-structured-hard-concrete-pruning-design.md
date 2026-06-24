# BRM-Net 结构化 Hard-Concrete 剪枝设计

## 目标

在不破坏现有 Houston 数据协议和实验引擎的前提下，将当前“BN 前连续缩放”升级为：

```text
Conv -> BatchNorm -> HardConcreteGate -> ReLU
```

使 65%、80%、90% 预算能够对应不同的硬通道结构，并将硬掩码模型导出为不含门控层的紧凑模型。

本规格只覆盖 P0 结构化轻量化闭环。模态 availability mask、modality dropout 和 MQE 质量监督属于后续 P1。

## 方案比较

### 方案 A：显式门控块和 BRM-Net 专用导出器

- 新建独立 `HardConcreteGate`。
- 编码器和分类头使用显式 Conv/BN/Gate/ReLU 块。
- 模型级共享融合接口门控。
- 根据明确的层依赖手工导出紧凑 BRM-Net。

优点：依赖最少，数据流明确，导出前后可逐层验证。  
缺点：导出器首先只支持当前 BRM-Net。

### 方案 B：继续扩展 `BudgetGatedConv2d`

优点：改动文件少。  
缺点：门控仍与卷积耦合，难以解决 BN 吸收和共享融合接口约束。

### 方案 C：直接引入 DepGraph/Torch-Pruning

优点：后续可支持复杂 ResNet。  
缺点：当前模型规模小，引入外部依赖会扩大环境和调试风险。

采用方案 A。P2 迁移到 ResNet 时，再评估用 DepGraph 替代专用导出器。

## 模型结构

### HardConcreteGate

每个输出通道维护一个可学习 `log_alpha`。接口：

```python
gate.expected_active_probability()  # 可微的通道激活概率
gate.soft_gate()                    # 确定性连续门控
gate.hard_mask(threshold=0.5)       # bool 硬掩码
gate.set_inference_mode("soft" | "hard")
```

训练时使用 Hard-Concrete 重参数采样；评估默认使用确定性 soft gate；导出验证使用 hard mask。

参数采用 Louizos et al. 的常用设置：

- temperature/beta：`2/3`
- stretch lower/gamma：`-0.1`
- stretch upper/zeta：`1.1`

初始值按非零概率公式反解：

```text
log_alpha =
    logit(initial_retention)
    + beta * log(-gamma / zeta)
```

使 `expected_active_probability()` 的初始均值等于目标保留率。目标 1.0 时数值截断，避免无穷值。`hard_mask()` 统一对 `expected_active_probability()` 使用阈值，训练统计、硬资源统计和导出不得使用不同口径。

### 编码器门控

两个模态分支的前两层使用独立门控：

```text
main_gate_1, main_gate_2
aux_gate_1, aux_gate_2
```

两个编码器末层输出不各自持有门控，而由模型级 `shared_fusion_gate` 同时作用于 main/aux 特征。这样保证：

- 两个特征张量保留完全相同的通道索引；
- 加权相加融合维度一致；
- 分类头第一层输入与共享通道索引一致。

分类头包含两个独立门控：

```text
head_gate_1, head_gate_2
```

总计 7 组结构门控。

## 资源预算

### 预期资源

预算损失不再使用门控概率均值，而按层计算预期资源：

```text
expected_conv_params =
    kernel_h * kernel_w * expected_input_channels * expected_output_channels

expected_conv_macs =
    output_h * output_w * expected_conv_params
```

首层输入通道固定；后续层输入通道由上一门控的期望激活数决定。BN、质量 MLP 和分类 Linear 的参数也计入参数量；MACs 主要统计 Conv/Linear。
卷积存在 bias 时将期望输出通道数计入参数量；当前 BRM-Net 卷积默认无 bias。

损失：

```text
resource_loss = (expected_resource_ratio - target_budget)^2
```

Runner 通过 `--budget-metric {params,macs}` 选择预算口径，默认 `macs`。输入 patch 尺寸来自现有 `--patch-size`。

### 硬资源

报告同时输出：

- expected params/MACs ratio；
- hard params/MACs ratio；
- baseline params/MACs；
- compact params/MACs；
- 每个门控组的 active/total channels。

论文中仅 compact model 的实际统计可称为压缩结果。

## 紧凑模型导出

新建 `CompactBRMNet`，构造参数为：

```python
main_width=(m1, m2, shared)
aux_width=(a1, a2, shared)
head_width=(h1, h2)
```

导出流程：

1. 从每组 gate 获取硬掩码。
2. 每组至少保留一个最高分通道，防止零宽网络。
3. 按索引复制 Conv 输出维、下一层 Conv 输入维、BN 参数和统计量。
4. 两个编码器末层使用同一 shared mask。
5. 复制两个 MQE 第一层的 shared 输入列。
6. 复制分类头 Linear 的 head2 输入列。
7. 返回无门控参数的 `CompactBRMNet` 和结构清单。

在 hard inference 模式下，原模型与 compact model 对同一输入的 logits 最大绝对误差必须小于 `1e-5`。

## 兼容策略

- 保留 `BudgetGatedConv2d` 和原统计 API，用于加载和复现历史 checkpoint。
- 新结构通过显式 `gate_type="hard_concrete"` 启用。
- Runner 默认切换到 Hard-Concrete；通过 `--gate-type legacy_sigmoid` 可运行旧结构。
- 旧 checkpoint 不自动加载到新结构，结构不匹配时给出明确错误。

## 输出文件

每个正式运行新增：

```text
resource_stats.json
compact_model.pt
compact_config.json
```

`resource_stats.json` 保存预期、硬掩码和紧凑模型资源，不把嵌套逐层统计塞入现有 CSV。

## 测试

至少覆盖：

1. Hard-Concrete 概率、采样范围、硬掩码和梯度。
2. 门控位于 BN 之后。
3. main/aux 末层共享同一个融合 gate。
4. 预期 Params/MACs 对门控参数可微。
5. 资源比例随关闭通道单调下降。
6. 导出器正确裁剪 Conv/BN/Linear。
7. hard masked model 与 compact model logits 一致。
8. compact model 不包含 `HardConcreteGate` 或旧门控参数。
9. Runner 参数、命名和资源结果落盘。

## 首轮实验

先运行 Houston official、seed 0：

- 预算：0.65、0.80、0.90；
- 20 epochs；
- metric：MACs；
- deterministic evaluation；
- 导出 compact model。

进入三 seed 的条件：

- 三档 hard/compact MACs 单调；
- 目标误差不超过 5%；
- 65% OA 相对 90% 下降不超过 2 pp；
- 导出一致性测试通过。

若 65% OA 下降超过 2 pp，则保留 80%/90% 为主结果，将 65% 标记为压力测试。
