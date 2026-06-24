# BRMNet 目标通道预算控制设计

## 目标

将现有“持续压低门控概率”的无界预算正则，改为可解释、可复现实验的目标通道保留率控制。首轮实验使用 65%、80%、90% 三档预算。

## 预算定义

对模型中全部 `BudgetGatedConv2d` 的输出通道进行全局统计。第 `l` 层有 `C_l` 个输出通道，门控概率为 `p_lc`：

```text
soft_retention = sum_l sum_c p_lc / sum_l C_l
```

该定义按每层输出通道数自然加权，不对各层做等权平均。

目标预算损失采用对称平方误差：

```text
budget_loss = (soft_retention - target_budget)^2
```

其中 `0 < target_budget <= 1`。没有预算门控层时应显式报错，避免产生虚假的预算指标。

## 统计指标

每次训练和验证至少记录：

- `target_budget`：配置的目标保留率。
- `soft_retention`：全模型可微的软保留率，用于训练和判断预算收敛。
- `hard_retention`：以门控概率 0.5 为阈值的硬保留率。
- `active_channels` / `total_channels`：硬门控激活通道数与总通道数。
- `layer_retention`：逐层软保留率、硬保留率及通道数。

硬保留率仅用于诊断。当前实现尚未执行结构化裁剪，因此不能据此宣称真实参数量、FLOPs 或推理延迟下降。

## 训练接口

训练脚本新增：

```text
--target-budget {0.65,0.80,0.90,1.00}
```

默认值为 `1.00`，作为完整预算基线。由于新预算损失已经归一化到 `[0, 1]`，`lambda_budget` 默认值调整为 `1.0`，实验中仍需通过收敛结果校准。

门控概率默认按目标预算初始化，即 `gate_score = logit(target_budget)`。这是为了消除统一 0.70 初始保留率对 20-epoch 预算比较造成的系统偏差；可通过 `--gate-init-retention` 显式覆盖。目标为 1.0 时数值截断为 0.9999。

运行目录和结果文件名必须包含预算标签，例如：

```text
..._gatedeterministic_budget65
```

汇总脚本必须按预算分组，防止不同目标预算的结果被混合统计。

## 首轮验证

固定：

- Houston 2013 HSI-LiDAR
- official split
- split seed 42
- train seed 0
- deterministic gate
- 20 epochs

分别运行目标预算 0.65、0.80、0.90。先判断软保留率误差和分类精度，再决定是否扩展到 3 seeds。若硬保留率始终为 1.0，应将其视为门控未形成离散结构的证据，而不是轻量化成功。
