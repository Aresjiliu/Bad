# 可用性条件计算改进记录

日期：2026-07-10

## 背景

参考 `BRM-Net_全面专业指导意见_创新性调研更新版 (1).md` 后，当前最需要补强的不是继续堆叠轻量化模块，而是把“面向模态缺失的轻量化”从普通剪枝问题推进为一个结构约束问题。原始实现虽然支持 `availability_mask`，但该掩码只在可靠性融合阶段生效，两个模态分支的编码器仍然始终执行。因此在论文叙述上只能说明“融合时屏蔽缺失模态”，无法支撑“缺失状态下计算量随可用模态自适应降低”的核心论点。

本次改进将该点落到代码中：当一个 batch 内某一模态整批不可用时，模型会直接跳过该模态编码器与质量估计器，并用零特征、零可靠性分数进入后续融合流程。

## 已实现内容

1. 新增 `brmnet_core/availability.py`
   - 提供 `encode_available_modalities`。
   - 判断 `availability_mask` 的模态列是否全为 0。
   - 对整批不可用的模态跳过 encoder 和 quality estimator。
   - 保持输出字典接口不变，仍返回 `main_feature`、`aux_feature`、`q_main`、`q_aux`、`fusion_weights` 和 `logits`。

2. 接入原始 BRM-Net
   - `brmnet_core/model.py` 的 `BRMNet.forward` 已改为通过可用性条件编码 helper 生成双分支特征。

3. 接入 compact export 模型
   - `brmnet_core/compact.py` 的 `CompactBRMNet.forward` 同样支持缺失模态跳过计算。
   - 这使 compact 模型不仅是参数裁剪后的静态小模型，还具备按模态状态改变执行路径的推理特性。

4. 新增回归测试
   - `tests/test_structured_brmnet.py` 覆盖原始结构模型在 `main_only` 和 `aux_only` 状态下跳过不可用分支。
   - `tests/test_compact_export.py` 覆盖 compact 模型在 `main_only` 状态下跳过辅助分支。

5. 新增状态相关资源统计
   - `brmnet_core/resources.py` 的 `estimate_brmnet_resources` 和 `estimate_compact_resources` 支持 `modality_state` 参数。
   - 可选状态包括 `full`、`main_only`、`aux_only`。
   - `scripts/run_brmnet_houston.py` 的 `resource_stats.json` 新增 `state_dependent` 字段，分别记录 `hard` 与 `compact` 在不同模态状态下的有效 Params/MACs。

6. 新增状态相关推理延迟统计
   - `brmnet_core/profiling.py` 提供 `profile_modality_state_latency`。
   - Runner 新增 `--latency-warmup` 和 `--latency-iterations` 参数。
   - `resource_stats.json` 新增 `latency_ms.hard` 与 `latency_ms.compact`，分别记录 `full`、`main_only`、`aux_only` 的平均推理耗时。
   - `scripts/summarize_brmnet_experiments.py` 会把 compact latency 汇总到 CSV/Markdown，便于直接形成论文资源效率表。

7. 新增 `w/o reliability` 消融接口
   - `ReliabilityGatedFusion` 新增 `mode` 参数，可选 `reliability` 和 `uniform`。
   - `uniform` 模式不使用质量分数决定融合权重，而是在可用模态之间均匀加权，并继续尊重 `availability_mask`。
   - Runner 新增 `--fusion-mode reliability|uniform`。
   - compact export 会保留 source model 的 fusion mode。
   - 实验矩阵新增 `without_reliability_uniform_fusion` 变体。
   - 汇总脚本将 `fusion_mode` 纳入分组与 Markdown 表格，避免 reliability 与 uniform 结果混合。

## 对论文创新性的意义

这次改动可以支撑方法部分从“剪枝 + 可靠性融合”的简单组合，升级为：

**Fusion-Compatible Budgeted Structural Export + Availability-Conditioned Computation**

可在论文中强调三点：

1. 结构门控不是孤立地压缩每个分支，而是保持双分支输出维度兼容，使结构化导出后的 compact 模型仍可进行可靠性加权融合。
2. 缺失模态不只是被融合权重屏蔽，而是在推理路径中直接停止对应分支计算。
3. 资源消耗不再是单一固定值，而应报告 `full`、`main_only`、`aux_only` 等模态状态下的 state-dependent MACs / latency。

## 后续实验优先级

P0：补充状态相关资源与延迟统计

- 已在资源估计层支持原始 hard 结构模型和 compact 模型的状态相关 Params/MACs。
- 已在 runner 中支持状态相关 latency profiling。
- 后续需要在真实实验 run 上重新生成 `resource_stats.json`：
  - `full`
  - `main_only`
  - `aux_only`
- 指标包括：
  - Params：已支持
  - MACs：已支持
  - CUDA latency：已支持
  - OA / AA / Kappa
- 目标：证明 compact export 与可用性条件计算同时带来资源收益。

推荐运行时保留默认 profiling 设置：

```powershell
conda run -n hslinets python scripts/run_brmnet_houston.py `
  --data-format legacy `
  --data-root D:\Academic\HSLiNets-main\Huston2013 `
  --split-protocol official `
  --target-budget 0.8 `
  --budget-metric macs `
  --epochs 20 `
  --compact-finetune-epochs 10 `
  --modality-dropout-prob 0.25 `
  --lambda-quality 0.2 `
  --aux-noise-std 0.1 `
  --seed 0 `
  --device cuda `
  --latency-warmup 5 `
  --latency-iterations 20 `
  --output-dir output/experiments/p0_reliability
```

P1：更新消融实验表

建议增加以下消融：

| 版本 | 结构导出 | 缺失分支跳过 | 可靠性融合 | 目的 |
|---|---:|---:|---:|---|
| Full baseline | 否 | 否 | 否 | 原始双分支上限 |
| Masked fusion only | 否 | 否 | 是 | 证明仅融合屏蔽不足 |
| Compact export only | 是 | 否 | 是 | 证明静态轻量化收益 |
| Compact + availability-conditioned | 是 | 是 | 是 | 本文完整方法 |
| w/o reliability / uniform fusion | 是 | 是 | 否 | 证明质量分数是否提供额外收益 |

P2：更新论文方法章节

建议将方法章节调整为：

1. Problem Formulation
2. Overall Framework
3. Fusion-Compatible Budgeted Structural Gating
4. Modality-Private and Fusion-Shared Gate Design
5. Availability-Conditioned Reliability Routing
6. Compact Export and State-Dependent Inference
7. Training Objectives

本次代码改动主要对应第 5、6 节。

## 当前已具备的优先实验

可直接生成包含 uniform 消融的核心实验矩阵：

```powershell
conda run -n hslinets python scripts/make_brmnet_experiment_matrix.py `
  --data-root D:\Academic\HSLiNets-main\Dataset `
  --python D:\software\anaconda3\envs\hslinets\python.exe `
  --ablation core `
  --seeds 0 1 2
```

其中 `without_reliability_uniform_fusion` 对应论文主消融表中的 `w/o reliability` 或 `average fusion` 行。

## 当前限制

当前实现只在 batch 级别跳过整批不可用模态。若一个 batch 内不同样本的缺失状态不同，为保持张量并行和代码稳定，仍会计算两个分支，再由融合层按样本掩码屏蔽。实验报告状态相关资源时，应按单一缺失状态构造 batch，例如全部为 `main_only` 或全部为 `aux_only`。
