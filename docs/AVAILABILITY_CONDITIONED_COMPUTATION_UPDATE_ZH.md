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

## 对论文创新性的意义

这次改动可以支撑方法部分从“剪枝 + 可靠性融合”的简单组合，升级为：

**Fusion-Compatible Budgeted Structural Export + Availability-Conditioned Computation**

可在论文中强调三点：

1. 结构门控不是孤立地压缩每个分支，而是保持双分支输出维度兼容，使结构化导出后的 compact 模型仍可进行可靠性加权融合。
2. 缺失模态不只是被融合权重屏蔽，而是在推理路径中直接停止对应分支计算。
3. 资源消耗不再是单一固定值，而应报告 `full`、`main_only`、`aux_only` 等模态状态下的 state-dependent MACs / latency。

## 后续实验优先级

P0：补充状态相关资源统计

- 对原始模型和 compact 模型分别统计：
  - `full`
  - `main_only`
  - `aux_only`
- 指标包括：
  - Params
  - MACs
  - CUDA latency
  - OA / AA / Kappa
- 目标：证明 compact export 与可用性条件计算同时带来资源收益。

P1：更新消融实验表

建议增加以下消融：

| 版本 | 结构导出 | 缺失分支跳过 | 可靠性融合 | 目的 |
|---|---:|---:|---:|---|
| Full baseline | 否 | 否 | 否 | 原始双分支上限 |
| Masked fusion only | 否 | 否 | 是 | 证明仅融合屏蔽不足 |
| Compact export only | 是 | 否 | 是 | 证明静态轻量化收益 |
| Compact + availability-conditioned | 是 | 是 | 是 | 本文完整方法 |

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

## 当前限制

当前实现只在 batch 级别跳过整批不可用模态。若一个 batch 内不同样本的缺失状态不同，为保持张量并行和代码稳定，仍会计算两个分支，再由融合层按样本掩码屏蔽。实验报告状态相关资源时，应按单一缺失状态构造 batch，例如全部为 `main_only` 或全部为 `aux_only`。
