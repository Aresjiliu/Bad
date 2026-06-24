# Target Budget Retention Implementation Plan

**Goal:** 将 BRMNet 门控正则改为 65%/80%/90% 可控的全局通道保留率，并完成 Houston official split 的单种子 CUDA 验证。

**Architecture:** 在 `budget_gates.py` 集中计算全局及逐层统计；损失函数只消费统一统计结果；Runner 负责参数、命名和结果落盘；汇总脚本按目标预算分组。

**Tech Stack:** Python, PyTorch, pytest, Houston 2013 HSI-LiDAR runner.

### Task 1: 门控统计与目标损失

**Files:**
- Modify: `brmnet_core/budget_gates.py`
- Test: `tests/test_budget_gates.py`

1. 编写多层、不同输出通道数的加权保留率失败测试。
2. 编写硬激活通道、逐层统计、非法预算和无门控层测试。
3. 实现全局统计对象与目标预算平方损失。
4. 运行预算门控单元测试。

### Task 2: 损失与训练指标

**Files:**
- Modify: `brmnet_core/losses.py`
- Modify: `brmnet_core/engine.py`
- Test: `tests/test_losses.py`
- Test: `tests/test_engine.py`

1. 编写目标预算损失及训练指标失败测试。
2. 将 `target_budget` 接入 `brmnet_loss`。
3. 在 epoch 指标中累计软、硬保留率和目标预算。
4. 运行损失与引擎测试。

### Task 3: Runner 与汇总

**Files:**
- Modify: `scripts/run_brmnet_houston.py`
- Modify: `scripts/summarize_brmnet_experiments.py`
- Modify: runner/summary related tests

1. 编写 CLI 默认值、预算标签和分组字段失败测试。
2. 接入 `--target-budget`，更新运行命名、配置和最终门控统计。
3. 门控概率默认按目标预算初始化，并记录有效初始值。
4. 汇总时按目标预算分组并输出预算误差。
5. 运行全量测试。

### Task 4: CUDA 实验与记录

**Files:**
- Modify: `docs/GPU_EXPERIMENT_REPORT_ZH.md`
- Modify: `docs/generated/brmnet_gpu_summary.csv`
- Modify: `docs/generated/brmnet_gpu_summary.md`

1. 使用 `hslinets` 环境运行 0.65、0.80、0.90，official split，seed 0，20 epochs。
2. 检查目标误差、分类指标和硬保留率；必要时仅校准 `lambda_budget`。
3. 生成汇总并记录不能宣称真实加速的边界。
4. 运行最终测试，提交代码并尝试推送。
