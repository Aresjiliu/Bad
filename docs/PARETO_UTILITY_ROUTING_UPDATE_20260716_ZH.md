# Pareto / Utility 路由策略推进记录

日期：2026-07-16

## 1. 本轮实现内容

本轮完成了质量条件预算路由的 P0 策略修正能力：

1. 新增 `select_pareto_tolerance_budget_profiles`。
   - 对每个模态/退化状态，先找该状态下 OA 最优的 profile。
   - 在 `OA >= best_OA - delta` 的候选中选择 MACs 最低的 profile。
   - 目标是替代过于保守的 hand oracle。

2. 新增 `build_sweep_reports`。
   - 支持 Pareto tolerance sweep。
   - 支持 utility penalty sweep。
   - 自动补充 `mean_metric_regret` 和 `mean_resource_saving_vs_reference`。

3. 扩展 `scripts/evaluate_budget_profile_routing.py`。
   - 新增 `--output-sweep-json`。
   - 新增 `--output-sweep-csv`。
   - 新增 `--pareto-tolerances`。
   - 新增 `--utility-resource-penalties`。

4. 增加单元测试。
   - 覆盖 Pareto 标签选择。
   - 覆盖 sweep 报告生成。
   - 覆盖 CLI sweep 输出。

## 2. 生成文件

新增或更新的结果文件：

- `docs/generated/brmnet_routing_profile_formal_sweep_seed0.csv`
- `docs/generated/brmnet_routing_profile_formal_sweep_seed0.json`
- `docs/generated/brmnet_routing_profile_formal_sweep_seed1.csv`
- `docs/generated/brmnet_routing_profile_formal_sweep_seed1.json`
- `docs/generated/brmnet_routing_profile_formal_sweep_seed0_seed1_summary.csv`

## 3. 2-seed 策略扫描结果

| Policy | 2-seed mean OA | 2-seed mean MACs | Mean regret | Mean saving vs 100% |
|---|---:|---:|---:|---:|
| pareto_delta_0 / utility_lambda_0 | 0.7434 | 0.9041 | 0.0000 | 0.0959 |
| pareto_delta_0.005 | 0.7433 | 0.8952 | 0.0001 | 0.1048 |
| pareto_delta_0.01 | 0.7424 | 0.8550 | 0.0010 | 0.1450 |
| utility_lambda_0.05 | 0.7419 | 0.8394 | 0.0015 | 0.1606 |
| pareto_delta_0.02 | 0.7398 | 0.8126 | 0.0036 | 0.1874 |
| utility_lambda_0.10 | 0.7383 | 0.7970 | 0.0051 | 0.2030 |
| utility_lambda_0.20 | 0.7283 | 0.7235 | 0.0151 | 0.2765 |

## 4. 当前判断

Pareto / utility sweep 证明了一个重要事实：当前动态路由已经可以系统控制精度-算力折中，但还没有在 2-seed mean OA 上超过旧的 static 80% 强基线。

更具体地说：

- `pareto_delta_0.01` 几乎不损失 OA，平均 regret 约 0.0010，同时相对 100% profile 节省约 0.145 MACs。
- `utility_lambda_0.05` 是当前更激进但仍相对稳的折中点，2-seed mean OA 为 0.7419，mean MACs 为 0.8394。
- `lambda=0.2` 过于偏向省算力，虽然 MACs 低，但 OA 损失明显。
- 当前策略扫描更适合作为“标签设计与策略选择依据”，还不能直接作为最终优于 static 80% 的主结果。

## 5. 对论文叙事的影响

论文主线应调整为：

1. 固定 profile 建立强基线。
2. 手工 oracle 暴露出启发式规则过保守的问题。
3. Pareto / utility 标签把路由目标改为数据驱动的精度-资源折中。
4. learned router 不再学习 hand oracle，而应学习 Pareto/utility 标签。

这样当前“不够理想”的实验结果可以转化为合理科研过程：先发现手写规则失败，再提出更合理的约束标签。

## 6. 下一步优先级

### P0-1：补 seed2

完成 seed2 的 65/80/100 profile 正式实验，然后重新生成：

- static profile 3-seed summary
- Pareto sweep 3-seed summary
- utility sweep 3-seed summary

### P0-2：确定标签方案

优先候选：

- `pareto_delta_0.005`
- `pareto_delta_0.01`
- `utility_lambda_0.05`

这三个策略在 2-seed 上保持较低 regret，同时有明确算力节省。

### P1：训练 learned Pareto router

不要再学习 hand oracle。下一版 learned router 应学习 Pareto/utility 标签，并增加：

- leave-one-state-out evaluation
- routing accuracy vs Pareto label
- selected policy 后的真实 OA/MAC
- profile selection heatmap

### P2：论文图表

建议新增三张图：

1. Pareto / utility sweep 曲线：x=mean MACs，y=mean OA。
2. 每个状态的 profile 选择热图。
3. delta/lambda 与 mean regret、mean MAC saving 的双轴图。
