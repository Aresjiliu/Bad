# 低预算结构化剪枝稳定性补强记录

日期：2026-06-25

## 1. 本次推进目标

前一轮 65% / 80% / 90% 结构化剪枝已经证明了“Hard-Concrete 训练 -> 目标资源投影 -> compact 导出 -> 微调评估”的闭环，但 65% compact OA 相对 90% 下降 2.40 pp，略超预期。因此本次优先处理两个最实际的问题：

1. compact 微调不能固定使用最后一个 epoch，应保存验证集最优 checkpoint；
2. 自动阈值搜索不能只追求全局 MACs 命中，应支持每个 gate 的最小激活比例，避免低预算时关键层被过度裁剪。

## 2. 已实现改动

### 2.1 每层最小激活比例约束

新增参数：

```bash
--min-active-ratio 0.25
```

作用位置：

- `brmnet_core.resources.find_resource_budget_threshold(...)`
- `scripts/run_brmnet_houston.py::write_structured_pruning_artifacts(...)`

实现方式：

- 自动阈值搜索仍然使用全局概率断点；
- 但候选阈值必须先满足每个 Hard-Concrete gate 至少保留 `ceil(channels * min_active_ratio)` 个通道；
- 再从满足约束的候选阈值中选择 Params/MACs 误差最小者。

该做法不会破坏 source hard 模型和 compact 模型的等价性，因为导出器仍使用同一个全局阈值和同一组 hard mask。

### 2.2 Compact 最优 checkpoint

新增参数：

```bash
--compact-val-fraction 0.1
--compact-selection-metric oa
```

实现方式：

- 主模型训练仍使用完整训练集；
- compact 微调前，从训练集按固定 seed 切出验证集；
- 每个 compact epoch 后在验证集评估；
- 默认按 `oa` 选择最佳 epoch；
- 最终 `compact_model.pt` 保存最佳 epoch 的权重，并写入 `best_epoch`、`selection_metric`、`selection_score` 和验证指标。

可选选择指标：

- `oa`：越大越好；
- `accuracy`：越大越好；
- `loss`：越小越好，内部转为负分数比较。

## 3. 推荐下一轮短实验

先只复跑 65% 和 80%，不要立刻扩展三随机种子。

```bash
D:\software\anaconda3\envs\hslinets\python.exe scripts\run_brmnet_houston.py ^
  --data-root D:\Academic\HSLiNets-main\Dataset ^
  --data-format raw ^
  --pair-modalities hsi+lidar ^
  --split-protocol official ^
  --epochs 20 ^
  --target-budget 0.65 ^
  --budget-metric macs ^
  --gate-type hard_concrete ^
  --gate-mode deterministic ^
  --min-active-ratio 0.25 ^
  --compact-val-fraction 0.1 ^
  --compact-selection-metric oa ^
  --compact-finetune-epochs 10 ^
  --device cuda ^
  --seed 0 ^
  --output-dir output\structured_pruning_stability
```

然后将 `--target-budget 0.65` 改为 `0.8` 再跑一次。

## 4. 判断标准

短实验只看四件事：

1. 实际 compact MACs 是否仍接近目标预算；
2. 65% compact OA 是否相比上一轮 `84.73%` 有提升；
3. 65% 相对 90% 的 OA 差距能否压到 2 pp 左右；
4. `compact_history.json` 中最佳 epoch 是否早于最后一轮，若是，说明 checkpoint 选择确实有价值。

如果 65% 仍然明显低于 90%，下一步再做逐层敏感度分配，而不是盲目提高模型复杂度。

