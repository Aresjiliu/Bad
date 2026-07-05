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

## 5. Seed 0 短实验结果

已在 Houston2013 official split、train seed 0、20 epoch 主模型训练、10 epoch compact 微调下复跑 65%、80% 和 90%。

| 目标 MACs | 实际 MACs | 实际 Params | Source OA | Compact OA | AA | Kappa | 最佳 compact epoch | 结构宽度 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 65% | 64.89% | 66.32% | 86.00% | 85.89% | 88.59% | 84.77% | 1 | main 24/45/101, aux 21/43/101, head 111/64 |
| 80% | 80.02% | 80.57% | 85.89% | 87.28% | 89.37% | 86.23% | 3 | main 29/57/112, aux 23/47/112, head 121/64 |
| 90% | 90.24% | 92.20% | 84.99% | 86.96% | 89.13% | 85.90% | 5 | main 29/61/124, aux 27/55/124, head 127/64 |

相对上一轮结果：

- 65% compact OA 从 `84.73%` 提升到 `85.89%`，提升 `+1.16 pp`；
- 80% compact OA 从 `84.90%` 提升到 `87.28%`，提升 `+2.38 pp`；
- 65% 与 80% 的差距为 `1.39 pp`，说明最佳 checkpoint 与最小通道约束对低预算稳定性有实际收益；
- 65% 与上一轮 90% compact OA `87.13%` 的差距约 `1.24 pp`，已经进入原先希望的 2 pp 范围。
- 90% 同配置 compact OA 为 `86.96%`，低于 80% 的 `87.28%`，但 source OA 本身也只有 `84.99%`，说明这一轮 90% 的主要问题是单 seed 训练波动，而不是资源投影或 compact 导出失效。

当前判断：

1. 可以将“低预算稳定性补强”写入汇报材料，作为一次明确的实验推进；
2. 资源命中、Params/MACs 和结构宽度已经单调，可以支撑“可控预算导出”的工程结论；
3. OA 非严格单调，不能把当前 seed-0 三档直接作为最终主实验，需要做训练稳定性补强；
4. 下一步优先实现 class-balanced compact validation split，并加入 source best checkpoint；随后复跑 80%/90% 或 seed 1 验证波动来源。

## 6. Seed 1 复核

为判断 90% 结果是否只是 seed 0 偶然波动，补跑 Houston2013 official split、train seed 1、90% 预算。

| 目标 MACs | Train seed | 实际 MACs | 实际 Params | Source OA | Compact OA | AA | Kappa | 最佳 compact epoch | 结构宽度 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 90% | 1 | 89.94% | 92.26% | 85.16% | 86.40% | 88.50% | 85.31% | 1 | main 29/60/126, aux 24/59/126, head 123/64 |

复核结论：

- 90% seed 1 的 compact OA 为 `86.40%`，仍低于 seed 0 的 80% compact OA `87.28%`；
- 90% 两个 seed 的 source OA 分别为 `84.99%` 和 `85.16%`，说明问题主要出现在源模型训练阶段，而不是 compact 导出阶段；
- compact 微调的验证集 OA 很容易达到 1.0，说明当前随机 10% train validation 太小、太容易饱和，不适合继续作为唯一模型选择依据；
- 下一步不应继续堆更多预算档，而应先补两个稳定性机制：主模型 best checkpoint、class-balanced validation split。

推荐下一步实验顺序：

1. 实现主模型训练期间的 validation split 与 best checkpoint；
2. 将 compact validation split 改为 class-balanced，避免 283 个样本的小验证集过易饱和；
3. 用 seed 0 复跑 80% 和 90%，检查源模型 OA 是否恢复合理顺序；
4. 若 80%/90% 稳定后，再启动 65/80/90 三随机种子主实验。

## 7. Source best checkpoint 与 class-balanced validation 更新

日期：2026-07-05

本轮按上一节建议补齐了两个稳定性机制：

1. 主模型训练阶段新增 `--val-fraction`、`--val-split-strategy`、`--selection-metric`，默认从训练集按类别均衡切出 10% validation，并保存 validation OA 最优的 source checkpoint。
2. Compact finetune 阶段新增 `--compact-val-split-strategy`，默认同样使用 class-balanced validation split，避免随机小验证集漏类。
3. `checkpoint.pt` 现在记录 `best_epoch`、`selection_metric`、`selection_score` 和 validation 指标，后续汇报可以明确说明没有用 test set 选择模型。
4. 本轮实验使用 Houston2013 official split、train seed 0、20 epoch source training、10 epoch compact finetune、`--min-active-ratio 0.1`。

复跑结果如下：

| 目标 MACs | Best source epoch | Source val OA | Source OA | Compact OA | Compact AA | Compact Kappa | 实际 MACs | 实际 Params | 结构宽度 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 65% | 9 | 100.00% | 87.22% | 87.60% | 89.40% | 86.59% | 65.09% | 64.66% | main 26/50/105, aux 22/41/105, head 96/64 |
| 80% | 9 | 100.00% | 86.99% | 86.90% | 88.73% | 85.82% | 80.12% | 79.53% | main 29/60/114, aux 27/48/114, head 111/64 |
| 90% | 9 | 100.00% | 86.67% | 86.23% | 88.41% | 85.12% | 90.08% | 90.74% | main 29/61/125, aux 31/58/125, head 118/64 |

关键判断：

1. 资源控制仍然成立：65/80/90 三档实际 MACs 分别为 65.09%、80.12%、90.08%，结构宽度也随预算单调增加。
2. validation OA 在第 9 轮达到 100%，说明 10% train validation 仍偏小，适合作为“避免 test-set selection”的规范化机制，但不适合作为强泛化判断。
3. 单 seed 下 OA 仍非单调：65% compact OA 反而最高，说明当前 BRM-Net 小主干受训练随机性、类别局部混淆和 budget 正则共同影响，不能把单 seed 的预算排序作为论文最终结论。
4. 与上一轮随机 compact validation 相比，新流程在方法论上更规范，但 80%/90% compact OA 略低；这不是失败，而是说明“验证集选择机制”不是当前精度瓶颈，下一步应转向多 seed 统计和更有工作量的主干迁移。

下一步建议：

1. 保留当前 BRM-Net 作为机制验证模型，论文中用它证明 hard-concrete 预算控制、紧凑导出、模态退化评估闭环。
2. 立即跑 65/80/90 的 seed 1、seed 2，得到均值和标准差，避免导师追问“为什么 65% 比 90% 高”时没有统计依据。
3. 若多 seed 仍显示 65% 不低于 90%，论文表述应改为“目标预算可控且精度在轻量化区间内保持稳定”，不要声称预算越高精度越高。
4. 下一阶段把该机制迁移到服务器上的较大双分支主干，当前小 BRM-Net 继续承担可复现实验和消融表，服务器主干承担硕士论文工作量与主结果。
