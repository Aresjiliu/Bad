# 论文与实验推进更新（2026-07-06）

## 1. 本轮已完成的实证推进

当前已经把 Houston2013-HS-LiDAR official split 的结构化剪枝 prototype 从单 seed 扩展到 3 seed：

- 数据协议：`D:\Academic\HSLiNets-main\Dataset`，raw format，official split。
- 验证策略：从训练集内采样 class-balanced validation，用 validation OA 选择 source 与 compact checkpoint。
- 训练设置：source 20 epochs，compact fine-tuning 10 epochs。
- 随机种子：0、1、2。
- 输出目录：`output/structured_pruning_validation`。

## 2. 3 seed 结果汇总

| Target MACs | Source OA | Compact OA | Compact AA | Compact Kappa | Params ratio | MACs ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 65% | 85.98 +/- 1.87 | 85.53 +/- 2.20 | 88.11 +/- 1.42 | 84.38 +/- 2.35 | 64.79 +/- 1.93 | 64.96 +/- 0.12 |
| 80% | 85.79 +/- 1.12 | 85.84 +/- 0.92 | 88.32 +/- 0.37 | 84.69 +/- 0.99 | 79.92 +/- 0.37 | 80.01 +/- 0.10 |
| 90% | 85.33 +/- 1.20 | 85.74 +/- 0.88 | 88.18 +/- 0.51 | 84.59 +/- 0.92 | 89.78 +/- 0.84 | 90.07 +/- 0.08 |

完整 run 级记录已保存到：

- `docs/generated/structured_pruning_multiseed_summary.md`
- `docs/generated/structured_pruning_multiseed_runs.csv`

## 3. 对论文主线的判断

这组结果适合支撑“机制闭环已经成立”，不适合作为最终主结果。

可以稳妥写入论文的结论：

- hard-concrete 结构化门控、阈值投影、compact exporter 能形成可部署闭环；
- compact 模型的实际 MACs 比例能稳定命中 65%、80%、90% 三档预算；
- class-balanced validation 能避免直接使用最后 epoch，实验选择更规范；
- prototype 已经具备多 seed 统计，不再只是偶然单次结果。

需要避免的结论：

- 不能声称预算越高精度越高，因为当前短训练 prototype 不呈现单调关系；
- 不能把该小模型当成硕士论文主工作量的最终模型；
- 不能把当前结果写成 SOTA，只能写成机制验证和消融基础。

## 4. 论文当前最合理的叙事方式

建议论文主线拆成两层：

1. 机制层：当前 `brmnet_core` 证明预算可控、结构可导出、结果可复现。
2. 主模型层：后续把机制迁移到更大双分支 backbone，并加入缺失模态训练闭环，形成硕士论文的主要工作量。

这样写的优势是：

- 当前已有结果不会被浪费；
- 老服务器代码和更大 backbone 可以作为论文主体继续承接；
- “轻量化 + 缺失模态鲁棒 + 动态可靠性融合”三条线能合成一个完整贡献，而不是单点小改。

## 5. 下一步优先级

P0：补齐缺失模态训练闭环。

- 在 RGF 中引入 availability mask，使缺失模态不参与 softmax 权重竞争。
- 在训练阶段加入 modality dropout，让模型见过 main-only、aux-only、degraded-aux 等场景。
- 输出完整、main-only、aux-only、noise 四种协议下的 OA/AA/Kappa。

P1：迁移到更大双分支 backbone。

- 从服务器版本或旧模型中抽取 ResNet/Couple-CNN 类双分支主干。
- 保留当前 hard-concrete gate、MQE/RGF、compact export 的可复用接口。
- 让论文主表使用较大模型，prototype 表作为机制消融或可复现实验。

P2：补论文图表。

- OA-Params、OA-MACs、OA-Latency 曲线。
- channel retention ratio 可视化。
- modality reliability weights under degradation。
- 分类图和 confusion matrix。

## 6. 本轮已同步到 Overleaf 草稿

已更新：

- `D:\Academic\paper_submission\brmnet_pricai2026\tables\prototype_hslidar_structured.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\sections\04_experiments.tex`
- `D:\Academic\paper_submission\brmnet_pricai2026\notes\experiments.md`

当前写法已经把 prototype 结果从单 seed 改为 3 seed mean +/- std，并明确这是 mechanism verification，而不是最终主表。

## 7. 代码推进状态

已完成缺失模态训练闭环的第一步实现：

- `ReliabilityGatedFusion` 支持 `availability_mask`，缺失模态不再参与 softmax 权重竞争；
- `BRMNet` 和 `CompactBRMNet` 的 forward 支持传入 `availability_mask`；
- `apply_degradation` 在 `main_only`、`aux_only`、`aux_noise` 评估模式下生成可用性 mask；
- 新增 `apply_modality_dropout`，训练时可按样本随机置零一个模态；
- `train_one_epoch` 新增 `modality_dropout_prob` 参数；
- `scripts/run_brmnet_houston.py` 新增 `--modality-dropout-prob` 命令行参数，source 与 compact fine-tuning 均可启用。

建议下一组实验：

```powershell
python scripts/run_brmnet_houston.py `
  --data-root D:\Academic\HSLiNets-main\Dataset `
  --data-format raw --split-protocol official `
  --target-budget 0.8 --min-active-ratio 0.1 `
  --epochs 20 --compact-finetune-epochs 10 `
  --val-fraction 0.1 --val-split-strategy class_balanced --selection-metric oa `
  --compact-val-fraction 0.1 --compact-val-split-strategy class_balanced --compact-selection-metric oa `
  --modality-dropout-prob 0.25 `
  --batch-size 64 --device cuda --seed 0 `
  --output-dir output\modality_dropout_validation
```

这组实验应优先观察 `main_only`、`aux_only` 和 `aux_noise` 的提升，而不是只看 full OA。

## 8. 初步 dropout 对照结果

已完成 80% budget、seed 0、`--modality-dropout-prob 0.25` 的初步实验。完整记录见：

- `docs/generated/modality_dropout_seed0_budget80.md`

核心现象：

| Model | Training | Full OA | Main-only OA | Aux-only OA | Aux-noise OA |
|---|---|---:|---:|---:|---:|
| Source | no dropout | 86.99 | 54.42 | 23.16 | 86.16 |
| Source | dropout 0.25 | 84.92 | 81.26 | 41.04 | 83.43 |
| Compact | no dropout | 86.90 | 49.75 | 24.13 | 87.02 |
| Compact | dropout 0.25 | 88.20 | 80.00 | 43.50 | 85.92 |

该结果说明：availability mask + modality dropout 对缺失模态鲁棒性是有效方向，尤其 main-only 提升明显；但当前只有 seed 0，必须补 seed 1/2 后才能写成论文结论。
