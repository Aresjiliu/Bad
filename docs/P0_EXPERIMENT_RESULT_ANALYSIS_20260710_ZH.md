# P0 可靠性校准实验结果分析

日期：2026-07-10

## 1. 实验设置

本轮完成了 P0 的首个完整训练实验。

```text
环境：conda hslinets
GPU：NVIDIA GeForce RTX 4060 Laptop GPU
数据：D:\Academic\HSLiNets-main\Huston2013
格式：legacy
协议：official
模型：BRM-Net hard-concrete
目标预算：80% MACs
seed：0
source epochs：20
compact fine-tune epochs：10
modality dropout：0.25
lambda_quality：0.2
aux_noise_std：0.1
输出目录：output/experiments/p0_reliability/
```

运行时间较长，单个 seed 约 30 分钟以上。实验完整生成了：

- `metrics.json/csv`
- `compact_metrics.json/csv`
- `resource_stats.json`
- `checkpoint.pt`
- `compact_model.pt`
- `history.json`
- `compact_history.json`

## 2. Compact 资源结果

本轮 compact model 成功达到目标预算附近：

| 指标 | 结果 |
|---|---:|
| Params ratio | 80.86% |
| MACs ratio | 80.09% |
| main widths | 28 / 55 / 118 |
| aux widths | 27 / 49 / 118 |
| head widths | 114 / 64 |

判断：结构化预算控制仍然稳定，compact export 是当前最可靠的论文证据。

## 3. Compact 退化矩阵结果

| Mode | OA | AA | q_aux | fusion_weight_aux |
|---|---:|---:|---:|---:|
| full | 81.30 | 84.54 | 0.999 | 0.503 |
| main_only | 74.13 | 78.94 | 0.000 | 0.000 |
| aux_only | 41.28 | 44.85 | 0.999 | 1.000 |
| aux_noise | 31.47 | 31.00 | 0.817 | 0.458 |
| aux_noise_low | 31.53 | 30.99 | 0.820 | 0.459 |
| aux_noise_mid | 31.57 | 30.86 | 0.817 | 0.458 |
| aux_noise_high | 22.70 | 21.40 | 0.056 | 0.283 |
| aux_downsample_2 | 79.80 | 83.04 | 0.999 | 0.503 |
| aux_downsample_4 | 74.49 | 78.64 | 0.999 | 0.503 |
| aux_occlusion_25 | 74.70 | 79.45 | 0.997 | 0.502 |
| aux_occlusion_50 | 64.79 | 71.05 | 0.607 | 0.406 |

## 4. 关键结论

### 4.1 可用性 mask 已经工作正常

`main_only` 时：

- `q_aux = 0.000`
- `fusion_weight_aux = 0.000`
- compact OA = 74.13%

这说明 availability mask 能严格屏蔽缺失辅助模态。相比早期无 modality dropout 的记录中 main-only compact OA 约 49.75%，本轮 74.13% 明显更稳。

`aux_only` 时：

- `fusion_weight_aux = 1.000`
- compact OA = 41.28%

这说明单辅助模态 fallback 能跑通，但 LiDAR-only 本身信息不足，性能仍较弱。

### 4.2 可靠性校准对严重噪声和严重遮挡有反应

`aux_noise_high`：

- `q_aux = 0.056`
- `fusion_weight_aux = 0.283`

`aux_occlusion_50`：

- `q_aux = 0.607`
- `fusion_weight_aux = 0.406`

这说明质量估计器并不是完全失效，对强退化能够降低辅助模态权重。

### 4.3 可靠性校准对 downsample 不敏感

`aux_downsample_2/4` 下：

- `q_aux ≈ 0.999`
- `fusion_weight_aux ≈ 0.503`

说明当前质量估计器没有学会识别下采样退化。可能原因：

1. 训练阶段没有显式加入 downsample degradation；
2. `lambda_quality=0.2` 只在已有 target 时生效，但训练时主要来自 modality dropout，不包含连续退化；
3. LiDAR 下采样后仍保留较多结构信息，模型特征层不容易区分。

### 4.4 中低噪声实验存在实现问题，已修正代码

本轮实验发现，原实现中：

```text
aux_noise_low = max(aux_noise_std, 0.05)
aux_noise_mid = max(aux_noise_std, 0.10)
aux_noise_high = max(aux_noise_std, 0.20)
```

当 `aux_noise_std=0.1` 时，low 和 mid 实际使用相同噪声强度。因此本轮 `aux_noise_low/mid` 结果不能作为正式曲线。

已修正为：

```text
low = 0.5 * aux_noise_std
mid = 1.0 * aux_noise_std
high = 2.0 * aux_noise_std
```

并新增测试保证 noise severity 单调递增。

### 4.5 当前 P0 主风险：完整模态 OA 偏低

compact full OA 为 81.30%，低于此前结构化剪枝三 seed 记录中 80% budget compact OA 均值 85.84%，也低于 seed 0 的 86.90%。

可能原因：

1. `lambda_quality=0.2` 较强，但训练阶段质量目标过于简单；
2. modality dropout 使训练更多关注 fallback，牺牲 full performance；
3. legacy 数据路径与此前 raw official protocol 可能存在差异；
4. 只跑 seed 0，不能判断是否为随机波动。

## 5. 论文写作判断

当前结果不能直接支持“全面提升精度”的强表述，但能支持一个更稳的论点：

> BRM-Net-RC 在约 80% MAC 预算下保持可部署 compact 结构，并通过 availability-aware routing 显著改善缺失辅助模态时的 fallback 行为；可靠性分数对严重退化有响应，但连续退化校准仍需进一步训练增强。

建议论文中暂时不要宣称：

- 对所有退化强度都可靠；
- full-modality 精度不下降；
- reliability estimator 已完全校准。

可以宣称：

- compact export 与预算控制稳定；
- availability mask 严格屏蔽缺失模态；
- modality dropout 显著提升 main-only fallback；
- severe degradation 下 q/weight 有下降趋势；
- 下一步通过连续退化训练增强可靠性校准。

## 6. 下一步优先级

### P0.1：重跑修正后的 seed 0 评估

由于 noise low/mid 实现已修正，需重新训练或至少重新评估：

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
  --output-dir output/experiments/p0_reliability_fixed_noise
```

### P0.2：降低质量损失，避免 full OA 过度下降

建议对 seed 0 做轻量 ablation：

| Setting | lambda_quality | modality_dropout |
|---|---:|---:|
| MD only | 0.0 | 0.25 |
| RC weak | 0.05 | 0.25 |
| RC current | 0.2 | 0.25 |

优先比较：

- full OA；
- main_only OA；
- aux_noise_high q/weight；
- aux_occlusion_50 q/weight。

### P0.3：训练时加入连续退化 target

当前训练时主要只有 modality dropout 的质量目标，缺少 noise/downsample/occlusion 的训练样本。若要让 reliability curve 更像论文图，需要训练时随机采样：

- aux noise low/mid/high；
- aux occlusion 25/50；
- aux downsample 2/4。

这比继续堆模块更重要。

## 7. 本轮结论

本轮实验完成了 P0 的首个端到端验证，并暴露了两个关键事实：

1. 预算控制和 compact export 仍然可靠，是论文的硬核心。
2. reliability-calibrated routing 方向成立，但当前训练信号不足，必须补连续退化训练或降低质量损失，否则 full OA 与中低退化表现不稳。

下一步不应急着跑 seeds 1/2。应先修正 fixed-noise 后做 seed 0 的小型 ablation，找到 `lambda_quality` 与退化训练策略，再扩展多 seed。

