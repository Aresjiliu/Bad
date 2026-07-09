# P0 可靠性校准与退化评估实现记录

日期：2026-07-10

## 已实现内容

本次实现对应论文升级方案中的 P0：可靠性校准条件路由的最小闭环。

### 1. 新增退化模式

`brmnet_core.engine.apply_degradation` 现在支持：

- `full`
- `main_only`
- `aux_only`
- `aux_noise`
- `aux_noise_low`
- `aux_noise_mid`
- `aux_noise_high`
- `aux_downsample_2`
- `aux_downsample_4`
- `aux_occlusion_25`
- `aux_occlusion_50`

其中 `aux_*` 对应 Houston-HS-LiDAR 中的辅助模态 LiDAR；迁移到 HS-MS 或 HS-SAR 时语义保持为 auxiliary modality。

### 2. 自动生成质量目标

退化评估会生成 `quality_targets`：

| mode | main target | aux target |
|---|---:|---:|
| full | 1.00 | 1.00 |
| main_only | 1.00 | 0.00 |
| aux_only | 0.00 | 1.00 |
| aux_noise_low | 1.00 | 0.80 |
| aux_noise_mid | 1.00 | 0.50 |
| aux_noise_high | 1.00 | 0.20 |
| aux_downsample_2 | 1.00 | 0.50 |
| aux_downsample_4 | 1.00 | 0.25 |
| aux_occlusion_25 | 1.00 | 0.75 |
| aux_occlusion_50 | 1.00 | 0.50 |

训练时 `apply_modality_dropout` 也会根据 availability mask 自动生成或修正质量目标，保证被 drop 的模态 target 为 0。

### 3. 可靠性与路由统计

`train_one_epoch` 和 `evaluate` 现在会统计并输出：

- `q_main`
- `q_aux`
- `fusion_weight_main`
- `fusion_weight_aux`

这些字段会被写入 `metrics.csv`、`metrics.json`、`compact_metrics.csv`、`compact_metrics.json`。

### 4. 汇总脚本支持

`scripts/summarize_brmnet_experiments.py` 已聚合新增字段：

- `q_main_mean/std`
- `q_aux_mean/std`
- `fusion_weight_main_mean/std`
- `fusion_weight_aux_mean/std`

后续可直接用于绘制 reliability calibration curve 和 routing curve。

## 推荐 P0 实验命令

先跑 80% budget、seed 0 的完整 RC 设置：

```powershell
python scripts/run_brmnet_houston.py `
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
  --seed 0
```

如果显存允许，继续跑 seeds 1/2：

```powershell
foreach ($s in 1,2) {
  python scripts/run_brmnet_houston.py `
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
    --seed $s
}
```

汇总结果：

```powershell
python scripts/summarize_brmnet_experiments.py `
  --root output/experiments `
  --output-prefix docs/generated/brmnet_p0_reliability_summary
```

## 论文中可新增的证据

新增字段能够支持两类图：

1. reliability calibration curve：退化强度 vs `q_aux`。
2. routing curve：退化强度 vs `fusion_weight_aux`。

如果曲线呈现退化越强、`q_aux` 和 `fusion_weight_aux` 越低，则可以支撑论文将模块表述为 reliability-calibrated conditional routing，而不是普通 weighted fusion。

## 验证情况

已通过：

```text
python -m pytest tests -q
104 passed, 1 skipped
```

完整仓库级 `python -m pytest -q` 仍会在旧目录 `lib/` 与 `missing3/` 的 torchvision 导入处失败，原因是当前环境中的 torchvision C++ op 注册异常，与本次 BRM-Net P0 改动无关。
