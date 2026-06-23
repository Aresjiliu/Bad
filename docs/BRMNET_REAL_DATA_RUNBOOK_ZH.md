# BRM-Net 真实数据接入运行说明

本文件记录 `brmnet-core-extraction` 分支中最小真实数据闭环。

## 当前入口

本地先做不读数据的配置检查：

```powershell
python scripts\run_brmnet_houston.py `
  --dry-run `
  --data-root ..\data\Huston2013 `
  --pair-modalities hsi+lidar `
  --class-num 15
```

确认数据路径后，在服务器上运行真实数据 smoke test：

```powershell
python scripts\run_brmnet_houston.py `
  --data-root ..\data\Huston2013 `
  --pair-modalities hsi+lidar `
  --epochs 1 `
  --batch-size 32 `
  --metrics-csv output\logs\brmnet_core\houston_degradation_matrix.csv
```

## 复用关系

- 旧数据加载：`src/huston2013_dataloader.py`
- 旧 batch 字段：`m_1`、`m_2`、`label`
- 新模型：`brmnet_core.BRMNet`
- 新训练评估：`brmnet_core.engine`
- 新结果表：`brmnet_core.reporting.write_metrics_csv`

## 首轮实验矩阵

脚本默认评估四种模式：

| mode | 含义 |
|---|---|
| `full` | 完整双模态 |
| `main_only` | 仅主模态，辅助模态置零 |
| `aux_only` | 仅辅助模态，主模态置零 |
| `aux_noise` | 辅助模态加入高斯噪声 |

## 下一步

1. 在服务器上用 `--dry-run` 确认通道数、类别数和参数量。
2. 确认 `--data-root` 指向真实 Houston2013 目录。
3. 跑 1 epoch smoke test，确认 batch shape、loss 和 CSV 输出。
4. 跑 50/65/80/90 多预算配置。
5. 将 CSV 汇总回论文表格和导师汇报。
