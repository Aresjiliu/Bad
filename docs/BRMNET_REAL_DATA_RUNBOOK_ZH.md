# BRM-Net Houston2013 真实数据运行说明

本说明对应 `brmnet-core-extraction` 分支，支持原始 Houston2013 数据的官方固定划分与随机像素划分。

## 原始数据

`--data-root` 目录需要包含：

| 文件 | MAT 键 |
|---|---|
| `2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat` | `ans` |
| `2013_IEEE_GRSS_DF_Contest_LiDAR.mat` | `LiDAR_data` |
| `GRSS2013.mat` | `name` |
| `2013_IEEE_GRSS_DF_Contest_Samples_TR.mat` | `TR_Samples` |

当前 raw 入口仅支持 `hsi+lidar`。旧预生成 patch 仍可通过 `--data-format legacy` 使用。

## 配置检查

不加载数据，只检查模型通道和参数量：

```powershell
python scripts\run_brmnet_houston.py `
  --dry-run `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset"
```

## 数据协议检查

官方竞赛固定划分：

```powershell
python scripts\run_brmnet_houston.py `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset" `
  --split-protocol official `
  --dataset-only `
  --num-workers 0 `
  --output-dir output\smoke
```

随机像素划分：

```powershell
python scripts\run_brmnet_houston.py `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset" `
  --split-protocol random `
  --split-seed 42 `
  --dataset-only `
  --num-workers 0 `
  --output-dir output\smoke
```

两套协议均应得到：

- 训练样本：2,832；
- 测试样本：12,197；
- 逐类训练数量完全一致；
- 训练坐标 SHA-256 不同。

## 一轮训练

官方划分：

```powershell
python scripts\run_brmnet_houston.py `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset" `
  --split-protocol official `
  --epochs 1 `
  --batch-size 64 `
  --num-workers 0 `
  --device cpu `
  --output-dir output\smoke
```

随机划分只需将协议改为：

```powershell
--split-protocol random --split-seed 42
```

有 CUDA 时将 `--device cpu` 改为 `--device cuda`。正式对比必须保持两套协议的模型、训练随机种子、epoch、batch size 和优化器参数一致。

## 输出目录

默认命名：

```text
output/experiments/houston2013_hsi-lidar_<protocol>_seed<split_seed>/
```

每次 raw 实验包含：

| 文件 | 含义 |
|---|---|
| `config.json` | 完整命令行配置 |
| `split.npz` | 训练/测试坐标和标签 |
| `split_summary.json` | 协议、样本数、逐类数量和坐标哈希 |
| `normalization.npz` | 全场景 HSI/LiDAR z-score 统计量 |
| `data_fingerprint.json` | 原始 MAT 文件路径、大小、修改时间和 SHA-256 |
| `metrics.csv` | 四种模态状态的评估指标 |
| `checkpoint.pt` | 模型、参数和数据协议元数据 |

脚本默认评估：

| mode | 含义 |
|---|---|
| `full` | 完整双模态 |
| `main_only` | 仅 HSI |
| `aux_only` | 仅 LiDAR |
| `aux_noise` | LiDAR 加高斯噪声 |

## 复现已有划分

使用保存的坐标文件：

```powershell
--split-protocol random --split-file path\to\split.npz
```

脚本会校验协议、坐标范围、GT 标签及全部标注像素覆盖关系，不会自动回退或重新随机划分。

## 论文使用边界

- 官方和随机划分结果不能在未注明协议的情况下放入同一排名表。
- 论文主表优先使用官方固定划分。
- 随机划分用于分析空间自相关和划分敏感性。
- 1 epoch 输出只用于验证代码闭环，不能作为论文结果。
- 不得使用 GT 填充分类图中的未预测区域。
