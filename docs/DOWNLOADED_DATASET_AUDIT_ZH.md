# 下载数据集审计报告

本报告由 `scripts/audit_downloaded_datasets.py` 生成，用于判断新下载数据集是否适合进入 BRM-Net 正式实验。

## Trento

- 路径：`D:\Academic\data\Trento-main`
- HSI shape：`[166, 600, 63]`
- 辅助模态 shape：`[166, 600, 2]`
- 标签 shape：`[166, 600]`
- 有限值检查：HSI `True`，辅助模态 `True`
- 标注像素：30214
- 类别数：6
- 类别不平衡比：21.92
- 是否检测到固定划分：False
- 采用建议：`smoke_then_protocolize`
- 原因：fixed train/test split is missing

类别分布：

| 类别 | 像素数 |
| --- | ---: |
| 1 | 4034 |
| 2 | 2903 |
| 3 | 479 |
| 4 | 9123 |
| 5 | 10501 |
| 6 | 3174 |

备注：
- LiDAR file has two channels in this download; the current dataset spec expected one DSM channel, so loader should expose an aux-channel selection option.
- Only allgrd.mat was detected as a label map; no explicit train/test masks were found.

## MUUFL Gulfport

- 路径：`D:\Academic\data\MUUFLGulfport-master`
- HSI shape：`[325, 220, 64]`
- 辅助模态 shape：`[[325, 220, 2], [325, 220, 2]]`
- 标签 shape：`[325, 220]`
- 有限值检查：HSI `True`，辅助模态 `True`
- 标注像素：53687
- 类别数：11
- 类别不平衡比：127.03
- 是否检测到固定划分：False
- 采用建议：`smoke_then_protocolize`
- 原因：fixed train/test split is missing; severe class imbalance requires AA/Kappa and per-class reporting

类别分布：

| 类别 | 像素数 |
| --- | ---: |
| 1 | 23246 |
| 2 | 4270 |
| 3 | 6882 |
| 4 | 1826 |
| 5 | 6687 |
| 6 | 466 |
| 7 | 2233 |
| 8 | 6240 |
| 9 | 1385 |
| 10 | 183 |
| 11 | 269 |

备注：
- The scene-label file matches the common 64-band MUUFL protocol and includes 11 material classes.
- Class imbalance is severe; class 10 and 11 have only 183 and 269 labeled pixels.
- No explicit train/test masks were detected in the downloaded folder.
