# Houston2013 双划分数据协议设计

日期：2026-06-24  
状态：待实现  
适用分支：`brmnet-core-extraction`

## 1. 目标

为 BRM-Net 建立统一、可复现的 Houston2013 HSI+LiDAR 数据协议，使以下两种划分能够在完全相同的模型、训练器、指标和退化评估流程中公平比较：

- `official`：IEEE GRSS 2013 竞赛固定训练坐标；
- `random`：按官方逐类训练样本数量，从全部标注像素中随机抽样。

本阶段只解决数据协议和实验可复现性，不迁移 HSLiNets 模型，不扩展到 Trento、Berlin 或其他模态组合。

## 2. 背景与问题

HSLiNets-main 中已经验证：Houston2013 的划分方式会显著改变实验结果。随机像素划分使训练集和测试集在空间上交错，容易利用遥感影像的空间自相关；官方固定划分更接近跨区域泛化，通常更困难。

现有 BRM-Net 入口只读取预生成的 `.mat` patch，缺少样本中心坐标、划分协议、随机种子、原始数据指纹和分类图位置。因此不能只比较两个数据目录中的结果，必须建立以原始影像和坐标索引为核心的数据协议层。

## 3. 方案选择

采用“统一数据协议层”，并保留旧 `.mat` loader 作为兼容入口。

不采用以下方案：

- 预生成两套 `.mat` 作为主要协议：会复制数据，且难以追踪坐标和生成可信分类图；
- 直接复制 HSLiNets pipeline：其 patch 抽取和划分存在重复随机过程，无法明确唯一划分清单；
- 只保留一种划分：无法量化划分方式对当前方法的影响。

## 4. 数据来源与契约

### 4.1 原始文件

默认从一个数据根目录读取：

- `2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat`
- `2013_IEEE_GRSS_DF_Contest_LiDAR.mat`
- `GRSS2013.mat`
- `2013_IEEE_GRSS_DF_Contest_Samples_TR.mat`

数据键不能依赖“MAT 文件最后一个键”。适配器应优先使用已知键，并在键缺失时输出包含实际键列表的明确错误。

当前本地文件已经确认使用以下键：

- HSI：`ans`
- LiDAR：`LiDAR_data`
- GT：`name`
- 官方 ROI：`TR_Samples`

### 4.2 标准化内部表示

- HSI：`float32[H, W, 144]`
- LiDAR：`float32[H, W, 1]`
- GT：`int64[H, W]`，背景为 0，类别为 1 至 15
- 坐标：`int64[N, 2]`，统一为 `(row, col)`
- 训练标签接口：转为 0 至 14
- 元数据和分类图：保留原始 1 至 15 类别编号

加载时必须校验 HSI、LiDAR 和 GT 的空间尺寸一致。

## 5. 划分协议

### 5.1 官方固定划分 `official`

从官方训练 ROI 文件解析 15 类训练像素坐标。其余满足 `GT > 0` 且不在训练坐标中的像素作为测试集。

`TR_Samples` 是包含 ENVI ROI 文本记录的对象数组，不是数值掩膜。解析器按 `ROI name`、`ROI npts` 和后续 `ID, X, Y, Lat, Lon` 记录构建每类坐标，并忽略文件头和空记录。

必须校验：

- 坐标位于影像范围内；
- 训练坐标无重复；
- 训练和测试坐标无交集；
- ROI 标签与 GT 标签一致；
- 每类训练数量与 Houston2013 官方数量一致；
- 训练总数为 2,832。
- 测试总数为 12,197。

若 ROI 文件坐标采用 1-based `(X, Y)`，解析器应显式转换为 0-based `(row, col)`，并通过 GT 标签一致性检查确认方向和偏移。不得依靠人工猜测静默继续。

### 5.2 随机像素划分 `random`

对每个类别，从全部有效标注坐标中按官方训练数量无放回抽样，其余坐标作为测试集。

要求：

- 使用局部 `numpy.random.Generator`；
- 默认种子为 42；
- 同一数据和种子必须生成相同坐标；
- 不修改 NumPy 全局随机状态；
- 每类训练数量与官方协议严格一致；
- 支持后续使用多个种子报告均值与标准差。

### 5.3 唯一划分

划分只执行一次。后续 Dataset、DataLoader、训练和分类图均消费保存后的坐标索引，不允许在 patch 抽取后再次随机划分。

## 6. 预处理与 Patch

### 6.1 归一化

- HSI：使用整幅无标签影像计算逐波段 z-score；
- LiDAR：使用整幅无标签影像计算单波段 z-score；
- 两种划分协议必须复用完全相同的统计量；
- 先标准化整幅影像，再按坐标提取训练和测试 patch。

该约定属于单场景高光谱分类中常用的 transductive 预处理：允许读取待分类场景的无标签像素分布，但不使用测试标签。这样可确保双协议对比中唯一变化是训练坐标，而不是归一化参数。元数据中保存每个模态的均值、标准差和零方差处理记录。

### 6.2 Patch 抽取

- 默认 patch size 为 7，必须为正奇数；
- 先对整幅影像执行反射填充，再按中心坐标提取 patch；
- 两种划分协议使用完全相同的 padding 和抽取函数；
- 不因边界位置删除样本；
- HSI 输出 `[144, P, P]`，LiDAR 输出 `[1, P, P]`。

数据增强只作用于训练集，并对两个模态执行相同空间变换。本阶段沿用水平和垂直翻转，不引入旋转或光谱扰动。

## 7. 模块边界

- `brmnet_core/data/houston.py`
  - 原始 MAT 数据读取；
  - 数据形状和键校验；
  - Houston2013 类别与训练数量常量。
- `brmnet_core/data/splits.py`
  - 官方 ROI 解析；
  - 随机逐类划分；
  - 划分完整性校验；
  - 坐标清单保存与加载。
- `brmnet_core/data/patch_dataset.py`
  - 全场景统计量计算；
  - 全图标准化；
  - 反射填充与 patch 抽取；
  - 双模态 Dataset。
- `brmnet_core/data/factory.py`
  - 根据协议和配置创建 train/test DataLoader；
  - 返回数据元数据。

旧 `brmnet_core.legacy.get_houston_loaders` 保留，不作为新实验的默认入口。

## 8. 命令行与数据流

扩展 `scripts/run_brmnet_houston.py`：

```text
raw MAT files
  -> load and validate
  -> build/load split coordinates
  -> fit/load shared scene normalization
  -> normalize full image
  -> coordinate-based patch datasets
  -> BRM-Net train/evaluate
  -> degradation matrix and reproducibility artifacts
```

新增核心参数：

- `--data-format raw|legacy`，默认 `raw`
- `--split-protocol official|random`
- `--split-seed 42`
- `--split-file <path>`，可选，优先加载已有划分
- `--num-workers`
- `--output-dir`

一次命令只运行一种划分协议，避免输出文件混淆。官方与随机协议由外层实验脚本分别调用。

## 9. 输出与可复现性

每次实验输出目录至少包含：

- `config.json`
- `split.npz`
- `split_summary.json`
- `data_fingerprint.json`
- `normalization.npz`
- `metrics.csv`
- `checkpoint.pt`

实验命名至少包含：

```text
houston2013_hsi-lidar_<protocol>_seed<seed>_budget<budget>
```

官方协议仍记录 `seed`，但明确标记其不影响划分，仅影响训练随机性。

## 10. 指标与分类图约束

统一报告 OA、AA、Cohen's Kappa、15 类逐类准确率、混淆矩阵，以及完整、主模态、辅助模态和辅助模态噪声退化结果。

分类图要求：

- 预测位置必须来自坐标清单；
- 未预测位置保持背景色或专用“未评估”颜色；
- 不得用 GT 填充边界或缺失预测；
- 图标题和文件名必须包含划分协议；
- 官方和随机划分使用相同类别颜色表。

分类图与混淆矩阵工具可在数据协议稳定后实现，不阻塞第一阶段训练闭环。

## 11. 错误处理

以下情况立即失败并给出可操作错误：

- MAT 文件或已知键缺失；
- 数据空间尺寸不一致；
- patch size 为偶数或非正数；
- 官方 ROI 坐标无法与 GT 对齐；
- 某类可用样本少于要求的训练数量；
- train/test 坐标重叠；
- split 文件与当前数据指纹不一致；
- split 文件协议或类别数不兼容。

不允许自动回退到随机划分，也不允许自动交换 HSI/LiDAR 文件。

## 12. 测试策略

### 12.1 单元测试

- 已知 MAT 键解析和缺失键错误；
- 随机划分逐类数量、互斥性和确定性；
- 官方 ROI 的坐标方向、1-based 转换和 GT 对齐；
- 全场景标准化统计及双协议复用；
- 反射填充后的边界 patch 形状；
- 标签从 1-based 到 0-based 的转换；
- split 保存后加载结果完全一致。

### 12.2 集成测试

使用小型合成影像执行两套 loader 构建、单 batch 前向和反向、1 epoch 训练、四种退化模式评估和输出 schema 检查。

### 12.3 真实数据 Smoke Test

在 `D:\Academic\HSLiNets-main\Dataset` 上分别运行：

- 官方协议 dry-run 和样本统计；
- 随机协议 seed 42 dry-run 和样本统计；
- 每种协议 1 epoch；
- 检查训练数 2,832、测试数和逐类数量；
- 检查坐标文件和指标文件。

## 13. 实验推进顺序

1. 实现原始数据读取、官方/随机划分和校验。
2. 实现坐标 Dataset、共享全场景归一化和 DataLoader factory。
3. 将 raw 数据入口接入现有 Houston runner。
4. 跑两种协议的 1 epoch smoke test。
5. 增加双协议批量实验脚本。
6. 在两种协议上运行完整、单模态和噪声退化矩阵。
7. 再叠加预算 65/80/90。
8. 最后生成分类图、混淆矩阵和划分空间分布图。

## 14. 论文使用边界

两种协议的结果不能混在同一排名表中而不注明划分方式。论文主表优先采用官方固定划分，随机划分作为“划分敏感性与空间泄漏分析”实验。

若随机划分精度显著高于官方划分，应解释为空间自相关与域偏移差异，而不能将其单独作为模型优越性的证据。
