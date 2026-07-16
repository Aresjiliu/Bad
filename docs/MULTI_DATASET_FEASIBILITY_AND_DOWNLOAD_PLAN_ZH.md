# BRM-Net 多数据集实验可行性与下载计划

日期：2026-07-16  
当前代码分支：`brmnet-core-extraction`  
当前环境：`conda activate hslinets`，CUDA 可用

## 结论摘要

为了增强论文说服力，下一阶段应从“单一 Houston2013 机制验证”推进到“Houston2013 + Trento + MUUFL”的三数据集验证。当前最实在的路线不是马上追求 Augsburg 或 Houston2018，而是先把 Trento 接入主代码，再接 MUUFL。理由是：Trento 与当前 Houston2013 同属 HS-LiDAR patch 分类，类别少、规模小、常被 MTNet/DFNet/CAMFNet 使用，最适合验证代码泛化；MUUFL 更难、类别不平衡明显，适合体现论文实验深度和鲁棒性分析；Augsburg 存在标签重构问题，Houston2018 极度不平衡且规模更大，暂不作为短期主线。

## 当前证据基础

当前主线已完成 Houston2013-HS-LiDAR 上的三类证据：

1. 结构化预算控制与 compact export：65/80/100 多预算 profile 可以稳定生成。
2. 缺失/退化模态鲁棒性：已有 full、main_only、aux_only、aux_noise、occlusion、downsample 等状态评估。
3. 质量条件预算路由：`pareto_delta_0.01` 在三种子 Houston profile sweep 中达到 mean OA 0.7537、mean MACs 0.8810，较 100% profile 约节省 11.9% MACs，mean regret 约 0.0010。

但单数据集仍有明显风险：评审或导师会认为方法只在 Houston2013 上调参有效，不能证明“多模态遥感分类中的通用轻量化与鲁棒路由”。

## 数据集优先级

| 优先级 | 数据集 | 模态 | 规模/类别 | 推荐原因 | 风险 |
| --- | --- | --- | --- | --- | --- |
| P0 | Houston2013-HS-LiDAR | 144-band HSI + 1-band LiDAR | 349×1905, 15 类 | 当前已跑通，作为主基准和方法开发集 | 官方 ROI 与随机划分不可混用 |
| P0 | Trento | 63-band HSI + 1-band LiDAR/DSM | 600×166, 6 类 | 最适合作为第一新增数据集；小、标准、文献常用 | 精度上限很高，需强调效率/鲁棒性而非只比 OA |
| P1 | MUUFL Gulfport | 64/72-band HSI + 2-band LiDAR | 325×220, 11 类 | 更难、更不平衡，适合体现工作量和鲁棒性 | 64/72 波段协议、训练样本设置差异大 |
| 暂缓 | Augsburg | 101-band HSI + LiDAR | 1364×1636, 6 类 | 可作为后续补充 | CAMFNet 使用重构标签，原始协议不清会造成不可比 |
| 暂缓 | Houston2018 | 48-band HSI + LiDAR | 601×2385, 20 类 | 可作为大规模压力测试 | 极端类别不平衡，当前 patch 管线耗时高 |
| 备选 | Berlin-HS-SAR | HSI + SAR | 8 类左右 | 可支撑“跨物理机理”叙事 | 当前主代码是 HS-LiDAR 假设，SAR 通道/预处理要单独定义 |

## 请优先下载的数据

### 第一批：Trento

请尽量下载并保留原始文件名，同时记录来源链接。

需要文件：

1. HSI cube，约 600×166×63。
2. LiDAR 或 DSM raster，约 600×166×1。
3. GT label map。
4. 训练/测试 mask，或论文公开 split 文件。
5. 如果下载包内有 `Train`、`Test`、`TRLabel`、`TSLabel`、`training`、`test` 等文件，请全部保留。

建议放置：

```text
D:\Academic\data\Trento\
  hsi.*
  lidar.*
  gt.*
  train_mask.*    # 如有
  test_mask.*     # 如有
  README_source.txt
```

### 第二批：MUUFL Gulfport

需要文件：

1. HSI cube：优先同时保留 raw 72-band 与 denoised 64-band 版本。
2. LiDAR rasters：若有两个 LiDAR 通道或 DSM/DEM 派生栅格，全部保留。
3. GT label map。
4. 训练/测试 mask 或 published split。
5. 数据说明文档，尤其是类别编号和 noisy bands 删除规则。

建议放置：

```text
D:\Academic\data\MUUFL\
  hsi_72.*        # 如有
  hsi_64.*        # 如有
  lidar_*.*
  gt.*
  train_mask.*    # 如有
  test_mask.*     # 如有
  README_source.txt
```

### 第三批：暂不急

Augsburg 与 Houston2018 只有在 Trento/MUUFL 跑通后再下载。若下载，重点确认是否有论文同款重构标签和固定划分；没有的话，不适合进入毕业论文主实验。

## 代码接入计划

当前代码问题：`scripts/run_brmnet_houston.py` 与 `brmnet_core.data.houston` 仍然是 Houston 专用。已经新增 `brmnet_core.data.dataset_specs` 作为数据集规格注册表，先固化各数据集的类别数、通道数、优先级和协议风险。

下一步代码应按以下顺序推进：

1. 新增通用 `MultimodalScene` 数据结构：字段包括 `hsi`、`aux`、`gt`、`train_mask`、`test_mask`、`metadata`。
2. 将 `HoustonPatchDataset` 泛化为 `MultimodalPatchDataset`，保留当前输入键 `m_1`、`m_2`、`label`、`coord`，避免破坏 engine。
3. 增加 `load_trento_scene(root)`，读取 Trento 原始 `.mat` 或常见 mask 命名。
4. 增加 `load_muufl_scene(root)`，先支持 64-band 协议，再保留 72-band 可选参数。
5. 将 `run_brmnet_houston.py` 扩展或新建 `run_brmnet_multidataset.py`，参数改为 `--dataset houston2013_hs_lidar|trento|muufl`。
6. 所有新 loader 必须先支持 `--dataset-only`，只检查样本数、类别分布、通道数、split 是否正确，再启动训练。

## 实验矩阵

### 最小毕业论文矩阵

| 数据集 | 种子 | 预算 | 模态状态 | 目标 |
| --- | --- | --- | --- | --- |
| Houston2013 | 0/1/2 | 65/80/100 | full, main_only, aux_only, noise, occlusion, downsample | 主结果与路由分析 |
| Trento | 0/1/2 | 80/100，必要时加 65 | full, main_only, aux_only, noise | 验证泛化，控制耗时 |
| MUUFL | 0/1/2 | 80/100，必要时加 65 | full, main_only, aux_only, noise | 验证困难不平衡场景 |

### 若时间不足的保底矩阵

1. Houston2013 保持当前完整三种子结果。
2. Trento 跑完整 3 seed：80/100 两个 profile + 缺失/噪声状态。
3. MUUFL 先跑 1 seed：80/100 两个 profile + full/main_only/aux_only/noise，用作补充实验或附录。

### 若时间充足的增强矩阵

1. Trento/MUUFL 都补齐 65/80/100 profile。
2. 对每个数据集生成 Pareto profile sweep。
3. 加入 leave-one-state-out learned router 报告，展示路由泛化情况。
4. 加入 classification map、confusion matrix、per-class AA，可视化类别不平衡影响。

## 论文写法建议

多数据集部分不应写成“所有数据集都达到 SOTA”。更稳妥的主张是：

1. Houston2013 用于完整方法开发、消融和质量条件预算路由。
2. Trento 用于验证轻量化与鲁棒融合机制可以迁移到另一 HS-LiDAR 场景。
3. MUUFL 用于验证方法在类别不平衡、LiDAR 多通道和更困难校园场景下的稳定性。
4. Augsburg/Houston2018 的未使用原因应主动说明：协议不一致或成本过高，避免被认为漏做。

可写入论文的表述：

> 为避免单一数据集导致的偶然性，本文在 Houston2013-HS-LiDAR、Trento 和 MUUFL Gulfport 三个公开 HS-LiDAR 数据集上进行实验。三者在场景类型、类别数量、光谱波段数和 LiDAR 数据形式上存在差异，能够从城市复杂场景、乡村场景和类别不平衡校园场景三个角度评估所提方法的精度、效率和鲁棒性。

## 当前立即行动

1. 用户下载 Trento，并保持原始压缩包与解压目录。
2. 下载后先运行 `dataset-only` 检查，不直接训练。
3. 代码侧下一步实现 Trento loader 与 `MultimodalPatchDataset`。
4. Trento 跑通后再下载 MUUFL，避免同时引入两个数据协议导致定位困难。
