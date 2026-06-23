# 当前服务器代码架构审查报告

> 审查对象：`D:\Academic\current\Bad`  
> 日期：2026-06-23  
> 目标：判断当前架构的问题、可复用代码资产，以及后续是否应继续在现有架构上叠加模块。

---

## 1. 总体结论

当前代码不是一个可继续直接扩展的清晰架构，而是一个实验堆叠型代码库。

它的价值主要在于：

1. 已经积累了大量轻量化、缺失模态、单模态和对比实验日志；
2. 已有预算感知通道门控的核心雏形；
3. 已有多模态遥感分类的数据加载、评价指标和训练流程；
4. 已有若干可作为论文“失败路线/消融/对比”的历史分支。

但它不适合作为“继续堆新架构”的基础。原因是：

- 主线分散，`prune/`、`MCL/`、`missing*`、`Fmc/`、`Drfuse/`、`claude/` 都像不同阶段的实验副本；
- 训练循环高度重复，`lib/model_develop.py` 中堆叠了大量相似函数；
- 配置和路径硬编码严重，实验可复现性弱；
- 模型 forward 返回值不统一，训练函数靠隐式约定工作；
- 日志没有统一 schema，论文表格需要人工二次核验；
- 缺失模态架构过重，与轻量化主线冲突。

因此，后续不建议继续在 `missing4/Drfuse/Fmc` 上做复杂融合堆叠。更稳妥的方式是：

> 保留当前代码作为“实验资产库”，从中抽出预算剪枝、数据加载、评价指标和日志证据；新论文主线另建一个薄的、清晰的 BRM-Net 训练分支。

---

## 2. 当前代码架构现状

### 2.1 代码规模

当前仓库约有：

| 类型 | 数量 |
|---|---:|
| Python 文件 | 210 |
| CSV 日志 | 164 |
| `.out` 原始日志 | 160 |
| Python 缓存 | 98 |
| shell 脚本 | 58 |
| 图片 | 53 |

这说明它更接近“服务器实验目录快照”，不是干净的软件工程项目。

### 2.2 主线分布

| 方向 | 主要目录 | 当前判断 |
|---|---|---|
| 预算剪枝/轻量化 | `prune/`, `MCL/`, `models/`, `configuration/prune_config.py` | 论文最值得保留的主线 |
| 缺失/退化模态 | `missing/`, `missing1/`, `missing2/`, `missing3/`, `missing4/`, `Fmc/`, `Drfuse/`, `claude/` | 有实验价值，但架构不宜继续扩展 |
| 基线与复现 | `src/`, `origin/`, `single/` | 可复用为 baseline 和 fallback 证据 |
| 工具与评价 | `datasets/`, `lib/`, `loss/`, `output/logs/` | 可抽取复用 |

---

## 3. 关键问题分析

### 3.1 架构主线不统一

当前仓库同时存在多条未收束路线：

- `MCL/`：早期多模态、gate、drop、DGD 等实验；
- `prune/`：预算剪枝主线；
- `missing*`：多轮缺失模态尝试；
- `Drfuse/`、`Fmc/`、`claude/`：共享/特异特征分解、FMC、DrFuse 类变体；
- `src/`：更早的 DGDNet/ShaSpec/MMFormer/transfer 复现或迁移脚本。

这些目录之间没有统一接口，也没有清晰的继承关系。继续在这个结构上加 MQE、RGF 或新的可靠性模块，会让论文方法更难解释。

**判断：**  
当前架构不能直接作为论文最终方法结构。它只能作为实验来源和代码素材库。

### 3.2 预算剪枝核心有价值，但实现仍是实验型

预算剪枝核心位于：

- `models/base_model.py::Conv2d_Prune`
- `models/base_model.py::Couple_CNN_Prune`
- `models/base_model.py::MDMB_fusion_prune`
- `models/resnet_ensemble.py::HSI_Lidar_Couple_Prune`
- `lib/model_develop.py::train_base_multi_share_unimodal_center_prune`

核心机制是：

1. `Conv2d_Prune` 为每个输出通道维护可学习参数 `sc` 和温度 `temp`；
2. 训练时用 `sigmoid(temp * sc)` 得到通道 mask；
3. 再用随机 Bernoulli 采样 `qc` 参与权重屏蔽；
4. `l1_loss()` 对 mask 做稀疏约束；
5. `update_temp()` 根据采样结果更新温度；
6. 训练函数把分类损失、单模态辅助损失和 L1 门控损失加权。

这部分是当前最有论文价值的代码。它可以支撑“预算感知通道门控/结构学习”的叙事。

但问题也明显：

- `Conv2d_Prune.forward()` 在 eval 模式下使用 `saved_weight`，而 `saved_weight` 是训练循环里手动更新的，不是模块内部稳定维护的状态；
- `custom_bernoulli(mask)` 带随机采样，缺少清晰的确定性部署转换接口；
- `osc`、`gama`、`l1_loss` 的含义没有被明确封装成“预算目标”；
- 真实 Params/FLOPs 裁剪需要单独的 compact/transfer 脚本，和训练逻辑割裂；
- 缺少一个统一的 `export_compact_model()` 或 `materialize_pruned_model()`。

**判断：**  
可以复用思想和部分代码，但不应原样作为最终投稿方法实现。需要包装成更清晰的 BCG 模块。

### 3.3 缺失模态/DrFuse 分支过重，且与轻量化主线冲突

`missing4/models1.py` 中的 `Drfuse` 结构包含：

- `SFDModule`：共享/特异特征分解；
- `CrossAttentionFusion`；
- `DynamicWeightFusion`；
- `CozeFusionClassifier` 或 `HierarchicalFusionClassifier`；
- 类内紧凑性损失；
- 多种共享/特异特征输出。

这条路线适合写“复杂缺失模态鲁棒融合”，但它和“在资源受限条件下轻量化”天然冲突：

- 模块数量多；
- 解释链条长；
- 损失函数多；
- 与预算剪枝代码没有统一接口；
- 论文中容易被质疑为堆模块。

**判断：**  
不建议继续把 DrFuse/FMC 作为论文主创新。它可以复用为：

- 缺失模态实验参考；
- ablation/失败路线；
- “为什么不采用复杂生成式/分解式恢复”的反面论证；
- 部分轻量可靠性融合设计的灵感来源。

### 3.4 训练函数严重重复，维护成本高

`lib/model_develop.py` 中存在大量类似训练函数，每个函数都包含：

- 创建日志目录；
- 保存 args；
- 学习率 scheduler；
- 训练循环；
- 测试；
- 保存 best 模型；
- 写 CSV。

这会带来几个问题：

- 一个 bug 可能在多个函数中重复存在；
- 新增一个损失或指标需要改很多位置；
- 不同实验的 CSV 列含义不一致；
- 结果汇总脚本很难可靠解析；
- 论文复现实验时难以说明“所有方法使用同一训练协议”。

**判断：**  
训练循环本身可复用，但需要抽象出统一 trainer。短期内不要大改历史函数，建议新建一个干净的 `brmnet_train.py` 或 `experiments/train_brmnet.py`。

### 3.5 配置与路径硬编码严重

典型问题：

- `configuration/prune_config.py` 在 import 时直接 `parse_args()`；
- 同一配置文件里修改 `CUDA_VISIBLE_DEVICES`；
- `args.data_root = '../data/' + args.data_root`；
- `prune/huston2013_multi_share_unimodal_center.py` 中硬编码 `args.model_root = args.model_root + "/prune_berlin_succ"`；
- `missing4/config.py` 中 `args.model_root` 被拼接了两次。

这些问题会直接影响复现实验和论文结果可信度。

**判断：**  
配置层不建议原样复用。后续新实验必须使用显式配置文件或命令行参数，避免 import config 时产生副作用。

### 3.6 日志有价值，但不能直接当论文表格

仓库中已有大量 `.csv` 和 `.out`，其中不少记录了 `accuracy_test` 和 `accuracy_best`。

第一轮汇总显示，若只按 best accuracy 排序，可以找到很多候选结果，例如：

- `output/logs/prune_huston_try/hsi_lidar_osc_0.7_l1_0.001_gama_1.01.csv`
- `output/logs/prune_huston_succ/hsi_lidar_osc_0.9_l1_0.001_gama_1.01.csv`
- `output/logs/prune_huston_succ/hsi_lidar_osc_0.7_l1_0.001_gama_1.01.csv`
- `MCL/dgd.out`
- `output/logs/_dgd/hsi_lidardgd.csv`

但这些日志不能直接写进论文，因为还缺：

- 数据集划分说明；
- seed；
- 模态组合；
- budget 与 `osc` 的映射；
- CSV 每列含义；
- 是否是 best checkpoint 还是最后一轮；
- 是否包含 AA/Kappa；
- Params/FLOPs/Latency 来源。

**判断：**  
日志可以复用，但必须先建立 canonical result table，每一行绑定 source log。

---

## 4. 可以复用的代码资产

### 4.1 强复用：预算剪枝核心

| 代码 | 复用方式 | 价值 |
|---|---|---|
| `Conv2d_Prune` | 改造成 `BudgetGatedConv2d` | 核心方法雏形 |
| `Couple_CNN_Prune` | 保留为轻量双分支 encoder | 支撑多模态预算门控 |
| `MDMB_fusion_prune` | 保留为融合层预算门控 | 支撑分支+融合层裁剪 |
| `HSI_Lidar_Couple_Prune` | 改造成 BRM-Net backbone | 可作为第三章/投稿主模型基础 |
| `train_base_multi_share_unimodal_center_prune` | 参考其 loss 组合 | 不建议原样继续扩展 |

建议后续抽象为：

```text
BudgetGatedConv2d
BudgetGatedEncoder
BudgetGatedFusionHead
BudgetRegularizer
CompactExporter
```

### 4.2 强复用：数据加载与评价指标

| 代码 | 复用方式 |
|---|---|
| `datasets/` | 保留数据集读取逻辑 |
| `src/huston2013_dataloader.py` | 保留 Houston2013 loader |
| `lib/model_develop.py::calc_accuracy_multi` | 可复用 OA/AA/Kappa 计算逻辑，但建议抽出独立 evaluator |
| `single/` | 单模态 baseline 证据 |

这些代码是论文实验闭环所必需的，不应重写。

### 4.3 中等复用：缺失模态分支

| 代码 | 复用建议 |
|---|---|
| `missing4/models1.py::SFDModule` | 不建议并入主模型；可作为复杂 baseline 或分析材料 |
| `DynamicWeightFusion` | 可借鉴“可靠性权重”的思想 |
| `CozeFusionClassifier` | 不建议直接复用，结构解释成本高 |
| `Fmc/`, `Drfuse/`, `claude/` | 作为历史实验和失败路线，不作为主线 |

### 4.4 强复用：实验日志

| 用途 | 候选来源 |
|---|---|
| 预算敏感性 | `output/logs/prune_huston_succ/`, `output/logs/prune_huston_try/`, `prune/log/` |
| baseline 对比 | `MCL/*.out`, `output/logs/_dgd/`, `output/logs/_base/` |
| 单模态 fallback | `single/*.out`, `output/logs/single_fc_modal_*` |
| 缺失/退化模态 | `missing2/log_file/`, `output/logs/depose*`, `output/logs/md_*` |

这些日志是当前最重要的“工作量证据”，但必须整理成带来源的表格。

### 4.5 新增可复用模块

已经新增：

```text
models/brmnet_reliability.py
```

包含：

- `ModalityQualityEstimator`
- `ReliabilityGatedFusion`
- `modality_quality_loss`

它目前是独立模块，未接入训练流程。建议后续只在一个受控新分支中接入，不要散落到 `missing*` 多个目录中。

---

## 5. 不建议继续复用或继续扩展的部分

### 5.1 不建议继续堆叠 `missing*`

`missing/`, `missing1/`, `missing2/`, `missing3/`, `missing4/` 是多轮探索的痕迹，不适合继续横向扩展。

风险：

- 新旧模块混杂；
- 很难讲清楚哪个是最终方法；
- 实验结果难以统一；
- 容易继续扩大而不是收敛。

### 5.2 不建议把 DrFuse/FMC 作为最终主方法

它们可以作为调研和对比，但不建议作为当前论文主线。原因是：

- 结构太重；
- 与轻量化目标冲突；
- 与预算剪枝代码没有自然耦合；
- 写作上容易显得“为了补工作量而堆复杂模块”。

### 5.3 不建议继续复制训练函数

以后不要再新增 `train_xxx` 大函数。新实验应采用一个统一训练入口，靠配置切换：

```text
method = baseline / budget_gate / budget_reliability
modality_state = full / main_only / aux_only / noise / downsample / occlusion
budget = 50 / 65 / 80 / 90
```

---

## 6. 建议的新架构

建议不要在原架构中直接修补，而是建立一个薄的新主线：

```text
brmnet/
  datasets/
    reuse existing loaders
  models/
    budget_gate.py
    encoders.py
    fusion.py
    reliability.py
    brmnet.py
  engine/
    trainer.py
    evaluator.py
    logger.py
  configs/
    houston_hsi_lidar_budget65.yaml
    berlin_hsi_sar_budget65.yaml
  tools/
    export_compact.py
    summarize_logs.py
```

最小 BRM-Net 架构：

```text
HSI input      -> BudgetGatedEncoder -> feature_hsi -> MQE -> q_hsi
Aux input      -> BudgetGatedEncoder -> feature_aux -> MQE -> q_aux
feature_hsi, feature_aux, q_hsi, q_aux -> ReliabilityGatedFusion
fused feature  -> BudgetGatedFusionHead -> classifier
```

训练目标：

```text
L = L_cls + lambda_b L_budget + lambda_q L_quality
```

暂时不建议加入：

- 生成式补全；
- 对抗损失；
- 多重共享/特异分解；
- 大规模 MoE；
- 多个复杂一致性损失。

---

## 7. 当前最实在的推进顺序

### 第一步：锁定论文证据

从现有日志中选 canonical logs：

1. Original baseline；
2. Ours 50/65/80/90；
3. complete modality；
4. main-only；
5. aux-only；
6. single modality；
7. missing/degradation baseline。

产出：

```text
docs/generated/canonical_results.md
```

### 第二步：抽取预算剪枝模块

从当前代码复用：

- `Conv2d_Prune` 思想；
- `Couple_CNN_Prune` 结构；
- `MDMB_fusion_prune` 结构；
- `l1_loss()` 与 `update_temp()` 逻辑。

但需要重命名和封装：

```text
Conv2d_Prune -> BudgetGatedConv2d
HSI_Lidar_Couple_Prune -> BudgetGatedMultimodalNet
```

### 第三步：接入轻量可靠性融合

只接入：

- MQE；
- RGF；
- quality loss。

不要接入 DrFuse 全套分解。

### 第四步：补最小退化实验

只做 P0：

| 实验 | 是否必须 |
|---|---|
| 完整模态 | 必须 |
| 仅 HSI | 必须 |
| 仅辅助模态 | 必须 |
| 辅助模态加噪 | 必须 |
| 50/65/80/90 预算 | 必须 |
| 权重随退化变化 | 必须 |

### 第五步：同步投稿工程

将结果回填：

```text
D:/Academic/paper_submission/brmnet_pricai2026/notes/experiments.md
D:/Academic/paper_submission/brmnet_pricai2026/tables/*.tex
```

---

## 8. 最终判断

当前代码最值得保留的不是“完整架构”，而是三类资产：

1. **预算剪枝机制**：能支撑论文核心方法；
2. **数据加载和评价体系**：能支撑实验复现；
3. **历史日志和对比结果**：能支撑论文工作量。

当前代码最不值得继续投入的是：

1. 多个 `missing*` 分支继续横向扩展；
2. DrFuse/FMC 复杂融合继续堆模块；
3. 在 `lib/model_develop.py` 中继续复制训练函数；
4. 直接用现有混乱日志写论文表格。

建议后续路线：

> 用当前仓库作为实验资产库，抽取预算剪枝主线和评价工具，新建一个干净的 BRM-Net 受控分支。论文创新应落在“预算感知结构选择 + 轻量可靠性融合 + 多预算/多退化评估”，而不是继续扩大复杂缺失模态架构。

