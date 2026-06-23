# BRM-Net 核心抽取分支设计

> 分支：`brmnet-core-extraction`  
> 目标：从服务器历史代码中抽取最有价值的核心模块，形成一个干净、可解释、可继续实验的 BRM-Net 分支基础。

## 1. 抽取原则

当前代码中最有价值的不是 DrFuse/FMC 等复杂缺失模态架构，而是：

1. 预算感知通道门控；
2. 多模态双分支编码；
3. 已有数据加载和评价体系；
4. 历史实验日志。

因此新分支只抽取最小核心，不继续继承 `missing*`、`Drfuse`、`Fmc` 的复杂模块。

## 2. 新增核心包

新增目录：

```text
brmnet_core/
```

包含：

| 文件 | 作用 |
|---|---|
| `budget_gates.py` | 从 `Conv2d_Prune` 抽取并清理出的 `BudgetGatedConv2d` |
| `encoders.py` | 双模态预算门控 encoder |
| `fusion.py` | MQE、RGF、预算门控分类头 |
| `model.py` | 最小 BRM-Net 组合模型 |
| `losses.py` | 分类、预算、质量监督损失组合 |
| `README.md` | 使用说明 |

## 3. 和旧代码的关系

| 旧代码 | 新分支处理 |
|---|---|
| `models/base_model.py::Conv2d_Prune` | 抽取为 `BudgetGatedConv2d`，去掉硬编码 `.cuda()` |
| `Couple_CNN_Prune` | 抽取为 `BudgetGatedEncoder` |
| `MDMB_fusion_prune` | 抽取为 `BudgetGatedFusionHead` |
| `missing4/Drfuse` | 不继承，只借鉴“质量/权重融合”的思想 |
| `models/brmnet_reliability.py` | 合并思想到 `brmnet_core/fusion.py` |

## 4. 当前未做的事情

当前只建立核心模块基础，尚未：

- 接入旧 dataloader；
- 写统一 trainer；
- 跑新实验；
- 替换旧日志体系；
- 导出 compact model。

这些应作为下一步受控实施，不应在历史分支里继续散改。

