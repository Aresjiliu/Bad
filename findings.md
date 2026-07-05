# 研究发现记录

## 2026-07-05 初始状态

- 当前分支：`brmnet-core-extraction`。
- 最新提交：`dd08705 Add class-balanced validation checkpointing`。
- 当前小 BRM-Net 已实现：Houston official/raw 数据协议、Hard-Concrete 结构化通道门控、真实 Params/MACs 预算、compact model 导出、source/compact validation best checkpoint、退化矩阵评估。
- 最新 seed0 结果显示：65/80/90 三档实际 MACs 命中稳定，但 OA 单 seed 非单调，65% compact OA 最高。
- 已有文档多次指出：小 BRM-Net 适合机制验证，不足以独立承担硕士论文工作量；下一步应迁移到服务器较大双分支主干。

## 2026-07-05 本地材料与代码发现

- `brmnet_core/model.py` 已包含 `ModalityQualityEstimator`、`ReliabilityGatedFusion`、共享融合 gate、Hard-Concrete encoder 和 fusion head。
- `brmnet_core/compact.py` 已能导出无 gate 的 compact BRM-Net，并保留 MQE/RGF。
- 当前缺口不是“没有可靠性模块”，而是缺少训练阶段的 `availability_mask`、modality dropout、退化强度质量监督，以及缺失模态下的 masked softmax。
- 旧 `models/resnet_ensemble.py` 中存在 `HSI_Lidar_Couple_Prune`、`ResNet18`、`Couple_CNN` 等较大主干素材，可作为论文工作量主模型迁移来源。
- `missing4/Drfuse/FMC` 提供缺失模态历史探索，但结构过重，不适合作为主线。

## 2026-07-05 外部论文趋势发现

- Missing modality survey 将模态缺失定义为独立问题，原因包括传感器限制、成本、隐私和数据丢失，说明第四章的“模态不稳定”问题成立。
- RingMoE 和 MAPEX 均体现“模态专家 + 动态路由/专家剪枝”趋势，但它们是大模型/基础模型方向；本项目应借鉴语言，不应照搬完整 MoE。
- MaMOL 将遥感缺失模态分类重构为条件计算问题，支持把 MQE/RGF 写成轻量条件路由。
- 2026 年高光谱压缩 benchmark 强调同时看 accuracy、memory、inference efficiency，说明论文主表必须有 Params/MACs/Latency/Model Size。
- ICTAI 2026 截稿已延至 2026-07-21；PRICAI 2026 已在 2026-06-27 截稿，ACCV 2026 截稿为 2026-07-05，当前不适合作为稳妥主投。
