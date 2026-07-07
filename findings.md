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
## 2026-07-06 多种子结果发现

- Houston2013-HS-LiDAR official split 中，65/80/90 target MACs 预算在 3 seed 下实际 MACs ratio 分别为 64.96 +/- 0.12、80.01 +/- 0.10、90.07 +/- 0.08，说明预算命中是当前最稳的可写结论。
- Compact OA 分别为 85.53 +/- 2.20、85.84 +/- 0.92、85.74 +/- 0.88，不呈现随预算单调提升，因此不应把当前 prototype 写成性能主表。
- 当前最稳妥论文写法是：prototype 作为 hard-concrete gate + compact export + validation selection 的机制验证和消融基础，下一步用 missing-modality training loop 和较大双分支 backbone 承担主工作量。
- availability mask 的实现使 RGF 从“质量分数可视化/隐式融合”推进到“显式缺失模态鲁棒融合”；下一步实验评价应重点看 main_only、aux_only、aux_noise，而不是只盯 full OA。
- 80% budget seed0 初步对照显示，compact 模型加入 dropout 0.25 后 full/main_only/aux_only/aux_noise OA 为 88.20/80.00/43.50/85.92；no-dropout 为 86.90/49.75/24.13/87.02。该方向值得补 seed 1/2，但目前不能作为最终结论。
