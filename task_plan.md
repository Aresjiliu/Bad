# 任务计划：下一步方向分析与调研

## 目标

基于当前 BRM-Net 结构化轻量化实验、已有文档和外部最新论文趋势，确定下一阶段最稳妥、最能体现硕士论文工作量的推进方向。

## 阶段

1. [completed] 汇总当前代码、实验结果和已有报告中的约束。
2. [completed] 调研 2024-2026 年相关方向：多模态遥感融合、模态缺失鲁棒、多模态轻量化/结构化剪枝、动态推理。
3. [completed] 对比候选路线的创新性、工作量、实验风险和实现成本。
4. [completed] 形成下一步主线、备选路线和近期可执行清单。
5. [completed] 输出中文汇报/决策文档。

## 当前决策问题

当前小 BRM-Net 已能证明 hard-concrete 预算控制和 compact 导出闭环，但模型过小，单纯继续做轻量化不够支撑论文工作量。下一步需要决定：继续扩展轻量化实验，还是迁移到更大双分支主干，并补充模态缺失/动态预算机制。

## 本轮结论

推荐路线为：以当前 `brmnet_core` 小模型作为机制验证和消融基础，同时启动服务器/旧代码中的 ResNet 类双分支主干迁移。短期先补多 seed 统计，随后实现 availability mask 与 modality dropout，使 MQE/RGF 从“输出可视化模块”变成真实的模态缺失鲁棒训练机制。

## 2026-07-06 论文写作推进

1. [completed] 检查 `D:\Academic\paper_submission\brmnet_pricai2026` Overleaf/LaTeX 工程。
2. [completed] 将当前已验证 Houston2013-HS-LiDAR 结构化剪枝实验整理为独立 LaTeX 表格。
3. [completed] 调整摘要、引言、方法、实验、讨论和结论，使当前稿件区分“已验证 prototype 结果”和“待核验三数据集主结果”。
4. [completed] 生成 Overleaf 上传包。
