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
## 2026-07-06 多种子实验与论文同步

1. [completed] 运行 Houston2013-HS-LiDAR official split 结构化剪枝 seed 1/2，并结合已有 seed 0 形成 3 seed 统计。
2. [completed] 生成 `docs/generated/structured_pruning_multiseed_summary.md` 和 `docs/generated/structured_pruning_multiseed_runs.csv`。
3. [completed] 将 Overleaf prototype 表格从单 seed 更新为 3 seed mean +/- std。
4. [completed] 新增中文更新文档 `docs/PAPER_EXPERIMENT_UPDATE_20260706_ZH.md`。
5. [completed] 按论文需求实现 availability mask + modality dropout 缺失模态训练闭环。
6. [completed] 运行 80% budget seed0 modality dropout 初步对照实验，重点比较 full/main_only/aux_only/aux_noise 四种模式。
7. [pending] 补 80% budget seed1/2 modality dropout 对照实验，确认鲁棒性收益是否稳定。

## 2026-07-10 continuous improvement plan

### Goal

Continue the BRM-Net thesis/paper work according to the latest guidance: make the contribution defensible through compact export, availability-conditioned inference, reliability ablations, degradation calibration, and accuracy-efficiency-robustness reporting.

### Phase A: Core Ablation Closure

1. [completed] Implement `--fusion-mode uniform` as `w/o reliability / average fusion`.
2. [completed] Generate the core experiment matrix including `without_reliability_uniform_fusion`.
3. [completed] Run `without_reliability_uniform_fusion` seed0 and summarize it in `docs/generated/brmnet_priority_summary.*`.
4. [completed] Run `without_reliability_uniform_fusion` seed1 and seed2 for 3-seed reliability ablation.
5. [completed] Refresh `full` with the current latency profiling fields.
6. [pending] Re-run or refresh `without_modality_dropout` and `without_budget_loss` if they are needed for the final ablation table.

### Phase B: Soft-Mask-Only vs Compact Export

1. [completed] Update the summary script to report source gated model and compact export metrics side by side.
2. [pending] Generate a paper-ready table comparing source gated and compact export under full/main_only/aux_only/noisy aux.
3. [pending] Investigate any large source/compact gap through export equivalence and compact fine-tuning history.

### Phase C: Reliability Degradation Calibration

1. [completed] Extract OA, q_aux, and fusion_weight_aux for aux_noise/high, downsample, and occlusion modes from the priority summary.
2. [completed] Test missing/non-missing quality supervision and identify its failure on degraded-but-available auxiliary inputs.
3. [completed] Implement and validate noise-based degraded-available quality supervision.
4. [pending] Extend quality degradation augmentation to occlusion and resolution-loss cases.
5. [pending] Compute Pearson/Spearman correlation between degradation severity and reliability/fusion/OA drop.
6. [pending] Generate degradation-curve figures and a Chinese analysis note.
7. [pending] Paper wording should emphasize `degradation-aware reliability supervision`; avoid claiming generic reliability learning until multi-type degradation is validated.

### Phase E: Degradation-Aware Method Upgrade

1. [completed] Add `quality_degradation_supervised` to the experiment matrix.
2. [completed] Complete Houston official seeds 0/1/2 for `quality_degradation_supervised`.
3. [completed] Add multi-type degradation augmentation: Gaussian noise, random occlusion/block masking, and downsample-upsample resolution loss.
4. [completed] Add ablations for degradation type: none, noise-only, multi-type.
5. [completed] Complete Houston official seeds 0/1/2 for multi-type degradation supervision.
6. [pending] Tune degradation type proportions or curriculum to recover clean full-modality OA while keeping occlusion/downsample gains.
7. [pending] Update method writing around three contributions: structured compact export, availability-conditioned inference, and degradation-aware reliability supervision.

### Phase F: Robustness Tradeoff Tuning

1. [completed] Run a compact tuning grid for multi-type probability 0.25 vs 0.5.
2. [completed] Add `quality_multi_degradation_p025` to the formal experiment matrix.
3. [completed] Select the current thesis/paper default by average adverse-state OA first, then full-modality OA as a tie-breaker.
4. [pending] Add a weighted degradation sampler if more tuning is needed after paper table generation.
5. [pending] Generate a paper-ready table with baseline, uniform, noise-only, p=0.50 multi-type, and p=0.25 multi-type degradation supervision.
6. [pending] Update method text to explain p=0.25 as a light degradation curriculum rather than an arbitrary hyperparameter.

### Phase D: Paper Figures and Writing

1. [pending] Generate retained-width visualization by branch and layer.
2. [pending] Generate AER Pareto plot: MAC/latency vs average OA over modality states.
3. [pending] Update LaTeX Method/Experiment sections and add Algorithm 1/2.
4. [pending] Remove internal-status wording such as prototype, preliminary, remaining experiments, and must be checked from the final manuscript.

### Execution Policy

- Use `conda run -n hslinets ...` for experiments.
- CUDA is available on RTX 4060 Laptop 8GB; run experiments serially.
- After each meaningful result, regenerate summaries, run targeted tests when code changes, commit, and push.
