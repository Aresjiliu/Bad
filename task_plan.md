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
4. [completed] Extend quality degradation augmentation to occlusion and resolution-loss cases.
5. [completed] Compute Pearson/Spearman correlation between degradation severity and reliability/fusion/OA drop.
6. [completed] Generate degradation-curve figures and a Chinese analysis note.
7. [completed] Paper wording now emphasizes `degradation-aware reliability supervision` and avoids claiming generic reliability learning without degradation training.

### Phase E: Degradation-Aware Method Upgrade

1. [completed] Add `quality_degradation_supervised` to the experiment matrix.
2. [completed] Complete Houston official seeds 0/1/2 for `quality_degradation_supervised`.
3. [completed] Add multi-type degradation augmentation: Gaussian noise, random occlusion/block masking, and downsample-upsample resolution loss.
4. [completed] Add ablations for degradation type: none, noise-only, multi-type.
5. [completed] Complete Houston official seeds 0/1/2 for multi-type degradation supervision.
6. [completed] Tune degradation type proportions or curriculum to recover clean full-modality OA while keeping occlusion/downsample gains.
7. [completed] Update method writing around three contributions: structured compact export, availability-conditioned inference, and degradation-aware reliability supervision.

### Phase F: Robustness Tradeoff Tuning

1. [completed] Run a compact tuning grid for multi-type probability 0.25 vs 0.5.
2. [completed] Add `quality_multi_degradation_p025` to the formal experiment matrix.
3. [completed] Select the current thesis/paper default by average adverse-state OA first, then full-modality OA as a tie-breaker.
4. [pending] Add a weighted degradation sampler if more tuning is needed after paper table generation.
5. [completed] Generate a paper-ready table with baseline, uniform, noise-only, p=0.50 multi-type, and p=0.25 multi-type degradation supervision.
6. [completed] Update the Overleaf experiment section with the robustness ablation result.
7. [completed] Update method text to explain p=0.25 as a light degradation curriculum rather than an arbitrary hyperparameter.

### Phase D: Paper Figures and Writing

1. [completed] Generate retained-width visualization by branch and layer.
2. [completed] Generate AER Pareto plot: MAC/latency vs average OA over modality states.
2a. [completed] Generate degradation robustness and fusion-diagnostic figure for current Houston robustness ablations.
3. [completed] Update LaTeX Method/Experiment sections and add Algorithm 1/2.
4. [completed] Remove internal-status wording such as prototype, preliminary, remaining experiments, and must be checked from the final manuscript.
5. [completed] Add related-work positioning table to clarify the difference from generic pruning, missing-modality learning, and expert routing.
6. [completed] Revise Figure 1 resource-objective layout and Figure 2 overview styling according to manuscript-review feedback.

### Execution Policy

- Use `conda run -n hslinets ...` for experiments.
- CUDA is available on RTX 4060 Laptop 8GB; run experiments serially.
- After each meaningful result, regenerate summaries, run targeted tests when code changes, commit, and push.

## 2026-07-15 Thesis Extension and Demonstration Plan

### Goal

Upgrade BRM-Net from a small static budget and robustness prototype into a thesis-level framework that couples modality quality with deployable compute allocation, validates the method on multiple datasets, and closes the loop with an inference demonstration system.

### Phase G: Thesis-Level Method Upgrade

1. [completed] Audit current reusable code, experiment evidence, local thesis plans, and recent literature.
2. [completed] Define the quality-conditioned budget routing direction and its novelty boundary.
3. [completed] Specify the multi-dataset experiment matrix, deployment evidence, thesis structure, and stop conditions.
4. [completed] Specify the minimum demonstration system architecture, pages, APIs, and defense scenarios.
5. [pending] Freeze current Houston artifacts with model/config hashes and source-vs-compact evidence.
6. [pending] Introduce a larger clean dual-branch backbone and branch-independent fixed-dimension projections.
7. [pending] Build 65/80/100 deployable branch profiles and a low-cost pre-encoder quality probe.
8. [pending] Implement oracle and learned quality-conditioned budget routing.
9. [pending] Add Trento and MUUFL protocols, then run three-seed experiments.
10. [pending] Add ONNX Runtime benchmarks and the demonstration-system MVP.

### 2026-07-15 implementation status

- [completed] Add a standalone pre-encoder quality probe with availability-aware quality and uncertainty outputs.
- [completed] Add the optional BRM-Net diagnostic output path without changing default inference behavior.
- [completed] Add a fixed-dimension branch projection primitive for future independent-width encoders.
- [completed] Train/supervise the probe through existing quality targets and aggregate its diagnostics in train/eval metrics.
- [completed] Add a standalone quality-to-budget profile router with oracle labels and supervised routing loss.
- [completed] Integrate the router with deployable 65/80/100 profile evaluation and oracle profile reporting.
- [completed] Train learned profile selection from oracle labels and compare against static 65/80/100 profiles on formal seed0.
- [completed] Add utility-derived labels and complete formal seed1 routing-profile experiments.
- [completed] Sweep utility penalty and implement Pareto tolerance labels because the hand oracle is too conservative.
- [completed] Run formal 20-epoch seed2 routing-profile experiments and generate 3-seed summaries.
- [pending] Replace in-sample learned routing with leave-one-state-out or validation-derived Pareto/utility labels.
- [pending] Train learned routing against `pareto_delta_0.01` and report leave-one-state-out accuracy plus achieved OA/MAC.

### 2026-07-16 multi-dataset evidence plan

- [completed] Implement leave-one-state-out learned routing against `pareto_delta_0.01` labels and generate seed0/1/2 reports.
- [completed] Add a dataset-spec registry for Houston2013, Trento, MUUFL, Augsburg, and Houston2018.
- [completed] Produce a dataset feasibility and download plan for thesis-level multi-dataset validation.
- [completed] After the user downloads Trento, implement `load_trento_scene` and a generic raw-loader path.
- [completed] Run Trento `--dataset-only` checks, then a smoke training run.
- [completed] After Trento is stable, download and integrate MUUFL with explicit 64-band scene-label protocol handling.

### 2026-07-16 downloaded dataset audit and Trento onboarding

- [completed] Audit downloaded Trento and MUUFL files for shape, finite values, label distribution, class count, imbalance, and split availability.
- [completed] Collect local same-task benchmark levels for Houston2013, Trento, and MUUFL.
- [completed] Implement `load_trento_scene` with `first|both|mean` auxiliary-channel handling.
- [completed] Generate fixed Trento seed0/1/2 coordinate splits from the downloaded labels.
- [completed] Add Trento raw-loader factory support and wire `--dataset trento` into the training runner.
- [completed] Run Trento `--dataset-only` checks and a one-epoch CUDA smoke experiment.
- [pending] Generalize `HoustonPatchDataset` naming or add `MultimodalPatchDataset` after Trento/MUUFL loaders stabilize.
- [completed] Implement MUUFL loader after Trento smoke is stable.

### 2026-07-16 Trento smoke-to-formal transition

- [completed] Validate Trento fixed split seed0: 819 train samples and 29,395 test samples.
- [completed] Validate Trento training/evaluation/compact-export path with `aux_channel_mode=first`.
- [completed] Run Trento seed0 formal baseline with 20 epochs at 100% budget.
- [completed] Run Trento seed0 80% budget with the current balanced degradation-aware setting.
- [pending] Decide whether Trento formal protocol should keep `first` DSM/LiDAR channel or compare `first|both|mean` as a small protocol ablation.
- [completed] Run Trento seed1/seed2 for both 100% baseline and 80% p=0.25 multi-degradation setting.
- [completed] Generate a Trento 3-seed summary table for the thesis and paper.
- [pending] Add Trento 3-seed results into the LaTeX experiment table after deciding the exact multi-dataset table layout.

### 2026-07-16 MUUFL onboarding and smoke validation

- [completed] Implement `load_muufl_scene` for the 64-band scene-label file, two-channel LiDAR `z` cube, and `-1` background labels.
- [completed] Add MUUFL raw-loader factory support and wire `--dataset muufl` into the runner.
- [completed] Generate fixed MUUFL seed0/1/2 coordinate splits with 1550 train samples and 52137 test samples per seed.
- [completed] Validate MUUFL seed0 `--dataset-only` on the downloaded data.
- [completed] Run a one-epoch CUDA smoke experiment for MUUFL seed0.
- [completed] Run MUUFL seed0/seed1/seed2 20-epoch 100% baseline.
- [completed] Run MUUFL seed0/seed1/seed2 80% p=0.25 multi-degradation setting.
- [completed] Generate MUUFL 3-seed formal summary tables and a Chinese thesis-interpretation note.
- [completed] Add MUUFL per-class accuracy summary to handle class imbalance explicitly.
- [completed] Build one unified Houston2013/Trento/MUUFL multi-dataset evidence table in Markdown/CSV.
- [completed] Convert the multi-dataset evidence table into LaTeX table snippets.
- [pending] Decide whether to add a MUUFL class-balanced sampler or weighted-loss ablation after checking thesis table space.

### 2026-07-17 paper content and experiment-depth plan

- [completed] Revise the external LaTeX paper to include multi-dataset validation in the abstract, introduction, experiments, discussion, and conclusion.
- [completed] Add a detailed multi-dataset table with Full OA, AA, Kappa, missing-modality diagnostics, occlusion robustness, MACs, and Params.
- [completed] Add MUUFL per-class table to the paper to avoid relying only on OA under severe class imbalance.
- [completed] Compile the paper with TeX Live 2026 and verify that the PDF builds.
- [completed] Generate MUUFL confusion matrix and class-delta visualization for a deeper failure-mode figure.
- [completed] Run MUUFL weighted CE seed0/seed1/seed2 ablation and summarize it as a diagnostic rather than a core method.
- [completed] Add the MUUFL weighted CE ablation table and interpretation into the external LaTeX paper.
- [completed] Write the latest research-status and next-plan document.
- [pending] Convert Pareto routing labels from state-level evidence to patch-level or validation-derived training samples.

### 2026-07-17 next stable implementation plan

- [completed] Design validation-derived Pareto routing samples from existing profile/degradation evaluation records.
- [completed] Implement a validation-state-derived routing dataset with seed/state quality features, Pareto labels, and label-stability diagnostics.
- [completed] Train a small router on validation-derived labels and evaluate achieved OA/MAC/regret with held-out seeds.
- [completed] Implement mean-profile stable Pareto labels from seed-averaged profile metrics.
- [completed] Add the stable-label router to the routing policy comparison table.
- [pending] Implement a true patch-level routing dataset that stores quality-probe features, availability/degradation state, confidence, selected profile label, and oracle regret.
- [completed] Stabilize the routing label rule at state level; patch-level remains future work.
- [completed] Add a paper-ready routing table comparing static 80%, static 100%, Pareto oracle, LOO router, validation-derived router, and stable-label router.
- [pending] Prepare a lightweight demonstration-system MVP driven by existing JSON metrics and generated figures.

### 2026-07-17 review-guided concrete action plan

- [completed] Translate the external review guidance into a concrete local action plan: `docs/REVIEW_GUIDANCE_CONCRETE_ACTION_PLAN_20260717_ZH.md`.
- [pending] Add an experiment-manifest freezer that records git commit, environment, config, metrics hashes, and compact-export evidence.
- [pending] Add a manuscript internal-term scanner and use it before each paper build.
- [pending] Implement reviewer-critical ablations: `soft-mask-only`, `w/o availability mask`, `w/o degradation quality`, and `uniform width scaling`.
- [pending] Add `quality_target_mode=fixed|continuous|rank` and run a Houston2013 seed0 calibration check.
- [pending] Generate continuous degradation curves and predicted-quality calibration plots.
- [pending] Dump patch-level routing samples with quality, uncertainty, confidence, correctness, profile oracle label, and regret.
- [pending] Compare static profiles, Pareto oracle, stable-label state router, and patch-level router in one policy table.
- [pending] Rebuild Figure 1/Figure 2 around Training, Export, Inference, AER Pareto, degradation curve, quality calibration, and failure analysis.
- [pending] Build a JSON-backed demonstration MVP from existing metrics and generated figures.

### 2026-07-17 paper-first split-track plan

- [completed] Read the dual-goal split-track guidance and freeze the near-term objective as paper submission first.
- [completed] Create the paper-first scope plan: `docs/PAPER_FIRST_TRACK_SUBMISSION_PLAN_20260717_ZH.md`.
- [completed] Slim the submission manuscript so the main text only argues compact export, fusion compatibility, and degradation-supervised reliability.
- [completed] Move routing, profile-router comparisons, weighted CE, and deep MUUFL class diagnostics out of the submission main text.
- [completed] Create a paper claims-evidence matrix and canonical result source for all main tables/figures.
- [in_progress] Build the submission ablation table around soft-mask-only, compact export, uniform width/fusion baseline, w/o availability, and w/o degradation reliability.
  - [completed] Add the source-gated versus physically exported compact-model table to support the compact-export claim.
  - [completed] Add a runnable `without_fusion_availability_mask` ablation path for the availability-mask claim.
  - [completed] Run the new w/o fusion availability mask ablation for seed0/1/2; it is strong enough as a missing-modality fusion diagnostic.
  - [pending] Decide whether to run strict uniform-width ablation or reword it as a limitation.
- [pending] Generate or refresh controlled-corruption reliability curves for the submission.
- [pending] Recompile the submission PDF and run a readiness check before adding any thesis-only work.

### 2026-07-18 paper-first ablation update

- [completed] Implement strict `uniform_width_export` as a target-matched fixed-width compact export strategy.
- [completed] Run Houston2013 seed0/seed1/seed2 for `uniform_width_export` under the 80% p=0.25 multi-degradation setting.
- [completed] Regenerate `docs/generated/brmnet_priority_summary.csv/.md` and the paper claims/evidence matrix.
- [completed] Add `docs/generated/uniform_width_export_ablation.csv/.md` and the external LaTeX table `tables/uniform_width_export_ablation.tex`.
- [completed] Insert the uniform-width export ablation into the submission experiment section and compile the paper successfully.
- [pending] Generate or refresh controlled-corruption reliability curves for the submission.
- [pending] Run the final submission readiness scan after the reliability-curve pass.
