# 进度记录

## 2026-07-05

- 创建持久化计划文件，用于本轮“进一步分析并调研，确定下一步方向”。
- 下一步将读取已有中文报告与核心代码，再进行外部论文调研。
- 已读取当前实验报告、架构审查、改进方向调研、论文架构升级文档、投稿工程文档。
- 已核查核心代码：当前已有 MQE/RGF、Hard-Concrete、compact export，但缺少模态可用性训练闭环。
- 已补充外部调研：missing modality survey、RingMoE、MAPEX、MaMOL、高光谱压缩 benchmark、星上推理综述、会议截稿信息。
- 已输出决策文档 `docs/NEXT_DIRECTION_DECISION_ZH.md`。
- 本地提交已完成；推送 GitHub 两次失败，错误分别为无法连接 443 端口和连接被重置。

## 2026-07-06

- 检查投稿工程 `D:\Academic\paper_submission\brmnet_pricai2026`。
- 新增 Overleaf 初稿表格 `tables/prototype_hslidar_structured.tex`。
- 更新 `sections/00_abstract.tex`、`01_intro.tex`、`02_related_work.tex`、`03_method.tex`、`04_experiments.tex`、`05_discussion.tex`、`06_conclusion.tex`。
- 更新 `notes/experiments.md`，新增 `notes/overleaf_status.md`。
- 生成 Overleaf 上传包 `D:\Academic\paper_submission\brmnet_pricai2026_overleaf_initial_20260706.zip`。
- 匿名检查通过；TODO 统计为 6；本机未检测到 `latexmk` 或 `pdflatex`，暂未本地编译 PDF。
## 2026-07-06 多种子实验

- 运行 Houston2013-HS-LiDAR official split 结构化剪枝 seed 1/2，与 seed 0 组成 3 seed 统计。
- 生成 `docs/generated/structured_pruning_multiseed_summary.md` 和 `docs/generated/structured_pruning_multiseed_runs.csv`。
- 更新 Overleaf prototype 表格为 3 seed mean +/- std，并在实验章节明确它是机制验证而非最终主表。
- 新增 `docs/PAPER_EXPERIMENT_UPDATE_20260706_ZH.md`，记录结果、论文写法和下一步优先级。
- 实现 availability mask + modality dropout：RGF mask softmax、BRMNet/CompactBRMNet forward mask、engine degradation mask、training-time modality dropout、Houston runner `--modality-dropout-prob`。
- 验证：`python -m unittest discover -s tests` 通过 101 个测试，1 个跳过；`scripts/run_brmnet_houston.py --dry-run --modality-dropout-prob 0.25 --device cpu` 通过。
- 运行 80% budget seed0 dropout 0.25 初步实验：compact full OA 88.20，main-only OA 80.00，aux-only OA 43.50；对应 no-dropout compact 为 86.90、49.75、24.13。

## 2026-07-10 continuous improvement session

- Read existing `task_plan.md`, `findings.md`, and `progress.md`.
- Appended a new continuous improvement plan covering core ablations, source-vs-compact reporting, degradation calibration, and paper figures/writing.
- Resource detection via skill script failed due missing `psutil`; switched to PyTorch/nvidia-smi checks.
- Confirmed CUDA availability on RTX 4060 Laptop GPU.
- Next action: run `without_reliability_uniform_fusion` seed1 and seed2 serially, then regenerate summary and commit results.

## 2026-07-10 experiment progress

- Completed `without_reliability_uniform_fusion` seed1 with current CUDA/latency profiling.
- Next action: run seed2, then regenerate `docs/generated/brmnet_priority_summary.*` and inspect 3-seed uniform-fusion ablation.
- Completed `without_reliability_uniform_fusion` seed2 and regenerated `docs/generated/brmnet_priority_summary.csv` / `.md`.
- Uniform-fusion ablation now has 3 seeds with real full/main_only/aux_only latency. This closes the first core ablation loop.
- Next action: refresh the `full` reliability-gated baseline under the current profiler so reliability-vs-uniform comparisons use comparable latency/resource fields.

## 2026-07-10 degradation-aware quality supervision

- Refreshed the `full` reliability-gated baseline for seeds 0/1/2 with the current profiler and regenerated `docs/generated/brmnet_priority_summary.*`.
- Added `quality_supervised` as a first check of direct quality loss. Seed0 showed that missing/non-missing supervision alone is not enough: `q_aux` stayed close to 1.0 under noisy auxiliary input and compact `aux_noise_high` OA dropped to 0.6929.
- Implemented train-time degraded-but-available auxiliary quality augmentation through `apply_aux_quality_degradation`, plus runner arguments and experiment-matrix support.
- Added `quality_degradation_supervised` to the core matrix and completed seeds 0/1/2 on Houston official split.
- Regenerated `docs/generated/brmnet_priority_matrix.csv`, `docs/generated/run_brmnet_priority_matrix.ps1`, `docs/generated/brmnet_priority_summary.csv`, and `docs/generated/brmnet_priority_summary.md`.
- Verified the code with `conda run -n hslinets python -m unittest discover -s tests`: 122 tests passed, 1 skipped.
- Next action: extend the degradation-aware supervision from noise-only to multi-type degradation, especially downsampling and occlusion, then convert the result into the paper method and ablation table.

## 2026-07-10 multi-type degradation supervision

- Added `--aux-quality-degradation-types` with support for `noise`, `downsample_2`, `downsample_4`, `occlusion_25`, and `occlusion_50`.
- Fixed PowerShell command generation for comma-separated degradation types in `scripts/make_brmnet_experiment_matrix.py`.
- Added `quality_multi_degradation_supervised` to the core matrix and regenerated `docs/generated/brmnet_priority_matrix.csv` / `run_brmnet_priority_matrix.ps1`.
- Completed Houston official seeds 0/1/2 for `quality_multi_degradation_supervised`.
- Regenerated `docs/generated/brmnet_priority_summary.csv` and `.md`; summary now covers 59 protocol/mode groups.
- Verified with `conda run -n hslinets python -m unittest discover -s tests`: 125 tests passed, 1 skipped.
- Next action: tune degradation curriculum/proportions because multi-type improves occlusion/downsample robustness but slightly lowers clean full-modality OA.

## 2026-07-10 light multi-type schedule

- Ran `quality_multi_degradation_p025` seeds 0/1/2 using multi-type degradation probability 0.25.
- Regenerated `docs/generated/brmnet_priority_summary.csv` and `.md`; summary now covers 70 protocol/mode groups.
- Added `quality_multi_degradation_p025` to the formal core experiment matrix; regenerated `docs/generated/brmnet_priority_matrix.csv` and `docs/generated/run_brmnet_priority_matrix.ps1`.
- Verified with `conda run -n hslinets python -m unittest discover -s tests`: 125 tests passed, 1 skipped.
- Current recommendation: treat `quality_multi_degradation_p025` as the balanced main candidate because it gives adverse-state average OA 0.8476 while preserving full OA 0.8690.

## 2026-07-10 paper table update

- Added `scripts/make_brmnet_paper_tables.py` to export the robustness ablation table from `docs/generated/brmnet_priority_summary.csv`.
- Generated `docs/generated/brmnet_robustness_table.csv` and `.md`.
- Generated Overleaf table `D:\Academic\paper_submission\brmnet_pricai2026\tables\robustness_ablation.tex`.
- Updated Overleaf experiment section `D:\Academic\paper_submission\brmnet_pricai2026\sections\04_experiments.tex` to cite the new three-seed robustness ablation and remove outdated preliminary wording.
- Verified local LaTeX build with `D:\texlive\2026\bin\windows\latexmk.exe -pdf -interaction=nonstopmode -halt-on-error paper.tex`.

## 2026-07-10 manuscript claim alignment

- Updated Overleaf abstract, introduction, experiments, discussion, and conclusion to align the paper claim with the new three-seed robustness ablation.
- Reframed the main technical claim as degradation-aware reliability supervision for degraded-but-available auxiliary modalities, not generic reliability weighting.
- Added the key paper-facing result: `quality_multi_degradation_p025` reaches 84.76% adverse-state average OA at 80.01% MACs while preserving 86.90% full-modality OA.
- Rebuilt the Overleaf manuscript successfully with `D:\texlive\2026\bin\windows\latexmk.exe -pdf -interaction=nonstopmode -halt-on-error paper.tex`.

## 2026-07-10 degradation reliability diagnostics

- Added `scripts/analyze_brmnet_degradation_reliability.py` to convert the priority summary into degradation diagnostics, correlation tables, and a publication-style multi-panel figure.
- Generated `docs/generated/brmnet_degradation_reliability.md`, `.csv` diagnostics, `.csv` correlations, `.pdf`, and `.png`.
- Main interpretation: multi-type degradation supervision improves adverse-state OA, but fusion weights are not fully calibrated quality scores; the paper should claim robust degradation-aware training rather than automatic monotonic suppression of every degraded modality.

## 2026-07-10 review-guided manuscript revision

- Applied the guidance from `D:\Download\BRM-Net_全面评审与调研指导意见.md` to the Overleaf manuscript.
- Retitled the paper around budgeted compact fusion and degradation-calibrated reliability.
- Rewrote the contribution statement to emphasize joint budget/availability/degradation formulation, fusion-compatible compact export, degradation-calibrated reliability fusion, and accuracy-efficiency-robustness evaluation.
- Added a related-work positioning table and Algorithm 1 for training plus compact export.
- Removed visible internal-status wording from the main manuscript path, including thesis-log, server reconciliation, target reference, and prototype-oriented claims.
- Added the degradation reliability diagnostic figure to the Overleaf project and rebuilt `paper.pdf` successfully with TeX Live 2026.

## 2026-07-11 figure revision pass

- Reworked the Overleaf Figure 1 TikZ layout so the resource objective is centered between the two channel-gate branches, with longer dashed objective arrows and more vertical separation between modality streams.
- Updated the overview result figure generator to use more distinct markers in Panel A, simplify the accuracy-efficiency legend, and move Panel C/D legends away from data-heavy regions.
- Regenerated `fig_brmnet_results_overview.*` for the Overleaf project and rebuilt `paper.pdf` successfully with TeX Live 2026.
- Verified the plotting-code change with `conda run -n hslinets python -m unittest discover -s tests`: 129 tests passed, 1 skipped.

## 2026-07-14 full pipeline documentation

- Added `docs/BRMNET_FULL_PIPELINE_ZH.md` as a clean Chinese runbook for the current `brmnet_core` and `scripts` mainline.
- Documented the end-to-end flow from Houston raw data loading, source training, degradation evaluation, hard-concrete compact export, compact fine-tuning, multi-seed summarization, and paper table/figure generation.
- Included Mermaid flowcharts, core module explanations, recommended commands, key parameter tables, output-file meanings, and paper-claim boundaries.

## 2026-07-15 thesis innovation and demonstration planning

- Audited the current BRM-Net method, experiment evidence, reusable legacy backbone assets, local thesis blueprints, and recent 2024-2026 literature.
- Identified the core gap: reliability currently changes fusion but does not control encoder computation, so budget and robustness remain only loosely coupled.
- Created `docs/THESIS_EXTENSION_AND_DEMO_PLAN_ZH.md` with a quality-conditioned budget routing design, multi-dataset experiment matrix, deployment plan, demonstration-system architecture, thesis chapter mapping, ten-week roadmap, and stop conditions.
- Added the implementation roadmap to `task_plan.md` and recorded the literature-driven decisions in `findings.md`.

## 2026-07-15 quality-routed foundation implementation

- Added `brmnet_core/quality_probe.py` with `PreEncoderQualityProbe`, including availability-aware quality and uncertainty outputs.
- Added the optional `use_pre_encoder_quality_probe` BRMNet path and four diagnostic output fields: `pre_q_main`, `pre_q_aux`, `pre_u_main`, and `pre_u_aux`.
- Added `brmnet_core/projection.py` with `FeatureProjection`; equal input/output widths are parameter-free identities.
- Followed red-green tests for both new modules: missing-module/unsupported-argument failures were observed before their implementations.
- Fresh verification: `conda run -n hslinets python -m unittest discover -s tests` completed with 134 tests passing and 1 skipped; the Houston runner CPU dry-run also completed successfully.

## 2026-07-15 pre-encoder quality supervision

- Added `pre_encoder_quality_loss`, wired `lambda_pre_quality` into `brmnet_loss`, and exposed `pre_quality`, `pre_q_*`, and `pre_u_*` in train/eval metrics.
- Added Houston runner switches `--lambda-pre-quality` and `--pre-encoder-quality-hidden`; the pre-encoder probe is enabled only when `lambda_pre_quality > 0`.
- Verified red-green behavior: missing `pre_encoder_quality_loss` and missing CLI arguments failed before implementation, then targeted tests passed.
- Fresh verification: `conda run -n hslinets python -m unittest discover -s tests` completed with 137 tests passing and 1 skipped.
- Ran a one-epoch CUDA pilot on HSLiNets Houston legacy patches at `output/quality_probe_pilot/...`: train `pre_quality=0.8769`, validation `pre_quality=0.8858`, validation OA `0.4382`, full test OA `0.4010`.
- Interpretation: the training/evaluation chain is now real, but a one-epoch pilot is only a smoke experiment. The probe outputs remain near 0.5 under many degraded states, so formal claims require longer training and routing ablations.

## 2026-07-15 quality-to-budget router primitive

- Added `brmnet_core/profile_router.py` with `QualityBudgetRouter`, `oracle_budget_profile_targets`, and `budget_profile_routing_loss`.
- The router maps four pre-encoder diagnostics `[pre_q_main, pre_q_aux, pre_u_main, pre_u_aux]` to 65/80/100 budget-profile probabilities, expected budget, selected profile, and selected budget.
- The oracle target function provides the first supervised labels: clean low-uncertainty states map to the full profile, severe low-quality/high-uncertainty states map to the low-budget profile, and intermediate states map to the middle profile.
- Verified with `conda run -n hslinets python -m unittest tests.test_profile_router` and full `conda run -n hslinets python -m unittest discover -s tests`: 140 tests passed and 1 skipped.

## 2026-07-15 routing profile evaluation workflow

- Added `evaluate_budget_profile_routing` and the CLI `scripts/evaluate_budget_profile_routing.py` to combine 65/80/100 compact profile metrics with pre-encoder quality features.
- Added resource-stat backfill so compact metrics with zero `expected_macs_ratio` are corrected from sibling `resource_stats.json`.
- Extended `scripts/make_brmnet_experiment_matrix.py` with `--ablation routing_profiles`, `--data-format`, and correct multi-token Python command generation for `conda run -n hslinets python`.
- Generated `docs/generated/brmnet_routing_profile_matrix.csv` and `docs/generated/run_brmnet_routing_profiles.ps1`.
- Completed three short Houston legacy seed0 profile pilots at 65/80/100 target MAC budgets under the quality-routing setting.
- Generated `docs/generated/brmnet_routing_profile_oracle_seed0.json` and `.csv`; the pilot oracle report covers 11 modality/degradation states with mean selected budget 0.7727 and mean expected MACs ratio 0.7361.
- Added `docs/QUALITY_ROUTING_PROFILE_PROGRESS_ZH.md` to summarize the method logic, pilot result boundary, and next formal experiment plan.

## 2026-07-16 learned routing and formal seed0 profile experiments

- Added reusable learned-router helpers: `train_quality_budget_router` and `predict_budget_profile_selection`.
- Extended `scripts/evaluate_budget_profile_routing.py` to produce static 65/80/100, oracle routing, and learned routing comparison reports.
- Generated the formal routing profile matrix for Houston2013: 3 budgets x 3 seeds, 20 epochs, 3 compact fine-tune epochs.
- Completed formal seed0 runs for 65%, 80%, and 100% profiles under the quality-routing setting.
- Generated `docs/generated/brmnet_routing_profile_formal_comparison_seed0.csv` and `.json`.
- Formal seed0 comparison: static 65% mean OA 0.7158 at 0.6121 MACs; static 80% 0.7231 at 0.7572 MACs; static 100% 0.7461 at 0.9420 MACs; oracle/learned routing 0.7501 at 0.8835 MACs.
- Completed formal seed1 runs for 65%, 80%, and 100% profiles.
- Added utility-derived budget labels through `select_utility_budget_profiles` and comparison-script support for `utility` / `learned_utility`.
- Generated `docs/generated/brmnet_routing_profile_formal_comparison_seed1.*` and `docs/generated/brmnet_routing_profile_formal_seed0_seed1_summary.csv`.
- Two-seed status: static 80% is the strongest fixed baseline so far; hand oracle is too conservative; utility labels reduce compute but require penalty/Pareto tuning.

## 2026-07-16 Pareto/utility routing sweep

- Added `select_pareto_tolerance_budget_profiles` to choose the lowest-MAC profile within a per-state OA tolerance of the best available profile.
- Added `build_sweep_reports` and CLI switches `--output-sweep-json`, `--output-sweep-csv`, `--pareto-tolerances`, and `--utility-resource-penalties`.
- Generated seed0/seed1 Pareto and utility sweeps:
  - `docs/generated/brmnet_routing_profile_formal_sweep_seed0.*`
  - `docs/generated/brmnet_routing_profile_formal_sweep_seed1.*`
  - `docs/generated/brmnet_routing_profile_formal_sweep_seed0_seed1_summary.csv`
- Current 2-seed sweep finding: `pareto_delta_0.01` reaches mean OA 0.7424 at 0.8550 MACs with about 0.0010 mean regret; `utility_lambda_0.05` reaches mean OA 0.7419 at 0.8394 MACs.
- Added `docs/PARETO_UTILITY_ROUTING_UPDATE_20260716_ZH.md` to summarize the strategy correction and next implementation steps.

## 2026-07-16 formal seed2 completion and 3-seed routing summary

- Completed formal seed2 runs for 65%, 80%, and 100% routing profiles on Houston2013 in the `hslinets` CUDA environment.
- Generated seed2 reports:
  - `docs/generated/brmnet_routing_profile_formal_comparison_seed2.*`
  - `docs/generated/brmnet_routing_profile_formal_oracle_seed2.*`
  - `docs/generated/brmnet_routing_profile_formal_sweep_seed2.*`
- Regenerated seed0/seed1 comparison reports with the current resource-stat backfill path so all three seeds use the same reporting convention.
- Generated 3-seed summaries:
  - `docs/generated/brmnet_routing_profile_formal_seed0_seed1_seed2_summary.csv`
  - `docs/generated/brmnet_routing_profile_formal_sweep_seed0_seed1_seed2_summary.csv`
- 3-seed comparison status: static 100% reaches mean OA 0.7483 at 1.0000 MACs; hand oracle / learned oracle reach 0.7461 at 0.9257 MACs; static 80% reaches 0.7399 at 0.8036 MACs.
- 3-seed sweep status: `pareto_delta_0.01` reaches mean OA 0.7537 at 0.8810 MACs with mean regret about 0.0010 and mean saving about 0.1190 versus 100% profile.

## 2026-07-16 leave-one-state-out routing and multi-dataset planning

- Implemented leave-one-state-out Pareto routing report generation in `scripts/evaluate_budget_profile_routing.py`.
- Added tests for the LOO report path and CLI output path in `tests/test_evaluate_budget_profile_routing.py`.
- Generated LOO reports for formal seed0/seed1/seed2 and the summary file `docs/generated/brmnet_routing_profile_formal_loo_pareto_delta0p01_seed0_seed1_seed2_summary.csv`.
- LOO 3-seed result: mean OA 0.7464, mean MACs 0.8929, routing accuracy versus Pareto labels 66.7%, mean regret 0.0082, mean saving 0.1071.
- Added `brmnet_core/data/dataset_specs.py` and `tests/test_dataset_specs.py` to record the priority dataset registry before real Trento/MUUFL files are available.
- Created `docs/MULTI_DATASET_FEASIBILITY_AND_DOWNLOAD_PLAN_ZH.md` with the recommended download order: Trento first, MUUFL second, Augsburg/Houston2018 deferred.
- Verification passed: `conda run -n hslinets python -m unittest discover -s tests` ran 158 tests, all passed with 1 skipped.

## 2026-07-16 downloaded dataset audit and Trento onboarding

- Located the downloaded datasets at `D:\Academic\data\Trento-main` and `D:\Academic\data\MUUFLGulfport-master`.
- Added `brmnet_core/data/dataset_audit.py` and `scripts/audit_downloaded_datasets.py`.
- Generated `docs/DOWNLOADED_DATASET_AUDIT_ZH.md` and `docs/generated/downloaded_dataset_audit.json`.
- Added `docs/NEW_DATASETS_ADOPTION_AND_BENCHMARK_ZH.md` with same-task benchmark levels and adoption decisions.
- Implemented `brmnet_core/data/trento.py` with `load_trento_scene` and auxiliary-channel modes `first`, `both`, and `mean`.
- Added `scripts/make_multidataset_splits.py` and generated fixed Trento splits:
  - `output/splits/trento_seed0.npz`
  - `output/splits/trento_seed1.npz`
  - `output/splits/trento_seed2.npz`
- Trento split protocol uses per-class training counts 129/125/105/154/184/122, producing 819 train samples and 29,395 test samples per seed.
- Verification passed: `conda run -n hslinets python -m unittest discover -s tests` ran 164 tests, all passed with 1 skipped.

## 2026-07-16 Trento runner integration and smoke run

- Added a Trento raw-loader factory that reuses normalized patch extraction and records split hashes, per-class train/test counts, patch size, and auxiliary-channel mode.
- Extended `scripts/run_brmnet_houston.py` with `--dataset houston2013|trento` and `--aux-channel-mode first|both|mean`.
- Verified that Trento `--dataset-only` uses 63 HSI channels plus the selected auxiliary channel and writes a `trento_...` run directory rather than a Houston-labelled output.
- Ran a real Trento seed0 dataset-only check on `D:\Academic\data\Trento-main` using `output\splits\trento_seed0.npz`.
- Ran a one-epoch CUDA smoke training run on Trento with 100% target budget and no compact fine-tuning. The run completed end to end and produced metrics, compact metrics, config, history, and resource stats under `output\trento_smoke\...`.
- Added `docs/TRENTO_SMOKE_RUN_20260716_ZH.md` to record the smoke result and the formal-experiment boundary.
- Current smoke metrics: full OA 0.9042, AA 0.7442, Kappa 0.8719; main-only OA 0.8600; auxiliary-only OA 0.5881; auxiliary occlusion 50% OA 0.6050.
- Next action: run a Trento 20-epoch 100% seed0 baseline, then compare the current 80% degradation-aware/lightweight setting.

## 2026-07-16 Trento formal seed0 experiments

- Completed Trento seed0 20-epoch 100% baseline under CUDA. Source full OA is 0.9947; compact full OA is 0.9924; compact MACs ratio is 1.0000.
- Completed Trento seed0 20-epoch 80% p=0.25 multi-degradation run with 10 compact fine-tune epochs. Compact full OA is 0.9952; compact MACs ratio is 0.8001; compact Params ratio is 0.8037.
- The 80% p=0.25 run improves compact high-noise/downsample/occlusion diagnostics relative to the 100% baseline, while aux-only drops substantially.
- Added `docs/TRENTO_FORMAL_SEED0_RESULTS_20260716_ZH.md` to record the formal seed0 comparison and thesis-writing interpretation.
- Push to GitHub was attempted after the Trento runner commit, but GitHub port 443 was unreachable from the current environment. Local branch remains ahead of origin.

## 2026-07-16 Trento 3-seed closure

- Completed Trento seed1 and seed2 for both 100% baseline and 80% p=0.25 multi-degradation, closing the first Trento 3-seed comparison.
- Generated `docs/generated/trento_formal_seed0_seed1_seed2_runs.csv`, `docs/generated/trento_formal_seed0_seed1_seed2_summary.csv`, and `.md`.
- 3-seed compact full OA: 100% baseline 0.9910 +/- 0.0036; 80% p=0.25 multi-degradation 0.9894 +/- 0.0094.
- 3-seed compact MACs: 100% baseline 1.0000; 80% p=0.25 multi-degradation 0.7990 +/- 0.0010.
- 3-seed robustness: 80% p=0.25 improves downsample4 OA from 0.9795 to 0.9851 and occlusion50 OA from 0.9646 to 0.9816, but compact latency does not yet improve.

## 2026-07-16 MUUFL onboarding and smoke run

- Implemented `brmnet_core/data/muufl.py` for the MUUFL 64-band scene-label file, including two-channel LiDAR `z` cube extraction and `-1` to background label mapping.
- Added MUUFL raw-loader factory support and extended `scripts/run_brmnet_houston.py` with `--dataset muufl`.
- Extended `scripts/make_multidataset_splits.py` and generated fixed MUUFL splits:
  - `output/splits/muufl_seed0.npz`
  - `output/splits/muufl_seed1.npz`
  - `output/splits/muufl_seed2.npz`
- MUUFL split protocol uses 150 samples for classes 1-9 and 100 samples for classes 10-11, producing 1550 train samples and 52137 test samples per seed.
- Verified real MUUFL seed0 `--dataset-only` on `D:\Academic\data\MUUFLGulfport-master` with channels `[64, 2]`.
- Ran a one-epoch CUDA smoke run under `output\muufl_smoke\...`; full OA 0.8080, AA 0.6973, Kappa 0.7498, main-only OA 0.6540, aux-only OA 0.3327, occlusion50 OA 0.7386.
- Added `docs/MUUFL_SMOKE_RUN_20260716_ZH.md` to record the protocol, command, metrics, and formal-experiment boundary.

## 2026-07-16 MUUFL formal 3-seed experiments

- Completed MUUFL seed0/seed1/seed2 for both 100% baseline and 80% p=0.25 multi-degradation settings.
- Generated:
  - `docs/generated/muufl_formal_seed0_seed1_seed2_runs.csv`
  - `docs/generated/muufl_formal_seed0_seed1_seed2_summary.csv`
  - `docs/generated/muufl_formal_seed0_seed1_seed2_summary.md`
  - `docs/MUUFL_FORMAL_SEED0_SEED1_SEED2_RESULTS_20260716_ZH.md`
- 3-seed compact full OA: 100% baseline 88.51 +/- 1.19; 80% p=0.25 multi-degradation 86.87 +/- 1.79.
- 3-seed compact MACs: 100% baseline 100.00%; 80% p=0.25 multi-degradation 79.97 +/- 0.07%.
- Robustness gain is substantial on MUUFL: 80% p=0.25 improves main-only OA by 11.94 points, aux-only OA by 30.96 points, downsample4 OA by 2.54 points, and occlusion50 OA by 5.91 points.
- Current conclusion: MUUFL should be used as the hard-data robustness case, not as a pure clean-OA win.
- Generated MUUFL per-class summaries under `docs/generated/muufl_formal_per_class_seed0_seed1_seed2_*`; the main clean full-OA losses are classes 3 and 9, while classes 4, 5, 7, and 10 improve under the 80% multi-degradation setting.

## 2026-07-17 paper and experiment-depth update

- Updated the external LaTeX paper at `D:\Academic\paper_submission\brmnet_pricai2026` to include Houston2013, Trento, and MUUFL as a three-dataset evidence chain.
- Revised the abstract, introduction contribution list, experiment section, discussion, and conclusion to frame the method as an accuracy-efficiency-robustness trade-off rather than a clean-accuracy-only method.
- Added paper tables:
  - `D:\Academic\paper_submission\brmnet_pricai2026\tables\multidataset_formal_evidence.tex`
  - `D:\Academic\paper_submission\brmnet_pricai2026\tables\multidataset_detailed_proposed.tex`
  - `D:\Academic\paper_submission\brmnet_pricai2026\tables\muufl_per_class_accuracy.tex`
- Generated repository-side detailed table files:
  - `docs/generated/multidataset_detailed_proposed_summary.csv`
  - `docs/generated/multidataset_detailed_proposed_summary.md`
  - `docs/generated/multidataset_detailed_proposed_table.tex`
- Added `docs/EXPERIMENT_DEPTH_AND_PAPER_REVISION_20260717_ZH.md` to record the current paper logic, remaining experimental-depth gaps, and next-priority ablations.
- Verified the paper with `D:\texlive\2026\bin\windows\latexmk.exe`; `paper.pdf` compiled successfully with no fatal errors, undefined references, or overfull boxes.

## 2026-07-17 MUUFL error analysis figure

- Added `scripts/plot_muufl_error_analysis.py` to aggregate MUUFL compact full-modality confusion matrices and class-wise accuracy deltas across three seeds.
- Added `tests/test_plot_muufl_error_analysis.py`; targeted tests passed.
- Generated:
  - `docs/generated/muufl_error_analysis.md`
  - `docs/generated/muufl_error_analysis_per_class.csv`
  - `docs/generated/muufl_error_analysis.pdf`
  - `docs/generated/muufl_error_analysis.png`
- Synced the figure to the external paper directory as `D:\Academic\paper_submission\brmnet_pricai2026\figures\generated\fig_muufl_error_analysis.pdf`.
- Updated the paper experiment section to cite the MUUFL error-analysis figure as evidence that robustness training redistributes clean full-modality class accuracy rather than uniformly improving every category.

## 2026-07-17 MUUFL weighted CE ablation and research plan

- Added class-weighted loss support through `brmnet_core.losses.brmnet_loss` and `scripts/run_brmnet_houston.py --class-weighting inverse_frequency`.
- Completed MUUFL weighted CE seed0/seed1/seed2 runs under the existing 80% MAC, p=0.25 multi-degradation setting.
- Added `scripts/summarize_muufl_weighted_ce.py` and `tests/test_summarize_muufl_weighted_ce.py`.
- Generated:
  - `docs/generated/muufl_weighted_ce_seed0_seed1_seed2_runs.csv`
  - `docs/generated/muufl_weighted_ce_seed0_seed1_seed2_summary.csv`
  - `docs/generated/muufl_weighted_ce_seed0_seed1_seed2_class_delta.csv`
  - `docs/generated/muufl_weighted_ce_seed0_seed1_seed2_summary.md`
- Updated the external LaTeX paper with `tables/muufl_weighted_ce_ablation.tex` and a conservative interpretation paragraph in `sections/04_experiments.tex`.
- Verified the paper with TeX Live 2026; `paper.pdf` compiled successfully with no undefined references, overfull boxes, fatal errors, or LaTeX errors detected by the final log grep.
- Added `docs/RESEARCH_STATUS_AND_NEXT_PLAN_20260717_ZH.md` to record the latest research status, current thesis positioning, and next implementation priorities.

## 2026-07-17 validation-derived Pareto routing

- Added `scripts/evaluate_validation_derived_routing.py` to turn Houston2013 65/80/100 profile results into seed/state routing samples.
- Added `tests/test_evaluate_validation_derived_routing.py`.
- Generated:
  - `docs/generated/brmnet_validation_derived_pareto_routing.json`
  - `docs/generated/brmnet_validation_derived_pareto_routing_summary.csv`
  - `docs/generated/brmnet_validation_derived_pareto_routing_samples.csv`
  - `docs/generated/brmnet_validation_derived_pareto_routing_label_stability.csv`
  - `docs/generated/brmnet_validation_derived_pareto_routing.md`
- The first validation-derived router uses 33 seed/state samples. It reaches mean OA 0.7459 at mean MACs 0.8622, compared with the earlier state-level LOO router mean OA 0.7464 at mean MACs 0.8929.
- Label stability analysis shows that Pareto targets are unstable across seeds for most states; only `aux_only` keeps the same 0.65 target across all three seeds.
- Added `--label-strategy mean_profile_pareto` to derive stable labels from seed-averaged profile metrics.
- Generated `docs/generated/brmnet_validation_mean_profile_pareto_routing.*`.
- The mean-profile stable-label router reaches mean OA 0.7519 at mean MACs 0.7343, with routing accuracy 0.7273 against the stable target labels. This is now the strongest learned routing result.

## 2026-07-17 paper-first split-track replanning

- Read `D:\Download\BRM-Net_毕业设计与学术投稿双目标分轨规划.md`.
- Confirmed the near-term objective should switch from thesis-wide expansion to paper submission first.
- Added `docs/PAPER_FIRST_TRACK_SUBMISSION_PLAN_20260717_ZH.md`.
- Updated `task_plan.md` with a paper-first split-track phase.
- Updated `findings.md` with the key split-track findings.
- Slimmed the external submission manuscript at `D:\Academic\paper_submission\brmnet_pricai2026` by removing routing-policy, weighted-CE, and deep MUUFL class-diagnostic content from the main text.
- Recompiled `paper.pdf` successfully with TeX Live 2026 after the slimming pass.
- Added `scripts/make_paper_submission_evidence.py` and `tests/test_make_paper_submission_evidence.py`.
- Generated `docs/generated/paper_canonical_results.csv` and `docs/PAPER_CLAIMS_EVIDENCE_MATRIX_20260717_ZH.md`.
- Regenerated the submission multi-dataset table so Houston2013 now reports the static 80% multi-degradation compact setting rather than Pareto routing.
- Recompiled the external submission PDF successfully after the table correction; log scan found no fatal, undefined, or overfull-box matches.
- Added `docs/generated/paper_source_compact_table.csv/.md` and the external LaTeX table `tables/source_vs_compact_export.tex`.
- Inserted the source-gated versus physically exported compact-model comparison into the submission experiments section and recompiled `paper.pdf` successfully.
- Local branch already had one unpushed commit from the previous plan update because GitHub connection reset during push.

## 2026-07-17 fusion availability-mask ablation

- Added `--disable-fusion-availability-mask` for the reviewer-critical `w/o availability mask` ablation.
- The switch disables only the fusion softmax availability mask while preserving branch-level availability-conditioned computation and modality-dropout/degradation generation, so the ablation isolates whether masked reliability fusion is necessary.
- Preserved the switch in `BRMNet`, `CompactBRMNet`, compact export metadata, the Houston runner dry-run summary, and the generated priority experiment matrix.
- Regenerated `docs/generated/brmnet_priority_matrix.csv` and `docs/generated/run_brmnet_priority_matrix.ps1`; the new `without_fusion_availability_mask` variant is scheduled for seed0/1/2.
- Verified with targeted unit tests and a seed0 dry-run using `conda run -n hslinets`.
- Ran Houston2013 seed0/1/2 for the new ablation on CUDA and refreshed `docs/generated/brmnet_priority_summary.csv/.md`.
- Added `docs/generated/fusion_availability_mask_ablation_seed0_seed1_seed2_summary.md`. The three-seed result shows main-only compact OA drops from 80.04 +/- 1.93 to 76.72 +/- 2.97 when fusion masking is disabled, while full and degraded-but-available states remain close.

## 2026-07-18 uniform-width export ablation

- Added a strict `uniform_width_export` compact-export strategy. It keeps a target-matched uniform width ratio in every gated block, instead of using the learned nonuniform hard-concrete channel pattern.
- Wired the strategy through `export_compact_brmnet`, `scripts/run_brmnet_houston.py --compact-export-strategy uniform_width`, and the generated priority experiment matrix.
- Ran Houston2013 seed0/seed1/seed2 on CUDA under the 80% p=0.25 multi-degradation setting.
- Refreshed `docs/generated/brmnet_priority_summary.csv/.md` and generated `docs/generated/uniform_width_export_ablation.csv/.md`.
- Updated `scripts/make_paper_submission_evidence.py` so the uniform-width evidence and LaTeX table are generated from the canonical summary.
- Inserted `tables/uniform_width_export_ablation.tex` into the external submission manuscript after the source-vs-compact table.
- Verified related unit tests and compiled `D:\Academic\paper_submission\brmnet_pricai2026\paper.pdf` with TeX Live 2026; the final log scan found no undefined references, LaTeX errors, fatal errors, or overfull boxes.

## 2026-07-18 controlled-corruption reliability curves

- Added `scripts/plot_brmnet_reliability_curves.py` and `tests/test_plot_brmnet_reliability_curves.py`.
- Generated controlled-corruption reliability curve data and notes:
  - `docs/generated/brmnet_controlled_corruption_reliability_curves.csv`
  - `docs/generated/brmnet_controlled_corruption_reliability_curves.md`
  - `docs/generated/brmnet_controlled_corruption_reliability_curves.pdf`
  - `docs/generated/brmnet_controlled_corruption_reliability_curves.png`
- Synced the paper figure copies to:
  - `D:\Academic\paper_submission\brmnet_pricai2026\figures\generated\fig_controlled_corruption_reliability_curves.pdf`
  - `D:\Academic\paper_submission\brmnet_pricai2026\figures\generated\fig_controlled_corruption_reliability_curves.png`
- Updated the external submission experiment section so Figure 3 now reports compact OA, auxiliary reliability, and auxiliary fusion weight across noise, resolution-loss, and occlusion severity.
- Verified targeted tests and recompiled the external paper with TeX Live 2026; the final log scan found no undefined references, LaTeX errors, fatal errors, citation warnings, or overfull boxes.

## 2026-07-19 long-term plan and submission readiness gate

- Added `docs/BRMNET_LONG_TERM_DUAL_TRACK_PLAN_20260719_ZH.md`.
- The long-term route now has two explicit tracks: paper submission first, then thesis extension with routing, diagnostics, and demo-system work.
- Added `scripts/scan_paper_submission_readiness.py` and `tests/test_scan_paper_submission_readiness.py`.
- The readiness scanner checks manuscript scope terms, internal process wording, over-strong reliability claims, missing table/figure files, undefined references, LaTeX log warnings/errors, and canonical evidence row status.
- Generated:
  - `docs/PAPER_SUBMISSION_READINESS_CHECK_20260719_ZH.md`
  - `docs/generated/paper_submission_readiness_findings.csv`
- Re-ran the scanner against the current external submission paper after confirming `paper.pdf` is up to date. Current status is PASS with 0 P0, 0 P1, and 0 P2 findings.

## 2026-07-19 advisor-facing submission report

- Added `scripts/make_advisor_submission_report.py` and `tests/test_make_advisor_submission_report.py`.
- Generated `docs/ADVISOR_SUBMISSION_STATUS_REPORT_20260719_ZH.md`.
- The report summarizes the paper track positioning, current PDF/readiness status, budget/export/fusion/reliability evidence, multi-dataset evidence, reliability-curve interpretation, risks, and questions for advisor decision.
- Re-ran `scripts/scan_paper_submission_readiness.py`; the current submission remains PASS with 0 findings.

## 2026-07-19 related-work citation audit

- Added `scripts/audit_paper_related_work.py` and `tests/test_audit_paper_related_work.py`.
- Generated `docs/RELATED_WORK_CITATION_AUDIT_20260719_ZH.md`.
- The audit checks citation density by section, topic keyword coverage in related work, cited keys missing from BibTeX, and unused BibTeX entries.
- Initial audit exposed two unused structural-pruning references: `guo2020dmcp` and `fang2023depgraph`.
- Updated the external submission related-work section at `D:\Academic\paper_submission\brmnet_pricai2026\sections\02_related_work.tex` so DMCP and DepGraph are explicitly used in the compact-export / structured-pruning positioning.
- Re-ran the audit after the writing patch. The paper now uses all 26 BibTeX entries, with 41 citation mentions and 26 unique cited keys in `02_related_work.tex`.
- Recompiled `D:\Academic\paper_submission\brmnet_pricai2026\paper.pdf` with TeX Live 2026 and re-ran the readiness scanner. Current submission status remains PASS with 0 findings.

## 2026-07-19 submission title/abstract/introduction polish

- Updated the external submission title to `Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification`.
- Revised `sections/00_abstract.tex` to state that the final model is physically compact, that export preserves terminal fusion dimensions, and that reliability estimates are diagnostic fusion signals rather than guaranteed physical quality calibration.
- Revised `sections/01_intro.tex` to add the concrete gap between soft training-time compression and deployable compact export, plus the multimodal constraint that branch pruning must preserve fusion compatibility.
- Generated `docs/PAPER_SUBMISSION_WRITING_POLISH_20260719_ZH.md` to track these external-paper edits from the main code repository.
- Recompiled the external paper with TeX Live 2026; the PDF remains 10 pages and the final hard-error log scan is clean.
- Re-ran `scripts/scan_paper_submission_readiness.py`; current status remains PASS with 0 findings.
- Completed a follow-up terminology pass across related work, method, experiments, discussion, and conclusion so the paper consistently says `fusion-compatible compact export` and `degradation-supervised reliability`.
- Replaced potentially confusing `routing` language in the submission main text with `availability-aware fusion` or `fusion weights`, while keeping dynamic profile routing explicitly outside the submission scope.
- Recompiled the external paper again; the latest PDF is `407697` bytes and the hard-error log scan remains clean.

## 2026-07-19 advisor review brief

- Added `docs/ADVISOR_REVIEW_BRIEF_20260719_ZH.md`.
- The brief summarizes the current submission track, compact-export evidence, fusion-compatibility evidence, availability-mask ablation, degradation-supervised reliability evidence, multi-dataset role assignment, and advisor decision questions.
- The brief also proposes the next decision: either enter final language/package polishing if the advisor accepts the current scope, or add a larger-backbone transfer / stronger lightweight baseline if innovation is still judged insufficient.

## 2026-07-19 submission package checklist

- Added `scripts/check_submission_package.py` and `tests/test_check_submission_package.py`.
- Generated:
  - `docs/SUBMISSION_PACKAGE_CHECKLIST_20260719_ZH.md`
  - `docs/generated/submission_package_checklist.csv`
- The checklist verifies required files, required directories, PDF size, section/table/figure inventory, LaTeX input references, graphics references, anonymous author block, local-path/identity tokens in manuscript TeX, and build-log hard-error patterns.
- Current package status is PASS with 0 findings: all referenced inputs and graphics exist, `Anonymous Authors` is present, no local path or identity token is detected in manuscript TeX, and the build log has no hard-error pattern matches.

## 2026-07-19 clean source package

- Added `scripts/make_submission_source_package.py` and `tests/test_make_submission_source_package.py`.
- Generated the current clean LaTeX source package candidate:
  - `D:\Academic\paper_submission\brmnet_pricai2026_submission_source_20260719.zip`
- Generated:
  - `docs/SUBMISSION_SOURCE_PACKAGE_MANIFEST_20260719_ZH.md`
  - `docs/generated/submission_source_package_manifest.csv`
- The source package contains 20 compile-essential files: `paper.tex`, `.latexmkrc`, bibliography, 7 section files, 8 referenced table files, and 2 referenced PDF figures.
- The package intentionally excludes build artifacts, preview images, notes, review pages, sources metadata, pycache files, and thesis-only tables/figures that are not referenced by the main manuscript.
- Verified that the package can be extracted and compiled independently with TeX Live 2026.
- Updated external submission `README.md` and `AGENTS.md` to remove stale PRICAI/LNAI, ICTAI, old title, and old dataset descriptions.
