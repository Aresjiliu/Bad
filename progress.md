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
