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
