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

## 2026-07-10 resources and current experimental state

- `get-available-resources` script failed in `hslinets` because `psutil` is not installed; avoid repeating that exact command unless dependency is installed.
- PyTorch 2.5.1 in `hslinets` reports CUDA available.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total; snapshot before experiments showed about 4868 MiB free.
- Execution strategy: run training experiments serially, not in parallel.
- `without_reliability_uniform_fusion` seed0 completed successfully and produced real latency fields in `resource_stats.json`.
- `without_reliability_uniform_fusion` seeds 0/1/2 are complete. Compact 3-seed OA: full 0.8672 +/- 0.0060, main_only 0.7635 +/- 0.0192, aux_only 0.4352 +/- 0.0319, aux_noise 0.8422 +/- 0.0327, aux_noise_high 0.7701 +/- 0.0350.
- Uniform fusion reaches acceptable full-modality OA but weak missing-modality OA. This is useful negative evidence: the reliability module should be justified by missing/noisy modality robustness, not by full-modality accuracy alone.
- Current `full` reliability baseline in the summary is older and lacks current latency fields for all seeds, so the next clean comparison requires rerunning/refreshed reliability-gated seeds with the current profiling code.

## 2026-07-10 degradation-aware reliability findings

- The refreshed `full` reliability-gated baseline is not sufficient by itself. Under compact export, 3-seed OA is 0.8697 for full modality but only 0.7208 for `aux_noise_high`; the learned auxiliary quality and fusion weights do not reliably suppress severely corrupted auxiliary inputs.
- Uniform fusion is a strong negative control rather than a strawman. It reaches compact full OA 0.8672 and `aux_noise_high` OA 0.7701, so the paper cannot claim that an unsupervised reliability module is automatically better than average fusion.
- Direct `quality_supervised` with only missing/non-missing targets fails on degraded-but-available inputs. In seed0, `aux_noise_high` has `q_aux` 0.9999999, auxiliary fusion weight 0.5001, and compact OA 0.6929. This proves that quality supervision must explicitly include degraded available modalities.
- `quality_degradation_supervised` is the current strongest direction. With noise-degraded available auxiliary samples during training, compact 3-seed `aux_noise_high` OA improves to 0.8515, compared with 0.7208 for the refreshed full baseline and 0.7701 for uniform fusion. The auxiliary fusion weight also drops to 0.3966 under high noise, showing that the routing behavior changes in the intended direction.
- The same variant keeps normal full-modality compact OA at 0.8745 and `aux_noise` OA at 0.8655, so the robustness gain is not obtained by simply sacrificing clean-modality performance.
- Limitation: the current quality-degradation augmentation is noise-only. It does not consistently solve occlusion or downsampling; `aux_occlusion_50` compact OA is 0.7332 and `aux_downsample_4` compact OA is 0.8014. The next method step should add multi-type degradation targets for noise, occlusion, and resolution loss.

## 2026-07-10 multi-type degradation findings

- Multi-type degradation supervision closes the previous occlusion/downsample gap. Compact 3-seed OA reaches 0.8438 on `aux_downsample_4` and 0.8466 on `aux_occlusion_50`, compared with 0.8014 and 0.7332 for noise-only degradation supervision.
- It keeps high-noise robustness essentially unchanged: compact `aux_noise_high` OA is 0.8511, close to the noise-only result 0.8515 and clearly above full baseline 0.7208 / uniform 0.7701.
- It produces clearer degradation-sensitive routing for occlusion and resolution loss. Under `aux_occlusion_50`, `q_aux` is 0.5480 and auxiliary fusion weight is 0.3891; under `aux_downsample_4`, `q_aux` is 0.6617 and auxiliary fusion weight is 0.4169.
- The tradeoff is clean full-modality performance: compact full OA is 0.8599, lower than noise-only degradation supervision 0.8745 and the refreshed full baseline 0.8697. This means the next improvement should tune the degradation schedule, not simply increase augmentation diversity.
- Paper framing should present the current result as a robustness-efficiency tradeoff: multi-type degradation improves adverse-modality states at about 80% MACs, while a curriculum or weighted degradation sampler is needed to recover clean full-modality accuracy.

## 2026-07-10 light multi-type degradation findings

- Reducing multi-type degradation probability from 0.50 to 0.25 gives the best current balance. Compact full OA recovers to 0.8690, close to the refreshed full baseline 0.8697 and above the p=0.50 multi-type result 0.8599.
- The adverse-state average over `aux_noise_high`, `aux_occlusion_50`, and `aux_downsample_4` is 0.8476 for p=0.25, slightly above p=0.50 multi-type 0.8472 and far above the refreshed full baseline 0.7489.
- p=0.25 is especially strong for occlusion: compact `aux_occlusion_50` OA is 0.8557, compared with 0.8466 for p=0.50, 0.7332 for noise-only quality degradation, and 0.7421 for the full baseline.
- The remaining weakness is high-noise routing calibration. p=0.25 has `q_aux` 0.9086 and auxiliary fusion weight 0.4772 under `aux_noise_high`, so it relies more on the auxiliary branch than p=0.50. This keeps clean accuracy but may reduce robustness on some seeds.
- Current default recommendation for the paper experiment table: use `quality_multi_degradation_p025` as the balanced main method, and report noise-only and p=0.50 multi-type as ablations that expose the robustness-clean-accuracy tradeoff.

## 2026-07-15 thesis extension findings

- The current codebase is reusable and should not be restarted: the clean mainline already contains structured budget learning, compact export, availability-conditioned fusion, degradation supervision, three-seed reporting, and 129 tests.
- The main thesis risk is conceptual coupling rather than missing modules. Current quality estimates affect fusion after feature extraction, so they cannot change the computation already spent by the encoders.
- Recent work including SimMLM, MaMOL, and DCMNet makes generic dynamic routing or missing-modality conditional computation an unsafe novelty claim.
- The most defensible new direction is a low-cost pre-encoder quality probe plus a router over a finite set of dense, exportable branch-width profiles. This distinguishes the project through resource constraints, real deployment, and degradation calibration.
- A first implementation should use separately exportable 65/80/100 profiles and oracle routing labels before attempting a shared-weight slimmable supernet.
- Thesis-level evidence requires at least Houston2013, Trento, and MUUFL Gulfport, three seeds, fixed split manifests, missing/degradation matrices, calibration metrics, and ONNX Runtime latency.
- The demonstration system should be an algorithm and deployment validation platform, not a claimed satellite production system. Core interactions are modality switches, degradation injection, budget/profile selection, backend selection, spatial predictions, and routing diagnostics.

## 2026-07-15 quality-routed foundation implementation

- `PreEncoderQualityProbe` now estimates raw-input quality and uncertainty before the expensive encoders. It uses depthwise-plus-pointwise branches and treats unavailable modalities as quality 0 / uncertainty 1.
- BRMNet exposes probe outputs only through the opt-in `use_pre_encoder_quality_probe` flag. The values are deliberately not connected to fusion or loss yet, so current metrics and resource accounting remain valid.
- `FeatureProjection` is available for future branch-independent encoder widths. Equal-width projections use `nn.Identity`, so the default path has no added parameters.
- The next implementation batch must add quality targets/metrics and a runner switch before GPU experiments; running the current training loop would not yet test the new hypothesis.

## 2026-07-15 pre-encoder quality supervision findings

- The pre-encoder probe is now trainable through the existing degradation/missing-modality quality targets. Its uncertainty head is supervised toward `1 - quality`, giving the thesis a clearer reliability-estimation story than quality-only scoring.
- Default behavior is preserved: with `--lambda-pre-quality 0`, the probe is not instantiated and the Houston dry-run remains at 466673 parameters. With `--lambda-pre-quality 0.5 --pre-encoder-quality-hidden 8`, parameters rise to 469174.
- The one-epoch CUDA pilot confirms that metrics are recorded end to end, but does not prove routing quality. Full test OA is 0.4010 and pre-encoder quality predictions remain close to 0.5, so the probe needs longer training before it can drive profile selection.
- The next thesis-level step should use the probe as an explicit routing input: first oracle labels over fixed 65/80/100 profiles, then a learned router with an added routing loss and calibration plots.

## 2026-07-15 quality-to-budget routing findings

- A finite-profile router is now represented in code, which is important for thesis defensibility: it avoids claiming an abstract dynamic network before there are deployable fixed profiles.
- The first oracle rule is intentionally coarse. It is suitable for bootstrapping experiments and calibration plots, but should not be overclaimed as an optimal policy.
- The next technical risk is profile materialization: current hard-concrete export can create target budgets, but the routing code still needs a run-level interface that evaluates or loads 65/80/100 profile checkpoints and reports oracle-vs-learned selection.

## 2026-07-15 routing profile pilot findings

- The 65/80/100 profile pipeline now runs end to end on HSLiNets Houston legacy data with CUDA in the `hslinets` conda environment.
- The short seed0 pilots confirm budget controllability: source expected MACs are about 0.654/0.802/1.000 for 65/80/100 targets, and compact MACs are about 0.649/0.799/1.000.
- The current oracle routing report is useful as a pipeline validation artifact, not as a thesis result. Compact OA is low because the pilots use only 3 epochs and no compact fine-tuning.
- The most useful near-term comparison is no longer only fixed-budget pruning. The thesis table should compare static profile selection, oracle profile routing, and learned quality-conditioned profile routing under the same missing/degradation states.
- The next implementation should train a learned router on oracle or validation-derived labels, then report routing accuracy, average MACs, average OA, and calibration curves.
