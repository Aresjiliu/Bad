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

## 2026-07-16 formal seed0 routing findings

- Formal seed0 changes the evidence level: after 20 epochs plus 3 compact fine-tune epochs, profile metrics are usable for method debugging rather than only pipeline validation.
- The seed0 static-profile tradeoff is monotonic in the expected direction: static 65% gives the lowest mean MACs and lowest mean OA, static 100% gives the highest fixed-profile OA and highest cost.
- Oracle routing reaches mean OA 0.7501 at mean MACs 0.8835, compared with static 100% mean OA 0.7461 at mean MACs 0.9420. This is the first concrete evidence that quality-conditioned profile selection can improve the accuracy-efficiency-robustness tradeoff.
- Learned routing currently matches oracle on the same mode-level labels. This verifies trainability but does not prove generalization; the paper should either mark it as in-sample policy fitting or add leave-one-state-out / validation-label experiments.
- The oracle policy mostly selects 100% for degraded-but-available states, 80% for `aux_occlusion_50`, and 65% for `aux_only` / `main_only`, suggesting the current hand rule is conservative. Future work can learn less conservative labels from validation OA-resource utility.

## 2026-07-16 formal seed0/seed1 routing findings

- With seed0 and seed1 completed, static 80% is the current strongest fixed-profile baseline: 2-seed mean OA 0.7503 at mean MACs 0.7554.
- Hand oracle / learned oracle reach 2-seed mean OA 0.7391 at mean MACs 0.8915, so the hand rule should not be the final thesis routing policy.
- Utility-derived labels using `OA - 0.2 * MACs` reach 2-seed mean OA 0.7425 at mean MACs 0.6441. This is more compute-efficient but underperforms static 80% in mean OA.
- Learned utility labels reach 0.7346 mean OA at 0.6166 MACs; the current 11-state mode-level training set is too small for a strong learned router.
- The next defensible method upgrade is to define labels through a Pareto or constrained-utility rule, e.g. choose the lowest-MAC profile within an OA tolerance of the per-state best profile, then train/evaluate the router with held-out states.

## 2026-07-16 Pareto/utility sweep findings

- Pareto-tolerance labels are now implemented and tested. They directly address the hand-oracle problem by choosing the lowest-cost profile among profiles whose OA is within a tolerance of the per-state best profile.
- On the current seed0/seed1 sweep, `pareto_delta_0.01` gives 2-seed mean OA 0.7424 at 0.8550 MACs, with only about 0.0010 mean regret from the per-state best profile.
- `utility_lambda_0.05` gives a similar tradeoff at lower cost: 2-seed mean OA 0.7419 at 0.8394 MACs. Larger penalties such as `lambda=0.2` are too aggressive for the current evidence.
- The method still does not beat the old static 80% baseline on mean OA, so the correct thesis framing is not "dynamic routing already wins everywhere"; it is "hand-crafted routing fails, Pareto/utility labels provide a controlled resource-accuracy target, and learned routing should be trained against these corrected labels."
- The next evidence gap is seed2 plus a learned Pareto/utility router with leave-one-state-out validation.

## 2026-07-16 seed2 and 3-seed routing findings

- Seed2 is now complete for 65/80/100 formal routing profiles, closing the first 3-seed evidence gap on Houston2013.
- Under the current resource-stat backfill convention, static 100% has the highest fixed-profile 3-seed mean OA: 0.7483 at 1.0000 MACs. Static 80% is no longer the highest-OA fixed baseline in the unified 3-seed table, but remains an important lower-cost baseline at 0.7399 OA and 0.8036 MACs.
- Hand oracle / learned hand-oracle routing reach 0.7461 mean OA at 0.9257 MACs. This is close to static 100% while saving compute, but its rule is still heuristic and should not be the final label source.
- Pareto sweep gives the strongest current routing target. `pareto_delta_0.01` reaches 0.7537 mean OA at 0.8810 MACs, with only about 0.0010 mean regret from the per-state best profile and about 11.9% MAC saving versus 100%.
- The next implementation should train the learned router against `pareto_delta_0.01` labels and evaluate leave-one-state-out generalization. This is now more defensible than learning either hand oracle or aggressive `utility_lambda_0.2` labels.

## 2026-07-16 leave-one-state-out routing and multi-dataset findings

- Leave-one-state-out learned Pareto routing is implemented and tested. It trains the profile router on all but one degradation/missing state, predicts the held-out state, and reports selected OA, MACs, routing accuracy versus Pareto labels, regret, and saving.
- The three-seed LOO summary is mixed rather than conclusive: mean OA is 0.7464, mean MACs 0.8929, routing accuracy versus Pareto labels is 66.7%, mean regret is 0.0082, and mean saving is 10.7%. Seed2 over-selects the 100% profile, which removes compute saving.
- This is useful negative evidence. Offline Pareto labels are strong, but the current learned router is underdetermined because it only sees 11 state-level samples per seed. The paper should not yet claim a mature generalizing router; it should claim that Pareto labels define a better routing target, while learned routing needs patch-level or validation-derived training samples.
- Multi-dataset validation is now the main credibility gap. Based on local notes and HSLiNets materials, the priority should be Houston2013 + Trento + MUUFL. Trento is the first new dataset because it is small, standard, and closest to the current HS-LiDAR pipeline. MUUFL is second because its class imbalance and two LiDAR rasters create a more convincing stress test.
- Augsburg and Houston2018 should be deferred. Augsburg depends on reconstructed labels in at least one recent paper, while Houston2018 has extreme class imbalance and much larger scale. Doing either before Trento/MUUFL would likely consume time without improving the thesis fastest.

## 2026-07-16 downloaded dataset audit findings

- Downloaded Trento is structurally clean: HSI shape is 166x600x63, auxiliary data is 166x600x2, labels are 166x600 with six foreground classes and 30,214 labeled pixels. The issue is protocol rather than data integrity: no explicit train/test mask was detected.
- Trento can be adopted immediately for smoke experiments. Because the auxiliary file has two channels while many papers describe one DSM/LiDAR channel, the loader should expose `first`, `both`, and `mean` auxiliary-channel modes and the formal protocol must state which mode is used.
- Downloaded MUUFL contains a usable 64-band scene-label file: HSI shape is 325x220x64, scene labels cover 11 classes, and LiDAR objects expose 325x220x2 arrays. This matches the common 64-band MUUFL protocol more closely than the raw 72-band files.
- MUUFL should be adopted after Trento, but its imbalance is severe: class 1 has 23,246 pixels while class 10 has 183 and class 11 has 269, giving an imbalance ratio around 127. OA alone is not acceptable; AA, Kappa, and per-class accuracy must be included.
- Same-task local benchmark levels show Trento is near-saturated in recent HS-LiDAR papers: MTNet 99.21 OA, DFNet 99.01, CAMFNet 99.49. MUUFL is harder: MTNet 89.53, CAMFNet 83.24 under a small-sample/pretraining protocol, and MSFFRN 90.70 under a 64-band protocol.

## 2026-07-16 Trento smoke findings

- Trento is now integrated into the same runner path as Houston raw data through `--dataset trento`, so it can reuse the existing training, degradation evaluation, compact export, and resource accounting code.
- The seed0 fixed split follows the local protocol counts 129/125/105/154/184/122, giving 819 train pixels and 29,395 test pixels. This is a usable formal split, but the thesis must state it clearly because the downloaded package did not include an official train/test mask.
- A one-epoch CUDA smoke run with `aux_channel_mode=first` reached full-modality OA 0.9042, AA 0.7442, and Kappa 0.8719. This only proves the pipeline is correct; it is not competitive with Trento literature levels near 99% OA.
- The smoke result is diagnostically useful: HSI-only OA is 0.8600, auxiliary-only OA is 0.5881, downsampled auxiliary OA is 0.8851, and 50% auxiliary occlusion OA is 0.6050. This suggests Trento can expose robustness behavior, but formal claims need longer training and multi-seed repeats.
- Resource accounting works on Trento: the 100% smoke run reports 442,248 baseline parameters and 8.62M MACs. Because the target budget was 100%, compact export correctly keeps almost the full model.

## 2026-07-16 Trento formal seed0 findings

- Trento seed0 20-epoch baseline reaches source full OA 0.9947 and compact full OA 0.9924, which is close to local literature benchmark levels and confirms that the Trento protocol is viable.
- The 80% p=0.25 multi-degradation setting is unexpectedly strong on seed0: compact full OA is 0.9952 at 0.8001 MACs and 0.8037 Params ratio.
- The same 80% setting improves the harsh degraded-auxiliary diagnostics over the 100% baseline: high-noise OA 0.9957 vs 0.9840, downsample-4 OA 0.9922 vs 0.9820, and 50% occlusion OA 0.9938 vs 0.9712.
- The tradeoff is clear in aux-only: 80% p=0.25 drops to 0.5206 OA compared with 0.7120 for the 100% baseline. The thesis should frame aux-only as an extreme diagnostic state, while the main deployment objective is full/main-available plus degraded auxiliary robustness.
- Trento now gives a stronger multi-dataset story than expected: the current method is not merely transferring Houston trends; it can produce near-saturated accuracy with measurable MAC reduction on a second HS-LiDAR dataset. Seed1/2 are required before making a final claim.

## 2026-07-16 Trento 3-seed findings

- Trento 3-seed validation is now complete for 100% baseline and 80% p=0.25 multi-degradation. The 80% setting reaches 0.9894 +/- 0.0094 compact full OA at 0.7990 +/- 0.0010 MACs, while 100% baseline reaches 0.9910 +/- 0.0036 full OA.
- The correct claim is therefore not that 80% is uniformly more accurate. The defensible claim is that 80% preserves near-saturated Trento accuracy while reducing MACs by about 20%.
- Robustness evidence is favorable for degradation states: 80% p=0.25 improves downsample4 OA to 0.9851 vs 0.9795 and occlusion50 OA to 0.9816 vs 0.9646. High-noise is also slightly higher, 0.9890 vs 0.9854, but with larger variance.
- Latency is not yet aligned with MACs on Trento: compact full latency averages 2.16 ms for 80% vs 2.02 ms for 100%. This should be treated as an open deployment-benchmark issue, not hidden.
- This result is thesis-useful because it supplies a second dataset with 3-seed accuracy-efficiency-robustness evidence, but the paper should report it as a compactness/robustness tradeoff rather than a clean accuracy win.

## 2026-07-16 MUUFL onboarding findings

- MUUFL is now integrated through the same raw runner path as Houston2013 and Trento. The usable local protocol is the 64-band scene-label file with two LiDAR channels from the `z` cube; original `-1` labels are unmapped pixels and are converted to background `0`.
- Fixed seed0/1/2 splits are generated with 1550 train samples per seed: classes 1-9 use 150 samples each, classes 10-11 use 100 samples each. This keeps the rare classes in the training set while leaving 52137 test samples.
- The one-epoch CUDA smoke is not a formal result, but it confirms end-to-end viability: full OA 0.8080, AA 0.6973, Kappa 0.7498; main-only OA 0.6540; aux-only OA 0.3327; occlusion50 OA 0.7386.
- The short-run per-class results expose the key thesis issue: rare classes 9 and 10 remain near 0 accuracy, so MUUFL formal reporting must include AA, Kappa, and class accuracy. If 20-epoch training still fails on these classes, the next method-side addition should be weighted loss or a class-balanced sampler ablation.

## 2026-07-16 MUUFL 3-seed formal findings

- MUUFL 3-seed formal validation is complete for the 100% baseline and the 80% p=0.25 multi-degradation setting.
- The 100% baseline has higher clean full-modality OA: 88.51 +/- 1.19 versus 86.87 +/- 1.79 for the 80% setting. This should be reported honestly rather than hidden.
- The 80% p=0.25 setting gives the stronger robustness-efficiency story: main-only OA improves from 71.09 to 83.03, aux-only OA improves from 26.24 to 57.20, downsample4 OA improves from 84.12 to 86.66, and occlusion50 OA improves from 80.08 to 85.99.
- Resource reduction is stable on MUUFL: MACs are 79.97 +/- 0.07 and Params are 82.25 +/- 0.33 for the compact 80% setting.
- The correct thesis framing is now clear across datasets: Houston supports the routing/quality-aware method line, Trento supports near-saturated accuracy with about 20% MAC reduction, and MUUFL supports robustness under harder imbalance and degraded/missing auxiliary modalities.
- The next writing step is a unified multi-dataset table plus a MUUFL-specific per-class/AA/Kappa discussion, because OA alone is not credible on this class distribution.
- MUUFL per-class analysis shows that the 80% clean full-OA drop is mainly from classes 1, 3, 8, and 9, while classes 4, 5, 7, and 10 improve and class 11 remains essentially stable. This weakens the need for an immediate weighted-loss pivot; per-class reporting should come first, with weighted CE or class-balanced sampling kept as an optional ablation.

## 2026-07-17 paper-depth findings

- The paper now has a stronger three-dataset evidence structure: Houston2013 is the method/routing dataset, Trento is the near-saturated compression dataset, and MUUFL is the hard robustness dataset.
- A detailed proposed-method table across Houston2013, Trento, and MUUFL reports Full OA, Full AA, Kappa, main-only OA, aux-only OA, occlusion50 OA, MACs, and Params. This is more defensible than reporting only clean OA.
- The current experimental-depth gap is no longer "lack of multiple datasets"; it is now "lack of deeper per-dataset analysis." The most valuable additions are MUUFL confusion/class-delta visualization, weighted CE or class-balanced sampler seed0 ablation, and patch-level router supervision.
- The paper wording was adjusted to preserve negative evidence: MUUFL clean full OA drops under the 80% setting, Trento latency is not guaranteed to decrease despite MAC reduction, and leave-one-state-out learned routing is not yet mature.
