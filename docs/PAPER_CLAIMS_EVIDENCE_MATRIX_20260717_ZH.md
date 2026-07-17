# BRM-Net 投稿版 Claims-Evidence Matrix（2026-07-17）

本文件服务投稿优先路线，只记录主文 compact export、fusion compatibility、degradation-supervised reliability 相关证据。Routing、weighted CE、MUUFL 深度类别诊断转入毕业论文或补充材料。

## 主张矩阵

| Claim | 投稿位置 | 状态 | 证据 |
|---|---|---|---|
| C1: BRM-Net exports physically compact models whose actual MAC ratios match the requested budgets. | Table 1 / Figure 2 | ready | E_BUDGET_65, E_BUDGET_80, E_BUDGET_90 |
| C2: The exported compact model is evaluated separately from the source gated model, supporting the physical-export claim. | Table 1 / planned compact-vs-source table | ready_for_table | E_SOURCE_COMPACT_FULL, E_SOURCE_COMPACT_MAIN_ONLY, E_SOURCE_COMPACT_AUX_ONLY |
| C3: Availability-aware training improves fallback behavior under missing single-modality inference. | Table 2 | ready | E_DROPOUT_MAIN_ONLY, E_DROPOUT_AUX_ONLY |
| C4: Degradation-supervised reliability improves robustness under noisy, low-resolution, and occluded auxiliary inputs. | Table 3 / Figure 3 | ready | E_ROBUST_FULL_BASELINE, E_ROBUST_UNIFORM, E_ROBUST_NOISE_ONLY, E_ROBUST_MULTI_P025 |
| C5: The compact degradation-aware setting transfers to external HSI-LiDAR scenes as an accuracy-efficiency-robustness trade-off. | Table 4 | ready | E_MULTI_HOUSTON, E_MULTI_TRENTO, E_MULTI_MUUFL |
| G1: Uniform-width compression, w/o availability mask, and w/o fusion-compatible terminal constraint remain incomplete or need stricter naming. | Readiness checklist | needs_ablation_or_rewording | GAP_UNIFORM_WIDTH, GAP_NO_AVAILABILITY, GAP_TERMINAL_TIE |

## Canonical Evidence

| Evidence | Claim | Dataset | Variant | Mode | Metric | Value | Source | Status | Note |
|---|---|---|---|---|---|---:|---|---|---|
| E_BUDGET_65 | C1 | Houston2013 | budgeted compact export | full | actual_macs_ratio | 64.96 +/- 0.12 | `docs/generated/structured_pruning_multiseed_runs.csv` | ready | Target MAC ratio 0.65; compact OA 85.53 +/- 2.20%. Params 64.79 +/- 1.93%. |
| E_BUDGET_80 | C1 | Houston2013 | budgeted compact export | full | actual_macs_ratio | 80.01 +/- 0.10 | `docs/generated/structured_pruning_multiseed_runs.csv` | ready | Target MAC ratio 0.80; compact OA 85.84 +/- 0.92%. Params 79.92 +/- 0.37%. |
| E_BUDGET_90 | C1 | Houston2013 | budgeted compact export | full | actual_macs_ratio | 90.07 +/- 0.08 | `docs/generated/structured_pruning_multiseed_runs.csv` | ready | Target MAC ratio 0.90; compact OA 85.74 +/- 0.88%. Params 89.78 +/- 0.84%. |
| E_SOURCE_COMPACT_FULL | C2 | Houston2013 | quality_multi_degradation_p025 | full | oa | 86.90 +/- 1.65 | `docs/generated/brmnet_priority_summary.csv` | ready_for_table | Source-vs-compact metrics are in the same summary row; source OA is 86.15%. |
| E_SOURCE_COMPACT_MAIN_ONLY | C2 | Houston2013 | quality_multi_degradation_p025 | main_only | oa | 80.04 +/- 1.93 | `docs/generated/brmnet_priority_summary.csv` | ready_for_table | Source-vs-compact metrics are in the same summary row; source OA is 78.18%. |
| E_SOURCE_COMPACT_AUX_ONLY | C2 | Houston2013 | quality_multi_degradation_p025 | aux_only | oa | 38.04 +/- 0.84 | `docs/generated/brmnet_priority_summary.csv` | ready_for_table | Source-vs-compact metrics are in the same summary row; source OA is 38.43%. |
| E_DROPOUT_MAIN_ONLY | C3 | Houston2013 | full | main_only | oa | 76.62 +/- 0.89 | `docs/generated/brmnet_priority_summary.csv` | ready | Compare with without_modality_dropout main_only row; full 3-seed refresh for no-dropout remains incomplete. |
| E_DROPOUT_AUX_ONLY | C3 | Houston2013 | full | aux_only | oa | 43.87 +/- 3.19 | `docs/generated/brmnet_priority_summary.csv` | ready | Compare with without_modality_dropout aux_only row; full 3-seed refresh for no-dropout remains incomplete. |
| E_ROBUST_FULL_BASELINE_FULL | C4 | Houston2013 | full | full | oa | 86.97 +/- 0.71 | `docs/generated/brmnet_priority_summary.csv` | ready | Full baseline |
| E_ROBUST_FULL_BASELINE_AUX_NOISE_HIGH | C4 | Houston2013 | full | aux_noise_high | oa | 72.08 +/- 1.45 | `docs/generated/brmnet_priority_summary.csv` | ready | Full baseline |
| E_ROBUST_FULL_BASELINE_AUX_DOWNSAMPLE_4 | C4 | Houston2013 | full | aux_downsample_4 | oa | 78.37 +/- 1.91 | `docs/generated/brmnet_priority_summary.csv` | ready | Full baseline |
| E_ROBUST_FULL_BASELINE_AUX_OCCLUSION_50 | C4 | Houston2013 | full | aux_occlusion_50 | oa | 74.21 +/- 5.64 | `docs/generated/brmnet_priority_summary.csv` | ready | Full baseline |
| E_ROBUST_UNIFORM_FULL | C4 | Houston2013 | without_reliability_uniform_fusion | full | oa | 86.72 +/- 0.60 | `docs/generated/brmnet_priority_summary.csv` | ready | Uniform fusion baseline; not a uniform-width baseline. |
| E_ROBUST_UNIFORM_AUX_NOISE_HIGH | C4 | Houston2013 | without_reliability_uniform_fusion | aux_noise_high | oa | 77.01 +/- 3.50 | `docs/generated/brmnet_priority_summary.csv` | ready | Uniform fusion baseline; not a uniform-width baseline. |
| E_ROBUST_UNIFORM_AUX_DOWNSAMPLE_4 | C4 | Houston2013 | without_reliability_uniform_fusion | aux_downsample_4 | oa | 78.35 +/- 2.05 | `docs/generated/brmnet_priority_summary.csv` | ready | Uniform fusion baseline; not a uniform-width baseline. |
| E_ROBUST_UNIFORM_AUX_OCCLUSION_50 | C4 | Houston2013 | without_reliability_uniform_fusion | aux_occlusion_50 | oa | 73.87 +/- 2.55 | `docs/generated/brmnet_priority_summary.csv` | ready | Uniform fusion baseline; not a uniform-width baseline. |
| E_ROBUST_NOISE_ONLY_FULL | C4 | Houston2013 | quality_degradation_supervised | full | oa | 87.45 +/- 0.95 | `docs/generated/brmnet_priority_summary.csv` | ready | Noise-only quality supervision. |
| E_ROBUST_NOISE_ONLY_AUX_NOISE_HIGH | C4 | Houston2013 | quality_degradation_supervised | aux_noise_high | oa | 85.15 +/- 1.45 | `docs/generated/brmnet_priority_summary.csv` | ready | Noise-only quality supervision. |
| E_ROBUST_NOISE_ONLY_AUX_DOWNSAMPLE_4 | C4 | Houston2013 | quality_degradation_supervised | aux_downsample_4 | oa | 80.14 +/- 2.98 | `docs/generated/brmnet_priority_summary.csv` | ready | Noise-only quality supervision. |
| E_ROBUST_NOISE_ONLY_AUX_OCCLUSION_50 | C4 | Houston2013 | quality_degradation_supervised | aux_occlusion_50 | oa | 73.32 +/- 2.40 | `docs/generated/brmnet_priority_summary.csv` | ready | Noise-only quality supervision. |
| E_ROBUST_MULTI_P025_FULL | C4 | Houston2013 | quality_multi_degradation_p025 | full | oa | 86.90 +/- 1.65 | `docs/generated/brmnet_priority_summary.csv` | ready | Selected light multi-degradation schedule. |
| E_ROBUST_MULTI_P025_AUX_NOISE_HIGH | C4 | Houston2013 | quality_multi_degradation_p025 | aux_noise_high | oa | 84.55 +/- 2.82 | `docs/generated/brmnet_priority_summary.csv` | ready | Selected light multi-degradation schedule. |
| E_ROBUST_MULTI_P025_AUX_DOWNSAMPLE_4 | C4 | Houston2013 | quality_multi_degradation_p025 | aux_downsample_4 | oa | 84.15 +/- 1.46 | `docs/generated/brmnet_priority_summary.csv` | ready | Selected light multi-degradation schedule. |
| E_ROBUST_MULTI_P025_AUX_OCCLUSION_50 | C4 | Houston2013 | quality_multi_degradation_p025 | aux_occlusion_50 | oa | 85.57 +/- 1.64 | `docs/generated/brmnet_priority_summary.csv` | ready | Selected light multi-degradation schedule. |
| E_MULTI_HOUSTON2013 | C5 | Houston2013 | 80% p=0.25 multi-degradation compact model | full | full_oa | 86.90 +/- 1.65 | `docs/generated/multidataset_detailed_proposed_summary.csv` | ready | Main-only 80.04 +/- 1.93; Aux-only 38.04 +/- 0.84; Occlusion50 85.57 +/- 1.64; MACs 80.01 +/- 0.13. |
| E_MULTI_TRENTO | C5 | Trento | 80% p=0.25 multi-degradation compact model | full | full_oa | 98.94 +/- 0.94 | `docs/generated/multidataset_detailed_proposed_summary.csv` | ready | Main-only 92.89 +/- 3.15; Aux-only 69.80 +/- 17.02; Occlusion50 98.16 +/- 1.87; MACs 79.90 +/- 0.10. |
| E_MULTI_MUUFL | C5 | MUUFL | 80% p=0.25 multi-degradation compact model | full | full_oa | 86.87 +/- 1.79 | `docs/generated/multidataset_detailed_proposed_summary.csv` | ready | Main-only 83.03 +/- 2.14; Aux-only 57.20 +/- 2.14; Occlusion50 85.99 +/- 2.08; MACs 79.97 +/- 0.07. |

## 当前投稿缺口

- `uniform fusion` 已完成，但它不是严格的 `uniform width scaling`。投稿中必须按真实含义命名；若要写 uniform width，需要另做固定宽度 baseline。
- `w/o availability mask` 仍未形成投稿级三种子消融。若短期不补，应降低 availability mask 的独立贡献强度。
- `w/o fusion-compatible terminal constraint` 当前没有稳定代码路径。不要把该消融写成已经完成。
- `without_modality_dropout` 目前不是完整三种子刷新结果，不适合单独支撑最终主张，可作为早期诊断或补跑。
