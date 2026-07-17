# Fusion Availability-Mask Ablation Summary

This diagnostic compares the selected 80% multi-degradation compact model with an ablation that disables only the availability mask inside the fusion softmax.

Implementation scope:

- Branch-level availability-conditioned computation is still enabled.
- Modality dropout and train-time degradation generation are still enabled.
- The ablation isolates whether unavailable zero features should be excluded from reliability-gated fusion weights.

| State | Full BRM-Net OA (%) | W/o fusion availability mask OA (%) | Delta (pp) |
|---|---:|---:|---:|
| Full | 86.90 +/- 1.65 | 87.02 +/- 1.47 | +0.12 |
| Main only | 80.04 +/- 1.93 | 76.72 +/- 2.97 | -3.32 |
| Aux only | 38.04 +/- 0.84 | 37.59 +/- 3.41 | -0.45 |
| Aux noise high | 84.55 +/- 2.82 | 85.72 +/- 1.24 | +1.16 |
| Aux downsample 4 | 84.15 +/- 1.46 | 84.44 +/- 1.32 | +0.29 |
| Aux occlusion 50 | 85.57 +/- 1.64 | 85.67 +/- 0.94 | +0.10 |

Interpretation:

The ablation mainly hurts the true missing-modality setting. When LiDAR is unavailable, the source/compact model without fusion masking still assigns nonzero softmax probability to the zero auxiliary feature, which attenuates the available HSI branch and reduces main-only OA by 3.32 percentage points. In contrast, full-modality and degraded-but-available settings remain close. This supports a precise paper claim: availability masking is most important for physically missing modalities, while degradation-supervised reliability handles corrupted but available auxiliary inputs.

Recommended paper use:

- Include this as `w/o availability mask` or `w/o fusion availability mask` in the Houston ablation table.
- Avoid claiming it proves all availability-conditioned computation is necessary, because branch skipping remains enabled in this diagnostic.
- Pair it with resource statistics to explain that the resource benefit of branch skipping is evaluated separately from fusion-weight masking.
