# Validation-derived Pareto routing

Samples: 33
Target distribution: {'0.65': 7, '0.8': 10, '1.0': 16}
Prediction distribution: {'0.65': 6, '0.8': 5, '1.0': 22}
Unstable target modes: aux_downsample_2, aux_downsample_4, aux_noise, aux_noise_high, aux_noise_low, aux_noise_mid, aux_occlusion_25, aux_occlusion_50, full, main_only

| Held-out seed | Mean OA | Mean MACs | Routing acc. | Mean regret | Mean saving | Train samples | Held-out samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7315 | 0.7382 | 0.2727 | 0.0245 | 0.2038 | 22.0 | 11.0 |
| 1 | 0.7294 | 0.9226 | 0.0909 | 0.0483 | 0.0194 | 22.0 | 11.0 |
| 2 | 0.7768 | 0.9257 | 0.6364 | 0.0001 | 0.0163 | 22.0 | 11.0 |
| mean | 0.7459 | 0.8622 | 0.3333 | 0.0243 | 0.0799 | 22.0 | 11.0 |

## Label stability

| Mode | Unique targets | Majority budget | Majority fraction | Distribution |
| --- | ---: | ---: | ---: | --- |
| aux_downsample_2 | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |
| aux_downsample_4 | 3 | 0.65 | 0.33 | `{"0.65": 1, "0.8": 1, "1.0": 1}` |
| aux_noise | 3 | 0.65 | 0.33 | `{"0.65": 1, "0.8": 1, "1.0": 1}` |
| aux_noise_high | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |
| aux_noise_low | 3 | 0.65 | 0.33 | `{"0.65": 1, "0.8": 1, "1.0": 1}` |
| aux_noise_mid | 3 | 0.65 | 0.33 | `{"0.65": 1, "0.8": 1, "1.0": 1}` |
| aux_occlusion_25 | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |
| aux_occlusion_50 | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |
| aux_only | 1 | 0.65 | 1.00 | `{"0.65": 3}` |
| full | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |
| main_only | 2 | 1.00 | 0.67 | `{"0.8": 1, "1.0": 2}` |

This is a validation-state-derived router: each seed/state pair is treated as one supervised sample. It is a stronger training protocol than fitting only eleven state-level samples, but it is still not a true patch-level router because patch-wise quality features are not stored by the current formal runs.
