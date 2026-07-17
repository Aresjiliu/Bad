# Validation-derived Pareto routing

Label strategy: mean_profile_pareto
Samples: 33
Target distribution: {'0.65': 12, '0.8': 18, '1.0': 3}
Prediction distribution: {'0.65': 8, '0.8': 23, '1.0': 2}
Unstable target modes: none

| Held-out seed | Mean OA | Mean MACs | Routing acc. | Mean regret | Mean saving | Train samples | Held-out samples |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.7289 | 0.7240 | 0.8182 | 0.0270 | 0.2181 | 22.0 | 11.0 |
| 1 | 0.7697 | 0.7583 | 0.5455 | 0.0080 | 0.1837 | 22.0 | 11.0 |
| 2 | 0.7570 | 0.7206 | 0.8182 | 0.0200 | 0.2214 | 22.0 | 11.0 |
| mean | 0.7519 | 0.7343 | 0.7273 | 0.0183 | 0.2077 | 22.0 | 11.0 |

## Label stability

| Mode | Unique targets | Majority budget | Majority fraction | Distribution |
| --- | ---: | ---: | ---: | --- |
| aux_downsample_2 | 1 | 0.80 | 1.00 | `{"0.8": 3}` |
| aux_downsample_4 | 1 | 0.65 | 1.00 | `{"0.65": 3}` |
| aux_noise | 1 | 0.65 | 1.00 | `{"0.65": 3}` |
| aux_noise_high | 1 | 1.00 | 1.00 | `{"1.0": 3}` |
| aux_noise_low | 1 | 0.80 | 1.00 | `{"0.8": 3}` |
| aux_noise_mid | 1 | 0.65 | 1.00 | `{"0.65": 3}` |
| aux_occlusion_25 | 1 | 0.80 | 1.00 | `{"0.8": 3}` |
| aux_occlusion_50 | 1 | 0.80 | 1.00 | `{"0.8": 3}` |
| aux_only | 1 | 0.65 | 1.00 | `{"0.65": 3}` |
| full | 1 | 0.80 | 1.00 | `{"0.8": 3}` |
| main_only | 1 | 0.80 | 1.00 | `{"0.8": 3}` |

This is a validation-state-derived router: each seed/state pair is treated as one supervised sample. It is a stronger training protocol than fitting only eleven state-level samples, but it is still not a true patch-level router because patch-wise quality features are not stored by the current formal runs. The mean-profile label strategy reduces seed-to-seed label noise by deriving labels from seed-averaged profile metrics.
