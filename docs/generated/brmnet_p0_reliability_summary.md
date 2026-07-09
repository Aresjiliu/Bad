# BRM-Net Experiment Summary

| Variant | Protocol | Gate | Metric | Budget | Lambda | Dropout | Epochs | Mode | Runs | OA | AA | Kappa | Params Ratio | MACs Ratio |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_downsample_2 | 1 | 0.7980 +/- 0.0000 | 0.8304 +/- 0.0000 | 0.7819 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_downsample_4 | 1 | 0.7449 +/- 0.0000 | 0.7864 +/- 0.0000 | 0.7250 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_noise | 1 | 0.3147 +/- 0.0000 | 0.3100 +/- 0.0000 | 0.2516 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_noise_high | 1 | 0.2270 +/- 0.0000 | 0.2140 +/- 0.0000 | 0.1543 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_noise_low | 1 | 0.3153 +/- 0.0000 | 0.3099 +/- 0.0000 | 0.2523 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_noise_mid | 1 | 0.3157 +/- 0.0000 | 0.3086 +/- 0.0000 | 0.2526 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_occlusion_25 | 1 | 0.7470 +/- 0.0000 | 0.7945 +/- 0.0000 | 0.7276 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_occlusion_50 | 1 | 0.6479 +/- 0.0000 | 0.7105 +/- 0.0000 | 0.6225 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_only | 1 | 0.4128 +/- 0.0000 | 0.4485 +/- 0.0000 | 0.3715 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | full | 1 | 0.8130 +/- 0.0000 | 0.8454 +/- 0.0000 | 0.7980 +/- 0.0000 | 0.8086 | 0.8009 |
| default | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | main_only | 1 | 0.7413 +/- 0.0000 | 0.7894 +/- 0.0000 | 0.7211 +/- 0.0000 | 0.8086 | 0.8009 |
