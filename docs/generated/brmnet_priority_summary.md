# BRM-Net Experiment Summary

| Variant | Protocol | Gate | Metric | Budget | Lambda | Dropout | Epochs | Mode | Runs | OA | AA | Kappa | Params Ratio | MACs Ratio |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| full | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_noise | 2 | 0.8409 +/- 0.0029 | 0.8683 +/- 0.0024 | 0.8277 +/- 0.0031 | 0.8331 | 0.8001 |
| full | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | aux_only | 2 | 0.4295 +/- 0.0141 | 0.4399 +/- 0.0287 | 0.3862 +/- 0.0184 | 0.8331 | 0.8001 |
| full | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | full | 2 | 0.8701 +/- 0.0063 | 0.8924 +/- 0.0036 | 0.8594 +/- 0.0068 | 0.8331 | 0.8001 |
| full | official | hard_concrete | macs | 0.80 | 1.00 | 0.25 | 20 | main_only | 2 | 0.7584 +/- 0.0190 | 0.7974 +/- 0.0209 | 0.7389 +/- 0.0206 | 0.8331 | 0.8001 |
| without_modality_dropout | official | hard_concrete | macs | 0.80 | 1.00 | 0.00 | 20 | aux_noise | 2 | 0.8613 +/- 0.0081 | 0.8873 +/- 0.0071 | 0.8499 +/- 0.0086 | 0.8071 | 0.7979 |
| without_modality_dropout | official | hard_concrete | macs | 0.80 | 1.00 | 0.00 | 20 | aux_only | 2 | 0.3227 +/- 0.0193 | 0.2894 +/- 0.0107 | 0.2665 +/- 0.0139 | 0.8071 | 0.7979 |
| without_modality_dropout | official | hard_concrete | macs | 0.80 | 1.00 | 0.00 | 20 | full | 2 | 0.8674 +/- 0.0083 | 0.8912 +/- 0.0077 | 0.8565 +/- 0.0090 | 0.8071 | 0.7979 |
| without_modality_dropout | official | hard_concrete | macs | 0.80 | 1.00 | 0.00 | 20 | main_only | 2 | 0.6334 +/- 0.0372 | 0.6882 +/- 0.0512 | 0.6040 +/- 0.0418 | 0.8071 | 0.7979 |
