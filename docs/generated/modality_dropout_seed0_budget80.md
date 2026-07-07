# Modality dropout preliminary validation

Date: 2026-07-06

Dataset/protocol: Houston2013 HS-LiDAR, raw format, official split, class-balanced validation.

Setting: target MACs budget 80%, seed 0, source 20 epochs, compact fine-tuning 10 epochs.

Dropout run: `--modality-dropout-prob 0.25`

## Source model

| Training | Mode | OA | AA | Kappa |
|---|---|---:|---:|---:|
| no dropout | full | 86.99 | 89.47 | 85.96 |
| no dropout | main_only | 54.42 | 60.17 | 50.91 |
| no dropout | aux_only | 23.16 | 18.34 | 16.21 |
| no dropout | aux_noise | 86.16 | 88.84 | 85.07 |
| dropout 0.25 | full | 84.92 | 87.27 | 83.70 |
| dropout 0.25 | main_only | 81.26 | 83.91 | 79.75 |
| dropout 0.25 | aux_only | 41.04 | 43.74 | 36.97 |
| dropout 0.25 | aux_noise | 83.43 | 86.21 | 82.08 |

## Compact model

| Training | Mode | OA | AA | Kappa |
|---|---|---:|---:|---:|
| no dropout | full | 86.90 | 88.73 | 85.82 |
| no dropout | main_only | 49.75 | 50.36 | 45.76 |
| no dropout | aux_only | 24.13 | 18.56 | 17.16 |
| no dropout | aux_noise | 87.02 | 88.92 | 85.95 |
| dropout 0.25 | full | 88.20 | 90.26 | 87.23 |
| dropout 0.25 | main_only | 80.00 | 83.14 | 78.39 |
| dropout 0.25 | aux_only | 43.50 | 44.50 | 39.34 |
| dropout 0.25 | aux_noise | 85.92 | 87.95 | 84.76 |

Dropout compact resources: 79.45% Params, 79.86% MACs.

## Interpretation

- This is only a single-seed preliminary comparison, so it should not be written as a final claim.
- The result strongly supports keeping availability masking and modality dropout in the thesis direction because missing-modality OA improves substantially.
- The full-modality compact result also improves in this seed, but this must be checked over seeds 1/2 before being used in the paper.
- The next experiment should repeat dropout 0.25 for seeds 1/2 at the 80% budget before extending to 65% and 90%.
