# MUUFL weighted CE ablation

## Three-seed summary

| Variant | Full OA | Full AA | Main-only OA | Aux-only OA | Noise-high OA | Downsample4 OA | Occlusion50 OA | MACs | Params |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 80% multi-deg. | 86.87 +/- 1.79 | 89.31 +/- 1.41 | 83.03 +/- 2.14 | 57.20 +/- 2.14 | 86.74 +/- 1.78 | 86.66 +/- 2.03 | 85.99 +/- 2.08 | 79.97 +/- 0.07 | 82.25 +/- 0.33 |
| 80% multi-deg. + weighted CE | 87.28 +/- 1.43 | 89.75 +/- 1.04 | 82.64 +/- 2.20 | 56.41 +/- 5.09 | 87.32 +/- 1.35 | 86.44 +/- 2.20 | 86.99 +/- 2.18 | 79.96 +/- 0.07 | 82.73 +/- 1.12 |

## Interpretation

Weighted CE changes full-modality OA by +0.41 pp and AA by +0.44 pp. The most important side effect is aux-only OA (-0.79 pp), which indicates that class balancing should not be promoted as an unconditional improvement.

## Per-class full-modality delta

| Class | Multi-deg. | Weighted CE | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 88.25 | 88.55 | +0.31 |
| 2 | 86.21 | 79.28 | -6.93 |
| 3 | 71.11 | 78.58 | +7.47 |
| 4 | 96.50 | 95.49 | -1.01 |
| 5 | 89.64 | 87.64 | -2.00 |
| 6 | 100.00 | 100.00 | +0.00 |
| 7 | 94.88 | 91.98 | -2.90 |
| 8 | 91.93 | 93.71 | +1.77 |
| 9 | 77.92 | 79.81 | +1.89 |
| 10 | 87.55 | 94.78 | +7.23 |
| 11 | 98.42 | 97.44 | -0.99 |
