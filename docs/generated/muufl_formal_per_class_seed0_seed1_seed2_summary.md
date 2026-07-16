# MUUFL full-modality per-class accuracy summary

| Class | 100% baseline | 80% p=0.25 multideg | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 90.93 +/- 1.95 | 88.25 +/- 2.23 | -2.68 |
| 2 | 84.62 +/- 3.67 | 86.21 +/- 1.68 | +1.59 |
| 3 | 76.64 +/- 5.06 | 71.11 +/- 4.83 | -5.53 |
| 4 | 88.98 +/- 7.85 | 96.50 +/- 2.63 | +7.52 |
| 5 | 87.59 +/- 4.39 | 89.64 +/- 0.77 | +2.05 |
| 6 | 99.79 +/- 0.18 | 100.00 +/- 0.00 | +0.21 |
| 7 | 93.02 +/- 3.47 | 94.88 +/- 3.05 | +1.86 |
| 8 | 94.59 +/- 2.07 | 91.93 +/- 2.45 | -2.65 |
| 9 | 83.67 +/- 2.02 | 77.92 +/- 3.80 | -5.75 |
| 10 | 86.35 +/- 12.19 | 87.55 +/- 12.88 | +1.20 |
| 11 | 98.62 +/- 1.37 | 98.42 +/- 1.71 | -0.20 |

Notes:
- Values are compact full-modality class accuracies from `compact_metrics.json`.
- Positive delta means the 80% multi-degradation setting improves the class over the 100% baseline.
