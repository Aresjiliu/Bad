# MUUFL error analysis

| Class | 100% baseline | 80% multi-deg. | Delta | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| 1 | 90.93 +/- 1.95 | 88.25 +/- 2.23 | -2.68 | decreased |
| 2 | 84.62 +/- 3.67 | 86.21 +/- 1.68 | +1.59 | stable |
| 3 | 76.64 +/- 5.06 | 71.11 +/- 4.83 | -5.53 | decreased |
| 4 | 88.98 +/- 7.85 | 96.50 +/- 2.63 | +7.52 | improved |
| 5 | 87.59 +/- 4.39 | 89.64 +/- 0.77 | +2.05 | improved |
| 6 | 99.79 +/- 0.18 | 100.00 +/- 0.00 | +0.21 | stable |
| 7 | 93.02 +/- 3.47 | 94.88 +/- 3.05 | +1.86 | stable |
| 8 | 94.59 +/- 2.07 | 91.93 +/- 2.45 | -2.65 | decreased |
| 9 | 83.67 +/- 2.02 | 77.92 +/- 3.80 | -5.75 | decreased |
| 10 | 86.35 +/- 12.19 | 87.55 +/- 12.88 | +1.20 | stable |
| 11 | 98.62 +/- 1.37 | 98.42 +/- 1.71 | -0.20 | stable |

The 80% multi-degradation model improves robustness states but redistributes clean full-modality class accuracy. Classes with negative deltas should be discussed as trade-off cases rather than hidden.
