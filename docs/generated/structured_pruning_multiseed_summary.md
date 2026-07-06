# BRM-Net structured pruning multi-seed validation

Date: 2026-07-06
Dataset/protocol: Houston2013 HS-LiDAR, raw format, official split, class-balanced validation.
Training: source 20 epochs, compact fine-tuning 10 epochs, seeds 0/1/2.

## Mean +/- std over 3 seeds

| Target MACs budget | Source OA | Compact OA | Compact AA | Compact Kappa | Params ratio | MACs ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 65% | 85.98 +/- 1.87 | 85.53 +/- 2.20 | 88.11 +/- 1.42 | 84.38 +/- 2.35 | 64.79 +/- 1.93 | 64.96 +/- 0.12 |
| 80% | 85.79 +/- 1.12 | 85.84 +/- 0.92 | 88.32 +/- 0.37 | 84.69 +/- 0.99 | 79.92 +/- 0.37 | 80.01 +/- 0.10 |
| 90% | 85.33 +/- 1.20 | 85.74 +/- 0.88 | 88.18 +/- 0.51 | 84.59 +/- 0.92 | 89.78 +/- 0.84 | 90.07 +/- 0.08 |

## Per-run records

| Budget | Seed | Source OA | Compact OA | Compact AA | Compact Kappa | Params | MACs | Widths main/aux/head |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 65 | 0 | 87.22 | 87.60 | 89.40 | 86.59 | 64.66 | 65.09 | 26/50/105; 22/41/105; 96/64 |
| 65 | 1 | 86.90 | 85.78 | 88.35 | 84.63 | 66.78 | 64.89 | 24/47/109; 18/44/109; 101/64 |
| 65 | 2 | 83.83 | 83.23 | 86.59 | 81.92 | 62.92 | 64.90 | 27/49/105; 24/46/105; 86/64 |
| 80 | 0 | 86.99 | 86.90 | 88.73 | 85.82 | 79.53 | 80.12 | 29/60/114; 27/48/114; 111/64 |
| 80 | 1 | 85.62 | 85.24 | 88.01 | 84.04 | 80.26 | 80.00 | 30/55/119; 22/48/119; 111/64 |
| 80 | 2 | 84.77 | 85.37 | 88.22 | 84.19 | 79.97 | 79.92 | 29/57/120; 25/50/120; 106/64 |
| 90 | 0 | 86.67 | 86.23 | 88.41 | 85.12 | 90.74 | 90.08 | 29/61/125; 31/58/125; 118/64 |
| 90 | 1 | 84.98 | 84.73 | 87.60 | 83.53 | 89.39 | 89.98 | 32/58/126; 28/57/126; 114/64 |
| 90 | 2 | 84.35 | 86.25 | 88.55 | 85.13 | 89.21 | 90.14 | 31/62/126; 30/59/126; 109/64 |

## Interpretation for paper writing

- The compact model consistently hits the requested MACs budgets, with actual MAC ratios around 64.96%, 80.01%, and 90.07%.
- Accuracy is not monotonic with budget in this short prototype setting. The 65% budget obtains the best mean compact OA, while the 80% budget has the largest variance because seed 1 is weak.
- This supports writing the current experiment as mechanism validation, not as the final SOTA result. The next paper-critical step is to add missing-modality training and migrate the mechanism onto a larger dual-branch backbone.
