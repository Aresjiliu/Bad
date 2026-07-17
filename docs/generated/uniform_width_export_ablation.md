# Uniform-Width Export Ablation

This diagnostic compares the selected learned nonuniform compact export with a target-matched uniform-width export. The uniform-width baseline keeps the same width ratio in every gated block and is not logit-equivalent to the learned hard-mask source model.

| State | Learned nonuniform OA | Uniform-width OA | Delta | Learned MACs | Uniform MACs |
|---|---:|---:|---:|---:|---:|
| Full | 86.90 +/- 1.65 | 86.29 +/- 0.77 | -0.61 | 80.01 | 79.45 |
| HSI only | 80.04 +/- 1.93 | 79.22 +/- 1.47 | -0.82 | 80.01 | 79.45 |
| LiDAR only | 38.04 +/- 0.84 | 37.76 +/- 1.46 | -0.28 | 80.01 | 79.45 |
| Noise-high | 84.55 +/- 2.82 | 84.93 +/- 0.83 | +0.38 | 80.01 | 79.45 |
| Downsample-4 | 84.15 +/- 1.46 | 82.88 +/- 0.75 | -1.27 | 80.01 | 79.45 |
| Occlusion-50 | 85.57 +/- 1.64 | 85.00 +/- 0.67 | -0.57 | 80.01 | 79.45 |
