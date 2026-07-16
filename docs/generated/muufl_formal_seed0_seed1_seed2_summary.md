# MUUFL formal 3-seed summary

| Variant | Seeds | Full OA | Main-only OA | Aux-only OA | Noise-high OA | Downsample4 OA | Occlusion50 OA | MACs | Params | Compact latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| muufl_100_baseline | 3 | 88.51 +/- 1.19 | 71.09 +/- 4.37 | 26.24 +/- 1.18 | 88.52 +/- 1.15 | 84.12 +/- 1.31 | 80.08 +/- 2.32 | 100.00 +/- 0.00 | 100.00 +/- 0.00 | 1.90 +/- 0.29 |
| muufl_80_p025_multideg | 3 | 86.87 +/- 1.79 | 83.03 +/- 2.14 | 57.20 +/- 2.14 | 86.74 +/- 1.78 | 86.66 +/- 2.03 | 85.99 +/- 2.08 | 79.97 +/- 0.07 | 82.25 +/- 0.33 | 1.96 +/- 0.10 |

Notes:
- Metrics are reported from `compact_metrics.json` after compact fine-tuning, using the fixed MUUFL splits in `output/splits/muufl_seed{0,1,2}.npz`.
- `muufl_80_p025_multideg` uses target MACs=0.8, modality dropout probability=0.25, and auxiliary quality degradation probability=0.25 with noise/downsample/occlusion degradation.
