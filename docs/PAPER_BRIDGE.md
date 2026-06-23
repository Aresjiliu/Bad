# Paper Bridge

## Goal

This file maps the server code snapshot to the PRICAI/ICTAI paper workspace:

```text
D:/Academic/paper_submission/brmnet_pricai2026
```

## Current Mapping

| Paper Component | Server Code Source | Status |
|---|---|---|
| Budget-aware channel gating / pruning | `prune/`, `MCL/`, `models/resnet_ensemble.py`, `configuration/prune_config.py` | Logs available |
| Multi-budget sensitivity | `output/logs/prune_*`, `prune/log/` | Needs final canonical log selection |
| Complete/missing modality behavior | `missing*`, `Drfuse/`, `Fmc/`, `output/logs/depose*`, `output/logs/md_*` | Logs available but need verification |
| Single-modality fallback | `single/`, `output/logs/single_fc_modal_*` | Logs available |
| Reliability-gated fusion | `models/brmnet_reliability.py` | Standalone module added; not yet wired |
| Paper tables | `docs/generated/result_summary.*` plus manual verification | Draft discovery only |

## Results That Look Paper-Relevant

First-pass result discovery suggests these logs are important:

| Purpose | Candidate Source |
|---|---|
| Strong pruning run | `output/logs/prune_huston_try/hsi_lidar_osc_0.7_l1_0.001_gama_1.01.csv` |
| 90%/high-budget pruning | `output/logs/prune_huston_succ/hsi_lidar_osc_0.9_l1_0.001_gama_1.01.csv` |
| 70%/balanced pruning | `output/logs/prune_huston_succ/hsi_lidar_osc_0.7_l1_0.001_gama_1.01.csv` |
| Baseline DGD-like run | `output/logs/_dgd/hsi_lidardgd.csv`, `MCL/dgd.out` |
| MCL variants | `output/logs/_mcl/`, `output/logs/mcl_drop/`, `output/logs/mcl_se/` |
| Single modality | `output/logs/single_fc_modal_*` |

Do not directly paste these numbers into the paper yet. Use the source logs to confirm dataset split, modality pair, seed, and metric columns.

## Next Server-Side Tasks

1. Select canonical logs for:
   - Original baseline;
   - Ours-Full;
   - 50/65/80/90 budget points;
   - complete, main-only, auxiliary-only modality settings.
2. Wire `models/brmnet_reliability.py` into a controlled branch if MQE/RGF experiments are still needed.
3. Run minimal degradation experiments:
   - auxiliary modality noise;
   - optional main modality noise;
   - optional downsample or occlusion.
4. Export a clean CSV table for the paper workspace:

```text
paper_submission/brmnet_pricai2026/notes/experiments.md
paper_submission/brmnet_pricai2026/tables/*.tex
```

## Naming Rule for New Runs

Use explicit names:

```text
<dataset>_<modalities>_<method>_budget<budget>_<degradation>_seed<seed>
```

Example:

```text
houston_hsi_lidar_brmnet_budget65_auxnoise_mid_seed0
```

