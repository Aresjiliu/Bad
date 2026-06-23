# Experiment Index

## Purpose

This file explains how current logs map to the thesis/paper narrative.

## Paper-Facing Claims

| Claim | Source Area | Verification Status |
|---|---|---|
| Budget pruning improves Params/FLOPs/Latency with small OA drop | `prune/`, `MCL/`, `output/logs/prune_*` | Needs final source-log selection |
| 50/65/80/90 budget sensitivity | `output/logs/prune_*`, `prune/log/` | Needs mapping from `osc`/log names to paper budget levels |
| Complete vs missing modalities | `missing*`, `Drfuse/`, `Fmc/`, `output/logs/` | Needs final table extraction |
| Single-modality fallback | `single/`, `output/logs/single_*` | Logs available |
| BRM-Net reliability module | Not yet wired into this repository | Pending implementation |

## Log Interpretation

CSV logs usually start with config rows:

```text
train_epoch,300
batch_size,32
...
```

Then per-epoch metric rows are appended without headers.
For common train loops, the last columns are usually:

```text
accuracy_test, accuracy_best
```

Some loops include OA, AA, and Kappa in intermediate columns.
Use `scripts/summarize_results.py` as a first-pass parser, then verify manually.

## Candidate Logs for Paper Tables

| Table | Candidate Sources |
|---|---|
| Main pruning comparison | `output/logs/prune_huston_succ/`, `output/logs/prune_berlin_succ/` |
| Budget sensitivity | `output/logs/prune_huston_succ/`, `output/logs/prune_huston_repair/`, `prune/log/` |
| Missing modality | `output/logs/depose*`, `output/logs/md_*`, `missing2/log_file/` |
| Single modality | `output/logs/single_fc_modal_*`, `single/*.out` |
| DrFuse/FMC variants | `output/logs/drfuse*`, `Drfuse/*.out`, `Fmc/*.out` |

## Immediate Cleanup Tasks

1. Select one canonical log per paper table row.
2. Add a source-log column to the paper experiment notes.
3. Rerun only missing P0 experiments instead of adding more uncontrolled variants.
4. Keep new logs named with dataset, modality pair, budget, method, seed, and degradation.

