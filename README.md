# BRM-Net Server Code Snapshot

This repository is the server-side experiment snapshot for the thesis/paper project on lightweight multimodal remote sensing classification.

Current research direction:

> Budget- and Modality-Reliability-Aware Lightweight Multimodal Remote Sensing Classification.

The repository contains multiple generations of experiments. Treat this codebase as an experimental archive plus a runnable baseline, not as a clean library.

## Main Lines

| Line | Directories | Purpose |
|---|---|---|
| Budget-aware pruning | `prune/`, `MCL/`, `models/`, `configuration/prune_config.py` | Main lightweight/pruning experiments for the BRM-Net paper |
| Missing/degraded modality | `missing/`, `missing1/`, `missing2/`, `missing3/`, `missing4/`, `Fmc/`, `Drfuse/`, `claude/` | Modality missing, feature decomposition, FMC/DrFuse variants |
| Baselines and datasets | `src/`, `origin/`, `single/`, `datasets/`, `lib/`, `loss/`, `models/` | Dataloaders, baseline training scripts, shared utilities |
| Outputs and logs | `output/`, `*.out`, `prune/log/` | Historical experiment logs and generated figures |

## Recommended Entry Points

Budget-aware pruning:

```bash
python prune/huston2013_multi_share_unimodal_center.py --gpu 0 --pair_modalities hsi+lidar --osc 0.7 --l1_loss 0.001 --gama 1.01
```

Missing/degraded modality branch:

```bash
python missing4/main.py --gpu 0 --pair_modalities hsi+lidar --identity mcl
```

Single-modality baselines:

```bash
python single/huston2013_single_train.py --gpu 0
```

## Important Notes

- The default data path is usually `../data/<dataset>`.
- The default output path is usually `../output/...`; several scripts mutate `args.model_root`, `args.log_root`, and `args.figure_root`.
- Historical logs are valuable for paper tables. Do not delete `.out`, `output/logs`, or `prune/log` until the paper tables are finalized.
- Many directories contain old experimental variants. Prefer using the runbook and code map before starting new experiments.

## Documentation

- `docs/CODE_MAP.md`: directory and entry-point map.
- `docs/RUNBOOK.md`: commands and experiment conventions.
- `docs/EXPERIMENT_INDEX.md`: current experiment log interpretation.
- `docs/generated/result_summary.*`: generated summaries from logs.

## Utility Scripts

Summarize logs:

```bash
python scripts/summarize_results.py --root .
```

Inventory repository files:

```bash
python scripts/inventory.py --root .
```

