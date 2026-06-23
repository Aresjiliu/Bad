# Code Map

## Repository Role

This repository is an experiment-heavy server snapshot. It contains baseline multimodal classification code, pruning variants, missing-modality variants, and historical logs.

## Core Shared Directories

| Path | Role | Notes |
|---|---|---|
| `datasets/` | Dataset loaders and preprocessing utilities | Includes Houston2013, Augsburg, LCZ helpers |
| `src/` | Baseline and transfer training scripts | Many shell launchers for Houston/Augsburg experiments |
| `lib/` | Shared training utilities, metrics, schedulers, model architecture helpers | `model_develop.py` contains many train loops and OA/AA/Kappa logic |
| `models/` | Main baseline and pruning-compatible model definitions | `resnet_ensemble.py` contains many HSI-LiDAR coupled model variants |
| `loss/` | MMD, Wasserstein, center loss, KD, class imbalance losses | Used by baseline and transfer branches |
| `configuration/` | Baseline/pruning/multimodal config files | `prune_config.py` is the pruning mainline config |

## Lightweight / Budget-Aware Branch

| Path | Role | Important Files |
|---|---|---|
| `prune/` | Current pruning/lightweight experiment branch | `huston2013_multi_share_unimodal_center.py`, `transfer.py`, `load_model.py` |
| `MCL/` | Earlier multi-modal channel/gate/pruning experiments | `huston2013_dw_prune*.py`, `huston2013_gate*.py`, `.out` logs |
| `Mlevel/` | Gate/multi-level related variants | `huston2013_gate*.py`, `missing.py` |
| `models1/` | Older lightweight/gate/depthwise model variants | `dw_prune*.py`, `gate_model*.py`, `compact.py` |

Recommended paper-facing line:

```text
configuration/prune_config.py
  -> prune/huston2013_multi_share_unimodal_center.py
  -> models.resnet_ensemble.HSI_Lidar_Couple_Prune
  -> lib.model_develop.train_base_multi_share_unimodal_center_prune
```

## Missing / Degraded Modality Branch

| Path | Role | Important Files |
|---|---|---|
| `missing/` | Early missing-modality experiments | `main_base.py`, `main_base_sfd.py`, `train.py` |
| `missing1/` | SFD/CKD/loss variants | `models1.py`, `train_model.py`, `analysis.py` |
| `missing2/` | Depose/ablation variants and logs | `models1.py`, `merge.py`, `log_file/` |
| `missing3/` | DrFuse/adversarial/FMC exploration | README files, `fmcnet_modules.py`, adversarial scripts |
| `missing4/` | FMC/DrFuse branch with more explicit config | `main.py`, `models1.py`, `fmc_module.py`, `fmc_inference.py` |
| `Drfuse/` | DrFuse variant | `main.py`, `models1.py`, `train_model.py` |
| `Fmc/` | FMC variant | `main.py`, `fmc_module.py`, `train_example.py` |
| `claude/` | Claude-generated/assisted DrFuse variant | `main.py`, `models1.py`, `analyse.md` |

Recommended reliability-facing line:

```text
missing4/config.py
  -> missing4/main.py
  -> missing4/models1.py::Drfuse
  -> missing4/train_model.py::train_single_ckd
```

## Baseline Branches

| Path | Role |
|---|---|
| `single/` | Single-modality baselines for HSI/MS/LiDAR |
| `origin/` | Teacher/original model utilities |
| `src/` | Many original DGDNet/ShaSpec/MMFormer/transfer scripts |

## Output Locations

| Path | Content |
|---|---|
| `output/logs/` | CSV logs with config rows followed by per-epoch metrics |
| `output/figures/` | Generated visualization images |
| `output/depose/` | Figure/decomposition outputs, ignored by Git |
| `prune/log/` | Raw pruning `.out` logs |
| top-level `*.out` and module `*.out` | Historical raw terminal logs |

