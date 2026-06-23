# Runbook

## Environment

The code expects PyTorch and common scientific Python dependencies.
Exact versions are not yet pinned in this snapshot.

Common imports found in the code:

- `torch`
- `torchvision`
- `numpy`
- `sklearn`
- `tqdm`
- `matplotlib`
- `scipy`

## Data Layout

Most configs assume data is outside this repository:

```text
../data/Huston2013
../data/<other-dataset>
```

Do not commit raw datasets into this repository.

## Budget-Aware Pruning Run

Primary entry:

```bash
python prune/huston2013_multi_share_unimodal_center.py \
  --gpu 0 \
  --pair_modalities hsi+lidar \
  --train_epoch 300 \
  --batch_size 32 \
  --osc 0.7 \
  --l1_loss 0.001 \
  --gama 1.01
```

Important config:

- `configuration/prune_config.py`
- `args.model_root` defaults to `../output/models`
- `args.log_root` defaults to `../output/logs`
- `args.name` is built from modalities, `osc`, `l1_loss`, and `gama`

Paper interpretation:

- `osc` roughly maps to target budget or retain ratio experiments.
- `l1_loss` controls gate sparsity pressure.
- `gama` controls gate/budget update behavior.

## Missing / Reliability Run

Primary current branch:

```bash
python missing4/main.py \
  --gpu 0 \
  --pair_modalities hsi+lidar \
  --identity mcl \
  --train_epoch 300
```

Important config:

- `missing4/config.py`
- `missing4/models1.py`
- `missing4/train_model.py`

Paper interpretation:

- Use this branch to support missing/degraded modality analysis.
- Future MQE/RGF implementation should be added as a small module before being wired into this branch.

## Result Summaries

Generate repository-level summaries:

```bash
python scripts/summarize_results.py --root .
```

Outputs:

```text
docs/generated/result_summary.csv
docs/generated/result_summary.md
```

These summaries are only discovery aids. Before writing a final table, manually verify the source logs.

## Git Hygiene

The repository currently tracks many historical outputs. For new work:

- Keep new generated outputs under `output/`.
- Keep new paper-facing summaries under `docs/generated/`.
- Do not commit `__pycache__/`, new `.out` logs, or large checkpoints unless there is a clear reason.

