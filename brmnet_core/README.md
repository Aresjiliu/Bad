# BRM-Net Core Extraction

This package is the clean branch foundation extracted from the historical server code.

It keeps only the high-value ideas:

1. budget-gated convolutions from the pruning branch;
2. lightweight dual-modality encoders;
3. modality quality estimation;
4. reliability-gated fusion;
5. a compact classifier head;
6. a minimal train/evaluate engine for old dataloaders.

It intentionally does not import old `missing*`, `Drfuse`, or `Fmc` code.

## Minimal Usage

```python
from brmnet_core import BRMNet
from brmnet_core.losses import brmnet_loss

model = BRMNet(main_channels=144, aux_channels=1, num_classes=15)
outputs = model(hsi, lidar)
losses = brmnet_loss(model, outputs, labels, lambda_budget=1e-3)
losses["total"].backward()
```

`BRMNet` and `CompactBRMNet` also accept an optional `availability_mask` with shape `[batch, 2]`.
The two columns indicate whether the main and auxiliary modalities are available.
Unavailable modalities are excluded from reliability softmax competition.

## Minimal Engine

`brmnet_core.engine` accepts tuple batches `(main, aux, labels)` and dict batches with aliases such as `hsi`, `lidar`, `sar`, `ms`, `label`, and `target`.

Evaluation supports the first degradation matrix needed by the report:

- `full`
- `main_only`
- `aux_only`
- `aux_noise`

Use `evaluate_degradation_matrix(...)` to run all four modes and return a metrics dictionary keyed by mode.

Training-time modality dropout is available through `train_one_epoch(..., modality_dropout_prob=p)`.
The Houston runner exposes the same mechanism as:

```powershell
python scripts/run_brmnet_houston.py --modality-dropout-prob 0.25
```

## Next Integration Point

Use this package as the basis for a new controlled training script instead of modifying historical experiment branches directly.
