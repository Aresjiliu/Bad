# BRM-Net Core Extraction

This package is the clean branch foundation extracted from the historical server code.

It keeps only the high-value ideas:

1. budget-gated convolutions from the pruning branch;
2. lightweight dual-modality encoders;
3. modality quality estimation;
4. reliability-gated fusion;
5. a compact classifier head.

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

## Next Integration Point

Use this package as the basis for a new controlled training script instead of modifying historical experiment branches directly.

