from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from .losses import brmnet_loss


@dataclass
class BRMNetBatch:
    main: torch.Tensor
    aux: torch.Tensor
    labels: torch.Tensor
    quality_targets: tuple[torch.Tensor, torch.Tensor] | None = None


def _first_present(batch: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in batch:
            return batch[key]
    raise KeyError(f"Missing required batch key. Tried: {', '.join(keys)}")


def unpack_batch(batch: object, device: torch.device | str) -> BRMNetBatch:
    """Move a tuple/dict batch to device and normalize field names."""

    device = torch.device(device)
    quality_targets = None

    if isinstance(batch, Mapping):
        main = _first_present(batch, ("main", "main_input", "hsi", "x_main", "m_1"))
        aux = _first_present(batch, ("aux", "aux_input", "lidar", "sar", "ms", "x_aux", "m_2"))
        labels = _first_present(batch, ("labels", "label", "target", "y"))
        if "q_main" in batch and "q_aux" in batch:
            quality_targets = (batch["q_main"], batch["q_aux"])
        elif "quality_main" in batch and "quality_aux" in batch:
            quality_targets = (batch["quality_main"], batch["quality_aux"])
    elif isinstance(batch, (tuple, list)) and len(batch) in (3, 5):
        main, aux, labels = batch[:3]
        if len(batch) == 5:
            quality_targets = (batch[3], batch[4])
    else:
        raise TypeError("Batch must be a dict or a tuple/list of length 3 or 5.")

    main = main.to(device=device, non_blocking=True)
    aux = aux.to(device=device, non_blocking=True)
    labels = labels.to(device=device, non_blocking=True).long()
    if quality_targets is not None:
        quality_targets = (
            quality_targets[0].to(device=device, non_blocking=True),
            quality_targets[1].to(device=device, non_blocking=True),
        )
    return BRMNetBatch(main=main, aux=aux, labels=labels, quality_targets=quality_targets)


def apply_degradation(
    batch: BRMNetBatch,
    degradation: str = "full",
    aux_noise_std: float = 0.0,
) -> BRMNetBatch:
    """Apply evaluation-time modality degradation used by the paper plan."""

    if degradation == "full":
        return batch
    if degradation == "main_only":
        return BRMNetBatch(batch.main, torch.zeros_like(batch.aux), batch.labels, batch.quality_targets)
    if degradation == "aux_only":
        return BRMNetBatch(torch.zeros_like(batch.main), batch.aux, batch.labels, batch.quality_targets)
    if degradation == "aux_noise":
        noise = torch.randn_like(batch.aux) * float(aux_noise_std)
        return BRMNetBatch(batch.main, batch.aux + noise, batch.labels, batch.quality_targets)
    raise ValueError(f"Unsupported degradation mode: {degradation}")


def _new_meter() -> dict[str, float]:
    return {"loss": 0.0, "cls": 0.0, "budget": 0.0, "quality": 0.0, "correct": 0.0, "samples": 0.0}


def _update_meter(meter: dict[str, float], losses: dict[str, torch.Tensor], logits: torch.Tensor, labels: torch.Tensor) -> None:
    batch_size = int(labels.numel())
    for key in ("loss", "cls", "budget", "quality"):
        loss_key = "total" if key == "loss" else key
        meter[key] += float(losses[loss_key].detach().cpu()) * batch_size
    meter["correct"] += float((logits.argmax(dim=1) == labels).sum().detach().cpu())
    meter["samples"] += float(batch_size)


def _finalize_meter(meter: dict[str, float]) -> dict[str, float]:
    samples = max(meter["samples"], 1.0)
    return {
        "loss": meter["loss"] / samples,
        "cls": meter["cls"] / samples,
        "budget": meter["budget"] / samples,
        "quality": meter["quality"] / samples,
        "accuracy": meter["correct"] / samples,
        "samples": int(meter["samples"]),
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[object],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    loss_kwargs: dict[str, float] | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    model.to(device)
    model.train()
    meter = _new_meter()
    loss_kwargs = loss_kwargs or {}

    for raw_batch in loader:
        batch = unpack_batch(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch.main, batch.aux)
        losses = brmnet_loss(model, outputs, batch.labels, quality_targets=batch.quality_targets, **loss_kwargs)
        losses["total"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        _update_meter(meter, losses, outputs["logits"], batch.labels)

    return _finalize_meter(meter)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[object],
    device: torch.device | str,
    loss_kwargs: dict[str, float] | None = None,
    degradation: str = "full",
    aux_noise_std: float = 0.0,
) -> dict[str, float]:
    model.to(device)
    model.eval()
    meter = _new_meter()
    loss_kwargs = loss_kwargs or {}

    for raw_batch in loader:
        batch = apply_degradation(unpack_batch(raw_batch, device), degradation=degradation, aux_noise_std=aux_noise_std)
        outputs = model(batch.main, batch.aux)
        losses = brmnet_loss(model, outputs, batch.labels, quality_targets=batch.quality_targets, **loss_kwargs)
        _update_meter(meter, losses, outputs["logits"], batch.labels)

    return _finalize_meter(meter)


def evaluate_degradation_matrix(
    model: nn.Module,
    loader: Iterable[object],
    device: torch.device | str,
    loss_kwargs: dict[str, float] | None = None,
    aux_noise_std: float = 0.0,
    modes: tuple[str, ...] = ("full", "main_only", "aux_only", "aux_noise"),
) -> dict[str, dict[str, float]]:
    return {
        mode: evaluate(
            model,
            loader,
            device,
            loss_kwargs=loss_kwargs,
            degradation=mode,
            aux_noise_std=aux_noise_std,
        )
        for mode in modes
    }
