from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .losses import brmnet_loss


@dataclass
class BRMNetBatch:
    main: torch.Tensor
    aux: torch.Tensor
    labels: torch.Tensor
    quality_targets: tuple[torch.Tensor, torch.Tensor] | None = None
    availability_mask: torch.Tensor | None = None


def _first_present(batch: Mapping[str, object], keys: tuple[str, ...]) -> object:
    for key in keys:
        if key in batch:
            return batch[key]
    raise KeyError(f"Missing required batch key. Tried: {', '.join(keys)}")


def unpack_batch(batch: object, device: torch.device | str) -> BRMNetBatch:
    """Move a tuple/dict batch to device and normalize field names."""

    device = torch.device(device)
    quality_targets = None
    availability_mask = None

    if isinstance(batch, Mapping):
        main = _first_present(batch, ("main", "main_input", "hsi", "x_main", "m_1"))
        aux = _first_present(batch, ("aux", "aux_input", "lidar", "sar", "ms", "x_aux", "m_2"))
        labels = _first_present(batch, ("labels", "label", "target", "y"))
        availability_mask = batch.get("availability_mask", batch.get("modality_mask"))
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
    if availability_mask is not None:
        availability_mask = availability_mask.to(device=device, non_blocking=True).float()
    return BRMNetBatch(
        main=main,
        aux=aux,
        labels=labels,
        quality_targets=quality_targets,
        availability_mask=availability_mask,
    )


def _availability(batch: BRMNetBatch, main_available: float, aux_available: float) -> torch.Tensor:
    mask = torch.tensor(
        [main_available, aux_available],
        dtype=batch.main.dtype,
        device=batch.main.device,
    ).expand(batch.labels.numel(), 2)
    if batch.availability_mask is not None:
        mask = mask * batch.availability_mask.to(device=batch.main.device, dtype=batch.main.dtype)
    return mask


def _quality_targets(batch: BRMNetBatch, main_quality: float, aux_quality: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(batch.labels.numel())
    main = torch.full(
        (batch_size, 1),
        float(main_quality),
        dtype=batch.main.dtype,
        device=batch.main.device,
    )
    aux = torch.full(
        (batch_size, 1),
        float(aux_quality),
        dtype=batch.main.dtype,
        device=batch.main.device,
    )
    return main, aux


def _downsample_like_input(x: torch.Tensor, factor: int) -> torch.Tensor:
    if factor <= 1:
        return x
    height, width = x.shape[-2:]
    small = F.interpolate(
        x,
        size=(max(1, height // factor), max(1, width // factor)),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(small, size=(height, width), mode="bilinear", align_corners=False)


def _occlude_top_rows(x: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = min(max(float(ratio), 0.0), 1.0)
    occluded = x.clone()
    rows = int(round(x.shape[-2] * ratio))
    if rows > 0:
        occluded[..., :rows, :] = 0
    return occluded


def apply_degradation(
    batch: BRMNetBatch,
    degradation: str = "full",
    aux_noise_std: float = 0.0,
) -> BRMNetBatch:
    """Apply evaluation-time modality degradation used by the paper plan."""

    if degradation == "full":
        return BRMNetBatch(
            batch.main,
            batch.aux,
            batch.labels,
            batch.quality_targets or _quality_targets(batch, 1.0, 1.0),
            batch.availability_mask,
        )
    if degradation == "main_only":
        return BRMNetBatch(
            batch.main,
            torch.zeros_like(batch.aux),
            batch.labels,
            _quality_targets(batch, 1.0, 0.0),
            _availability(batch, 1.0, 0.0),
        )
    if degradation == "aux_only":
        return BRMNetBatch(
            torch.zeros_like(batch.main),
            batch.aux,
            batch.labels,
            _quality_targets(batch, 0.0, 1.0),
            _availability(batch, 0.0, 1.0),
        )
    if degradation == "aux_noise":
        noise = torch.randn_like(batch.aux) * float(aux_noise_std)
        return BRMNetBatch(
            batch.main,
            batch.aux + noise,
            batch.labels,
            _quality_targets(batch, 1.0, 0.8),
            _availability(batch, 1.0, 1.0),
        )
    base_noise_std = float(aux_noise_std) if aux_noise_std > 0 else 0.1
    noise_modes = {
        "aux_noise_low": (0.5 * base_noise_std, 0.8),
        "aux_noise_mid": (1.0 * base_noise_std, 0.5),
        "aux_noise_high": (2.0 * base_noise_std, 0.2),
    }
    if degradation in noise_modes:
        std, target = noise_modes[degradation]
        noise = torch.randn_like(batch.aux) * std
        return BRMNetBatch(
            batch.main,
            batch.aux + noise,
            batch.labels,
            _quality_targets(batch, 1.0, target),
            _availability(batch, 1.0, 1.0),
        )
    downsample_modes = {
        "aux_downsample_2": (2, 0.5),
        "aux_downsample_4": (4, 0.25),
    }
    if degradation in downsample_modes:
        factor, target = downsample_modes[degradation]
        return BRMNetBatch(
            batch.main,
            _downsample_like_input(batch.aux, factor),
            batch.labels,
            _quality_targets(batch, 1.0, target),
            _availability(batch, 1.0, 1.0),
        )
    occlusion_modes = {
        "aux_occlusion_25": (0.25, 0.75),
        "aux_occlusion_50": (0.50, 0.50),
    }
    if degradation in occlusion_modes:
        ratio, target = occlusion_modes[degradation]
        return BRMNetBatch(
            batch.main,
            _occlude_top_rows(batch.aux, ratio),
            batch.labels,
            _quality_targets(batch, 1.0, target),
            _availability(batch, 1.0, 1.0),
        )
    raise ValueError(f"Unsupported degradation mode: {degradation}")


def apply_modality_dropout(
    batch: BRMNetBatch,
    probability: float = 0.0,
    generator: torch.Generator | None = None,
) -> BRMNetBatch:
    """Randomly remove one modality per selected training sample."""

    probability = float(probability)
    if probability <= 0.0:
        return batch
    if probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    batch_size = int(batch.labels.numel())
    device = batch.main.device
    drop_selected = (torch.rand(batch_size, generator=generator) < probability).to(device=device)
    drop_main = (torch.rand(batch_size, generator=generator) < 0.5).to(device=device) & drop_selected
    drop_aux = drop_selected & ~drop_main

    mask = torch.ones(batch_size, 2, dtype=batch.main.dtype, device=device)
    if batch.availability_mask is not None:
        mask = batch.availability_mask.to(device=device, dtype=batch.main.dtype).clone()
    mask[drop_main, 0] = 0.0
    mask[drop_aux, 1] = 0.0

    main = batch.main * mask[:, 0].reshape(batch_size, 1, 1, 1)
    aux = batch.aux * mask[:, 1].reshape(batch_size, 1, 1, 1)
    quality_targets = batch.quality_targets
    if quality_targets is None:
        quality_targets = (mask[:, 0:1].clone(), mask[:, 1:2].clone())
    else:
        quality_targets = (
            quality_targets[0].to(device=device, dtype=batch.main.dtype) * mask[:, 0:1],
            quality_targets[1].to(device=device, dtype=batch.main.dtype) * mask[:, 1:2],
        )
    return BRMNetBatch(main, aux, batch.labels, quality_targets, mask)


def apply_aux_quality_degradation(
    batch: BRMNetBatch,
    probability: float = 0.0,
    aux_noise_std: float = 0.1,
    aux_quality_target: float = 0.5,
    degradation_types: Iterable[str] | str = ("noise",),
    generator: torch.Generator | None = None,
) -> BRMNetBatch:
    """Add train-time auxiliary degradation while keeping the modality available."""

    probability = float(probability)
    if probability <= 0.0:
        return batch
    if probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")

    if isinstance(degradation_types, str):
        modes = tuple(mode.strip() for mode in degradation_types.split(",") if mode.strip())
    else:
        modes = tuple(str(mode).strip() for mode in degradation_types if str(mode).strip())
    if not modes:
        raise ValueError("degradation_types must include at least one mode")

    quality_by_mode = {
        "noise": float(aux_quality_target),
        "downsample_2": 0.5,
        "downsample_4": 0.25,
        "occlusion_25": 0.75,
        "occlusion_50": 0.5,
    }
    unknown_modes = [mode for mode in modes if mode not in quality_by_mode]
    if unknown_modes:
        raise ValueError(f"Unsupported auxiliary quality degradation types: {', '.join(unknown_modes)}")

    batch_size = int(batch.labels.numel())
    device = batch.main.device
    dtype = batch.main.dtype
    base_mask = (
        torch.ones(batch_size, 2, dtype=dtype, device=device)
        if batch.availability_mask is None
        else batch.availability_mask.to(device=device, dtype=dtype).clone()
    )
    selected = (torch.rand(batch_size, generator=generator) < probability).to(device=device)
    aux_available = base_mask[:, 1] > 0
    selected = selected & aux_available

    aux = batch.aux.clone()
    mode_indices = torch.randint(len(modes), (batch_size,), generator=generator).to(device=device)

    quality_targets = batch.quality_targets
    if quality_targets is None:
        q_main = base_mask[:, 0:1].clone()
        q_aux = base_mask[:, 1:2].clone()
    else:
        q_main = quality_targets[0].to(device=device, dtype=dtype) * base_mask[:, 0:1]
        q_aux = quality_targets[1].to(device=device, dtype=dtype) * base_mask[:, 1:2]

    for mode_index, mode in enumerate(modes):
        mode_selected = selected & (mode_indices == mode_index)
        if not bool(mode_selected.any()):
            continue
        if mode == "noise":
            noise = torch.randn(batch.aux.shape, dtype=batch.aux.dtype, device=device, generator=generator)
            aux[mode_selected] = aux[mode_selected] + noise[mode_selected] * float(aux_noise_std)
        elif mode == "downsample_2":
            aux[mode_selected] = _downsample_like_input(aux[mode_selected], 2)
        elif mode == "downsample_4":
            aux[mode_selected] = _downsample_like_input(aux[mode_selected], 4)
        elif mode == "occlusion_25":
            aux[mode_selected] = _occlude_top_rows(aux[mode_selected], 0.25)
        elif mode == "occlusion_50":
            aux[mode_selected] = _occlude_top_rows(aux[mode_selected], 0.50)
        q_aux[mode_selected] = quality_by_mode[mode]

    q_aux = q_aux * base_mask[:, 1:2]
    return BRMNetBatch(batch.main, aux, batch.labels, (q_main, q_aux), base_mask)


def _forward_model(model: nn.Module, batch: BRMNetBatch) -> dict[str, torch.Tensor]:
    if batch.availability_mask is None:
        return model(batch.main, batch.aux)
    signature = inspect.signature(model.forward)
    if "availability_mask" in signature.parameters:
        return model(batch.main, batch.aux, availability_mask=batch.availability_mask)
    return model(batch.main, batch.aux)


def _new_meter() -> dict[str, float]:
    return {
        "loss": 0.0,
        "cls": 0.0,
        "budget": 0.0,
        "quality": 0.0,
        "pre_quality": 0.0,
        "soft_retention": 0.0,
        "hard_retention": 0.0,
        "target_budget": 0.0,
        "resource_ratio": 0.0,
        "expected_params_ratio": 0.0,
        "expected_macs_ratio": 0.0,
        "q_main": 0.0,
        "q_aux": 0.0,
        "pre_q_main": 0.0,
        "pre_q_aux": 0.0,
        "pre_u_main": 0.0,
        "pre_u_aux": 0.0,
        "fusion_weight_main": 0.0,
        "fusion_weight_aux": 0.0,
        "correct": 0.0,
        "samples": 0.0,
    }


def _update_meter(
    meter: dict[str, float],
    losses: dict[str, torch.Tensor],
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
) -> None:
    batch_size = int(labels.numel())
    for key in (
        "loss",
        "cls",
        "budget",
        "quality",
        "pre_quality",
        "soft_retention",
        "hard_retention",
        "target_budget",
        "resource_ratio",
        "expected_params_ratio",
        "expected_macs_ratio",
    ):
        loss_key = "total" if key == "loss" else key
        meter[key] += float(losses[loss_key].detach().cpu()) * batch_size
    if "q_main" in outputs:
        meter["q_main"] += float(outputs["q_main"].detach().mean().cpu()) * batch_size
    if "q_aux" in outputs:
        meter["q_aux"] += float(outputs["q_aux"].detach().mean().cpu()) * batch_size
    for key in ("pre_q_main", "pre_q_aux", "pre_u_main", "pre_u_aux"):
        if key in outputs:
            meter[key] += float(outputs[key].detach().mean().cpu()) * batch_size
    if "fusion_weights" in outputs:
        weights = outputs["fusion_weights"].detach()
        meter["fusion_weight_main"] += float(weights[:, 0].mean().cpu()) * batch_size
        meter["fusion_weight_aux"] += float(weights[:, 1].mean().cpu()) * batch_size
    logits = outputs["logits"]
    meter["correct"] += float((logits.argmax(dim=1) == labels).sum().detach().cpu())
    meter["samples"] += float(batch_size)


def _finalize_meter(meter: dict[str, float]) -> dict[str, float]:
    samples = max(meter["samples"], 1.0)
    return {
        "loss": meter["loss"] / samples,
        "cls": meter["cls"] / samples,
        "budget": meter["budget"] / samples,
        "quality": meter["quality"] / samples,
        "pre_quality": meter["pre_quality"] / samples,
        "soft_retention": meter["soft_retention"] / samples,
        "hard_retention": meter["hard_retention"] / samples,
        "target_budget": meter["target_budget"] / samples,
        "resource_ratio": meter["resource_ratio"] / samples,
        "expected_params_ratio": meter["expected_params_ratio"] / samples,
        "expected_macs_ratio": meter["expected_macs_ratio"] / samples,
        "q_main": meter["q_main"] / samples,
        "q_aux": meter["q_aux"] / samples,
        "pre_q_main": meter["pre_q_main"] / samples,
        "pre_q_aux": meter["pre_q_aux"] / samples,
        "pre_u_main": meter["pre_u_main"] / samples,
        "pre_u_aux": meter["pre_u_aux"] / samples,
        "fusion_weight_main": meter["fusion_weight_main"] / samples,
        "fusion_weight_aux": meter["fusion_weight_aux"] / samples,
        "accuracy": meter["correct"] / samples,
        "samples": int(meter["samples"]),
    }


def _classification_metrics(confusion: torch.Tensor) -> dict[str, object]:
    confusion = confusion.to(dtype=torch.float64, device="cpu")
    total = float(confusion.sum())
    correct = float(confusion.diag().sum())
    oa = correct / total if total else 0.0
    support = confusion.sum(dim=1)
    class_accuracy = torch.where(
        support > 0,
        confusion.diag() / support,
        torch.zeros_like(support),
    )
    present = support > 0
    aa = float(class_accuracy[present].mean()) if bool(present.any()) else 0.0
    expected = (
        float((confusion.sum(dim=1) * confusion.sum(dim=0)).sum()) / (total * total)
        if total
        else 0.0
    )
    kappa = (oa - expected) / (1.0 - expected) if expected < 1.0 else 0.0
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "class_accuracy": class_accuracy.tolist(),
        "confusion_matrix": confusion.to(dtype=torch.int64).tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[object],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    loss_kwargs: dict[str, float] | None = None,
    grad_clip: float | None = None,
    modality_dropout_prob: float = 0.0,
    aux_quality_degradation_prob: float = 0.0,
    aux_quality_degradation_noise_std: float = 0.1,
    aux_quality_degradation_target: float = 0.5,
    aux_quality_degradation_types: Iterable[str] | str = ("noise",),
) -> dict[str, float]:
    model.to(device)
    model.train()
    meter = _new_meter()
    loss_kwargs = loss_kwargs or {}

    for raw_batch in loader:
        batch = unpack_batch(raw_batch, device)
        batch = apply_modality_dropout(batch, probability=modality_dropout_prob)
        batch = apply_aux_quality_degradation(
            batch,
            probability=aux_quality_degradation_prob,
            aux_noise_std=aux_quality_degradation_noise_std,
            aux_quality_target=aux_quality_degradation_target,
            degradation_types=aux_quality_degradation_types,
        )
        optimizer.zero_grad(set_to_none=True)
        outputs = _forward_model(model, batch)
        losses = brmnet_loss(model, outputs, batch.labels, quality_targets=batch.quality_targets, **loss_kwargs)
        losses["total"].backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        optimizer.step()
        _update_meter(meter, losses, outputs, batch.labels)

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
    confusion = None

    for raw_batch in loader:
        batch = apply_degradation(unpack_batch(raw_batch, device), degradation=degradation, aux_noise_std=aux_noise_std)
        outputs = _forward_model(model, batch)
        losses = brmnet_loss(model, outputs, batch.labels, quality_targets=batch.quality_targets, **loss_kwargs)
        _update_meter(meter, losses, outputs, batch.labels)
        predictions = outputs["logits"].argmax(dim=1)
        num_classes = int(outputs["logits"].shape[1])
        batch_confusion = torch.bincount(
            batch.labels * num_classes + predictions,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)
        if confusion is None:
            confusion = batch_confusion.detach().cpu()
        else:
            confusion += batch_confusion.detach().cpu()

    metrics = _finalize_meter(meter)
    if confusion is not None:
        metrics.update(_classification_metrics(confusion))
        metrics["accuracy"] = metrics["oa"]
    return metrics


def evaluate_degradation_matrix(
    model: nn.Module,
    loader: Iterable[object],
    device: torch.device | str,
    loss_kwargs: dict[str, float] | None = None,
    aux_noise_std: float = 0.0,
    modes: tuple[str, ...] = (
        "full",
        "main_only",
        "aux_only",
        "aux_noise",
        "aux_noise_low",
        "aux_noise_mid",
        "aux_noise_high",
        "aux_downsample_2",
        "aux_downsample_4",
        "aux_occlusion_25",
        "aux_occlusion_50",
    ),
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
