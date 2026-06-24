from __future__ import annotations

import torch

from .compact import CompactBRMNet
from .hard_concrete import HardConcreteGate


def _selected_indices(gate: HardConcreteGate, threshold: float) -> torch.Tensor:
    probabilities = gate.expected_active_probability().detach()
    mask = probabilities >= threshold
    if not bool(mask.any()):
        mask[probabilities.argmax()] = True
    return mask.nonzero(as_tuple=False).flatten()


def _copy_conv_block(source, target, input_indices: torch.Tensor | None, output_indices: torch.Tensor) -> None:
    weight = source.conv.weight.index_select(0, output_indices)
    if input_indices is not None:
        weight = weight.index_select(1, input_indices)
    target.conv.weight.copy_(weight)
    if source.conv.bias is not None and target.conv.bias is not None:
        target.conv.bias.copy_(source.conv.bias.index_select(0, output_indices))

    target.bn.weight.copy_(source.bn.weight.index_select(0, output_indices))
    target.bn.bias.copy_(source.bn.bias.index_select(0, output_indices))
    target.bn.running_mean.copy_(source.bn.running_mean.index_select(0, output_indices))
    target.bn.running_var.copy_(source.bn.running_var.index_select(0, output_indices))
    target.bn.num_batches_tracked.copy_(source.bn.num_batches_tracked)


def _copy_quality_estimator(source, target, shared_indices: torch.Tensor) -> None:
    target.net[2].weight.copy_(source.net[2].weight.index_select(1, shared_indices))
    target.net[2].bias.copy_(source.net[2].bias)
    target.net[4].weight.copy_(source.net[4].weight)
    target.net[4].bias.copy_(source.net[4].bias)


@torch.no_grad()
def export_compact_brmnet(
    model,
    threshold: float = 0.5,
) -> tuple[CompactBRMNet, dict[str, object]]:
    if getattr(model, "gate_type", None) != "hard_concrete":
        raise ValueError("Compact export requires a hard_concrete BRMNet.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    indices = {
        "main_1": _selected_indices(model.main_encoder.net[0].gate, threshold),
        "main_2": _selected_indices(model.main_encoder.net[1].gate, threshold),
        "aux_1": _selected_indices(model.aux_encoder.net[0].gate, threshold),
        "aux_2": _selected_indices(model.aux_encoder.net[1].gate, threshold),
        "shared": _selected_indices(model.shared_fusion_gate, threshold),
        "head_1": _selected_indices(model.classifier.net[0].gate, threshold),
        "head_2": _selected_indices(model.classifier.net[1].gate, threshold),
    }
    widths = {name: int(value.numel()) for name, value in indices.items()}
    compact = CompactBRMNet(
        main_channels=model.main_encoder.net[0].conv.in_channels,
        aux_channels=model.aux_encoder.net[0].conv.in_channels,
        num_classes=model.classifier.linear.out_features,
        main_width=(widths["main_1"], widths["main_2"], widths["shared"]),
        aux_width=(widths["aux_1"], widths["aux_2"], widths["shared"]),
        head_width=(widths["head_1"], widths["head_2"]),
        reliability_temperature=model.reliability_fusion.temperature,
    )
    reference = next(model.parameters())
    compact.to(device=reference.device, dtype=reference.dtype)

    _copy_conv_block(model.main_encoder.net[0], compact.main_encoder.net[0], None, indices["main_1"])
    _copy_conv_block(
        model.main_encoder.net[1],
        compact.main_encoder.net[1],
        indices["main_1"],
        indices["main_2"],
    )
    _copy_conv_block(
        model.main_encoder.net[2],
        compact.main_encoder.net[2],
        indices["main_2"],
        indices["shared"],
    )
    _copy_conv_block(model.aux_encoder.net[0], compact.aux_encoder.net[0], None, indices["aux_1"])
    _copy_conv_block(
        model.aux_encoder.net[1],
        compact.aux_encoder.net[1],
        indices["aux_1"],
        indices["aux_2"],
    )
    _copy_conv_block(
        model.aux_encoder.net[2],
        compact.aux_encoder.net[2],
        indices["aux_2"],
        indices["shared"],
    )
    _copy_quality_estimator(model.main_quality, compact.main_quality, indices["shared"])
    _copy_quality_estimator(model.aux_quality, compact.aux_quality, indices["shared"])
    _copy_conv_block(
        model.classifier.net[0],
        compact.classifier.net[0],
        indices["shared"],
        indices["head_1"],
    )
    _copy_conv_block(
        model.classifier.net[1],
        compact.classifier.net[1],
        indices["head_1"],
        indices["head_2"],
    )
    compact.classifier.linear.weight.copy_(
        model.classifier.linear.weight.index_select(1, indices["head_2"])
    )
    compact.classifier.linear.bias.copy_(model.classifier.linear.bias)
    compact.train(model.training)

    metadata = {
        "threshold": float(threshold),
        "widths": {
            "main": [widths["main_1"], widths["main_2"], widths["shared"]],
            "aux": [widths["aux_1"], widths["aux_2"], widths["shared"]],
            "head": [widths["head_1"], widths["head_2"]],
        },
        "indices": {
            name: value.detach().cpu().tolist()
            for name, value in indices.items()
        },
    }
    return compact, metadata
