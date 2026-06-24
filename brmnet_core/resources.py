from __future__ import annotations

from dataclasses import dataclass

import torch

from .hard_concrete import HardConcreteGate, iter_hard_concrete_gates


Scalar = int | float | torch.Tensor


@dataclass(frozen=True)
class BRMNetResourceStats:
    params: Scalar
    macs: Scalar
    params_ratio: Scalar
    macs_ratio: Scalar


def _active_channels(gate: HardConcreteGate, mode: str) -> Scalar:
    if mode == "baseline":
        return gate.channels
    if mode == "expected":
        return gate.expected_active_probability().sum()
    mask = gate.hard_mask()
    return max(int(mask.sum().detach().cpu()), 1)


def _conv_output_size(size: int, conv) -> int:
    kernel = conv.kernel_size[0]
    stride = conv.stride[0]
    padding = conv.padding[0]
    dilation = conv.dilation[0]
    return (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


def _conv_cost(conv, input_channels: Scalar, output_channels: Scalar, output_size: int) -> tuple[Scalar, Scalar]:
    kernel_area = conv.kernel_size[0] * conv.kernel_size[1]
    weights = input_channels * output_channels * kernel_area
    if conv.groups != 1:
        weights = weights / conv.groups
    params = weights
    if conv.bias is not None:
        params = params + output_channels
    macs = weights * output_size * output_size
    return params, macs


def _linear_cost(input_features: Scalar, output_features: int, bias: bool) -> tuple[Scalar, Scalar]:
    weights = input_features * output_features
    params = weights + (output_features if bias else 0)
    return params, weights


def _raw_resources(model, patch_size: int, mode: str) -> tuple[Scalar, Scalar]:
    main_blocks = model.main_encoder.net
    aux_blocks = model.aux_encoder.net
    head_blocks = model.classifier.net

    main_widths = [
        _active_channels(main_blocks[0].gate, mode),
        _active_channels(main_blocks[1].gate, mode),
        _active_channels(model.shared_fusion_gate, mode),
    ]
    aux_widths = [
        _active_channels(aux_blocks[0].gate, mode),
        _active_channels(aux_blocks[1].gate, mode),
        main_widths[2],
    ]
    head_widths = [
        _active_channels(head_blocks[0].gate, mode),
        _active_channels(head_blocks[1].gate, mode),
    ]

    params: Scalar = 0
    macs: Scalar = 0
    for blocks, input_channels, widths in (
        (main_blocks, main_blocks[0].conv.in_channels, main_widths),
        (aux_blocks, aux_blocks[0].conv.in_channels, aux_widths),
    ):
        spatial = patch_size
        previous = input_channels
        for block, width in zip(blocks, widths):
            spatial = _conv_output_size(spatial, block.conv)
            block_params, block_macs = _conv_cost(block.conv, previous, width, spatial)
            params = params + block_params + 2 * width
            macs = macs + block_macs
            previous = width

    shared_width = main_widths[2]
    for estimator in (model.main_quality, model.aux_quality):
        first_linear = estimator.net[2]
        second_linear = estimator.net[4]
        first_params, first_macs = _linear_cost(
            shared_width, first_linear.out_features, first_linear.bias is not None
        )
        second_params, second_macs = _linear_cost(
            first_linear.out_features,
            second_linear.out_features,
            second_linear.bias is not None,
        )
        params = params + first_params + second_params
        macs = macs + first_macs + second_macs

    spatial = _conv_output_size(
        _conv_output_size(
            _conv_output_size(patch_size, main_blocks[0].conv),
            main_blocks[1].conv,
        ),
        main_blocks[2].conv,
    )
    previous = shared_width
    for block, width in zip(head_blocks, head_widths):
        spatial = _conv_output_size(spatial, block.conv)
        block_params, block_macs = _conv_cost(block.conv, previous, width, spatial)
        params = params + block_params + 2 * width
        macs = macs + block_macs
        previous = width

    final_linear = model.classifier.linear
    linear_params, linear_macs = _linear_cost(
        head_widths[1], final_linear.out_features, final_linear.bias is not None
    )
    params = params + linear_params
    macs = macs + linear_macs
    return params, macs


def estimate_brmnet_resources(model, patch_size: int, mode: str = "expected") -> BRMNetResourceStats:
    if getattr(model, "gate_type", None) != "hard_concrete":
        raise ValueError("Resource estimation requires a hard_concrete BRMNet.")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if mode not in {"baseline", "expected", "hard"}:
        raise ValueError(f"Unsupported resource mode: {mode}")

    baseline_params, baseline_macs = _raw_resources(model, patch_size, "baseline")
    params, macs = _raw_resources(model, patch_size, mode)
    return BRMNetResourceStats(
        params=params,
        macs=macs,
        params_ratio=params / baseline_params,
        macs_ratio=macs / baseline_macs,
    )


def resource_budget_loss(
    model,
    target_budget: float,
    patch_size: int,
    metric: str = "macs",
) -> tuple[torch.Tensor, BRMNetResourceStats]:
    if not 0.0 < target_budget <= 1.0:
        raise ValueError(f"target_budget must be in (0, 1], got {target_budget}")
    if metric not in {"params", "macs"}:
        raise ValueError(f"Unsupported budget metric: {metric}")
    stats = estimate_brmnet_resources(model, patch_size=patch_size, mode="expected")
    ratio = stats.params_ratio if metric == "params" else stats.macs_ratio
    if not isinstance(ratio, torch.Tensor):
        ratio = torch.as_tensor(float(ratio))
    return torch.square(ratio - float(target_budget)), stats


def initialize_uniform_resource_budget(
    model,
    target_budget: float,
    patch_size: int,
    metric: str = "macs",
    tolerance: float = 1e-6,
    max_iterations: int = 60,
) -> float:
    if not 0.0 < target_budget <= 1.0:
        raise ValueError(f"target_budget must be in (0, 1], got {target_budget}")
    if metric not in {"params", "macs"}:
        raise ValueError(f"Unsupported budget metric: {metric}")
    gates = list(iter_hard_concrete_gates(model))
    if not gates:
        raise ValueError("Uniform resource initialization requires HardConcreteGate modules.")

    def set_retention(value: float) -> float:
        for gate in gates:
            gate.set_expected_active_probability(value)
        stats = estimate_brmnet_resources(model, patch_size=patch_size, mode="expected")
        ratio = stats.params_ratio if metric == "params" else stats.macs_ratio
        return float(ratio.detach().cpu())

    low = max(gate.epsilon for gate in gates)
    high = 1.0 - low
    minimum = set_retention(low)
    maximum = set_retention(high)
    endpoint_tolerance = max(tolerance, 1e-5)
    if target_budget >= maximum and target_budget - maximum <= endpoint_tolerance:
        set_retention(high)
        return high
    if target_budget <= minimum and minimum - target_budget <= endpoint_tolerance:
        set_retention(low)
        return low
    if target_budget < minimum - tolerance or target_budget > maximum + tolerance:
        raise ValueError(
            f"target_budget {target_budget} is outside achievable {metric} ratio "
            f"[{minimum:.6f}, {maximum:.6f}]"
        )

    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        ratio = set_retention(midpoint)
        if abs(ratio - target_budget) <= tolerance:
            return midpoint
        if ratio < target_budget:
            low = midpoint
        else:
            high = midpoint
    retention = (low + high) / 2.0
    set_retention(retention)
    return retention


def estimate_compact_resources(model, patch_size: int) -> BRMNetResourceStats:
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    main_blocks = model.main_encoder.net
    aux_blocks = model.aux_encoder.net
    head_blocks = model.classifier.net

    params = 0
    macs = 0
    for blocks in (main_blocks, aux_blocks):
        spatial = patch_size
        for block in blocks:
            spatial = _conv_output_size(spatial, block.conv)
            block_params, block_macs = _conv_cost(
                block.conv,
                block.conv.in_channels,
                block.conv.out_channels,
                spatial,
            )
            params += int(block_params) + 2 * block.conv.out_channels
            macs += int(block_macs)

    shared_width = main_blocks[-1].conv.out_channels
    for estimator in (model.main_quality, model.aux_quality):
        for linear in (estimator.net[2], estimator.net[4]):
            linear_params, linear_macs = _linear_cost(
                linear.in_features,
                linear.out_features,
                linear.bias is not None,
            )
            params += int(linear_params)
            macs += int(linear_macs)

    spatial = patch_size
    for block in main_blocks:
        spatial = _conv_output_size(spatial, block.conv)
    for block in head_blocks:
        spatial = _conv_output_size(spatial, block.conv)
        block_params, block_macs = _conv_cost(
            block.conv,
            block.conv.in_channels,
            block.conv.out_channels,
            spatial,
        )
        params += int(block_params) + 2 * block.conv.out_channels
        macs += int(block_macs)

    final_linear = model.classifier.linear
    linear_params, linear_macs = _linear_cost(
        final_linear.in_features,
        final_linear.out_features,
        final_linear.bias is not None,
    )
    params += int(linear_params)
    macs += int(linear_macs)
    return BRMNetResourceStats(
        params=params,
        macs=macs,
        params_ratio=1.0,
        macs_ratio=1.0,
    )


def find_resource_budget_threshold(
    model,
    target_budget: float,
    patch_size: int,
    metric: str = "macs",
) -> tuple[float, BRMNetResourceStats]:
    if not 0.0 < target_budget <= 1.0:
        raise ValueError(f"target_budget must be in (0, 1], got {target_budget}")
    if metric not in {"params", "macs"}:
        raise ValueError(f"Unsupported budget metric: {metric}")
    gates = list(iter_hard_concrete_gates(model))
    if not gates:
        raise ValueError("Resource threshold search requires HardConcreteGate modules.")

    probabilities = sorted(
        {
            float(value)
            for gate in gates
            for value in gate.expected_active_probability().detach().cpu().tolist()
        }
    )
    candidates = [0.0]
    candidates.extend(
        (left + right) / 2.0
        for left, right in zip(probabilities, probabilities[1:])
    )
    candidates.append(1.0)

    best_threshold = 0.0
    best_stats = None
    best_error = float("inf")
    for threshold in candidates:
        for gate in gates:
            gate.hard_threshold = threshold
        stats = estimate_brmnet_resources(model, patch_size=patch_size, mode="hard")
        ratio = stats.params_ratio if metric == "params" else stats.macs_ratio
        error = abs(float(ratio) - target_budget)
        if error < best_error:
            best_threshold = threshold
            best_stats = stats
            best_error = error

    for gate in gates:
        gate.hard_threshold = best_threshold
    if best_stats is None:
        raise RuntimeError("Unable to determine a resource budget threshold.")
    return best_threshold, best_stats
