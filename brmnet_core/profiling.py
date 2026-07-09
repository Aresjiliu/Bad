from __future__ import annotations

import time

import torch


MODALITY_STATE_MASKS = {
    "full": (1.0, 1.0),
    "main_only": (1.0, 0.0),
    "aux_only": (0.0, 1.0),
}


def make_availability_mask(
    state: str,
    batch_size: int,
    device: torch.device | str,
) -> torch.Tensor:
    if state not in MODALITY_STATE_MASKS:
        raise ValueError(f"Unsupported modality state: {state}")
    return torch.tensor(
        MODALITY_STATE_MASKS[state],
        device=device,
        dtype=torch.float32,
    ).repeat(batch_size, 1)


def profile_modality_state_latency(
    model: torch.nn.Module,
    main_input: torch.Tensor,
    aux_input: torch.Tensor,
    states: tuple[str, ...] = ("full", "main_only", "aux_only"),
    warmup: int = 5,
    iterations: int = 20,
) -> dict[str, dict[str, float | int]]:
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    was_training = model.training
    model.eval()
    device = main_input.device
    uses_cuda = device.type == "cuda"
    results: dict[str, dict[str, float | int]] = {}
    with torch.no_grad():
        for state in states:
            mask = make_availability_mask(state, main_input.shape[0], device)
            for _ in range(warmup):
                model(main_input, aux_input, availability_mask=mask)
            if uses_cuda:
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(iterations):
                model(main_input, aux_input, availability_mask=mask)
            if uses_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results[state] = {
                "mean": elapsed_ms / iterations,
                "iterations": iterations,
                "warmup": warmup,
                "batch_size": int(main_input.shape[0]),
                "device": str(device),
            }
    model.train(was_training)
    return results
