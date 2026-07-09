from __future__ import annotations

import torch
from torch import nn


def _is_fully_unavailable(availability_mask: torch.Tensor | None, index: int) -> bool:
    if availability_mask is None:
        return False
    return bool(torch.count_nonzero(availability_mask[:, index]).item() == 0)


def encode_available_modalities(
    main_encoder: nn.Module,
    aux_encoder: nn.Module,
    main_quality: nn.Module,
    aux_quality: nn.Module,
    main_input: torch.Tensor,
    aux_input: torch.Tensor,
    availability_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Skip branch computation when a modality is absent for the whole batch."""

    skip_main = _is_fully_unavailable(availability_mask, 0)
    skip_aux = _is_fully_unavailable(availability_mask, 1)
    if skip_main and skip_aux:
        raise ValueError("At least one modality must be available for each sample.")

    main_feature = None if skip_main else main_encoder(main_input)
    aux_feature = None if skip_aux else aux_encoder(aux_input)

    if main_feature is None:
        if aux_feature is None:
            raise RuntimeError("Unreachable modality availability state.")
        main_feature = torch.zeros_like(aux_feature)
        q_main = torch.zeros(aux_feature.shape[0], 1, device=aux_feature.device, dtype=aux_feature.dtype)
    else:
        q_main = main_quality(main_feature)

    if aux_feature is None:
        aux_feature = torch.zeros_like(main_feature)
        q_aux = torch.zeros(main_feature.shape[0], 1, device=main_feature.device, dtype=main_feature.dtype)
    else:
        q_aux = aux_quality(aux_feature)

    return main_feature, aux_feature, q_main, q_aux
