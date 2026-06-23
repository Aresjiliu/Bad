from __future__ import annotations

from types import SimpleNamespace
from collections.abc import Sequence


MODALITY_CHANNELS = {
    "hsi": 144,
    "hsi1": 244,
    "ms": 8,
    "lidar": 1,
    "sar": 4,
    "dsm": 1,
}


def normalize_pair_modalities(pair_modalities: str | Sequence[str]) -> list[str]:
    if isinstance(pair_modalities, str):
        pair_modalities = pair_modalities.split("+")
    pair = [str(item).strip() for item in pair_modalities if str(item).strip()]
    if len(pair) != 2:
        raise ValueError("BRM-Net currently expects exactly two modalities, e.g. 'hsi+lidar'.")
    return pair


def infer_channels(pair_modalities: str | Sequence[str]) -> tuple[int, int]:
    pair = normalize_pair_modalities(pair_modalities)
    try:
        return MODALITY_CHANNELS[pair[0]], MODALITY_CHANNELS[pair[1]]
    except KeyError as exc:
        known = ", ".join(sorted(MODALITY_CHANNELS))
        raise KeyError(f"Unknown modality {exc.args[0]!r}. Known modalities: {known}") from exc


def build_houston_args(
    data_root: str,
    pair_modalities: str | Sequence[str] = "hsi+lidar",
    batch_size: int = 32,
    patch_size: int = 7,
    num_workers: int = 4,
) -> SimpleNamespace:
    return SimpleNamespace(
        data_root=data_root,
        pair_modalities=normalize_pair_modalities(pair_modalities),
        batch_size=int(batch_size),
        patch_size=int(patch_size),
        num_workers=int(num_workers),
    )


def get_houston_loaders(args: SimpleNamespace):
    from src.huston2013_dataloader import huston2013_multi_dataloader

    train_loader = huston2013_multi_dataloader(train=True, args=args)
    test_loader = huston2013_multi_dataloader(train=False, args=args)
    return train_loader, test_loader
