from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.io import loadmat


TRENTO_HSI_FILENAME = "Italy_hsi.mat"
TRENTO_AUX_FILENAME = "Italy_lidar.mat"
TRENTO_GT_FILENAME = "allgrd.mat"

TRENTO_HSI_KEY = "data"
TRENTO_AUX_KEY = "data"
TRENTO_GT_KEY = "mask_test"


@dataclass(frozen=True)
class TrentoScene:
    hsi: np.ndarray
    aux: np.ndarray
    gt: np.ndarray
    source_paths: Mapping[str, Path]
    aux_channel_mode: str


def _load_key(path: Path, key: str) -> np.ndarray:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in content:
        available = sorted(name for name in content if not name.startswith("__"))
        raise KeyError(f"{path.name} must contain {key!r}; available keys: {available}")
    return np.asarray(content[key])


def _validate_numeric(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def _select_aux_channels(aux: np.ndarray, mode: str) -> np.ndarray:
    if aux.ndim == 2:
        aux = aux[..., np.newaxis]
    if aux.ndim != 3:
        raise ValueError(f"Trento auxiliary modality must be 2-D or 3-D, got {aux.shape}")
    if mode == "first":
        return aux[..., :1]
    if mode == "both":
        return aux
    if mode == "mean":
        return aux.mean(axis=-1, keepdims=True)
    raise ValueError("aux_channel_mode must be one of: first, both, mean")


def load_trento_scene(
    root: str | Path,
    aux_channel_mode: str = "first",
) -> TrentoScene:
    root_path = Path(root).expanduser().resolve()
    source_paths = {
        "hsi": root_path / TRENTO_HSI_FILENAME,
        "aux": root_path / TRENTO_AUX_FILENAME,
        "gt": root_path / TRENTO_GT_FILENAME,
    }
    missing = [
        f"{name}: {path}"
        for name, path in source_paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing required Trento files:\n" + "\n".join(missing))

    hsi = _load_key(source_paths["hsi"], TRENTO_HSI_KEY)
    aux = _select_aux_channels(_load_key(source_paths["aux"], TRENTO_AUX_KEY), aux_channel_mode)
    gt = _load_key(source_paths["gt"], TRENTO_GT_KEY)

    _validate_numeric(hsi, "HSI")
    _validate_numeric(aux, "auxiliary modality")
    _validate_numeric(gt, "GT")
    if hsi.ndim != 3:
        raise ValueError(f"Trento HSI must be 3-D, got {hsi.shape}")
    if gt.ndim != 2:
        raise ValueError(f"Trento GT must be 2-D, got {gt.shape}")
    if not np.equal(gt, np.floor(gt)).all():
        raise ValueError("Trento GT values must be integers")
    if gt.size and ((gt < 0).any() or (gt > 6).any()):
        raise ValueError("Trento GT values must be within 0..6")

    spatial_shapes = {
        "hsi": hsi.shape[:2],
        "aux": aux.shape[:2],
        "gt": gt.shape,
    }
    if len(set(spatial_shapes.values())) != 1:
        raise ValueError(f"Trento spatial shapes must match: {spatial_shapes}")

    return TrentoScene(
        hsi=hsi.astype(np.float32, copy=False),
        aux=aux.astype(np.float32, copy=False),
        gt=gt.astype(np.int64, copy=False),
        source_paths=MappingProxyType(dict(source_paths)),
        aux_channel_mode=aux_channel_mode,
    )
