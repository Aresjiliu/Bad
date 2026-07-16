from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.io import loadmat


MUUFL_SCENE_LABELS_DIR = "MUUFLGulfportSceneLabels"
MUUFL_SCENE_FILENAME = "muufl_gulfport_campus_1_hsi_220_label.mat"
MUUFL_STRUCT_KEY = "hsi"
MUUFL_HSI_FIELD = "Data"
MUUFL_AUX_FIELD = "Lidar"
MUUFL_GT_FIELD = "sceneLabels"


@dataclass(frozen=True)
class MUUFLScene:
    hsi: np.ndarray
    aux: np.ndarray
    gt: np.ndarray
    source_paths: Mapping[str, Path]
    aux_channel_mode: str


def _get_field(obj: object, names: tuple[str, ...]) -> object:
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, np.ndarray) and obj.dtype.names and name in obj.dtype.names:
            return obj[name].item() if obj[name].shape == () else obj[name]
        if isinstance(obj, dict) and name in obj:
            return obj[name]
    available = getattr(obj, "_fieldnames", None)
    if available is None and isinstance(obj, np.ndarray) and obj.dtype.names:
        available = obj.dtype.names
    raise KeyError(f"missing MUUFL field {names}; available fields: {available}")


def _load_hsi_struct(path: Path) -> object:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if MUUFL_STRUCT_KEY not in content:
        available = sorted(name for name in content if not name.startswith("__"))
        raise KeyError(f"{path.name} must contain {MUUFL_STRUCT_KEY!r}; available keys: {available}")
    return content[MUUFL_STRUCT_KEY]


def _as_numeric_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _stack_lidar(lidar: object) -> np.ndarray:
    array = np.asarray(lidar)
    if array.ndim == 3:
        return array
    if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
        return array[..., np.newaxis]

    def _normalize_candidate(candidate: object) -> np.ndarray | None:
        candidate_array = np.asarray(candidate)
        if not np.issubdtype(candidate_array.dtype, np.number):
            return None
        if candidate_array.ndim == 2:
            return candidate_array[..., np.newaxis]
        if candidate_array.ndim == 3:
            return candidate_array
        return None

    channel_names = (
        ("Intensity", "intensity", "int", "I"),
        ("Elevation", "elevation", "elev", "E"),
    )
    channels: list[np.ndarray] = []
    for names in channel_names:
        try:
            channels.append(_as_numeric_array(_get_field(lidar, names), "MUUFL LiDAR channel"))
        except KeyError:
            continue
    if not channels and isinstance(lidar, np.ndarray):
        for item in lidar.reshape(-1):
            item_array = _normalize_candidate(item)
            if item_array is not None:
                if item_array.shape[-1] > 1:
                    return item_array
                channels.append(item_array[..., 0])
                continue
            elif hasattr(item, "_fieldnames"):
                for field in item._fieldnames:
                    candidate = _normalize_candidate(getattr(item, field))
                    if candidate is not None:
                        if candidate.shape[-1] > 1:
                            return candidate
                        channels.append(candidate[..., 0])
                        break
    if not channels:
        raise ValueError("MUUFL LiDAR field must expose at least one 2-D numeric channel")
    spatial = {channel.shape for channel in channels}
    if len(spatial) != 1:
        raise ValueError(f"MUUFL LiDAR channel shapes must match: {spatial}")
    return np.stack(channels, axis=-1)


def _extract_labels(scene_labels: object) -> np.ndarray:
    array = np.asarray(scene_labels)
    if array.ndim == 2 and np.issubdtype(array.dtype, np.number):
        return array
    label_names = (
        "labels",
        "Labels",
        "label",
        "Label",
        "map",
        "Map",
        "truth",
        "Truth",
        "groundTruth",
        "classLabels",
        "class_labels",
    )
    for name in label_names:
        try:
            candidate = np.asarray(_get_field(scene_labels, (name,)))
        except KeyError:
            continue
        if candidate.ndim == 2 and np.issubdtype(candidate.dtype, np.number):
            return candidate
    if hasattr(scene_labels, "_fieldnames"):
        for field in scene_labels._fieldnames:
            candidate = np.asarray(getattr(scene_labels, field))
            if candidate.ndim == 2 and np.issubdtype(candidate.dtype, np.number):
                return candidate
    raise ValueError("MUUFL sceneLabels must contain a 2-D numeric label map")


def _select_aux_channels(aux: np.ndarray, mode: str) -> np.ndarray:
    if aux.ndim == 2:
        aux = aux[..., np.newaxis]
    if aux.ndim != 3:
        raise ValueError(f"MUUFL auxiliary modality must be 2-D or 3-D, got {aux.shape}")
    if mode == "first":
        return aux[..., :1]
    if mode == "both":
        return aux
    if mode == "mean":
        return aux.mean(axis=-1, keepdims=True)
    raise ValueError("aux_channel_mode must be one of: first, both, mean")


def load_muufl_scene(root: str | Path, aux_channel_mode: str = "first") -> MUUFLScene:
    root_path = Path(root).expanduser().resolve()
    scene_path = root_path / MUUFL_SCENE_LABELS_DIR / MUUFL_SCENE_FILENAME
    if not scene_path.is_file():
        raise FileNotFoundError(f"Missing required MUUFL scene-label file: {scene_path}")

    hsi_struct = _load_hsi_struct(scene_path)
    hsi = _as_numeric_array(_get_field(hsi_struct, (MUUFL_HSI_FIELD,)), "MUUFL HSI")
    lidar = _select_aux_channels(
        _as_numeric_array(_stack_lidar(_get_field(hsi_struct, (MUUFL_AUX_FIELD,))), "MUUFL LiDAR"),
        aux_channel_mode,
    )
    gt = _as_numeric_array(_extract_labels(_get_field(hsi_struct, (MUUFL_GT_FIELD,))), "MUUFL labels")

    if hsi.ndim != 3:
        raise ValueError(f"MUUFL HSI must be 3-D, got {hsi.shape}")
    if gt.ndim != 2:
        raise ValueError(f"MUUFL GT must be 2-D, got {gt.shape}")
    if not np.equal(gt, np.floor(gt)).all():
        raise ValueError("MUUFL GT values must be integers")
    gt = np.where(gt == -1, 0, gt)
    if gt.size and ((gt < 0).any() or (gt > 11).any()):
        raise ValueError("MUUFL GT values must be within 0..11")

    spatial_shapes = {
        "hsi": hsi.shape[:2],
        "aux": lidar.shape[:2],
        "gt": gt.shape,
    }
    if len(set(spatial_shapes.values())) != 1:
        raise ValueError(f"MUUFL spatial shapes must match: {spatial_shapes}")

    return MUUFLScene(
        hsi=hsi.astype(np.float32, copy=False),
        aux=lidar.astype(np.float32, copy=False),
        gt=gt.astype(np.int64, copy=False),
        source_paths=MappingProxyType({"scene": scene_path}),
        aux_channel_mode=aux_channel_mode,
    )
