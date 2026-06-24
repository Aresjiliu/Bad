from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.io import loadmat


HOUSTON_HSI_FILENAME = "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat"
HOUSTON_LIDAR_FILENAME = "2013_IEEE_GRSS_DF_Contest_LiDAR.mat"
HOUSTON_GT_FILENAME = "GRSS2013.mat"
HOUSTON_ROI_FILENAME = "2013_IEEE_GRSS_DF_Contest_Samples_TR.mat"

HOUSTON_HSI_KEY = "ans"
HOUSTON_LIDAR_KEY = "LiDAR_data"
HOUSTON_GT_KEY = "name"
HOUSTON_ROI_KEY = "TR_Samples"

HOUSTON_TRAIN_COUNTS = {
    1: 198,
    2: 190,
    3: 192,
    4: 188,
    5: 186,
    6: 182,
    7: 196,
    8: 191,
    9: 193,
    10: 191,
    11: 181,
    12: 192,
    13: 184,
    14: 181,
    15: 187,
}

HOUSTON_CLASS_NAMES = {
    1: "grass_healthy",
    2: "grass_stressed",
    3: "grass_synthetic",
    4: "tree",
    5: "soil",
    6: "water",
    7: "residential",
    8: "commercial",
    9: "road",
    10: "highway",
    11: "railway",
    12: "parking_lot1",
    13: "parking_lot2",
    14: "tennis_court",
    15: "running_track",
}


@dataclass(frozen=True)
class HoustonScene:
    hsi: np.ndarray
    lidar: np.ndarray
    gt: np.ndarray
    roi_records: np.ndarray | None
    source_paths: Mapping[str, Path]


def _load_known_key(path: Path, key: str) -> np.ndarray:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in content:
        available = sorted(name for name in content if not name.startswith("__"))
        raise KeyError(
            f"{path.name} must contain {key!r}; available keys: {available}"
        )
    return content[key]


def _validate_real_finite(array: np.ndarray, name: str) -> None:
    if not np.issubdtype(array.dtype, np.number) or not np.isrealobj(array):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_gt_values(gt: np.ndarray) -> None:
    _validate_real_finite(gt, "GT")
    if not np.equal(gt, np.floor(gt)).all():
        raise ValueError("GT values must be integers")
    if gt.size and ((gt < 0).any() or (gt > 15).any()):
        raise ValueError("GT values must be within 0..15")


def _validate_float32_range(array: np.ndarray, name: str) -> None:
    limit = np.finfo(np.float32).max
    if array.size and ((array < -limit).any() or (array > limit).any()):
        raise ValueError(f"{name} values must be within float32 range")


def load_houston_scene(
    root: str | Path,
    require_roi: bool = True,
) -> HoustonScene:
    root_path = Path(root).expanduser().resolve()
    source_paths = {
        "hsi": root_path / HOUSTON_HSI_FILENAME,
        "lidar": root_path / HOUSTON_LIDAR_FILENAME,
        "gt": root_path / HOUSTON_GT_FILENAME,
    }
    if require_roi:
        source_paths["roi"] = root_path / HOUSTON_ROI_FILENAME

    missing = [
        f"{modality.upper()}: {path}"
        for modality, path in source_paths.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required Houston files:\n" + "\n".join(missing)
        )

    hsi = np.asarray(_load_known_key(source_paths["hsi"], HOUSTON_HSI_KEY))
    lidar = np.asarray(_load_known_key(source_paths["lidar"], HOUSTON_LIDAR_KEY))
    gt = np.asarray(_load_known_key(source_paths["gt"], HOUSTON_GT_KEY))

    _validate_real_finite(hsi, "HSI")
    _validate_real_finite(lidar, "LiDAR")
    _validate_float32_range(hsi, "HSI")
    _validate_float32_range(lidar, "LiDAR")
    _validate_gt_values(gt)

    if hsi.ndim != 3:
        raise ValueError(f"HSI must be 3-D, got shape {hsi.shape}")
    if lidar.ndim == 2:
        lidar = lidar[..., np.newaxis]
    if lidar.ndim != 3:
        raise ValueError(f"LiDAR must be 2-D or 3-D, got shape {lidar.shape}")
    if gt.ndim != 2:
        raise ValueError(f"GT must be 2-D, got shape {gt.shape}")

    spatial_shapes = {
        "hsi": hsi.shape[:2],
        "lidar": lidar.shape[:2],
        "gt": gt.shape,
    }
    if len(set(spatial_shapes.values())) != 1:
        raise ValueError(f"Houston spatial shapes must match: {spatial_shapes}")

    hsi = hsi.astype(np.float32, copy=False)
    lidar = lidar.astype(np.float32, copy=False)
    gt = gt.astype(np.int64, copy=False)

    roi_records = None
    if require_roi:
        roi_records = np.asarray(
            _load_known_key(source_paths["roi"], HOUSTON_ROI_KEY)
        )

    return HoustonScene(
        hsi=hsi,
        lidar=lidar,
        gt=gt,
        roi_records=roi_records,
        source_paths=MappingProxyType(dict(source_paths)),
    )
