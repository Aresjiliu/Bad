from dataclasses import dataclass
from pathlib import Path

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
    source_paths: dict[str, Path]


def _load_known_key(path: Path, key: str) -> np.ndarray:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in content:
        available = sorted(name for name in content if not name.startswith("__"))
        raise KeyError(
            f"{path.name} must contain {key!r}; available keys: {available}"
        )
    return content[key]


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

    hsi = np.asarray(
        _load_known_key(source_paths["hsi"], HOUSTON_HSI_KEY),
        dtype=np.float32,
    )
    lidar = np.asarray(
        _load_known_key(source_paths["lidar"], HOUSTON_LIDAR_KEY),
        dtype=np.float32,
    )
    gt = np.asarray(
        _load_known_key(source_paths["gt"], HOUSTON_GT_KEY),
        dtype=np.int64,
    )

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

    roi_records = None
    if require_roi:
        roi_path = root_path / HOUSTON_ROI_FILENAME
        if not roi_path.is_file():
            raise FileNotFoundError(f"Required Houston ROI file not found: {roi_path}")
        roi_records = np.asarray(_load_known_key(roi_path, HOUSTON_ROI_KEY))
        source_paths["roi"] = roi_path

    return HoustonScene(
        hsi=hsi,
        lidar=lidar,
        gt=gt,
        roi_records=roi_records,
        source_paths=source_paths,
    )
