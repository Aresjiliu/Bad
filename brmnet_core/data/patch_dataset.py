from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


def _readonly_float32(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return np.frombuffer(contiguous.tobytes(), dtype=np.float32).reshape(contiguous.shape)


def _validate_scene_array(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if result.ndim != 3:
        raise ValueError(f"{name} shape must be (H, W, C), got {result.shape}")
    if not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_integer_array(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array)
    if not np.issubdtype(result.dtype, np.number) or not np.isrealobj(result):
        raise ValueError(f"{name} must contain real numeric values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.equal(result, np.floor(result)).all():
        raise ValueError(f"{name} must contain integer values")
    return result


@dataclass(frozen=True)
class NormalizationStats:
    hsi_mean: np.ndarray
    hsi_std: np.ndarray
    lidar_mean: np.ndarray
    lidar_std: np.ndarray

    def __post_init__(self) -> None:
        for name in ("hsi_mean", "hsi_std", "lidar_mean", "lidar_std"):
            object.__setattr__(self, name, _readonly_float32(getattr(self, name)))


@dataclass(frozen=True)
class NormalizedScene:
    hsi: np.ndarray
    lidar: np.ndarray

    def __post_init__(self) -> None:
        hsi = np.ascontiguousarray(self.hsi, dtype=np.float32)
        lidar = np.ascontiguousarray(self.lidar, dtype=np.float32)
        hsi.setflags(write=False)
        lidar.setflags(write=False)
        object.__setattr__(self, "hsi", hsi)
        object.__setattr__(self, "lidar", lidar)


def normalize_scene(
    hsi: np.ndarray,
    lidar: np.ndarray,
) -> tuple[NormalizedScene, NormalizationStats]:
    hsi_array = _validate_scene_array(hsi, "HSI")
    lidar_array = _validate_scene_array(lidar, "LiDAR")
    if hsi_array.shape[:2] != lidar_array.shape[:2]:
        raise ValueError(
            "HSI and LiDAR spatial shapes must match: "
            f"{hsi_array.shape[:2]} != {lidar_array.shape[:2]}"
        )

    hsi_mean = hsi_array.mean(axis=(0, 1), dtype=np.float64)
    hsi_std = hsi_array.std(axis=(0, 1), dtype=np.float64)
    lidar_mean = lidar_array.mean(axis=(0, 1), dtype=np.float64)
    lidar_std = lidar_array.std(axis=(0, 1), dtype=np.float64)
    hsi_std = np.where(hsi_std < 1e-8, 1.0, hsi_std)
    lidar_std = np.where(lidar_std < 1e-8, 1.0, lidar_std)

    normalized_hsi = np.empty(hsi_array.shape, dtype=np.float32)
    normalized_lidar = np.empty(lidar_array.shape, dtype=np.float32)
    np.subtract(
        hsi_array,
        hsi_mean.astype(np.float32),
        out=normalized_hsi,
        casting="unsafe",
    )
    np.divide(normalized_hsi, hsi_std.astype(np.float32), out=normalized_hsi)
    np.subtract(
        lidar_array,
        lidar_mean.astype(np.float32),
        out=normalized_lidar,
        casting="unsafe",
    )
    np.divide(normalized_lidar, lidar_std.astype(np.float32), out=normalized_lidar)
    if not np.isfinite(normalized_hsi).all() or not np.isfinite(normalized_lidar).all():
        raise ValueError("scene normalization produced non-finite values")

    stats = NormalizationStats(
        hsi_mean=hsi_mean,
        hsi_std=hsi_std,
        lidar_mean=lidar_mean,
        lidar_std=lidar_std,
    )
    return NormalizedScene(normalized_hsi, normalized_lidar), stats


class HoustonPatchDataset(Dataset):
    def __init__(
        self,
        hsi: np.ndarray,
        lidar: np.ndarray,
        coords: np.ndarray,
        labels: np.ndarray,
        patch_size: int = 7,
        augment: bool = False,
    ) -> None:
        hsi_array = _validate_scene_array(hsi, "HSI")
        lidar_array = _validate_scene_array(lidar, "LiDAR")
        if hsi_array.shape[:2] != lidar_array.shape[:2]:
            raise ValueError("HSI and LiDAR spatial shapes must match")
        if (
            isinstance(patch_size, (bool, np.bool_))
            or not isinstance(patch_size, (int, np.integer))
            or patch_size <= 0
            or patch_size % 2 == 0
        ):
            raise ValueError("patch_size must be a positive odd integer")

        coords_array = np.asarray(coords)
        labels_array = np.asarray(labels)
        if coords_array.ndim != 2 or coords_array.shape[1] != 2:
            raise ValueError(f"coords shape must be (N, 2), got {coords_array.shape}")
        if labels_array.ndim != 1:
            raise ValueError(f"labels shape must be (N,), got {labels_array.shape}")
        if len(coords_array) != len(labels_array):
            raise ValueError("coordinate/label length mismatch")
        coords_array = _validate_integer_array(coords_array, "coords")
        labels_array = _validate_integer_array(labels_array, "labels")

        coords_array = np.asarray(coords_array, dtype=np.int64)
        labels_array = np.asarray(labels_array, dtype=np.int64)
        height, width = hsi_array.shape[:2]
        if coords_array.size and (
            np.any(coords_array[:, 0] < 0)
            or np.any(coords_array[:, 0] >= height)
            or np.any(coords_array[:, 1] < 0)
            or np.any(coords_array[:, 1] >= width)
        ):
            raise ValueError("coords contain out-of-bounds values")
        if labels_array.size and np.any(labels_array <= 0):
            raise ValueError("labels must use positive 1-based class ids")

        self.coords = np.array(coords_array, copy=True)
        self.labels = np.array(labels_array, copy=True)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self.hsi = hsi_array.astype(np.float32, copy=False)
        self.lidar = lidar_array.astype(np.float32, copy=False)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, col = self.coords[index]
        half = self.patch_size // 2
        height, width = self.hsi.shape[:2]
        row_start = max(0, row - half)
        row_end = min(height, row + half + 1)
        col_start = max(0, col - half)
        col_end = min(width, col + half + 1)
        hsi_patch = self.hsi[row_start:row_end, col_start:col_end]
        lidar_patch = self.lidar[row_start:row_end, col_start:col_end]
        padding = (
            (max(0, half - row), max(0, row + half + 1 - height)),
            (max(0, half - col), max(0, col + half + 1 - width)),
            (0, 0),
        )
        if any(before or after for before, after in padding[:2]):
            hsi_patch = np.pad(hsi_patch, padding, mode="reflect")
            lidar_patch = np.pad(lidar_patch, padding, mode="reflect")

        main = torch.from_numpy(np.ascontiguousarray(hsi_patch.transpose(2, 0, 1)))
        aux = torch.from_numpy(np.ascontiguousarray(lidar_patch.transpose(2, 0, 1)))
        if self.augment:
            if torch.rand(()) < 0.5:
                main = torch.flip(main, dims=(-1,))
                aux = torch.flip(aux, dims=(-1,))
            if torch.rand(()) < 0.5:
                main = torch.flip(main, dims=(-2,))
                aux = torch.flip(aux, dims=(-2,))

        return {
            "m_1": main,
            "m_2": aux,
            "label": torch.tensor(int(self.labels[index]) - 1, dtype=torch.int64),
            "coord": torch.tensor(self.coords[index], dtype=torch.int64),
        }
