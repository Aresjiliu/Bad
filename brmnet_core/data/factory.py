from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .houston import (
    HOUSTON_CLASS_NAMES,
    HOUSTON_TRAIN_COUNTS,
    HoustonScene,
)
from .patch_dataset import (
    HoustonPatchDataset,
    NormalizationStats,
    normalize_scene,
)
from .splits import (
    CoordinateSplit,
    build_official_split,
    build_random_split,
    load_coordinate_split,
    save_coordinate_split,
)
from .trento import TrentoScene


@dataclass(frozen=True)
class HoustonDataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    split: CoordinateSplit
    normalization: NormalizationStats
    metadata: dict[str, object]


@dataclass(frozen=True)
class TrentoDataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    split: CoordinateSplit
    normalization: NormalizationStats
    metadata: dict[str, object]


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {
        str(class_id): int(np.count_nonzero(labels == class_id))
        for class_id in sorted(np.unique(labels).tolist())
    }


def _coordinate_hash(coords: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()


def _validate_split_against_scene(split: CoordinateSplit, scene: HoustonScene) -> None:
    height, width = scene.gt.shape
    for name, coords, labels in (
        ("train", split.train_coords, split.train_labels),
        ("test", split.test_coords, split.test_labels),
    ):
        if coords.size and (
            np.any(coords[:, 0] >= height) or np.any(coords[:, 1] >= width)
        ):
            raise ValueError(f"{name} split coordinates exceed scene bounds")
        if len(coords):
            scene_labels = scene.gt[coords[:, 0], coords[:, 1]]
            if not np.array_equal(scene_labels, labels):
                raise ValueError(f"{name} split labels do not match scene GT")

    expected = {tuple(coord) for coord in np.argwhere(scene.gt > 0).tolist()}
    actual = {
        tuple(coord)
        for coord in np.concatenate((split.train_coords, split.test_coords), axis=0).tolist()
    }
    if actual != expected:
        raise ValueError("split coordinates do not cover exactly all labeled scene pixels")


def _validate_split_bounds_and_labels(
    split: CoordinateSplit,
    gt: np.ndarray,
    require_full_coverage: bool,
) -> None:
    height, width = gt.shape
    for name, coords, labels in (
        ("train", split.train_coords, split.train_labels),
        ("test", split.test_coords, split.test_labels),
    ):
        if coords.size and (
            np.any(coords[:, 0] >= height) or np.any(coords[:, 1] >= width)
        ):
            raise ValueError(f"{name} split coordinates exceed scene bounds")
        if len(coords):
            scene_labels = gt[coords[:, 0], coords[:, 1]]
            if not np.array_equal(scene_labels, labels):
                raise ValueError(f"{name} split labels do not match scene GT")

    if require_full_coverage:
        expected = {tuple(coord) for coord in np.argwhere(gt > 0).tolist()}
        actual = {
            tuple(coord)
            for coord in np.concatenate((split.train_coords, split.test_coords), axis=0).tolist()
        }
        if actual != expected:
            raise ValueError("split coordinates do not cover exactly all labeled scene pixels")


def build_houston_raw_loaders(
    scene: HoustonScene,
    protocol: str,
    split_seed: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    train_counts: Mapping[int, int] = HOUSTON_TRAIN_COUNTS,
    split_file: str | Path | None = None,
) -> HoustonDataBundle:
    if protocol not in {"official", "random"}:
        raise ValueError("protocol must be 'official' or 'random'")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    if split_file:
        split = load_coordinate_split(split_file)
        if split.protocol != protocol:
            raise ValueError(
                f"split protocol {split.protocol!r} does not match requested {protocol!r}"
            )
    elif protocol == "official":
        if scene.roi_records is None:
            raise ValueError("official protocol requires Houston ROI records")
        split = build_official_split(
            scene.gt,
            scene.roi_records,
            HOUSTON_CLASS_NAMES,
            train_counts,
        )
    else:
        split = build_random_split(scene.gt, train_counts, split_seed)

    _validate_split_against_scene(split, scene)
    normalized, stats = normalize_scene(scene.hsi, scene.lidar)
    train_dataset = HoustonPatchDataset(
        normalized.hsi,
        normalized.lidar,
        split.train_coords,
        split.train_labels,
        patch_size=patch_size,
        augment=True,
    )
    test_dataset = HoustonPatchDataset(
        normalized.hsi,
        normalized.lidar,
        split.test_coords,
        split.test_labels,
        patch_size=patch_size,
        augment=False,
    )
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metadata = {
        "protocol": split.protocol,
        "seed": split.seed,
        "train_samples": len(split.train_coords),
        "test_samples": len(split.test_coords),
        "train_class_counts": _class_counts(split.train_labels),
        "test_class_counts": _class_counts(split.test_labels),
        "train_coords_sha256": _coordinate_hash(split.train_coords),
        "test_coords_sha256": _coordinate_hash(split.test_coords),
        "patch_size": int(patch_size),
    }
    return HoustonDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        split=split,
        normalization=stats,
        metadata=metadata,
    )


TRENTO_TRAIN_COUNTS = {
    1: 129,
    2: 125,
    3: 105,
    4: 154,
    5: 184,
    6: 122,
}


def build_trento_raw_loaders(
    scene: TrentoScene,
    split_seed: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    train_counts: Mapping[int, int] = TRENTO_TRAIN_COUNTS,
    split_file: str | Path | None = None,
) -> TrentoDataBundle:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")

    if split_file:
        split = load_coordinate_split(split_file)
    else:
        split = build_random_split(scene.gt, train_counts, split_seed)

    _validate_split_bounds_and_labels(split, scene.gt, require_full_coverage=False)
    normalized, stats = normalize_scene(scene.hsi, scene.aux)
    train_dataset = HoustonPatchDataset(
        normalized.hsi,
        normalized.lidar,
        split.train_coords,
        split.train_labels,
        patch_size=patch_size,
        augment=True,
    )
    test_dataset = HoustonPatchDataset(
        normalized.hsi,
        normalized.lidar,
        split.test_coords,
        split.test_labels,
        patch_size=patch_size,
        augment=False,
    )
    generator = torch.Generator()
    generator.manual_seed(int(split_seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    metadata = {
        "dataset": "trento",
        "protocol": split.protocol,
        "seed": split.seed,
        "train_samples": len(split.train_coords),
        "test_samples": len(split.test_coords),
        "train_class_counts": _class_counts(split.train_labels),
        "test_class_counts": _class_counts(split.test_labels),
        "train_coords_sha256": _coordinate_hash(split.train_coords),
        "test_coords_sha256": _coordinate_hash(split.test_coords),
        "patch_size": int(patch_size),
        "aux_channel_mode": scene.aux_channel_mode,
    }
    return TrentoDataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        split=split,
        normalization=stats,
        metadata=metadata,
    )


def _file_fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def write_houston_data_artifacts(
    output_dir: str | Path,
    bundle: HoustonDataBundle,
    scene: HoustonScene,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "split": destination / "split.npz",
        "split_summary": destination / "split_summary.json",
        "normalization": destination / "normalization.npz",
        "data_fingerprint": destination / "data_fingerprint.json",
    }
    save_coordinate_split(paths["split"], bundle.split)
    paths["split_summary"].write_text(
        json.dumps(bundle.metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(
        paths["normalization"],
        hsi_mean=bundle.normalization.hsi_mean,
        hsi_std=bundle.normalization.hsi_std,
        lidar_mean=bundle.normalization.lidar_mean,
        lidar_std=bundle.normalization.lidar_std,
    )
    fingerprints = {
        name: _file_fingerprint(Path(path))
        for name, path in scene.source_paths.items()
    }
    paths["data_fingerprint"].write_text(
        json.dumps({"files": fingerprints}, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths
