from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import numpy as np


def _readonly_int64(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.int64)
    return np.frombuffer(contiguous.tobytes(), dtype=np.int64).reshape(
        contiguous.shape
    )


def _coordinate_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(coord) for coord in coords.tolist()}


@dataclass(frozen=True)
class CoordinateSplit:
    protocol: str
    seed: int | None
    train_coords: np.ndarray
    train_labels: np.ndarray
    test_coords: np.ndarray
    test_labels: np.ndarray

    def __post_init__(self) -> None:
        if self.protocol not in {"official", "random"}:
            raise ValueError("protocol must be 'official' or 'random'")
        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
                self.seed, (int, np.integer)
            ):
                raise ValueError("seed must be an integer or None")
            object.__setattr__(self, "seed", int(self.seed))

        arrays = {
            "train_coords": _readonly_int64(self.train_coords),
            "train_labels": _readonly_int64(self.train_labels),
            "test_coords": _readonly_int64(self.test_coords),
            "test_labels": _readonly_int64(self.test_labels),
        }
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

        for name in ("train_coords", "test_coords"):
            coords = arrays[name]
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"{name} shape must be (N, 2), got {coords.shape}")
            if np.any(coords < 0):
                raise ValueError(f"{name} coordinates must be non-negative")

        for name in ("train_labels", "test_labels"):
            labels = arrays[name]
            if labels.ndim != 1:
                raise ValueError(f"{name} shape must be (N,), got {labels.shape}")

        if len(self.train_coords) != len(self.train_labels):
            raise ValueError("train coordinate/label length mismatch")
        if len(self.test_coords) != len(self.test_labels):
            raise ValueError("test coordinate/label length mismatch")

        train_set = _coordinate_set(self.train_coords)
        test_set = _coordinate_set(self.test_coords)
        if len(train_set) != len(self.train_coords):
            raise ValueError("duplicate coordinates in train split")
        if len(test_set) != len(self.test_coords):
            raise ValueError("duplicate coordinates in test split")
        if train_set & test_set:
            raise ValueError("train/test coordinate overlap")


def build_random_split(
    gt: np.ndarray,
    train_counts: Mapping[int, int],
    seed: int,
) -> CoordinateSplit:
    gt_array = np.asarray(gt)
    if gt_array.ndim != 2:
        raise ValueError(f"GT must be 2-D, got shape {gt_array.shape}")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be an integer")

    count_class_ids = set()
    for class_id in train_counts:
        if (
            isinstance(class_id, (bool, np.bool_))
            or not isinstance(class_id, (int, np.integer))
            or class_id <= 0
        ):
            raise ValueError(f"invalid class id: {class_id!r}")
        count_class_ids.add(int(class_id))
    gt_class_ids = {
        int(class_id) for class_id in np.unique(gt_array) if class_id > 0
    }
    missing = sorted(gt_class_ids - count_class_ids)
    extra = sorted(count_class_ids - gt_class_ids)
    if missing or extra:
        raise ValueError(
            f"train_counts class mismatch: missing={missing}, extra={extra}"
        )

    train_parts = []
    train_label_parts = []
    test_parts = []
    test_label_parts = []
    rng = np.random.default_rng(seed)

    for class_id in sorted(train_counts):
        count = train_counts[class_id]
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (int, np.integer))
            or count <= 0
        ):
            raise ValueError(
                f"training count for class {class_id} must be a positive integer"
            )

        coords = np.argwhere(gt_array == class_id)
        if count > len(coords):
            raise ValueError(
                f"class {class_id} has {len(coords)} samples, fewer than "
                f"requested {count} training samples"
            )
        shuffled = coords[rng.permutation(len(coords))]
        train_coords = shuffled[:count]
        test_coords = shuffled[count:]
        train_parts.append(train_coords)
        train_label_parts.append(np.full(count, class_id, dtype=np.int64))
        test_parts.append(test_coords)
        test_label_parts.append(
            np.full(len(test_coords), class_id, dtype=np.int64)
        )

    def concatenate_coords(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    def concatenate_labels(parts: list[np.ndarray]) -> np.ndarray:
        if not parts:
            return np.empty((0,), dtype=np.int64)
        return np.concatenate(parts, axis=0)

    return CoordinateSplit(
        protocol="random",
        seed=int(seed),
        train_coords=concatenate_coords(train_parts),
        train_labels=concatenate_labels(train_label_parts),
        test_coords=concatenate_coords(test_parts),
        test_labels=concatenate_labels(test_label_parts),
    )


def _record_tokens(record: object) -> list[str]:
    values = np.asarray(record, dtype=object).reshape(-1)
    tokens = []
    for value in values:
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if isinstance(value, str):
            token = value.strip()
            if token:
                tokens.append(token)
    return tokens


def _parse_integer(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        return None


def parse_envi_roi_records(
    records: np.ndarray,
    class_names: Mapping[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw 1-based ``(X, Y)`` coordinates and 1-based class labels."""
    name_to_id = {}
    for class_id, name in class_names.items():
        if name in name_to_id:
            raise ValueError(f"duplicate class name: {name!r}")
        name_to_id[name] = int(class_id)

    roi_data: dict[int, dict[str, object]] = {}
    current_class_id: int | None = None

    for record in np.asarray(records, dtype=object).reshape(-1):
        tokens = _record_tokens(record)
        if not tokens:
            continue
        line = " ".join(tokens)

        name_match = re.search(r"\bROI\s+name\s*:\s*(\S+)", line, re.IGNORECASE)
        if name_match:
            roi_name = name_match.group(1).strip()
            if roi_name not in name_to_id:
                raise ValueError(f"unknown ROI name: {roi_name!r}")
            current_class_id = name_to_id[roi_name]
            if current_class_id in roi_data:
                raise ValueError(f"duplicate ROI name: {roi_name!r}")
            roi_data[current_class_id] = {
                "name": roi_name,
                "npts": None,
                "coords": [],
            }
            continue

        npts_match = re.search(r"\bROI\s+npts\s*:\s*(\S+)", line, re.IGNORECASE)
        if npts_match:
            if current_class_id is None:
                raise ValueError("ROI npts record without ROI context")
            npts = _parse_integer(npts_match.group(1))
            if npts is None or npts < 0:
                raise ValueError(f"invalid ROI npts: {npts_match.group(1)!r}")
            entry = roi_data[current_class_id]
            if entry["npts"] is not None:
                raise ValueError(f"duplicate ROI npts for {entry['name']!r}")
            entry["npts"] = npts
            continue

        if len(tokens) == 5:
            identifiers = [_parse_integer(token) for token in tokens[:3]]
            if all(value is not None for value in identifiers):
                if current_class_id is None:
                    raise ValueError("coordinate row without ROI context")
                _, x_coord, y_coord = identifiers
                roi_data[current_class_id]["coords"].append((x_coord, y_coord))

    coordinate_parts = []
    label_parts = []
    for class_id in sorted(roi_data):
        entry = roi_data[class_id]
        coords = np.asarray(entry["coords"], dtype=np.int64).reshape(-1, 2)
        declared_count = entry["npts"]
        if declared_count is None:
            raise ValueError(f"ROI npts missing for {entry['name']!r}")
        if declared_count != len(coords):
            raise ValueError(
                f"ROI npts mismatch for {entry['name']!r}: "
                f"declared {declared_count}, parsed {len(coords)}"
            )
        coordinate_parts.append(coords)
        label_parts.append(np.full(len(coords), class_id, dtype=np.int64))

    if not coordinate_parts:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.concatenate(coordinate_parts, axis=0),
        np.concatenate(label_parts, axis=0),
    )


def build_official_split(
    gt: np.ndarray,
    records: np.ndarray,
    class_names: Mapping[int, str],
    train_counts: Mapping[int, int] | None = None,
) -> CoordinateSplit:
    gt_array = np.asarray(gt)
    if gt_array.ndim != 2:
        raise ValueError(f"GT must be 2-D, got shape {gt_array.shape}")

    xy_coords, train_labels = parse_envi_roi_records(records, class_names)
    train_coords = np.column_stack(
        (xy_coords[:, 1] - 1, xy_coords[:, 0] - 1)
    ).astype(np.int64, copy=False)

    height, width = gt_array.shape
    if train_coords.size and (
        np.any(train_coords[:, 0] < 0)
        or np.any(train_coords[:, 0] >= height)
        or np.any(train_coords[:, 1] < 0)
        or np.any(train_coords[:, 1] >= width)
    ):
        raise ValueError("official training coordinate out of bounds")

    if train_counts is not None:
        expected_class_ids = {int(class_id) for class_id in class_names}
        count_class_ids = {int(class_id) for class_id in train_counts}
        if count_class_ids != expected_class_ids:
            raise ValueError(
                "train_counts class ids must exactly match class_names class ids"
            )
        for class_id in sorted(train_counts):
            expected = train_counts[class_id]
            if (
                isinstance(expected, (bool, np.bool_))
                or not isinstance(expected, (int, np.integer))
                or expected <= 0
            ):
                raise ValueError(
                    f"expected count for class {class_id} must be a positive integer"
                )
            actual = int(np.count_nonzero(train_labels == class_id))
            if actual != expected:
                raise ValueError(
                    f"class {class_id} expected {expected} training samples, "
                    f"parsed {actual}"
                )

    if len(train_coords):
        gt_labels = gt_array[train_coords[:, 0], train_coords[:, 1]]
        mismatches = np.flatnonzero(gt_labels != train_labels)
        if len(mismatches):
            index = int(mismatches[0])
            raise ValueError(
                "GT label mismatch at "
                f"{train_coords[index].tolist()}: ROI={train_labels[index]}, "
                f"GT={gt_labels[index]}"
            )

    all_labeled_coords = np.argwhere(gt_array > 0)
    train_set = _coordinate_set(train_coords)
    test_coords = np.asarray(
        [
            coord
            for coord in all_labeled_coords.tolist()
            if tuple(coord) not in train_set
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    test_labels = gt_array[test_coords[:, 0], test_coords[:, 1]]

    return CoordinateSplit(
        protocol="official",
        seed=None,
        train_coords=train_coords,
        train_labels=train_labels,
        test_coords=test_coords,
        test_labels=test_labels,
    )


def save_coordinate_split(path: str | Path, split: CoordinateSplit) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        protocol=np.array(split.protocol),
        seed=np.array(0 if split.seed is None else split.seed, dtype=np.int64),
        seed_is_none=np.array(split.seed is None, dtype=np.bool_),
        train_coords=split.train_coords,
        train_labels=split.train_labels,
        test_coords=split.test_coords,
        test_labels=split.test_labels,
    )


def load_coordinate_split(path: str | Path) -> CoordinateSplit:
    with np.load(Path(path), allow_pickle=False) as data:
        seed_value = int(data["seed"].item())
        seed_is_none = bool(data["seed_is_none"].item())
        return CoordinateSplit(
            protocol=str(data["protocol"].item()),
            seed=None if seed_is_none else seed_value,
            train_coords=data["train_coords"],
            train_labels=data["train_labels"],
            test_coords=data["test_coords"],
            test_labels=data["test_labels"],
        )
