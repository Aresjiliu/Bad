from collections.abc import Iterable

import numpy as np


def audit_label_map(
    labels: np.ndarray,
    background_values: Iterable[int] = (0,),
) -> dict[str, object]:
    label_array = np.asarray(labels)
    if label_array.ndim != 2:
        raise ValueError(f"labels must be a 2-D array, got {label_array.shape}")
    if not np.issubdtype(label_array.dtype, np.number) or not np.isrealobj(label_array):
        raise ValueError("labels must contain real numeric values")
    if not np.isfinite(label_array).all():
        raise ValueError("labels must contain only finite values")
    if not np.equal(label_array, np.floor(label_array)).all():
        raise ValueError("labels must contain integer values")

    labels_int = label_array.astype(np.int64, copy=False)
    background = {int(value) for value in background_values}
    foreground = labels_int[~np.isin(labels_int, list(background))]
    unique, counts = np.unique(foreground, return_counts=True)
    class_counts = {int(label): int(count) for label, count in zip(unique, counts)}
    positive_counts = [count for count in class_counts.values() if count > 0]
    imbalance_ratio = (
        float(max(positive_counts) / min(positive_counts)) if positive_counts else 0.0
    )

    return {
        "shape": tuple(int(value) for value in labels_int.shape),
        "labeled_pixels": int(foreground.size),
        "background_pixels": int(labels_int.size - foreground.size),
        "class_count": len(class_counts),
        "class_counts": class_counts,
        "imbalance_ratio": imbalance_ratio,
        "empty_classes": [],
    }


def recommend_dataset_usage(
    expected_classes: int,
    audit: dict[str, object],
    has_fixed_split: bool,
    severe_imbalance_threshold: float = 50.0,
) -> dict[str, object]:
    reasons: list[str] = []
    class_count = int(audit["class_count"])
    if class_count != expected_classes:
        reasons.append(
            f"class count mismatch: expected {expected_classes}, found {class_count}"
        )
    if int(audit["labeled_pixels"]) <= 0:
        reasons.append("no labeled pixels were found")

    if reasons:
        return {"status": "do_not_use_yet", "reasons": reasons}

    if not has_fixed_split:
        reasons.append("fixed train/test split is missing")
    if float(audit["imbalance_ratio"]) >= severe_imbalance_threshold:
        reasons.append("severe class imbalance requires AA/Kappa and per-class reporting")

    if reasons:
        return {"status": "smoke_then_protocolize", "reasons": reasons}
    return {"status": "use_for_formal_experiments", "reasons": ["dataset audit passed"]}
