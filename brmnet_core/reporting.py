from __future__ import annotations

import csv
from pathlib import Path


def write_metrics_csv(path: str | Path, metrics_by_mode: dict[str, dict[str, float]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys: list[str] = []
    for metrics in metrics_by_mode.values():
        for key in metrics:
            if key not in metric_keys:
                metric_keys.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mode", *metric_keys])
        writer.writeheader()
        for mode, metrics in metrics_by_mode.items():
            writer.writerow({"mode": mode, **metrics})
