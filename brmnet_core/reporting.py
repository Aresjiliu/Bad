from __future__ import annotations

import csv
import json
from pathlib import Path


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def write_metrics_csv(path: str | Path, metrics_by_mode: dict[str, dict[str, object]]) -> None:
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
            writer.writerow(
                {"mode": mode, **{key: _csv_value(value) for key, value in metrics.items()}}
            )


def write_metrics_json(
    path: str | Path,
    metrics_by_mode: dict[str, dict[str, object]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics_by_mode, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
