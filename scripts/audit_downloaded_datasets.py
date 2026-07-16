import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brmnet_core.data.dataset_audit import audit_label_map, recommend_dataset_usage


def _as_serializable(value):
    if isinstance(value, dict):
        return {str(key): _as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _shape(array) -> list[int]:
    return [int(value) for value in np.asarray(array).shape]


def audit_trento(root: Path) -> dict[str, object]:
    hsi = loadmat(root / "Italy_hsi.mat", squeeze_me=True, struct_as_record=False)["data"]
    lidar = loadmat(root / "Italy_lidar.mat", squeeze_me=True, struct_as_record=False)["data"]
    labels = loadmat(root / "allgrd.mat", squeeze_me=True, struct_as_record=False)["mask_test"]
    label_audit = audit_label_map(labels, background_values=(0,))
    has_fixed_split = any(
        path.is_file()
        for path in (
            root / "train_mask.mat",
            root / "test_mask.mat",
            root / "TRLabel.mat",
            root / "TSLabel.mat",
        )
    )
    decision = recommend_dataset_usage(
        expected_classes=6,
        audit=label_audit,
        has_fixed_split=has_fixed_split,
    )
    return {
        "dataset": "Trento",
        "root": str(root),
        "hsi_shape": _shape(hsi),
        "aux_shape": _shape(lidar),
        "label_shape": _shape(labels),
        "hsi_finite": bool(np.isfinite(hsi).all()),
        "aux_finite": bool(np.isfinite(lidar).all()),
        "label_audit": label_audit,
        "has_fixed_split": has_fixed_split,
        "decision": decision,
        "notes": [
            "LiDAR file has two channels in this download; the current dataset spec expected one DSM channel, so loader should expose an aux-channel selection option.",
            "Only allgrd.mat was detected as a label map; no explicit train/test masks were found.",
        ],
    }


def _muufl_scene(root: Path):
    path = root / "MUUFLGulfportSceneLabels" / "muufl_gulfport_campus_1_hsi_220_label.mat"
    return loadmat(path, squeeze_me=True, struct_as_record=False)["hsi"], path


def audit_muufl(root: Path) -> dict[str, object]:
    scene, path = _muufl_scene(root)
    hsi = np.asarray(scene.Data)
    lidar_shapes = []
    for lidar_item in np.asarray(scene.Lidar).flat:
        lidar_shapes.append(_shape(lidar_item.z))
    labels = np.asarray(scene.sceneLabels.labels)
    label_audit = audit_label_map(labels, background_values=(-1, 0))
    has_fixed_split = any(
        candidate.is_file()
        for candidate in (
            root / "train_mask.mat",
            root / "test_mask.mat",
            root / "TRLabel.mat",
            root / "TSLabel.mat",
        )
    )
    decision = recommend_dataset_usage(
        expected_classes=11,
        audit=label_audit,
        has_fixed_split=has_fixed_split,
        severe_imbalance_threshold=50.0,
    )
    return {
        "dataset": "MUUFL Gulfport",
        "root": str(root),
        "source_file": str(path),
        "hsi_shape": _shape(hsi),
        "aux_shape": lidar_shapes,
        "label_shape": _shape(labels),
        "hsi_finite": bool(np.isfinite(hsi).all()),
        "aux_finite": all(
            bool(np.isfinite(np.asarray(lidar_item.z)).all())
            for lidar_item in np.asarray(scene.Lidar).flat
        ),
        "label_audit": label_audit,
        "has_fixed_split": has_fixed_split,
        "decision": decision,
        "notes": [
            "The scene-label file matches the common 64-band MUUFL protocol and includes 11 material classes.",
            "Class imbalance is severe; class 10 and 11 have only 183 and 269 labeled pixels.",
            "No explicit train/test masks were detected in the downloaded folder.",
        ],
    }


def write_markdown(path: Path, reports: list[dict[str, object]]) -> None:
    lines = [
        "# 下载数据集审计报告",
        "",
        "本报告由 `scripts/audit_downloaded_datasets.py` 生成，用于判断新下载数据集是否适合进入 BRM-Net 正式实验。",
        "",
    ]
    for report in reports:
        audit = report["label_audit"]
        decision = report["decision"]
        lines.extend(
            [
                f"## {report['dataset']}",
                "",
                f"- 路径：`{report['root']}`",
                f"- HSI shape：`{report['hsi_shape']}`",
                f"- 辅助模态 shape：`{report['aux_shape']}`",
                f"- 标签 shape：`{report['label_shape']}`",
                f"- 有限值检查：HSI `{report['hsi_finite']}`，辅助模态 `{report['aux_finite']}`",
                f"- 标注像素：{audit['labeled_pixels']}",
                f"- 类别数：{audit['class_count']}",
                f"- 类别不平衡比：{float(audit['imbalance_ratio']):.2f}",
                f"- 是否检测到固定划分：{report['has_fixed_split']}",
                f"- 采用建议：`{decision['status']}`",
                f"- 原因：{'; '.join(decision['reasons'])}",
                "",
                "类别分布：",
                "",
                "| 类别 | 像素数 |",
                "| --- | ---: |",
            ]
        )
        for label, count in sorted(audit["class_counts"].items(), key=lambda item: int(item[0])):
            lines.append(f"| {label} | {count} |")
        lines.extend(["", "备注："])
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit downloaded Trento and MUUFL datasets.")
    parser.add_argument("--trento-root", default=r"D:\Academic\data\Trento-main")
    parser.add_argument("--muufl-root", default=r"D:\Academic\data\MUUFLGulfport-master")
    parser.add_argument("--output-json", default="docs/generated/downloaded_dataset_audit.json")
    parser.add_argument("--output-md", default="docs/DOWNLOADED_DATASET_AUDIT_ZH.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    reports = [
        audit_trento(Path(args.trento_root)),
        audit_muufl(Path(args.muufl_root)),
    ]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_as_serializable(reports), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(Path(args.output_md), reports)
    print(json.dumps(_as_serializable(reports), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
