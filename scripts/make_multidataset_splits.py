import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brmnet_core.data import build_random_split, save_coordinate_split
from brmnet_core.data.trento import load_trento_scene


DEFAULT_TRAIN_COUNTS = {
    "trento": {
        1: 129,
        2: 125,
        3: 105,
        4: 154,
        5: 184,
        6: 122,
    },
}


def parse_train_counts(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    if not value:
        raise ValueError("train counts must not be empty")
    for item in value.split(","):
        class_id, count = item.split(":", maxsplit=1)
        result[int(class_id)] = int(count)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create fixed coordinate splits for non-Houston datasets.")
    parser.add_argument("--dataset", choices=("trento",), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-counts", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset == "trento":
        scene = load_trento_scene(args.root)
        train_counts = (
            parse_train_counts(args.train_counts)
            if args.train_counts
            else DEFAULT_TRAIN_COUNTS["trento"]
        )
        split = build_random_split(scene.gt, train_counts, seed=args.seed)
    else:
        raise ValueError(f"unsupported dataset: {args.dataset}")

    save_coordinate_split(args.output, split)
    summary = {
        "dataset": args.dataset,
        "seed": split.seed,
        "train_samples": int(len(split.train_coords)),
        "test_samples": int(len(split.test_coords)),
        "train_counts": {
            str(class_id): int((split.train_labels == class_id).sum())
            for class_id in sorted(set(split.train_labels.tolist()))
        },
    }
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
