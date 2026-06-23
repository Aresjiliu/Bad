from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    by_ext = Counter(p.suffix.lower() or "<no_ext>" for p in files)
    by_top = Counter(p.relative_to(root).parts[0] for p in files)

    print(f"Root: {root}")
    print(f"Files: {len(files)}")
    print("\nTop-level distribution:")
    for name, count in by_top.most_common():
        print(f"{name}: {count}")

    print("\nExtension distribution:")
    for ext, count in by_ext.most_common():
        print(f"{ext}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

