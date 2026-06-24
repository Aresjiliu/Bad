from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brmnet_core import BRMNet
from brmnet_core.data import (
    build_houston_raw_loaders,
    load_houston_scene,
    write_houston_data_artifacts,
)
from brmnet_core.engine import evaluate_degradation_matrix, train_one_epoch
from brmnet_core.legacy import (
    build_houston_args,
    get_houston_loaders,
    infer_channels,
    normalize_pair_modalities,
)
from brmnet_core.reporting import write_metrics_csv, write_metrics_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal BRM-Net Houston2013 experiment loop.")
    parser.add_argument("--data-root", default="../data/Huston2013")
    parser.add_argument("--data-format", choices=("raw", "legacy"), default="raw")
    parser.add_argument("--pair-modalities", default="hsi+lidar")
    parser.add_argument("--split-protocol", choices=("official", "random"), default="official")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-file", default="")
    parser.add_argument("--class-num", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-budget", type=float, default=1e-3)
    parser.add_argument("--lambda-quality", type=float, default=0.0)
    parser.add_argument("--aux-noise-std", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--output-dir", default="output/experiments")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build args/model and exit without loading .mat data.")
    return parser


def build_run_paths(
    output_dir: str | Path,
    pair_modalities: str,
    protocol: str,
    split_seed: int,
) -> dict[str, Path]:
    pair_name = "-".join(normalize_pair_modalities(pair_modalities))
    run_name = f"houston2013_{pair_name}_{protocol}_seed{split_seed}"
    run_dir = Path(output_dir) / run_name
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.csv",
        "metrics_json": run_dir / "metrics.json",
        "checkpoint": run_dir / "checkpoint.pt",
        "config": run_dir / "config.json",
        "history": run_dir / "history.json",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint_if_present(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def main(argv: list[str] | None = None) -> dict[str, object]:
    cli_args = build_parser().parse_args(argv)
    seed_everything(cli_args.seed)

    device = torch.device(cli_args.device)
    pair_modalities = normalize_pair_modalities(cli_args.pair_modalities)
    legacy_args = build_houston_args(
        data_root=cli_args.data_root,
        pair_modalities=pair_modalities,
        batch_size=cli_args.batch_size,
        patch_size=cli_args.patch_size,
        num_workers=cli_args.num_workers,
    )
    main_channels, aux_channels = infer_channels(legacy_args.pair_modalities)

    if cli_args.dry_run:
        summary = {
            "data_root": legacy_args.data_root,
            "data_format": cli_args.data_format,
            "split_protocol": cli_args.split_protocol,
            "split_seed": cli_args.split_seed,
            "pair_modalities": legacy_args.pair_modalities,
            "channels": [main_channels, aux_channels],
            "class_num": cli_args.class_num,
        }
        model = BRMNet(
            main_channels=main_channels,
            aux_channels=aux_channels,
            num_classes=cli_args.class_num,
        )
        summary["parameters"] = sum(param.numel() for param in model.parameters())
        print(json.dumps(summary, ensure_ascii=False))
        return {"dry_run": summary}

    paths = build_run_paths(
        cli_args.output_dir,
        cli_args.pair_modalities,
        cli_args.split_protocol,
        cli_args.split_seed,
    )
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["config"].write_text(
        json.dumps(vars(cli_args), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    data_metadata: dict[str, object]
    if cli_args.data_format == "raw":
        if pair_modalities != ["hsi", "lidar"]:
            raise ValueError("raw Houston data currently supports only hsi+lidar")
        require_roi = cli_args.split_protocol == "official" and not cli_args.split_file
        scene = load_houston_scene(cli_args.data_root, require_roi=require_roi)
        bundle = build_houston_raw_loaders(
            scene=scene,
            protocol=cli_args.split_protocol,
            split_seed=cli_args.split_seed,
            patch_size=cli_args.patch_size,
            batch_size=cli_args.batch_size,
            num_workers=cli_args.num_workers,
            split_file=cli_args.split_file or None,
        )
        write_houston_data_artifacts(paths["run_dir"], bundle, scene)
        train_loader = bundle.train_loader
        test_loader = bundle.test_loader
        data_metadata = bundle.metadata
    else:
        if cli_args.dataset_only:
            raise ValueError("--dataset-only is supported only with --data-format raw")
        train_loader, test_loader = get_houston_loaders(legacy_args)
        data_metadata = {
            "protocol": "legacy",
            "train_samples": len(train_loader.dataset),
            "test_samples": len(test_loader.dataset),
        }

    if cli_args.dataset_only:
        result = {"dataset": data_metadata, "run_dir": str(paths["run_dir"])}
        print(json.dumps(result, ensure_ascii=False))
        return result

    model = BRMNet(
        main_channels=main_channels,
        aux_channels=aux_channels,
        num_classes=cli_args.class_num,
    )
    model.to(device)
    load_checkpoint_if_present(model, cli_args.checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    loss_kwargs = {"lambda_budget": cli_args.lambda_budget, "lambda_quality": cli_args.lambda_quality}

    history = []
    for epoch in range(cli_args.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_kwargs=loss_kwargs)
        epoch_record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
        }
        history.append(epoch_record)
        paths["history"].write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(epoch_record, ensure_ascii=False))

    metrics_by_mode = evaluate_degradation_matrix(
        model,
        test_loader,
        device,
        loss_kwargs=loss_kwargs,
        aux_noise_std=cli_args.aux_noise_std,
    )
    metrics_path = Path(cli_args.metrics_csv) if cli_args.metrics_csv else paths["metrics"]
    write_metrics_csv(metrics_path, metrics_by_mode)
    write_metrics_json(paths["metrics_json"], metrics_by_mode)

    checkpoint_path = (
        Path(cli_args.save_checkpoint)
        if cli_args.save_checkpoint
        else paths["checkpoint"]
    )
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(cli_args),
                "data_metadata": data_metadata,
            },
            checkpoint_path,
        )

    print(
        json.dumps(
            {
                "metrics_csv": str(metrics_path),
                "checkpoint": str(checkpoint_path),
                "metrics": metrics_by_mode,
            },
            ensure_ascii=False,
        )
    )
    return metrics_by_mode


if __name__ == "__main__":
    main()
