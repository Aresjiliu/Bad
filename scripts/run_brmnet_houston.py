from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brmnet_core import BRMNet
from brmnet_core.engine import evaluate_degradation_matrix, train_one_epoch
from brmnet_core.legacy import build_houston_args, get_houston_loaders, infer_channels
from brmnet_core.reporting import write_metrics_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal BRM-Net Houston2013 experiment loop.")
    parser.add_argument("--data-root", default="../data/Huston2013")
    parser.add_argument("--pair-modalities", default="hsi+lidar")
    parser.add_argument("--class-num", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-budget", type=float, default=1e-3)
    parser.add_argument("--lambda-quality", type=float, default=0.0)
    parser.add_argument("--aux-noise-std", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="output/models/brmnet_core_houston.pt")
    parser.add_argument("--metrics-csv", default="output/logs/brmnet_core/houston_degradation_matrix.csv")
    parser.add_argument("--dry-run", action="store_true", help="Build args/model and exit without loading .mat data.")
    return parser


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


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    cli_args = build_parser().parse_args(argv)
    seed_everything(cli_args.seed)

    device = torch.device(cli_args.device)
    legacy_args = build_houston_args(
        data_root=cli_args.data_root,
        pair_modalities=cli_args.pair_modalities,
        batch_size=cli_args.batch_size,
        patch_size=cli_args.patch_size,
    )
    main_channels, aux_channels = infer_channels(legacy_args.pair_modalities)

    model = BRMNet(main_channels=main_channels, aux_channels=aux_channels, num_classes=cli_args.class_num)
    model.to(device)
    load_checkpoint_if_present(model, cli_args.checkpoint, device)
    if cli_args.dry_run:
        summary = {
            "data_root": legacy_args.data_root,
            "pair_modalities": legacy_args.pair_modalities,
            "channels": [main_channels, aux_channels],
            "class_num": cli_args.class_num,
            "parameters": sum(param.numel() for param in model.parameters()),
        }
        print(json.dumps(summary, ensure_ascii=False))
        return {"dry_run": summary}

    train_loader, test_loader = get_houston_loaders(legacy_args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    loss_kwargs = {"lambda_budget": cli_args.lambda_budget, "lambda_quality": cli_args.lambda_quality}

    for epoch in range(cli_args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_kwargs=loss_kwargs)
        print(json.dumps({"epoch": epoch + 1, "train": train_metrics}, ensure_ascii=False))

    metrics_by_mode = evaluate_degradation_matrix(
        model,
        test_loader,
        device,
        loss_kwargs=loss_kwargs,
        aux_noise_std=cli_args.aux_noise_std,
    )
    write_metrics_csv(cli_args.metrics_csv, metrics_by_mode)

    if cli_args.save_checkpoint:
        checkpoint_path = Path(cli_args.save_checkpoint)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": vars(cli_args)}, checkpoint_path)

    print(json.dumps({"metrics_csv": cli_args.metrics_csv, "metrics": metrics_by_mode}, ensure_ascii=False))
    return metrics_by_mode


if __name__ == "__main__":
    main()
