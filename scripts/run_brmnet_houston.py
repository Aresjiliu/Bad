from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brmnet_core import (
    BRMNet,
    collect_budget_stats,
    estimate_brmnet_resources,
    estimate_compact_resources,
    export_compact_brmnet,
    find_resource_budget_threshold,
    initialize_uniform_resource_budget,
    iter_hard_concrete_gates,
    profile_modality_state_latency,
    set_gate_stochastic,
    set_hard_concrete_inference_mode,
    set_hard_concrete_stochastic,
)
from brmnet_core.data import (
    build_houston_raw_loaders,
    load_houston_scene,
    write_houston_data_artifacts,
)
from brmnet_core.engine import evaluate, evaluate_degradation_matrix, train_one_epoch, unpack_batch
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
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--val-split-strategy",
        choices=("class_balanced", "random"),
        default="class_balanced",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("oa", "accuracy", "loss"),
        default="oa",
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--lambda-budget", type=float, default=1.0)
    parser.add_argument("--target-budget", type=float, choices=(0.65, 0.8, 0.9, 1.0), default=1.0)
    parser.add_argument(
        "--min-active-ratio",
        type=float,
        default=0.0,
        help="Minimum active channel ratio per hard-concrete gate during automatic export threshold search.",
    )
    parser.add_argument(
        "--gate-init-retention",
        type=float,
        default=None,
        help="Initial gate probability. Defaults to --target-budget.",
    )
    parser.add_argument("--lambda-quality", type=float, default=0.0)
    parser.add_argument(
        "--lambda-pre-quality",
        type=float,
        default=0.0,
        help="Auxiliary loss weight for the pre-encoder quality and uncertainty probe.",
    )
    parser.add_argument(
        "--pre-encoder-quality-hidden",
        type=int,
        default=16,
        help="Hidden channel width of the optional pre-encoder quality probe.",
    )
    parser.add_argument(
        "--modality-dropout-prob",
        type=float,
        default=0.0,
        help="Training-time probability of dropping exactly one modality per selected sample.",
    )
    parser.add_argument(
        "--aux-quality-degradation-prob",
        type=float,
        default=0.0,
        help="Training-time probability of degrading the auxiliary modality while keeping it available.",
    )
    parser.add_argument(
        "--aux-quality-degradation-target",
        type=float,
        default=0.5,
        help="Quality target assigned to train-time degraded auxiliary samples.",
    )
    parser.add_argument(
        "--aux-quality-degradation-types",
        default="noise",
        help="Comma-separated train-time auxiliary degradation types: noise, downsample_2, downsample_4, occlusion_25, occlusion_50.",
    )
    parser.add_argument(
        "--gate-type",
        choices=("hard_concrete", "legacy_sigmoid"),
        default="hard_concrete",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("reliability", "uniform"),
        default="reliability",
        help="Use reliability-weighted fusion or uniform available-modality fusion for ablation.",
    )
    parser.add_argument("--budget-metric", choices=("params", "macs"), default="macs")
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Hard export threshold. Defaults to an automatic target-resource projection.",
    )
    parser.add_argument("--compact-finetune-epochs", type=int, default=10)
    parser.add_argument("--compact-val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--compact-val-split-strategy",
        choices=("class_balanced", "random"),
        default="class_balanced",
    )
    parser.add_argument(
        "--compact-selection-metric",
        choices=("oa", "accuracy", "loss"),
        default="oa",
    )
    parser.add_argument(
        "--gate-mode",
        choices=("stochastic", "deterministic"),
        default="deterministic",
    )
    parser.add_argument("--aux-noise-std", type=float, default=0.1)
    parser.add_argument(
        "--latency-warmup",
        type=int,
        default=5,
        help="Warmup forward passes for state-dependent latency profiling.",
    )
    parser.add_argument(
        "--latency-iterations",
        type=int,
        default=20,
        help="Timed forward passes for state-dependent latency profiling.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--save-checkpoint", default="")
    parser.add_argument("--metrics-csv", default="")
    parser.add_argument("--output-dir", default="output/experiments")
    parser.add_argument("--dataset-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build args/model and exit without loading .mat data.")
    return parser


def retention_to_gate_score(retention: float, epsilon: float = 1e-4) -> float:
    if not 0.0 < retention <= 1.0:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    bounded = min(float(retention), 1.0 - epsilon)
    return math.log(bounded / (1.0 - bounded))


def build_run_paths(
    output_dir: str | Path,
    pair_modalities: str,
    protocol: str,
    split_seed: int,
    train_seed: int,
    gate_mode: str = "deterministic",
    target_budget: float = 1.0,
    gate_type: str = "hard_concrete",
    budget_metric: str = "macs",
) -> dict[str, Path]:
    pair_name = "-".join(normalize_pair_modalities(pair_modalities))
    gate_label = (
        gate_type
        if gate_type == "hard_concrete"
        else f"{gate_type}-{gate_mode}"
    )
    run_name = (
        f"houston2013_{pair_name}_{protocol}_splitseed{split_seed}"
        f"_trainseed{train_seed}_gate{gate_label}"
        f"_budget{round(target_budget * 100):02d}_metric{budget_metric}"
    )
    run_dir = Path(output_dir) / run_name
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.csv",
        "metrics_json": run_dir / "metrics.json",
        "checkpoint": run_dir / "checkpoint.pt",
        "config": run_dir / "config.json",
        "history": run_dir / "history.json",
        "gate_stats": run_dir / "gate_stats.json",
        "resource_stats": run_dir / "resource_stats.json",
        "compact_model": run_dir / "compact_model.pt",
        "compact_config": run_dir / "compact_config.json",
        "compact_metrics": run_dir / "compact_metrics.csv",
        "compact_metrics_json": run_dir / "compact_metrics.json",
        "compact_history": run_dir / "compact_history.json",
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_loader_for_validation(
    loader: DataLoader,
    val_fraction: float,
    seed: int,
    strategy: str = "random",
) -> tuple[DataLoader, DataLoader | None]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    if strategy not in {"random", "class_balanced"}:
        raise ValueError(f"Unsupported validation split strategy: {strategy}")
    if val_fraction == 0.0:
        return loader, None

    dataset_size = len(loader.dataset)
    if dataset_size < 2:
        return loader, None
    generator = torch.Generator().manual_seed(int(seed))
    if strategy == "class_balanced":
        train_subset, val_subset = _class_balanced_split(loader.dataset, val_fraction, generator)
    else:
        val_size = max(1, int(round(dataset_size * val_fraction)))
        val_size = min(val_size, dataset_size - 1)
        train_size = dataset_size - val_size
        train_subset, val_subset = random_split(
            loader.dataset,
            [train_size, val_size],
            generator=generator,
        )
    train_loader = DataLoader(
        train_subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def _dataset_labels(dataset) -> np.ndarray | None:
    if isinstance(dataset, Subset):
        parent_labels = _dataset_labels(dataset.dataset)
        if parent_labels is None:
            return None
        return parent_labels[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, TensorDataset) and len(dataset.tensors) >= 3:
        return dataset.tensors[2].detach().cpu().numpy()
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    return None


def _class_balanced_split(dataset, val_fraction: float, generator: torch.Generator) -> tuple[Subset, Subset]:
    labels = _dataset_labels(dataset)
    if labels is None:
        dataset_size = len(dataset)
        val_size = max(1, int(round(dataset_size * val_fraction)))
        val_size = min(val_size, dataset_size - 1)
        train_size = dataset_size - val_size
        train_subset, val_subset = random_split(
            dataset,
            [train_size, val_size],
            generator=generator,
        )
        return train_subset, val_subset

    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(dataset):
        raise ValueError("dataset labels must have the same length as the dataset")

    val_indices: list[int] = []
    for class_id in sorted(np.unique(labels).tolist()):
        class_indices = np.flatnonzero(labels == class_id)
        if len(class_indices) < 2:
            continue
        class_tensor = torch.as_tensor(class_indices, dtype=torch.long)
        permutation = torch.randperm(len(class_tensor), generator=generator)
        class_val_size = max(1, int(round(len(class_indices) * val_fraction)))
        class_val_size = min(class_val_size, len(class_indices) - 1)
        val_indices.extend(class_tensor[permutation[:class_val_size]].tolist())

    if not val_indices:
        dataset_size = len(dataset)
        val_size = max(1, int(round(dataset_size * val_fraction)))
        val_size = min(val_size, dataset_size - 1)
        val_indices = torch.randperm(dataset_size, generator=generator)[:val_size].tolist()

    val_set = set(int(index) for index in val_indices)
    train_indices = [index for index in range(len(dataset)) if index not in val_set]
    return Subset(dataset, train_indices), Subset(dataset, sorted(val_set))


def compact_selection_score(metrics: dict[str, float], metric: str) -> float:
    if metric not in metrics:
        raise KeyError(f"Missing compact selection metric: {metric}")
    value = float(metrics[metric])
    return -value if metric == "loss" else value


def load_checkpoint_if_present(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    if not checkpoint:
        return
    state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)


def _as_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _resource_payload(stats, baseline=None) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "params": int(_as_float(stats.params)),
        "macs": int(_as_float(stats.macs)),
        "params_ratio": _as_float(stats.params_ratio),
        "macs_ratio": _as_float(stats.macs_ratio),
    }
    if baseline is not None:
        payload["global_params_ratio"] = int(_as_float(stats.params)) / int(_as_float(baseline.params))
        payload["global_macs_ratio"] = int(_as_float(stats.macs)) / int(_as_float(baseline.macs))
    return payload


def write_structured_pruning_artifacts(
    model: torch.nn.Module,
    run_dir: str | Path,
    patch_size: int,
    threshold: float | None,
    sample_main: torch.Tensor,
    sample_aux: torch.Tensor,
    target_budget: float | None = None,
    budget_metric: str = "macs",
    min_active_ratio: float = 0.0,
    latency_warmup: int = 5,
    latency_iterations: int = 20,
) -> tuple[torch.nn.Module, dict[str, object]]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline = estimate_brmnet_resources(model, patch_size=patch_size, mode="baseline")
    expected = estimate_brmnet_resources(model, patch_size=patch_size, mode="expected")
    if threshold is None:
        if target_budget is None:
            raise ValueError("target_budget is required for automatic threshold projection")
        threshold, hard = find_resource_budget_threshold(
            model,
            target_budget=target_budget,
            patch_size=patch_size,
            metric=budget_metric,
            min_active_ratio=min_active_ratio,
        )
    else:
        for gate in iter_hard_concrete_gates(model):
            gate.hard_threshold = float(threshold)
        hard = estimate_brmnet_resources(model, patch_size=patch_size, mode="hard")
    set_hard_concrete_inference_mode(model, "hard")
    model.eval()
    compact, metadata = export_compact_brmnet(model, threshold=threshold)
    compact.eval()
    compact_resources = estimate_compact_resources(compact, patch_size=patch_size)
    modality_states = ("full", "main_only", "aux_only")
    state_dependent = {
        "hard": {
            state: _resource_payload(
                estimate_brmnet_resources(
                    model,
                    patch_size=patch_size,
                    mode="hard",
                    modality_state=state,
                ),
                baseline=baseline,
            )
            for state in modality_states
        },
        "compact": {
            state: _resource_payload(
                estimate_compact_resources(
                    compact,
                    patch_size=patch_size,
                    modality_state=state,
                ),
                baseline=baseline,
            )
            for state in modality_states
        },
    }
    latency_ms = {
        "hard": profile_modality_state_latency(
            model,
            sample_main,
            sample_aux,
            warmup=latency_warmup,
            iterations=latency_iterations,
        ),
        "compact": profile_modality_state_latency(
            compact,
            sample_main,
            sample_aux,
            warmup=latency_warmup,
            iterations=latency_iterations,
        ),
    }

    with torch.no_grad():
        source_logits = model(sample_main, sample_aux)["logits"]
        compact_logits = compact(sample_main, sample_aux)["logits"]
    logit_delta = source_logits - compact_logits
    equivalence_error = float(logit_delta.abs().max().detach().cpu())
    equivalence_l2_relative = float(
        (logit_delta.norm() / source_logits.norm().clamp_min(1e-12)).detach().cpu()
    )

    gate_records = []
    seen = set()
    for name, module in model.named_modules():
        if id(module) in seen or not hasattr(module, "expected_active_probability"):
            continue
        seen.add(id(module))
        probabilities = module.expected_active_probability()
        active = int(module.hard_mask(threshold).sum().detach().cpu())
        gate_records.append(
            {
                "name": name,
                "expected_retention": float(probabilities.mean().detach().cpu()),
                "active_channels": active,
                "total_channels": int(probabilities.numel()),
            }
        )

    stats: dict[str, object] = {
        "threshold": float(threshold),
        "min_active_ratio": float(min_active_ratio),
        "baseline": {
            "params": int(baseline.params),
            "macs": int(baseline.macs),
            "params_ratio": 1.0,
            "macs_ratio": 1.0,
        },
        "expected": {
            "params": _as_float(expected.params),
            "macs": _as_float(expected.macs),
            "params_ratio": _as_float(expected.params_ratio),
            "macs_ratio": _as_float(expected.macs_ratio),
        },
        "hard": {
            "params": int(hard.params),
            "macs": int(hard.macs),
            "params_ratio": _as_float(hard.params_ratio),
            "macs_ratio": _as_float(hard.macs_ratio),
        },
        "compact": {
            "params": int(compact_resources.params),
            "macs": int(compact_resources.macs),
            "params_ratio": int(compact_resources.params) / int(baseline.params),
            "macs_ratio": int(compact_resources.macs) / int(baseline.macs),
        },
        "state_dependent": state_dependent,
        "latency_ms": latency_ms,
        "equivalence_max_abs_error": equivalence_error,
        "equivalence_l2_relative_error": equivalence_l2_relative,
        "gates": gate_records,
        "structure": metadata,
    }
    (run_dir / "resource_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "compact_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {"model": compact.state_dict(), "config": metadata},
        run_dir / "compact_model.pt",
    )
    return compact, stats


def _build_model(cli_args, main_channels: int, aux_channels: int) -> tuple[BRMNet, float, float | None]:
    if cli_args.gate_type == "legacy_sigmoid":
        retention = (
            cli_args.target_budget
            if cli_args.gate_init_retention is None
            else cli_args.gate_init_retention
        )
        score = retention_to_gate_score(retention)
        model = BRMNet(
            main_channels=main_channels,
            aux_channels=aux_channels,
            num_classes=cli_args.class_num,
            init_score=score,
            gate_type="legacy_sigmoid",
            fusion_mode=cli_args.fusion_mode,
            use_pre_encoder_quality_probe=cli_args.lambda_pre_quality > 0.0,
            pre_encoder_quality_hidden=cli_args.pre_encoder_quality_hidden,
        )
        return model, retention, score

    initial_retention = (
        0.9 if cli_args.gate_init_retention is None else cli_args.gate_init_retention
    )
    model = BRMNet(
        main_channels=main_channels,
        aux_channels=aux_channels,
        num_classes=cli_args.class_num,
        gate_type="hard_concrete",
        initial_retention=initial_retention,
        fusion_mode=cli_args.fusion_mode,
        use_pre_encoder_quality_probe=cli_args.lambda_pre_quality > 0.0,
        pre_encoder_quality_hidden=cli_args.pre_encoder_quality_hidden,
    )
    if cli_args.gate_init_retention is None:
        initial_retention = initialize_uniform_resource_budget(
            model,
            target_budget=cli_args.target_budget,
            patch_size=cli_args.patch_size,
            metric=cli_args.budget_metric,
        )
    return model, initial_retention, None


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
    model, gate_init_retention, gate_init_score = _build_model(
        cli_args,
        main_channels,
        aux_channels,
    )

    if cli_args.dry_run:
        summary = {
            "data_root": legacy_args.data_root,
            "data_format": cli_args.data_format,
            "split_protocol": cli_args.split_protocol,
            "split_seed": cli_args.split_seed,
            "pair_modalities": legacy_args.pair_modalities,
            "channels": [main_channels, aux_channels],
            "class_num": cli_args.class_num,
            "target_budget": cli_args.target_budget,
            "gate_init_retention": gate_init_retention,
            "gate_type": cli_args.gate_type,
            "fusion_mode": cli_args.fusion_mode,
            "budget_metric": cli_args.budget_metric,
            "lambda_pre_quality": cli_args.lambda_pre_quality,
            "pre_encoder_quality_probe": cli_args.lambda_pre_quality > 0.0,
            "pre_encoder_quality_hidden": cli_args.pre_encoder_quality_hidden,
        }
        summary["parameters"] = sum(param.numel() for param in model.parameters())
        print(json.dumps(summary, ensure_ascii=False))
        return {"dry_run": summary}

    paths = build_run_paths(
        cli_args.output_dir,
        cli_args.pair_modalities,
        cli_args.split_protocol,
        cli_args.split_seed,
        cli_args.seed,
        cli_args.gate_mode,
        cli_args.target_budget,
        cli_args.gate_type,
        cli_args.budget_metric,
    )
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    config = vars(cli_args).copy()
    config["effective_gate_init_retention"] = gate_init_retention
    config["effective_gate_init_score"] = gate_init_score
    paths["config"].write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
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

    train_loader, val_loader = split_loader_for_validation(
        train_loader,
        val_fraction=cli_args.val_fraction,
        seed=cli_args.seed,
        strategy=cli_args.val_split_strategy,
    )
    data_metadata = {
        **data_metadata,
        "effective_train_samples": len(train_loader.dataset),
        "validation_samples": len(val_loader.dataset) if val_loader is not None else 0,
        "validation_fraction": cli_args.val_fraction,
        "validation_split_strategy": cli_args.val_split_strategy,
    }

    if cli_args.gate_type == "legacy_sigmoid":
        set_gate_stochastic(model, cli_args.gate_mode == "stochastic")
    else:
        set_hard_concrete_stochastic(model, cli_args.gate_mode == "stochastic")
    model.to(device)
    load_checkpoint_if_present(model, cli_args.checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cli_args.lr, weight_decay=cli_args.weight_decay)
    loss_kwargs = {
        "lambda_budget": cli_args.lambda_budget,
        "lambda_quality": cli_args.lambda_quality,
        "lambda_pre_quality": cli_args.lambda_pre_quality,
        "target_budget": cli_args.target_budget,
    }
    if cli_args.gate_type == "hard_concrete":
        loss_kwargs.update(
            {
                "patch_size": cli_args.patch_size,
                "budget_metric": cli_args.budget_metric,
            }
        )

    history = []
    best_source_state = copy.deepcopy(model.state_dict())
    best_source_score = float("-inf")
    best_source_epoch = 0
    best_source_validation = None
    for epoch in range(cli_args.epochs):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_kwargs=loss_kwargs,
            modality_dropout_prob=cli_args.modality_dropout_prob,
            aux_quality_degradation_prob=cli_args.aux_quality_degradation_prob,
            aux_quality_degradation_noise_std=cli_args.aux_noise_std,
            aux_quality_degradation_target=cli_args.aux_quality_degradation_target,
            aux_quality_degradation_types=cli_args.aux_quality_degradation_types,
        )
        validation_metrics = None
        selection_score = None
        if val_loader is not None:
            validation_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_kwargs=loss_kwargs,
            )
            selection_score = compact_selection_score(validation_metrics, cli_args.selection_metric)
            if selection_score > best_source_score:
                best_source_score = selection_score
                best_source_epoch = epoch + 1
                best_source_validation = validation_metrics
                best_source_state = copy.deepcopy(model.state_dict())
        epoch_record = {
            "epoch": epoch + 1,
            "elapsed_seconds": time.perf_counter() - epoch_start,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_score": selection_score,
        }
        history.append(epoch_record)
        paths["history"].write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(epoch_record, ensure_ascii=False))
    if val_loader is not None:
        model.load_state_dict(best_source_state)

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
    compact_metrics_by_mode = None
    resource_stats = None
    if cli_args.gate_type == "hard_concrete":
        sample_batch = unpack_batch(next(iter(test_loader)), device)
        compact_model, resource_stats = write_structured_pruning_artifacts(
            model=model,
            run_dir=paths["run_dir"],
            patch_size=cli_args.patch_size,
            threshold=cli_args.gate_threshold,
            sample_main=sample_batch.main,
            sample_aux=sample_batch.aux,
            target_budget=cli_args.target_budget,
            budget_metric=cli_args.budget_metric,
            min_active_ratio=cli_args.min_active_ratio,
            latency_warmup=cli_args.latency_warmup,
            latency_iterations=cli_args.latency_iterations,
        )
        compact_optimizer = torch.optim.AdamW(
            compact_model.parameters(),
            lr=cli_args.lr,
            weight_decay=cli_args.weight_decay,
        )
        compact_train_loader, compact_val_loader = split_loader_for_validation(
            train_loader,
            val_fraction=cli_args.compact_val_fraction,
            seed=cli_args.seed,
            strategy=cli_args.compact_val_split_strategy,
        )
        compact_history = []
        best_compact_state = copy.deepcopy(compact_model.state_dict())
        best_compact_score = float("-inf")
        best_compact_epoch = 0
        best_compact_validation = None
        for compact_epoch in range(cli_args.compact_finetune_epochs):
            epoch_start = time.perf_counter()
            compact_train_metrics = train_one_epoch(
                compact_model,
                compact_train_loader,
                compact_optimizer,
                device,
                loss_kwargs={
                    "lambda_budget": 0.0,
                    "lambda_quality": cli_args.lambda_quality,
                    "lambda_pre_quality": 0.0,
                },
                modality_dropout_prob=cli_args.modality_dropout_prob,
                aux_quality_degradation_prob=cli_args.aux_quality_degradation_prob,
                aux_quality_degradation_noise_std=cli_args.aux_noise_std,
                aux_quality_degradation_target=cli_args.aux_quality_degradation_target,
                aux_quality_degradation_types=cli_args.aux_quality_degradation_types,
            )
            compact_validation_metrics = None
            compact_score = None
            if compact_val_loader is not None:
                compact_validation_metrics = evaluate(
                    compact_model,
                    compact_val_loader,
                    device,
                    loss_kwargs={
                        "lambda_budget": 0.0,
                        "lambda_quality": cli_args.lambda_quality,
                        "lambda_pre_quality": 0.0,
                    },
                )
                compact_score = compact_selection_score(
                    compact_validation_metrics,
                    cli_args.compact_selection_metric,
                )
                if compact_score > best_compact_score:
                    best_compact_score = compact_score
                    best_compact_epoch = compact_epoch + 1
                    best_compact_validation = compact_validation_metrics
                    best_compact_state = copy.deepcopy(compact_model.state_dict())
            compact_record = {
                "epoch": compact_epoch + 1,
                "elapsed_seconds": time.perf_counter() - epoch_start,
                "train": compact_train_metrics,
                "validation": compact_validation_metrics,
                "selection_score": compact_score,
            }
            compact_history.append(compact_record)
            paths["compact_history"].write_text(
                json.dumps(compact_history, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps({"compact_finetune": compact_record}, ensure_ascii=False))
        if compact_val_loader is not None:
            compact_model.load_state_dict(best_compact_state)
        torch.save(
            {
                "model": compact_model.state_dict(),
                "config": resource_stats["structure"],
                "finetune_epochs": cli_args.compact_finetune_epochs,
                "best_epoch": best_compact_epoch,
                "selection_metric": cli_args.compact_selection_metric,
                "selection_score": best_compact_score if compact_val_loader is not None else None,
                "validation": best_compact_validation,
            },
            paths["compact_model"],
        )
        compact_metrics_by_mode = evaluate_degradation_matrix(
            compact_model,
            test_loader,
            device,
            loss_kwargs={
                "lambda_budget": 0.0,
                "lambda_quality": cli_args.lambda_quality,
                "lambda_pre_quality": 0.0,
            },
            aux_noise_std=cli_args.aux_noise_std,
        )
        write_metrics_csv(paths["compact_metrics"], compact_metrics_by_mode)
        write_metrics_json(paths["compact_metrics_json"], compact_metrics_by_mode)
    else:
        gate_stats = collect_budget_stats(model)
        paths["gate_stats"].write_text(
            json.dumps(
                {
                    "target_budget": cli_args.target_budget,
                    "soft_retention": float(gate_stats.soft_retention.detach().cpu()),
                    "hard_retention": gate_stats.hard_retention,
                    "active_channels": gate_stats.active_channels,
                    "total_channels": gate_stats.total_channels,
                    "layers": [
                        {
                            "name": layer.name,
                            "soft_retention": layer.soft_retention,
                            "hard_retention": layer.hard_retention,
                            "active_channels": layer.active_channels,
                            "total_channels": layer.total_channels,
                        }
                        for layer in gate_stats.layers
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

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
                "args": config,
                "data_metadata": data_metadata,
                "best_epoch": best_source_epoch,
                "selection_metric": cli_args.selection_metric,
                "selection_score": best_source_score if val_loader is not None else None,
                "validation": best_source_validation,
            },
            checkpoint_path,
        )

    print(
        json.dumps(
            {
                "metrics_csv": str(metrics_path),
                "checkpoint": str(checkpoint_path),
                "best_epoch": best_source_epoch,
                "selection_metric": cli_args.selection_metric,
                "selection_score": best_source_score if val_loader is not None else None,
                "metrics": metrics_by_mode,
                "compact_metrics": compact_metrics_by_mode,
                "resource_stats": resource_stats,
            },
            ensure_ascii=False,
        )
    )
    return compact_metrics_by_mode or metrics_by_mode


if __name__ == "__main__":
    main()
