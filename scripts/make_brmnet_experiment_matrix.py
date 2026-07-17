from __future__ import annotations

import argparse
import csv
import itertools
import shlex
from pathlib import Path


def _float_label(value: float) -> str:
    return str(value).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create the prioritized BRM-Net experiment matrix and runnable commands."
    )
    parser.add_argument("--data-root", default="../data/Huston2013")
    parser.add_argument("--data-format", choices=("raw", "legacy"), default="raw")
    parser.add_argument("--output-dir", default="output/experiments_priority")
    parser.add_argument("--matrix-csv", default="docs/generated/brmnet_priority_matrix.csv")
    parser.add_argument("--commands-ps1", default="docs/generated/run_brmnet_priority_matrix.ps1")
    parser.add_argument("--python", default="python")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--compact-finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--ablation",
        choices=("core", "full", "routing_profiles"),
        default="core",
        help="core gives reviewer-critical ablations; full adds budget sensitivity; routing_profiles creates 65/80/100 profile runs.",
    )
    return parser


def _routing_profile_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    priority = 0
    for seed, budget in itertools.product(args.seeds, (0.65, 0.8, 1.0)):
        priority += 1
        rows.append(
            {
                "priority": priority,
                "run_tag": f"quality_routing_profile_budget{_float_label(budget)}_seed{seed}",
                "dataset": "Houston2013-HS-LiDAR",
                "split_protocol": "official",
                "split_seed": 42,
                "seed": seed,
                "target_budget": float(budget),
                "variant": "quality_routing_profile",
                "gate_type": "hard_concrete",
                "fusion_mode": "reliability",
                "disable_fusion_availability_mask": False,
                "compact_export_strategy": "learned_threshold",
                "lambda_budget": 1.0,
                "lambda_quality": 1.0,
                "lambda_pre_quality": 0.5,
                "pre_encoder_quality_hidden": 8,
                "aux_quality_degradation_prob": 0.25,
                "aux_quality_degradation_types": "noise,downsample_4,occlusion_50",
                "modality_dropout_prob": 0.25,
                "epochs": args.epochs,
                "compact_finetune_epochs": args.compact_finetune_epochs,
                "note": "Fixed deployable budget profile for quality-conditioned routing experiments.",
            }
        )
    return rows


def experiment_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.ablation == "routing_profiles":
        return _routing_profile_rows(args)

    budgets = args.budgets
    if args.ablation == "full":
        budgets = sorted(set([0.65, 0.8, 0.9, *budgets]))

    variants = [
        {
            "variant": "full",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Full BRM-Net: budget gates + compact export + availability-aware modality dropout.",
        },
        {
            "variant": "without_modality_dropout",
            "target_budget": None,
            "modality_dropout_prob": 0.0,
            "lambda_budget": 1.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Tests whether missing-modality robustness comes from dropout training.",
        },
        {
            "variant": "budget_only",
            "target_budget": None,
            "modality_dropout_prob": 0.0,
            "lambda_budget": 1.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Compression mechanism without explicit missing-modality training.",
        },
        {
            "variant": "without_budget_loss",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 0.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Reliability/dropout behavior without an active budget penalty.",
        },
        {
            "variant": "without_reliability_uniform_fusion",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "uniform",
            "note": "Ablates reliability weighting by averaging available modalities.",
        },
        {
            "variant": "quality_supervised",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Adds direct quality supervision so reliability routing can learn degradation-aware scores.",
        },
        {
            "variant": "quality_degradation_supervised",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.5,
            "aux_quality_degradation_types": "noise",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Trains reliability scores with degraded-but-available auxiliary samples.",
        },
        {
            "variant": "quality_multi_degradation_supervised",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.5,
            "aux_quality_degradation_types": "noise,downsample_4,occlusion_50",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Trains reliability scores with noise, resolution-loss, and occlusion auxiliary degradation.",
        },
        {
            "variant": "quality_multi_degradation_p025",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.25,
            "aux_quality_degradation_types": "noise,downsample_4,occlusion_50",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "note": "Light multi-degradation supervision for balancing clean OA and adverse-modality robustness.",
        },
        {
            "variant": "without_fusion_availability_mask",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.25,
            "aux_quality_degradation_types": "noise,downsample_4,occlusion_50",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "disable_fusion_availability_mask": True,
            "note": "Ablates masked fusion by letting unavailable zero features still enter the reliability softmax.",
        },
        {
            "variant": "uniform_width_export",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 1.0,
            "aux_quality_degradation_prob": 0.25,
            "aux_quality_degradation_types": "noise,downsample_4,occlusion_50",
            "gate_type": "hard_concrete",
            "fusion_mode": "reliability",
            "compact_export_strategy": "uniform_width",
            "note": "Ablates learned nonuniform structural export with a target-matched uniform-width compact model.",
        },
        {
            "variant": "legacy_sigmoid_reference",
            "target_budget": None,
            "modality_dropout_prob": 0.25,
            "lambda_budget": 1.0,
            "lambda_quality": 0.0,
            "aux_quality_degradation_prob": 0.0,
            "aux_quality_degradation_types": "noise",
            "gate_type": "legacy_sigmoid",
            "fusion_mode": "reliability",
            "note": "Legacy gate reference; not a deployable hard-concrete export baseline.",
        },
    ]

    rows: list[dict[str, object]] = []
    priority = 0
    for seed, budget, variant in itertools.product(args.seeds, budgets, variants):
        target_budget = float(variant["target_budget"] if variant["target_budget"] is not None else budget)
        if args.ablation == "core" and target_budget != 0.8:
            continue
        priority += 1
        run_tag = f"{variant['variant']}_budget{_float_label(target_budget)}_seed{seed}"
        rows.append(
            {
                "priority": priority,
                "run_tag": run_tag,
                "dataset": "Houston2013-HS-LiDAR",
                "split_protocol": "official",
                "split_seed": 42,
                "seed": seed,
                "target_budget": target_budget,
                "variant": variant["variant"],
                "gate_type": variant["gate_type"],
                "fusion_mode": variant["fusion_mode"],
                "disable_fusion_availability_mask": variant.get("disable_fusion_availability_mask", False),
                "compact_export_strategy": variant.get("compact_export_strategy", "learned_threshold"),
                "lambda_budget": variant["lambda_budget"],
                "lambda_quality": variant["lambda_quality"],
                "lambda_pre_quality": 0.0,
                "pre_encoder_quality_hidden": 16,
                "aux_quality_degradation_prob": variant["aux_quality_degradation_prob"],
                "aux_quality_degradation_types": variant["aux_quality_degradation_types"],
                "modality_dropout_prob": variant["modality_dropout_prob"],
                "epochs": args.epochs,
                "compact_finetune_epochs": args.compact_finetune_epochs,
                "note": variant["note"],
            }
        )
    return rows


def command_for_row(row: dict[str, object], args: argparse.Namespace) -> str:
    parts = [
        *shlex.split(args.python),
        "scripts/run_brmnet_houston.py",
        "--data-root",
        args.data_root,
        "--data-format",
        args.data_format,
        "--pair-modalities",
        "hsi+lidar",
        "--split-protocol",
        str(row["split_protocol"]),
        "--split-seed",
        str(row["split_seed"]),
        "--seed",
        str(row["seed"]),
        "--target-budget",
        str(row["target_budget"]),
        "--gate-type",
        str(row["gate_type"]),
        "--fusion-mode",
        str(row["fusion_mode"]),
        "--compact-export-strategy",
        str(row.get("compact_export_strategy", "learned_threshold")),
        "--lambda-budget",
        str(row["lambda_budget"]),
        "--lambda-quality",
        str(row["lambda_quality"]),
        "--lambda-pre-quality",
        str(row.get("lambda_pre_quality", 0.0)),
        "--pre-encoder-quality-hidden",
        str(row.get("pre_encoder_quality_hidden", 16)),
        "--aux-quality-degradation-prob",
        str(row["aux_quality_degradation_prob"]),
        "--aux-quality-degradation-types",
        str(row["aux_quality_degradation_types"]),
        "--modality-dropout-prob",
        str(row["modality_dropout_prob"]),
        "--epochs",
        str(row["epochs"]),
        "--compact-finetune-epochs",
        str(row["compact_finetune_epochs"]),
        "--batch-size",
        str(args.batch_size),
        "--budget-metric",
        "macs",
        "--output-dir",
        str(Path(args.output_dir) / str(row["variant"])),
    ]
    if row.get("disable_fusion_availability_mask"):
        parts.append("--disable-fusion-availability-mask")
    return " ".join(_quote_shell_arg(part) for part in parts)


def _quote_shell_arg(part: object) -> str:
    text = str(part)
    if "," in text:
        return "'" + text.replace("'", "''") + "'"
    return shlex.quote(text)


def write_outputs(rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    matrix_path = Path(args.matrix_csv)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    commands_path = Path(args.commands_ps1)
    commands_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "Set-Location (Split-Path -Parent $PSScriptRoot)",
        "Set-Location ..",
        "",
    ]
    for row in rows:
        lines.append(f"# priority {row['priority']}: {row['run_tag']} - {row['note']}")
        lines.append(command_for_row(row, args))
        lines.append("")
    commands_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    rows = experiment_rows(args)
    write_outputs(rows, args)
    print(f"Wrote {len(rows)} experiments to {args.matrix_csv}")
    print(f"Wrote runnable PowerShell commands to {args.commands_ps1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
