from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


CLAIMS = [
    {
        "claim_id": "C1",
        "claim": "BRM-Net exports physically compact models whose actual MAC ratios match the requested budgets.",
        "paper_location": "Table 1 / Figure 2",
        "status": "ready",
        "evidence": "E_BUDGET_65, E_BUDGET_80, E_BUDGET_90",
    },
    {
        "claim_id": "C2",
        "claim": "The exported compact model is evaluated separately from the source gated model, supporting the physical-export claim.",
        "paper_location": "Table 1 / planned compact-vs-source table",
        "status": "ready_for_table",
        "evidence": "E_SOURCE_COMPACT_FULL, E_SOURCE_COMPACT_MAIN_ONLY, E_SOURCE_COMPACT_AUX_ONLY",
    },
    {
        "claim_id": "C3",
        "claim": "Availability-aware training improves fallback behavior under missing single-modality inference.",
        "paper_location": "Table 2",
        "status": "ready",
        "evidence": "E_DROPOUT_MAIN_ONLY, E_DROPOUT_AUX_ONLY",
    },
    {
        "claim_id": "C4",
        "claim": "Degradation-supervised reliability improves robustness under noisy, low-resolution, and occluded auxiliary inputs.",
        "paper_location": "Table 3 / Figure 3",
        "status": "ready",
        "evidence": "E_ROBUST_FULL_BASELINE, E_ROBUST_UNIFORM, E_ROBUST_NOISE_ONLY, E_ROBUST_MULTI_P025",
    },
    {
        "claim_id": "C5",
        "claim": "The compact degradation-aware setting transfers to external HSI-LiDAR scenes as an accuracy-efficiency-robustness trade-off.",
        "paper_location": "Table 4",
        "status": "ready",
        "evidence": "E_MULTI_HOUSTON, E_MULTI_TRENTO, E_MULTI_MUUFL",
    },
    {
        "claim_id": "G1",
        "claim": "Uniform-width compression, w/o availability mask, and w/o fusion-compatible terminal constraint remain incomplete or need stricter naming.",
        "paper_location": "Readiness checklist",
        "status": "needs_ablation_or_rewording",
        "evidence": "GAP_UNIFORM_WIDTH, GAP_NO_AVAILABILITY, GAP_TERMINAL_TIE",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build paper-first claims/evidence artifacts for BRM-Net.")
    parser.add_argument("--priority-summary", default="docs/generated/brmnet_priority_summary.csv")
    parser.add_argument("--structured-runs", default="docs/generated/structured_pruning_multiseed_runs.csv")
    parser.add_argument("--multidataset-summary", default="docs/generated/multidataset_detailed_proposed_summary.csv")
    parser.add_argument("--output-csv", default="docs/generated/paper_canonical_results.csv")
    parser.add_argument("--output-md", default="docs/PAPER_CLAIMS_EVIDENCE_MATRIX_20260717_ZH.md")
    parser.add_argument(
        "--latex-multidataset-output",
        default="D:/Academic/paper_submission/brmnet_pricai2026/tables/multidataset_formal_evidence.tex",
    )
    parser.add_argument(
        "--repo-latex-multidataset-output",
        default="docs/generated/multidataset_formal_evidence_table.tex",
    )
    return parser


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def _std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _fmt_pct(mean: float, std: float | None = None) -> str:
    if std is None:
        return f"{mean * 100:.2f}"
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def _add_row(
    rows: list[dict[str, str]],
    *,
    evidence_id: str,
    claim_id: str,
    dataset: str,
    variant: str,
    mode: str,
    seeds: str,
    metric: str,
    mean: float,
    std: float | None,
    unit: str,
    source_file: str,
    paper_target: str,
    status: str,
    note: str,
) -> None:
    rows.append(
        {
            "evidence_id": evidence_id,
            "claim_id": claim_id,
            "dataset": dataset,
            "variant": variant,
            "mode": mode,
            "seeds": seeds,
            "metric": metric,
            "mean": f"{mean:.6f}",
            "std": "" if std is None else f"{std:.6f}",
            "formatted": _fmt_pct(mean, std) if unit == "ratio" else f"{mean:.4f}",
            "unit": unit,
            "source_file": source_file,
            "paper_target": paper_target,
            "status": status,
            "note": note,
        }
    )


def _structured_budget_rows(structured_runs: list[dict[str, str]], source_file: str) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in structured_runs:
        grouped[row["budget"]].append(row)

    rows: list[dict[str, str]] = []
    for budget in ("65", "80", "90"):
        items = grouped.get(budget, [])
        if not items:
            continue
        target = float(budget) / 100.0
        macs = [float(item["macs_ratio"]) / 100.0 for item in items]
        params = [float(item["params_ratio"]) / 100.0 for item in items]
        compact_oa = [float(item["compact_oa"]) / 100.0 for item in items]
        _add_row(
            rows,
            evidence_id=f"E_BUDGET_{budget}",
            claim_id="C1",
            dataset="Houston2013",
            variant="budgeted compact export",
            mode="full",
            seeds=str(len(items)),
            metric="actual_macs_ratio",
            mean=_mean(macs),
            std=_std(macs),
            unit="ratio",
            source_file=source_file,
            paper_target="Table 1 / Figure 2",
            status="ready",
            note=f"Target MAC ratio {target:.2f}; compact OA {_fmt_pct(_mean(compact_oa), _std(compact_oa))}%. Params {_fmt_pct(_mean(params), _std(params))}%.",
        )
    return rows


def _priority_lookup(priority_rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["variant"], row["mode"]): row for row in priority_rows}


def _priority_row(
    rows: list[dict[str, str]],
    lookup: dict[tuple[str, str], dict[str, str]],
    *,
    evidence_id: str,
    claim_id: str,
    variant: str,
    mode: str,
    metric_key: str,
    dataset: str = "Houston2013",
    paper_target: str,
    status: str = "ready",
    note: str = "",
) -> None:
    row = lookup[(variant, mode)]
    _add_row(
        rows,
        evidence_id=evidence_id,
        claim_id=claim_id,
        dataset=dataset,
        variant=variant,
        mode=mode,
        seeds=row["runs"],
        metric=metric_key,
        mean=float(row[f"{metric_key}_mean"]),
        std=float(row[f"{metric_key}_std"]),
        unit="ratio",
        source_file="docs/generated/brmnet_priority_summary.csv",
        paper_target=paper_target,
        status=status,
        note=note,
    )


def _priority_rows(priority_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    lookup = _priority_lookup(priority_rows)
    rows: list[dict[str, str]] = []

    for mode in ("full", "main_only", "aux_only"):
        _priority_row(
            rows,
            lookup,
            evidence_id=f"E_SOURCE_COMPACT_{mode.upper()}",
            claim_id="C2",
            variant="quality_multi_degradation_p025",
            mode=mode,
            metric_key="oa",
            paper_target="planned compact-vs-source table",
            status="ready_for_table",
            note=(
                "Source-vs-compact metrics are in the same summary row; source OA is "
                f"{float(lookup[('quality_multi_degradation_p025', mode)]['source_oa_mean']) * 100:.2f}%."
            ),
        )

    _priority_row(
        rows,
        lookup,
        evidence_id="E_DROPOUT_MAIN_ONLY",
        claim_id="C3",
        variant="full",
        mode="main_only",
        metric_key="oa",
        paper_target="Table 2",
        note="Compare with without_modality_dropout main_only row; full 3-seed refresh for no-dropout remains incomplete.",
    )
    _priority_row(
        rows,
        lookup,
        evidence_id="E_DROPOUT_AUX_ONLY",
        claim_id="C3",
        variant="full",
        mode="aux_only",
        metric_key="oa",
        paper_target="Table 2",
        note="Compare with without_modality_dropout aux_only row; full 3-seed refresh for no-dropout remains incomplete.",
    )

    robustness_specs = [
        ("E_ROBUST_FULL_BASELINE", "full", "Full baseline"),
        ("E_ROBUST_UNIFORM", "without_reliability_uniform_fusion", "Uniform fusion baseline; not a uniform-width baseline."),
        ("E_ROBUST_NOISE_ONLY", "quality_degradation_supervised", "Noise-only quality supervision."),
        ("E_ROBUST_MULTI_P025", "quality_multi_degradation_p025", "Selected light multi-degradation schedule."),
    ]
    for evidence_id, variant, note in robustness_specs:
        for mode in ("full", "aux_noise_high", "aux_downsample_4", "aux_occlusion_50"):
            _priority_row(
                rows,
                lookup,
                evidence_id=f"{evidence_id}_{mode.upper()}",
                claim_id="C4",
                variant=variant,
                mode=mode,
                metric_key="oa",
                paper_target="Table 3 / Figure 3",
                note=note,
            )
    return rows


def _multidataset_rows(multidataset_rows: list[dict[str, str]], source_file: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in multidataset_rows:
        dataset = row.get("Dataset", row.get("dataset", "unknown"))
        full_oa = _dataset_metric(row, "Full OA", "full_oa")
        full_oa_std = _dataset_metric(row, "", "full_oa_std", default=0.0)
        main_only = _dataset_metric_display(row, "Main-only OA", "main_only_oa", "main_only_oa_std")
        aux_only = _dataset_metric_display(row, "Aux-only OA", "aux_only_oa", "aux_only_oa_std")
        occlusion50 = _dataset_metric_display(row, "Occlusion50 OA", "occlusion50_oa", "occlusion50_oa_std")
        macs = _dataset_metric_display(row, "MACs", "macs", "macs_std")
        _add_row(
            rows,
            evidence_id=f"E_MULTI_{dataset.upper()}",
            claim_id="C5",
            dataset=dataset,
            variant="80% p=0.25 multi-degradation compact model",
            mode="full",
            seeds="3",
            metric="full_oa",
            mean=full_oa / 100.0,
            std=full_oa_std / 100.0,
            unit="ratio",
            source_file=source_file,
            paper_target="Table 4",
            status="ready",
            note=(
                f"Main-only {main_only}; Aux-only {aux_only}; "
                f"Occlusion50 {occlusion50}; MACs {macs}."
            ),
        )
    return rows


def _dataset_metric(row: dict[str, str], display_key: str, raw_key: str, default: float | None = None) -> float:
    if raw_key and raw_key in row:
        return float(row[raw_key])
    if display_key and display_key in row:
        return float(row[display_key].split("+/-")[0])
    if default is not None:
        return default
    raise KeyError(display_key or raw_key)


def _dataset_metric_display(row: dict[str, str], display_key: str, raw_key: str, raw_std_key: str) -> str:
    if display_key in row:
        return row[display_key]
    return f"{float(row[raw_key]):.2f} +/- {float(row[raw_std_key]):.2f}"


def _latex_pm(value: str) -> str:
    parts = [part.strip() for part in value.split("+/-")]
    if len(parts) == 2:
        return rf"${parts[0]}{{\pm}}{parts[1]}$"
    return value


def write_csv(rows: list[dict[str, str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_claims_md(rows: list[dict[str, str]], path: str | Path) -> None:
    evidence_by_claim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        evidence_by_claim[row["claim_id"]].append(row)

    lines = [
        "# BRM-Net 投稿版 Claims-Evidence Matrix（2026-07-17）",
        "",
        "本文件服务投稿优先路线，只记录主文 compact export、fusion compatibility、degradation-supervised reliability 相关证据。Routing、weighted CE、MUUFL 深度类别诊断转入毕业论文或补充材料。",
        "",
        "## 主张矩阵",
        "",
        "| Claim | 投稿位置 | 状态 | 证据 |",
        "|---|---|---|---|",
    ]
    for claim in CLAIMS:
        lines.append(
            f"| {claim['claim_id']}: {claim['claim']} | {claim['paper_location']} | {claim['status']} | {claim['evidence']} |"
        )

    lines.extend(["", "## Canonical Evidence", "", "| Evidence | Claim | Dataset | Variant | Mode | Metric | Value | Source | Status | Note |", "|---|---|---|---|---|---|---:|---|---|---|"])
    for row in rows:
        lines.append(
            f"| {row['evidence_id']} | {row['claim_id']} | {row['dataset']} | {row['variant']} | {row['mode']} | "
            f"{row['metric']} | {row['formatted']} | `{row['source_file']}` | {row['status']} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## 当前投稿缺口",
            "",
            "- `uniform fusion` 已完成，但它不是严格的 `uniform width scaling`。投稿中必须按真实含义命名；若要写 uniform width，需要另做固定宽度 baseline。",
            "- `w/o availability mask` 仍未形成投稿级三种子消融。若短期不补，应降低 availability mask 的独立贡献强度。",
            "- `w/o fusion-compatible terminal constraint` 当前没有稳定代码路径。不要把该消融写成已经完成。",
            "- `without_modality_dropout` 目前不是完整三种子刷新结果，不适合单独支撑最终主张，可作为早期诊断或补跑。",
        ]
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_multidataset_latex(multidataset_rows: list[dict[str, str]], path: str | Path) -> None:
    body = []
    for row in multidataset_rows:
        dataset = row.get("Dataset", row.get("dataset", "unknown"))
        full_oa = _dataset_metric_display(row, "Full OA", "full_oa", "full_oa_std")
        main_only = _dataset_metric_display(row, "Main-only OA", "main_only_oa", "main_only_oa_std")
        occlusion50 = _dataset_metric_display(row, "Occlusion50 OA", "occlusion50_oa", "occlusion50_oa_std")
        macs = _dataset_metric_display(row, "MACs", "macs", "macs_std")
        body.append(
            " & ".join(
                [
                    dataset,
                    "80\\% multi-degradation compact",
                    _latex_pm(full_oa),
                    _latex_pm(main_only),
                    _latex_pm(occlusion50),
                    _latex_pm(macs),
                ]
            )
            + r" \\"
        )
    content = r"""\begin{table*}[t]
\centering
\caption{External-scene validation of the 80\% multi-degradation compact model. Values are compact-model percentages averaged over three seeds. The table reports the same static exported-model setting across datasets; dynamic profile routing is excluded from the submission scope.}
\label{tab:multidataset_evidence}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrr}
\toprule
Dataset & Setting & Full OA & Main-only OA & Occlusion-50 OA & MACs \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> list[dict[str, str]]:
    args = build_parser().parse_args(argv)
    structured_rows = _read_csv(args.structured_runs)
    priority_rows = _read_csv(args.priority_summary)
    multidataset_rows = _read_csv(args.multidataset_summary)

    rows: list[dict[str, str]] = []
    rows.extend(_structured_budget_rows(structured_rows, args.structured_runs))
    rows.extend(_priority_rows(priority_rows))
    rows.extend(_multidataset_rows(multidataset_rows, args.multidataset_summary))

    write_csv(rows, args.output_csv)
    write_claims_md(rows, args.output_md)
    write_multidataset_latex(multidataset_rows, args.latex_multidataset_output)
    write_multidataset_latex(multidataset_rows, args.repo_latex_multidataset_output)
    return rows


if __name__ == "__main__":
    main()
