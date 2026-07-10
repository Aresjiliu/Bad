from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


VARIANT_LABELS = {
    "full": "Full baseline",
    "without_reliability_uniform_fusion": "Uniform fusion",
    "quality_degradation_supervised": "Noise quality",
    "quality_multi_degradation_supervised": "Multi-deg. p=0.50",
    "quality_multi_degradation_p025": "Multi-deg. p=0.25",
}

MODE_INFO = {
    "full": ("complete", 0.0),
    "aux_noise_low": ("noise", 0.10),
    "aux_noise_mid": ("noise", 0.25),
    "aux_noise_high": ("noise", 0.50),
    "aux_downsample_2": ("resolution", 0.50),
    "aux_downsample_4": ("resolution", 0.75),
    "aux_occlusion_25": ("occlusion", 0.25),
    "aux_occlusion_50": ("occlusion", 0.50),
}


def _float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0.0)


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2:
        return 0.0
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in x_values))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in y_values))
    if x_den == 0.0 or y_den == 0.0:
        return 0.0
    return numerator / (x_den * y_den)


def _spearman(x_values: list[float], y_values: list[float]) -> float:
    return _pearson(_rank(x_values), _rank(y_values))


def load_degradation_rows(summary_csv: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variant = row["variant"]
            mode = row["mode"]
            if variant not in VARIANT_LABELS or mode not in MODE_INFO:
                continue
            family, severity = MODE_INFO[mode]
            full_drop = 0.0
            rows.append(
                {
                    "variant": variant,
                    "method": VARIANT_LABELS[variant],
                    "mode": mode,
                    "family": family,
                    "severity": severity,
                    "runs": int(row["runs"]),
                    "oa": _float(row, "oa_mean"),
                    "oa_std": _float(row, "oa_std"),
                    "q_aux": _float(row, "q_aux_mean"),
                    "q_aux_std": _float(row, "q_aux_std"),
                    "fusion_aux": _float(row, "fusion_weight_aux_mean"),
                    "fusion_aux_std": _float(row, "fusion_weight_aux_std"),
                    "macs": _float(row, "compact_macs_ratio_mean"),
                    "full_drop": full_drop,
                }
            )

    full_oa_by_variant = {
        row["variant"]: row["oa"]
        for row in rows
        if row["mode"] == "full"
    }
    for row in rows:
        row["full_drop"] = full_oa_by_variant.get(row["variant"], row["oa"]) - row["oa"]
    return rows


def compute_correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    correlations: list[dict[str, object]] = []
    metrics = (
        ("oa", "OA"),
        ("full_drop", "Full-OA drop"),
        ("q_aux", "Aux reliability"),
        ("fusion_aux", "Aux fusion weight"),
    )
    for variant in VARIANT_LABELS:
        for family in ("noise", "resolution", "occlusion"):
            group = [
                row
                for row in rows
                if row["variant"] == variant and row["family"] == family
            ]
            if len(group) < 2:
                continue
            x_values = [float(row["severity"]) for row in group]
            for metric_key, metric_label in metrics:
                y_values = [float(row[metric_key]) for row in group]
                correlations.append(
                    {
                        "variant": variant,
                        "method": VARIANT_LABELS[variant],
                        "family": family,
                        "metric": metric_label,
                        "n": len(group),
                        "pearson_r": _pearson(x_values, y_values),
                        "spearman_r": _spearman(x_values, y_values),
                    }
                )
    return correlations


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_note(
    rows: list[dict[str, object]],
    correlations: list[dict[str, object]],
    output_md: str | Path,
) -> None:
    by_variant = {variant: [] for variant in VARIANT_LABELS}
    for row in rows:
        by_variant[row["variant"]].append(row)

    lines = [
        "# BRM-Net 退化可靠性诊断分析",
        "",
        "本分析基于 `docs/generated/brmnet_priority_summary.csv` 中的三种退化状态：噪声、分辨率损失和遮挡。相关性只用于诊断可靠性分支是否随退化强度发生合理变化；由于每个退化族的点数很少，不能作为严格统计显著性结论。",
        "",
        "## 关键结论",
        "",
    ]
    p025 = by_variant["quality_multi_degradation_p025"]
    adverse_modes = {"aux_noise_high", "aux_downsample_4", "aux_occlusion_50"}
    p025_adverse = [
        row for row in p025 if row["mode"] in adverse_modes
    ]
    if p025_adverse:
        adverse_avg = sum(float(row["oa"]) for row in p025_adverse) / len(p025_adverse)
        full_oa = next(float(row["oa"]) for row in p025 if row["mode"] == "full")
        lines.append(
            f"- 当前主方案 `quality_multi_degradation_p025` 的完整模态 OA 为 {full_oa * 100:.2f}%，三个强退化状态平均 OA 为 {adverse_avg * 100:.2f}%。"
        )
    lines.extend(
        [
            "- 从结果形态看，单纯的可靠性加权不足以稳定解决退化但可用的辅助模态；必须在训练中显式制造退化并提供质量监督。",
            "- 噪声-only 监督主要修复高噪声状态，但对遮挡状态仍然不足，说明退化类型覆盖本身就是方法贡献的一部分。",
            "- p=0.25 的轻量多退化 schedule 比 p=0.50 更适合作为当前论文主结果，因为它在完整模态精度和强退化鲁棒性之间更平衡。",
            "- 需要注意，fusion weight 并不在所有退化族上都单调下降，尤其噪声场景下仍存在权重升高现象。因此论文中更稳妥的说法是“退化监督改善了鲁棒表示与融合行为”，而不是“可靠性权重总能自动抑制退化模态”。",
            "",
            "## 退化状态明细",
            "",
            "| Method | Mode | Severity | OA (%) | q_aux | Aux fusion | Full-OA drop (%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["mode"] == "full" or row["mode"] in adverse_modes:
            lines.append(
                f"| {row['method']} | {row['mode']} | {float(row['severity']):.2f} | "
                f"{float(row['oa']) * 100:.2f} | {float(row['q_aux']):.3f} | "
                f"{float(row['fusion_aux']):.3f} | {float(row['full_drop']) * 100:.2f} |"
            )

    lines.extend(
        [
            "",
            "## 相关性诊断",
            "",
            "| Method | Family | Metric | n | Pearson r | Spearman r |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in correlations:
        if row["metric"] in {"OA", "Aux fusion weight"}:
            lines.append(
                f"| {row['method']} | {row['family']} | {row['metric']} | {row['n']} | "
                f"{float(row['pearson_r']):.3f} | {float(row['spearman_r']):.3f} |"
            )

    lines.extend(
        [
            "",
            "## 下一步写作使用方式",
            "",
            "论文中应把该分析作为机制诊断，而不是主要结果表。主文可以引用退化鲁棒性表说明 p=0.25 多退化监督的收益；补充材料或汇报中再展示相关性图，解释可靠性分支为什么需要退化目标监督。写作时应避免把 q_aux 或 fusion weight 解释为完全校准的物理质量分数，当前证据更适合支持“训练目标使模型在退化状态下保持更稳定的分类性能”。",
            "",
        ]
    )
    Path(output_md).write_text("\n".join(lines), encoding="utf-8")


def plot_diagnostics(rows: list[dict[str, object]], output_prefix: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    methods = [
        "full",
        "without_reliability_uniform_fusion",
        "quality_degradation_supervised",
        "quality_multi_degradation_supervised",
        "quality_multi_degradation_p025",
    ]
    colors = {
        "full": "#0072B2",
        "without_reliability_uniform_fusion": "#999999",
        "quality_degradation_supervised": "#E69F00",
        "quality_multi_degradation_supervised": "#D55E00",
        "quality_multi_degradation_p025": "#009E73",
    }
    markers = {
        "full": "o",
        "without_reliability_uniform_fusion": "s",
        "quality_degradation_supervised": "^",
        "quality_multi_degradation_supervised": "D",
        "quality_multi_degradation_p025": "P",
    }
    by_key = {(row["variant"], row["mode"]): row for row in rows}
    adverse_modes = ["aux_noise_high", "aux_downsample_4", "aux_occlusion_50"]
    adverse_labels = ["Noise-high", "Downsample-4", "Occlusion-50"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)

    ax = axes[0]
    x_positions = range(len(adverse_modes))
    for variant in methods:
        values = [float(by_key[(variant, mode)]["oa"]) * 100 for mode in adverse_modes]
        ax.plot(
            list(x_positions),
            values,
            marker=markers[variant],
            linewidth=1.4,
            markersize=4.5,
            color=colors[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.set_xticks(list(x_positions), adverse_labels, rotation=18, ha="right")
    ax.set_ylabel("Compact OA (%)")
    ax.set_title("A. Strong degradation robustness")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    noise_modes = ["aux_noise_low", "aux_noise_mid", "aux_noise_high"]
    for variant in ("full", "quality_degradation_supervised", "quality_multi_degradation_p025"):
        x = [float(by_key[(variant, mode)]["severity"]) for mode in noise_modes]
        y = [float(by_key[(variant, mode)]["fusion_aux"]) for mode in noise_modes]
        ax.plot(
            x,
            y,
            marker=markers[variant],
            linewidth=1.4,
            markersize=4.5,
            color=colors[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.set_xlabel("Noise severity")
    ax.set_ylabel("Aux fusion weight")
    ax.set_title("B. Noise fusion diagnostic")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[2]
    occlusion_modes = ["aux_occlusion_25", "aux_occlusion_50"]
    for variant in ("full", "quality_degradation_supervised", "quality_multi_degradation_p025"):
        x = [float(by_key[(variant, mode)]["severity"]) for mode in occlusion_modes]
        y = [float(by_key[(variant, mode)]["fusion_aux"]) for mode in occlusion_modes]
        ax.plot(
            x,
            y,
            marker=markers[variant],
            linewidth=1.4,
            markersize=4.5,
            color=colors[variant],
            label=VARIANT_LABELS[variant],
        )
    ax.set_xlabel("Occlusion ratio")
    ax.set_ylabel("Aux fusion weight")
    ax.set_title("C. Occlusion fusion diagnostic")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.5, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
        frameon=False,
    )
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-csv",
        default="docs/generated/brmnet_priority_summary.csv",
    )
    parser.add_argument(
        "--output-prefix",
        default="docs/generated/brmnet_degradation_reliability",
    )
    args = parser.parse_args()
    output_prefix = Path(args.output_prefix)
    rows = load_degradation_rows(args.summary_csv)
    correlations = compute_correlations(rows)
    write_csv(rows, output_prefix.with_name(output_prefix.name + "_diagnostics.csv"))
    write_csv(correlations, output_prefix.with_name(output_prefix.name + "_correlations.csv"))
    write_note(rows, correlations, output_prefix.with_suffix(".md"))
    plot_diagnostics(rows, output_prefix)
    print(f"Wrote degradation reliability analysis to {output_prefix.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
