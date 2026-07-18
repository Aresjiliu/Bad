from __future__ import annotations

import argparse
import csv
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an advisor-facing BRM-Net submission status report.")
    parser.add_argument("--canonical-csv", default="docs/generated/paper_canonical_results.csv")
    parser.add_argument("--priority-summary", default="docs/generated/brmnet_priority_summary.csv")
    parser.add_argument("--readiness-md", default="docs/PAPER_SUBMISSION_READINESS_CHECK_20260719_ZH.md")
    parser.add_argument(
        "--reliability-md",
        default="docs/generated/brmnet_controlled_corruption_reliability_curves.md",
    )
    parser.add_argument("--paper-pdf", default="D:/Academic/paper_submission/brmnet_pricai2026/paper.pdf")
    parser.add_argument("--output-md", default="docs/ADVISOR_SUBMISSION_STATUS_REPORT_20260719_ZH.md")
    return parser


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _row_by_evidence(rows: list[dict[str, str]], evidence_id: str) -> dict[str, str]:
    for row in rows:
        if row["evidence_id"] == evidence_id:
            return row
    raise KeyError(evidence_id)


def _priority_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["variant"], row["mode"]): row for row in rows}


def _fmt(row: dict[str, str]) -> str:
    return row["formatted"].replace("+/-", "±")


def _pct(value: str) -> float:
    return float(value) * 100.0


def _readiness_status(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("**Status:**"):
            return line.replace("**Status:**", "").strip()
    return "UNKNOWN"


def _extract_bullets(path: str | Path, heading: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    capture = False
    bullets: list[str] = []
    for line in lines:
        if line.strip() == heading:
            capture = True
            continue
        if capture and line.startswith("## "):
            break
        if capture and line.startswith("- "):
            bullets.append(line)
    return bullets


def build_report(
    canonical_rows: list[dict[str, str]],
    priority_rows: list[dict[str, str]],
    readiness_status: str,
    reliability_bullets: list[str],
    paper_pdf: str | Path,
) -> str:
    priority = _priority_lookup(priority_rows)
    budget65 = _row_by_evidence(canonical_rows, "E_BUDGET_65")
    budget80 = _row_by_evidence(canonical_rows, "E_BUDGET_80")
    budget90 = _row_by_evidence(canonical_rows, "E_BUDGET_90")
    source_full = _row_by_evidence(canonical_rows, "E_SOURCE_COMPACT_FULL")
    uniform_full = _row_by_evidence(canonical_rows, "E_UNIFORM_WIDTH_FULL")
    fusion_mask = _row_by_evidence(canonical_rows, "E_FUSION_MASK_MAIN_ONLY")
    robust_baseline = _row_by_evidence(canonical_rows, "E_ROBUST_FULL_BASELINE_AUX_DOWNSAMPLE_4")
    robust_p025 = _row_by_evidence(canonical_rows, "E_ROBUST_MULTI_P025_AUX_DOWNSAMPLE_4")
    houston = _row_by_evidence(canonical_rows, "E_MULTI_HOUSTON2013")
    trento = _row_by_evidence(canonical_rows, "E_MULTI_TRENTO")
    muufl = _row_by_evidence(canonical_rows, "E_MULTI_MUUFL")

    learned_full = priority[("quality_multi_degradation_p025", "full")]
    uniform_full_row = priority[("uniform_width_export", "full")]
    full_delta = _pct(uniform_full_row["oa_mean"]) - _pct(learned_full["oa_mean"])
    downsample_gain = _pct(robust_p025["mean"]) - _pct(robust_baseline["mean"])
    fusion_mask_delta = _pct(fusion_mask["mean"]) - _pct(priority[("quality_multi_degradation_p025", "main_only")]["oa_mean"])

    pdf_path = Path(paper_pdf)
    pdf_line = (
        f"`{pdf_path}`，大小 {pdf_path.stat().st_size / 1024:.1f} KB"
        if pdf_path.is_file()
        else f"`{pdf_path}`，未找到"
    )

    lines = [
        "# BRM-Net 投稿版导师汇报（2026-07-19）",
        "",
        "## 1. 当前定位",
        "",
        "当前建议先走投稿轨道，论文主线固定为：",
        "",
        "> Fusion-Compatible Budgeted Compact Export with Degradation-Supervised Reliability for Multimodal Remote Sensing Classification",
        "",
        "投稿版只证明一个集中问题：在多模态遥感分类中，能否把预算可控的结构化 compact export 与退化监督可靠性融合结合起来，在真实导出的轻量模型上保持完整、缺失和退化模态下的稳定表现。",
        "",
        "暂不把 dynamic routing、weighted CE、patch-level router、demo system 作为投稿贡献。这些内容保留到毕业论文扩展轨道。",
        "",
        "## 2. 当前稿件状态",
        "",
        f"- Readiness scanner 状态：**{readiness_status}**。",
        f"- 当前 PDF：{pdf_line}。",
        "- 当前主文已聚焦 compact export、fusion compatibility、availability mask、degradation-supervised reliability 和多数据集 trade-off。",
        "",
        "## 3. 已完成的关键证据",
        "",
        "| 证据问题 | 当前结果 | 解释 |",
        "|---|---:|---|",
        f"| 预算 65% | {_fmt(budget65)} actual MAC ratio | 证明目标预算与实际导出资源可对齐。 |",
        f"| 预算 80% | {_fmt(budget80)} actual MAC ratio | 当前主实验默认预算。 |",
        f"| 预算 90% | {_fmt(budget90)} actual MAC ratio | 形成预算 sweep，不是单点结果。 |",
        f"| Source vs compact | {_fmt(source_full)} compact full OA | 证明结果来自物理导出模型，不是只看软门控源模型。 |",
        f"| Uniform-width 对照 | {_fmt(uniform_full)} full OA | 比 learned nonuniform export 低 {abs(full_delta):.2f} pp；说明统一缩宽不是完全等价替代。 |",
        f"| w/o fusion mask | {_fmt(fusion_mask)} HSI-only OA | 比主方案低 {abs(fusion_mask_delta):.2f} pp；说明缺失模态需要 fusion availability mask。 |",
        f"| Downsample-4 robustness | {_fmt(robust_p025)} OA | 比 compact baseline 高 {downsample_gain:.2f} pp；说明退化监督对低分辨率辅助模态有效。 |",
        "",
        "## 4. 多数据集结果",
        "",
        "| 数据集 | 投稿版使用方式 | Full OA | 说明 |",
        "|---|---|---:|---|",
        f"| Houston2013 | 主实验与消融 | {_fmt(houston)} | 官方 split，证据最完整。 |",
        f"| Trento | 外部近饱和场景 | {_fmt(trento)} | 约 80% MAC 下保持接近饱和准确率。 |",
        f"| MUUFL | 困难 stress test | {_fmt(muufl)} | clean full OA 非绝对最优，但 missing/degraded modality trade-off 更有信息量。 |",
        "",
        "## 5. 可靠性曲线结论",
        "",
    ]
    lines.extend(reliability_bullets or ["- 当前 reliability curve 摘要未生成。"])
    lines.extend(
        [
            "",
            "这部分在论文中应作为机制诊断，而不是写成完全校准的物理质量估计。当前稿件已经采用保守表述。",
            "",
            "## 6. 当前风险",
            "",
            "- 创新性风险：模型本身仍偏轻量，投稿叙事必须强调“真实 compact export + 退化监督可靠性 + 多状态评估协议”的组合贡献，而不是只说轻量化。",
            "- 可靠性解释风险：q_aux 和 fusion weight 不是所有退化族都严格单调，不能写成完全校准分数。",
            "- 多数据集风险：MUUFL 是 trade-off 证据，不是 clean full OA 全面胜利，需要继续诚实表述。",
            "- 篇幅风险：routing、weighted CE、demo 等毕业论文内容不能重新进入投稿主文。",
            "",
            "## 7. 建议请导师决策的问题",
            "",
            "1. 当前投稿是否以 Houston2013 为主、Trento/MUUFL 为外部验证，而不是继续新增第四个数据集？",
            "2. MUUFL 是否保留在主文多数据集表中，还是放到补充材料以降低解释成本？",
            "3. 是否接受当前题目中的 `Fusion-Compatible Budgeted Compact Export` 表述？",
            "4. 下一步是否先进入语言润色和 related work 补强，而不是继续加实验？",
            "",
            "## 8. 下一步执行建议",
            "",
            "- 短期：把该汇报交给导师，确认投稿主线和 MUUFL 展示方式。",
            "- 中期：做 related work 引用密度审查，补 compact export、missing modality、degradation-aware fusion 三类引用。",
            "- 长期：投稿版冻结后，转入毕业论文扩展轨道，整理 routing、weighted CE、演示系统和失败分析章节。",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical_rows = _read_csv(args.canonical_csv)
    priority_rows = _read_csv(args.priority_summary)
    readiness_status = _readiness_status(args.readiness_md)
    reliability_bullets = _extract_bullets(args.reliability_md, "## Main Readout")
    content = build_report(
        canonical_rows,
        priority_rows,
        readiness_status,
        reliability_bullets,
        args.paper_pdf,
    )
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    print(f"Wrote advisor submission report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
