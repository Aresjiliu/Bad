# BRM-Net 投稿源码包 Manifest（2026-07-19）

- Zip path: `D:\Academic\paper_submission\brmnet_pricai2026_submission_source_20260719.zip`
- File count: 20
- Source bytes before compression: 177265

## 打包原则

- 只包含 LaTeX 编译必需文件：`paper.tex`、被 `\input{}` 引用的 sections/tables、被 `\includegraphics{}` 引用的 figures、bibliography 和 `.latexmkrc`。
- 不包含构建产物：`aux/bbl/blg/fdb_latexmk/fls/log/synctex/pdf`。
- 不包含预览图、review_pages、notes、sources、pycache、私有脚本缓存和未在主文引用的 thesis-only 表图。

## Files

| Path | Kind | Bytes |
|---|---|---:|
| `.latexmkrc` | latexmk config | 255 |
| `bib/references.bib` | bibliography | 9203 |
| `figures/generated/fig_brmnet_results_overview.pdf` | referenced figure | 76126 |
| `figures/generated/fig_controlled_corruption_reliability_curves.pdf` | referenced figure | 41011 |
| `paper.tex` | root tex | 918 |
| `sections/00_abstract.tex` | tex input | 2342 |
| `sections/01_intro.tex` | tex input | 3808 |
| `sections/02_related_work.tex` | tex input | 7015 |
| `sections/03_method.tex` | tex input | 13255 |
| `sections/04_experiments.tex` | tex input | 11210 |
| `sections/05_discussion.tex` | tex input | 3446 |
| `sections/06_conclusion.tex` | tex input | 1293 |
| `tables/fusion_availability_mask_ablation.tex` | tex input | 922 |
| `tables/multidataset_detailed_proposed.tex` | tex input | 991 |
| `tables/multidataset_formal_evidence.tex` | tex input | 904 |
| `tables/prototype_hslidar_structured.tex` | tex input | 852 |
| `tables/prototype_modality_dropout.tex` | tex input | 788 |
| `tables/robustness_ablation.tex` | tex input | 817 |
| `tables/source_vs_compact_export.tex` | tex input | 1020 |
| `tables/uniform_width_export_ablation.tex` | tex input | 1089 |
