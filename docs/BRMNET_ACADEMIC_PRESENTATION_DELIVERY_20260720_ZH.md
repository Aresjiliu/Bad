# BRM-Net 学术汇报 PPT 交付说明（2026-07-20）

## 输出文件

- PPTX：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation.pptx`
- PDF 检查版：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation.pdf`
- 生成脚本：`D:\Academic\current\Bad\scripts\make_brmnet_academic_presentation.py`

## 使用的材料

已读取并参考当前投稿论文目录 `D:\Academic\paper_submission\brmnet_pricai2026` 中的最新论文源文件：

- `sections/00_abstract.tex`
- `sections/01_intro.tex`
- `sections/02_related_work.tex`
- `sections/03_method.tex`
- `sections/04_experiments.tex`
- `sections/05_discussion.tex`
- `sections/06_conclusion.tex`
- `paper.pdf`
- `tables/*.tex`
- `figures/generated/*.png`

用户提到的 `00_abstract(1).tex` 等带 `(1)` 后缀文件在当前论文目录中未发现，因此采用无后缀的最新 section 文件。

## Slide-by-slide 内容清单

1. **Title**：论文题目、汇报定位与三条核心信息。
2. **Research motivation**：多模态遥感轻量化的部署动机。
3. **Problem**：资源预算、模态缺失、模态退化的耦合问题。
4. **Key gap**：结构剪枝、弹性网络、多模态融合、缺失模态方法之间的空缺。
5. **BRM-Net overview**：Inputs → Gated encoders → Reliability → Masked fusion → Compact export。
6. **Problem formulation**：资源约束、availability mask、degradation operator、compact model 定义。
7. **Budget-aware channel gating**：hard-concrete gates 与 target/actual MAC 对齐。
8. **Fusion-compatible compact export**：source gated model 与 compact model 的真实导出对比。
9. **Availability-aware reliability fusion**：availability mask 的作用与 ablation。
10. **Degradation-supervised quality learning**：质量目标、退化监督和 conservative reliability 表述。
11. **Training and export pipeline**：训练、门控投影、结构导出、微调、评估五阶段流程。
12. **Experimental protocol and datasets**：Houston2013、Trento、MUUFL 的角色分工。
13. **Budget agreement results**：65/80/90 预算下 actual MAC 与 compact OA。
14. **Accuracy-efficiency trade-off**：Pareto 图与论文 overview 图。
15. **Missing-modality robustness**：modality dropout 对单模态 fallback 的影响。
16. **Degraded-modality robustness**：退化鲁棒性 ablation 与 adverse-state average。
17. **Multi-dataset evidence**：Houston / Trento / MUUFL 三数据集证据。
18. **Ablation study**：uniform fusion、noise quality、multi-degradation、w/o mask 的结论。
19. **Visualization**：retained width、reliability diagnostic、MUUFL error analysis。
20. **Discussion**：clean-robustness trade-off、证据边界和限制。
21. **Contribution summary**：三项主贡献收束。
22. **Future work and Q&A**：毕业论文扩展、增强证据、演示系统与 Q&A。

## 缺失图表 / 数据 TODO

- Houston2013 的最终 confusion matrix 或 class-wise map 尚未作为正式论文图提供。
- 最终 latency table 未纳入本 PPT 的主证据链，因为当前投稿表格主要使用 MAC/parameter ratio。
- 当前退化实验覆盖 noise、downsample、occlusion，尚未形成更密集的连续 corruption grid。

## 运行过的检查命令与结果

1. 生成 PPTX：

   `python scripts\make_brmnet_academic_presentation.py`

   结果：成功生成 `BRM-Net_Academic_Presentation.pptx`，共 22 页。

2. PowerPoint COM 打开并导出：

   使用 PowerPoint COM 将 PPTX 导出为 PDF 和逐页 PNG。

   结果：成功生成 `BRM-Net_Academic_Presentation.pdf` 和 `BRM-Net_Academic_Presentation_pages_v2`，证明 PPTX 可由 PowerPoint 正常打开。

3. 内容与占位符检查：

   使用 `python-pptx` 读取 PPTX。

   结果：`slides 22`；未发现 `待补充最终实验表格`、`Lorem`、`Ipsum`、`Click to add`、`Slidesgo`、`Presentation title` 等占位符或模板残留。

4. 视觉检查：

   已生成 contact sheet，并抽查第 1、14、16、17、19、22 页。

   第一轮发现深色页卡片文字对比度不足；已修正 `dark_card` 样式并重新导出。第二轮复查通过，未发现明显裁切、遮挡或文字溢出。

## 设计说明

本 PPT 采用深蓝、青色、灰白为主色，少量橙色强调 degradation/reliability。叙事集中在投稿主线：

1. Fusion-compatible budgeted compact export；
2. Degradation-supervised reliability fusion；
3. Accuracy-efficiency-robustness evaluation。

Routing、weighted CE、profile label stabilization 等内容仅放在 future work / extended thesis exploration 中，避免冲淡投稿论文主贡献。
