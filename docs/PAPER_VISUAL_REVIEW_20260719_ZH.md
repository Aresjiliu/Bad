# BRM-Net 投稿 PDF 视觉检查记录（2026-07-19）

## 检查对象

- PDF：`D:\Academic\paper_submission\brmnet_pricai2026\paper.pdf`
- 当前页数：10 pages
- 当前 PDF 大小：约 398 KB
- 页面渲染目录：`docs/generated/submission_pdf_pages/`（本地检查用，不纳入 Git）

## 检查方法

使用 bundled Python runtime 和 `pypdfium2` 将 PDF 渲染为页面 PNG，并生成 contact sheet 做全页检查。重点检查：

- 标题页和摘要是否过密；
- Figure 1/2/3 是否可读；
- 表格是否溢出或贴边；
- 图例是否遮挡数据；
- caption 是否过长影响阅读；
- 参考文献是否异常换行或溢出。

## 检查结论

整体版面目前可以进入导师审阅。没有发现必须立即修复的视觉错误：

- 10 页均正常渲染；
- Figure 2 的四个 panel 可读，图例未遮挡柱状图；
- Figure 3 的三行三列可靠性曲线可读，颜色和线型区分度可接受；
- 主文表格没有明显越界；
- 参考文献位于第 10 页，没有发现溢出版面；
- LaTeX 日志硬错误扫描未发现 `Undefined`、`LaTeX Error`、`Fatal`、`Overfull`、`Warning: Citation`。

## 已做的视觉微调

检查中发现 Figure 2 和 Figure 3 的 caption 偏长，会占用较多版面并降低快速阅读效率。因此已在外部投稿稿中压缩两处 caption：

- `D:\Academic\paper_submission\brmnet_pricai2026\sections\04_experiments.tex`
- Figure 2 caption 改为直接说明四个 panel 的含义和横向偏移原因；
- Figure 3 caption 改为直接说明列、行、误差棒和可靠性解释边界。

压缩后第 8 页和第 9 页正文留白更自然，图文衔接更紧凑。

## 后续可选微调

这些不是阻塞项，可以放到最终润色阶段：

- Figure 2 Panel A 中 80% 附近点仍然密集，但已通过横向 offset 和阴影区缓解；
- Figure 3 顶部图例横向占用较宽，若最终模板页宽变化，可以考虑改成两行图例；
- Discussion 第 8-9 页段落较长，最终语言润色时可适当压缩；
- 参考文献中 URL 换行较碎，但当前是 underfull 而非 overfull，不影响提交。
