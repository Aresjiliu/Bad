# BRM-Net PPT v2 QA 报告（2026-07-20）

## 检查对象

`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v2.pptx`

## 自动检查

1. **生成命令**

   `python scripts\make_brmnet_academic_presentation_zh_v2.py`

   结果：成功生成 22 页 PPT，结构为 18 页主汇报 + 4 页备份页。

2. **PowerPoint 打开与导出**

   使用 Microsoft PowerPoint COM 打开 PPTX，并导出：

   - `BRM-Net_Academic_Presentation_ZH_v2.pdf`
   - `BRM-Net_Academic_Presentation_ZH_v2_pages_v2\幻灯片*.PNG`

   结果：导出成功，说明文件可由 PowerPoint 正常打开。

3. **Office 包结构验证**

   `python C:\Users\蒋冠军\.codex\skills\pptx\scripts\office\validate.py docs\slides\BRM-Net_Academic_Presentation_ZH_v2.pptx`

   结果：`All validations PASSED!`

4. **文本内容检查**

   使用 `python-pptx` 检查页数、占位符和公式源码泄漏。

   重点检查项：

   - `Lorem`
   - `Ipsum`
   - `Click to add`
   - `Slidesgo`
   - `Presentation title`
   - `softmax(log q + mask)`
   - `theta`
   - `lambda_b`
   - `<=`

   结果：

   - `slides 22`
   - `Lorem False`
   - `Ipsum False`
   - `Click to add False`
   - `Slidesgo False`
   - `Presentation title False`
   - `softmax(log q + mask) False`
   - `theta False`
   - `lambda_b False`
   - `<= False`
   - `z_l in False`
   - `f_theta False`

## 视觉检查

已检查：

- contact sheet：`BRM-Net_Academic_Presentation_ZH_v2_contact_sheet.png`
- 第 6 页：双分支三阶段方法图
- 第 7 页：门控与资源约束公式
- 第 9 页：masked reliability fusion 公式
- 第 18 页：总结页

第一轮发现问题：

- 第 6 页部分英文标签被强制换行，如 `HSI/main`、`Encoder`、`Classifier`。
- 第 9 页说明卡片中仍有 `a_m`、`q_m` 等 ASCII 变量。

修复：

- 第 6 页标签改为 `HSI`、`Aux`、`Enc`、`Gate`、`Q head`、`Fusion`、`Head`。
- 第 9 页说明卡片改成中文解释，不再在说明文字中暴露 ASCII 变量。

第二轮复查：

- 第 6 页：双分支结构清晰，未见裁切。
- 第 9 页：公式以渲染图展示，说明文字无 ASCII 变量泄漏。
- 第 18 页：总结页可读，无内容溢出。

## 剩余限制

- 公式当前以高分辨率透明图嵌入，不是 PowerPoint 原生可编辑公式。
- 备份页中的论文图仍来自现有论文 PNG，但已拆成单独页面以提升可读性。
- 若要进一步达到完全矢量级要求，可后续改为 SVG/EMF 公式和重绘全部实验图。

## 字号检查说明

使用 `python-pptx` 统计可识别字号时，每页最小字号多为 9/10 pt；这些来自页脚、页码和顶部章节标签。主体文字已重排为大字号：

- 普通页标题约 31 pt；
- 结果页 take-home message 约 21 pt；
- 主体解释文字约 20-22 pt；
- 表格文字约 16 pt；
- 深色总结页主体文字约 30 pt。
