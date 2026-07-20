# BRM-Net_Academic_Presentation_ZH_v3 QA 报告

本轮目标仅修复版面问题：文字重叠、文本框溢出、布局拥挤、连线穿字、图表标签裁切和表格不可读。不改变论文主线和实验数据。

## 输出文件

- PPTX：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v3.pptx`
- PDF：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v3.pdf`
- 全部 PNG：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v3_pages\`
- Montage：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v3_montage.png`
- 生成脚本：`D:\Academic\current\Bad\scripts\make_brmnet_academic_presentation_zh_v3.py`

## 本轮重点修复

- Slide 6：重排为 Training / Compact Export / Inference 三列；导出路径下移为独立区域；模块短标签化；虚线导出路径不穿过文字。
- Slide 8：改为四步横向流程卡片；移除拥挤表格；保留单一 take-home message。
- Slide 2：修复箭头换行；部署导向评价改为更高的单文本框卡片。
- Slide 5：三项贡献卡片改为单文本框卡片，避免标题正文叠压。
- Slide 7：解释框改为单文本框卡片，避免 bullet 溢出。
- Slide 10：缩短标题，避免标题末尾异常换行。
- Slide 13：右侧说明卡片改为更高卡片，修复正文溢出。
- Slide 15：调整右侧数值与说明间距，消除重叠。
- Slide 17：底部三卡片改为更高单文本框卡片。
- 全局：正文/卡片/模块文字不低于 18pt；图表坐标轴与图例设置为不低于 15pt。

## 检查命令与结果

```powershell
$env:PYTHONPATH='D:\Academic\current\Bad\output\ppt_deps'
python scripts\make_brmnet_academic_presentation_zh_v3.py
```

结果：成功生成 22 页 `BRM-Net_Academic_Presentation_ZH_v3.pptx`。

```powershell
python C:\Users\蒋冠军\.codex\skills\pptx\scripts\office\validate.py docs\slides\BRM-Net_Academic_Presentation_ZH_v3.pptx
```

结果：`All validations PASSED!`

```powershell
Get-ChildItem docs\slides\BRM-Net_Academic_Presentation_ZH_v3_pages\*.PNG | Measure-Object
```

结果：22 张 PNG。

```powershell
# 使用 python-pptx 检查占位符和字号
placeholder_hits = 0
font_lt_18 = only footer / page number / section tag
```

说明：当前环境未安装 `markitdown`，因此改用 `python-pptx` 完成文本抽取、占位符搜索和字号统计。

## 逐页视觉 QA

| Slide | text overlap | text overflow | font < 18 pt | line crossing text | cropped chart labels | unreadable table |
|---:|---|---|---|---|---|---|
| 01 | 无 | 无 | 无 | 无 | 无 | 无 |
| 02 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 03 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 04 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 05 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 06 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 07 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 08 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 09 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 10 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 11 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 12 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 13 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 14 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 15 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 16 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 17 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 18 | 无 | 无 | 无 | 无 | 无 | 无 |
| 19 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 20 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 21 | 无 | 无 | 仅章节/页脚 | 无 | 无 | 无 |
| 22 | 无 | 无 | 无 | 无 | 无 | 无 |

## 人工视觉检查结论

已查看最终 montage 和分组 QA 大图：

- `BRM-Net_Academic_Presentation_ZH_v3_montage.png`
- `BRM-Net_Academic_Presentation_ZH_v3_QA_final_01_06.png`
- `BRM-Net_Academic_Presentation_ZH_v3_QA_final_07_12.png`
- `BRM-Net_Academic_Presentation_ZH_v3_QA_final_13_18.png`
- `BRM-Net_Academic_Presentation_ZH_v3_QA_final_19_22.png`

结论：未发现标题、正文、表格、图例之间的重叠；未发现文本框正文溢出；Slide 6 和 Slide 8 的重点问题已解决。
