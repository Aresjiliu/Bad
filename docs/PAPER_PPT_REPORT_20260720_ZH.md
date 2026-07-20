# 当前论文导师汇报 PPT 制作说明（2026-07-20）

## 产物位置

- PPT：`D:\Academic\current\Bad\docs\slides\BRMNet_submission_report_20260720.pptx`
- PDF：`D:\Academic\current\Bad\docs\slides\BRMNet_submission_report_20260720.pdf`
- 生成脚本：`D:\Academic\current\Bad\scripts\make_current_paper_ppt.py`

## 模板来源

本次汇报使用 `D:\Academic\ppt\Elegant Black & White Thesis Defense by Slidesgo.pptx` 作为视觉模板基础，保留黑白论文答辩风格，但重新生成了页面结构、内容层级和图文排版。

## 汇报主线

本 PPT 面向“投稿优先”的导师沟通场景，不再把毕业论文扩展、演示系统、动态路由等内容全部塞进投稿主线，而是强调当前论文已经形成较闭环的投稿版本：

1. 研究问题：轻量化、多模态缺失、模态退化是耦合问题。
2. 方法框架：结构化门控、融合兼容导出、availability mask、退化监督可靠性分支。
3. 证据链：目标预算对齐、learned nonuniform export 优于 uniform width、availability mask 保护物理缺失模态、退化监督提升 corrupted-but-available 输入表现。
4. 多数据集定位：Houston2013 做主实验，Trento 做外部验证，MUUFL 做困难压力测试。
5. 导师决策：题目、MUUFL 展示位置、是否停止加小实验、是否补强创新性 baseline。

## 页面结构

共 13 页：

1. 标题页
2. 投稿优先与毕业论文扩展分轨
3. 问题定义
4. 方法框架
5. 目标预算与实际导出资源对齐
6. learned nonuniform export 对比 uniform width
7. availability mask 的实验含义
8. 退化监督可靠性曲线
9. 多数据集角色分工
10. 投稿准备状态
11. 当前风险与处理方式
12. 需要导师拍板的 4 个问题
13. 下一步执行计划

## 视觉检查与修正

已使用 PowerPoint COM 导出 PDF 和逐页 PNG 进行视觉检查。本轮修正过两个主要问题：

- 模板原始尺寸为 10:5.625，生成脚本最初按 13.333:7.5 排版，导致右侧内容裁切；已在脚本中显式设置 16:9 宽屏尺寸。
- 第 2 页、第 7 页部分中文说明文字过长；已开启文本自动换行并压缩为更适合汇报的短句。

导出的逐页 PNG 和 contact sheet 仅作为本地 QA 产物，已加入 `.gitignore`，不纳入仓库提交。

## 后续建议

这版 PPT 已适合给导师快速汇报当前投稿路线。下一轮如果继续改，应优先基于导师反馈调整第 12 页的决策问题，而不是继续增加页面数量。
