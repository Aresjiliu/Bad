# BRM-Net PPT 修订日志（v2，2026-07-20）

## 输出文件

- 改进版 PPTX：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v2.pptx`
- PDF 检查版：`D:\Academic\current\Bad\docs\slides\BRM-Net_Academic_Presentation_ZH_v2.pdf`
- 生成脚本：`D:\Academic\current\Bad\scripts\make_brmnet_academic_presentation_zh_v2.py`
- 公式资源：`D:\Academic\current\Bad\docs\slides\formula_assets\`

## 主要修订

1. **重构叙事结构**
   - 从原 22 页普通汇报改为 **18 页主汇报 + 4 页备份页**。
   - 将 retained width、可靠性诊断曲线、MUUFL 错误分析移入备份页，避免主汇报过密。

2. **修复公式展示**
   - 第 7、9、10 页不再直接显示 ASCII/LaTeX 源码。
   - 使用高分辨率透明公式图展示资源约束、通道门控、masked reliability fusion 和总损失。
   - 第 9 页已替换原 `softmax(log q + mask)` 简化写法，改为论文中的 masked reliability fusion 公式。

3. **重画方法总览**
   - 第 6 页从单线流程改为双模态并行分支。
   - 明确区分 Training、Compact Export、Inference 三阶段。
   - 使用实线表示前向特征流，虚线表示导出/约束路径。

4. **放大字号与减少小卡片**
   - 主体文字大幅放大，结果页使用大图、大数字和 take-home message。
   - 避免“大面积空白 + 小字号卡片”的版式。

5. **拆分过密图页**
   - 第 13 页只保留 Pareto 主图。
   - retained width、reliability diagnostic、MUUFL error analysis 分别放入备份页。

6. **统一结果页表达**
   - 每个结果页底部保留一句明确 take-home message。
   - 强调 actual MAC 对齐、missing fallback 增益、adverse average 提升和 clean-robustness trade-off。

## 页序

主汇报：

1. 标题
2. 研究动机
3. 三重约束问题
4. 现有方法缺口
5. 三项核心贡献
6. BRM-Net 总体框架
7. 预算门控与资源约束
8. 融合兼容 compact export
9. 可用性感知可靠融合
10. 退化监督可靠性
11. 实验协议
12. 预算对齐与真实导出
13. accuracy-efficiency 比较
14. 缺失模态鲁棒性
15. 连续退化鲁棒性
16. 多数据集验证
17. 消融、trade-off 与局限
18. 总结与 Q&A

备份页：

19. retained width
20. 退化可靠性诊断曲线
21. MUUFL 错误分析
22. 毕业论文扩展方向

## 未改变的内容

- 未修改任何实验结果数值。
- 未引入新数据或编造新结论。
- routing、weighted CE、profile stabilization 仍保留为毕业论文扩展，不进入投稿主线。
