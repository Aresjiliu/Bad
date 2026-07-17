# 基于评审意见的下一步切实改进计划（2026-07-17）

## 1. 总判断

参考 `D:\Download\BRM-Net_全面评审与调研指导意见 (1).md` 后，当前项目不建议推倒重来。更稳妥的路线是保留已经形成的 BRM-Net 主线，把论文定位从“简单轻量化 + 模态缺失”收束为：

> 面向多模态遥感分类的资源预算、模态可用性与模态退化联合约束，设计一个可以真实导出 compact model 的 fusion-compatible lightweight fusion 框架，并用 accuracy-efficiency-robustness 三维证据证明部署价值。

现有代码和实验已经支撑了三件事：可导出预算结构、退化感知可靠性融合、多数据集三 seed 评价。当前最主要短板不是没有工作量，而是证据链还不够闭合：关键消融不足、质量监督目标偏经验、路由标签仍偏 state-level、复现与图表证据还需要冻结。

## 2. 评审意见到当前状态的映射

### 已基本具备的部分

- 真实 compact export：已有 hard-concrete gate、compact exporter、source-vs-compact 测试和 MAC/Params 统计基础。
- 多状态鲁棒性：已有 full、main-only、aux-only、noise-high、downsample4、occlusion50 等评估状态。
- 多数据集工作量：Houston2013、Trento、MUUFL 已完成三 seed formal evidence。
- 质量探针与可靠性：已有 pre-encoder quality probe、quality/uncertainty 诊断、退化监督训练选项。
- Profile/Pareto routing：已有 65/80/100 profile bank、state-level LOO router、validation-derived router、mean-profile stable-label router。
- 论文工程：外部 LaTeX 稿件已经有方法、实验、消融、表格和图件基础，格式问题基本不是当前主矛盾。

### 仍需补齐的硬伤

- 缺少评审最看重的对照：`soft-mask-only`、`w/o availability mask`、`w/o degradation quality`、`uniform width scaling`、`w/o terminal tie`。
- 质量监督目标还偏固定经验值，需要补充连续退化目标或 rank-guided target，避免被认为是手工标签。
- 路由目前主要是状态级样本，尚未做到 patch-level quality feature 和 oracle/regret 记录。
- 退化曲线还不够连续，论文中需要从单点退化扩展为 noise/downsample/occlusion 曲线。
- 复现实验清单不够完整，需要冻结 git commit、conda 环境、数据协议、run config、metrics hash、compact export hash。
- 演示系统尚未落地，答辩时缺少“部署价值”的可视化入口。

## 3. 下一步优先级

### P0：先把论文证据链封口（1-2 天）

目标：让当前论文不再像阶段性实验记录，而是一个可复现、可审稿的完整方法。

切实措施：

1. 新增实验冻结脚本 `scripts/freeze_experiment_manifest.py`。
   - 输入：formal run 根目录、profile run 根目录、paper figure/table 根目录。
   - 输出：`docs/generated/experiment_manifest_*.json`。
   - 记录：git commit、conda env、CUDA/PyTorch 版本、数据集路径摘要、配置参数、metrics.json 的 sha256、compact 模型参数量/MACs。
2. 新增稿件内部表述扫描脚本 `scripts/scan_manuscript_internal_terms.py`。
   - 扫描 `prototype`、`preliminary`、`remaining`、`server log`、`thesis only`、`should be checked` 等内部过程词。
   - 输出可直接修订的行号清单。
3. 在论文实验节补充 latency protocol 表。
   - 明确 GPU/CPU、batch size、patch size、precision、warmup、repeat、CUDA sync、是否包含数据传输、是否使用 exported compact model。
4. 把现有多数据集三 seed 表、MUUFL per-class/confusion、routing policy 表统一成“主表 + 补充表”的叙事。

完成标准：

- 论文中不再出现内部工作流措辞。
- 每个核心表格能追溯到一个 metrics 文件和一个 git commit。
- compact export 的资源收益不只停留在 gate ratio，而有参数量、MACs、latency 与 manifest 证据。

### P1：补最关键的审稿消融（3-5 天）

目标：回应“方法是不是只是普通剪枝、普通 mask、普通退化增强”的质疑。

优先做 4 个低风险高价值对照：

1. `soft-mask-only`
   - 训练时保留 gate/budget loss，但评估时不导出 compact model，只用源模型 soft/hard mask。
   - 对比点：证明 BRM-Net 的贡献不是训练时 mask，而是能导出真实 compact 结构。
2. `w/o availability mask`
   - 去掉 fusion 权重中的 availability 约束，只保留质量/可靠性得分。
   - 对比点：证明缺失模态不是简单 zero-fill 能处理，availability-conditioned fusion 必要。
3. `w/o degradation quality`
   - 保留预算导出和模态 dropout，但去掉退化质量监督。
   - 对比点：证明质量监督对 noise/downsample/occlusion 不是装饰项。
4. `uniform width scaling`
   - 构造一个固定 80% 宽度的双分支 baseline。
   - 对比点：证明结构化预算搜索优于手工等比例缩放。

视时间补 2 个中等成本对照：

5. `w/o terminal tie`
   - 如果当前 exporter 已有终端 gate 绑定，则增加禁用开关；否则先在论文中降低该点的贡献权重，避免过度声明。
6. `L1 / Network Slimming style baseline`
   - 作为压缩类 baseline 放到补充实验，不抢主线。

建议先只在 Houston2013 seed0 跑通，若结论稳定，再扩展到 seed0/1/2；如果算力紧张，Trento/MUUFL 只保留主方法和 uniform baseline。

完成标准：

- 至少有一张 reviewer-critical ablation 表：
  `Full BRM-Net / soft-mask-only / w-o availability / w-o degradation quality / uniform width`。
- 表中同时报告 OA、AA、Kappa、MACs、Params、Latency、missing/degraded states。

### P2：把质量监督从经验标签升级为可解释目标（3-7 天）

目标：解决评审指出的“quality target 太经验、公式太简陋”的问题。

切实措施：

1. 增加 `quality_target_mode`：
   - `fixed`：保留当前固定退化目标，作为历史兼容。
   - `continuous`：根据退化强度生成连续质量目标。
   - `rank`：只约束 clean > mild > severe 的排序关系。
2. 连续目标建议：
   - noise：`q = exp(-alpha * sigma / sigma_max)` 或 `q = 1 - sigma / sigma_max`。
   - downsample：`q = 1 / r` 或 `q = exp(-beta * log2(r))`。
   - occlusion：`q = 1 - area_ratio`。
3. rank-guided target：
   - 对 clean、mild、severe 质量分数加 margin ranking loss。
   - 用于降低“绝对质量值是手工设定”的风险。
4. 图表补充：
   - 画 degradation level vs predicted quality。
   - 画 degradation level vs OA。
   - 画 predicted quality vs routing budget。

完成标准：

- 论文方法节有一段正式的 quality target formulation。
- 至少有一个 continuous/rank 版本在 Houston2013 seed0 上跑通。
- 如果性能不优于 fixed，也可写成“fixed target is retained for stability, continuous/rank target is analyzed as a calibration alternative”，不能硬吹。

### P3：把 routing 从状态级推向 patch-level 证据（4-7 天）

目标：让 quality-conditioned budget routing 不再只是 11 个 degradation state 的小样本实验。

切实措施：

1. 新增 patch-level routing dump：
   - 对 validation/test patch 记录 `dataset`、`seed`、`state`、`label`、`pred`、`correct`、`pre_q_main`、`pre_q_aux`、`pre_u_main`、`pre_u_aux`、`confidence`。
2. 基于 profile bank 生成 patch-level oracle/regret：
   - 对 65/80/100 profile 分别记录该 patch 是否预测正确。
   - 用 Pareto tolerance 或 utility rule 生成 patch-level profile label。
3. 训练轻量 router：
   - 输入：quality、uncertainty、availability、confidence。
   - 输出：65/80/100 profile。
   - 指标：achieved OA、MACs、regret、saving、routing accuracy、label entropy。
4. 论文定位：
   - 若 patch-level 稳定：作为正式方法组件。
   - 若 patch-level 不稳定：作为补充分析，主文保留 mean-profile stable-label router，诚实说明 patch-level 标签噪声。

完成标准：

- 有一个 patch-level CSV/Parquet 数据文件和 summary json。
- 有一张 router policy comparison 表，包含 static 80、static 100、Pareto oracle、state router、stable-label router、patch router。

### P4：科研图表与答辩演示系统（2-5 天）

目标：提升论文观感，并让“部署价值”在答辩中可演示。

切实措施：

1. Figure 1 保持 Training / Export / Inference 三段式。
   - Resource objective 放在上下分支中线位置。
   - 增加浅色背景框区分 budgeted encoder、fusion、compact export、availability-conditioned inference。
   - 明确显示 missing branch skip，而不是只显示 fusion mask。
2. Figure 2 从“单图拼接”改为结果组图。
   - A：Accuracy-efficiency Pareto。
   - B：continuous degradation curve。
   - C：quality calibration。
   - D：MUUFL failure/per-class analysis。
3. 新增 retained width heatmap。
   - 显示不同 budget 下各层保留通道，证明结构不是均匀缩放。
4. 演示系统 MVP：
   - 优先做静态 JSON + Streamlit 或 FastAPI + 简单前端。
   - 页面包括：数据集选择、模态状态选择、profile 选择、accuracy-efficiency-robustness 表、routing 决策、失败案例。
   - 不重新训练模型，读取现有 metrics 和 figures。

完成标准：

- 论文主图能直接解释“训练、导出、推理”的完整闭环。
- 演示系统能在答辩时展示三个问题：缺失模态怎么办、预算降低多少、鲁棒性损失多少。

## 4. 近期执行顺序

建议严格按以下顺序推进，避免发散：

1. `P0-1`：实现 experiment manifest，冻结现有结果。
2. `P0-2`：扫描并修订论文内部过程措辞。
3. `P1-1`：实现并运行 `soft-mask-only` 和 `uniform width`，先 Houston2013 seed0。
4. `P1-2`：实现并运行 `w/o availability mask` 和 `w/o degradation quality`。
5. `P2-1`：实现 `quality_target_mode=fixed|continuous|rank`，先做单 seed 验证。
6. `P2-2`：生成 continuous degradation curves 和 quality calibration 图。
7. `P3-1`：导出 patch-level routing dump。
8. `P3-2`：训练 patch-level router，并与 stable-label router 比较。
9. `P4-1`：重绘 Figure 1 / Figure 2，补 retained width heatmap。
10. `P4-2`：做 JSON-backed demo MVP。

## 5. 论文写作取舍

主文应突出四个贡献：

1. Fusion-compatible budgeted structural export：不是普通软 mask，而是可导出 compact model。
2. Degradation-calibrated reliability fusion：不是只处理模态缺失，还处理噪声、分辨率下降、遮挡。
3. Availability-conditioned compact inference：缺失模态不仅在融合处 mask，还应体现推理路径/资源统计。
4. Accuracy-efficiency-robustness Pareto evaluation：用三维 trade-off 评价部署价值，而不是只报 OA。

必须避免的写法：

- 不要把 hard-concrete、budget loss、masked softmax、modality dropout、reliability score 单独说成新颖。
- 不要声称所有数据集 clean OA 都提升。
- 不要把 weighted CE 写成核心创新，它只能作为类别不均衡诊断。
- 不要把当前 state-level router 说成成熟 patch-level adaptive inference。

## 6. 停止条件

如果时间继续紧张，最低可交付版本应包含：

- 三数据集三 seed 主表。
- 真实 compact export 的 MACs/Params/Latency 证据。
- `soft-mask-only`、`uniform width`、`w/o availability`、`w/o degradation quality` 四个关键消融。
- continuous degradation curve 或 quality calibration 至少一类图。
- routing policy comparison 表，其中明确 stable-label router 的收益和 patch-level 的状态。
- 完整 experiment manifest。

达到这些条件后，论文即使单点算法创新不强，也能通过“问题定义完整、实验闭环扎实、部署证据充分、失败分析诚实”来支撑硕士毕业工作量。
