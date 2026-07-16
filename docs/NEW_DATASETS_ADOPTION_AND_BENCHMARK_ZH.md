# Trento 与 MUUFL 新数据集采用评估与同类实验水平

日期：2026-07-16  
数据路径：

- Trento：`D:\Academic\data\Trento-main`
- MUUFL：`D:\Academic\data\MUUFLGulfport-master`

配套审计输出：

- `docs/DOWNLOADED_DATASET_AUDIT_ZH.md`
- `docs/generated/downloaded_dataset_audit.json`

## 一句话结论

两个数据集都值得继续推进，但不能立刻作为“正式主实验”直接跑。当前更稳妥的顺序是：先接入 Trento 并完成 smoke 实验，再用明确的每类训练样本数生成固定 split；随后接入 MUUFL，并把 AA/Kappa、per-class accuracy 和类别不平衡分析作为重点。两者当前共同问题是：下载包里没有直接检测到标准 train/test mask，因此必须先把划分协议写死并保存 split 文件，避免论文结果不可复现。

## 数据质量审计

| 数据集 | HSI | 辅助模态 | 标签 | 有限值 | 类别 | 标注像素 | 不平衡比 | 固定划分 | 采用建议 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | --- | --- |
| Trento | 166×600×63 | 166×600×2 | 166×600 | 通过 | 6 | 30,214 | 21.92 | 未检测到 | smoke 后协议化 |
| MUUFL | 325×220×64 | 两组 325×220×2 | 325×220 | 通过 | 11 | 53,687 | 127.03 | 未检测到 | smoke 后协议化 |

### Trento 判断

优点：

- 文件结构简单：`Italy_hsi.mat`、`Italy_lidar.mat`、`allgrd.mat`。
- HSI 为 63 波段，符合主流 Trento HS-LiDAR 设定。
- 标签类别 1..6 齐全，空间尺寸与 HSI/LiDAR 对齐。
- 数据规模小，适合第一批接入与快速调试。

风险：

- `Italy_lidar.mat` 是 2 通道，而多数文献描述为 1 个 LiDAR/DSM 通道。loader 需要提供 `--aux-channel-mode first|both|mean`。
- 当前只检测到 `allgrd.mat`，没有标准 train/test mask。必须生成并保存固定 split。
- Trento 文献 OA 通常接近 99%，如果只报告 OA，很难体现创新；应重点报告 MACs、Params、Latency、缺失/退化鲁棒性和预算路由。

结论：采用，优先级最高，但先做 smoke，不直接写成正式结果。

### MUUFL 判断

优点：

- `muufl_gulfport_campus_1_hsi_220_label.mat` 内含 64-band HSI，尺寸 325×220，符合常见 64-band MUUFL 协议。
- `sceneLabels.labels` 覆盖 11 类，类别数符合 HS-LiDAR 分类论文设定。
- LiDAR 数据存在两组对象，每组 `z` 为 325×220×2，说明可以构造多通道辅助模态。
- 类别不平衡严重，适合体现论文对困难数据集的分析深度。

风险：

- 类别不平衡比约 127，class 10/11 只有 183/269 个像素。OA 会被大类主导，必须报告 AA/Kappa/per-class。
- 下载包未检测到标准 train/test mask。需要采用文献常用 60/类、Building 600 的协议，或采用固定随机 split 并在论文中说明。
- LiDAR 两组对象如何选择需要确认：短期可先拼接或取第一组，正式实验前必须固定协议。

结论：采用，但排在 Trento 之后。MUUFL 更适合成为“难数据集/不平衡数据集”证据，而不是第一批调试对象。

## 同类任务实验水平

下表只作为目标水平与论文叙事边界，不表示我们必须追平所有 OA。不同论文的训练样本数、划分协议、是否使用预训练并不统一。

| 方法 | 年份 | Houston2013 OA | Trento OA | MUUFL OA | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| MTNet | 2023 | 90.14 | 99.21 | 89.53 | 多 Transformer，参数约 4.72M |
| DFNet | 2024 | 89.83 | 99.01 | - | 解耦 + 对比学习 + 蒸馏 |
| CAMFNet | 2025 | 94.83 | 99.49 | 83.24 | 20/类小样本 + 无标注预训练，协议更特殊 |
| MSFFRN | 2025 | 90.39 | - | 90.70 | 空频融合，MUUFL 使用 64-band，Building 类 600 训练样本 |
| HSLiNets | 2025 | 99% 级别 | 99% 级别 | - | 波段顺序/patch 设置影响极大，需谨慎引用绝对 OA |

### 目标设定

对本论文，Trento/MUUFL 的目标不应是单纯冲 SOTA，而是验证三件事：

1. 同一套预算控制和 compact export 能否跨数据集运行。
2. 缺失/退化模态下，质量感知融合是否比普通融合更稳。
3. 质量条件预算路由是否能在不同数据集上保持“接近高预算精度，同时节省 MACs”的趋势。

建议论文目标：

- Trento：full OA 应达到 98% 以上才有基本说服力；核心展示为 80/100 profile 的效率差异和缺失模态鲁棒性。
- MUUFL：full OA 若达到 85% 以上即可作为可用起点；更重要的是 AA/Kappa 和小类表现。若 OA 接近 89%/90%，可作为强结果。

## 推荐实验协议

### Trento

第一阶段 smoke：

- 使用所有标注像素生成固定随机 split。
- 训练样本数优先参考 HSLiNets 配置：class 1..6 分别为 129/125/105/154/184/122。
- 保存 split 到 `output/splits/trento_seed{seed}.json`。
- 先跑 seed0、budget 1.0、少量 epoch，验证 loader 与模型尺寸。

第二阶段正式：

- seed 0/1/2。
- budget 80/100，若时间足够补 65。
- 模态状态：full、main_only、aux_only、aux_noise_high。
- 若结果稳定，再补 occlusion/downsample。

### MUUFL

第一阶段 smoke：

- 使用 64-band scene-label 文件。
- 先选定 LiDAR 协议：建议短期使用第一组 LiDAR 的 2 通道 `z`，后续比较 `first` vs `both`。
- 训练样本数建议参考 MSFFRN：大多数类 60，Building 类 600；若小类不足，按可用数上限截断并记录。

第二阶段正式：

- seed 0/1/2。
- budget 80/100；65 作为补充。
- 必须报告 OA/AA/Kappa/per-class。
- 论文讨论中主动说明类别不平衡导致 OA 不足以代表真实性能。

## 下一步代码任务

1. 实现 `MultimodalPatchDataset`，替代 Houston 专用命名。
2. 实现 `load_trento_scene`，支持 2-channel LiDAR 的 `first|both|mean` 选择。
3. 实现固定 per-class split 生成和保存，避免每次随机变动。
4. 给 runner 增加 `--dataset trento` 与 `--dataset-only`。
5. Trento dataset-only 通过后，跑 1 epoch CUDA smoke。
6. 再接 MUUFL loader。

## 当前采用决策

- Trento：采用，立即进入代码接入。
- MUUFL：采用，但等 Trento runner 稳定后再接。
- 两者都不能直接用当前无 split 状态进入论文主表，必须先协议化。
