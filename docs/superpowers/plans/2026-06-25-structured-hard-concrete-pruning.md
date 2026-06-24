# Structured Hard-Concrete Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 BRM-Net 实现 BN 后 Hard-Concrete 通道门控、真实 Params/MACs 预算、融合接口共享约束和可验证的紧凑模型导出。

**Architecture:** 保留旧 `BudgetGatedConv2d` 以复现历史结果，新增独立 `HardConcreteGate` 和显式 Conv/BN/Gate/ReLU 模块。新 BRM-Net 使用模型级共享融合门控；资源估计器按结构依赖计算预期与硬 Params/MACs；专用 exporter 将硬掩码复制到无门控的 `CompactBRMNet`。

**Tech Stack:** Python 3.10, PyTorch 2.5.1, unittest, CUDA 12.1.

---

### Task 1: Hard-Concrete Gate

**Files:**
- Create: `brmnet_core/hard_concrete.py`
- Create: `tests/test_hard_concrete.py`
- Modify: `brmnet_core/__init__.py`

- [x] **Step 1: Write failing probability and initialization tests**

```python
gate = HardConcreteGate(4, initial_retention=0.65)
self.assertTrue(torch.allclose(
    gate.expected_active_probability(),
    torch.full((4,), 0.65),
    atol=1e-5,
))
```

Also assert invalid channel counts, retention, temperature and stretch bounds raise `ValueError`.

- [x] **Step 2: Run tests and verify RED**

Run:

```powershell
D:\software\anaconda3\envs\hslinets\python.exe -m unittest tests.test_hard_concrete -v
```

Expected: import failure because `HardConcreteGate` does not exist.

- [x] **Step 3: Implement initialization and expected probability**

Implement:

```python
class HardConcreteGate(nn.Module):
    def __init__(
        self,
        channels: int,
        initial_retention: float = 0.9,
        temperature: float = 2.0 / 3.0,
        lower: float = -0.1,
        upper: float = 1.1,
    ) -> None: ...

    def expected_active_probability(self) -> torch.Tensor:
        offset = self.temperature * math.log(-self.lower / self.upper)
        return torch.sigmoid(self.log_alpha - offset)
```

Initialize `log_alpha` by inverting this equation.

- [x] **Step 4: Write failing forward-mode tests**

Test that:

- training samples remain in `[0, 1]`;
- evaluation soft mode is deterministic;
- hard mode returns only 0/1 values;
- output has shape `[1, C, 1, 1]`;
- gradients reach `log_alpha`;
- `set_inference_mode` rejects unknown modes.

- [x] **Step 5: Implement soft sampling, hard masks and forward**

Use reparameterized binary concrete sampling in training. In evaluation:

- `soft`: stretched sigmoid clamped to `[0,1]`;
- `hard`: `expected_active_probability() >= threshold`.

- [x] **Step 6: Run tests and commit**

Run the focused test and then:

```powershell
git add brmnet_core\hard_concrete.py brmnet_core\__init__.py tests\test_hard_concrete.py
git commit -m "Add hard concrete channel gates"
```

### Task 2: Explicit Gated Blocks and Shared Fusion Gate

**Files:**
- Create: `brmnet_core/gated_blocks.py`
- Modify: `brmnet_core/encoders.py`
- Modify: `brmnet_core/fusion.py`
- Modify: `brmnet_core/model.py`
- Create: `tests/test_structured_brmnet.py`

- [x] **Step 1: Write failing block-order test**

Construct `HardConcreteConvBlock` and assert its registered children are ordered:

```python
["conv", "bn", "gate", "activation"]
```

Verify output shape and that gate receives the BN output by registering forward hooks.

- [x] **Step 2: Run test and verify RED**

Expected: `HardConcreteConvBlock` import failure.

- [x] **Step 3: Implement focused block**

The block owns `conv`, `bn`, optional `gate`, and ReLU. It must accept an externally supplied gate so the model can reuse `shared_fusion_gate`.

- [x] **Step 4: Write failing model topology tests**

Construct:

```python
model = BRMNet(..., gate_type="hard_concrete", initial_retention=0.8)
```

Assert:

- main/aux internal gates are different objects;
- both final encoder blocks reference `model.shared_fusion_gate`;
- classifier has two independent gates;
- exactly seven unique `HardConcreteGate` instances exist;
- legacy model still contains `BudgetGatedConv2d`.

- [x] **Step 5: Implement hard-concrete encoder, head and model path**

Add `gate_type` with values `hard_concrete` and `legacy_sigmoid`. Preserve legacy constructors and state layout in the legacy path. New forward behavior remains output-compatible.

- [x] **Step 6: Run focused and regression tests**

Run:

```powershell
D:\software\anaconda3\envs\hslinets\python.exe -m unittest tests.test_hard_concrete tests.test_structured_brmnet tests.test_brmnet_engine -v
```

- [x] **Step 7: Commit**

```powershell
git add brmnet_core tests\test_structured_brmnet.py
git commit -m "Add BN-after structured BRMNet gates"
```

### Task 3: Differentiable Resource Estimator

**Files:**
- Create: `brmnet_core/resources.py`
- Create: `tests/test_resources.py`
- Modify: `brmnet_core/losses.py`
- Modify: `brmnet_core/engine.py`
- Modify: `brmnet_core/__init__.py`

- [x] **Step 1: Write failing baseline resource test**

For a small fixed-width model, independently calculate Conv/BN/Linear parameters and MACs at patch size 7. Assert `estimate_brmnet_resources(..., mode="baseline")` matches.

- [x] **Step 2: Write failing expected and hard resource tests**

Set gate probabilities to known values and assert:

- expected resources are differentiable;
- hard resources are integers;
- reducing one gate decreases resources;
- shared fusion channels affect both encoder output convolutions and classifier input.

- [x] **Step 3: Run tests and verify RED**

Expected: resource estimator import failure.

- [x] **Step 4: Implement `BRMNetResourceStats`**

Expose:

```python
estimate_brmnet_resources(model, patch_size, mode)
resource_budget_loss(model, target_budget, patch_size, metric)
```

Modes are `baseline`, `expected`, and `hard`; metrics are `params` and `macs`. Include quality-estimator and classifier Linear costs.

- [x] **Step 5: Replace hard-concrete loss metrics**

For `gate_type="hard_concrete"`, `brmnet_loss` must use `resource_budget_loss`. Return:

- `resource_ratio`;
- `expected_params_ratio`;
- `expected_macs_ratio`;
- legacy retention metrics for legacy models only.

- [x] **Step 6: Extend engine meters and run tests**

Update scalar accumulation without changing classification metrics. Run all focused tests.

- [x] **Step 7: Commit**

```powershell
git add brmnet_core tests\test_resources.py tests\test_brmnet_engine.py
git commit -m "Add differentiable BRMNet resource budgets"
```

### Task 4: Compact Model and Exporter

**Files:**
- Create: `brmnet_core/compact.py`
- Create: `brmnet_core/export.py`
- Create: `tests/test_compact_export.py`
- Modify: `brmnet_core/__init__.py`

- [x] **Step 1: Write failing compact topology test**

Construct `CompactBRMNet` with asymmetric branch widths and assert forward output shape. Assert it contains no `HardConcreteGate` and no `BudgetGatedConv2d`.

- [x] **Step 2: Implement compact model**

Use ordinary Conv/BN/ReLU blocks and preserve the existing MQE/RGF/classifier output contract.

- [x] **Step 3: Write failing exporter shape-copy tests**

Set explicit masks for all seven gates. Export and assert every Conv, BN, MQE Linear and final Linear dimension matches selected indices.

- [x] **Step 4: Implement channel-copy helpers and exporter**

Implement:

```python
export_compact_brmnet(model, threshold=0.5)
```

Each mask must retain at least the highest-probability channel. Return model plus serializable width/index metadata.

- [x] **Step 5: Write failing numerical equivalence test**

Set source model to evaluation hard mode, export it, feed identical random inputs, and assert:

```python
torch.testing.assert_close(source_logits, compact_logits, atol=1e-5, rtol=1e-5)
```

- [x] **Step 6: Fix exporter until equivalence passes**

Copy all Conv/BN/Linear weights and running statistics using the exact selected indices.

- [x] **Step 7: Commit**

```powershell
git add brmnet_core tests\test_compact_export.py
git commit -m "Export compact structured BRMNet models"
```

### Task 5: Runner Integration and Resource Artifacts

**Files:**
- Modify: `scripts/run_brmnet_houston.py`
- Modify: `scripts/summarize_brmnet_experiments.py`
- Modify: `tests/test_run_brmnet_houston.py`
- Modify: `tests/test_summarize_brmnet_experiments.py`
- Modify: `docs/BRMNET_REAL_DATA_RUNBOOK_ZH.md`

- [x] **Step 1: Write failing CLI and path tests**

Add expectations:

```text
--gate-type hard_concrete
--budget-metric macs
--gate-threshold 0.5
```

Run names must include gate type, budget and budget metric.

- [x] **Step 2: Implement parser, model construction and loss kwargs**

Legacy stochastic/deterministic mode applies only to `legacy_sigmoid`. Hard-concrete evaluation defaults to soft mode and export uses hard mode.

- [x] **Step 3: Write failing artifact test**

Extract an artifact-writing helper, call it with a temporary directory and a real small `BRMNet`, and require:

```text
resource_stats.json
compact_model.pt
compact_config.json
```

- [x] **Step 4: Implement export and artifact writing**

Record baseline/expected/hard/compact Params and MACs, ratios, active channels and numerical equivalence error.

- [x] **Step 5: Update summary grouping**

Group by `gate_type`, `target_budget`, and `budget_metric`. Add compact Params/MACs ratios to summary output.

- [x] **Step 6: Run runner and summary tests**

- [x] **Step 7: Commit**

```powershell
git add scripts tests docs\BRMNET_REAL_DATA_RUNBOOK_ZH.md
git commit -m "Integrate structured pruning experiment runner"
```

### Task 6: Full Verification and CUDA Budget Smoke Tests

**Files:**
- Modify: `docs/STRUCTURED_PRUNING_VERIFICATION_ZH.md`
- Modify: `D:\Academic\task_plan.md`
- Modify: `D:\Academic\findings.md`
- Modify: `D:\Academic\progress.md`

- [x] **Step 1: Run all tests with real Houston data**

```powershell
$env:BRMNET_HOUSTON_DATA_ROOT='D:\Academic\HSLiNets-main\Dataset'
D:\software\anaconda3\envs\hslinets\python.exe -m unittest discover -s tests -v
```

Expected: zero failures and zero skips.

- [x] **Step 2: Run static checks**

```powershell
git diff --check
D:\software\anaconda3\envs\hslinets\python.exe -m py_compile brmnet_core\*.py scripts\run_brmnet_houston.py
```

- [x] **Step 3: Run CUDA seed-0 experiments**

Run official split, 20 epochs, MACs budgets 0.65/0.80/0.90. Execute sequentially to avoid GPU contention.

- [x] **Step 4: Verify acceptance criteria**

Check:

- compact MAC ratios are monotonic;
- target error is within 5%;
- exporter logit error is below `1e-5`;
- report actual OA difference between 65% and 90%;
- do not start three-seed experiments unless criteria pass.

- [x] **Step 5: Write verification report and update planning files**

Record failed criteria explicitly; do not relabel soft or expected ratios as real compression.

- [x] **Step 6: Final verification and commit**

Run the full suite again, inspect `git diff --check`, then commit:

```powershell
git add docs
git commit -m "Document structured pruning verification"
```

The three planning files under `D:\Academic` are outside this Git repository and are updated without adding them to this commit.
