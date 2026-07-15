# Quality-Routed Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a testable pre-encoder modality-quality probe and a fixed-dimension branch projection primitive without changing the default BRM-Net inference contract.

**Architecture:** A standalone probe receives raw HSI and auxiliary patches before deep encoding and returns per-modality quality and uncertainty scores. Availability masks force an unavailable modality to quality zero and uncertainty one. A reusable projection block maps branch features to a common fusion width; integration with budget profiles and export is intentionally deferred until the primitives and resource accounting are established.

**Tech Stack:** Python 3, PyTorch 2.5, `unittest`, existing `hslinets` Conda environment.

---

### Task 1: Pre-encoder quality probe

**Files:**

- Create: `brmnet_core/quality_probe.py`
- Modify: `brmnet_core/__init__.py`
- Test: `tests/test_quality_probe.py`

- [x] **Step 1: Write the failing tests**

```python
from brmnet_core.quality_probe import PreEncoderQualityProbe

def test_probe_returns_bounded_modality_scores():
    probe = PreEncoderQualityProbe(main_channels=4, aux_channels=1, hidden_channels=4)
    outputs = probe(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))
    assert set(outputs) == {"pre_q_main", "pre_q_aux", "pre_u_main", "pre_u_aux"}
    for value in outputs.values():
        assert value.shape == (2, 1)
        assert torch.all((0.0 <= value) & (value <= 1.0))
```

- [x] **Step 2: Run the probe tests and verify ImportError**

Run: `conda run -n hslinets python -m unittest tests.test_quality_probe`

Expected: FAIL because `brmnet_core.quality_probe` does not exist.

- [x] **Step 3: Write the minimal probe implementation**

```python
class PreEncoderQualityProbe(nn.Module):
    def __init__(self, main_channels, aux_channels, hidden_channels=16): ...
    def forward(self, main_input, aux_input, availability_mask=None): ...
```

Each modality uses a depthwise convolution, pointwise projection, global pooling, and two scalar sigmoid heads. If `availability_mask[:, i] == 0`, return quality `0` and uncertainty `1` for that modality.

- [x] **Step 4: Run the probe tests and verify pass**

Run: `conda run -n hslinets python -m unittest tests.test_quality_probe`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add brmnet_core/quality_probe.py brmnet_core/__init__.py tests/test_quality_probe.py
git commit -m "Add pre-encoder modality quality probe"
```

### Task 2: Optional BRM-Net pre-encoder diagnostics

**Files:**

- Modify: `brmnet_core/model.py`
- Test: `tests/test_structured_brmnet.py`

- [x] **Step 1: Write the failing test**

```python
def test_model_optionally_reports_pre_encoder_quality():
    model = BRMNet(main_channels=4, aux_channels=1, num_classes=3, use_pre_encoder_quality_probe=True)
    outputs = model(torch.randn(2, 4, 7, 7), torch.randn(2, 1, 7, 7))
    assert outputs["pre_q_main"].shape == (2, 1)
    assert outputs["pre_u_aux"].shape == (2, 1)
```

- [x] **Step 2: Run the targeted test and verify TypeError**

Run: `conda run -n hslinets python -m unittest tests.test_structured_brmnet.StructuredBRMNetTest.test_model_optionally_reports_pre_encoder_quality`

Expected: FAIL because `BRMNet` does not accept `use_pre_encoder_quality_probe`.

- [x] **Step 3: Implement the opt-in diagnostic path**

Add `use_pre_encoder_quality_probe: bool = False` and `pre_encoder_quality_hidden: int = 16` constructor arguments. Instantiate the probe only when requested. Add its four values to output dictionaries only when enabled. Do not use them for current fusion or loss.

- [x] **Step 4: Run target and full tests**

Run: `conda run -n hslinets python -m unittest tests.test_structured_brmnet`

Run: `conda run -n hslinets python -m unittest discover -s tests`

Expected: PASS; default model outputs and current resource tests remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add brmnet_core/model.py tests/test_structured_brmnet.py
git commit -m "Expose optional pre-encoder quality diagnostics"
```

### Task 3: Fixed-dimension branch projection primitive

**Files:**

- Create: `brmnet_core/projection.py`
- Modify: `brmnet_core/__init__.py`
- Test: `tests/test_projection.py`

- [x] **Step 1: Write the failing tests**

```python
from brmnet_core.projection import FeatureProjection

def test_projection_maps_branch_features_to_common_width():
    projection = FeatureProjection(in_channels=12, out_channels=8)
    assert projection(torch.randn(2, 12, 4, 4)).shape == (2, 8, 4, 4)
```

The second test must verify `FeatureProjection(8, 8)` is an identity module so default paths add no parameters.

- [x] **Step 2: Run tests and verify ImportError**

Run: `conda run -n hslinets python -m unittest tests.test_projection`

Expected: FAIL because `brmnet_core.projection` does not exist.

- [x] **Step 3: Implement the minimal projection**

For unequal widths, use `Conv2d(1x1, bias=False)`, `BatchNorm2d`, and `ReLU`. For equal widths, expose `nn.Identity` as the projection implementation.

- [x] **Step 4: Run target and full tests**

Run: `conda run -n hslinets python -m unittest tests.test_projection`

Run: `conda run -n hslinets python -m unittest discover -s tests`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add brmnet_core/projection.py brmnet_core/__init__.py tests/test_projection.py
git commit -m "Add fixed-dimension branch projection primitive"
```

### Task 4: Record foundation status

**Files:**

- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [ ] **Step 1: Document the verified API and deliberate scope boundary**

Record that the probe is diagnostic only in this batch, and that quality-driven profile selection, profile export, resource accounting, and long-running experiments are the next batch.

- [ ] **Step 2: Commit and push after full verification**

```powershell
git add task_plan.md findings.md progress.md docs/superpowers/plans/2026-07-15-quality-routed-foundation.md
git commit -m "Document quality-routed foundation progress"
git push
```
