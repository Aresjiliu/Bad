# Houston2013 Dual-Split Data Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a raw Houston2013 data path that reproducibly supports official contest and random pixel splits through one coordinate-based BRM-Net training interface.

**Architecture:** A focused `brmnet_core.data` package will load validated raw MAT arrays, create immutable coordinate splits, normalize the shared scene, and expose paired HSI/LiDAR patch datasets. The existing runner will select the new path with `--data-format raw` while retaining the legacy pre-generated MAT path for historical experiments.

**Tech Stack:** Python 3, NumPy, SciPy, PyTorch, `unittest`, JSON/NPZ artifacts.

---

## File Map

- Create `brmnet_core/data/__init__.py`: public data API.
- Create `brmnet_core/data/houston.py`: raw MAT loading, known keys, dataset constants, fingerprints.
- Create `brmnet_core/data/splits.py`: official ROI parsing, random split generation, validation, NPZ persistence.
- Create `brmnet_core/data/patch_dataset.py`: scene normalization, reflection padding, paired coordinate dataset.
- Create `brmnet_core/data/factory.py`: DataLoader construction and metadata aggregation.
- Create `tests/test_houston_data.py`: raw loading and validation tests.
- Create `tests/test_houston_splits.py`: official/random split and persistence tests.
- Create `tests/test_houston_patch_dataset.py`: normalization, boundary patch and label tests.
- Create `tests/test_houston_factory.py`: loader integration tests.
- Modify `scripts/run_brmnet_houston.py`: raw/legacy selection, split artifacts and dataset-only mode.
- Modify `brmnet_core/__init__.py`: export the stable raw-data factory API.
- Modify `docs/BRMNET_REAL_DATA_RUNBOOK_ZH.md`: exact commands for both protocols.

### Task 1: Raw Houston2013 Loader

**Files:**
- Create: `brmnet_core/data/__init__.py`
- Create: `brmnet_core/data/houston.py`
- Create: `tests/test_houston_data.py`

- [ ] **Step 1: Write failing loader tests**

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from brmnet_core.data.houston import HoustonScene, load_houston_scene


class HoustonDataTest(unittest.TestCase):
    def test_loads_known_mat_keys_and_normalizes_shapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            savemat(root / "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat",
                    {"ans": np.ones((3, 4, 2), dtype=np.uint16)})
            savemat(root / "2013_IEEE_GRSS_DF_Contest_LiDAR.mat",
                    {"LiDAR_data": np.ones((3, 4), dtype=np.float32)})
            savemat(root / "GRSS2013.mat",
                    {"name": np.array([[0, 1, 1, 0], [2, 2, 0, 0], [0, 0, 0, 0]], dtype=np.uint8)})

            scene = load_houston_scene(root, require_roi=False)

            self.assertIsInstance(scene, HoustonScene)
            self.assertEqual(scene.hsi.shape, (3, 4, 2))
            self.assertEqual(scene.lidar.shape, (3, 4, 1))
            self.assertEqual(scene.gt.shape, (3, 4))
            self.assertEqual(scene.hsi.dtype, np.float32)
            self.assertEqual(scene.gt.dtype, np.int64)

    def test_rejects_missing_known_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            savemat(root / "2013_IEEE_GRSS_DF_Contest_CASI_349_1905_144.mat",
                    {"wrong": np.ones((3, 4, 2))})
            savemat(root / "2013_IEEE_GRSS_DF_Contest_LiDAR.mat",
                    {"LiDAR_data": np.ones((3, 4, 1))})
            savemat(root / "GRSS2013.mat", {"name": np.zeros((3, 4))})

            with self.assertRaisesRegex(KeyError, "ans"):
                load_houston_scene(root, require_roi=False)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_data -v
```

Expected: import failure for missing `brmnet_core.data.houston`.

- [ ] **Step 3: Implement known-key loading and validation**

Implement:

```python
@dataclass(frozen=True)
class HoustonScene:
    hsi: np.ndarray
    lidar: np.ndarray
    gt: np.ndarray
    roi_records: np.ndarray | None
    source_paths: dict[str, Path]


def _load_known_key(path: Path, key: str) -> np.ndarray:
    content = loadmat(path, squeeze_me=True, struct_as_record=False)
    if key not in content:
        available = sorted(name for name in content if not name.startswith("__"))
        raise KeyError(f"{path.name} must contain {key!r}; available keys: {available}")
    return content[key]


def load_houston_scene(root: str | Path, require_roi: bool = True) -> HoustonScene:
    # Load ans, LiDAR_data, name and optionally TR_Samples.
    # Convert arrays to float32/int64, expand 2-D LiDAR to H,W,1,
    # and raise ValueError when spatial shapes differ.
```

Also define exact filenames, MAT keys, class names, and:

```python
HOUSTON_TRAIN_COUNTS = {
    1: 198, 2: 190, 3: 192, 4: 188, 5: 186,
    6: 182, 7: 196, 8: 191, 9: 193, 10: 191,
    11: 181, 12: 192, 13: 184, 14: 181, 15: 187,
}
```

- [ ] **Step 4: Add spatial mismatch and ROI requirement tests**

Add tests that assert:

```python
with self.assertRaisesRegex(ValueError, "spatial shapes"):
    load_houston_scene(root, require_roi=False)

with self.assertRaisesRegex(FileNotFoundError, "Samples_TR"):
    load_houston_scene(root, require_roi=True)
```

- [ ] **Step 5: Run loader tests and full regression**

Run:

```powershell
python -m unittest tests.test_houston_data -v
python -m unittest tests.test_brmnet_engine tests.test_brmnet_legacy tests.test_brmnet_reporting
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add brmnet_core/data/__init__.py brmnet_core/data/houston.py tests/test_houston_data.py
git commit -m "Add validated Houston raw data loader"
git push origin brmnet-core-extraction
```

### Task 2: Official and Random Coordinate Splits

**Files:**
- Create: `brmnet_core/data/splits.py`
- Create: `tests/test_houston_splits.py`
- Modify: `brmnet_core/data/__init__.py`

- [ ] **Step 1: Write failing random split tests**

```python
def test_random_split_is_deterministic_and_class_balanced(self):
    gt = np.array([
        [1, 1, 1, 1, 0],
        [2, 2, 2, 2, 0],
    ])
    counts = {1: 2, 2: 2}

    first = build_random_split(gt, counts, seed=42)
    second = build_random_split(gt, counts, seed=42)

    np.testing.assert_array_equal(first.train_coords, second.train_coords)
    np.testing.assert_array_equal(first.test_coords, second.test_coords)
    self.assertEqual(first.protocol, "random")
    self.assertEqual(first.seed, 42)
    self.assertEqual(len(first.train_coords), 4)
    self.assertEqual(len(first.test_coords), 4)
```

Add a test that seeds NumPy's global RNG before and after `build_random_split` and proves its sequence is unchanged.

- [ ] **Step 2: Run random split tests and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_splits.HoustonSplitTest.test_random_split_is_deterministic_and_class_balanced -v
```

Expected: import failure for missing split API.

- [ ] **Step 3: Implement split value object and random split**

Implement:

```python
@dataclass(frozen=True)
class CoordinateSplit:
    protocol: str
    seed: int | None
    train_coords: np.ndarray
    train_labels: np.ndarray
    test_coords: np.ndarray
    test_labels: np.ndarray


def build_random_split(
    gt: np.ndarray,
    train_counts: Mapping[int, int],
    seed: int,
) -> CoordinateSplit:
    rng = np.random.default_rng(seed)
    # For each class, permute np.argwhere(gt == class_id), select the
    # requested count, and concatenate class-ordered train/test arrays.
```

Validation must reject unavailable class counts and train/test overlap.

- [ ] **Step 4: Write failing official ROI parser tests**

Use a small object array matching the real ENVI structure:

```python
records = np.array([
    np.array(["; ROI name:", "grass_healthy"]),
    np.array(["; ROI npts:", "2"]),
    np.array(["; ID", "X", "Y", "Lat", "Lon"]),
    np.array(["1", "2", "1", "0", "0"]),
    np.array(["2", "3", "1", "0", "0"]),
    np.array(["; ROI name:", "grass_stressed"]),
    np.array(["; ROI npts:", "1"]),
    np.array(["; ID", "X", "Y", "Lat", "Lon"]),
    np.array(["1", "1", "2", "0", "0"]),
], dtype=object)
```

Assert that 1-based `(X,Y)` becomes zero-based `(row,col)`:

```python
split = build_official_split(gt, records, class_names={1: "grass_healthy", 2: "grass_stressed"})
np.testing.assert_array_equal(split.train_coords, np.array([[0, 1], [0, 2], [1, 0]]))
```

Also assert that a coordinate/GT label mismatch raises `ValueError`.

- [ ] **Step 5: Run official parser tests and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_splits.HoustonSplitTest.test_official_roi_converts_xy_and_validates_gt -v
```

Expected: failure because `build_official_split` is missing.

- [ ] **Step 6: Implement official parser and split persistence**

Implement:

```python
def parse_envi_roi_records(
    records: np.ndarray,
    class_names: Mapping[int, str],
) -> tuple[np.ndarray, np.ndarray]:
    # Normalize each record to stripped strings.
    # Track ROI name and declared npts.
    # Parse only five-field rows whose first three fields are integers.
    # Map ROI name to class id and verify each declared count.


def build_official_split(
    gt: np.ndarray,
    records: np.ndarray,
    class_names: Mapping[int, str],
    train_counts: Mapping[int, int] | None = None,
) -> CoordinateSplit:
    # Convert X,Y to row,col, validate GT labels, and use all remaining GT>0
    # coordinates as test coordinates.


def save_coordinate_split(path: str | Path, split: CoordinateSplit) -> None:
    np.savez_compressed(
        path,
        protocol=np.array(split.protocol),
        seed=np.array(-1 if split.seed is None else split.seed, dtype=np.int64),
        train_coords=split.train_coords,
        train_labels=split.train_labels,
        test_coords=split.test_coords,
        test_labels=split.test_labels,
    )


def load_coordinate_split(path: str | Path) -> CoordinateSplit:
    with np.load(path, allow_pickle=False) as data:
        seed_value = int(data["seed"])
        return CoordinateSplit(
            protocol=str(data["protocol"]),
            seed=None if seed_value == -1 else seed_value,
            train_coords=data["train_coords"].astype(np.int64, copy=False),
            train_labels=data["train_labels"].astype(np.int64, copy=False),
            test_coords=data["test_coords"].astype(np.int64, copy=False),
            test_labels=data["test_labels"].astype(np.int64, copy=False),
        )
```

- [ ] **Step 7: Add round-trip and invalid-overlap tests**

Assert `save_coordinate_split` followed by `load_coordinate_split` preserves every field and dtype. Add direct validation coverage for duplicated train coordinates and train/test overlap.

- [ ] **Step 8: Run split tests and full regression**

Run:

```powershell
python -m unittest tests.test_houston_splits -v
python -m unittest discover -s tests -p "test_*.py"
```

Expected: all tests pass.

- [ ] **Step 9: Commit and push**

```powershell
git add brmnet_core/data/__init__.py brmnet_core/data/splits.py tests/test_houston_splits.py
git commit -m "Add Houston official and random split protocols"
git push origin brmnet-core-extraction
```

### Task 3: Shared Normalization and Coordinate Patch Dataset

**Files:**
- Create: `brmnet_core/data/patch_dataset.py`
- Create: `tests/test_houston_patch_dataset.py`
- Modify: `brmnet_core/data/__init__.py`

- [ ] **Step 1: Write failing normalization tests**

```python
def test_scene_normalization_uses_all_unlabeled_pixels(self):
    hsi = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    lidar = np.array([[[2.0], [4.0]], [[6.0], [8.0]]], dtype=np.float32)

    normalized, stats = normalize_scene(hsi, lidar)

    self.assertAlmostEqual(float(normalized.hsi.mean()), 0.0, places=6)
    self.assertAlmostEqual(float(normalized.hsi.std()), 1.0, places=6)
    self.assertEqual(stats.hsi_mean.shape, (1,))
    self.assertEqual(stats.lidar_mean.shape, (1,))
```

Add a zero-variance test asserting finite zero outputs and stored effective standard deviation of 1.

- [ ] **Step 2: Run normalization tests and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_patch_dataset.HoustonPatchDatasetTest.test_scene_normalization_uses_all_unlabeled_pixels -v
```

Expected: import failure for missing `normalize_scene`.

- [ ] **Step 3: Implement shared normalization**

Implement immutable `NormalizationStats` and `NormalizedScene` dataclasses. Compute means/stds across axes `(0, 1)`, replace standard deviations below `1e-8` with `1.0`, and return `float32` arrays.

- [ ] **Step 4: Write failing patch dataset tests**

```python
def test_extracts_reflection_padded_boundary_patch_and_zero_based_label(self):
    hsi = np.arange(3 * 3, dtype=np.float32).reshape(3, 3, 1)
    lidar = (hsi + 100).copy()
    coords = np.array([[0, 0]])
    labels = np.array([2])

    dataset = HoustonPatchDataset(hsi, lidar, coords, labels, patch_size=3, augment=False)
    sample = dataset[0]

    self.assertEqual(tuple(sample["m_1"].shape), (1, 3, 3))
    self.assertEqual(tuple(sample["m_2"].shape), (1, 3, 3))
    self.assertEqual(sample["label"], 1)
    self.assertTrue(torch.isfinite(sample["m_1"]).all())
```

Add tests rejecting patch sizes `0`, `2`, and mismatched coordinate/label lengths.

- [ ] **Step 5: Run dataset tests and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_patch_dataset.HoustonPatchDatasetTest.test_extracts_reflection_padded_boundary_patch_and_zero_based_label -v
```

Expected: failure because `HoustonPatchDataset` is missing.

- [ ] **Step 6: Implement coordinate dataset**

Implement:

```python
class HoustonPatchDataset(Dataset):
    def __init__(self, hsi, lidar, coords, labels, patch_size=7, augment=False):
        # Validate shapes and labels, reflect-pad spatial axes once.

    def __getitem__(self, index):
        # Slice the padded scene around row,col.
        # Apply the same random horizontal/vertical flips to both modalities.
        # Return float tensors under "m_1" and "m_2", a zero-based integer
        # under "label", and the original int64 [row, col] under "coord".
```

Use `torch.rand` for augmentation decisions so `seed_everything` controls training transforms.

- [ ] **Step 7: Run dataset tests and full regression**

Run:

```powershell
python -m unittest tests.test_houston_patch_dataset -v
python -m unittest discover -s tests -p "test_*.py"
```

Expected: all tests pass.

- [ ] **Step 8: Commit and push**

```powershell
git add brmnet_core/data/__init__.py brmnet_core/data/patch_dataset.py tests/test_houston_patch_dataset.py
git commit -m "Add Houston coordinate patch dataset"
git push origin brmnet-core-extraction
```

### Task 4: Raw DataLoader Factory and Artifacts

**Files:**
- Create: `brmnet_core/data/factory.py`
- Create: `tests/test_houston_factory.py`
- Modify: `brmnet_core/data/__init__.py`
- Modify: `brmnet_core/__init__.py`

- [ ] **Step 1: Write failing factory integration test**

Build a small `HoustonScene` directly and assert:

```python
bundle = build_houston_raw_loaders(
    scene=scene,
    protocol="random",
    split_seed=7,
    patch_size=3,
    batch_size=2,
    num_workers=0,
    train_counts={1: 1, 2: 1},
)

batch = next(iter(bundle.train_loader))
self.assertEqual(tuple(batch["m_1"].shape[1:]), (2, 3, 3))
self.assertEqual(tuple(batch["m_2"].shape[1:]), (1, 3, 3))
self.assertEqual(bundle.split.protocol, "random")
self.assertEqual(bundle.metadata["train_samples"], 2)
```

- [ ] **Step 2: Run factory test and verify RED**

Run:

```powershell
python -m unittest tests.test_houston_factory -v
```

Expected: import failure for missing factory.

- [ ] **Step 3: Implement loader bundle**

Implement:

```python
@dataclass(frozen=True)
class HoustonDataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    split: CoordinateSplit
    normalization: NormalizationStats
    metadata: dict[str, object]


def build_houston_raw_loaders(
    scene: HoustonScene,
    protocol: str,
    split_seed: int,
    patch_size: int,
    batch_size: int,
    num_workers: int,
    train_counts: Mapping[int, int] = HOUSTON_TRAIN_COUNTS,
    split_file: str | Path | None = None,
) -> HoustonDataBundle:
    # Select official/random split, normalize once, create datasets/loaders,
    # and expose exact sample/class counts.
```

`official` must require `scene.roi_records`. `split_file` loads exact coordinates and must match the requested protocol.

- [ ] **Step 4: Add artifact writer test and implementation**

Test and implement:

```python
def write_houston_data_artifacts(
    output_dir: str | Path,
    bundle: HoustonDataBundle,
    scene: HoustonScene,
) -> dict[str, Path]:
    # Write split.npz, split_summary.json, normalization.npz,
    # and data_fingerprint.json with SHA-256 for each source file.
```

Assert all four files exist and JSON contains protocol, seed, train/test totals, and per-class counts.

- [ ] **Step 5: Run factory tests and full regression**

Run:

```powershell
python -m unittest tests.test_houston_factory -v
python -m unittest discover -s tests -p "test_*.py"
```

Expected: all tests pass.

- [ ] **Step 6: Commit and push**

```powershell
git add brmnet_core/data/__init__.py brmnet_core/data/factory.py brmnet_core/__init__.py tests/test_houston_factory.py
git commit -m "Add Houston raw DataLoader factory"
git push origin brmnet-core-extraction
```

### Task 5: Runner Integration and Real-Data Verification

**Files:**
- Modify: `scripts/run_brmnet_houston.py`
- Create: `tests/test_run_brmnet_houston.py`
- Modify: `docs/BRMNET_REAL_DATA_RUNBOOK_ZH.md`

- [ ] **Step 1: Write failing CLI parser tests**

```python
def test_raw_data_arguments(self):
    args = build_parser().parse_args([
        "--data-format", "raw",
        "--data-root", "D:/data/Houston",
        "--split-protocol", "official",
        "--dataset-only",
        "--num-workers", "0",
    ])
    self.assertEqual(args.data_format, "raw")
    self.assertEqual(args.split_protocol, "official")
    self.assertTrue(args.dataset_only)
    self.assertEqual(args.num_workers, 0)
```

Also test that defaults are `raw`, `official`, split seed `42`, and output directory `output/experiments`.

- [ ] **Step 2: Run CLI tests and verify RED**

Run:

```powershell
python -m unittest tests.test_run_brmnet_houston -v
```

Expected: parser rejects unknown raw-data arguments.

- [ ] **Step 3: Add raw/legacy CLI and loader selection**

Add:

```python
parser.add_argument("--data-format", choices=("raw", "legacy"), default="raw")
parser.add_argument("--split-protocol", choices=("official", "random"), default="official")
parser.add_argument("--split-seed", type=int, default=42)
parser.add_argument("--split-file", default="")
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--output-dir", default="output/experiments")
parser.add_argument("--dataset-only", action="store_true")
```

For `raw`, call `load_houston_scene`, `build_houston_raw_loaders`, and `write_houston_data_artifacts`. For `legacy`, retain the current `build_houston_args/get_houston_loaders` path. Never catch raw-data errors and fall back to legacy.

`--dataset-only` must print JSON metadata after building loaders/artifacts and exit before constructing or training the model.

- [ ] **Step 4: Make experiment paths protocol-specific**

Derive:

```python
run_name = f"houston2013_hsi-lidar_{protocol}_seed{split_seed}"
run_dir = Path(output_dir) / run_name
metrics_path = run_dir / "metrics.csv"
checkpoint_path = run_dir / "checkpoint.pt"
config_path = run_dir / "config.json"
```

Write `config.json` before training and store data metadata in the checkpoint.

- [ ] **Step 5: Run unit and syntax verification**

Run:

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m py_compile brmnet_core\data\*.py scripts\run_brmnet_houston.py
python scripts\run_brmnet_houston.py --help
```

Expected: all tests pass, compilation succeeds, help lists both protocols.

- [ ] **Step 6: Run real official dataset-only verification**

Run:

```powershell
python scripts\run_brmnet_houston.py `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset" `
  --split-protocol official `
  --dataset-only `
  --num-workers 0 `
  --output-dir output\smoke
```

Expected JSON:

```json
{"protocol":"official","train_samples":2832,"test_samples":12197}
```

Verify `split.npz`, `split_summary.json`, `normalization.npz`, and `data_fingerprint.json` exist.

- [ ] **Step 7: Run real random dataset-only verification**

Run the same command with:

```powershell
--split-protocol random --split-seed 42
```

Expected: training 2,832, testing 12,197, and class counts equal the official protocol.

- [ ] **Step 8: Run one-epoch smoke tests**

Run official first:

```powershell
python scripts\run_brmnet_houston.py `
  --data-format raw `
  --data-root "D:\Academic\HSLiNets-main\Dataset" `
  --split-protocol official `
  --epochs 1 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --output-dir output\smoke
```

Then run random seed 42 with the same training arguments. If CUDA is unavailable, use `--device cpu` and record the runtime limitation; do not change model or data settings between protocols.

- [ ] **Step 9: Update Chinese runbook**

Document:

- raw file names and keys;
- official and random dataset-only commands;
- one-epoch commands;
- output artifact meanings;
- warning that official and random results must not be mixed without protocol labels.

- [ ] **Step 10: Commit and push**

```powershell
git add scripts/run_brmnet_houston.py tests/test_run_brmnet_houston.py docs/BRMNET_REAL_DATA_RUNBOOK_ZH.md
git commit -m "Integrate Houston dual-split raw experiments"
git push origin brmnet-core-extraction
```

### Task 6: Final Verification and Planning Records

**Files:**
- Modify: `D:\Academic\task_plan.md`
- Modify: `D:\Academic\findings.md`
- Modify: `D:\Academic\progress.md`

- [ ] **Step 1: Run complete verification**

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m py_compile brmnet_core\*.py brmnet_core\data\*.py scripts\run_brmnet_houston.py
git diff --check
git status --short --branch
```

Expected: tests pass, compilation succeeds, no whitespace errors, and only intentional experiment outputs remain untracked or ignored.

- [ ] **Step 2: Compare real split artifacts**

Read both `split_summary.json` files and confirm:

- both train totals are 2,832;
- both test totals are 12,197;
- all per-class train totals match;
- coordinate hashes differ;
- official protocol has no split seed effect;
- random protocol records seed 42.

- [ ] **Step 3: Update planning records**

Mark stages 31 and 32 complete only if raw data creation and both real 1-epoch runs succeed. If training cannot run because of environment constraints, mark stage 31 complete and leave stage 32 pending with the exact error and completed dataset-only evidence.

- [ ] **Step 4: Push any final documentation-only correction**

If repository documentation changed after the Task 5 commit:

```powershell
git add docs
git commit -m "Document Houston dual-split verification"
git push origin brmnet-core-extraction
```
