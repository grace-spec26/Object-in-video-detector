# No-Wound YOLO Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CoTracker's second-step frame/coordinate export controls with one atomic button that appends submitted no-wound frames to the YOLO dataset with matching empty labels.

**Architecture:** Add a reusable array-frame export function to the existing YOLO exporter so dataset naming, splitting, YAML generation, and validation stay centralized. Guard each export with in-process and filesystem locks, and roll back the complete batch if any conversion, write, or validation fails. Keep the Gradio callback thin and change only the positional outputs associated with the two removed controls.

**Tech Stack:** Python 3.9+, Pillow, NumPy, Gradio, `unittest`

## Global Constraints

- Export every loaded clean frame; tracking, coordinates, and SAM masks are not required.
- Use a deterministic 80% train and 20% validation split with both splits non-empty.
- Append unique names in the existing `train_img00001` and `val_img00001` sequences.
- Every JPEG must have a same-stem zero-byte `.txt` label.
- Existing non-empty YOLO polygon validation remains unchanged.
- Concurrent exports must allocate unique names, and a failed export must leave no partial batch.
- Do not overwrite existing dataset files or commit generated dataset contents.

---

### Task 1: No-Wound Dataset Exporter

**Files:**
- Modify: `tests/test_yolo_segmentation_dataset_export.py`
- Modify: `export_yolo_segmentation_dataset.py`

**Interfaces:**
- Consumes: a sequence of RGB/RGBA/grayscale array-like frames, an output directory, `train_ratio`, and `image_quality`.
- Produces: `export_no_wound_frames_to_yolo_dataset(frames, output_dir="dataset", train_ratio=0.8, image_quality=95) -> ExportStats`.

- [ ] **Step 1: Write failing exporter and validator tests**

Import the new function and validator, then add tests that export five NumPy frames, expect four train images and one validation image, assert every label is exactly `b""`, and validate a mixed dataset containing both polygon and empty labels. Add an append test that pre-creates `train_img00005` and `val_img00003` and expects the next files to begin at indices 6 and 4.

```python
from export_yolo_segmentation_dataset import (
    export_no_wound_frames_to_yolo_dataset,
    export_yolo_segmentation_dataset,
    validate_yolo_segmentation_dataset,
)

frames = np.zeros((5, 20, 30, 3), dtype=np.uint8)
stats = export_no_wound_frames_to_yolo_dataset(frames, dataset_dir)
self.assertEqual((stats.train_images, stats.val_images), (4, 1))
self.assertEqual(label_path.read_bytes(), b"")
validate_yolo_segmentation_dataset(dataset_dir)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m unittest tests/test_yolo_segmentation_dataset_export.py
```

Expected: fail because `export_no_wound_frames_to_yolo_dataset` does not exist and empty labels are rejected.

- [ ] **Step 3: Implement the no-wound exporter**

Add the public function, validate at least two frames, prevalidate every frame, prepare existing split folders, calculate `_split_count`, append using `_next_split_image_index`, and write each image/empty-label pair. Hold a per-dataset thread lock plus `fcntl` lock across allocation, writing, YAML generation, and validation. Track every created path and remove the full batch while restoring the prior YAML on failure.

```python
def export_no_wound_frames_to_yolo_dataset(
    frames: Sequence[object],
    output_dir: Path | str = "dataset",
    train_ratio: float = 0.8,
    image_quality: int = 95,
) -> ExportStats:
    frame_list = list(frames)
    if len(frame_list) < 2:
        raise ValueError("At least two frames are required so train and val folders are non-empty.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")

    dataset_root = Path(output_dir)
    train_count = _split_count(len(frame_list), train_ratio)
    _prepare_dataset_dir(dataset_root, overwrite=False)
    _write_empty_frame_records(dataset_root, "train", frame_list[:train_count], image_quality)
    _write_empty_frame_records(dataset_root, "val", frame_list[train_count:], image_quality)
    _write_dataset_yaml(dataset_root)
    validate_yolo_segmentation_dataset(dataset_root)
    return ExportStats(len(frame_list), len(frame_list), 0, 0, train_count, len(frame_list) - train_count)
```

Use `Image.fromarray(np.asarray(frame)).convert("RGB")` for loaded video frames. If writing either member of a pair fails, delete both output paths before re-raising.

- [ ] **Step 4: Permit valid empty labels**

In `validate_yolo_segmentation_dataset`, remove the rejection of zero non-empty rows. Continue applying all class, point-count, parity, and normalization checks to every row that is present.

```python
rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines()]
for row in (row for row in rows if row):
    values = row.split()
    # Existing polygon validation remains here.
```

- [ ] **Step 5: Run exporter tests and verify GREEN**

Run the Task 1 test command again. Expected: all exporter tests pass.

---

### Task 2: CoTracker Atomic Export Control

**Files:**
- Modify: `co-tracker-main/tests/test_gradio_app_wiring.py`
- Modify: `co-tracker-main/gradio_demo/app.py`

**Interfaces:**
- Consumes: the existing clean `video` Gradio state.
- Produces: `export_no_wound_frames_from_state(video_frames) -> str`, shown in `export_status`.

- [ ] **Step 1: Write failing UI wiring tests**

Assert that the second step contains exactly one disabled-at-load button, that submit enables it, and that its callback receives only the clean video state. Assert the old two button declarations and callback wiring are absent.

```python
self.assertIn(
    'no_wound_export_button = gr.Button("Export No-Wound Frames to YOLO", interactive=False)',
    second_step_block,
)
self.assertNotIn("store_frames_button = gr.Button", second_step_block)
self.assertNotIn("store_coordinates_button = gr.Button", second_step_block)

match = re.search(r"no_wound_export_button\.click\((.*?)\n\s*\)", app_source, re.DOTALL)
self.assertIn("fn = export_no_wound_frames_from_state", match.group(1))
self.assertIn("video", match.group(1))
```

- [ ] **Step 2: Run wiring tests and verify RED**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m unittest co-tracker-main/tests/test_gradio_app_wiring.py
```

Expected: fail because the new button and callback do not exist.

- [ ] **Step 3: Add the thin Gradio callback**

Replace the two old store callbacks with:

```python
def export_no_wound_frames_from_state(video_frames):
    if video_frames is None:
        message = "Submit a video before exporting no-wound frames."
        gr.Warning(message, duration=5)
        return message

    try:
        from export_yolo_segmentation_dataset import export_no_wound_frames_to_yolo_dataset
        stats = export_no_wound_frames_to_yolo_dataset(video_frames, DEFAULT_YOLO_DATASET_DIR)
    except Exception as exc:
        message = f"Failed to export no-wound frames: {exc}"
        gr.Warning(message, duration=5)
        return message

    return (
        f"Exported {stats.exported_images} no-wound frame(s) to {DEFAULT_YOLO_DATASET_DIR}: "
        f"{stats.train_images} train and {stats.val_images} val image(s), each with an empty label."
    )
```

Ensure `PROJECT_ROOT` is on `sys.path` before the root exporter import, matching the existing YOLO callback.

- [ ] **Step 4: Replace controls and positional wiring**

Declare only `no_wound_export_button`, enable it in the successful preprocess return, preserve it as interactive through Track/re-process, and replace the two old click callbacks with one:

```python
no_wound_export_button = gr.Button("Export No-Wound Frames to YOLO", interactive=False)

no_wound_export_button.click(
    fn=export_no_wound_frames_from_state,
    inputs=[video],
    outputs=[export_status],
    queue=False,
)
```

Remove one positional return/update wherever the former two button outputs appeared, and change `reprocess_with_refinements` from 16 outputs to 15. Update its processed-video-skip mutation from `result[11]` to `result[10]` after the removed coordinate-button slot.

- [ ] **Step 5: Run CoTracker tests and verify GREEN**

Run:

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m unittest discover -s co-tracker-main/tests -p 'test_*.py'
```

Expected: all CoTracker tests pass.

---

### Task 3: End-to-End Verification

**Files:**
- Verify only; no generated dataset files are committed.

**Interfaces:**
- Consumes: completed Task 1 and Task 2 changes.
- Produces: fresh test and syntax evidence for commit readiness.

- [ ] **Step 1: Run all focused tests**

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m unittest tests/test_yolo_segmentation_dataset_export.py
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m unittest discover -s co-tracker-main/tests -p 'test_*.py'
```

- [ ] **Step 2: Compile changed Python files**

```bash
PYTHONPYCACHEPREFIX=/tmp/codex-pycache-system .venv_local/bin/python -m py_compile export_yolo_segmentation_dataset.py co-tracker-main/gradio_demo/app.py tests/test_yolo_segmentation_dataset_export.py co-tracker-main/tests/test_gradio_app_wiring.py
```

- [ ] **Step 3: Review repository scope**

Run `git status --short`, `git diff --check`, and `git diff --stat`. Confirm generated datasets, model weights, environments, caches, logs, and unrelated user files are unstaged.

- [ ] **Step 4: Commit and push**

Stage only the plan, exporter, app, and their tests. Commit with `Add no-wound YOLO frame export`, then push the current branch.
