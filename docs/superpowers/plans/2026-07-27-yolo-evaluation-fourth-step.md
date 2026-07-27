# YOLO Evaluation Fourth Step Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Fourth Step Gradio workflow for uploading an evaluation video and trained YOLO `.pt` model, then previewing model detections on the video.

**Architecture:** Add Fourth Step components in `ui_layout.py`, wire the button in `app.py`, and put YOLO video inference in a focused `yolo_evaluation_service.py` helper. The service streams frame-level progress, writes an MP4 preview, and stays independent from CoTracker/SAM state.

**Tech Stack:** Python, Gradio, OpenCV, mediapy test adapters, NumPy, Ultralytics YOLO, unittest.

---

### Task 1: Add Failing UI And Wiring Tests

**Files:**
- Modify: `co-tracker-main/tests/test_gradio_app_wiring.py`

- [ ] **Step 1: Add tests for Fourth Step layout, callback wiring, and dependency**

Add these tests to `GradioAppWiringTest`:

```python
    def test_fourth_step_has_yolo_evaluation_controls(self):
        layout_source = UI_LAYOUT_PATH.read_text()

        self.assertIn("## Fourth step: Evaluation of model.", layout_source)
        fourth_step_block = layout_source.split("## Fourth step: Evaluation of model.", maxsplit=1)[1]
        self.assertIn("evaluation_video_input = gr.File", fourth_step_block)
        self.assertIn('label="Evaluation Video"', fourth_step_block)
        self.assertIn("evaluation_yolo_model_input = gr.File", fourth_step_block)
        self.assertIn('label="Trained YOLO Model"', fourth_step_block)
        self.assertIn('evaluation_preview_button = gr.Button("Preview model on video"', fourth_step_block)
        self.assertIn("evaluation_progress = gr.HTML", fourth_step_block)
        self.assertIn("evaluation_output_video = gr.Video", fourth_step_block)
        self.assertIn('label="YOLO Model Preview"', fourth_step_block)

    def test_fourth_step_components_are_returned_from_layout_namespace(self):
        layout_source = UI_LAYOUT_PATH.read_text()

        for component_name in (
            "evaluation_video_input",
            "evaluation_yolo_model_input",
            "evaluation_preview_button",
            "evaluation_progress",
            "evaluation_output_video",
        ):
            self.assertIn(f"{component_name}={component_name}", layout_source)

    def test_yolo_evaluation_preview_button_is_wired_to_service(self):
        app_source = APP_PATH.read_text()

        self.assertIn("from yolo_evaluation_service import", app_source)
        match = re.search(r"evaluation_preview_button\\.click\\((.*?)\\n\\s*\\)", app_source, re.DOTALL)

        self.assertIsNotNone(match)
        self.assertIn("fn = preview_yolo_model_on_video", match.group(1))
        self.assertIn("evaluation_video_input", match.group(1))
        self.assertIn("evaluation_yolo_model_input", match.group(1))
        self.assertIn("evaluation_progress", match.group(1))
        self.assertIn("evaluation_output_video", match.group(1))

    def test_cotracker_gradio_requirements_include_ultralytics_for_yolo_evaluation(self):
        requirements_path = APP_PATH.parent / "requirements.txt"
        requirements = requirements_path.read_text()

        self.assertRegex(requirements, r"(?m)^ultralytics\\b")
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv_local/bin/python -m unittest co-tracker-main/tests/test_gradio_app_wiring.py
```

Expected: the new Fourth Step tests fail because the UI, service import, callback, and dependency are not implemented.

### Task 2: Add Failing YOLO Evaluation Service Tests

**Files:**
- Create: `co-tracker-main/tests/test_yolo_evaluation_service.py`

- [ ] **Step 1: Add service tests using fake readers, writers, streaming capture, and YOLO model**

Create a test file with:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

import yolo_evaluation_service as service  # noqa: E402


class FakeBoxes:
    def __init__(self):
        self.xyxy = np.asarray([[2, 2, 12, 12]], dtype=np.float32)
        self.cls = np.asarray([0], dtype=np.float32)
        self.conf = np.asarray([0.91], dtype=np.float32)


class FakeResult:
    names = {0: "wound"}

    def __init__(self):
        self.boxes = FakeBoxes()


class FakeYoloModel:
    names = {0: "wound"}

    def __init__(self):
        self.calls = []

    def __call__(self, frame, verbose=False):
        self.calls.append(np.asarray(frame).copy())
        return [FakeResult()]


class YoloEvaluationServiceTest(unittest.TestCase):
    def test_missing_video_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "best.pt"
            model_path.write_bytes(b"weights")

            results = list(service.preview_yolo_model_on_video(None, str(model_path)))

        self.assertEqual(results[-1][1], None)
        self.assertIn("Upload an evaluation video", results[-1][0])

    def test_missing_model_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            video_path.write_bytes(b"video")

            results = list(service.preview_yolo_model_on_video(str(video_path), None))

        self.assertEqual(results[-1][1], None)
        self.assertIn("Upload a trained YOLO .pt model", results[-1][0])

    def test_non_pt_model_reports_clear_status_and_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.onnx"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            results = list(service.preview_yolo_model_on_video(str(video_path), str(model_path)))

        self.assertEqual(results[-1][1], None)
        self.assertIn("must be a .pt file", results[-1][0])

    def test_processes_each_frame_draws_detections_and_writes_preview(self):
        frames = np.zeros((2, 20, 24, 3), dtype=np.uint8)
        fake_model = FakeYoloModel()
        written = {}

        def fake_writer(path, output_frames, fps):
            written["path"] = path
            written["frames"] = np.asarray(output_frames)
            written["fps"] = fps
            Path(path).write_bytes(b"preview")

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "clip.mp4"
            model_path = Path(tmp) / "best.pt"
            video_path.write_bytes(b"video")
            model_path.write_bytes(b"weights")

            results = list(
                service.preview_yolo_model_on_video(
                    str(video_path),
                    str(model_path),
                    model_loader=lambda path: fake_model,
                    video_reader=lambda path: frames,
                    video_writer=fake_writer,
                    output_dir=tmp,
                )
            )

        progress_html = "\n".join(result[0] for result in results)
        self.assertIn("0/2 frame", progress_html)
        self.assertIn("1/2 frame", progress_html)
        self.assertIn("2/2 frame", progress_html)
        self.assertEqual(len(fake_model.calls), 2)
        self.assertEqual(written["frames"].shape, frames.shape)
        self.assertEqual(written["fps"], 24.0)
        self.assertTrue(np.any(written["frames"] != frames))
        self.assertEqual(results[-1][1], written["path"])
        self.assertIn("complete", results[-1][0].lower())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv_local/bin/python -m unittest co-tracker-main/tests/test_yolo_evaluation_service.py
```

Expected: import fails because `yolo_evaluation_service.py` does not exist.

### Task 3: Implement The YOLO Evaluation Service

**Files:**
- Create: `co-tracker-main/gradio_demo/yolo_evaluation_service.py`

- [ ] **Step 1: Add service implementation**

Create the helper with these public names and responsibilities:

- `YOLO_EVALUATION_PROGRESS_READY`: initial HTML status.
- `YoloEvaluationError`: validation and runtime error type.
- `coerce_uploaded_path(uploaded)`: accepts Gradio file dictionaries or file path strings.
- `validate_yolo_evaluation_inputs(video_path, model_path)`: returns two `Path` values after checking required inputs, existence, and `.pt` extension.
- `format_yolo_evaluation_progress_html(processed, total, message, state="running")`: returns escaped progress/status HTML.
- `as_uint8_rgb_frame(frame)`: returns an RGB `uint8` frame with 3 channels.
- `to_numpy(value)`: converts NumPy, torch tensor, or tensor-like values to NumPy arrays.
- `extract_yolo_detections(result, fallback_names=None)`: returns `(xyxy, class_id, confidence, label)` tuples.
- `draw_yolo_detections(frame, result, fallback_names=None)`: draws boxes, labels, and confidence scores.
- `load_yolo_model(model_path, model_loader=None)`: lazy-imports `ultralytics.YOLO` when no test loader is provided.
- `run_yolo_on_frame(model, frame)`: calls the model and returns the first result.
- `open_cv2_video_capture(video_path)`: opens the default streaming video reader and returns capture, total frame count, and FPS.
- `open_cv2_video_writer(frame, fps, output_dir=None)`: opens the default streaming MP4 writer from the first output frame shape.
- `preview_yolo_model_on_video(video_path, model_path, *, model_loader=None, video_reader=None, video_writer=None, output_dir=None)`: generator yielding `(progress_html, output_video_path)` tuples.

- [ ] **Step 2: Run service tests and verify GREEN**

Run:

```bash
.venv_local/bin/python -m unittest co-tracker-main/tests/test_yolo_evaluation_service.py
```

Expected: all service tests pass.

### Task 4: Implement UI And Callback Wiring

**Files:**
- Modify: `co-tracker-main/gradio_demo/ui_layout.py`
- Modify: `co-tracker-main/gradio_demo/app.py`
- Modify: `co-tracker-main/gradio_demo/requirements.txt`

- [ ] **Step 1: Add requirements entry**

Add:

```text
ultralytics
```

- [ ] **Step 2: Wire app imports**

Import `preview_yolo_model_on_video` and `YOLO_EVALUATION_PROGRESS_READY` in both relative and direct import blocks in `app.py`.

- [ ] **Step 3: Add layout parameter and Fourth Step components**

Add `yolo_evaluation_progress_ready` to `build_demo_layout()`, create the Fourth Step under Third Step, and return the five new component names in the `SimpleNamespace`.

- [ ] **Step 4: Add callback**

In `configure_app_callbacks()`, bind:

```python
    evaluation_preview_button.click(
        fn=preview_yolo_model_on_video,
        inputs=[
            evaluation_video_input,
            evaluation_yolo_model_input,
        ],
        outputs=[
            evaluation_progress,
            evaluation_output_video,
        ],
    )
```

- [ ] **Step 5: Run wiring tests and verify GREEN**

Run:

```bash
.venv_local/bin/python -m unittest co-tracker-main/tests/test_gradio_app_wiring.py
```

Expected: all wiring tests pass.

### Task 5: Verify The Feature Surface

**Files:**
- Verify: changed source and test files

- [ ] **Step 1: Run focused test suite**

Run:

```bash
.venv_local/bin/python -m unittest co-tracker-main/tests/test_yolo_evaluation_service.py co-tracker-main/tests/test_gradio_app_wiring.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Run broader regression tests**

Run:

```bash
.venv_local/bin/python -m unittest discover -s co-tracker-main/tests
```

Expected: all tests pass or any unrelated pre-existing failures are documented with exact output.

- [ ] **Step 3: Check generated and large files are not staged**

Run:

```bash
git status --short
git diff --cached --name-status
```

Expected: staged files include only the plan, source files, tests, requirements, and `.gitignore` if it changes. Staged files must not include virtual environments, model weights, outputs, uploads, logs, `.DS_Store`, pycache, or generated datasets.

### Task 6: Commit And Push

**Files:**
- Commit the scoped implementation changes

- [ ] **Step 1: Stage only intended files**

Run:

```bash
git add docs/superpowers/plans/2026-07-27-yolo-evaluation-fourth-step.md \
  co-tracker-main/gradio_demo/yolo_evaluation_service.py \
  co-tracker-main/gradio_demo/ui_layout.py \
  co-tracker-main/gradio_demo/app.py \
  co-tracker-main/gradio_demo/requirements.txt \
  co-tracker-main/tests/test_yolo_evaluation_service.py \
  co-tracker-main/tests/test_gradio_app_wiring.py
```

- [ ] **Step 2: Verify staged files**

Run:

```bash
git status --short
git diff --cached --name-status
```

Expected: only intended files are staged.

- [ ] **Step 3: Commit and push**

Run:

```bash
git commit -m "Add YOLO evaluation fourth step"
git push
```

Expected: commit succeeds and push updates the current branch.
