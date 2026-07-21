# SAM Processed Video Preview Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `Preview SAM on Processed Video` so a click immediately reads the skip setting, reports the selected-frame count, then renders the chronological CoTracker-prompted SAM mask subsequence into the Step 3 video review box at the original FPS.

**Architecture:** Add a non-queued preparation callback in `sam_preview_service.py`, chain the existing queued video generator after it in `app.py`, and keep the generator responsible for model wait, CoTracker prompt conversion, mask prediction, overlay rendering, and MP4 encoding. Reuse `predict_sam_preview_mask()` for processed-video prediction so predictor locking and image-cache behavior match the single-frame SAM preview path.

**Tech Stack:** Python, Gradio event chaining, NumPy, OpenCV, mediapy, unittest.

---

## File Structure

- Modify `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/sam_preview_service.py`
  - Add `prepare_sam_video_preview()` beside the existing video progress helpers.
  - Route processed-video SAM prediction through `predict_sam_preview_mask()`.
  - Surface prediction and video-encoding failures as visible Gradio errors with status text.
- Modify `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/app.py`
  - Import `prepare_sam_video_preview`.
  - Wire `processed_sam_video_button.click(... queue=False)` to preparation output, then `.then(... queue=True)` to `preview_sam_video_for_processed_frames`.
- Modify `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_sam_preview.py`
  - Add direct service tests for preparation progress and invalid/missing-state behavior.
  - Add regression coverage that video review uses `predict_sam_preview_mask()` and preserves original FPS for `skip = 2`.
  - Add prediction and encoding error visibility tests.
- Modify `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_app_wiring.py`
  - Update source-level wiring test to expect preparation click followed by queued generator `.then()`.

---

### Task 1: Preparation Callback Tests

**Files:**
- Test: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_sam_preview.py`
- Modify: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/sam_preview_service.py`

- [ ] **Step 1: Write the failing tests**

Add these methods to `GradioSamPreviewTest`:

```python
    def test_prepare_processed_video_review_reads_skip_and_reports_selected_frames(self):
        video_frames = np.zeros((10, 20, 40, 3), dtype=np.uint8)

        progress_html, video_path, status = app.prepare_sam_video_preview(video_frames, 2)

        self.assertIn("Preparing SAM video preview", progress_html)
        self.assertIn("0/4 selected frame(s)", progress_html)
        self.assertIsNone(video_path)
        self.assertIn("4 selected frame(s)", status)
        self.assertIn("from 10 total video frame(s)", status)

    def test_prepare_processed_video_review_reports_missing_video_state(self):
        progress_html, video_path, status = app.prepare_sam_video_preview(None, 0)

        self.assertEqual(progress_html, app.SAM_VIDEO_PROGRESS_READY)
        self.assertIsNone(video_path)
        self.assertIn("Submit and track a video", status)

    def test_prepare_processed_video_review_rejects_negative_skip_value(self):
        with self.assertRaises(RuntimeError) as raised:
            app.prepare_sam_video_preview(np.zeros((2, 20, 40, 3), dtype=np.uint8), -1)

        self.assertIn("non-negative", str(raised.exception))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_prepare_processed_video_review_reads_skip_and_reports_selected_frames \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_prepare_processed_video_review_reports_missing_video_state \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_prepare_processed_video_review_rejects_negative_skip_value -v
```

Expected: fail with `AttributeError` because `app.prepare_sam_video_preview` is not imported or defined.

- [ ] **Step 3: Implement the preparation callback**

In `sam_preview_service.py`, add these functions near `selected_sam_video_frame_indices()`:

```python
def parse_sam_video_skip_count(value):
    if value is None:
        return 0
    if isinstance(value, str) and not value.strip():
        return 0

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return parse_frame_skip_count(value)
    if numeric_value < 0:
        raise ValueError("Skip frame count must be non-negative. Use 0 to keep every frame.")
    return parse_frame_skip_count(value)


def prepare_sam_video_preview(video_frames, sam_video_skip_frames):
    if video_frames is None:
        message = "Submit and track a video before running SAM video review."
        gr.Warning(message, duration=5)
        return SAM_VIDEO_PROGRESS_READY, None, message

    try:
        skip_count = parse_sam_video_skip_count(sam_video_skip_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    frame_count = len(video_frames)
    selected_frame_count = len(selected_sam_video_frame_indices(frame_count, skip_count))
    return (
        format_sam_video_progress_html(
            0,
            selected_frame_count,
            f"Preparing SAM video preview for 0/{selected_frame_count} selected frame(s)",
        ),
        None,
        (
            f"Preparing SAM video preview for {selected_frame_count} selected frame(s) "
            f"from {frame_count} total video frame(s)."
        ),
    )
```

In `app.py`, import `prepare_sam_video_preview` in both relative and fallback import blocks.

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command from Step 2.

Expected: all three tests pass.

---

### Task 2: Gradio Event Chain Tests

**Files:**
- Test: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_app_wiring.py`
- Modify: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/app.py`

- [ ] **Step 1: Write the failing wiring test**

Replace `test_processed_sam_video_button_runs_sam_on_all_processed_frames` with:

```python
    def test_processed_sam_video_button_prepares_then_runs_queued_sam_video_review(self):
        app_source = read_combined_source()
        prep_match = re.search(
            r"processed_sam_video_start\s*=\s*processed_sam_video_button\.click\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )
        then_match = re.search(
            r"processed_sam_video_start\.then\((.*?)\n\s*\)",
            app_source,
            re.DOTALL,
        )

        self.assertIsNotNone(prep_match)
        self.assertIn("fn = prepare_sam_video_preview", prep_match.group(1))
        self.assertIn("video", prep_match.group(1))
        self.assertIn("processed_sam_video_skip_frames", prep_match.group(1))
        self.assertIn("processed_sam_video_progress", prep_match.group(1))
        self.assertIn("processed_sam_video", prep_match.group(1))
        self.assertIn("export_status", prep_match.group(1))
        self.assertIn("queue = False", prep_match.group(1))

        self.assertIsNotNone(then_match)
        self.assertIn("fn = preview_sam_video_for_processed_frames", then_match.group(1))
        for state_name in (
            "video",
            "video_preview",
            "query_points",
            "selected_tracks",
            "selected_visibility",
            "selected_point_labels",
            "tracked_video_preview",
            "video_fps",
            "processed_sam_model_dropdown",
            "processed_sam_video_skip_frames",
            "refinement_query_points",
            "tracked_prompt_sources",
        ):
            self.assertIn(state_name, then_match.group(1))
        self.assertIn("processed_sam_video_progress", then_match.group(1))
        self.assertIn("processed_sam_video", then_match.group(1))
        self.assertIn("export_status", then_match.group(1))
        self.assertIn("queue = True", then_match.group(1))
        self.assertIn('show_progress = "hidden"', then_match.group(1))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_app_wiring.GradioAppWiringTest.test_processed_sam_video_button_prepares_then_runs_queued_sam_video_review -v
```

Expected: fail because the app still wires the button directly to the queued generator.

- [ ] **Step 3: Implement event chaining**

In `app.py`, replace the direct `processed_sam_video_button.click(...)` block with:

```python
    processed_sam_video_start = processed_sam_video_button.click(
        fn = prepare_sam_video_preview,
        inputs = [
            video,
            processed_sam_video_skip_frames,
        ],
        outputs = [
            processed_sam_video_progress,
            processed_sam_video,
            export_status,
        ],
        queue = False,
    )

    processed_sam_video_start.then(
        fn = preview_sam_video_for_processed_frames,
        inputs = [
            video,
            video_preview,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            tracked_video_preview,
            video_fps,
            processed_sam_model_dropdown,
            processed_sam_video_skip_frames,
            refinement_query_points,
            tracked_prompt_sources,
        ],
        outputs = [
            processed_sam_video_progress,
            processed_sam_video,
            export_status,
        ],
        show_progress = "hidden",
        queue = True,
    )
```

- [ ] **Step 4: Run wiring test to verify it passes**

Run the same unittest command from Step 2.

Expected: the wiring test passes.

---

### Task 3: Shared Prediction Path and Skip/FPS Regression Tests

**Files:**
- Test: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_sam_preview.py`
- Modify: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/sam_preview_service.py`

- [ ] **Step 1: Write failing tests for shared helper use and `skip = 2` original FPS**

Add this method to `GradioSamPreviewTest`:

```python
    def test_processed_video_review_uses_shared_prediction_helper_and_original_fps_for_skip_two(self):
        video_frames = np.zeros((10, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((10, 10, 20, 3), dtype=np.uint8)
        selected_tracks = np.asarray(
            [[[float(frame_index + 1), 5.0] for frame_index in range(10)]],
            dtype=np.float32,
        )
        runtime = {
            "predictor": FakeSamPredictor(),
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }
        calls = []
        written = {}

        def fake_predict(runtime_arg, frame, point_coords, point_labels):
            calls.append(
                {
                    "runtime": runtime_arg,
                    "frame": np.asarray(frame).copy(),
                    "point_coords": np.asarray(point_coords).copy(),
                    "point_labels": np.asarray(point_labels).copy(),
                }
            )
            mask = np.zeros((20, 40), dtype=bool)
            mask[1:5, 1:5] = True
            return np.asarray([mask]), np.asarray([0.9], dtype=np.float32), None

        def fake_write_video(path, frames, fps):
            written["path"] = path
            written["frames"] = np.asarray(frames)
            written["fps"] = fps

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ), mock.patch.object(
            sam_preview_service,
            "predict_sam_preview_mask",
            side_effect=fake_predict,
        ), mock.patch.object(
            sam_preview_service.mediapy,
            "write_video",
            side_effect=fake_write_video,
        ):
            results = list(
                app.preview_sam_video_for_processed_frames(
                    video_frames,
                    video_preview,
                    [[] for _ in range(10)],
                    selected_tracks,
                    np.ones((1, 10), dtype=bool),
                    [1],
                    video_preview.copy(),
                    24,
                    "sam2.1_hiera_small.pt",
                    2,
                    [[] for _ in range(10)],
                    [],
                )
            )

        self.assertEqual([call["point_coords"][0, 0] for call in calls], [2.0, 8.0, 14.0, 20.0])
        self.assertTrue(all(call["runtime"] is runtime for call in calls))
        self.assertEqual(written["frames"].shape[0], 4)
        self.assertEqual(written["fps"], 24.0)
        self.assertEqual(results[-1][1], written["path"])
        self.assertIn("4/4 selected frame(s)", results[-1][2])
        self.assertEqual(len(runtime["predictor"].predict_calls), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_processed_video_review_uses_shared_prediction_helper_and_original_fps_for_skip_two -v
```

Expected: fail because `preview_sam_video_for_processed_frames()` calls `predictor.set_image()` and `predictor.predict()` directly, so the patched helper is not called.

- [ ] **Step 3: Implement shared helper use**

In `preview_sam_video_for_processed_frames()`, remove the local `predictor = runtime["predictor"]` assignment and replace:

```python
        predictor.set_image(frame)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            multimask_output=True,
            normalize_coords=True,
        )
```

with:

```python
        try:
            masks, scores, _ = predict_sam_preview_mask(runtime, frame, point_coords, point_labels)
        except Exception as exc:
            message = (
                f"SAM video review failed while predicting video frame {frame_index + 1}/{frame_count}: {exc}"
            )
            yield (
                format_sam_video_progress_html(
                    selected_index - 1,
                    selected_frame_count,
                    f"SAM prediction failed at selected frame {selected_index}/{selected_frame_count}",
                ),
                None,
                message,
            )
            raise gr.Error(message) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_processed_video_review_uses_shared_prediction_helper_and_original_fps_for_skip_two \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_processed_video_review_honors_skip_frames_and_reports_progress -v
```

Expected: both tests pass.

---

### Task 4: Visible Prediction and Encoding Errors

**Files:**
- Test: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/tests/test_gradio_sam_preview.py`
- Modify: `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/co-tracker-main/gradio_demo/sam_preview_service.py`

- [ ] **Step 1: Write failing tests for prediction and encoding failures**

Add these methods to `GradioSamPreviewTest`:

```python
    def test_processed_video_review_surfaces_prediction_errors(self):
        video_frames = np.zeros((1, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((1, 10, 20, 3), dtype=np.uint8)
        runtime = {
            "predictor": FakeSamPredictor(),
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ), mock.patch.object(
            sam_preview_service,
            "predict_sam_preview_mask",
            side_effect=ValueError("predict failed"),
        ):
            generator = app.preview_sam_video_for_processed_frames(
                video_frames,
                video_preview,
                [[]],
                np.asarray([[[5.0, 5.0]]], dtype=np.float32),
                np.ones((1, 1), dtype=bool),
                [1],
                video_preview.copy(),
                24,
                "sam2.1_hiera_small.pt",
                0,
                [[]],
                [],
            )
            with self.assertRaises(RuntimeError) as raised:
                list(generator)

        self.assertIn("predict failed", str(raised.exception))

    def test_processed_video_review_surfaces_video_encoding_errors(self):
        video_frames = np.zeros((1, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((1, 10, 20, 3), dtype=np.uint8)
        runtime = {
            "predictor": FakeSamPredictor(),
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ), mock.patch.object(
            sam_preview_service.mediapy,
            "write_video",
            side_effect=ValueError("encode failed"),
        ):
            generator = app.preview_sam_video_for_processed_frames(
                video_frames,
                video_preview,
                [[]],
                np.asarray([[[5.0, 5.0]]], dtype=np.float32),
                np.ones((1, 1), dtype=bool),
                [1],
                video_preview.copy(),
                24,
                "sam2.1_hiera_small.pt",
                0,
                [[]],
                [],
            )
            with self.assertRaises(RuntimeError) as raised:
                list(generator)

        self.assertIn("encode failed", str(raised.exception))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_processed_video_review_surfaces_prediction_errors \
  co-tracker-main.tests.test_gradio_sam_preview.GradioSamPreviewTest.test_processed_video_review_surfaces_video_encoding_errors -v
```

Expected: fail because prediction errors are still uncaught and encoding errors are not converted into visible Gradio errors.

- [ ] **Step 3: Implement encoding error handling**

Wrap the existing `mediapy.write_video(...)` call:

```python
    try:
        mediapy.write_video(video_file_path, np.asarray(review_frames), fps=output_fps)
    except Exception as exc:
        message = f"SAM video review failed while encoding preview video: {exc}"
        yield (
            format_sam_video_progress_html(
                selected_frame_count,
                selected_frame_count,
                "SAM video review encoding failed",
            ),
            None,
            message,
        )
        raise gr.Error(message) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run the same unittest command from Step 2.

Expected: both tests pass.

---

### Task 5: Focused Regression and Git Workflow

**Files:**
- Verify modified Python files and git staging only.

- [ ] **Step 1: Run focused test files**

Run:

```bash
.venv_local/bin/python -m unittest \
  co-tracker-main.tests.test_gradio_sam_preview \
  co-tracker-main.tests.test_gradio_app_wiring -v
```

Expected: all tests in both files pass.

- [ ] **Step 2: Verify plan requirements**

Check each requirement from `/Users/wanglihang/Desktop/feel_intern/cotracker+MSAM/docs/superpowers/specs/2026-07-22-sam-processed-video-preview-design.md` against the implemented code and tests:

```text
skip read on click: prepare_sam_video_preview(video, skip) and app click inputs include both values
skip semantics: selected_sam_video_frame_indices() uses should_process_frame_for_skip()
chronological selected frames: generator iterates selected_frame_indices in order
CoTracker prompts to SAM: sam_point_prompts_for_frame(... prefer_tracked_points=True ...)
visible mask overlay: draw_sam_preview() frames are appended to review_frames
unselected/unmasked absent: only review_frames from successful SAM predictions are encoded
original FPS: mediapy.write_video(... fps=output_fps) where output_fps uses video_fps
visible click feedback: non-queued preparation callback updates progress/status
visible errors: invalid skip, missing state, model load, prediction, and encoding errors surface
no export side effects: no writes to raw-mask-data or dataset in this flow
```

- [ ] **Step 3: Run git status and stage according to AGENTS.md**

Run:

```bash
git status --short
git add -A
git status --short
```

Before committing, unstage any generated files, virtualenvs, model weights, datasets, uploads, outputs, run logs, `.DS_Store`, or `__pycache__` files if they appear staged.

- [ ] **Step 4: Commit and push**

Run:

```bash
git commit -m "Fix processed SAM video preview flow"
git push
```

Expected: commit succeeds and push succeeds on branch `codex/no-wound-yolo-export`.

---

## Self-Review

Spec coverage:

- Preparation callback reads skip and reports selected frame count: Task 1.
- Non-queued preparation before queued generator: Task 2.
- `skip = 2` selects `0, 3, 6, 9` and original FPS is preserved: Task 3.
- CoTracker prompts reach SAM in chronological order: Task 3 uses selected tracks and point-coordinate assertions.
- SAM masks are drawn and only masked frames are encoded: covered by existing processed-video tests plus Task 3.
- Invalid skip, missing state, model loading, prediction, and encoding failures are visible: existing model-load tests plus Tasks 1 and 4.
- No raw-mask or dataset writes: Task 5 checklist.

Placeholder scan:

- The plan contains no deferred placeholders. All code steps include concrete snippets and exact commands.

Type consistency:

- `prepare_sam_video_preview(video_frames, sam_video_skip_frames)` is imported into `app.py`, called by Gradio with `[video, processed_sam_video_skip_frames]`, and tested through `app.prepare_sam_video_preview`.
- `predict_sam_preview_mask(runtime, frame, point_coords, point_labels)` is already defined and becomes the only processed-video SAM prediction path.
