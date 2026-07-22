import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import cv2 as cv2_module


os.environ.setdefault("COTRACKER_DISABLE_EXAMPLES", "1")
os.environ.setdefault("GRADIO_SKIP_PYI_GENERATION", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYDANTIC_DISABLE_PLUGINS", "__all__")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

class DummyGradioComponent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def click(self, *args, **kwargs):
        return self

    def then(self, *args, **kwargs):
        return self

    def change(self, *args, **kwargs):
        return self

    def select(self, *args, **kwargs):
        return self

    def launch(self, *args, **kwargs):
        return None


gradio_stub = types.ModuleType("gradio")
for component_name in (
    "Accordion",
    "Blocks",
    "Button",
    "Column",
    "Dropdown",
    "File",
    "HTML",
    "Image",
    "Markdown",
    "Number",
    "Radio",
    "Row",
    "Slider",
    "State",
    "Textbox",
    "Video",
):
    setattr(gradio_stub, component_name, DummyGradioComponent)
gradio_stub.Error = RuntimeError
gradio_stub.Examples = lambda *args, **kwargs: DummyGradioComponent(*args, **kwargs)
gradio_stub.SelectData = type("SelectData", (), {})
gradio_stub.Warning = lambda *args, **kwargs: None
gradio_stub.update = lambda **kwargs: kwargs
gradio_stub.data_classes = types.SimpleNamespace(
    PredictBody=types.SimpleNamespace(model_fields={})
)
gradio_stub.networking = types.SimpleNamespace(url_ok=lambda _: True)
mediapy_stub = types.ModuleType("mediapy")
mediapy_stub.read_video = lambda *args, **kwargs: None
mediapy_stub.resize_video = lambda frames, size: frames
mediapy_stub.write_video = lambda *args, **kwargs: None
matplotlib_stub = types.ModuleType("matplotlib")
matplotlib_stub.colormaps = types.SimpleNamespace(
    get_cmap=lambda name: (lambda value: (1.0, 0.0, 0.0, 1.0))
)


with mock.patch.dict(
    sys.modules,
    {
        "gradio": gradio_stub,
        "gradio.data_classes": gradio_stub.data_classes,
        "gradio.networking": gradio_stub.networking,
        "mediapy": mediapy_stub,
        "matplotlib": matplotlib_stub,
    },
):
    import app  # noqa: E402


sam_preview_service = app.sam_preview_service


class FakeSamPredictor:
    def __init__(self, masks=None, scores=None):
        self.image = None
        self.predict_calls = []
        self.masks = masks
        self.scores = scores

    def set_image(self, frame):
        self.image = np.asarray(frame).copy()

    def predict(self, point_coords, point_labels, multimask_output, normalize_coords):
        self.predict_calls.append(
            {
                "point_coords": np.asarray(point_coords).copy(),
                "point_labels": np.asarray(point_labels).copy(),
                "multimask_output": multimask_output,
                "normalize_coords": normalize_coords,
            }
        )
        height, width = self.image.shape[:2]
        if self.masks is None:
            mask = np.zeros((1, height, width), dtype=bool)
            mask[0, 10:30, 40:80] = True
        else:
            mask = np.asarray(self.masks, dtype=bool)
        scores = np.asarray(self.scores if self.scores is not None else [0.9], dtype=np.float32)
        return mask, scores, np.zeros((len(mask), 256, 256), dtype=np.float32)


class GradioSamPreviewTest(unittest.TestCase):
    def test_processed_video_review_first_yield_prepares_skip_selection(self):
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

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ):
            generator = app.preview_sam_video_for_processed_frames(
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
            progress_html, video_path, status = next(generator)

        self.assertIn("Preparing SAM video preview", progress_html)
        self.assertIn("0/4 selected frame(s)", progress_html)
        self.assertIsNone(video_path)
        self.assertIn("4 selected frame(s)", status)
        self.assertIn("from 10 total video frame(s)", status)
        self.assertEqual(len(runtime["predictor"].predict_calls), 0)

    def test_processed_video_review_reports_missing_video_state(self):
        generator = app.preview_sam_video_for_processed_frames(
            None,
            None,
            [],
            None,
            None,
            None,
            None,
            24,
            "sam2.1_hiera_small.pt",
            0,
            [],
            [],
        )
        progress_html, video_path, status = next(generator)

        self.assertEqual(progress_html, app.SAM_VIDEO_PROGRESS_READY)
        self.assertIsNone(video_path)
        self.assertIn("Submit and track a video", status)

    def test_processed_video_review_rejects_negative_skip_value(self):
        video_frames = np.zeros((2, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((2, 10, 20, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError) as raised:
            next(
                app.preview_sam_video_for_processed_frames(
                    video_frames,
                    video_preview,
                    [[], []],
                    np.asarray([[[5.0, 5.0], [6.0, 5.0]]], dtype=np.float32),
                    np.ones((1, 2), dtype=bool),
                    [1],
                    video_preview.copy(),
                    24,
                    "sam2.1_hiera_small.pt",
                    -1,
                    [[], []],
                    [],
                )
            )

        self.assertIn("non-negative", str(raised.exception))

    def _sample_video(self):
        video_frames = np.zeros((2, 100, 200, 3), dtype=np.uint8)
        video_preview = np.zeros((2, 50, 100, 3), dtype=np.uint8)
        query_points = [
            [],
            [
                (25.0, 10.0, 1, 1),
                (70.0, 20.0, 1, 0),
            ],
        ]
        return video_frames, video_preview, query_points

    def test_preview_forwards_scaled_point_prompts_to_sam_predictor(self):
        video_frames, video_preview, query_points = self._sample_video()
        predictor = FakeSamPredictor()
        runtime = {
            "predictor": predictor,
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }

        with mock.patch.object(sam_preview_service, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
            _, status = app.preview_sam_on_frame(
                video_frames,
                video_preview,
                query_points,
                None,
                None,
                None,
                1,
                "sam2.1_hiera_small.pt",
            )

        self.assertEqual(len(predictor.predict_calls), 1)
        call = predictor.predict_calls[0]
        np.testing.assert_allclose(
            call["point_coords"],
            np.asarray([[50.0, 20.0], [140.0, 40.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(call["point_labels"], np.asarray([1, 0], dtype=np.int32))
        self.assertTrue(call["multimask_output"])
        self.assertTrue(call["normalize_coords"])
        self.assertIn("point_coords=[[50.0, 20.0], [140.0, 40.0]]", status)
        self.assertIn("point_labels=[1, 0]", status)

    def test_preview_prefers_mask_that_includes_positive_and_excludes_negative_points(self):
        video_frames, video_preview, query_points = self._sample_video()
        bad_mask = np.zeros((100, 200), dtype=bool)
        bad_mask[35:48, 132:150] = True
        good_mask = np.zeros((100, 200), dtype=bool)
        good_mask[15:32, 44:68] = True
        predictor = FakeSamPredictor(
            masks=np.stack([bad_mask, good_mask], axis=0),
            scores=np.asarray([0.99, 0.2], dtype=np.float32),
        )
        runtime = {
            "predictor": predictor,
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }

        with mock.patch.object(sam_preview_service, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
            preview, _ = app.preview_sam_on_frame(
                video_frames,
                video_preview,
                query_points,
                None,
                None,
                None,
                1,
                "sam2.1_hiera_small.pt",
            )

        self.assertGreater(preview[25, 50, 1], 0)
        self.assertEqual(preview[42, 140].tolist(), [255, 0, 0])

    def test_preview_draws_prompt_points_while_sam_model_is_loading(self):
        video_frames, video_preview, query_points = self._sample_video()

        with mock.patch.object(
            sam_preview_service,
            "get_sam_preview_runtime_if_ready",
            return_value=(None, "SAM preview model sam2.1_hiera_small.pt is still loading."),
        ):
            preview, status = app.preview_sam_on_frame(
                video_frames,
                video_preview,
                query_points,
                None,
                None,
                None,
                1,
                "sam2.1_hiera_small.pt",
            )

        self.assertEqual(preview[20, 50].tolist(), [0, 255, 0])
        self.assertIn("point_coords=[[50.0, 20.0], [140.0, 40.0]]", status)
        self.assertIn("point_labels=[1, 0]", status)

    def test_model_switch_preview_resets_to_prompt_frame_without_sam_mask(self):
        video_frames, video_preview, query_points = self._sample_video()

        preview, status = app.preview_sam_prompt_frame_on_model_switch(
            video_frames,
            video_preview,
            query_points,
            None,
            None,
            None,
            1,
            "sam2.1_hiera_base_plus.pt",
        )

        self.assertEqual(preview[20, 50].tolist(), [0, 255, 0])
        self.assertEqual(preview[42, 60].tolist(), [0, 0, 0])
        self.assertIn("SAM model changed to SAM2.1 Hiera Base Plus", status)
        self.assertIn("Loaded prompts: point_coords=[[50.0, 20.0], [140.0, 40.0]]", status)

    def test_model_switch_starts_runtime_preload_without_streaming_queue(self):
        video_frames, video_preview, query_points = self._sample_video()

        with mock.patch.object(sam_preview_service, "start_sam_preview_preload") as start_preload:
            preview, progress, status = app.sam_model_switch_preview_with_progress(
                video_frames,
                video_preview,
                query_points,
                None,
                None,
                None,
                1,
                "sam2.1_hiera_base_plus.pt",
            )

        start_preload.assert_called_once_with("sam2.1_hiera_base_plus.pt")
        self.assertEqual(preview[20, 50].tolist(), [0, 255, 0])
        self.assertIn("SAM2.1 Hiera Base Plus", progress)
        self.assertIn("Press Preview SAM to run this model", status)

    def test_preview_waits_for_ready_runtime_when_checkpoint_is_local(self):
        video_frames, video_preview, query_points = self._sample_video()
        predictor = FakeSamPredictor()
        runtime = {
            "predictor": predictor,
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2 Base Plus",
            "device": "cpu",
            "image_cache_key": None,
        }
        wait_values = []

        def fake_runtime_if_ready(_sam_model, wait_for_ready_seconds=0):
            wait_values.append(wait_for_ready_seconds)
            if wait_for_ready_seconds > 0:
                return runtime, None
            return None, "SAM preview model is still loading."

        with mock.patch.object(
            sam_preview_service,
            "get_sam_preview_runtime_if_ready",
            side_effect=fake_runtime_if_ready,
        ):
            preview, status = app.preview_sam_on_frame(
                video_frames,
                video_preview,
                query_points,
                None,
                None,
                None,
                1,
                "sam2.1_hiera_base_plus.pt",
            )

        self.assertGreater(wait_values[0], 0)
        self.assertEqual(len(predictor.predict_calls), 1)
        self.assertIn("Fake SAM2 Base Plus", status)
        self.assertGreater(preview[25, 50, 1], 0)

    def test_sam_model_progress_reports_checkpoint_download_percentage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam2.1_hiera_small.pt"
            temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".download")
            temporary_checkpoint.write_bytes(b"x" * 25)
            model_option = {
                "label": "SAM2.1 Hiera Small",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }

            with mock.patch.object(sam_preview_service, "resolve_sam_preview_model_option", return_value=model_option):
                percent, message = app.sam_model_checkpoint_download_progress(
                    "sam2.1_hiera_small.pt"
                )

        self.assertEqual(percent, 25)
        self.assertIn("Downloading SAM2.1 Hiera Small", message)
        self.assertIn("25/100 bytes", message)

    def test_sam_model_progress_counts_active_curl_range_resume_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam2.1_hiera_small.pt"
            temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".download")
            range_checkpoint = temporary_checkpoint.with_suffix(temporary_checkpoint.suffix + ".range")
            temporary_checkpoint.write_bytes(b"x" * 25)
            range_checkpoint.write_bytes(b"y" * 15)
            model_option = {
                "label": "SAM2.1 Hiera Small",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }

            with mock.patch.object(sam_preview_service, "resolve_sam_preview_model_option", return_value=model_option):
                percent, message = app.sam_model_checkpoint_download_progress(
                    "sam2.1_hiera_small.pt"
                )

        self.assertEqual(percent, 40)
        self.assertIn("Downloading SAM2.1 Hiera Small", message)
        self.assertIn("40/100 bytes", message)

    def test_sam_video_progress_bar_displays_completed_frame_fraction(self):
        progress_html = sam_preview_service.format_sam_video_progress_html(
            1,
            7,
            "Processed selected frame",
        )

        self.assertIn(">1/7", progress_html)
        self.assertIn("14%", progress_html)

    def test_sam_model_progress_prefers_partial_download_over_sparse_placeholder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam2.1_hiera_large.pt"
            checkpoint.write_bytes(b"")
            with checkpoint.open("r+b") as handle:
                handle.truncate(100)
            temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".download")
            temporary_checkpoint.write_bytes(b"x" * 25)
            model_option = {
                "label": "SAM2.1 Hiera Large",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }

            with mock.patch.object(sam_preview_service, "resolve_sam_preview_model_option", return_value=model_option):
                percent, message = app.sam_model_checkpoint_download_progress(
                    "sam2.1_hiera_large.pt"
                )

        self.assertEqual(percent, 25)
        self.assertIn("Downloading SAM2.1 Hiera Large", message)
        self.assertIn("25/100 bytes", message)

    def test_sam_model_progress_does_not_trust_sparse_partial_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam2.1_hiera_large.pt"
            temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".download")
            temporary_checkpoint.write_bytes(b"x" * 25)
            model_option = {
                "label": "SAM2.1 Hiera Large",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }

            def fake_unavailable(path):
                return Path(path).name.endswith(".download")

            with mock.patch.object(sam_preview_service, "resolve_sam_preview_model_option", return_value=model_option):
                with mock.patch.object(
                    sam_preview_service,
                    "sam_checkpoint_file_looks_unavailable",
                    side_effect=fake_unavailable,
                ):
                    percent, message = app.sam_model_checkpoint_download_progress(
                        "sam2.1_hiera_large.pt"
                    )

        self.assertEqual(percent, 0)
        self.assertIn("partial download is a local placeholder", message)

    def test_failed_sam_model_progress_keeps_actual_download_percentage(self):
        model_name = "sam2.1_hiera_large.pt"
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / model_name
            checkpoint.write_bytes(b"")
            with checkpoint.open("r+b") as handle:
                handle.truncate(100)
            temporary_checkpoint = checkpoint.with_suffix(checkpoint.suffix + ".download")
            temporary_checkpoint.write_bytes(b"x" * 25)
            model_option = {
                "label": "SAM2.1 Hiera Large",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }
            with app.sam_preview_preload_lock:
                original_errors = dict(app.sam_preview_preload_errors)
                original_started = set(app.sam_preview_preload_started)
                app.sam_preview_preload_errors[model_name] = "curl exited with status 56"
                app.sam_preview_preload_started.discard(model_name)
            try:
                with mock.patch.object(sam_preview_service, "resolve_sam_preview_model_option", return_value=model_option):
                    progress_html = app.current_sam_model_progress_html(model_name)
            finally:
                with app.sam_preview_preload_lock:
                    app.sam_preview_preload_errors.clear()
                    app.sam_preview_preload_errors.update(original_errors)
                    app.sam_preview_preload_started.clear()
                    app.sam_preview_preload_started.update(original_started)

        self.assertIn("25%", progress_html)
        self.assertIn("failed to load", progress_html)
        self.assertIn("curl exited with status 56", progress_html)
        self.assertNotIn(">100%</span>", progress_html)

    def test_sam_model_progress_marks_loaded_runtime_complete(self):
        model_name = "sam2.1_hiera_small.pt"
        with app.sam_preview_runtime_lock:
            original_runtimes = dict(app.sam_preview_runtimes)
            app.sam_preview_runtimes[model_name] = {
                "model_label": "SAM2.1 Hiera Small",
                "device": "cpu",
            }
        try:
            progress_html = app.current_sam_model_progress_html(model_name)
        finally:
            with app.sam_preview_runtime_lock:
                app.sam_preview_runtimes.clear()
                app.sam_preview_runtimes.update(original_runtimes)

        self.assertIn("SAM2.1 Hiera Small loaded", progress_html)
        self.assertIn("100%", progress_html)

    def test_sam_model_progress_does_not_block_while_runtime_load_holds_lock(self):
        model_name = "sam2.1_hiera_large.pt"
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / model_name
            checkpoint.write_bytes(b"partial")
            model_option = {
                "label": "SAM2.1 Hiera Large",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }
            with app.sam_preview_runtime_lock:
                with mock.patch.object(
                    sam_preview_service,
                    "resolve_sam_preview_model_option",
                    return_value=model_option,
                ):
                    result = []
                    worker = threading.Thread(
                        target=lambda: result.append(app.current_sam_model_progress_html(model_name)),
                    )
                    worker.start()
                    worker.join(timeout=0.2)
                    finished_while_locked = not worker.is_alive()

        if worker.is_alive():
            worker.join(timeout=1)
        self.assertTrue(finished_while_locked)
        self.assertEqual(len(result), 1)
        self.assertIn("incomplete", result[0])

    def test_sam_model_progress_keeps_placeholder_message_while_loading(self):
        model_name = "sam2.1_hiera_large.pt"
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / model_name
            with checkpoint.open("wb") as handle:
                handle.truncate(100)
            model_option = {
                "label": "SAM2.1 Hiera Large",
                "checkpoint": checkpoint,
                "expected_size": 100,
            }
            with app.sam_preview_preload_lock:
                original_started = set(app.sam_preview_preload_started)
                original_errors = dict(app.sam_preview_preload_errors)
                app.sam_preview_preload_started.add(model_name)
                app.sam_preview_preload_errors.pop(model_name, None)
            try:
                with mock.patch.object(
                    sam_preview_service,
                    "resolve_sam_preview_model_option",
                    return_value=model_option,
                ):
                    progress_html = app.current_sam_model_progress_html(model_name)
            finally:
                with app.sam_preview_preload_lock:
                    app.sam_preview_preload_started.clear()
                    app.sam_preview_preload_started.update(original_started)
                    app.sam_preview_preload_errors.clear()
                    app.sam_preview_preload_errors.update(original_errors)

        self.assertIn("local placeholder", progress_html)
        self.assertNotIn("Preparing SAM2.1 Hiera Large", progress_html)

    def test_get_sam_preview_runtime_loads_without_holding_runtime_lock(self):
        model_name = "sam2.1_hiera_large.pt"
        fake_sam2_wrapper = types.ModuleType("sam2_coordinate_wrapper")
        lock_was_available_during_load = []

        def fake_load_sam2_predictor(model_name, download_checkpoint):
            acquired = app.sam_preview_runtime_lock.acquire(blocking=False)
            lock_was_available_during_load.append(acquired)
            if acquired:
                app.sam_preview_runtime_lock.release()
            return object(), "cpu"

        fake_sam2_wrapper.load_sam2_predictor = fake_load_sam2_predictor
        fake_sam2_wrapper.resolve_sam2_model_option = lambda model_name: {
            "label": "SAM2.1 Hiera Large",
        }

        with app.sam_preview_runtime_lock:
            original_runtimes = dict(app.sam_preview_runtimes)
            app.sam_preview_runtimes.pop(model_name, None)
        try:
            with mock.patch.dict(sys.modules, {"sam2_coordinate_wrapper": fake_sam2_wrapper}):
                runtime = app.get_sam_preview_runtime(model_name)
        finally:
            with app.sam_preview_runtime_lock:
                app.sam_preview_runtimes.clear()
                app.sam_preview_runtimes.update(original_runtimes)

        self.assertEqual(lock_was_available_during_load, [True])
        self.assertEqual(runtime["model_label"], "SAM2.1 Hiera Large")

    def test_processed_preview_scales_refinement_points_from_tracked_preview_space(self):
        video_frames = np.zeros((2, 100, 200, 3), dtype=np.uint8)
        video_preview = np.zeros((2, 50, 100, 3), dtype=np.uint8)
        tracked_video_preview = np.zeros((2, 80, 160, 3), dtype=np.uint8)
        refinement_points = [
            [],
            [
                (80.0, 40.0, 1, 1),
                (120.0, 60.0, 1, 0),
            ],
        ]
        predictor = FakeSamPredictor()
        runtime = {
            "predictor": predictor,
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }

        with mock.patch.object(sam_preview_service, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
            _, status = app.preview_sam_for_selected_frame(
                video_frames,
                video_preview,
                [[], []],
                np.empty((0, 2, 2), dtype=np.float32),
                np.empty((0, 2), dtype=bool),
                [],
                0,
                1,
                tracked_video_preview,
                "sam2.1_hiera_small.pt",
                refinement_points,
                [],
            )

        self.assertEqual(len(predictor.predict_calls), 1)
        np.testing.assert_allclose(
            predictor.predict_calls[0]["point_coords"],
            np.asarray([[100.0, 50.0], [150.0, 75.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            predictor.predict_calls[0]["point_labels"],
            np.asarray([1, 0], dtype=np.int32),
        )
        self.assertIn("point_coords=[[100.0, 50.0], [150.0, 75.0]]", status)

    def test_processed_video_review_honors_skip_frames_and_reports_progress(self):
        video_frames = np.zeros((4, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((4, 10, 20, 3), dtype=np.uint8)
        selected_tracks = np.asarray(
            [
                [
                    [5.0, 5.0],
                    [6.0, 5.0],
                    [7.0, 5.0],
                    [8.0, 5.0],
                ],
            ],
            dtype=np.float32,
        )
        runtime = {
            "predictor": FakeSamPredictor(),
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }
        written = {}

        def fake_write_video(path, frames, fps):
            written["path"] = path
            written["frames"] = np.asarray(frames)
            written["fps"] = fps

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ), mock.patch.object(
            sam_preview_service.mediapy,
            "write_video",
            side_effect=fake_write_video,
        ):
            results = list(
                app.preview_sam_video_for_processed_frames(
                    video_frames,
                    video_preview,
                    [[], [], [], []],
                    selected_tracks,
                    np.ones((1, 4), dtype=bool),
                    [1],
                    video_preview.copy(),
                    24,
                    "sam2.1_hiera_small.pt",
                    1,
                    [[], [], [], []],
                    [],
                )
            )

        self.assertGreater(len(results), 1)
        self.assertIn("0/2 selected frame(s)", results[0][0])
        final_progress_html, video_path, status = results[-1]
        self.assertIn("100%", final_progress_html)
        self.assertIn("path", written)
        self.assertEqual(video_path, written["path"])
        self.assertEqual(written["frames"].shape[0], 2)
        self.assertEqual(len(runtime["predictor"].predict_calls), 2)
        self.assertIn("2/2 selected frame(s)", status)
        self.assertIn("from 4 total video frame(s)", status)

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

    def test_processed_video_review_yields_processing_status_before_running_sam(self):
        video_frames = np.zeros((4, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((4, 10, 20, 3), dtype=np.uint8)
        selected_tracks = np.asarray(
            [
                [
                    [5.0, 5.0],
                    [6.0, 5.0],
                    [7.0, 5.0],
                    [8.0, 5.0],
                ],
            ],
            dtype=np.float32,
        )
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
        ):
            generator = app.preview_sam_video_for_processed_frames(
                video_frames,
                video_preview,
                [[], [], [], []],
                selected_tracks,
                np.ones((1, 4), dtype=bool),
                [1],
                video_preview.copy(),
                24,
                "sam2.1_hiera_small.pt",
                1,
                [[], [], [], []],
                [],
            )
            first_progress_html, _, first_status = next(generator)
            second_progress_html, _, second_status = next(generator)
            third_progress_html, _, third_status = next(generator)
            fourth_progress_html, _, fourth_status = next(generator)

        self.assertIn("0/2 selected frame(s)", first_progress_html)
        self.assertIn("Preparing SAM video preview", first_status)
        self.assertIn("0/2 selected frame(s)", second_progress_html)
        self.assertIn("Loading SAM model", second_status)
        self.assertIn("0/2 selected frame(s)", third_progress_html)
        self.assertIn("Starting SAM video review", third_status)
        self.assertIn("Processing selected frame 1/2", fourth_progress_html)
        self.assertIn("video frame 1/4", fourth_status)
        self.assertEqual(len(runtime["predictor"].predict_calls), 0)

    def test_processed_video_review_reports_model_load_error_without_blocking(self):
        model_name = "sam2.1_hiera_large.pt"
        video_frames = np.zeros((2, 20, 40, 3), dtype=np.uint8)
        video_preview = np.zeros((2, 10, 20, 3), dtype=np.uint8)
        selected_tracks = np.asarray(
            [
                [
                    [5.0, 5.0],
                    [6.0, 5.0],
                ],
            ],
            dtype=np.float32,
        )

        with app.sam_preview_preload_lock:
            original_started = set(app.sam_preview_preload_started)
            original_errors = dict(app.sam_preview_preload_errors)
            app.sam_preview_preload_started.discard(model_name)
            app.sam_preview_preload_errors[model_name] = "download failed"
        try:
            with mock.patch.object(
                sam_preview_service,
                "get_loaded_sam_preview_runtime",
                return_value=None,
            ), mock.patch.object(
                sam_preview_service,
                "start_sam_preview_preload",
            ) as start_preload, mock.patch.object(
                sam_preview_service,
                "get_sam_preview_runtime",
                side_effect=AssertionError("blocking runtime load should not run"),
            ), mock.patch.object(
                sam_preview_service,
                "sam_model_checkpoint_download_progress",
                return_value=(0, "SAM2.1 Hiera Large checkpoint waiting to download"),
            ):
                results = list(
                    app.preview_sam_video_for_processed_frames(
                        video_frames,
                        video_preview,
                        [[], []],
                        selected_tracks,
                        np.ones((1, 2), dtype=bool),
                        [1],
                        video_preview.copy(),
                        24,
                        model_name,
                        0,
                        [[], []],
                        [],
                    )
                )
        finally:
            with app.sam_preview_preload_lock:
                app.sam_preview_preload_started.clear()
                app.sam_preview_preload_started.update(original_started)
                app.sam_preview_preload_errors.clear()
                app.sam_preview_preload_errors.update(original_errors)

        self.assertGreaterEqual(len(results), 2)
        start_preload.assert_called_once_with(model_name)
        final_progress_html, video_path, status = results[-1]
        self.assertIn("0/2", final_progress_html)
        self.assertIsNone(video_path)
        self.assertIn("failed to load", status)
        self.assertIn("download failed", status)
        self.assertIn("checkpoint waiting to download", status)

    def test_processed_video_review_writes_unmasked_frames_when_no_frames_run_sam(self):
        video_frames = np.zeros((2, 20, 40, 3), dtype=np.uint8)
        video_frames[0, :, :, 0] = 25
        video_frames[1, :, :, 1] = 75
        video_preview = np.zeros((2, 10, 20, 3), dtype=np.uint8)
        runtime = {
            "predictor": FakeSamPredictor(),
            "predictor_lock": threading.Lock(),
            "model_label": "Fake SAM2",
            "device": "cpu",
            "image_cache_key": None,
        }
        written = {}

        def fake_write_video(path, frames, fps):
            written["path"] = path
            written["frames"] = np.asarray(frames)
            written["fps"] = fps

        with mock.patch.object(
            sam_preview_service,
            "get_loaded_sam_preview_runtime",
            return_value=runtime,
        ), mock.patch.object(
            sam_preview_service.mediapy,
            "write_video",
            side_effect=fake_write_video,
        ):
            results = list(
                app.preview_sam_video_for_processed_frames(
                    video_frames,
                    video_preview,
                    [[], []],
                    np.empty((0, 2, 2), dtype=np.float32),
                    np.empty((0, 2), dtype=bool),
                    [],
                    video_preview.copy(),
                    24,
                    "sam2.1_hiera_small.pt",
                    0,
                    [[], []],
                    [],
                )
            )

        final_progress_html, video_path, status = results[-1]
        self.assertIn("100%", final_progress_html)
        self.assertIn("path", written)
        self.assertEqual(video_path, written["path"])
        np.testing.assert_array_equal(written["frames"], video_frames)
        self.assertEqual(len(runtime["predictor"].predict_calls), 0)
        self.assertIn("0 SAM-masked frame(s)", status)
        self.assertIn("2 unmasked frame(s)", status)
        self.assertIn("unmasked 2 without points", status)

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


if __name__ == "__main__":
    unittest.main()
