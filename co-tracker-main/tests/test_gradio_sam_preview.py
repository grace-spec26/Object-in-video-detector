import os
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


os.environ.setdefault("COTRACKER_DISABLE_EXAMPLES", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

import gradio as gr  # noqa: E402


with mock.patch.object(gr.Blocks, "launch", lambda self, *args, **kwargs: None):
    import app  # noqa: E402


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

        with mock.patch.object(app, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
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

        with mock.patch.object(app, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
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
            app,
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

        with mock.patch.object(app, "get_sam_preview_runtime_if_ready", return_value=(runtime, None)):
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

        with mock.patch.object(app, "get_sam_preview_runtime", return_value=runtime), mock.patch.object(
            app.mediapy,
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
        final_progress_html, video_path, status = results[-1]
        self.assertIn("100%", final_progress_html)
        self.assertEqual(video_path, written["path"])
        self.assertEqual(written["frames"].shape[0], 4)
        self.assertEqual(len(runtime["predictor"].predict_calls), 2)
        self.assertIn("2/4 frame(s)", status)
        self.assertIn("skipped 2 by skip setting", status)


if __name__ == "__main__":
    unittest.main()
