import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

from tracking_helpers import (  # noqa: E402
    get_cached_cotracker_model,
    get_tracking_resolution,
    resize_video_for_tracking,
)


class FakeModel:
    def __init__(self):
        self.devices = []
        self.eval_calls = 0

    def to(self, device):
        self.devices.append(device)
        return self

    def eval(self):
        self.eval_calls += 1
        return self


class TrackingHelpersTest(unittest.TestCase):
    def test_get_tracking_resolution_uses_video_style_height_and_preserves_aspect_ratio(self):
        self.assertEqual(get_tracking_resolution("720P", source_hw=(1080, 1920)), (720, 1280))
        self.assertEqual(get_tracking_resolution("1080P", source_hw=(2160, 3840)), (1080, 1920))

    def test_get_tracking_resolution_does_not_upscale_small_videos(self):
        self.assertEqual(get_tracking_resolution("1080P", source_hw=(480, 640)), (480, 640))

    def test_resize_video_for_tracking_keeps_frame_count_and_uses_selected_resolution(self):
        video = np.zeros((2, 1080, 1920, 3), dtype=np.uint8)
        video[0, 100, 200] = [255, 0, 0]

        resized = resize_video_for_tracking(video, "640P")

        self.assertEqual(resized.shape, (2, 640, 1138, 3))
        self.assertEqual(resized.dtype, np.uint8)

    def test_get_cached_cotracker_model_loads_once_per_device(self):
        loaded_models = []

        def loader():
            model = FakeModel()
            loaded_models.append(model)
            return model

        cache = {}

        first = get_cached_cotracker_model("cpu", cache=cache, loader=loader)
        second = get_cached_cotracker_model("cpu", cache=cache, loader=loader)
        third = get_cached_cotracker_model("cuda", cache=cache, loader=loader)

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(loaded_models), 2)
        self.assertEqual(first.devices, ["cpu"])
        self.assertEqual(first.eval_calls, 1)
        self.assertEqual(third.devices, ["cuda"])


if __name__ == "__main__":
    unittest.main()
