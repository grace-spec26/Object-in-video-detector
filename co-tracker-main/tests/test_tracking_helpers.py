import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

from tracking_helpers import (  # noqa: E402
    expand_sampled_time_axis,
    get_cached_cotracker_model,
    get_tracking_resolution,
    map_frame_index_to_sampled,
    resolve_torch_device,
    resize_video_for_tracking,
    subsample_video_tensor,
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


class FakeCuda:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeMps:
    def __init__(self, available):
        self.available = available

    def is_available(self):
        return self.available


class FakeBackends:
    def __init__(self, mps_available):
        self.mps = FakeMps(mps_available)


class FakeTorch:
    def __init__(self, cuda_available=False, mps_available=False):
        self.cuda = FakeCuda(cuda_available)
        self.backends = FakeBackends(mps_available)


class TrackingHelpersTest(unittest.TestCase):
    def test_resolve_torch_device_prefers_cuda_then_mps_then_cpu(self):
        self.assertEqual(resolve_torch_device(FakeTorch(cuda_available=True, mps_available=True)), "cuda")
        self.assertEqual(resolve_torch_device(FakeTorch(cuda_available=False, mps_available=True)), "mps")
        self.assertEqual(resolve_torch_device(FakeTorch(cuda_available=False, mps_available=False)), "cpu")

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

    def test_subsample_video_tensor_keeps_every_second_frame_on_time_axis(self):
        video_tensor = np.arange(1 * 5 * 1 * 1 * 1).reshape(1, 5, 1, 1, 1)

        sampled = subsample_video_tensor(video_tensor, stride=2)

        np.testing.assert_array_equal(sampled[:, :, 0, 0, 0], np.array([[0, 2, 4]]))

    def test_map_frame_index_to_sampled_uses_previous_available_frame(self):
        self.assertEqual(map_frame_index_to_sampled(0, sampled_frame_count=3, stride=2), 0)
        self.assertEqual(map_frame_index_to_sampled(1, sampled_frame_count=3, stride=2), 0)
        self.assertEqual(map_frame_index_to_sampled(5, sampled_frame_count=3, stride=2), 2)

    def test_expand_sampled_time_axis_restores_full_frame_count(self):
        sampled_tracks = np.array(
            [
                [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]],
                [[40.0, 41.0], [50.0, 51.0], [60.0, 61.0]],
            ],
            dtype=np.float32,
        )

        expanded = expand_sampled_time_axis(sampled_tracks, total_frames=6, stride=2, axis=1)

        self.assertEqual(expanded.shape, (2, 6, 2))
        np.testing.assert_array_equal(expanded[:, 0], sampled_tracks[:, 0])
        np.testing.assert_array_equal(expanded[:, 1], sampled_tracks[:, 0])
        np.testing.assert_array_equal(expanded[:, 4], sampled_tracks[:, 2])
        np.testing.assert_array_equal(expanded[:, 5], sampled_tracks[:, 2])

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

    def test_get_cached_cotracker_model_loads_local_patched_repo_by_default(self):
        class FakeHub:
            def __init__(self):
                self.calls = []

            def load(self, *args, **kwargs):
                self.calls.append((args, kwargs))
                return FakeModel()

        class FakeTorchModule:
            def __init__(self):
                self.hub = FakeHub()

        fake_torch = FakeTorchModule()
        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch

        try:
            model = get_cached_cotracker_model("mps", cache={})
        finally:
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertEqual(model.devices, ["mps"])
        self.assertEqual(len(fake_torch.hub.calls), 1)
        args, kwargs = fake_torch.hub.calls[0]
        self.assertEqual(args[1], "cotracker3_online")
        self.assertEqual(kwargs, {"source": "local"})
        self.assertTrue((Path(args[0]) / "hubconf.py").is_file())


if __name__ == "__main__":
    unittest.main()
