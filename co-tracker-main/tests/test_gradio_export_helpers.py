import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

from export_helpers import (  # noqa: E402
    scale_tracks_to_frame_space,
    store_coordinate_arrays,
    store_original_frames,
)


class ExportHelpersTest(unittest.TestCase):
    def test_store_original_frames_writes_all_frames_with_stable_names(self):
        frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
        frames[0, :, :] = [255, 0, 0]
        frames[1, :, :] = [0, 255, 0]

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "frame_999999.png").write_text("stale", encoding="utf-8")

            written = store_original_frames(frames, output_dir)

            self.assertEqual(
                [path.name for path in written],
                ["frame_000000.png", "frame_000001.png"],
            )
            self.assertFalse((output_dir / "frame_999999.png").exists())
            self.assertTrue((output_dir / "frame_000000.png").exists())
            self.assertTrue((output_dir / "frame_000001.png").exists())

    def test_store_coordinate_arrays_scales_preview_tracks_to_original_pixels(self):
        tracks = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ],
            dtype=np.float32,
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            written = store_coordinate_arrays(
                tracks=tracks,
                output_dir=output_dir,
                source_hw=(100, 200),
                target_hw=(300, 400),
            )

            self.assertEqual(
                [path.name for path in written],
                ["frame_000000.json", "frame_000001.json", "coordinates.json"],
            )
            self.assertEqual(
                json.loads((output_dir / "frame_000000.json").read_text(encoding="utf-8")),
                [[20.0, 60.0], [100.0, 180.0]],
            )
            self.assertEqual(
                json.loads((output_dir / "coordinates.json").read_text(encoding="utf-8")),
                [
                    [[20.0, 60.0], [100.0, 180.0]],
                    [[60.0, 120.0], [140.0, 240.0]],
                ],
            )

    def test_scale_tracks_to_frame_space_clips_to_target_bounds(self):
        tracks = np.array([[[250.0, -10.0]]], dtype=np.float32)

        scaled = scale_tracks_to_frame_space(
            tracks=tracks,
            source_hw=(100, 200),
            target_hw=(300, 400),
        )

        np.testing.assert_allclose(scaled, np.array([[[399.0, 0.0]]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
