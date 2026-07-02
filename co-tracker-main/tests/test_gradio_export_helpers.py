import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gradio_demo"))

from export_helpers import (  # noqa: E402
    build_sam_point_prompt_payload,
    scale_tracks_to_frame_space,
    store_coordinate_arrays,
    store_original_frames,
    visible_labeled_points_for_frame,
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

    def test_store_coordinate_arrays_filters_invisible_tracks(self):
        tracks = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ],
            dtype=np.float32,
        )
        visibility = np.array(
            [
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            store_coordinate_arrays(
                tracks=tracks,
                output_dir=output_dir,
                source_hw=(100, 200),
                target_hw=(300, 400),
                visibility=visibility,
            )

            self.assertEqual(
                json.loads((output_dir / "frame_000000.json").read_text(encoding="utf-8")),
                [[20.0, 60.0]],
            )
            self.assertEqual(
                json.loads((output_dir / "frame_000001.json").read_text(encoding="utf-8")),
                [[140.0, 240.0]],
            )

    def test_store_coordinate_arrays_rejects_mismatched_visibility_shape(self):
        tracks = np.zeros((2, 3, 2), dtype=np.float32)
        visibility = np.ones((3, 2), dtype=bool)

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Visibility"):
                store_coordinate_arrays(
                    tracks=tracks,
                    output_dir=Path(tmp),
                    visibility=visibility,
                )

    def test_store_coordinate_arrays_writes_sam_prompt_json_with_labels(self):
        tracks = np.array(
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[50.0, 60.0], [70.0, 80.0]],
                [[90.0, 95.0], [100.0, 110.0]],
            ],
            dtype=np.float32,
        )
        labels = [1, 0, 1]
        visibility = np.array(
            [
                [True, True],
                [True, False],
                [False, True],
            ],
            dtype=bool,
        )

        with TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            store_coordinate_arrays(
                tracks=tracks,
                output_dir=output_dir,
                source_hw=(100, 200),
                target_hw=(300, 400),
                visibility=visibility,
                point_labels=labels,
            )

            frame_0 = json.loads((output_dir / "frame_000000.json").read_text(encoding="utf-8"))
            self.assertEqual(frame_0["image_size"], {"width": 400, "height": 300})
            self.assertEqual(frame_0["frame_index"], 0)
            self.assertEqual(
                frame_0["objects"][0],
                {
                    "class_id": 1,
                    "positive_points": [[20.0, 60.0]],
                    "negative_points": [[100.0, 180.0]],
                    "point_coords": [[20.0, 60.0], [100.0, 180.0]],
                    "point_labels": [1, 0],
                },
            )

            frame_1 = json.loads((output_dir / "frame_000001.json").read_text(encoding="utf-8"))
            self.assertEqual(frame_1["objects"][0]["positive_points"], [[60.0, 120.0], [200.0, 299.0]])
            self.assertEqual(frame_1["objects"][0]["negative_points"], [])
            self.assertEqual(frame_1["objects"][0]["point_labels"], [1, 1])

            aggregate = json.loads((output_dir / "coordinates.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate["format"], "sam_point_prompts")
            self.assertEqual(len(aggregate["frames"]), 2)

    def test_store_coordinate_arrays_rejects_mismatched_point_labels(self):
        tracks = np.zeros((2, 3, 2), dtype=np.float32)

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "point_labels"):
                store_coordinate_arrays(
                    tracks=tracks,
                    output_dir=Path(tmp),
                    point_labels=[1],
                )

    def test_visible_labeled_points_for_frame_filters_visibility_and_labels(self):
        tracks = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )
        visibility = np.array([[False, True], [True, False]], dtype=bool)

        coords, labels = visible_labeled_points_for_frame(
            tracks,
            1,
            point_labels=[1, 0],
            visibility=visibility,
        )

        np.testing.assert_allclose(coords, np.array([[3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(labels, np.array([1], dtype=np.int32))

    def test_build_sam_point_prompt_payload_splits_positive_and_negative_points(self):
        payload = build_sam_point_prompt_payload(
            np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            [1, 0, 1],
            image_size=(10, 20),
        )

        self.assertEqual(payload["image_size"], {"width": 20, "height": 10})
        self.assertEqual(payload["objects"][0]["positive_points"], [[1.0, 2.0], [5.0, 6.0]])
        self.assertEqual(payload["objects"][0]["negative_points"], [[3.0, 4.0]])
        self.assertEqual(payload["objects"][0]["point_labels"], [1, 0, 1])

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
