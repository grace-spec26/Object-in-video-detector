import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobilesam_coordinate_wrapper import (  # noqa: E402
    NEGATIVE_MODES,
    build_augmented_prompt_json,
    format_coordinate_progress_html,
    generate_expanded_box_negative_points,
    iter_coordinate_prompt_folder_steps,
    prepare_coordinate_prompt_json,
    run_mobilesam_for_frame,
    run_coordinate_prompt_folders,
    select_frames_for_frame_step,
)


class FakePredictor:
    def __init__(self, masks=None, scores=None):
        self.calls = []
        self.masks = masks
        self.scores = scores

    def set_image(self, image):
        self.image_shape = image.shape

    def predict(self, point_coords, point_labels, multimask_output, box=None):
        self.calls.append(
            {
                "coords": np.asarray(point_coords).tolist(),
                "labels": np.asarray(point_labels).tolist(),
                "multimask_output": multimask_output,
                "box": None if box is None else np.asarray(box).tolist(),
            }
        )
        if self.masks is not None:
            return (
                np.asarray(self.masks),
                np.asarray(self.scores, dtype=np.float32),
                None,
            )
        mask = np.zeros(self.image_shape[:2], dtype=bool)
        mask[1:3, 1:3] = True
        return np.asarray([mask]), np.asarray([0.9], dtype=np.float32), None


class MobileSAMCoordinateWrapperTest(unittest.TestCase):
    def test_build_augmented_prompt_json_defaults_to_box_8_points(self):
        prompt = build_augmented_prompt_json(
            positive_points=[[50, 50], [70, 90]],
            image_width=200,
            image_height=200,
        )

        obj = prompt["objects"][0]
        np.testing.assert_allclose(obj["box"], [30.0, 30.0, 90.0, 110.0])
        np.testing.assert_allclose(
            obj["negative_points"],
            [
                [30.0, 30.0],
                [90.0, 30.0],
                [90.0, 110.0],
                [30.0, 110.0],
                [60.0, 30.0],
                [90.0, 70.0],
                [60.0, 110.0],
                [30.0, 70.0],
            ],
        )
        self.assertEqual(obj["point_labels"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(prompt["padding_ratio"], 0.35)
        self.assertEqual(prompt["negative_mode"], "box_8_points")

    def test_build_augmented_prompt_json_adds_expanded_corner_negatives(self):
        prompt = build_augmented_prompt_json(
            positive_points=[[10, 20], [30, 50]],
            image_width=100,
            image_height=80,
            padding_ratio=0.15,
            min_padding_px=0,
            min_negative_distance=0,
            negative_mode="box_4_corners",
        )

        obj = prompt["objects"][0]
        np.testing.assert_allclose(obj["box"], [7.0, 15.5, 33.0, 54.5])
        np.testing.assert_allclose(
            obj["point_coords"],
            [
                [10.0, 20.0],
                [30.0, 50.0],
                [7.0, 15.5],
                [33.0, 15.5],
                [33.0, 54.5],
                [7.0, 54.5],
            ],
        )
        self.assertEqual(obj["point_labels"], [1, 1, 0, 0, 0, 0])
        self.assertEqual(obj["positive_points"], [[10.0, 20.0], [30.0, 50.0]])
        np.testing.assert_allclose(
            obj["negative_points"],
            [[7.0, 15.5], [33.0, 15.5], [33.0, 54.5], [7.0, 54.5]],
        )

    def test_build_augmented_prompt_json_clamps_negative_points_to_image(self):
        prompt = build_augmented_prompt_json(
            positive_points=[[0, 0], [99, 79]],
            image_width=100,
            image_height=80,
            padding_ratio=0.15,
            min_padding_px=0,
            min_negative_distance=0,
            negative_mode="box_4_corners",
        )

        self.assertEqual(
            prompt["objects"][0]["negative_points"],
            [[0.0, 0.0], [99.0, 0.0], [99.0, 79.0], [0.0, 79.0]],
        )

    def test_oriented_side_points_generate_negatives_on_both_sides_of_diagonal_line(self):
        prompt = build_augmented_prompt_json(
            positive_points=[[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]],
            image_width=100,
            image_height=100,
            min_padding_px=10,
            min_negative_distance=0,
            negative_mode="oriented_side_points",
        )

        obj = prompt["objects"][0]
        self.assertEqual(prompt["negative_mode"], "oriented_side_points")
        self.assertEqual(obj["negative_mode"], "oriented_side_points")
        self.assertEqual(obj["point_labels"], [1, 1, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(
            obj["negative_points"],
            [
                [12.928932, 27.071068],
                [32.928932, 47.071068],
                [52.928932, 67.071068],
                [27.071068, 12.928932],
                [47.071068, 32.928932],
                [67.071068, 52.928932],
            ],
            rtol=1e-5,
            atol=1e-5,
        )

    def test_box_8_oriented_combines_box_8_and_oriented_side_negatives(self):
        positive_points = np.asarray(
            [[20.0, 20.0], [40.0, 40.0], [60.0, 60.0]],
            dtype=np.float32,
        )

        prompt = build_augmented_prompt_json(
            positive_points=positive_points,
            image_width=100,
            image_height=100,
            padding_ratio=0.15,
            min_padding_px=10,
            min_negative_distance=0,
            negative_mode="box_8_oriented",
        )
        box_8_negatives = generate_expanded_box_negative_points(
            positive_points,
            image_width=100,
            image_height=100,
            padding_ratio=0.15,
            min_padding_px=10,
            min_negative_distance=0,
            negative_mode="box_8_points",
        )
        oriented_negatives = generate_expanded_box_negative_points(
            positive_points,
            image_width=100,
            image_height=100,
            padding_ratio=0.15,
            min_padding_px=10,
            min_negative_distance=0,
            negative_mode="oriented_side_points",
        )

        obj = prompt["objects"][0]
        self.assertIn("box_8_oriented", NEGATIVE_MODES)
        self.assertEqual(prompt["negative_mode"], "box_8_oriented")
        self.assertEqual(obj["negative_mode"], "box_8_oriented")
        np.testing.assert_allclose(
            obj["negative_points"],
            np.concatenate([box_8_negatives, oriented_negatives], axis=0),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(
            obj["point_labels"],
            [1, 1, 1] + [0] * (len(box_8_negatives) + len(oriented_negatives)),
        )

    def test_prepare_coordinate_prompt_json_does_not_overwrite_source(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "frame_000000.json"
            output_path = tmp_path / "raw-mask-data" / "coordinates" / "frame_000000.json"
            source_json = [[40.0, 40.0], [60.0, 60.0]]
            source_path.write_text(json.dumps(source_json), encoding="utf-8")

            prompt_objects = prepare_coordinate_prompt_json(
                source_path,
                image_width=100,
                image_height=100,
                output_path=output_path,
            )

            self.assertEqual(json.loads(source_path.read_text(encoding="utf-8")), source_json)
            augmented = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(augmented["objects"][0]["positive_points"], source_json)
            self.assertEqual(augmented["objects"][0]["point_labels"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            self.assertEqual(augmented["negative_mode"], "box_8_points")
            self.assertEqual(len(prompt_objects), 1)

    def test_prepare_coordinate_prompt_json_preserves_source_point_labels(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "frame_000000.json"
            output_path = tmp_path / "raw-mask-data" / "coordinates" / "frame_000000.json"
            source_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "class_id": 1,
                                "point_coords": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                                "point_labels": [1, 0, 1],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            prompt_objects = prepare_coordinate_prompt_json(
                source_path,
                image_width=20,
                image_height=20,
                output_path=output_path,
                padding_ratio=0.15,
                min_padding_px=0,
                min_negative_distance=0,
                negative_mode="box_4_corners",
            )

            augmented = json.loads(output_path.read_text(encoding="utf-8"))
            obj = augmented["objects"][0]
            np.testing.assert_allclose(
                obj["point_coords"],
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                ],
            )
            self.assertEqual(obj["point_labels"], [1, 0, 1])
            self.assertEqual(obj["positive_points"], [[1.0, 1.0], [3.0, 3.0]])
            self.assertEqual(obj["negative_points"], [[2.0, 2.0]])
            self.assertNotIn("box", obj)
            self.assertEqual(prompt_objects[0][1].tolist(), [1, 0, 1])

    def test_run_mobilesam_for_frame_passes_generated_negative_labels_to_predictor(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frame_path = tmp_path / "frame_000000.png"
            coordinate_path = tmp_path / "frame_000000.json"
            mask_path = tmp_path / "mask" / "frame_000000.png"
            preview_path = tmp_path / "masked_frame" / "frame_000000.jpg"
            augmented_path = tmp_path / "coordinates" / "frame_000000.json"

            Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(frame_path)
            coordinate_path.write_text(
                json.dumps(
                    {
                        "objects": [
                            {
                                "class_id": 1,
                                "point_coords": [[1.0, 1.0], [4.0, 4.0]],
                                "point_labels": [1, 0],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            predictor = FakePredictor()
            run_mobilesam_for_frame(
                predictor=predictor,
                frame_path=frame_path,
                coordinate_json_path=coordinate_path,
                mask_path=mask_path,
                preview_path=preview_path,
                augmented_coordinate_json_path=augmented_path,
                padding_ratio=0.15,
                min_padding_px=0,
                min_negative_distance=0,
                negative_mode="box_4_corners",
            )

            np.testing.assert_allclose(
                predictor.calls[0]["coords"],
                [[1.0, 1.0], [4.0, 4.0]],
            )
            self.assertEqual(predictor.calls[0]["labels"], [1, 0])
            self.assertIsNone(predictor.calls[0]["box"])
            self.assertTrue(mask_path.exists())
            self.assertTrue(preview_path.exists())

    def test_run_mobilesam_for_frame_prefers_mask_matching_prompt_points(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frame_path = tmp_path / "frame_000000.png"
            coordinate_path = tmp_path / "frame_000000.json"
            mask_path = tmp_path / "mask" / "frame_000000.png"

            Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(frame_path)
            coordinate_path.write_text(json.dumps([[10.0, 10.0]]), encoding="utf-8")

            oversized_mask = np.ones((40, 40), dtype=bool)
            prompt_fit_mask = np.zeros((40, 40), dtype=bool)
            prompt_fit_mask[8:13, 8:13] = True
            predictor = FakePredictor(
                masks=[oversized_mask, prompt_fit_mask],
                scores=[0.99, 0.2],
            )

            run_mobilesam_for_frame(
                predictor=predictor,
                frame_path=frame_path,
                coordinate_json_path=coordinate_path,
                mask_path=mask_path,
            )

            saved_mask = np.asarray(Image.open(mask_path))
            self.assertEqual(saved_mask[10, 10], 1)
            self.assertEqual(saved_mask[0, 0], 0)
            np.testing.assert_allclose(predictor.calls[0]["box"], [0.0, 0.0, 30.0, 30.0])

    def test_select_frames_for_frame_step_skips_first_and_last_five_before_direct_step(self):
        frame_paths = [Path(f"frame_{index:06d}.png") for index in range(20)]

        selected = select_frames_for_frame_step(
            frame_paths,
            frame_step=3,
        )

        self.assertEqual(
            [path.name for path in selected],
            [
                "frame_000005.png",
                "frame_000008.png",
                "frame_000011.png",
                "frame_000014.png",
            ],
        )

    def test_select_frames_for_frame_step_returns_empty_when_only_boundary_frames_remain(self):
        frame_paths = [Path(f"frame_{index:06d}.png") for index in range(10)]

        selected = select_frames_for_frame_step(
            frame_paths,
            frame_step=1,
        )

        self.assertEqual(selected, [])

    def test_run_coordinate_prompt_folders_samples_frames_and_writes_outputs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)
            output_root.mkdir(parents=True)
            for archive_name in ("frames.zip", "mask.zip", "masked_frame.zip"):
                (output_root / archive_name).write_text("stale", encoding="utf-8")

            for index in range(12):
                Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (coordinates_dir / f"frame_{index:06d}.json").write_text(
                    json.dumps([[40.0, 40.0], [60.0, 60.0]]),
                    encoding="utf-8",
                )

            predictor = FakePredictor()
            result = run_coordinate_prompt_folders(
                frames_dir=frames_dir,
                coordinates_dir=coordinates_dir,
                output_root=output_root,
                predictor=predictor,
                frame_step=3,
            )

            self.assertEqual(result["processed_frames"], 1)
            self.assertEqual(
                [path.name for path in result["frame_paths"]],
                [
                    "frame_000005.png",
                ],
            )
            self.assertFalse((output_root / "frames" / "frame_000000.png").exists())
            self.assertTrue((output_root / "frames" / "frame_000005.png").exists())
            self.assertTrue((output_root / "coordinates" / "frame_000005.json").exists())
            self.assertTrue((output_root / "mask" / "frame_000005.png").exists())
            self.assertTrue((output_root / "masked_frame" / "frame_000005.jpg").exists())
            self.assertIsNone(result["frames_zip"])
            self.assertIsNone(result["masks_zip"])
            self.assertIsNone(result["previews_zip"])
            self.assertFalse((output_root / "frames.zip").exists())
            self.assertFalse((output_root / "mask.zip").exists())
            self.assertFalse((output_root / "masked_frame.zip").exists())
            self.assertEqual(predictor.calls[0]["labels"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_iter_coordinate_prompt_folder_steps_yields_per_frame_updates(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)
            output_root.mkdir(parents=True)
            for archive_name in ("frames.zip", "mask.zip", "masked_frame.zip"):
                (output_root / archive_name).write_text("stale", encoding="utf-8")

            for index in range(12):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (coordinates_dir / f"frame_{index:06d}.json").write_text(
                    json.dumps([[1.0, 1.0], [2.0, 2.0]]),
                    encoding="utf-8",
                )

            updates = list(
                iter_coordinate_prompt_folder_steps(
                    frames_dir=frames_dir,
                    coordinates_dir=coordinates_dir,
                    output_root=output_root,
                    predictor=FakePredictor(),
                    frame_step=3,
                )
            )

            self.assertEqual([update["completed"] for update in updates], [0, 1, 1])
            self.assertEqual([update["total"] for update in updates], [1, 1, 1])
            self.assertEqual(updates[0]["stage"], "starting")
            self.assertEqual(updates[-1]["stage"], "done")
            self.assertIsNone(updates[-1]["result"]["frames_zip"])
            self.assertIsNone(updates[-1]["result"]["masks_zip"])
            self.assertIsNone(updates[-1]["result"]["previews_zip"])
            self.assertFalse((output_root / "frames.zip").exists())
            self.assertFalse((output_root / "mask.zip").exists())
            self.assertFalse((output_root / "masked_frame.zip").exists())

    def test_format_coordinate_progress_html_renders_visible_bar(self):
        html = format_coordinate_progress_html(
            completed=1,
            total=4,
            message="Processing frame_000006.png",
        )

        self.assertIn("coordinate-progress", html)
        self.assertIn("width: 25%", html)
        self.assertIn("1 / 4", html)
        self.assertIn("Processing frame_000006.png", html)


if __name__ == "__main__":
    unittest.main()
