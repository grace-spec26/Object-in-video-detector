import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mobilesam_coordinate_wrapper import (  # noqa: E402
    build_augmented_prompt_json,
    format_coordinate_progress_html,
    iter_coordinate_prompt_folder_steps,
    prepare_coordinate_prompt_json,
    run_mobilesam_for_frame,
    run_coordinate_prompt_folders,
    select_frames_for_target_fps,
)


class FakePredictor:
    def __init__(self):
        self.calls = []

    def set_image(self, image):
        self.image_shape = image.shape

    def predict(self, point_coords, point_labels, multimask_output):
        self.calls.append(
            {
                "coords": np.asarray(point_coords).tolist(),
                "labels": np.asarray(point_labels).tolist(),
                "multimask_output": multimask_output,
            }
        )
        mask = np.zeros(self.image_shape[:2], dtype=bool)
        mask[1:3, 1:3] = True
        return np.asarray([mask]), np.asarray([0.9], dtype=np.float32), None


class MobileSAMCoordinateWrapperTest(unittest.TestCase):
    def test_build_augmented_prompt_json_adds_expanded_corner_negatives(self):
        prompt = build_augmented_prompt_json(
            positive_points=[[10, 20], [30, 50]],
            image_width=100,
            image_height=80,
            padding_ratio=0.15,
        )

        obj = prompt["objects"][0]
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
        )

        self.assertEqual(
            prompt["objects"][0]["negative_points"],
            [[0.0, 0.0], [99.0, 0.0], [99.0, 79.0], [0.0, 79.0]],
        )

    def test_prepare_coordinate_prompt_json_does_not_overwrite_source(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "frame_000000.json"
            output_path = tmp_path / "raw-mask-data" / "coordinates" / "frame_000000.json"
            source_json = [[2.0, 3.0], [6.0, 9.0]]
            source_path.write_text(json.dumps(source_json), encoding="utf-8")

            prompt_objects = prepare_coordinate_prompt_json(
                source_path,
                image_width=20,
                image_height=20,
                output_path=output_path,
                padding_ratio=0.15,
            )

            self.assertEqual(json.loads(source_path.read_text(encoding="utf-8")), source_json)
            augmented = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(augmented["objects"][0]["positive_points"], source_json)
            self.assertEqual(augmented["objects"][0]["point_labels"], [1, 1, 0, 0, 0, 0])
            self.assertEqual(len(prompt_objects), 1)

    def test_prepare_coordinate_prompt_json_generates_negatives_from_source_positives(self):
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
            )

            augmented = json.loads(output_path.read_text(encoding="utf-8"))
            obj = augmented["objects"][0]
            np.testing.assert_allclose(
                obj["point_coords"],
                [
                    [1.0, 1.0],
                    [3.0, 3.0],
                    [0.7, 0.7],
                    [3.3, 0.7],
                    [3.3, 3.3],
                    [0.7, 3.3],
                ],
            )
            self.assertEqual(obj["point_labels"], [1, 1, 0, 0, 0, 0])
            self.assertEqual(obj["positive_points"], [[1.0, 1.0], [3.0, 3.0]])
            np.testing.assert_allclose(
                obj["negative_points"],
                [[0.7, 0.7], [3.3, 0.7], [3.3, 3.3], [0.7, 3.3]],
            )
            self.assertEqual(prompt_objects[0][1].tolist(), [1, 1, 0, 0, 0, 0])

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
            )

            np.testing.assert_allclose(
                predictor.calls[0]["coords"],
                [[1.0, 1.0], [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
            )
            self.assertEqual(predictor.calls[0]["labels"], [1, 0, 0, 0, 0])
            self.assertTrue(mask_path.exists())
            self.assertTrue(preview_path.exists())

    def test_select_frames_for_target_fps_uses_source_fps_stride(self):
        frame_paths = [Path(f"frame_{index:06d}.png") for index in range(13)]

        selected = select_frames_for_target_fps(
            frame_paths,
            target_fps=5,
            source_fps=30,
        )

        self.assertEqual(
            [path.name for path in selected],
            ["frame_000000.png", "frame_000006.png", "frame_000012.png"],
        )

    def test_run_coordinate_prompt_folders_samples_frames_and_writes_outputs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)

            for index in range(12):
                Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (coordinates_dir / f"frame_{index:06d}.json").write_text(
                    json.dumps([[1.0, 1.0], [2.0, 2.0]]),
                    encoding="utf-8",
                )

            predictor = FakePredictor()
            result = run_coordinate_prompt_folders(
                frames_dir=frames_dir,
                coordinates_dir=coordinates_dir,
                output_root=output_root,
                predictor=predictor,
                target_fps=5,
                source_fps=30,
            )

            self.assertEqual(result["processed_frames"], 2)
            self.assertEqual(
                [path.name for path in result["frame_paths"]],
                ["frame_000000.png", "frame_000006.png"],
            )
            self.assertTrue((output_root / "frames" / "frame_000000.png").exists())
            self.assertTrue((output_root / "coordinates" / "frame_000000.json").exists())
            self.assertTrue((output_root / "mask" / "frame_000000.png").exists())
            self.assertTrue((output_root / "masked_frame" / "frame_000000.jpg").exists())
            self.assertTrue(result["frames_zip"].exists())
            self.assertTrue(result["masks_zip"].exists())
            self.assertTrue(result["previews_zip"].exists())
            self.assertEqual(predictor.calls[0]["labels"], [1, 1, 0, 0, 0, 0])

    def test_iter_coordinate_prompt_folder_steps_yields_per_frame_updates(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)

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
                    target_fps=5,
                    source_fps=30,
                )
            )

            self.assertEqual([update["completed"] for update in updates], [0, 1, 2, 2])
            self.assertEqual([update["total"] for update in updates], [2, 2, 2, 2])
            self.assertEqual(updates[0]["stage"], "starting")
            self.assertEqual(updates[-1]["stage"], "done")
            self.assertTrue(updates[-1]["result"]["frames_zip"].exists())
            self.assertTrue(updates[-1]["result"]["masks_zip"].exists())
            self.assertTrue(updates[-1]["result"]["previews_zip"].exists())

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
