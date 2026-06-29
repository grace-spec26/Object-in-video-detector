import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam2_coordinate_wrapper import (  # noqa: E402
    DEFAULT_SAM2_MODEL,
    SAM2_MODEL_CHOICES,
    iter_sam2_coordinate_prompt_folder_steps,
    load_sam2_prompt_objects,
    resolve_sam2_model_option,
    run_sam2_for_frame,
    select_frames_for_frame_step,
)


class FakeSAM2Predictor:
    def __init__(self):
        self.calls = []

    def set_image(self, image):
        self.image_shape = image.shape

    def predict(
        self,
        point_coords,
        point_labels,
        box,
        multimask_output,
        normalize_coords,
    ):
        self.calls.append(
            {
                "coords": np.asarray(point_coords).tolist(),
                "labels": np.asarray(point_labels).tolist(),
                "box": None if box is None else np.asarray(box).tolist(),
                "multimask_output": multimask_output,
                "normalize_coords": normalize_coords,
            }
        )
        mask = np.zeros(self.image_shape[:2], dtype=bool)
        mask[1:3, 1:3] = True
        return np.asarray([mask]), np.asarray([0.95], dtype=np.float32), None


class SAM2CoordinateWrapperTest(unittest.TestCase):
    def test_sam2_model_options_include_all_hiera_21_checkpoints(self):
        self.assertEqual(
            SAM2_MODEL_CHOICES,
            (
                "sam2.1_hiera_tiny.pt",
                "sam2.1_hiera_small.pt",
                "sam2.1_hiera_base_plus.pt",
                "sam2.1_hiera_large.pt",
            ),
        )
        self.assertEqual(DEFAULT_SAM2_MODEL, "sam2.1_hiera_small.pt")

        option = resolve_sam2_model_option("sam2.1_hiera_base_plus.pt")

        self.assertEqual(option["checkpoint_name"], "sam2.1_hiera_base_plus.pt")
        self.assertEqual(option["config"], "configs/sam2.1/sam2.1_hiera_b+.yaml")
        self.assertTrue(str(option["checkpoint"]).endswith("sam2/checkpoints/sam2.1_hiera_base_plus.pt"))
        self.assertIn("sam2.1_hiera_base_plus.pt", option["url"])

    def test_sam2_model_option_rejects_unknown_model(self):
        with self.assertRaisesRegex(ValueError, "sam2.1_hiera_small.pt"):
            resolve_sam2_model_option("bad-model.pt")

    def test_load_sam2_prompt_objects_uses_augmented_json_directly(self):
        with TemporaryDirectory() as tmp:
            coordinate_path = Path(tmp) / "frame_000000.json"
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

            prompt_objects, prompt_json = load_sam2_prompt_objects(
                coordinate_path,
                image_width=8,
                image_height=8,
            )

            coords, labels, class_id, prompt_box = prompt_objects[0]
            self.assertEqual(class_id, 1)
            self.assertEqual(coords.tolist(), [[1.0, 1.0], [4.0, 4.0]])
            self.assertEqual(labels.tolist(), [1, 0])
            self.assertIsNone(prompt_box)
            self.assertEqual(prompt_json["engine"], "sam2")

    def test_run_sam2_for_frame_writes_mask_and_preview(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frame_path = tmp_path / "frame_000000.png"
            coordinate_path = tmp_path / "frame_000000.json"
            mask_path = tmp_path / "mask" / "frame_000000.png"
            preview_path = tmp_path / "masked_frame" / "frame_000000.jpg"

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

            predictor = FakeSAM2Predictor()
            run_sam2_for_frame(
                predictor=predictor,
                frame_path=frame_path,
                coordinate_json_path=coordinate_path,
                mask_path=mask_path,
                preview_path=preview_path,
            )

            self.assertEqual(predictor.calls[0]["labels"], [1, 0])
            self.assertIsNone(predictor.calls[0]["box"])
            self.assertTrue(predictor.calls[0]["normalize_coords"])
            self.assertTrue(mask_path.exists())
            self.assertTrue(preview_path.exists())

    def test_iter_sam2_coordinate_prompt_folder_steps_does_not_clear_inputs(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "raw-mask-data" / "frames"
            coordinates_dir = tmp_path / "raw-mask-data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)

            for index in range(12):
                Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (coordinates_dir / f"frame_{index:06d}.json").write_text(
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

            updates = list(
                iter_sam2_coordinate_prompt_folder_steps(
                    frames_dir=frames_dir,
                    coordinates_dir=coordinates_dir,
                    output_root=output_root,
                    predictor=FakeSAM2Predictor(),
                    target_fps=30,
                    source_fps=30,
                )
            )

            self.assertEqual(updates[-1]["stage"], "done")
            self.assertTrue((frames_dir / "frame_000000.png").exists())
            self.assertTrue((coordinates_dir / "frame_000000.json").exists())
            self.assertTrue((output_root / "mask" / "frame_000005.png").exists())
            self.assertTrue((output_root / "masked_frames" / "frame_000005.jpg").exists())

    def test_iter_sam2_coordinate_prompt_folder_steps_writes_processed_coordinates(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            source_coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            output_frames_dir = output_root / "frames"
            processed_coordinates_dir = output_root / "coordinates"
            frames_dir.mkdir(parents=True)
            source_coordinates_dir.mkdir(parents=True)
            output_frames_dir.mkdir(parents=True)
            processed_coordinates_dir.mkdir(parents=True)
            stale_frame_path = output_frames_dir / "stale.png"
            stale_frame_path.write_bytes(b"stale")
            stale_coordinate_path = processed_coordinates_dir / "stale.json"
            stale_coordinate_path.write_text("{}", encoding="utf-8")

            coordinate_payload = {
                "objects": [
                    {
                        "class_id": 1,
                        "point_coords": [[40.0, 40.0], [60.0, 60.0], [4.0, 4.0]],
                        "point_labels": [1, 1, 0],
                    }
                ]
            }
            for index in range(12):
                Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (source_coordinates_dir / f"frame_{index:06d}.json").write_text(
                    json.dumps(coordinate_payload),
                    encoding="utf-8",
                )

            predictor = FakeSAM2Predictor()
            updates = list(
                iter_sam2_coordinate_prompt_folder_steps(
                    frames_dir=frames_dir,
                    coordinates_dir=source_coordinates_dir,
                    output_root=output_root,
                    predictor=predictor,
                    target_fps=30,
                    source_fps=30,
                )
            )

            result = updates[-1]["result"]
            output_frame_path = output_frames_dir / "frame_000005.png"
            processed_coordinate_path = processed_coordinates_dir / "frame_000005.json"
            self.assertEqual(result["source_frames_dir"], frames_dir)
            self.assertEqual(result["frames_dir"], output_frames_dir)
            self.assertEqual(result["source_coordinates_dir"], source_coordinates_dir)
            self.assertEqual(result["coordinates_dir"], processed_coordinates_dir)
            self.assertTrue(output_frame_path.exists())
            self.assertFalse(stale_frame_path.exists())
            self.assertTrue(processed_coordinate_path.exists())
            self.assertFalse(stale_coordinate_path.exists())

            processed_json = json.loads(processed_coordinate_path.read_text(encoding="utf-8"))
            self.assertEqual(
                processed_json["source_coordinate_json"],
                str(source_coordinates_dir / "frame_000005.json"),
            )
            self.assertEqual(
                processed_json["objects"][0]["positive_points"],
                [[40.0, 40.0], [60.0, 60.0]],
            )
            self.assertEqual(
                processed_json["objects"][0]["negative_points"],
                [
                    [20.0, 20.0],
                    [80.0, 20.0],
                    [80.0, 80.0],
                    [20.0, 80.0],
                    [50.0, 20.0],
                    [80.0, 50.0],
                    [50.0, 80.0],
                    [20.0, 50.0],
                ],
            )
            self.assertEqual(
                processed_json["objects"][0]["point_labels"],
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            self.assertEqual(processed_json["negative_mode"], "box_8_points")
            self.assertEqual(predictor.calls[0]["labels"], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            self.assertEqual(predictor.calls[0]["box"], [20.0, 20.0, 80.0, 80.0])

    def test_select_frames_for_frame_step_skips_first_and_last_five_before_direct_step(self):
        frame_paths = [Path(f"frame_{index:06d}.png") for index in range(20)]

        selected = select_frames_for_frame_step(frame_paths, frame_step=3)

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

        selected = select_frames_for_frame_step(frame_paths, frame_step=1)

        self.assertEqual(selected, [])

    def test_iter_sam2_coordinate_prompt_folder_steps_respects_frame_step(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
            output_root = tmp_path / "raw-mask-data"
            frames_dir.mkdir(parents=True)
            coordinates_dir.mkdir(parents=True)
            output_root.mkdir(parents=True)
            for archive_name in ("frames.zip", "mask.zip", "masked_frames.zip"):
                (output_root / archive_name).write_text("stale", encoding="utf-8")

            for index in range(12):
                Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(
                    frames_dir / f"frame_{index:06d}.png"
                )
                (coordinates_dir / f"frame_{index:06d}.json").write_text(
                    json.dumps(
                        {
                            "objects": [
                                {
                                    "class_id": 1,
                                    "point_coords": [[1.0, 1.0], [3.0, 3.0]],
                                    "point_labels": [1, 1],
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )

            updates = list(
                iter_sam2_coordinate_prompt_folder_steps(
                    frames_dir=frames_dir,
                    coordinates_dir=coordinates_dir,
                    output_root=output_root,
                    predictor=FakeSAM2Predictor(),
                    frame_step=3,
                )
            )

            stages = [update["stage"] for update in updates]
            result = updates[-1]["result"]
            self.assertEqual(stages[:3], ["scanned", "prepared-output", "starting"])
            self.assertEqual(result["processed_frames"], 1)
            self.assertFalse((output_root / "frames" / "frame_000000.png").exists())
            self.assertTrue((output_root / "frames" / "frame_000005.png").exists())
            self.assertFalse((output_root / "frames" / "frame_000003.png").exists())
            self.assertFalse((output_root / "frames" / "frame_000006.png").exists())
            self.assertFalse((output_root / "frames" / "frame_000009.png").exists())
            self.assertFalse((output_root / "frames" / "frame_000001.png").exists())
            self.assertTrue((output_root / "coordinates" / "frame_000005.json").exists())
            self.assertIsNone(result["frames_zip"])
            self.assertIsNone(result["masks_zip"])
            self.assertIsNone(result["previews_zip"])
            self.assertFalse((output_root / "frames.zip").exists())
            self.assertFalse((output_root / "mask.zip").exists())
            self.assertFalse((output_root / "masked_frames.zip").exists())


if __name__ == "__main__":
    unittest.main()
