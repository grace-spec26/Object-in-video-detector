import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam2_coordinate_wrapper import (  # noqa: E402
    iter_sam2_coordinate_prompt_folder_steps,
    load_sam2_prompt_objects,
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
        multimask_output,
        normalize_coords,
    ):
        self.calls.append(
            {
                "coords": np.asarray(point_coords).tolist(),
                "labels": np.asarray(point_labels).tolist(),
                "multimask_output": multimask_output,
                "normalize_coords": normalize_coords,
            }
        )
        mask = np.zeros(self.image_shape[:2], dtype=bool)
        mask[1:3, 1:3] = True
        return np.asarray([mask]), np.asarray([0.95], dtype=np.float32), None


class SAM2CoordinateWrapperTest(unittest.TestCase):
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

            coords, labels, class_id = prompt_objects[0]
            self.assertEqual(class_id, 1)
            self.assertEqual(coords.tolist(), [[1.0, 1.0], [4.0, 4.0]])
            self.assertEqual(labels.tolist(), [1, 0])
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

            for index in range(2):
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
            self.assertTrue((output_root / "mask" / "frame_000000.png").exists())
            self.assertTrue((output_root / "masked_frames" / "frame_000000.jpg").exists())

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

            Image.fromarray(np.zeros((6, 6, 3), dtype=np.uint8)).save(
                frames_dir / "frame_000000.png"
            )
            (source_coordinates_dir / "frame_000000.json").write_text(
                json.dumps(
                        {
                            "objects": [
                                {
                                    "class_id": 1,
                                    "point_coords": [[1.0, 1.0], [3.0, 3.0], [4.0, 4.0]],
                                    "point_labels": [1, 1, 0],
                                }
                            ]
                        }
                ),
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
            output_frame_path = output_frames_dir / "frame_000000.png"
            processed_coordinate_path = processed_coordinates_dir / "frame_000000.json"
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
                str(source_coordinates_dir / "frame_000000.json"),
            )
            self.assertEqual(
                processed_json["objects"][0]["positive_points"],
                [[1.0, 1.0], [3.0, 3.0]],
            )
            self.assertEqual(
                processed_json["objects"][0]["point_labels"],
                [1, 1, 0, 0, 0, 0],
            )
            self.assertEqual(predictor.calls[0]["labels"], [1, 1, 0, 0, 0, 0])

    def test_select_frames_for_frame_step_uses_direct_user_step(self):
        frame_paths = [Path(f"frame_{index:06d}.png") for index in range(13)]

        selected = select_frames_for_frame_step(frame_paths, frame_step=3)

        self.assertEqual(
            [path.name for path in selected],
            [
                "frame_000000.png",
                "frame_000003.png",
                "frame_000006.png",
                "frame_000009.png",
                "frame_000012.png",
            ],
        )

    def test_iter_sam2_coordinate_prompt_folder_steps_respects_frame_step(self):
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frames_dir = tmp_path / "data" / "frames"
            coordinates_dir = tmp_path / "data" / "coordinates"
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
            self.assertEqual(result["processed_frames"], 4)
            self.assertTrue((output_root / "frames" / "frame_000000.png").exists())
            self.assertTrue((output_root / "frames" / "frame_000003.png").exists())
            self.assertTrue((output_root / "frames" / "frame_000006.png").exists())
            self.assertTrue((output_root / "frames" / "frame_000009.png").exists())
            self.assertFalse((output_root / "frames" / "frame_000001.png").exists())
            self.assertTrue((output_root / "coordinates" / "frame_000009.json").exists())


if __name__ == "__main__":
    unittest.main()
