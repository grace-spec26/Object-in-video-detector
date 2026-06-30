import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image, ImageDraw


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from export_yolo_segmentation_dataset import export_yolo_segmentation_dataset  # noqa: E402


class YoloSegmentationDatasetExportTest(unittest.TestCase):
    def _write_frame_and_mask(self, frames_dir, masks_dir, name, mask_regions):
        image = Image.new("RGB", (30, 20), (40, 80, 120))
        image.save(frames_dir / f"{name}.png")

        mask = Image.new("L", (30, 20), 0)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2 in mask_regions:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=1)
        mask.save(masks_dir / f"{name}.png")

    def test_exports_yolo_polygon_dataset_with_renamed_jpgs_and_split(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw-mask-data"
            frames_dir = raw_root / "frames"
            masks_dir = raw_root / "mask"
            dataset_dir = root / "dataset"
            frames_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            self._write_frame_and_mask(frames_dir, masks_dir, "frame_000005", [(2, 3, 8, 10)])
            self._write_frame_and_mask(frames_dir, masks_dir, "frame_000020", [(4, 5, 9, 12), (16, 6, 23, 14)])
            self._write_frame_and_mask(frames_dir, masks_dir, "frame_000030", [(1, 1, 2, 2)])
            self._write_frame_and_mask(frames_dir, masks_dir, "frame_000040", [(10, 2, 18, 9)])
            self._write_frame_and_mask(frames_dir, masks_dir, "frame_000050", [(5, 11, 14, 18)])

            stats = export_yolo_segmentation_dataset(
                raw_mask_root=raw_root,
                output_dir=dataset_dir,
                train_ratio=0.8,
                min_area_px=6,
                approx_epsilon=1.0,
            )

            self.assertEqual(stats.exported_images, 4)
            self.assertEqual(stats.skipped_masks, 1)
            self.assertEqual(stats.total_wound_instances, 5)
            self.assertEqual(stats.train_images, 3)
            self.assertEqual(stats.val_images, 1)

            self.assertTrue((dataset_dir / "train" / "images" / "img1.jpg").exists())
            self.assertTrue((dataset_dir / "train" / "labels" / "img1.txt").exists())
            self.assertTrue((dataset_dir / "train" / "images" / "img2.jpg").exists())
            self.assertTrue((dataset_dir / "train" / "labels" / "img2.txt").exists())
            self.assertFalse((dataset_dir / "train" / "images" / "frame_000005.png").exists())
            self.assertTrue((dataset_dir / "val" / "images" / "img4.jpg").exists())
            self.assertTrue((dataset_dir / "val" / "labels" / "img4.txt").exists())

            second_label_rows = (dataset_dir / "train" / "labels" / "img2.txt").read_text(
                encoding="utf-8"
            ).strip().splitlines()
            self.assertEqual(len(second_label_rows), 2)

            for label_path in sorted((dataset_dir / "train" / "labels").glob("*.txt")) + sorted(
                (dataset_dir / "val" / "labels").glob("*.txt")
            ):
                rows = [row for row in label_path.read_text(encoding="utf-8").splitlines() if row]
                self.assertGreaterEqual(len(rows), 1)
                for row in rows:
                    values = row.split()
                    self.assertEqual(values[0], "0")
                    coords = [float(value) for value in values[1:]]
                    self.assertGreaterEqual(len(coords), 6)
                    self.assertEqual(len(coords) % 2, 0)
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in coords))

            dataset_yaml = (dataset_dir / "dataset.yaml").read_text(encoding="utf-8")
            self.assertIn("path: dataset", dataset_yaml)
            self.assertIn("train: train/images", dataset_yaml)
            self.assertIn("val: val/images", dataset_yaml)
            self.assertIn("0: wound", dataset_yaml)


if __name__ == "__main__":
    unittest.main()
