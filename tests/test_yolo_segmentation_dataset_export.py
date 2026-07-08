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

    def _write_frame_and_mask_values(self, frames_dir, masks_dir, name, valued_regions):
        image = Image.new("RGB", (30, 20), (40, 80, 120))
        image.save(frames_dir / f"{name}.png")

        mask = Image.new("L", (30, 20), 0)
        draw = ImageDraw.Draw(mask)
        for x1, y1, x2, y2, value in valued_regions:
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=value)
        mask.save(masks_dir / f"{name}.png")

    def _write_existing_dataset_item(self, dataset_dir, split, stem):
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        Image.new("RGB", (30, 20), (10, 20, 30)).save(images_dir / f"{stem}.jpg")
        labels_dir.joinpath(f"{stem}.txt").write_text(
            "0 0.100000 0.100000 0.800000 0.100000 0.800000 0.800000\n",
            encoding="utf-8",
        )

    def _write_sample_raw_mask_data(self, raw_root):
        frames_dir = raw_root / "frames"
        masks_dir = raw_root / "mask"
        frames_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        self._write_frame_and_mask(frames_dir, masks_dir, "frame_000005", [(2, 3, 8, 10)])
        self._write_frame_and_mask(frames_dir, masks_dir, "frame_000020", [(4, 5, 9, 12), (16, 6, 23, 14)])
        self._write_frame_and_mask(frames_dir, masks_dir, "frame_000030", [(1, 1, 2, 2)])
        self._write_frame_and_mask(frames_dir, masks_dir, "frame_000040", [(10, 2, 18, 9)])
        self._write_frame_and_mask(frames_dir, masks_dir, "frame_000050", [(5, 11, 14, 18)])

    def test_exports_yolo_polygon_dataset_with_renamed_jpgs_and_split(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw-mask-data"
            dataset_dir = root / "dataset"
            self._write_sample_raw_mask_data(raw_root)

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

            self.assertTrue((dataset_dir / "images" / "train" / "train_img00001.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "train" / "train_img00001.txt").exists())
            self.assertTrue((dataset_dir / "images" / "train" / "train_img00002.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "train" / "train_img00002.txt").exists())
            self.assertFalse((dataset_dir / "images" / "train" / "frame_000005.png").exists())
            self.assertTrue((dataset_dir / "images" / "val" / "val_img00001.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "val" / "val_img00001.txt").exists())
            self.assertFalse((dataset_dir / "train").exists())
            self.assertFalse((dataset_dir / "val").exists())

            image_names = [
                path.name
                for path in sorted((dataset_dir / "images" / "train").glob("*.jpg"))
                + sorted((dataset_dir / "images" / "val").glob("*.jpg"))
            ]
            self.assertEqual(len(image_names), len(set(image_names)))

            second_label_rows = (dataset_dir / "labels" / "train" / "train_img00002.txt").read_text(
                encoding="utf-8"
            ).strip().splitlines()
            self.assertEqual(len(second_label_rows), 2)

            for label_path in sorted((dataset_dir / "labels" / "train").glob("*.txt")) + sorted(
                (dataset_dir / "labels" / "val").glob("*.txt")
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
            self.assertIn("train: images/train", dataset_yaml)
            self.assertIn("val: images/val", dataset_yaml)
            self.assertIn("0: wound", dataset_yaml)

    def test_appends_to_existing_dataset_with_next_split_specific_names(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw-mask-data"
            dataset_dir = root / "dataset"
            self._write_sample_raw_mask_data(raw_root)

            for index in range(1, 6):
                self._write_existing_dataset_item(dataset_dir, "train", f"train_img{index:05d}")
            for index in range(1, 4):
                self._write_existing_dataset_item(dataset_dir, "val", f"val_img{index:05d}")

            stats = export_yolo_segmentation_dataset(
                raw_mask_root=raw_root,
                output_dir=dataset_dir,
                train_ratio=0.8,
                min_area_px=6,
                approx_epsilon=1.0,
            )

            self.assertEqual(stats.exported_images, 4)
            self.assertEqual(stats.train_images, 3)
            self.assertEqual(stats.val_images, 1)

            self.assertTrue((dataset_dir / "images" / "train" / "train_img00001.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "train" / "train_img00001.txt").exists())
            self.assertTrue((dataset_dir / "images" / "train" / "train_img00006.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "train" / "train_img00006.txt").exists())
            self.assertTrue((dataset_dir / "images" / "train" / "train_img00008.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "train" / "train_img00008.txt").exists())
            self.assertTrue((dataset_dir / "images" / "val" / "val_img00003.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "val" / "val_img00003.txt").exists())
            self.assertTrue((dataset_dir / "images" / "val" / "val_img00004.jpg").exists())
            self.assertTrue((dataset_dir / "labels" / "val" / "val_img00004.txt").exists())

            train_images = sorted((dataset_dir / "images" / "train").glob("*.jpg"))
            val_images = sorted((dataset_dir / "images" / "val").glob("*.jpg"))
            self.assertEqual(len(train_images), 8)
            self.assertEqual(len(val_images), 4)
            image_names = [path.name for path in train_images + val_images]
            self.assertEqual(len(image_names), len(set(image_names)))

    def test_ignores_225_mask_pixels_when_converting_to_yolo_polygons(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root = root / "raw-mask-data"
            frames_dir = raw_root / "frames"
            masks_dir = raw_root / "mask"
            dataset_dir = root / "dataset"
            frames_dir.mkdir(parents=True)
            masks_dir.mkdir(parents=True)

            self._write_frame_and_mask_values(
                frames_dir,
                masks_dir,
                "frame_000001",
                [
                    (2, 3, 8, 10, 1),
                    (18, 2, 29, 18, 225),
                ],
            )
            self._write_frame_and_mask_values(
                frames_dir,
                masks_dir,
                "frame_000002",
                [
                    (3, 4, 10, 13, 1),
                    (16, 1, 27, 17, 225),
                ],
            )

            stats = export_yolo_segmentation_dataset(
                raw_mask_root=raw_root,
                output_dir=dataset_dir,
                train_ratio=0.5,
                min_area_px=6,
                approx_epsilon=1.0,
            )

            self.assertEqual(stats.exported_images, 2)
            label_row = (dataset_dir / "labels" / "train" / "train_img00001.txt").read_text(
                encoding="utf-8"
            ).strip()
            coords = [float(value) for value in label_row.split()[1:]]
            x_coords = coords[0::2]
            self.assertLess(max(x_coords), 0.5)


if __name__ == "__main__":
    unittest.main()
