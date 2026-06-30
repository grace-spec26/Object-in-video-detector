#!/usr/bin/env python3
"""Export MobileSAM wound masks as Ultralytics YOLO segmentation labels.

Final YOLO labels use one polygon row per wound instance:
class_index x1 y1 x2 y2 ... xn yn

Coordinates are normalized to 0-1 in image (x, y) order. Class 0 is wound.
PNG masks remain intermediate data only; final training labels are .txt files.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image


Point = Tuple[float, float]
Pixel = Tuple[int, int]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ExportStats:
    exported_images: int
    labels: int
    skipped_masks: int
    total_wound_instances: int
    train_images: int
    val_images: int


@dataclass(frozen=True)
class ExportRecord:
    source_frame: Path
    output_name: str
    width: int
    height: int
    polygons: List[List[Point]]


def export_yolo_segmentation_dataset(
    raw_mask_root: Path | str = "raw-mask-data",
    output_dir: Path | str = "dataset",
    frames_dir: Optional[Path | str] = None,
    masks_dir: Optional[Path | str] = None,
    train_ratio: float = 0.8,
    min_area_px: int = 20,
    approx_epsilon: float = 2.0,
    wound_value: int = 1,
    image_quality: int = 95,
) -> ExportStats:
    """Convert raw-mask-data frame/mask pairs into YOLO polygon dataset files."""

    raw_root = Path(raw_mask_root)
    frame_root = Path(frames_dir) if frames_dir else raw_root / "frames"
    mask_root = Path(masks_dir) if masks_dir else _default_masks_dir(raw_root)
    dataset_root = Path(output_dir)

    _validate_inputs(frame_root, mask_root, train_ratio, min_area_px, approx_epsilon)

    records: List[ExportRecord] = []
    skipped_masks = 0
    output_index = 1

    for frame_path in _iter_frame_paths(frame_root):
        mask_path = mask_root / f"{frame_path.stem}.png"
        if not mask_path.exists():
            skipped_masks += 1
            print(f"[skip] Missing mask for frame: {frame_path.name}")
            continue

        polygons, width, height = _polygons_from_mask(
            mask_path=mask_path,
            wound_value=wound_value,
            min_area_px=min_area_px,
            approx_epsilon=approx_epsilon,
        )
        if not polygons:
            skipped_masks += 1
            continue

        records.append(
            ExportRecord(
                source_frame=frame_path,
                output_name=f"img{output_index}",
                width=width,
                height=height,
                polygons=polygons,
            )
        )
        output_index += 1

    if len(records) < 2:
        raise ValueError(
            "At least two valid frame/mask pairs are required so train and val folders are non-empty."
        )

    train_count = _split_count(len(records), train_ratio)
    train_records = records[:train_count]
    val_records = records[train_count:]

    _prepare_dataset_dir(dataset_root)
    _write_records(dataset_root, "train", train_records, image_quality=image_quality)
    _write_records(dataset_root, "val", val_records, image_quality=image_quality)
    _write_dataset_yaml(dataset_root)

    validate_yolo_segmentation_dataset(dataset_root)

    stats = ExportStats(
        exported_images=len(records),
        labels=len(records),
        skipped_masks=skipped_masks,
        total_wound_instances=sum(len(record.polygons) for record in records),
        train_images=len(train_records),
        val_images=len(val_records),
    )
    _print_stats(stats)
    return stats


def validate_yolo_segmentation_dataset(dataset_dir: Path | str) -> None:
    """Validate the Ultralytics segmentation dataset structure and label rows."""

    dataset_root = Path(dataset_dir)
    for split in ("train", "val"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"
        image_paths = sorted(images_dir.glob("*.jpg"))
        label_paths = sorted(labels_dir.glob("*.txt"))

        if not image_paths:
            raise ValueError(f"{split} image folder is empty: {images_dir}")
        if not label_paths:
            raise ValueError(f"{split} label folder is empty: {labels_dir}")

        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        missing_labels = sorted(image_stems - label_stems)
        extra_labels = sorted(label_stems - image_stems)
        if missing_labels:
            raise ValueError(f"{split} images missing label files: {missing_labels}")
        if extra_labels:
            raise ValueError(f"{split} labels missing images: {extra_labels}")

        for label_path in label_paths:
            rows = [row.strip() for row in label_path.read_text(encoding="utf-8").splitlines()]
            rows = [row for row in rows if row]
            if not rows:
                raise ValueError(f"Label file has no wound polygons: {label_path}")

            for row in rows:
                values = row.split()
                if values[0] != "0":
                    raise ValueError(f"Invalid class index in {label_path}: {values[0]}")
                if len(values) < 7 or len(values[1:]) % 2:
                    raise ValueError(f"Polygon must contain at least 3 x/y points: {label_path}")
                coords = [float(value) for value in values[1:]]
                if any(value < 0.0 or value > 1.0 for value in coords):
                    raise ValueError(f"Polygon coordinate outside 0-1 range: {label_path}")


def _default_masks_dir(raw_root: Path) -> Path:
    mask_dir = raw_root / "mask"
    if mask_dir.exists():
        return mask_dir
    return raw_root / "masks"


def _validate_inputs(
    frames_dir: Path,
    masks_dir: Path,
    train_ratio: float,
    min_area_px: int,
    approx_epsilon: float,
) -> None:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frame folder does not exist: {frames_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Mask folder does not exist: {masks_dir}")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if min_area_px < 1:
        raise ValueError("--min-area-px must be at least 1")
    if approx_epsilon < 0:
        raise ValueError("--approx-epsilon must be non-negative")


def _iter_frame_paths(frames_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _polygons_from_mask(
    mask_path: Path,
    wound_value: int,
    min_area_px: int,
    approx_epsilon: float,
) -> Tuple[List[List[Point]], int, int]:
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
        width, height = mask.size
        wound_pixels = _wound_pixels(mask, wound_value=wound_value)

    polygons: List[List[Point]] = []
    for component in _connected_components(wound_pixels):
        if len(component) < min_area_px:
            continue

        loops = _component_contour_loops(component)
        if not loops:
            continue

        contour = max(loops, key=lambda loop: abs(_signed_area(loop)))
        if len(contour) < 3 or abs(_signed_area(contour)) < float(min_area_px):
            continue

        simplified = _simplify_closed_polygon(contour, approx_epsilon)
        if len(simplified) < 3:
            simplified = _convex_hull(contour)
        if len(simplified) < 3:
            continue

        normalized = [
            (_clamp(x / width, 0.0, 1.0), _clamp(y / height, 0.0, 1.0))
            for x, y in simplified
        ]
        if len(normalized) >= 3:
            polygons.append(normalized)

    return polygons, width, height


def _wound_pixels(mask: Image.Image, wound_value: int) -> Set[Pixel]:
    width, height = mask.size
    data = mask.tobytes()
    return {
        (index % width, index // width)
        for index, value in enumerate(data)
        if value == wound_value and index // width < height
    }


def _connected_components(pixels: Set[Pixel]) -> Iterable[Set[Pixel]]:
    remaining = set(pixels)
    neighbors = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )

    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            x, y = stack.pop()
            for dx, dy in neighbors:
                neighbor = (x + dx, y + dy)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        yield component


def _component_contour_loops(component: Set[Pixel]) -> List[List[Point]]:
    # Trace the outer pixel-edge contours of a connected component.
    edges: Dict[Point, List[Point]] = {}

    def add_edge(start: Point, end: Point) -> None:
        edges.setdefault(start, []).append(end)

    for x, y in component:
        if (x, y - 1) not in component:
            add_edge((float(x), float(y)), (float(x + 1), float(y)))
        if (x + 1, y) not in component:
            add_edge((float(x + 1), float(y)), (float(x + 1), float(y + 1)))
        if (x, y + 1) not in component:
            add_edge((float(x + 1), float(y + 1)), (float(x), float(y + 1)))
        if (x - 1, y) not in component:
            add_edge((float(x), float(y + 1)), (float(x), float(y)))

    loops: List[List[Point]] = []
    while edges:
        start = next(iter(edges))
        current = start
        loop = [start]
        while True:
            destinations = edges.get(current)
            if not destinations:
                break
            next_point = destinations.pop()
            if not destinations:
                del edges[current]
            current = next_point
            if current == start:
                break
            loop.append(current)

        if len(loop) >= 3:
            loops.append(loop)

    return loops


def _simplify_closed_polygon(points: Sequence[Point], epsilon: float) -> List[Point]:
    unique_points = _remove_duplicate_neighbors(points)
    if epsilon <= 0 or len(unique_points) <= 3:
        return unique_points

    closed = list(unique_points) + [unique_points[0]]
    simplified = _rdp(closed, epsilon)
    if simplified and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]
    return _remove_duplicate_neighbors(simplified)


def _remove_duplicate_neighbors(points: Sequence[Point]) -> List[Point]:
    cleaned: List[Point] = []
    for point in points:
        if not cleaned or point != cleaned[-1]:
            cleaned.append(point)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _rdp(points: Sequence[Point], epsilon: float) -> List[Point]:
    if len(points) < 3:
        return list(points)

    first = points[0]
    last = points[-1]
    max_distance = -1.0
    max_index = 0
    for index in range(1, len(points) - 1):
        distance = _distance_to_segment(points[index], first, last)
        if distance > max_distance:
            max_distance = distance
            max_index = index

    if max_distance > epsilon:
        left = _rdp(points[: max_index + 1], epsilon)
        right = _rdp(points[max_index:], epsilon)
        return left[:-1] + right
    return [first, last]


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)

    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    t = _clamp(t, 0.0, 1.0)
    nearest_x = sx + t * dx
    nearest_y = sy + t * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def _convex_hull(points: Sequence[Point]) -> List[Point]:
    sorted_points = sorted(set(points))
    if len(sorted_points) <= 1:
        return sorted_points

    def cross(origin: Point, a: Point, b: Point) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower: List[Point] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: List[Point] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _signed_area(points: Sequence[Point]) -> float:
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def _split_count(total: int, train_ratio: float) -> int:
    train_count = int(total * train_ratio)
    return min(max(train_count, 1), total - 1)


def _prepare_dataset_dir(dataset_root: Path) -> None:
    for split in ("train", "val"):
        split_dir = dataset_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        (split_dir / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "labels").mkdir(parents=True, exist_ok=True)


def _write_records(
    dataset_root: Path,
    split: str,
    records: Sequence[ExportRecord],
    image_quality: int,
) -> None:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"

    for record in records:
        image_path = images_dir / f"{record.output_name}.jpg"
        label_path = labels_dir / f"{record.output_name}.txt"
        with Image.open(record.source_frame) as image:
            image.convert("RGB").save(image_path, format="JPEG", quality=image_quality)

        rows = []
        for polygon in record.polygons:
            coords = " ".join(f"{value:.6f}" for point in polygon for value in point)
            rows.append(f"0 {coords}")
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_dataset_yaml(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"path: {dataset_root.name}\n"
        "train: train/images\n"
        "val: val/images\n"
        "\n"
        "names:\n"
        "  0: wound\n"
    )
    (dataset_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def _print_stats(stats: ExportStats) -> None:
    print(f"exported_images={stats.exported_images}")
    print(f"labels={stats.labels}")
    print(f"skipped_masks={stats.skipped_masks}")
    print(f"total_wound_instances={stats.total_wound_instances}")
    print(f"train_images={stats.train_images}")
    print(f"val_images={stats.val_images}")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert raw-mask-data MobileSAM masks to Ultralytics YOLO polygon labels."
    )
    parser.add_argument("--raw-mask-root", default="raw-mask-data", help="Root containing frames/ and mask(s)/")
    parser.add_argument("--frames-dir", default=None, help="Override input frame folder")
    parser.add_argument("--masks-dir", default=None, help="Override input PNG mask folder")
    parser.add_argument("--output-dir", default="dataset", help="Output Ultralytics dataset folder")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio, default 0.8")
    parser.add_argument("--min-area-px", type=int, default=20, help="Minimum wound component area in pixels")
    parser.add_argument(
        "--approx-epsilon",
        type=float,
        default=2.0,
        help="Polygon simplification distance in pixels, similar to approxPolyDP epsilon",
    )
    parser.add_argument("--wound-value", type=int, default=1, help="Mask pixel value treated as wound")
    parser.add_argument("--image-quality", type=int, default=95, help="JPEG quality for exported images")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_yolo_segmentation_dataset(
        raw_mask_root=args.raw_mask_root,
        output_dir=args.output_dir,
        frames_dir=args.frames_dir,
        masks_dir=args.masks_dir,
        train_ratio=args.train_ratio,
        min_area_px=args.min_area_px,
        approx_epsilon=args.approx_epsilon,
        wound_value=args.wound_value,
        image_quality=args.image_quality,
    )


if __name__ == "__main__":
    main()
