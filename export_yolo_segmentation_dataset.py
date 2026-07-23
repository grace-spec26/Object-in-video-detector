#!/usr/bin/env python3
"""Export MobileSAM wound masks as Ultralytics YOLO segmentation labels.

Final YOLO labels use one polygon row per wound instance:
class_index x1 y1 x2 y2 ... xn yn

Coordinates are normalized to 0-1 in image (x, y) order. Class 0 is wound.
PNG masks remain intermediate data only; final training labels are .txt files.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image


Point = Tuple[float, float]
Pixel = Tuple[int, int]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_DATASET_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_DATASET_THREAD_LOCKS_GUARD = threading.Lock()


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
    width: int
    height: int
    polygons: List[List[Point]]


@dataclass(frozen=True)
class SingleFrameExportResult:
    image_path: Path
    label_path: Path
    split: str
    total_wound_instances: int


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
    overwrite: bool = False,
) -> ExportStats:
    """Convert raw-mask-data frame/mask pairs into YOLO polygon dataset files."""

    raw_root = Path(raw_mask_root)
    frame_root = Path(frames_dir) if frames_dir else raw_root / "frames"
    mask_root = Path(masks_dir) if masks_dir else _default_masks_dir(raw_root)
    dataset_root = Path(output_dir)

    _validate_inputs(frame_root, mask_root, train_ratio, min_area_px, approx_epsilon)

    records: List[ExportRecord] = []
    skipped_masks = 0

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
                width=width,
                height=height,
                polygons=polygons,
            )
        )

    if len(records) < 2:
        raise ValueError(
            "At least two valid frame/mask pairs are required so train and val folders are non-empty."
        )

    train_count = _split_count(len(records), train_ratio)
    train_records = records[:train_count]
    val_records = records[train_count:]

    with _dataset_export_lock(dataset_root):
        _prepare_dataset_dir(dataset_root, overwrite=overwrite)
        train_start_index = _next_split_image_index(dataset_root, "train")
        val_start_index = _next_split_image_index(dataset_root, "val")

        _write_records(
            dataset_root,
            "train",
            train_records,
            image_quality=image_quality,
            start_index=train_start_index,
        )
        _write_records(
            dataset_root,
            "val",
            val_records,
            image_quality=image_quality,
            start_index=val_start_index,
        )
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


def export_single_mask_frame_to_yolo_split(
    frame: object,
    mask: object,
    output_dir: Path | str = "dataset",
    split: str = "train",
    min_area_px: int = 20,
    approx_epsilon: float = 2.0,
    image_quality: int = 95,
) -> SingleFrameExportResult:
    """Append one frame and its binary mask as a YOLO segmentation item."""

    split_name = str(split)
    if split_name not in {"train", "val"}:
        raise ValueError("split must be 'train' or 'val'.")
    if min_area_px < 1:
        raise ValueError("min_area_px must be at least 1.")
    if approx_epsilon < 0:
        raise ValueError("approx_epsilon must be non-negative.")
    if not 1 <= image_quality <= 100:
        raise ValueError("image_quality must be between 1 and 100.")

    frame_array = _as_export_frame_array(frame)
    polygons, width, height = _polygons_from_mask_array(
        mask,
        min_area_px=min_area_px,
        approx_epsilon=approx_epsilon,
    )
    if frame_array.shape[0] != height or frame_array.shape[1] != width:
        raise ValueError("Frame and mask dimensions must match.")
    if not polygons:
        raise ValueError("SAM mask did not contain any valid YOLO polygons.")

    dataset_root = Path(output_dir).expanduser()
    created_paths: List[Path] = []
    with _dataset_export_lock(dataset_root):
        _prepare_dataset_dir(dataset_root, overwrite=False)
        images_dir = _split_images_dir(dataset_root, split_name)
        labels_dir = _split_labels_dir(dataset_root, split_name)
        next_index = _next_split_image_index(dataset_root, split_name)

        output_name = f"{split_name}_img{next_index:05d}"
        image_path = images_dir / f"{output_name}.jpg"
        label_path = labels_dir / f"{output_name}.txt"
        while image_path.exists() or label_path.exists():
            next_index += 1
            output_name = f"{split_name}_img{next_index:05d}"
            image_path = images_dir / f"{output_name}.jpg"
            label_path = labels_dir / f"{output_name}.txt"

        try:
            Image.fromarray(frame_array).convert("RGB").save(
                image_path,
                format="JPEG",
                quality=image_quality,
            )
            created_paths.append(image_path)
            rows = []
            for polygon in polygons:
                coords = " ".join(f"{value:.6f}" for point in polygon for value in point)
                rows.append(f"0 {coords}")
            label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            created_paths.append(label_path)
            _write_dataset_yaml(dataset_root)
        except Exception:
            for created_path in reversed(created_paths):
                created_path.unlink(missing_ok=True)
            raise

    return SingleFrameExportResult(
        image_path=image_path,
        label_path=label_path,
        split=split_name,
        total_wound_instances=len(polygons),
    )


def export_no_wound_frames_to_yolo_dataset(
    frames: Sequence[object],
    output_dir: Path | str = "dataset",
    train_ratio: float = 0.8,
    image_quality: int = 95,
) -> ExportStats:
    """Append clean frames as YOLO negative samples with empty label files."""

    frame_list = list(frames)
    if len(frame_list) < 2:
        raise ValueError(
            "At least two frames are required so train and val folders are non-empty."
        )
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 1 <= image_quality <= 100:
        raise ValueError("image_quality must be between 1 and 100.")

    frame_arrays = [_as_export_frame_array(frame) for frame in frame_list]
    dataset_root = Path(output_dir)
    train_count = _split_count(len(frame_arrays), train_ratio)
    train_frames = frame_arrays[:train_count]
    val_frames = frame_arrays[train_count:]

    with _dataset_export_lock(dataset_root):
        _prepare_dataset_dir(dataset_root, overwrite=False)
        dataset_yaml = dataset_root / "dataset.yaml"
        previous_yaml = dataset_yaml.read_bytes() if dataset_yaml.exists() else None
        created_paths: List[Path] = []
        try:
            _write_empty_frame_records(
                dataset_root,
                "train",
                train_frames,
                image_quality=image_quality,
                start_index=_next_split_image_index(dataset_root, "train"),
                created_paths=created_paths,
            )
            _write_empty_frame_records(
                dataset_root,
                "val",
                val_frames,
                image_quality=image_quality,
                start_index=_next_split_image_index(dataset_root, "val"),
                created_paths=created_paths,
            )
            _write_dataset_yaml(dataset_root)
            validate_yolo_segmentation_dataset(dataset_root)
        except Exception:
            for created_path in reversed(created_paths):
                created_path.unlink(missing_ok=True)
            if previous_yaml is None:
                dataset_yaml.unlink(missing_ok=True)
            else:
                dataset_yaml.write_bytes(previous_yaml)
            raise

    stats = ExportStats(
        exported_images=len(frame_list),
        labels=len(frame_list),
        skipped_masks=0,
        total_wound_instances=0,
        train_images=len(train_frames),
        val_images=len(val_frames),
    )
    _print_stats(stats)
    return stats


def validate_yolo_segmentation_dataset(dataset_dir: Path | str) -> None:
    """Validate the Ultralytics segmentation dataset structure and label rows."""

    dataset_root = Path(dataset_dir)
    for split in ("train", "val"):
        images_dir = _split_images_dir(dataset_root, split)
        labels_dir = _split_labels_dir(dataset_root, split)
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

    return _polygons_from_wound_pixels(
        wound_pixels,
        width=width,
        height=height,
        min_area_px=min_area_px,
        approx_epsilon=approx_epsilon,
    ), width, height


def _polygons_from_mask_array(
    mask: object,
    min_area_px: int,
    approx_epsilon: float,
) -> Tuple[List[List[Point]], int, int]:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a single-channel mask; got shape {mask_array.shape}.")

    mask_array = mask_array.astype(bool)
    height, width = mask_array.shape
    y_coords, x_coords = np.nonzero(mask_array)
    wound_pixels = set(zip(x_coords.tolist(), y_coords.tolist()))
    polygons = _polygons_from_wound_pixels(
        wound_pixels,
        width=width,
        height=height,
        min_area_px=min_area_px,
        approx_epsilon=approx_epsilon,
    )
    return polygons, width, height


def _polygons_from_wound_pixels(
    wound_pixels: Set[Pixel],
    width: int,
    height: int,
    min_area_px: int,
    approx_epsilon: float,
) -> List[List[Point]]:
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

    return polygons


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


def _prepare_dataset_dir(dataset_root: Path, overwrite: bool) -> None:
    for split in ("train", "val"):
        images_dir = _split_images_dir(dataset_root, split)
        labels_dir = _split_labels_dir(dataset_root, split)
        if overwrite:
            for folder in (images_dir, labels_dir):
                if folder.exists():
                    for child in folder.rglob("*"):
                        if child.is_file():
                            child.unlink()
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


def _next_split_image_index(dataset_root: Path, split: str) -> int:
    images_dir = _split_images_dir(dataset_root, split)
    prefix = f"{split}_img"
    max_index = 0
    image_count = 0

    for image_path in images_dir.glob("*.jpg"):
        image_count += 1
        stem = image_path.stem
        if stem.startswith(prefix):
            suffix = stem[len(prefix) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))

    return max(max_index, image_count) + 1


def _write_records(
    dataset_root: Path,
    split: str,
    records: Sequence[ExportRecord],
    image_quality: int,
    start_index: int,
) -> None:
    images_dir = _split_images_dir(dataset_root, split)
    labels_dir = _split_labels_dir(dataset_root, split)

    next_index = start_index
    for record in records:
        output_name = f"{split}_img{next_index:05d}"
        image_path = images_dir / f"{output_name}.jpg"
        label_path = labels_dir / f"{output_name}.txt"
        while image_path.exists() or label_path.exists():
            next_index += 1
            output_name = f"{split}_img{next_index:05d}"
            image_path = images_dir / f"{output_name}.jpg"
            label_path = labels_dir / f"{output_name}.txt"

        with Image.open(record.source_frame) as image:
            image.convert("RGB").save(image_path, format="JPEG", quality=image_quality)

        rows = []
        for polygon in record.polygons:
            coords = " ".join(f"{value:.6f}" for point in polygon for value in point)
            rows.append(f"0 {coords}")
        label_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        next_index += 1


def _write_empty_frame_records(
    dataset_root: Path,
    split: str,
    frames: Sequence[object],
    image_quality: int,
    start_index: int,
    created_paths: List[Path],
) -> None:
    images_dir = _split_images_dir(dataset_root, split)
    labels_dir = _split_labels_dir(dataset_root, split)

    next_index = start_index
    for frame in frames:
        output_name = f"{split}_img{next_index:05d}"
        image_path = images_dir / f"{output_name}.jpg"
        label_path = labels_dir / f"{output_name}.txt"
        while image_path.exists() or label_path.exists():
            next_index += 1
            output_name = f"{split}_img{next_index:05d}"
            image_path = images_dir / f"{output_name}.jpg"
            label_path = labels_dir / f"{output_name}.txt"

        try:
            Image.fromarray(frame).convert("RGB").save(
                image_path,
                format="JPEG",
                quality=image_quality,
            )
            label_path.write_bytes(b"")
        except Exception:
            image_path.unlink(missing_ok=True)
            label_path.unlink(missing_ok=True)
            raise

        created_paths.extend((image_path, label_path))
        next_index += 1


def _as_export_frame_array(frame: object) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.ndim == 3 and frame_array.shape[-1] == 1:
        frame_array = frame_array[..., 0]
    if frame_array.ndim == 2:
        pass
    elif frame_array.ndim == 3 and frame_array.shape[-1] in (3, 4):
        pass
    else:
        raise ValueError(
            f"Expected a grayscale, RGB, or RGBA frame; got shape {frame_array.shape}."
        )
    if frame_array.dtype != np.uint8:
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)
    return frame_array


def _dataset_thread_lock(dataset_root: Path) -> threading.Lock:
    lock_key = str(dataset_root.expanduser().resolve())
    with _DATASET_THREAD_LOCKS_GUARD:
        return _DATASET_THREAD_LOCKS.setdefault(lock_key, threading.Lock())


@contextmanager
def _dataset_export_lock(dataset_root: Path):
    dataset_root = dataset_root.expanduser().resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    lock_path = dataset_root / ".yolo-export.lock"

    with _dataset_thread_lock(dataset_root):
        with lock_path.open("a+b") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_dataset_yaml(dataset_root: Path) -> None:
    dataset_root.mkdir(parents=True, exist_ok=True)
    yaml_text = (
        f"path: {dataset_root.name}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: wound\n"
    )
    (dataset_root / "dataset.yaml").write_text(yaml_text, encoding="utf-8")


def _split_images_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / "images" / split


def _split_labels_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / "labels" / split


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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear existing train/val image and label files before exporting",
    )
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
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
