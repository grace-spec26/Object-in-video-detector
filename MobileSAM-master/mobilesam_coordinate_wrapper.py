import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_MASK_DATA_DIR = PROJECT_ROOT / "raw-mask-data"
SUPPORTED_FRAME_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def resolve_checkpoint(checkpoint: Optional[Path]) -> Path:
    if checkpoint is not None:
        resolved = checkpoint
    else:
        resolved = Path(__file__).resolve().parent / "weights" / "mobile_sam.pt"
    if not resolved.exists():
        raise FileNotFoundError(f"MobileSAM checkpoint not found: {resolved}")
    return resolved


def load_predictor(checkpoint: Optional[Path] = None, device: Optional[str] = None):
    import torch
    from mobile_sam import SamPredictor, sam_model_registry

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = sam_model_registry["vit_t"](checkpoint=str(resolve_checkpoint(checkpoint)))
    model = model.to(device=resolved_device)
    model.eval()
    return SamPredictor(model)


def _clear_matching_files(output_dir: Path, patterns: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _json_ready_points(points: np.ndarray) -> List[List[float]]:
    return [[float(x), float(y)] for x, y in points.tolist()]


def _as_point_array(points: Any, source: Path) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected an array of [x, y] points in {source}")
    if len(coords) == 0:
        raise ValueError(f"No positive points found in {source}")
    return coords


def _as_optional_point_array(points: Any, source: Path) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    coords = np.asarray(points, dtype=np.float32)
    if coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected an array of [x, y] points in {source}")
    return coords


def _clamp_points(points: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    clamped = np.asarray(points, dtype=np.float32).copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, image_width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, image_height - 1)
    return clamped


def _point_record_label(point: Dict[str, Any]) -> int:
    if "label" in point:
        return int(point["label"])
    return 0 if str(point.get("type", "")).lower() == "negative" else 1


def _point_records_to_arrays(
    records: Sequence[Any],
    source: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = []
    labels = []
    visible = []
    for point in records:
        if isinstance(point, dict):
            if "x" not in point or "y" not in point:
                raise ValueError(f"Point record missing x/y in {source}")
            coords.append([float(point["x"]), float(point["y"])])
            labels.append(_point_record_label(point))
            visible.append(bool(point.get("visible", True)))
        else:
            raw_point = np.asarray(point, dtype=np.float32)
            if raw_point.shape != (2,):
                raise ValueError(f"Expected point records as [x, y] pairs in {source}")
            coords.append([float(raw_point[0]), float(raw_point[1])])
            labels.append(1)
            visible.append(True)

    if not coords:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=bool),
        )
    return (
        np.asarray(coords, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        np.asarray(visible, dtype=bool),
    )


def _labels_for_coords(labels: Any, coords: np.ndarray, source: Path) -> np.ndarray:
    if labels is None:
        return np.ones((len(coords),), dtype=np.int32)
    label_array = np.asarray(labels, dtype=np.int32)
    if label_array.ndim != 1 or len(label_array) != len(coords):
        raise ValueError(f"point_labels length does not match point_coords in {source}")
    return label_array


def _filter_positive_visible_points(
    coords: np.ndarray,
    labels: Optional[np.ndarray] = None,
    visible: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask = np.ones((len(coords),), dtype=bool)
    if labels is not None:
        mask &= labels == 1
    if visible is not None:
        mask &= visible
    return coords[mask]


def _positive_coords_from_object(
    obj: Dict[str, Any],
    source: Path,
    visible_only: bool,
) -> Tuple[np.ndarray, int]:
    class_id = int(obj.get("class_id", 1))
    point_records = obj.get("points") or []
    visible_mask = None

    if "positive_points" in obj:
        coords = _as_optional_point_array(obj.get("positive_points"), source)
        labels = None
    elif "point_coords" in obj:
        coords = _as_optional_point_array(obj.get("point_coords"), source)
        labels = None
        if obj.get("point_labels") is not None:
            labels = _labels_for_coords(obj.get("point_labels"), coords, source)
        elif point_records:
            _, record_labels, record_visible = _point_records_to_arrays(point_records, source)
            if len(record_labels) == len(coords):
                labels = record_labels
            if visible_only and len(record_visible) == len(coords):
                visible_mask = record_visible
    elif point_records:
        coords, labels, record_visible = _point_records_to_arrays(point_records, source)
        visible_mask = record_visible if visible_only else None
    else:
        raise ValueError(f"Unsupported object coordinate format in {source}")

    coords = _filter_positive_visible_points(coords, labels=labels, visible=visible_mask)
    return coords, class_id


def _extract_positive_point_sets(
    data: Any,
    source: Path,
    visible_only: bool,
) -> List[Tuple[np.ndarray, int]]:
    if isinstance(data, list):
        if data and isinstance(data[0], dict):
            coords, labels, record_visible = _point_records_to_arrays(data, source)
            visible_mask = record_visible if visible_only else None
            coords = _filter_positive_visible_points(coords, labels=labels, visible=visible_mask)
            return [(coords, 1)]
        coords = _as_point_array(data, source)
        return [(coords, 1)]

    if not isinstance(data, dict):
        raise ValueError(f"Unsupported coordinate JSON format in {source}")

    objects = data.get("objects")
    if objects:
        point_sets: List[Tuple[np.ndarray, int]] = []
        for obj in objects:
            coords, class_id = _positive_coords_from_object(obj, source, visible_only)
            if len(coords) > 0:
                point_sets.append((coords, class_id))

        if not point_sets:
            raise ValueError(f"No positive points found in {source}")
        return point_sets

    if isinstance(data.get("points"), list) and data["points"] and isinstance(data["points"][0], dict):
        coords, labels, record_visible = _point_records_to_arrays(data["points"], source)
        visible_mask = record_visible if visible_only else None
        coords = _filter_positive_visible_points(coords, labels=labels, visible=visible_mask)
        return [(coords, int(data.get("class_id", 1)))]

    for key in ("point_coords", "points", "coordinates"):
        if key in data:
            coords = _as_point_array(data[key], source)
            labels = None
            if data.get("point_labels") is not None:
                labels = _labels_for_coords(data.get("point_labels"), coords, source)
            coords = _filter_positive_visible_points(coords, labels=labels)
            return [(coords, int(data.get("class_id", 1)))]

    raise ValueError(f"Unsupported coordinate JSON format in {source}")


def generate_expanded_box_negative_points(
    positive_points: np.ndarray,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
) -> np.ndarray:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image width and height must be positive.")
    if padding_ratio < 0:
        raise ValueError("Padding ratio must be non-negative.")

    positives = _clamp_points(positive_points, image_width, image_height)
    min_x = float(np.min(positives[:, 0]))
    max_x = float(np.max(positives[:, 0]))
    min_y = float(np.min(positives[:, 1]))
    max_y = float(np.max(positives[:, 1]))

    width = max_x - min_x
    height = max_y - min_y
    pad_x = max(width * float(padding_ratio), 1.0 if width == 0 else 0.0)
    pad_y = max(height * float(padding_ratio), 1.0 if height == 0 else 0.0)

    expanded_min_x = max(0.0, min_x - pad_x)
    expanded_max_x = min(float(image_width - 1), max_x + pad_x)
    expanded_min_y = max(0.0, min_y - pad_y)
    expanded_max_y = min(float(image_height - 1), max_y + pad_y)

    return np.asarray(
        [
            [expanded_min_x, expanded_min_y],
            [expanded_max_x, expanded_min_y],
            [expanded_max_x, expanded_max_y],
            [expanded_min_x, expanded_max_y],
        ],
        dtype=np.float32,
    )


def build_augmented_prompt_object(
    positive_points: Any,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
    class_id: int = 1,
) -> Dict[str, Any]:
    positives = _clamp_points(
        _as_point_array(positive_points, Path("<coordinates>")),
        image_width=image_width,
        image_height=image_height,
    )
    negatives = generate_expanded_box_negative_points(
        positives,
        image_width=image_width,
        image_height=image_height,
        padding_ratio=padding_ratio,
    )
    point_coords = np.concatenate([positives, negatives], axis=0)
    point_labels = [1] * len(positives) + [0] * len(negatives)

    return {
        "class_id": int(class_id),
        "positive_points": _json_ready_points(positives),
        "negative_points": _json_ready_points(negatives),
        "point_coords": _json_ready_points(point_coords),
        "point_labels": point_labels,
    }


def build_augmented_prompt_json(
    positive_points: Any,
    image_width: int,
    image_height: int,
    padding_ratio: float = 0.15,
    class_id: int = 1,
    source_coordinate_json: Optional[Path] = None,
) -> Dict[str, Any]:
    return {
        "source_coordinate_json": str(source_coordinate_json) if source_coordinate_json else None,
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "padding_ratio": float(padding_ratio),
        "objects": [
            build_augmented_prompt_object(
                positive_points=positive_points,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                class_id=class_id,
            )
        ],
    }


def _prompt_objects_from_augmented_json(
    augmented_prompt_json: Dict[str, Any],
    image_width: int,
    image_height: int,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    prompt_objects: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for obj in augmented_prompt_json.get("objects", []):
        coords = _clamp_points(
            _as_point_array(obj.get("point_coords", []), Path("<augmented-coordinates>")),
            image_width=image_width,
            image_height=image_height,
        )
        labels = np.asarray(obj.get("point_labels", []), dtype=np.int32)
        if labels.ndim != 1 or len(labels) != len(coords):
            raise ValueError("point_labels length does not match point_coords")
        prompt_objects.append((coords, labels, int(obj.get("class_id", 1))))
    return prompt_objects


def prepare_coordinate_prompt_json(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    output_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
    visible_only: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    if not coordinate_json_path.exists():
        raise FileNotFoundError(f"Coordinate JSON not found: {coordinate_json_path}")

    data = json.loads(coordinate_json_path.read_text(encoding="utf-8"))
    objects = []
    for positive_points, class_id in _extract_positive_point_sets(
        data,
        source=coordinate_json_path,
        visible_only=visible_only,
    ):
        objects.append(
            build_augmented_prompt_object(
                positive_points=positive_points,
                image_width=image_width,
                image_height=image_height,
                padding_ratio=padding_ratio,
                class_id=class_id,
            )
        )

    augmented_prompt_json = {
        "source_coordinate_json": str(coordinate_json_path),
        "image_size": {"width": int(image_width), "height": int(image_height)},
        "padding_ratio": float(padding_ratio),
        "objects": objects,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(augmented_prompt_json, indent=2),
            encoding="utf-8",
        )

    return _prompt_objects_from_augmented_json(
        augmented_prompt_json,
        image_width=image_width,
        image_height=image_height,
    )


def load_prompt_objects(
    coordinate_json_path: Path,
    image_width: int,
    image_height: int,
    visible_only: bool = True,
    augmented_output_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    return prepare_coordinate_prompt_json(
        coordinate_json_path=coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        output_path=augmented_output_path,
        padding_ratio=padding_ratio,
        visible_only=visible_only,
    )


def _coerce_frame_step(frame_step: Any, field_name: str = "frame_step") -> int:
    try:
        step_value = float(frame_step)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive whole number.") from exc
    if not step_value.is_integer():
        raise ValueError(f"{field_name} must be a positive whole number.")
    step = int(step_value)
    if step < 1:
        raise ValueError(f"{field_name} must be at least 1.")
    return step


def select_frames_for_frame_step(
    frame_paths: Sequence[Path],
    frame_step: float,
) -> List[Path]:
    sorted_paths = sorted(frame_paths)
    return list(sorted_paths[::_coerce_frame_step(frame_step)])


def select_frames_for_target_fps(
    frame_paths: Sequence[Path],
    target_fps: float,
    source_fps: float = 30.0,
) -> List[Path]:
    return select_frames_for_frame_step(frame_paths, frame_step=target_fps)


def list_frame_paths(frames_dir: Path) -> List[Path]:
    return sorted(
        path for path in frames_dir.iterdir() if path.suffix.lower() in SUPPORTED_FRAME_EXTENSIONS
    )


def save_preview(frame_rgb: np.ndarray, mask: np.ndarray, preview_path: Path) -> None:
    overlay = frame_rgb.astype(np.float32).copy()
    wound = mask == 1
    ignore = mask == 255
    overlay[wound] = overlay[wound] * 0.45 + np.array([255, 40, 40], dtype=np.float32) * 0.55
    overlay[ignore] = overlay[ignore] * 0.45 + np.array([255, 220, 40], dtype=np.float32) * 0.55
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(preview_path)


def run_mobilesam_for_frame(
    predictor,
    frame_path: Path,
    coordinate_json_path: Path,
    mask_path: Path,
    preview_path: Optional[Path] = None,
    augmented_coordinate_json_path: Optional[Path] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
) -> None:
    if not frame_path.exists():
        raise FileNotFoundError(f"Frame not found: {frame_path}")

    frame_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
    image_height, image_width = frame_rgb.shape[:2]
    prompt_objects = load_prompt_objects(
        coordinate_json_path,
        image_width=image_width,
        image_height=image_height,
        visible_only=visible_only,
        augmented_output_path=augmented_coordinate_json_path,
        padding_ratio=padding_ratio,
    )

    output_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if not prompt_objects:
        output_mask[:, :] = 255
    else:
        predictor.set_image(frame_rgb)
        for point_coords, point_labels, class_id in prompt_objects:
            masks, scores, _ = predictor.predict(
                point_coords=point_coords.astype(np.float32),
                point_labels=point_labels.astype(np.int32),
                multimask_output=True,
            )
            best_index = int(np.argmax(scores))
            best_mask = masks[best_index].astype(bool)
            best_score = float(scores[best_index])
            if best_score < score_threshold:
                output_mask[best_mask] = 255
            else:
                output_mask[best_mask] = 1 if class_id == 1 else int(class_id)

    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_mask, mode="L").save(mask_path)
    if preview_path is not None:
        save_preview(frame_rgb, output_mask, preview_path)


def _make_zip_archive(source_dir: Path, archive_path: Path) -> Path:
    if archive_path.exists():
        archive_path.unlink()
    archive_base = archive_path.with_suffix("")
    created_path = shutil.make_archive(
        str(archive_base),
        "zip",
        root_dir=source_dir.parent,
        base_dir=source_dir.name,
    )
    return Path(created_path)


def _progress_iter(items: Sequence[Path], progress: Optional[Any], desc: str) -> Iterable[Path]:
    if progress is not None and hasattr(progress, "tqdm"):
        return progress.tqdm(items, desc=desc)
    return items


def format_coordinate_progress_html(completed: int, total: int, message: str = "Ready") -> str:
    safe_total = max(int(total), 0)
    safe_completed = max(0, int(completed))
    if safe_total > 0:
        safe_completed = min(safe_completed, safe_total)
        percent = int(round((safe_completed / safe_total) * 100))
    else:
        percent = 0

    safe_message = html.escape(str(message))
    return (
        '<div class="coordinate-progress" '
        'style="width: 100%; padding: 10px 0 2px 0;">'
        '<div style="display: flex; justify-content: space-between; '
        'font-size: 14px; margin-bottom: 6px; color: #374151;">'
        f"<span>{safe_message}</span>"
        f"<span>{safe_completed} / {safe_total}</span>"
        "</div>"
        '<div style="height: 14px; width: 100%; background: #e5e7eb; '
        'border-radius: 999px; overflow: hidden;">'
        f'<div style="height: 100%; width: {percent}%; '
        'background: linear-gradient(90deg, #2563eb, #16a34a); '
        'border-radius: 999px; transition: width 160ms ease;"></div>'
        "</div>"
        f'<div style="font-size: 12px; margin-top: 4px; color: #6b7280;">{percent}%</div>'
        "</div>"
    )


def iter_coordinate_prompt_folder_steps(
    frames_dir: Path,
    coordinates_dir: Path,
    output_root: Path = DEFAULT_RAW_MASK_DATA_DIR,
    predictor=None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    frame_step: Optional[float] = None,
    target_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
) -> Iterable[Dict[str, Any]]:
    frames_dir = Path(frames_dir)
    coordinates_dir = Path(coordinates_dir)
    output_root = Path(output_root)
    frames_output_dir = output_root / "frames"
    coordinates_output_dir = output_root / "coordinates"
    masks_output_dir = output_root / "mask"
    previews_output_dir = output_root / "masked_frame"

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    effective_frame_step = frame_step if frame_step is not None else target_fps
    if effective_frame_step is None:
        effective_frame_step = 1
    selected_frame_paths = select_frames_for_frame_step(
        frame_paths,
        frame_step=effective_frame_step,
    )
    missing_json = [
        coordinates_dir / f"{frame_path.stem}.json"
        for frame_path in selected_frame_paths
        if not (coordinates_dir / f"{frame_path.stem}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Missing coordinate JSON for frame: {missing_json[0]}")

    _clear_matching_files(frames_output_dir, ("*.png", "*.jpg", "*.jpeg"))
    _clear_matching_files(coordinates_output_dir, ("*.json",))
    _clear_matching_files(masks_output_dir, ("*.png", "*.jpg", "*.jpeg"))
    _clear_matching_files(previews_output_dir, ("*.png", "*.jpg", "*.jpeg"))

    if predictor is None:
        predictor = load_predictor(checkpoint=checkpoint, device=device)

    total = len(selected_frame_paths)
    yield {
        "stage": "starting",
        "completed": 0,
        "total": total,
        "message": f"Starting MobileSAM for {total} frame(s)",
        "result": None,
    }

    for frame_index, frame_path in enumerate(selected_frame_paths, start=1):
        copied_frame_path = frames_output_dir / frame_path.name
        shutil.copy2(frame_path, copied_frame_path)

        run_mobilesam_for_frame(
            predictor=predictor,
            frame_path=frame_path,
            coordinate_json_path=coordinates_dir / f"{frame_path.stem}.json",
            mask_path=masks_output_dir / f"{frame_path.stem}.png",
            preview_path=previews_output_dir / f"{frame_path.stem}.jpg",
            augmented_coordinate_json_path=coordinates_output_dir / f"{frame_path.stem}.json",
            padding_ratio=float(padding_ratio),
            score_threshold=score_threshold,
            visible_only=visible_only,
        )

        yield {
            "stage": "processing",
            "completed": frame_index,
            "total": total,
            "message": f"Processed {frame_path.name}",
            "result": None,
        }

    frames_zip = _make_zip_archive(frames_output_dir, output_root / "frames.zip")
    masks_zip = _make_zip_archive(masks_output_dir, output_root / "mask.zip")
    previews_zip = _make_zip_archive(previews_output_dir, output_root / "masked_frame.zip")
    result = {
        "processed_frames": len(selected_frame_paths),
        "frame_paths": selected_frame_paths,
        "frames_dir": frames_output_dir,
        "coordinates_dir": coordinates_output_dir,
        "masks_dir": masks_output_dir,
        "previews_dir": previews_output_dir,
        "frames_zip": frames_zip,
        "masks_zip": masks_zip,
        "previews_zip": previews_zip,
    }

    yield {
        "stage": "done",
        "completed": total,
        "total": total,
        "message": f"Completed {total} frame(s)",
        "result": result,
    }


def run_coordinate_prompt_folders(
    frames_dir: Path,
    coordinates_dir: Path,
    output_root: Path = DEFAULT_RAW_MASK_DATA_DIR,
    predictor=None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    frame_step: Optional[float] = None,
    target_fps: Optional[float] = None,
    source_fps: Optional[float] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    last_update = None
    for update in iter_coordinate_prompt_folder_steps(
        frames_dir=frames_dir,
        coordinates_dir=coordinates_dir,
        output_root=output_root,
        predictor=predictor,
        checkpoint=checkpoint,
        device=device,
        frame_step=frame_step,
        target_fps=target_fps,
        source_fps=source_fps,
        padding_ratio=padding_ratio,
        score_threshold=score_threshold,
        visible_only=visible_only,
    ):
        last_update = update
        if progress is not None and callable(progress):
            progress(
                (int(update["completed"]), int(update["total"])),
                desc=str(update["message"]),
            )

    if last_update and last_update.get("result"):
        return last_update["result"]

    raise RuntimeError("MobileSAM coordinate folder processing did not complete.")


def run_directory(
    frames_dir: Path,
    coordinates_dir: Path,
    masks_dir: Path,
    preview_dir: Optional[Path] = None,
    augmented_coordinates_dir: Optional[Path] = None,
    checkpoint: Optional[Path] = None,
    device: Optional[str] = None,
    padding_ratio: float = 0.15,
    score_threshold: float = 0.0,
    visible_only: bool = True,
    dry_run: bool = False,
) -> None:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not coordinates_dir.exists():
        raise FileNotFoundError(f"Coordinates directory not found: {coordinates_dir}")

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise RuntimeError(f"No frames found in: {frames_dir}")

    missing_json = [
        coordinates_dir / f"{frame_path.stem}.json"
        for frame_path in frame_paths
        if not (coordinates_dir / f"{frame_path.stem}.json").exists()
    ]
    if missing_json:
        raise FileNotFoundError(f"Missing coordinate JSON for frame: {missing_json[0]}")

    if dry_run:
        print(f"frames={len(frame_paths)}")
        print(f"frames_dir={frames_dir}")
        print(f"coordinates_dir={coordinates_dir}")
        print(f"masks_dir={masks_dir}")
        print(f"preview_dir={preview_dir}")
        print(f"augmented_coordinates_dir={augmented_coordinates_dir}")
        return

    if augmented_coordinates_dir is None:
        augmented_coordinates_dir = DEFAULT_RAW_MASK_DATA_DIR / "coordinates"

    predictor = load_predictor(checkpoint=checkpoint, device=device)
    for frame_path in frame_paths:
        json_path = coordinates_dir / f"{frame_path.stem}.json"
        mask_path = masks_dir / f"{frame_path.stem}.png"
        preview_path = preview_dir / f"{frame_path.stem}.jpg" if preview_dir else None
        augmented_path = augmented_coordinates_dir / f"{frame_path.stem}.json"
        run_mobilesam_for_frame(
            predictor=predictor,
            frame_path=frame_path,
            coordinate_json_path=json_path,
            mask_path=mask_path,
            preview_path=preview_path,
            augmented_coordinate_json_path=augmented_path,
            padding_ratio=padding_ratio,
            score_threshold=score_threshold,
            visible_only=visible_only,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MobileSAM wound masks from raw frames and CoTracker coordinate JSON."
    )
    parser.add_argument("--frames-dir", required=True, type=Path, help="Directory containing raw exported frames.")
    parser.add_argument("--coordinates-dir", required=True, type=Path, help="Directory containing matching JSON prompts.")
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=DEFAULT_RAW_MASK_DATA_DIR / "mask",
        help="Output directory for single-channel PNG masks.",
    )
    parser.add_argument("--preview-dir", type=Path, default=None, help="Optional output directory for masked previews.")
    parser.add_argument(
        "--augmented-coordinates-dir",
        type=Path,
        default=DEFAULT_RAW_MASK_DATA_DIR / "coordinates",
        help="Output directory for augmented prompt JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="If set, copy sampled frames, augmented coordinates, masks, and zip downloads under this folder.",
    )
    parser.add_argument(
        "--frame-step",
        type=float,
        default=None,
        help="Process every Nth frame. For example, 3 processes frames 0, 3, 6...",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Deprecated alias for --frame-step.",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=30.0,
        help="Deprecated; frame sampling now uses --frame-step directly.",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.15,
        help="Expanded-box padding ratio used to create four negative corner prompts.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional MobileSAM checkpoint path.")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cpu or cuda.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Masks below this score are encoded as 255 ignore instead of wound.",
    )
    parser.add_argument(
        "--include-invisible-points",
        action="store_true",
        help="Use points marked visible=false in the coordinate JSON.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate frame/JSON matching without loading MobileSAM.")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    frame_step = args.frame_step if args.frame_step is not None else args.target_fps
    if frame_step is None:
        frame_step = 1
    if args.output_root is not None or args.frame_step is not None or args.target_fps is not None:
        if args.dry_run:
            frame_paths = list_frame_paths(args.frames_dir)
            selected_frame_paths = select_frames_for_frame_step(
                frame_paths,
                frame_step=frame_step,
            )
            print(f"frames={len(frame_paths)}")
            print(f"selected_frames={len(selected_frame_paths)}")
            print(f"frames_dir={args.frames_dir}")
            print(f"coordinates_dir={args.coordinates_dir}")
            print(f"output_root={args.output_root or DEFAULT_RAW_MASK_DATA_DIR}")
            print(f"frame_step={_coerce_frame_step(frame_step)}")
            return

        result = run_coordinate_prompt_folders(
            frames_dir=args.frames_dir,
            coordinates_dir=args.coordinates_dir,
            output_root=args.output_root or DEFAULT_RAW_MASK_DATA_DIR,
            checkpoint=args.checkpoint,
            device=args.device,
            frame_step=frame_step,
            padding_ratio=args.padding_ratio,
            score_threshold=args.score_threshold,
            visible_only=not args.include_invisible_points,
        )
        print(f"processed_frames={result['processed_frames']}")
        print(f"frames_dir={result['frames_dir']}")
        print(f"coordinates_dir={result['coordinates_dir']}")
        print(f"masks_dir={result['masks_dir']}")
        print(f"previews_dir={result['previews_dir']}")
        print(f"frames_zip={result['frames_zip']}")
        print(f"masks_zip={result['masks_zip']}")
        print(f"previews_zip={result['previews_zip']}")
        return

    run_directory(
        frames_dir=args.frames_dir,
        coordinates_dir=args.coordinates_dir,
        masks_dir=args.masks_dir,
        preview_dir=args.preview_dir,
        augmented_coordinates_dir=args.augmented_coordinates_dir,
        checkpoint=args.checkpoint,
        device=args.device,
        padding_ratio=args.padding_ratio,
        score_threshold=args.score_threshold,
        visible_only=not args.include_invisible_points,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
