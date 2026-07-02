import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_COORDINATES_DIR = PROJECT_ROOT / "data" / "coordinates"
POSITIVE_POINT_LABEL = 1
NEGATIVE_POINT_LABEL = 0


def _clear_matching_files(output_dir: Path, patterns: Iterable[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _as_uint8_image(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Frames must be RGB or RGBA images with shape (H, W, 3/4).")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
        image = image * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def store_original_frames(frames: np.ndarray, output_dir: Path = DEFAULT_FRAMES_DIR) -> List[Path]:
    """Store original video frames as lossless PNG files named frame_000000.png."""
    frames_array = np.asarray(frames)
    if frames_array.ndim != 4 or frames_array.shape[-1] not in (3, 4):
        raise ValueError("Video frames must have shape (T, H, W, 3/4).")

    output_dir = Path(output_dir)
    _clear_matching_files(output_dir, ("frame_*.png", "frame_*.jpg", "frame_*.jpeg"))

    written_paths: List[Path] = []
    for frame_index, frame in enumerate(frames_array):
        image = _as_uint8_image(frame)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

        frame_path = output_dir / f"frame_{frame_index:06d}.png"
        if not cv2.imwrite(str(frame_path), image):
            raise RuntimeError(f"Failed to write frame: {frame_path}")
        written_paths.append(frame_path)

    return written_paths


def scale_tracks_to_frame_space(
    tracks: np.ndarray,
    source_hw: Sequence[int],
    target_hw: Sequence[int],
) -> np.ndarray:
    """Scale tracks from preview/image-space pixels into stored-frame pixels."""
    tracks_array = np.asarray(tracks, dtype=np.float32)
    if tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
        raise ValueError("Tracks must have shape (N, T, 2).")

    source_height, source_width = [float(value) for value in source_hw]
    target_height, target_width = [float(value) for value in target_hw]
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("Source and target frame dimensions must be positive.")

    scaled = tracks_array.copy()
    scaled[..., 0] *= target_width / source_width
    scaled[..., 1] *= target_height / source_height
    scaled[..., 0] = np.clip(scaled[..., 0], 0, target_width - 1)
    scaled[..., 1] = np.clip(scaled[..., 1], 0, target_height - 1)
    return scaled


def normalize_point_labels(point_labels: Optional[Sequence[int]], point_count: int) -> np.ndarray:
    """Return one SAM prompt label per selected track."""
    if point_count < 0:
        raise ValueError("point_count must be non-negative.")
    if point_labels is None:
        return np.ones((point_count,), dtype=np.int32)

    labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    if len(labels) != point_count:
        raise ValueError("point_labels length must match the number of tracks.")
    if not np.isin(labels, [NEGATIVE_POINT_LABEL, POSITIVE_POINT_LABEL]).all():
        raise ValueError("point_labels must contain only 1 for positive and 0 for negative points.")
    return labels


def labeled_query_points_for_frame(
    query_points: Optional[Sequence[Sequence[Sequence[float]]]],
    frame_index: int,
    source_hw: Optional[Sequence[int]] = None,
    target_hw: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return same-frame clicked query points as SAM coordinates and labels."""
    empty_coords = np.empty((0, 2), dtype=np.float32)
    empty_labels = np.empty((0,), dtype=np.int32)
    if query_points is None or len(query_points) == 0:
        return empty_coords, empty_labels

    frame_index = int(frame_index)
    if frame_index < 0 or frame_index >= len(query_points):
        return empty_coords, empty_labels

    coords: List[List[float]] = []
    labels: List[int] = []
    for point in query_points[frame_index]:
        if len(point) < 2:
            raise ValueError("Each query point must contain at least x and y coordinates.")
        coords.append([float(point[0]), float(point[1])])
        labels.append(int(point[3]) if len(point) >= 4 else POSITIVE_POINT_LABEL)

    if not coords:
        return empty_coords, empty_labels

    coords_array = np.asarray(coords, dtype=np.float32)
    labels_array = normalize_point_labels(labels, len(coords_array))
    if source_hw is not None and target_hw is not None:
        source_height, source_width = [float(value) for value in source_hw]
        target_height, target_width = [float(value) for value in target_hw]
        if min(source_height, source_width, target_height, target_width) <= 0:
            raise ValueError("Source and target frame dimensions must be positive.")
        coords_array = coords_array.copy()
        coords_array[:, 0] *= target_width / source_width
        coords_array[:, 1] *= target_height / source_height
        coords_array[:, 0] = np.clip(coords_array[:, 0], 0, target_width - 1)
        coords_array[:, 1] = np.clip(coords_array[:, 1], 0, target_height - 1)
    return coords_array, labels_array


def visible_labeled_points_for_frame(
    tracks: np.ndarray,
    frame_index: int,
    point_labels: Optional[Sequence[int]] = None,
    visibility: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return visible point coordinates and SAM labels for one frame."""
    tracks_array = np.asarray(tracks, dtype=np.float32)
    if tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
        raise ValueError("Tracks must have shape (N, T, 2).")

    point_count, frame_count = tracks_array.shape[:2]
    if frame_count <= 0:
        raise ValueError("Tracks must contain at least one frame.")

    labels = normalize_point_labels(point_labels, point_count)
    frame_index = int(np.clip(int(frame_index), 0, frame_count - 1))

    visible_mask = np.ones((point_count,), dtype=bool)
    if visibility is not None:
        visibility_array = np.asarray(visibility, dtype=bool)
        if visibility_array.shape != tracks_array.shape[:2]:
            raise ValueError("Visibility must have shape (N, T) to match tracks.")
        visible_mask = visibility_array[:, frame_index]

    return tracks_array[:, frame_index, :][visible_mask], labels[visible_mask]


def build_sam_point_prompt_payload(
    point_coords: np.ndarray,
    point_labels: Sequence[int],
    image_size: Optional[Sequence[int]] = None,
    class_id: int = 1,
) -> Dict[str, object]:
    """Build the SAM-friendly JSON object used by MobileSAM/SAM2 wrappers."""
    coords = np.asarray(point_coords, dtype=np.float32)
    if coords.size == 0:
        coords = np.empty((0, 2), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[-1] != 2:
        raise ValueError("point_coords must have shape (N, 2).")

    labels = normalize_point_labels(point_labels, len(coords))
    positives = coords[labels == POSITIVE_POINT_LABEL]
    negatives = coords[labels == NEGATIVE_POINT_LABEL]

    payload: Dict[str, object] = {
        "objects": [
            {
                "class_id": int(class_id),
                "positive_points": positives.tolist(),
                "negative_points": negatives.tolist(),
                "point_coords": coords.tolist(),
                "point_labels": [int(label) for label in labels.tolist()],
            }
        ]
    }
    if image_size is not None:
        height, width = [int(value) for value in image_size]
        payload["image_size"] = {"width": width, "height": height}
    return payload


def store_coordinate_arrays(
    tracks: np.ndarray,
    output_dir: Path = DEFAULT_COORDINATES_DIR,
    source_hw: Optional[Sequence[int]] = None,
    target_hw: Optional[Sequence[int]] = None,
    visibility: Optional[np.ndarray] = None,
    point_labels: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Store selected-point tracks as per-frame JSON arrays or SAM prompt objects."""
    output_dir = Path(output_dir)
    _clear_matching_files(output_dir, ("frame_*.json", "coordinates.json"))

    tracks_array = np.asarray(tracks, dtype=np.float32)
    if source_hw is not None and target_hw is not None:
        tracks_array = scale_tracks_to_frame_space(tracks_array, source_hw, target_hw)
    elif tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
        raise ValueError("Tracks must have shape (N, T, 2).")

    if visibility is not None:
        visibility_array = np.asarray(visibility, dtype=bool)
        if visibility_array.shape != tracks_array.shape[:2]:
            raise ValueError("Visibility must have shape (N, T) to match tracks.")
    else:
        visibility_array = None

    labels_array = None
    if point_labels is not None:
        labels_array = normalize_point_labels(point_labels, tracks_array.shape[0])

    tracks_by_frame = np.transpose(tracks_array, (1, 0, 2))
    if visibility_array is None and labels_array is None:
        all_coordinates = tracks_by_frame.tolist()
    elif labels_array is None:
        visibility_by_frame = np.transpose(visibility_array, (1, 0))
        all_coordinates = [
            frame_tracks[frame_visibility].tolist()
            for frame_tracks, frame_visibility in zip(tracks_by_frame, visibility_by_frame)
        ]
    else:
        image_size = target_hw if target_hw is not None else source_hw
        all_coordinates = []
        for frame_index in range(tracks_array.shape[1]):
            frame_tracks, frame_labels = visible_labeled_points_for_frame(
                tracks_array,
                frame_index,
                point_labels=labels_array,
                visibility=visibility_array,
            )
            frame_payload = build_sam_point_prompt_payload(
                frame_tracks,
                frame_labels,
                image_size=image_size,
            )
            frame_payload["frame_index"] = frame_index
            all_coordinates.append(frame_payload)

    written_paths: List[Path] = []
    for frame_index, frame_coordinates in enumerate(all_coordinates):
        coordinate_path = output_dir / f"frame_{frame_index:06d}.json"
        coordinate_path.write_text(json.dumps(frame_coordinates, indent=2), encoding="utf-8")
        written_paths.append(coordinate_path)

    aggregate_path = output_dir / "coordinates.json"
    aggregate_payload = all_coordinates
    if labels_array is not None:
        aggregate_payload = {
            "format": "sam_point_prompts",
            "frames": all_coordinates,
        }
    aggregate_path.write_text(json.dumps(aggregate_payload, indent=2), encoding="utf-8")
    written_paths.append(aggregate_path)
    return written_paths
