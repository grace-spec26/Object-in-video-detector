import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_DIR = PROJECT_ROOT / "data" / "frames"
DEFAULT_COORDINATES_DIR = PROJECT_ROOT / "data" / "coordinates"


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


def store_coordinate_arrays(
    tracks: np.ndarray,
    output_dir: Path = DEFAULT_COORDINATES_DIR,
    source_hw: Optional[Sequence[int]] = None,
    target_hw: Optional[Sequence[int]] = None,
) -> List[Path]:
    """Store selected-point tracks as per-frame JSON arrays of [x, y] pixels."""
    output_dir = Path(output_dir)
    _clear_matching_files(output_dir, ("frame_*.json", "coordinates.json"))

    tracks_array = np.asarray(tracks, dtype=np.float32)
    if source_hw is not None and target_hw is not None:
        tracks_array = scale_tracks_to_frame_space(tracks_array, source_hw, target_hw)
    elif tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
        raise ValueError("Tracks must have shape (N, T, 2).")

    tracks_by_frame = np.transpose(tracks_array, (1, 0, 2))
    all_coordinates = tracks_by_frame.tolist()

    written_paths: List[Path] = []
    for frame_index, frame_coordinates in enumerate(all_coordinates):
        coordinate_path = output_dir / f"frame_{frame_index:06d}.json"
        coordinate_path.write_text(json.dumps(frame_coordinates, indent=2), encoding="utf-8")
        written_paths.append(coordinate_path)

    aggregate_path = output_dir / "coordinates.json"
    aggregate_path.write_text(json.dumps(all_coordinates, indent=2), encoding="utf-8")
    written_paths.append(aggregate_path)
    return written_paths
