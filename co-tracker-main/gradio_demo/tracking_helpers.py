from pathlib import Path
import shutil
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


TRACKING_RESOLUTION_OPTIONS = ("1080P", "720P", "640P", "520P")
DEFAULT_TRACKING_RESOLUTION = "520P"
TRACKING_FRAME_STRIDE = 2
DEFAULT_SAM_PREVIEW_VIDEO_FILENAME = "sam_video_preview.mp4"
_TRACKING_RESOLUTION_HEIGHTS = {
    "520P": 520,
    "640P": 640,
    "720P": 720,
    "1080P": 1080,
}
_MODEL_CACHE = {}
_LOCAL_COTRACKER_REPO = Path(__file__).resolve().parents[1]


def parse_max_frame_count(value) -> int:
    """Return a non-negative frame cap; 0 means load the full uploaded video."""
    if value is None:
        return 0
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0

    try:
        max_frames = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Max frames to load must be a whole number. Use 0 to load all frames.") from exc

    return max(0, max_frames)


def parse_frame_skip_count(value) -> int:
    """Return a non-negative frame skip count; 0 means keep every frame."""
    if value is None:
        return 0
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return 0

    try:
        skip_count = int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("Skip frame count must be a whole number. Use 0 to keep every frame.") from exc

    return max(0, skip_count)


def _even(value: int) -> int:
    return max(2, value - (value % 2))


def get_tracking_resolution(label: str, source_hw: Sequence[int]) -> Tuple[int, int]:
    """Return an aspect-preserving (height, width) for the selected tracking size."""
    if label not in _TRACKING_RESOLUTION_HEIGHTS:
        raise ValueError(f"Unknown tracking resolution: {label}")

    source_height, source_width = [int(value) for value in source_hw]
    if source_height <= 0 or source_width <= 0:
        raise ValueError("Source video dimensions must be positive.")

    target_height = min(_TRACKING_RESOLUTION_HEIGHTS[label], source_height)
    target_width = round(source_width * target_height / source_height)
    return _even(target_height), _even(target_width)


def resize_video_for_tracking(video: np.ndarray, resolution_label: str) -> np.ndarray:
    """Resize a video for CoTracker with OpenCV, preserving frame count and aspect."""
    video_array = np.asarray(video)
    if video_array.ndim != 4 or video_array.shape[-1] != 3:
        raise ValueError("Video must have shape (T, H, W, 3).")

    target_height, target_width = get_tracking_resolution(
        resolution_label,
        source_hw=video_array.shape[1:3],
    )
    interpolation = cv2.INTER_AREA if target_height < video_array.shape[1] else cv2.INTER_LINEAR
    resized_frames = [
        cv2.resize(frame, (target_width, target_height), interpolation=interpolation)
        for frame in video_array
    ]
    return np.stack(resized_frames, axis=0).astype(video_array.dtype, copy=False)


def get_frame_skip_stride(skip_count: int) -> int:
    """Return the keep-every-N stride for a skip-after-each-frame count."""
    return max(1, int(skip_count) + 1)


def should_process_frame_for_skip(frame_index: int, skip_count: int) -> bool:
    """Return whether frame_index should be processed when skipping after each processed frame."""
    frame_index = int(frame_index)
    if frame_index < 0:
        return False
    return frame_index % get_frame_skip_stride(skip_count) == 0


def sample_video_for_frame_skip(video: np.ndarray, source_fps: float, skip_count: int):
    """Keep one frame, skip skip_count frames, and repeat."""
    video_array = np.asarray(video)
    if video_array.ndim != 4:
        raise ValueError("Video must have shape (T, H, W, C).")
    if float(source_fps) <= 0:
        raise ValueError("source_fps must be positive.")

    stride = get_frame_skip_stride(skip_count)
    if stride == 1:
        return video_array, float(source_fps), stride
    return video_array[::stride], float(source_fps) / float(stride), stride


def subsample_video_tensor(video_tensor, stride: int = TRACKING_FRAME_STRIDE):
    """Keep one frame every stride frames on the batched video time axis."""
    if stride <= 0:
        raise ValueError("stride must be positive.")
    return video_tensor[:, ::stride]


def map_frame_index_to_sampled(
    frame_index: int,
    sampled_frame_count: int,
    stride: int = TRACKING_FRAME_STRIDE,
) -> int:
    """Map an original frame index to the previous available sampled frame index."""
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if sampled_frame_count <= 0:
        raise ValueError("sampled_frame_count must be positive.")
    return min(max(int(frame_index) // stride, 0), sampled_frame_count - 1)


def get_online_chunk_start_indices(frame_count: int, step: int):
    """Return online CoTracker chunk starts, including one chunk for short clips."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    if step <= 0:
        raise ValueError("step must be positive.")

    stop = max(frame_count - step, 1)
    return list(range(0, stop, step))


def expand_sampled_time_axis(
    sampled_values: np.ndarray,
    total_frames: int,
    stride: int = TRACKING_FRAME_STRIDE,
    axis: int = 1,
) -> np.ndarray:
    """Expand sampled predictions back to the original frame count."""
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive.")

    sampled_array = np.asarray(sampled_values)
    sampled_length = sampled_array.shape[axis]
    if sampled_length <= 0:
        raise ValueError("sampled_values must contain at least one sampled frame.")

    take_indices = np.minimum(np.arange(total_frames) // stride, sampled_length - 1)
    return np.take(sampled_array, take_indices, axis=axis)


def resolve_torch_device(torch_module) -> str:
    """Prefer CUDA, then Apple MPS, then CPU for CoTracker inference."""
    if torch_module.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def get_cached_cotracker_model(
    device: str,
    cache: Optional[Dict[str, object]] = None,
    loader: Optional[Callable[[], object]] = None,
):
    """Load CoTracker once per device and reuse it for subsequent track calls."""
    model_cache = _MODEL_CACHE if cache is None else cache
    if device not in model_cache:
        if loader is None:
            import torch

            loader = lambda: torch.hub.load(
                str(_LOCAL_COTRACKER_REPO),
                "cotracker3_online",
                source="local",
            )

        model = loader().to(device)
        if hasattr(model, "eval"):
            model.eval()
        model_cache[device] = model

    return model_cache[device]


def _coerce_positive_fps(fps) -> float:
    try:
        parsed_fps = float(fps)
    except (TypeError, ValueError):
        parsed_fps = 24.0
    return parsed_fps if parsed_fps > 0 else 24.0


def _write_rgb_frames_to_mp4(frames: np.ndarray, output_path: Path, fps: float) -> None:
    frame_array = np.asarray(frames)
    if frame_array.ndim != 4 or frame_array.shape[-1] not in (3, 4):
        raise ValueError("SAM video review frames must have shape (T, H, W, 3/4).")
    if frame_array.shape[0] == 0:
        raise ValueError("SAM video review has no frames to save.")

    if frame_array.dtype != np.uint8:
        if np.issubdtype(frame_array.dtype, np.floating) and frame_array.max(initial=0) <= 1.0:
            frame_array = frame_array * 255
        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

    if frame_array.shape[-1] == 4:
        frame_array = frame_array[..., :3]

    height, width = frame_array.shape[1:3]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        _coerce_positive_fps(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer for: {output_path}")

    try:
        for frame in frame_array:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_sam_video_review(
    video_review,
    output_dir: Union[Path, str],
    fps: float = 24.0,
    filename: str = DEFAULT_SAM_PREVIEW_VIDEO_FILENAME,
) -> Path:
    """Save a SAM video review to a user-selected directory as an MP4."""
    if video_review is None:
        raise ValueError("Run SAM video review before saving the preview.")
    if isinstance(video_review, dict):
        video_review = video_review.get("name") or video_review.get("path") or video_review.get("data")
        if video_review is None:
            raise ValueError("Run SAM video review before saving the preview.")

    destination_dir = Path(output_dir).expanduser() if output_dir else Path.cwd()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / filename

    if isinstance(video_review, (str, Path)):
        source_path = Path(video_review).expanduser()
        if not source_path.exists():
            raise FileNotFoundError(f"SAM video review file does not exist: {source_path}")
        if source_path.resolve() != output_path.resolve():
            shutil.copy2(source_path, output_path)
        return output_path

    _write_rgb_frames_to_mp4(np.asarray(video_review), output_path, fps)
    return output_path
