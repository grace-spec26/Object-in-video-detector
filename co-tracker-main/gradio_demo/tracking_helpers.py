from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np


TRACKING_RESOLUTION_OPTIONS = ("1080P", "720P", "640P", "520P")
DEFAULT_TRACKING_RESOLUTION = "520P"
TRACKING_FRAME_STRIDE = 2
_TRACKING_RESOLUTION_HEIGHTS = {
    "520P": 520,
    "640P": 640,
    "720P": 720,
    "1080P": 1080,
}
_MODEL_CACHE = {}
_LOCAL_COTRACKER_REPO = Path(__file__).resolve().parents[1]


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
