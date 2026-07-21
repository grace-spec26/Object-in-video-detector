import hashlib
import html
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import gradio as gr
import mediapy
import numpy as np

try:
    from .export_helpers import (
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        visible_labeled_points_for_frame,
    )
    from .refinement_helpers import pending_refinement_points
    from .tracking_helpers import parse_frame_skip_count, should_process_frame_for_skip
except ImportError:
    from export_helpers import (
        labeled_query_points_for_frame,
        scale_tracks_to_frame_space,
        visible_labeled_points_for_frame,
    )
    from refinement_helpers import pending_refinement_points
    from tracking_helpers import parse_frame_skip_count, should_process_frame_for_skip


POINT_PROMPT_RADIUS = 3
POINT_COLORS = {
    1: (0, 255, 0),
    0: (255, 0, 0),
}
SAM_VIDEO_PROGRESS_READY = """
<div style="width: 100%; padding: 6px 0;">
  <div style="display: flex; justify-content: space-between; font-size: 13px; color: #667085; margin-bottom: 4px;">
    <span>SAM video review ready</span><span>0%</span>
  </div>
  <div style="height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
    <div style="height: 100%; width: 0%; background: #2563eb;"></div>
  </div>
</div>
"""
SAM_MODEL_PROGRESS_READY = """
<div style="width: 100%; padding: 6px 0;">
  <div style="display: flex; justify-content: space-between; font-size: 13px; color: #667085; margin-bottom: 4px;">
    <span>SAM image model not loaded</span><span>0%</span>
  </div>
  <div style="height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
    <div style="height: 100%; width: 0%; background: #2563eb;"></div>
  </div>
</div>
"""
SAM_IMAGE_MODEL_CHOICES = (
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
)
DEFAULT_SAM_IMAGE_MODEL = "sam2.1_hiera_small.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOBILE_SAM_ROOT = PROJECT_ROOT / "MobileSAM-master"
sam_preview_runtime_lock = threading.Lock()
sam_preview_runtimes = {}
sam_preview_preload_lock = threading.Lock()
sam_preview_preload_started = set()
sam_preview_preload_errors = {}


def get_loaded_sam_preview_runtime(model_name, *, blocking=True):
    acquired = sam_preview_runtime_lock.acquire(blocking=blocking)
    if not acquired:
        return None

    try:
        return sam_preview_runtimes.get(model_name)
    finally:
        sam_preview_runtime_lock.release()


def format_sam_model_progress_html(percent, message, color="#2563eb"):
    bounded_percent = int(np.clip(int(round(float(percent))), 0, 100))
    safe_message = html.escape(str(message))
    safe_color = html.escape(str(color))
    return f"""
<div style="width: 100%; padding: 6px 0;">
  <div style="display: flex; justify-content: space-between; font-size: 13px; color: #344054; margin-bottom: 4px;">
    <span>{safe_message}</span><span>{bounded_percent}%</span>
  </div>
  <div style="height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
    <div style="height: 100%; width: {bounded_percent}%; background: {safe_color};"></div>
  </div>
</div>
"""


def resolve_sam_preview_model_option(model_name):
    if str(MOBILE_SAM_ROOT) not in sys.path:
        sys.path.insert(0, str(MOBILE_SAM_ROOT))

    from sam2_coordinate_wrapper import resolve_sam2_model_option

    return resolve_sam2_model_option(model_name)


def sam_checkpoint_file_looks_unavailable(checkpoint_path):
    try:
        stat_result = Path(checkpoint_path).stat()
    except FileNotFoundError:
        return False

    allocated_blocks = getattr(stat_result, "st_blocks", None)
    if stat_result.st_size <= 0 or allocated_blocks is None:
        return False

    allocated_bytes = int(allocated_blocks) * 512
    return allocated_bytes < stat_result.st_size * 0.5


def sam_model_checkpoint_download_progress(model_name):
    model_option = resolve_sam_preview_model_option(model_name)
    model_label = model_option.get("label", str(model_name))
    checkpoint_path = Path(model_option["checkpoint"])
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".download")
    expected_size = int(model_option.get("expected_size") or 0)
    checkpoint_exists = checkpoint_path.exists()
    checkpoint_unavailable = (
        sam_checkpoint_file_looks_unavailable(checkpoint_path)
        if checkpoint_exists
        else False
    )

    if temporary_path.exists() and (checkpoint_unavailable or not checkpoint_exists):
        if sam_checkpoint_file_looks_unavailable(temporary_path):
            return 0, f"{model_label} partial download is a local placeholder; restarting download"
        downloaded_size = temporary_path.stat().st_size
        if expected_size > 0:
            percent = int(round((downloaded_size / expected_size) * 100))
            return percent, f"Downloading {model_label}: {downloaded_size}/{expected_size} bytes"
        return 0, f"Downloading {model_label}: {downloaded_size} bytes"

    if checkpoint_exists:
        checkpoint_size = checkpoint_path.stat().st_size
        if checkpoint_unavailable:
            return 0, f"{model_label} checkpoint is a local placeholder; waiting to redownload"
        if expected_size > 0 and checkpoint_size < expected_size:
            percent = int(round((checkpoint_size / expected_size) * 100))
            return percent, f"{model_label} checkpoint is incomplete: {checkpoint_size}/{expected_size} bytes"
        return 100, f"{model_label} checkpoint downloaded"

    if temporary_path.exists():
        downloaded_size = temporary_path.stat().st_size
        if expected_size > 0:
            percent = int(round((downloaded_size / expected_size) * 100))
            return percent, f"Downloading {model_label}: {downloaded_size}/{expected_size} bytes"
        return 0, f"Downloading {model_label}: {downloaded_size} bytes"

    return 0, f"{model_label} checkpoint waiting to download"


def current_sam_model_progress_html(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    if model_name not in SAM_IMAGE_MODEL_CHOICES:
        return format_sam_model_progress_html(0, f"Unknown SAM image model: {model_name}", "#dc2626")

    runtime = get_loaded_sam_preview_runtime(model_name, blocking=False)
    if runtime is not None:
        return format_sam_model_progress_html(
            100,
            f"{runtime['model_label']} loaded on {runtime['device']}",
        )

    with sam_preview_preload_lock:
        previous_error = sam_preview_preload_errors.get(model_name)
        is_loading = model_name in sam_preview_preload_started

    try:
        percent, message = sam_model_checkpoint_download_progress(model_name)
    except Exception as exc:
        return format_sam_model_progress_html(
            0,
            f"SAM image model {model_name} progress unavailable: {exc}",
            "#dc2626",
        )

    if previous_error and not is_loading:
        return format_sam_model_progress_html(
            percent,
            f"SAM image model {model_name} failed to load: {previous_error}. {message}",
            "#dc2626",
        )

    if is_loading and percent >= 100:
        try:
            model_label = resolve_sam_preview_model_option(model_name).get("label", model_name)
        except Exception:
            model_label = model_name
        message = f"{model_label} checkpoint ready; loading model runtime"
    elif is_loading and percent == 0 and "local placeholder" not in message and "incomplete" not in message:
        try:
            model_label = resolve_sam_preview_model_option(model_name).get("label", model_name)
        except Exception:
            model_label = model_name
        message = f"Preparing {model_label} download or model load"

    return format_sam_model_progress_html(percent, message)


def stream_sam_model_loading_progress(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    start_sam_preview_preload(model_name)

    while True:
        yield current_sam_model_progress_html(model_name)

        is_loaded = get_loaded_sam_preview_runtime(model_name, blocking=False) is not None
        with sam_preview_preload_lock:
            has_error = model_name in sam_preview_preload_errors
            is_loading = model_name in sam_preview_preload_started

        if is_loaded or (has_error and not is_loading):
            return
        time.sleep(1)


def draw_query_point(frame, x, y, point_label):
    point_color = POINT_COLORS.get(int(point_label), POINT_COLORS[1])
    x, y = int(round(x)), int(round(y))
    frame = cv2.circle(frame, (x, y), POINT_PROMPT_RADIUS, point_color, -1)
    frame = cv2.circle(frame, (x, y), POINT_PROMPT_RADIUS, (255, 255, 255), 1)
    return frame


def get_sam_preview_runtime(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    if model_name not in SAM_IMAGE_MODEL_CHOICES:
        allowed = ", ".join(SAM_IMAGE_MODEL_CHOICES)
        raise ValueError(f"SAM image model must be one of: {allowed}")

    runtime = get_loaded_sam_preview_runtime(model_name)
    if runtime is not None:
        return runtime

    if str(MOBILE_SAM_ROOT) not in sys.path:
        sys.path.insert(0, str(MOBILE_SAM_ROOT))

    from sam2_coordinate_wrapper import load_sam2_predictor, resolve_sam2_model_option

    model_option = resolve_sam2_model_option(model_name)
    predictor, device = load_sam2_predictor(
        model_name=model_name,
        download_checkpoint=True,
    )
    runtime = {
        "predictor": predictor,
        "predictor_lock": threading.Lock(),
        "device": device,
        "model_name": model_name,
        "model_label": model_option["label"],
        "image_cache_key": None,
    }

    with sam_preview_runtime_lock:
        existing_runtime = sam_preview_runtimes.get(model_name)
        if existing_runtime is not None:
            return existing_runtime
        sam_preview_runtimes[model_name] = runtime
    return runtime


def preload_sam_preview_runtime(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    try:
        runtime = get_sam_preview_runtime(model_name)
        with sam_preview_preload_lock:
            sam_preview_preload_errors.pop(model_name, None)
            sam_preview_preload_started.discard(model_name)
        print(
            f"SAM preview model preloaded: {runtime['model_label']} on {runtime['device']}",
            flush=True,
        )
    except Exception as exc:
        with sam_preview_preload_lock:
            sam_preview_preload_errors[model_name] = str(exc)
            sam_preview_preload_started.discard(model_name)
        print(f"SAM preview model preload failed for {model_name}: {exc}", flush=True)


def start_sam_preview_preload(sam_model=DEFAULT_SAM_IMAGE_MODEL):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    already_loaded = get_loaded_sam_preview_runtime(model_name) is not None
    with sam_preview_preload_lock:
        if model_name in sam_preview_preload_started or already_loaded:
            return
        sam_preview_preload_started.add(model_name)

    thread = threading.Thread(
        target=preload_sam_preview_runtime,
        args=(model_name,),
        daemon=True,
        name=f"sam-preview-preload-{model_name}",
    )
    thread.start()


def get_sam_preview_runtime_if_ready(sam_model):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    runtime_message = (
        f"SAM preview model {model_name} is still loading or downloading. "
        "Try Preview SAM again in a moment."
    )

    acquired = sam_preview_runtime_lock.acquire(blocking=False)
    if not acquired:
        start_sam_preview_preload(model_name)
        return None, runtime_message

    try:
        runtime = sam_preview_runtimes.get(model_name)
        if runtime is not None:
            return runtime, None
    finally:
        sam_preview_runtime_lock.release()

    with sam_preview_preload_lock:
        previous_error = sam_preview_preload_errors.get(model_name)
    start_sam_preview_preload(model_name)
    if previous_error:
        return (
            None,
            f"SAM preview model {model_name} failed to load previously: {previous_error}. "
            "Retrying in the background.",
        )
    return None, runtime_message


def as_uint8_rgb_frame(frame):
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError("Expected an RGB/RGBA frame for SAM preview.")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max(initial=0) <= 1.0:
            image = image * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image.copy()


def sam_preview_frame_cache_key(frame):
    image = np.ascontiguousarray(as_uint8_rgb_frame(frame))
    digest = hashlib.blake2b(image.tobytes(), digest_size=16).hexdigest()
    return image.shape, str(image.dtype), digest


def predict_sam_preview_mask(runtime, frame, point_coords, point_labels):
    frame_cache_key = sam_preview_frame_cache_key(frame)
    predictor = runtime["predictor"]
    with runtime["predictor_lock"]:
        if runtime.get("image_cache_key") != frame_cache_key:
            predictor.set_image(frame)
            runtime["image_cache_key"] = frame_cache_key
        return predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            multimask_output=True,
            normalize_coords=True,
        )


def sam_preview_mask_values_at_points(mask, points):
    if len(points) == 0:
        return np.empty((0,), dtype=bool)
    mask_array = np.asarray(mask).astype(bool)
    height, width = mask_array.shape[:2]
    rounded = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
    xs = np.clip(rounded[:, 0], 0, width - 1)
    ys = np.clip(rounded[:, 1], 0, height - 1)
    return mask_array[ys, xs].astype(bool)


def select_sam_preview_mask(masks, scores, point_coords, point_labels):
    masks_array = np.asarray(masks).astype(bool)
    if len(masks_array) == 0:
        raise ValueError("SAM returned no masks for preview.")

    coords = np.asarray(point_coords, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.int32)
    positives = coords[labels == 1]
    negatives = coords[labels == 0]
    scored_indices = []
    for index, mask in enumerate(masks_array):
        positive_hits = sam_preview_mask_values_at_points(mask, positives)
        negative_hits = sam_preview_mask_values_at_points(mask, negatives)
        positive_score = float(np.mean(positive_hits)) if len(positive_hits) else 0.0
        negative_score = float(np.mean(~negative_hits)) if len(negative_hits) else 1.0
        sam_score = float(scores[index]) if index < len(scores) else 0.0
        area = float(np.count_nonzero(mask))
        combined_score = 5.0 * positive_score + 4.0 * negative_score + sam_score
        scored_indices.append((combined_score, sam_score, -area, index))
    return int(max(scored_indices)[-1])


def draw_sam_preview(frame, mask, point_coords, point_labels):
    preview = as_uint8_rgb_frame(frame)
    mask_array = np.asarray(mask).astype(bool)
    if mask_array.shape != preview.shape[:2]:
        raise ValueError("SAM mask size does not match frame size.")

    overlay_color = np.asarray([0, 180, 255], dtype=np.float32)
    preview_float = preview.astype(np.float32)
    preview_float[mask_array] = preview_float[mask_array] * 0.55 + overlay_color * 0.45
    preview = np.clip(preview_float, 0, 255).astype(np.uint8)

    height, width = preview.shape[:2]
    for (x, y), label in zip(np.asarray(point_coords), np.asarray(point_labels)):
        x = int(np.clip(round(float(x)), 0, width - 1))
        y = int(np.clip(round(float(y)), 0, height - 1))
        preview = draw_query_point(preview, x, y, int(label))
    return preview


def format_sam_prompt_summary(point_coords, point_labels, max_points=12):
    coords = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
    labels = np.asarray(point_labels, dtype=np.int32).reshape(-1)
    display_count = min(len(coords), int(max_points))
    display_coords = [
        [round(float(x), 2), round(float(y), 2)]
        for x, y in coords[:display_count]
    ]
    display_labels = [int(label) for label in labels[:display_count]]
    suffix = ""
    if len(coords) > display_count:
        suffix = f", showing first {display_count} of {len(coords)}"
    return f"point_coords={display_coords}, point_labels={display_labels}{suffix}"


def format_sam_video_progress_html(completed_frames, total_frames, message):
    total = max(1, int(total_frames))
    completed = int(np.clip(int(completed_frames), 0, total))
    percent = int(round((completed / total) * 100))
    progress_label = f"{completed}/{total} ({percent}%)"
    return f"""
<div style="width: 100%; padding: 6px 0;">
  <div style="display: flex; justify-content: space-between; font-size: 13px; color: #344054; margin-bottom: 4px;">
    <span>{message}</span><span>{progress_label}</span>
  </div>
  <div style="height: 8px; background: #e5e7eb; border-radius: 999px; overflow: hidden;">
    <div style="height: 100%; width: {percent}%; background: #2563eb;"></div>
  </div>
</div>
"""


def selected_sam_video_frame_indices(frame_count, skip_count):
    return [
        frame_index
        for frame_index in range(max(0, int(frame_count)))
        if should_process_frame_for_skip(frame_index, skip_count)
    ]


def wait_for_sam_video_runtime(sam_model, selected_frame_count, frame_count):
    model_name = str(sam_model or DEFAULT_SAM_IMAGE_MODEL)
    runtime = get_loaded_sam_preview_runtime(model_name, blocking=False)
    if runtime is not None:
        return runtime

    start_sam_preview_preload(model_name)
    while True:
        runtime = get_loaded_sam_preview_runtime(model_name, blocking=False)
        if runtime is not None:
            return runtime

        with sam_preview_preload_lock:
            previous_error = sam_preview_preload_errors.get(model_name)
            is_loading = model_name in sam_preview_preload_started

        try:
            _, model_message = sam_model_checkpoint_download_progress(model_name)
        except Exception as exc:
            model_message = f"SAM image model {model_name} progress unavailable: {exc}"

        if previous_error and not is_loading:
            yield (
                format_sam_video_progress_html(
                    0,
                    selected_frame_count,
                    f"0/{selected_frame_count} selected frame(s) - SAM model failed",
                ),
                None,
                (
                    f"SAM video review cannot start because SAM image model {model_name} "
                    f"failed to load: {previous_error}. {model_message}."
                ),
            )
            return None

        yield (
            format_sam_video_progress_html(
                0,
                selected_frame_count,
                f"0/{selected_frame_count} selected frame(s) - Waiting for SAM model",
            ),
            None,
            (
                f"Waiting for SAM model before processing {selected_frame_count} selected frame(s) "
                f"from {frame_count} total video frame(s): {model_message}."
            ),
        )
        time.sleep(1)


def sam_point_prompts_for_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_index,
    prefer_tracked_points=False,
    refinement_query_points=None,
    tracked_prompt_sources=None,
    refinement_source_frames=None,
):
    point_coords = np.empty((0, 2), dtype=np.float32)
    point_labels = np.empty((0,), dtype=np.int32)
    prompt_source = "selected"

    if prefer_tracked_points and selected_tracks is not None and selected_point_labels is not None:
        scaled_tracks = scale_tracks_to_frame_space(
            selected_tracks,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        point_coords, point_labels = visible_labeled_points_for_frame(
            scaled_tracks,
            frame_index,
            point_labels=selected_point_labels,
            visibility=selected_visibility,
        )
        prompt_source = "tracked"

    if len(point_coords) == 0:
        point_coords, point_labels = labeled_query_points_for_frame(
            query_points,
            frame_index,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        prompt_source = "selected"

    if (
        len(point_coords) == 0
        and not prefer_tracked_points
        and selected_tracks is not None
        and selected_point_labels is not None
    ):
        scaled_tracks = scale_tracks_to_frame_space(
            selected_tracks,
            source_hw=video_preview_array.shape[1:3],
            target_hw=video_frames.shape[1:3],
        )
        point_coords, point_labels = visible_labeled_points_for_frame(
            scaled_tracks,
            frame_index,
            point_labels=selected_point_labels,
            visibility=selected_visibility,
        )
        prompt_source = "tracked"

    pending_refinement_query_points = pending_refinement_points(
        refinement_query_points,
        tracked_prompt_sources,
        frame_count=len(video_frames),
    )
    refinement_source_hw = (
        np.asarray(refinement_source_frames).shape[1:3]
        if refinement_source_frames is not None
        else video_preview_array.shape[1:3]
    )
    refinement_coords, refinement_labels = labeled_query_points_for_frame(
        pending_refinement_query_points,
        frame_index,
        source_hw=refinement_source_hw,
        target_hw=video_frames.shape[1:3],
    )
    if len(refinement_coords) > 0:
        if len(point_coords) > 0:
            point_coords = np.concatenate([point_coords, refinement_coords], axis=0)
            point_labels = np.concatenate([point_labels, refinement_labels], axis=0)
            prompt_source = f"{prompt_source} + refinement"
        else:
            point_coords = refinement_coords
            point_labels = refinement_labels
            prompt_source = "refinement"

    return point_coords, point_labels, prompt_source


def preview_sam_on_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_num,
    sam_model,
    prefer_tracked_points=False,
    refinement_query_points=None,
    tracked_prompt_sources=None,
    refinement_source_frames=None,
):
    if video_frames is None or video_preview_array is None:
        message = "Submit a video before previewing SAM."
        gr.Warning(message, duration=5)
        return None, message

    frame_index = int(np.clip(int(frame_num), 0, len(video_frames) - 1))
    point_coords, point_labels, prompt_source = sam_point_prompts_for_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        frame_index,
        prefer_tracked_points=prefer_tracked_points,
        refinement_query_points=refinement_query_points,
        tracked_prompt_sources=tracked_prompt_sources,
        refinement_source_frames=refinement_source_frames,
    )

    if len(point_coords) == 0:
        message = f"No selected or visible tracked points on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message
    if not np.any(point_labels == 1):
        message = f"SAM needs at least one visible positive point on frame {frame_index}."
        gr.Warning(message, duration=5)
        return as_uint8_rgb_frame(video_frames[frame_index]), message

    frame = as_uint8_rgb_frame(video_frames[frame_index])
    prompt_summary = format_sam_prompt_summary(point_coords, point_labels)
    runtime, loading_message = get_sam_preview_runtime_if_ready(sam_model)
    if runtime is None:
        empty_mask = np.zeros(frame.shape[:2], dtype=bool)
        prompt_preview = draw_sam_preview(frame, empty_mask, point_coords, point_labels)
        return prompt_preview, f"{loading_message} Loaded prompts: {prompt_summary}."

    masks, scores, _ = predict_sam_preview_mask(runtime, frame, point_coords, point_labels)
    best_mask = masks[select_sam_preview_mask(masks, scores, point_coords, point_labels)]
    preview = draw_sam_preview(frame, best_mask, point_coords, point_labels)
    positive_count = int(np.sum(point_labels == 1))
    negative_count = int(np.sum(point_labels == 0))
    return (
        preview,
        f"SAM preview frame {frame_index} from {prompt_source} points with "
        f"{runtime['model_label']} on {runtime['device']} "
        f"({positive_count} positive, {negative_count} negative point(s)). "
        f"Loaded prompts: {prompt_summary}.",
    )


def preview_sam_for_selected_frame(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    query_frame_num,
    tracked_frame_num,
    tracked_video_preview,
    sam_model,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    has_processed_selection = tracked_video_preview is not None and selected_tracks is not None
    frame_num = tracked_frame_num if has_processed_selection else query_frame_num
    refinement_source_frames = tracked_video_preview if has_processed_selection else None
    return preview_sam_on_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        frame_num,
        sam_model,
        prefer_tracked_points=has_processed_selection,
        refinement_query_points=refinement_query_points,
        tracked_prompt_sources=tracked_prompt_sources,
        refinement_source_frames=refinement_source_frames,
    )


def preview_sam_on_frame_with_progress(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    frame_num,
    sam_model,
):
    preview, status = preview_sam_on_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        frame_num,
        sam_model,
    )
    return preview, current_sam_model_progress_html(sam_model), status


def preview_sam_for_selected_frame_with_progress(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    query_frame_num,
    tracked_frame_num,
    tracked_video_preview,
    sam_model,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    preview, status = preview_sam_for_selected_frame(
        video_frames,
        video_preview_array,
        query_points,
        selected_tracks,
        selected_visibility,
        selected_point_labels,
        query_frame_num,
        tracked_frame_num,
        tracked_video_preview,
        sam_model,
        refinement_query_points=refinement_query_points,
        tracked_prompt_sources=tracked_prompt_sources,
    )
    return preview, current_sam_model_progress_html(sam_model), status


def preview_sam_video_for_processed_frames(
    video_frames,
    video_preview_array,
    query_points,
    selected_tracks,
    selected_visibility,
    selected_point_labels,
    tracked_video_preview,
    video_fps,
    sam_model,
    sam_video_skip_frames,
    refinement_query_points=None,
    tracked_prompt_sources=None,
):
    if video_frames is None or video_preview_array is None:
        message = "Submit and track a video before running SAM video review."
        gr.Warning(message, duration=5)
        yield SAM_VIDEO_PROGRESS_READY, None, message
        return
    if tracked_video_preview is None or selected_tracks is None or selected_point_labels is None:
        message = "Track selected points before running SAM video review."
        gr.Warning(message, duration=5)
        yield SAM_VIDEO_PROGRESS_READY, None, message
        return

    try:
        skip_count = parse_frame_skip_count(sam_video_skip_frames)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    frame_count = len(video_frames)
    selected_frame_indices = selected_sam_video_frame_indices(frame_count, skip_count)
    selected_frame_count = len(selected_frame_indices)
    yield (
        format_sam_video_progress_html(
            0,
            selected_frame_count,
            f"0/{selected_frame_count} selected frame(s) - Loading SAM model",
        ),
        None,
        (
            f"Loading SAM model for {selected_frame_count} selected frame(s) "
            f"from {frame_count} total video frame(s)..."
        ),
    )

    runtime = yield from wait_for_sam_video_runtime(sam_model, selected_frame_count, frame_count)
    if runtime is None:
        return

    predictor = runtime["predictor"]
    review_frames = []
    processed_count = 0
    skipped_by_frame_skip = max(0, frame_count - selected_frame_count)
    skipped_no_points = 0
    skipped_no_positive = 0
    yield (
        format_sam_video_progress_html(
            0,
            selected_frame_count,
            f"0/{selected_frame_count} selected frame(s) - Starting SAM video review",
        ),
        None,
        (
            f"Starting SAM video review for {selected_frame_count} selected frame(s) "
            f"from {frame_count} total video frame(s)."
        ),
    )

    for selected_index, frame_index in enumerate(selected_frame_indices, start=1):
        frame = as_uint8_rgb_frame(video_frames[frame_index])
        yield (
            format_sam_video_progress_html(
                selected_index - 1,
                selected_frame_count,
                f"Processing selected frame {selected_index}/{selected_frame_count}",
            ),
            None,
            (
                f"Processing selected frame {selected_index}/{selected_frame_count} "
                f"(video frame {frame_index + 1}/{frame_count})."
            ),
        )

        point_coords, point_labels, _ = sam_point_prompts_for_frame(
            video_frames,
            video_preview_array,
            query_points,
            selected_tracks,
            selected_visibility,
            selected_point_labels,
            frame_index,
            prefer_tracked_points=True,
            refinement_query_points=refinement_query_points,
            tracked_prompt_sources=tracked_prompt_sources,
            refinement_source_frames=tracked_video_preview,
        )
        if len(point_coords) == 0:
            skipped_no_points += 1
            yield (
                format_sam_video_progress_html(
                    selected_index,
                    selected_frame_count,
                    f"Checked selected frame {selected_index}/{selected_frame_count}",
                ),
                None,
                (
                    f"SAM video review checked {selected_index}/{selected_frame_count} "
                    f"selected frame(s); video frame {frame_index + 1}/{frame_count} has no points."
                ),
            )
            continue
        if not np.any(point_labels == 1):
            skipped_no_positive += 1
            yield (
                format_sam_video_progress_html(
                    selected_index,
                    selected_frame_count,
                    f"Checked selected frame {selected_index}/{selected_frame_count}",
                ),
                None,
                (
                    f"SAM video review checked {selected_index}/{selected_frame_count} "
                    f"selected frame(s); video frame {frame_index + 1}/{frame_count} has no positive points."
                ),
            )
            continue

        predictor.set_image(frame)
        masks, scores, _ = predictor.predict(
            point_coords=point_coords.astype(np.float32),
            point_labels=point_labels.astype(np.int32),
            multimask_output=True,
            normalize_coords=True,
        )
        best_mask = masks[select_sam_preview_mask(masks, scores, point_coords, point_labels)]
        review_frames.append(draw_sam_preview(frame, best_mask, point_coords, point_labels))
        processed_count += 1
        yield (
            format_sam_video_progress_html(
                selected_index,
                selected_frame_count,
                f"Processed {selected_index}/{selected_frame_count} selected frame(s)",
            ),
            None,
            (
                f"SAM video review processed {processed_count} frame(s); "
                f"checked {selected_index}/{selected_frame_count} selected frame(s)."
            ),
        )

    skipped_parts = []
    if skipped_by_frame_skip:
        skipped_parts.append(f"{skipped_by_frame_skip} by skip setting")
    if skipped_no_points:
        skipped_parts.append(f"{skipped_no_points} without points")
    if skipped_no_positive:
        skipped_parts.append(f"{skipped_no_positive} without positive points")
    skipped_text = f"; skipped {', '.join(skipped_parts)}" if skipped_parts else ""

    if not review_frames:
        yield (
            format_sam_video_progress_html(selected_frame_count, selected_frame_count, "SAM video review complete"),
            None,
            (
                "No SAM-processed frames were available for video review after checking "
                f"{selected_frame_count}/{selected_frame_count} selected frame(s) "
                f"from {frame_count} total video frame(s){skipped_text}."
            ),
        )
        return

    video_file_name = uuid.uuid4().hex + ".mp4"
    video_path = os.path.join(os.path.dirname(__file__), "tmp")
    video_file_path = os.path.join(video_path, video_file_name)
    os.makedirs(video_path, exist_ok=True)
    output_fps = float(video_fps or 24)
    if output_fps <= 0:
        output_fps = 24
    mediapy.write_video(video_file_path, np.asarray(review_frames), fps=output_fps)

    yield (
        format_sam_video_progress_html(selected_frame_count, selected_frame_count, "SAM video review complete"),
        video_file_path,
        (
            f"SAM video review complete for {processed_count}/{selected_frame_count} selected frame(s) "
            f"from {frame_count} total video frame(s) with {runtime['model_label']} "
            f"on {runtime['device']}{skipped_text}."
        ),
    )
